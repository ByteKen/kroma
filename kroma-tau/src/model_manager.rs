use crate::dto::{
    AppUpdateDownloadProgressDto, ModelDownloadProgressDto, ModelManifestEntryStatusDto,
    ModelManifestStatusDto,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub const MODEL_DOWNLOAD_PROGRESS_EVENT: &str = "kroma://model-download-progress";
pub const APP_UPDATE_PROGRESS_EVENT: &str = "kroma://app-update-progress";

const EMBEDDED_MODELS_MANIFEST: &str = include_str!("../../models/models.json");
const INSTALLED_INDEX_FILE: &str = "installed.json";
const DOWNLOAD_CHUNK_BYTES: usize = 64 * 1024;
const PROGRESS_EMIT_INTERVAL: Duration = Duration::from_millis(100);

fn default_required() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelManifest {
    pub schema_version: u32,
    pub release_tag: String,
    pub models: Vec<ModelManifestEntry>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelManifestEntry {
    pub id: String,
    pub name: String,
    pub category: String,
    pub version: String,
    pub relative_path: String,
    pub url: String,
    pub size_bytes: u64,
    pub sha256: String,
    #[serde(default = "default_required")]
    pub required: bool,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InstalledModelReceipt {
    pub version: String,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InstalledModelsIndex {
    #[serde(default)]
    pub entries: BTreeMap<String, InstalledModelReceipt>,
}

fn sanitize_relative_path(raw: &str) -> Result<PathBuf, String> {
    let trimmed = raw.trim().replace('\\', "/");
    if trimmed.is_empty() {
        return Err("model relativePath cannot be empty".to_string());
    }
    let path = Path::new(&trimmed);
    let mut clean = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::Normal(part) => clean.push(part),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(format!("unsafe model relativePath '{raw}'"));
            }
        }
    }
    if clean.as_os_str().is_empty() {
        return Err(format!("invalid model relativePath '{raw}'"));
    }
    Ok(clean)
}

/// Loads `models.json` from the first available local search directory override, falling back
/// to the embedded compile-time manifest so packaged builds always have defaults.
pub fn load_models_manifest(search_dirs: &[PathBuf]) -> Result<ModelManifest, String> {
    for dir in search_dirs {
        let candidate = dir.join("models.json");
        if candidate.is_file()
            && let Ok(raw) = fs::read_to_string(&candidate)
            && let Ok(parsed) = serde_json::from_str::<ModelManifest>(&raw)
            && !parsed.models.is_empty()
        {
            return Ok(parsed);
        }
    }

    serde_json::from_str::<ModelManifest>(EMBEDDED_MODELS_MANIFEST)
        .map_err(|err| format!("failed to parse embedded models.json manifest: {err}"))
}

fn load_installed_index(managed_models_dir: &Path) -> InstalledModelsIndex {
    let path = managed_models_dir.join(INSTALLED_INDEX_FILE);
    fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str::<InstalledModelsIndex>(&raw).ok())
        .unwrap_or_default()
}

fn save_installed_index(
    managed_models_dir: &Path,
    index: &InstalledModelsIndex,
) -> Result<(), String> {
    fs::create_dir_all(managed_models_dir)
        .map_err(|err| format!("failed to create models dir: {err}"))?;
    let target = managed_models_dir.join(INSTALLED_INDEX_FILE);
    let temp = managed_models_dir.join(format!("{INSTALLED_INDEX_FILE}.tmp"));
    let json = serde_json::to_string_pretty(index)
        .map_err(|err| format!("failed to serialize installed models index: {err}"))?;
    fs::write(&temp, json).map_err(|err| format!("failed to write installed index: {err}"))?;
    replace_file_atomically(&temp, &target)
}

fn replace_file_atomically(src_part: &Path, dest: &Path) -> Result<(), String> {
    if dest.exists() {
        fs::remove_file(dest).map_err(|err| {
            format!(
                "failed to replace existing file '{}': {err}",
                dest.display()
            )
        })?;
    }
    fs::rename(src_part, dest).map_err(|err| {
        format!(
            "failed to finalize downloaded file '{}': {err}",
            dest.display()
        )
    })
}

/// Evaluates manifest entries against `managed_models_dir` and fallback directories without
/// reading multi-hundred-MB weights on normal boots (uses receipt version + exact byte size).
pub fn inspect_manifest_status(
    manifest: &ModelManifest,
    managed_models_dir: &Path,
    fallback_search_dirs: &[PathBuf],
) -> Result<ModelManifestStatusDto, String> {
    let installed = load_installed_index(managed_models_dir);
    let mut entries = Vec::with_capacity(manifest.models.len());
    let mut missing_count = 0usize;
    let mut outdated_count = 0usize;
    let mut pending_download_bytes = 0u64;

    for entry in &manifest.models {
        let rel = sanitize_relative_path(&entry.relative_path)?;
        let managed_path = managed_models_dir.join(&rel);
        let receipt = installed.entries.get(&entry.id);

        let (status, resolved_path, installed_version) = if managed_path.is_file() {
            let file_len = fs::metadata(&managed_path).map(|m| m.len()).unwrap_or(0);
            let size_ok = entry.size_bytes == 0 || file_len == entry.size_bytes;
            let version_ok = receipt.is_none_or(|r| r.version == entry.version);
            if size_ok && version_ok {
                (
                    "ready",
                    Some(managed_path.to_string_lossy().to_string()),
                    Some(receipt.map_or_else(|| entry.version.clone(), |r| r.version.clone())),
                )
            } else {
                (
                    "outdated",
                    Some(managed_path.to_string_lossy().to_string()),
                    receipt.map(|r| r.version.clone()),
                )
            }
        } else {
            let mut found_ready: Option<PathBuf> = None;
            let mut found_mismatch: Option<PathBuf> = None;
            for dir in fallback_search_dirs {
                if dir == managed_models_dir {
                    continue;
                }
                let candidate = dir.join(&rel);
                if candidate.is_file() {
                    let file_len = fs::metadata(&candidate).map(|m| m.len()).unwrap_or(0);
                    let size_ok = entry.size_bytes == 0 || file_len == entry.size_bytes;
                    if size_ok {
                        found_ready = Some(candidate);
                        break;
                    } else if found_mismatch.is_none() {
                        found_mismatch = Some(candidate);
                    }
                }
            }
            if let Some(path) = found_ready {
                (
                    "ready",
                    Some(path.to_string_lossy().to_string()),
                    Some(entry.version.clone()),
                )
            } else if let Some(path) = found_mismatch {
                ("outdated", Some(path.to_string_lossy().to_string()), None)
            } else {
                ("missing", None, None)
            }
        };

        if entry.required {
            if status == "missing" {
                missing_count += 1;
                pending_download_bytes = pending_download_bytes.saturating_add(entry.size_bytes);
            } else if status == "outdated" {
                outdated_count += 1;
                pending_download_bytes = pending_download_bytes.saturating_add(entry.size_bytes);
            }
        }

        entries.push(ModelManifestEntryStatusDto {
            id: entry.id.clone(),
            name: entry.name.clone(),
            category: entry.category.clone(),
            version: entry.version.clone(),
            installed_version,
            relative_path: entry.relative_path.clone(),
            size_bytes: entry.size_bytes,
            required: entry.required,
            status,
            resolved_path,
        });
    }

    let all_ready = entries
        .iter()
        .all(|item| !item.required || item.status == "ready");

    Ok(ModelManifestStatusDto {
        schema_version: manifest.schema_version,
        release_tag: manifest.release_tag.clone(),
        models_dir: managed_models_dir.to_string_lossy().to_string(),
        all_ready,
        missing_count,
        outdated_count,
        pending_download_bytes,
        entries,
    })
}

fn stream_and_verify_to_file<R, F>(
    reader: &mut R,
    dest: &Path,
    expected_size: u64,
    expected_sha256: &str,
    cancel_flag: &AtomicBool,
    mut on_chunk: F,
) -> Result<(u64, String), String>
where
    R: Read,
    F: FnMut(u64),
{
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create destination folder '{}': {err}",
                parent.display()
            )
        })?;
    }

    let part_path = PathBuf::from(format!("{}.part", dest.display()));
    let mut out = File::create(&part_path)
        .map_err(|err| format!("failed to create temp file '{}': {err}", part_path.display()))?;

    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; DOWNLOAD_CHUNK_BYTES];
    let mut written = 0u64;

    loop {
        if cancel_flag.load(Ordering::Relaxed) {
            drop(out);
            let _ = fs::remove_file(&part_path);
            return Err("Model download cancelled.".to_string());
        }

        let read_len = match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(err) => {
                drop(out);
                let _ = fs::remove_file(&part_path);
                return Err(format!("download stream interrupted: {err}"));
            }
        };

        if let Err(err) = out.write_all(&buf[..read_len]) {
            drop(out);
            let _ = fs::remove_file(&part_path);
            return Err(format!("failed writing model to disk: {err}"));
        }
        hasher.update(&buf[..read_len]);
        written = written.saturating_add(read_len as u64);
        on_chunk(written);
    }

    if let Err(err) = out.flush() {
        drop(out);
        let _ = fs::remove_file(&part_path);
        return Err(format!("failed flushing model to disk: {err}"));
    }
    drop(out);

    if expected_size > 0 && written != expected_size {
        let _ = fs::remove_file(&part_path);
        return Err(format!(
            "downloaded file size mismatch (expected {expected_size} bytes, got {written} bytes)"
        ));
    }

    let actual_sha = hex::encode(hasher.finalize());
    if !expected_sha256.trim().is_empty() && !actual_sha.eq_ignore_ascii_case(expected_sha256.trim())
    {
        let _ = fs::remove_file(&part_path);
        return Err(format!(
            "SHA-256 verification failed (expected {}, got {actual_sha})",
            expected_sha256.trim()
        ));
    }

    if let Err(err) = replace_file_atomically(&part_path, dest) {
        let _ = fs::remove_file(&part_path);
        return Err(err);
    }

    Ok((written, actual_sha))
}

fn build_download_agent() -> ureq::Agent {
    let tls_config = ureq::tls::TlsConfig::builder()
        .provider(ureq::tls::TlsProvider::NativeTls)
        .root_certs(ureq::tls::RootCerts::PlatformVerifier)
        .build();
    let config = ureq::Agent::config_builder()
        .tls_config(tls_config)
        .timeout_connect(Some(Duration::from_secs(15)))
        .timeout_recv_body(Some(Duration::from_secs(600)))
        .build();
    config.into()
}

/// Downloads all missing or outdated **required** models defined by `manifest` into
/// `managed_models_dir`, streaming progress updates via `on_progress` and verifying SHA-256.
pub fn download_pending_models<F>(
    manifest: &ModelManifest,
    managed_models_dir: &Path,
    fallback_search_dirs: &[PathBuf],
    cancel_flag: &AtomicBool,
    on_progress: F,
) -> Result<ModelManifestStatusDto, String>
where
    F: FnMut(ModelDownloadProgressDto),
{
    download_selected_models(
        manifest,
        None,
        managed_models_dir,
        fallback_search_dirs,
        cancel_flag,
        on_progress,
    )
}

/// Downloads a specific opt-in model pack (e.g. `bisenet-face-parsing` or `lama-inpainting`)
/// on demand into `managed_models_dir` with live progress events and SHA-256 verification.
pub fn download_optional_model_by_id<F>(
    manifest: &ModelManifest,
    model_id: &str,
    managed_models_dir: &Path,
    fallback_search_dirs: &[PathBuf],
    cancel_flag: &AtomicBool,
    on_progress: F,
) -> Result<ModelManifestStatusDto, String>
where
    F: FnMut(ModelDownloadProgressDto),
{
    let trimmed = model_id.trim();
    if !manifest.models.iter().any(|m| m.id == trimmed) {
        return Err(format!("Unknown model pack '{trimmed}'."));
    }
    download_selected_models(
        manifest,
        Some(trimmed),
        managed_models_dir,
        fallback_search_dirs,
        cancel_flag,
        on_progress,
    )
}

fn download_selected_models<F>(
    manifest: &ModelManifest,
    only_model_id: Option<&str>,
    managed_models_dir: &Path,
    fallback_search_dirs: &[PathBuf],
    cancel_flag: &AtomicBool,
    mut on_progress: F,
) -> Result<ModelManifestStatusDto, String>
where
    F: FnMut(ModelDownloadProgressDto),
{
    cancel_flag.store(false, Ordering::Relaxed);
    let initial = inspect_manifest_status(manifest, managed_models_dir, fallback_search_dirs)?;

    let pending: Vec<&ModelManifestEntry> = manifest
        .models
        .iter()
        .filter(|entry| match only_model_id {
            Some(target_id) => entry.id == target_id,
            None => entry.required,
        })
        .filter(|entry| {
            initial
                .entries
                .iter()
                .find(|s| s.id == entry.id)
                .is_none_or(|s| s.status != "ready")
        })
        .collect();

    if pending.is_empty() {
        on_progress(ModelDownloadProgressDto {
            phase: "completed",
            model_id: String::new(),
            model_name: "All AI models ready".to_string(),
            index: manifest.models.len(),
            total_count: manifest.models.len(),
            downloaded_bytes: 0,
            total_bytes: 0,
            overall_downloaded_bytes: 0,
            overall_total_bytes: 0,
            bytes_per_second: 0,
        });
        return Ok(initial);
    }

    fs::create_dir_all(managed_models_dir)
        .map_err(|err| format!("failed to prepare models directory: {err}"))?;

    let agent = build_download_agent();

    let mut installed_index = load_installed_index(managed_models_dir);
    let overall_total_bytes: u64 = pending.iter().map(|e| e.size_bytes).sum();
    let total_count = pending.len();
    let mut overall_completed_before = 0u64;

    for (idx, entry) in pending.iter().enumerate() {
        if cancel_flag.load(Ordering::Relaxed) {
            return Err("Model download cancelled.".to_string());
        }

        let rel = sanitize_relative_path(&entry.relative_path)?;
        let dest = managed_models_dir.join(&rel);

        on_progress(ModelDownloadProgressDto {
            phase: "downloading",
            model_id: entry.id.clone(),
            model_name: entry.name.clone(),
            index: idx + 1,
            total_count,
            downloaded_bytes: 0,
            total_bytes: entry.size_bytes,
            overall_downloaded_bytes: overall_completed_before,
            overall_total_bytes,
            bytes_per_second: 0,
        });

        let mut response = agent
            .get(&entry.url)
            .header(
                "User-Agent",
                concat!("KromaDesktop/", env!("CARGO_PKG_VERSION")),
            )
            .call()
            .map_err(|err| format!("Could not download {}: {err}", entry.name))?;

        let response_len = response
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|&len| len > 0);
        let expected_total = if entry.size_bytes > 0 {
            entry.size_bytes
        } else {
            response_len.unwrap_or(0)
        };

        let started_at = Instant::now();
        let mut last_emit = Instant::now();
        let mut reader = response.body_mut().as_reader();

        let (written, actual_sha) = stream_and_verify_to_file(
            &mut reader,
            &dest,
            entry.size_bytes,
            &entry.sha256,
            cancel_flag,
            |downloaded| {
                let now = Instant::now();
                if now.duration_since(last_emit) >= PROGRESS_EMIT_INTERVAL
                    || (expected_total > 0 && downloaded >= expected_total)
                {
                    last_emit = now;
                    let elapsed_secs = started_at.elapsed().as_secs_f64().max(0.05);
                    let bps = (downloaded as f64 / elapsed_secs).round().max(0.0) as u64;
                    on_progress(ModelDownloadProgressDto {
                        phase: "downloading",
                        model_id: entry.id.clone(),
                        model_name: entry.name.clone(),
                        index: idx + 1,
                        total_count,
                        downloaded_bytes: downloaded,
                        total_bytes: expected_total,
                        overall_downloaded_bytes: overall_completed_before
                            .saturating_add(downloaded),
                        overall_total_bytes: overall_total_bytes
                            .max(overall_completed_before.saturating_add(downloaded)),
                        bytes_per_second: bps,
                    });
                }
            },
        )
        .map_err(|err| format!("{}: {err}", entry.name))?;

        overall_completed_before = overall_completed_before.saturating_add(written);

        installed_index.entries.insert(
            entry.id.clone(),
            InstalledModelReceipt {
                version: entry.version.clone(),
                sha256: actual_sha,
                size_bytes: written,
            },
        );
        save_installed_index(managed_models_dir, &installed_index)?;
    }

    on_progress(ModelDownloadProgressDto {
        phase: "completed",
        model_id: String::new(),
        model_name: "AI models verified".to_string(),
        index: total_count,
        total_count,
        downloaded_bytes: overall_completed_before,
        total_bytes: overall_total_bytes.max(overall_completed_before),
        overall_downloaded_bytes: overall_completed_before,
        overall_total_bytes: overall_total_bytes.max(overall_completed_before),
        bytes_per_second: 0,
    });

    inspect_manifest_status(manifest, managed_models_dir, fallback_search_dirs)
}

pub fn sanitize_installer_file_name(raw: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty()
        || trimmed.contains('/')
        || trimmed.contains('\\')
        || trimmed.contains("..")
        || trimmed.chars().any(char::is_control)
    {
        return Err(format!("invalid installer filename '{raw}'"));
    }
    let lower = trimmed.to_ascii_lowercase();
    if !(lower.ends_with(".exe") || lower.ends_with(".msi") || lower.ends_with(".dmg")) {
        return Err(format!(
            "unsupported installer extension in '{raw}' (expected .exe, .msi, or .dmg)"
        ));
    }
    Ok(trimmed.to_string())
}

/// Streams a desktop installer bundle (`.exe`, `.msi`, or `.dmg`) from GitHub Releases into
/// `%TEMP%/kroma_updates/<asset_name>`, emitting live progress updates for the in-app updater modal.
pub fn download_app_update_installer<F>(
    url: &str,
    asset_name: &str,
    expected_size: u64,
    mut on_progress: F,
) -> Result<PathBuf, String>
where
    F: FnMut(AppUpdateDownloadProgressDto),
{
    let trimmed_url = url.trim();
    if !trimmed_url.starts_with("https://") || trimmed_url.chars().any(char::is_control) {
        return Err("Update download URL must use https://".to_string());
    }
    let safe_name = sanitize_installer_file_name(asset_name)?;
    let updates_dir = std::env::temp_dir().join("kroma_updates");
    fs::create_dir_all(&updates_dir)
        .map_err(|err| format!("failed to prepare updates directory: {err}"))?;
    let dest = updates_dir.join(&safe_name);

    on_progress(AppUpdateDownloadProgressDto {
        phase: "downloading",
        asset_name: safe_name.clone(),
        downloaded_bytes: 0,
        total_bytes: expected_size,
        bytes_per_second: 0,
    });

    let agent = build_download_agent();
    let mut response = agent
        .get(trimmed_url)
        .header(
            "User-Agent",
            concat!("KromaDesktop/", env!("CARGO_PKG_VERSION")),
        )
        .call()
        .map_err(|err| format!("Could not download {safe_name}: {err}"))?;

    let response_len = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&len| len > 0);
    let expected_total = if expected_size > 0 {
        expected_size
    } else {
        response_len.unwrap_or(0)
    };

    let started_at = Instant::now();
    let mut last_emit = Instant::now();
    let mut reader = response.body_mut().as_reader();
    let cancel_flag = AtomicBool::new(false);

    let (written, _) = stream_and_verify_to_file(
        &mut reader,
        &dest,
        expected_size,
        "",
        &cancel_flag,
        |downloaded| {
            let now = Instant::now();
            if now.duration_since(last_emit) >= PROGRESS_EMIT_INTERVAL
                || (expected_total > 0 && downloaded >= expected_total)
            {
                last_emit = now;
                let elapsed_secs = started_at.elapsed().as_secs_f64().max(0.05);
                let bps = (downloaded as f64 / elapsed_secs).round().max(0.0) as u64;
                on_progress(AppUpdateDownloadProgressDto {
                    phase: "downloading",
                    asset_name: safe_name.clone(),
                    downloaded_bytes: downloaded,
                    total_bytes: expected_total.max(downloaded),
                    bytes_per_second: bps,
                });
            }
        },
    )?;

    on_progress(AppUpdateDownloadProgressDto {
        phase: "launching",
        asset_name: safe_name,
        downloaded_bytes: written,
        total_bytes: expected_total.max(written),
        bytes_per_second: 0,
    });

    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample_manifest_with_payload(payload: &[u8]) -> (ModelManifest, String) {
        let mut hasher = Sha256::new();
        hasher.update(payload);
        let sha = hex::encode(hasher.finalize());
        let manifest = ModelManifest {
            schema_version: 1,
            release_tag: "models-v1".to_string(),
            models: vec![ModelManifestEntry {
                id: "test-model".to_string(),
                name: "Test Model".to_string(),
                category: "face_detection".to_string(),
                version: "1.0.0".to_string(),
                relative_path: "ocr/test_model.onnx".to_string(),
                url: "https://example.invalid/test_model.onnx".to_string(),
                size_bytes: payload.len() as u64,
                sha256: sha.clone(),
                required: true,
            }],
        };
        (manifest, sha)
    }

    #[test]
    fn download_agent_uses_native_tls_with_platform_roots() {
        let agent = build_download_agent();
        let tls = agent.config().tls_config();
        assert_eq!(tls.provider(), ureq::tls::TlsProvider::NativeTls);
        assert!(matches!(
            tls.root_certs(),
            ureq::tls::RootCerts::PlatformVerifier
        ));
    }

    #[test]
    fn embedded_manifest_parses_all_six_active_models() {
        let manifest = load_models_manifest(&[]).expect("embedded manifest should parse");
        assert_eq!(manifest.schema_version, 1);
        assert_eq!(manifest.release_tag, "models-v1");
        assert_eq!(manifest.models.len(), 8);
        let required_count = manifest.models.iter().filter(|m| m.required).count();
        let optional_count = manifest.models.iter().filter(|m| !m.required).count();
        assert_eq!(required_count, 6);
        assert_eq!(optional_count, 2);
        assert!(manifest.models.iter().any(|m| m.id == "bisenet-face-parsing" && !m.required));
        assert!(manifest.models.iter().any(|m| m.id == "lama-inpainting" && !m.required));
        assert!(manifest.models.iter().all(|m| !m.sha256.is_empty() && m.size_bytes > 0));
    }

    #[test]
    fn sanitize_relative_path_rejects_traversal() {
        assert!(sanitize_relative_path("../escape.onnx").is_err());
        assert!(sanitize_relative_path("/abs/model.onnx").is_err());
        assert_eq!(
            sanitize_relative_path("ocr/ppocrv5_mobile_det.onnx").unwrap(),
            PathBuf::from("ocr").join("ppocrv5_mobile_det.onnx")
        );
    }

    #[test]
    fn sanitize_installer_file_name_accepts_exe_msi_dmg_and_rejects_traversal() {
        assert_eq!(
            sanitize_installer_file_name("Kroma_0.1.1_x64-setup.exe").unwrap(),
            "Kroma_0.1.1_x64-setup.exe"
        );
        assert_eq!(
            sanitize_installer_file_name("Kroma_0.1.1_aarch64.dmg").unwrap(),
            "Kroma_0.1.1_aarch64.dmg"
        );
        assert!(sanitize_installer_file_name("../Kroma_0.1.1_x64-setup.exe").is_err());
        assert!(sanitize_installer_file_name("Kroma_0.1.1.bat").is_err());
    }

    #[test]
    fn inspect_classifies_missing_ready_and_outdated_models() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("kroma_model_mgr_test_{stamp}"));
        let managed = root.join("managed");
        let fallback = root.join("fallback");
        let _ = fs::create_dir_all(&managed);
        let _ = fs::create_dir_all(&fallback);

        let payload = b"kroma-onnx-dummy-bytes";
        let (mut manifest, sha) = sample_manifest_with_payload(payload);

        // 1. Missing initially
        let s1 = inspect_manifest_status(&manifest, &managed, std::slice::from_ref(&fallback))
            .expect("inspect 1");
        assert!(!s1.all_ready);
        assert_eq!(s1.missing_count, 1);
        assert_eq!(s1.entries[0].status, "missing");

        // 2. Stream write & verify into managed dir -> ready
        let dest = managed.join("ocr").join("test_model.onnx");
        let cancel = AtomicBool::new(false);
        let mut cursor = Cursor::new(payload.to_vec());
        let (written, actual_sha) =
            stream_and_verify_to_file(&mut cursor, &dest, payload.len() as u64, &sha, &cancel, |_| {
            })
            .expect("stream verify should succeed");
        assert_eq!(written, payload.len() as u64);
        assert_eq!(actual_sha, sha);

        let mut idx = InstalledModelsIndex::default();
        idx.entries.insert(
            "test-model".to_string(),
            InstalledModelReceipt {
                version: "1.0.0".to_string(),
                sha256: actual_sha,
                size_bytes: written,
            },
        );
        save_installed_index(&managed, &idx).expect("save receipt");

        let s2 = inspect_manifest_status(&manifest, &managed, std::slice::from_ref(&fallback))
            .expect("inspect 2");
        assert!(s2.all_ready);
        assert_eq!(s2.missing_count, 0);
        assert_eq!(s2.entries[0].status, "ready");

        // 3. Version bump in manifest -> outdated
        manifest.models[0].version = "2.0.0".to_string();
        let s3 = inspect_manifest_status(&manifest, &managed, std::slice::from_ref(&fallback))
            .expect("inspect 3");
        assert!(!s3.all_ready);
        assert_eq!(s3.outdated_count, 1);
        assert_eq!(s3.entries[0].status, "outdated");

        // 4. SHA-256 mismatch rejects & cleans up .part file
        let bad_dest = managed.join("bad.onnx");
        let mut bad_cursor = Cursor::new(b"tampered".to_vec());
        let err = stream_and_verify_to_file(
            &mut bad_cursor,
            &bad_dest,
            8,
            "0000000000000000000000000000000000000000000000000000000000000000",
            &cancel,
            |_| {},
        );
        assert!(err.is_err());
        assert!(!bad_dest.exists());
        assert!(!PathBuf::from(format!("{}.part", bad_dest.display())).exists());

        let _ = fs::remove_dir_all(&root);
    }
}
