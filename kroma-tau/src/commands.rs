use crate::dto::{
    build_face_overlay, confidence_label, AmbiguousCandidateDto, BatchSyncResultDto,
    ClearCacheResultDto, ClusterDto, ClusterSlateOcrResultDto, DevelopRecipeDto,
    DevelopSyncMaskDto, DirectorySummaryDto, EstimateExportSizeRequestDto,
    EstimateExportSizeResponseDto, ExportShootRequestDto, ExportShootResultDto, FaceOverlayDto,
    FaceParseMapDto, GeneratedAiMaskDto, GeneratedInpaintPatchDto, HistogramDto,
    ImportedCubeLutDto, InpaintStrokeDto, ModelDownloadOutcomeDto, ModelInspectionDto,
    ModelManifestStatusDto, OcrTextLineDto, PhotoDto, PhotoOcrResultDto, PhotoRecipeSaveItemDto,
    PreviewPatchDto, PreviewsReadyBatchDto, RelativeDevelopDeltaDto, RuntimeStatusDto,
    ScanProgressDto, ShootSessionDto, SlateRenameApplyItemDto, StorageCacheSummaryDto,
    ThumbnailRenderProgressDto, VirtualFolderDto, WhiteBalanceDto,
};
use crate::face_cache;
use crate::model_manager::{
    download_app_update_installer, download_optional_model_by_id, download_pending_models,
    inspect_manifest_status, load_models_manifest, APP_UPDATE_PROGRESS_EVENT,
    MODEL_DOWNLOAD_PROGRESS_EVENT,
};
use crate::state::{
    ActiveShootContext, AppState, CachedDevelopBuffer, CachedOcrEngine, DevelopDecodePriority,
    OrganizationHistory, ShootSessionMode, ThumbnailProgressSnapshot, ThumbnailProgressTracker,
};
use image::codecs::jpeg::JpegEncoder;
use kroma_app::ports::develop_engine::{ImageDevelopPort, OutputImageFormat};
use kroma_app::ports::organization_plan_store::OrganizationPlanStorePort;
use kroma_app::ports::organizer::OrganizeTransferMode;
use kroma_app::ports::recipe_store::DevelopRecipeStorePort;
use kroma_app::use_cases::auto_wb_from_slate::{
    CalibrateWhiteBalanceFromSlateInput, CalibrateWhiteBalanceFromSlateUseCase,
};
use kroma_app::use_cases::batch_nudge_develop::{
    BatchRelativeNudgeInput, BatchRelativeNudgeUseCase,
};
use kroma_app::use_cases::batch_sync_develop::{BatchSyncDevelopInput, BatchSyncDevelopUseCase};
use kroma_app::use_cases::cluster_shoot::{cluster_preextracted_scans, PhotoScanTarget};
use kroma_app::use_cases::edit_photo::{
    SavePhotoDevelopRecipeInput, SavePhotoDevelopRecipeUseCase,
};
use kroma_app::use_cases::organize_shoot::{
    OrganizeExecutionMode, OrganizeShootFoldersUseCase, OrganizeShootInput,
};
use kroma_domain::develop::{CropTransform, DevelopRecipe, NormalizedRect};
use kroma_domain::masking::AiMaskKind;
use kroma_domain::identity::{
    parse_filename_capture_hints, BoundingBox, ClusterId, ClusterPhotoEntry, DetectedFace,
    FaceEmbedding, MatchConfidence, Point2D, ShootOrganizationPlan, SimilarityThresholds,
    SubjectCluster, VirtualFolderId, YuNetLandmarks, SFACE_EMBEDDING_DIM,
};
use kroma_domain::organization::OutputNamingTemplate;
use kroma_domain::photo::PhotoId;
use kroma_domain::retouch::{FaceParseMap, PortraitFaceGuide};
use kroma_infra::develop::cube_lut::CubeLut3D;
use kroma_infra::develop::image_decoder::{
    decode_develop_buffer, decode_fast_preview_max_dim, decode_high_bit_depth_working_buffer,
    decode_kr16_downscaled_rgb32f, decode_working_wire_downscaled_rgb32f,
    downscale_rgb32f_unclamped, encode_kr16_rgba16f, encode_kr32_rgba32f,
    extract_exif_capture_timestamp_epoch_sec, is_16bit_png_path, is_raw_photo_path,
    is_supported_photo_path, is_tiff_photo_path, is_web_native_image_path,
    read_native_photo_dimensions, supported_raw_extensions, DecodedDevelopBuffer,
};
use kroma_infra::develop::pixel_engine::RayonImageDevelopEngine;
use kroma_infra::develop::wgpu_engine::{
    is_wgpu_develop_available, wgpu_develop_adapter_name, wgpu_develop_backend_label,
};
use kroma_infra::organizer::local_file_organizer::LocalFileOrganizer;
use kroma_infra::storage::json_organization_plan_store::JsonOrganizationPlanStore;
use kroma_infra::storage::json_sidecar_store::JsonSidecarRecipeStore;
use kroma_infra::vision::face_parser::OnnxBiSeNetFaceParser;
use kroma_infra::vision::lama_inpainter::{
    generate_inpaint_patch, InpaintStrokeInput, OnnxLamaInpainter,
};
use kroma_infra::vision::model_inspector::{
    inspect_onnx_model as inspect_onnx_model_file, ModelCompatibility, ModelRole,
};
use kroma_infra::vision::onnx_runtime::{
    gpu_execution_provider_label, is_gpu_acceleration_enabled,
    is_gpu_execution_provider_available, set_gpu_acceleration_enabled,
};
use kroma_infra::vision::ppocr_v5::{PpOcrV5Config, PpOcrV5Engine};
use kroma_infra::vision::subject_segmenter::{
    generate_ai_mask_for_rgb_with_face_parser, OnnxSubjectSegmenter,
};
use kroma_infra::vision::yunet_detector::{YuNetConfig, YuNetOnnxDetector};
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager, State};

const SCAN_PROGRESS_EVENT: &str = "kroma://scan-progress";
const PREVIEWS_READY_EVENT: &str = "kroma://previews-ready";
const THUMBNAIL_RENDER_PROGRESS_EVENT: &str = "kroma://thumbnail-render-progress";
const PREVIEW_MAX_DIM: u32 = 1024;
const THUMBNAIL_32BIT_MAX_DIM: u32 = 640;
const THUMBNAIL_32BIT_JPEG_QUALITY: u8 = 86;
const THUMBNAIL_32BIT_BATCH_SIZE: usize = 6;
const MAX_DISK_WORK_CACHE_FILES: usize = 24;
const OCR_HIGH_RES_RETRY_CONFIDENCE: f32 = 0.85;
const INITIAL_SYNC_RAW_PREVIEWS: usize = 36;
const BACKGROUND_PREVIEW_BATCH_SIZE: usize = 32;
/// Standard JPEG/PNG/WebP files at or above this byte size are downscaled into cached `1024px`
/// JPEG preview thumbnails (`v3_web_*_preview.jpg`) so WebView2 never decodes 24–60 MP originals
/// inside contact sheets, filmstrips, export lists, or Home recent cards.
const WEB_NATIVE_PREVIEW_PROXY_MIN_BYTES: u64 = 300_000;

fn emit_progress(app: &AppHandle, phase: &'static str, completed: usize, total: usize, msg: String) {
    let _ = app.emit(
        SCAN_PROGRESS_EVENT,
        ScanProgressDto {
            phase,
            completed,
            total,
            message: msg,
        },
    );
}

fn persist_organization_plan(ctx: &ActiveShootContext) -> Result<(), String> {
    // Only full folder shoots own the folder's organization_plan.json; single-photo or ad-hoc
    // selected-file sessions must never overwrite a parent folder's saved clustering plan.
    if !ctx.is_clustered || ctx.session_mode != ShootSessionMode::Folder {
        return Ok(());
    }
    JsonOrganizationPlanStore::new(&ctx.sidecars_dir)
        .save_plan(&ctx.plan)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn exit_app(app: AppHandle) {
    app.exit(0);
}

#[tauri::command]
pub fn open_external_url(url: String) -> Result<(), String> {
    let trimmed = url.trim();
    if !(trimmed.starts_with("https://") || trimmed.starts_with("http://")) {
        return Err("Only http:// and https:// URLs are permitted".to_string());
    }
    if trimmed.chars().any(char::is_control) {
        return Err("URL contains invalid control characters".to_string());
    }

    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("rundll32")
            .args(["url.dll,FileProtocolHandler", trimmed])
            .spawn()
            .map_err(|err| format!("Failed to open browser: {err}"))?;
    }

    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(trimmed)
            .spawn()
            .map_err(|err| format!("Failed to open browser: {err}"))?;
    }

    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        std::process::Command::new("xdg-open")
            .arg(trimmed)
            .spawn()
            .map_err(|err| format!("Failed to open browser: {err}"))?;
    }

    Ok(())
}

#[tauri::command]
pub async fn download_and_install_app_update(
    url: String,
    asset_name: String,
    size_bytes: Option<u64>,
    app: AppHandle,
) -> Result<String, String> {
    let app_for_progress = app.clone();
    let expected_size = size_bytes.unwrap_or(0);
    let installer_path = tauri::async_runtime::spawn_blocking(move || {
        download_app_update_installer(&url, &asset_name, expected_size, |progress| {
            let _ = app_for_progress.emit(APP_UPDATE_PROGRESS_EVENT, progress);
        })
    })
    .await
    .map_err(|err| format!("app update download worker failed: {err}"))??;

    #[cfg(target_os = "windows")]
    {
        let is_msi = installer_path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("msi"));
        if is_msi {
            std::process::Command::new("msiexec")
                .arg("/i")
                .arg(&installer_path)
                .spawn()
                .map_err(|err| format!("Failed to launch MSI installer: {err}"))?;
        } else {
            std::process::Command::new(&installer_path)
                .spawn()
                .map_err(|err| format!("Failed to launch installer: {err}"))?;
        }
    }

    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(&installer_path)
            .spawn()
            .map_err(|err| format!("Failed to open DMG installer: {err}"))?;
    }

    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        std::process::Command::new("xdg-open")
            .arg(&installer_path)
            .spawn()
            .map_err(|err| format!("Failed to open installer: {err}"))?;
    }

    // Exit Kroma shortly after spawning the installer so Windows NSIS can replace the binary cleanly
    let app_for_exit = app.clone();
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(650));
        app_for_exit.exit(0);
    });

    Ok(installer_path.to_string_lossy().to_string())
}

#[tauri::command]
pub fn reveal_main_window(window: tauri::WebviewWindow) {
    if !window.is_visible().unwrap_or(true) {
        let _ = window.show();
        let _ = window.set_focus();
    }
}

#[tauri::command]
pub async fn import_cube_lut_file(
    source_path: String,
    app: AppHandle,
) -> Result<ImportedCubeLutDto, String> {
    let luts_dir = app
        .path()
        .app_data_dir()
        .unwrap_or_else(|_| std::env::temp_dir().join("kroma_app_data"))
        .join("luts");

    tauri::async_runtime::spawn_blocking(move || {
        let src = Path::new(source_path.trim());
        if !src.is_file() {
            return Err(format!("LUT file '{}' not found", src.display()));
        }
        let content =
            fs::read_to_string(src).map_err(|err| format!("Failed to read .cube file: {err}"))?;
        let parsed = CubeLut3D::parse(&content).map_err(|err| err.to_string())?;

        let raw_name = src
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("Custom_LUT.cube")
            .trim();
        let safe_file_name: String = raw_name
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | ' ') {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let final_file_name = if safe_file_name.to_ascii_lowercase().ends_with(".cube") {
            safe_file_name
        } else {
            format!("{safe_file_name}.cube")
        };

        fs::create_dir_all(&luts_dir)
            .map_err(|err| format!("Failed to create LUT library directory: {err}"))?;
        let dest = luts_dir.join(&final_file_name);
        fs::write(&dest, &content)
            .map_err(|err| format!("Failed to save LUT to library: {err}"))?;

        Ok(ImportedCubeLutDto {
            name: final_file_name,
            cube_path: dest.to_string_lossy().to_string(),
            grid_size: parsed.size(),
            content,
        })
    })
    .await
    .map_err(|err| format!("import .cube LUT worker failed: {err}"))?
}

#[tauri::command]
pub async fn read_cube_lut_file(cube_path: String) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let p = Path::new(cube_path.trim());
        if !p.is_file() {
            return Err(format!("LUT file '{}' not found", p.display()));
        }
        if p.extension()
            .and_then(|e| e.to_str())
            .is_none_or(|ext| !ext.eq_ignore_ascii_case("cube"))
        {
            return Err("Only .cube 3D LUT files can be loaded".to_string());
        }
        fs::read_to_string(p).map_err(|err| format!("Failed to read .cube file: {err}"))
    })
    .await
    .map_err(|err| format!("read .cube LUT worker failed: {err}"))?
}

fn build_runtime_status_dto(state: &AppState) -> RuntimeStatusDto {
    let (yunet, recognizer) = state.resolve_model_paths();
    let (default_yunet, default_recognizer) = state.resolve_default_model_paths();
    let (ocr_det, ocr_rec, ocr_dict) = state.resolve_ocr_model_paths();
    let segmenter = state.resolve_segmenter_model_path();
    let default_segmenter = state.resolve_default_segmenter_model_path();
    let face_parser = state.resolve_face_parser_model_path();
    let lama = state.resolve_lama_model_path();
    let detector_model_label = AppState::detector_model_label(yunet.as_deref());
    let recognition_model_label = AppState::recognition_model_label(recognizer.as_deref());
    let default_detector_model_label = AppState::detector_model_label(default_yunet.as_deref());
    let default_recognition_model_label =
        AppState::recognition_model_label(default_recognizer.as_deref());
    let ocr_det_model_label = AppState::ocr_det_model_label(ocr_det.as_deref());
    let ocr_rec_model_label = AppState::ocr_rec_model_label(ocr_rec.as_deref());
    let segmenter_model_label = AppState::segmenter_model_label(segmenter.as_deref());
    let default_segmenter_model_label =
        AppState::segmenter_model_label(default_segmenter.as_deref());
    let face_parser_model_label = AppState::face_parser_model_label(face_parser.as_deref());
    let lama_model_label = AppState::lama_model_label(lama.as_deref());
    let models_ready = yunet.is_some() && recognizer.is_some();
    let ocr_models_ready = ocr_det.is_some() && ocr_rec.is_some() && ocr_dict.is_some();
    let segmenter_models_ready = segmenter.is_some();
    let face_parser_ready = face_parser.is_some();
    let lama_ready = lama.is_some();
    let gpu_available = is_gpu_execution_provider_available() || is_wgpu_develop_available();
    let gpu_acceleration_enabled = is_gpu_acceleration_enabled();
    let backend_label = if is_gpu_execution_provider_available() && gpu_acceleration_enabled {
        gpu_execution_provider_label().to_string()
    } else {
        "CPU".to_string()
    };
    let develop_gpu_backend_label = if gpu_acceleration_enabled && is_wgpu_develop_available() {
        wgpu_develop_backend_label()
    } else {
        "CPU".to_string()
    };
    let develop_gpu_adapter_name = wgpu_develop_adapter_name();

    RuntimeStatusDto {
        yunet_model_path: yunet.map(|p| p.to_string_lossy().to_string()),
        sface_model_path: recognizer.map(|p| p.to_string_lossy().to_string()),
        detector_model_label,
        recognition_model_label,
        default_yunet_model_path: default_yunet.map(|p| p.to_string_lossy().to_string()),
        default_sface_model_path: default_recognizer.map(|p| p.to_string_lossy().to_string()),
        default_detector_model_label,
        default_recognition_model_label,
        is_custom_yunet: state.has_custom_yunet(),
        is_custom_sface: state.has_custom_recognizer(),
        models_ready,
        ocr_models_ready,
        ocr_det_model_label,
        ocr_rec_model_label,
        ocr_det_model_path: ocr_det.map(|p| p.to_string_lossy().to_string()),
        ocr_rec_model_path: ocr_rec.map(|p| p.to_string_lossy().to_string()),
        segmenter_model_path: segmenter.map(|p| p.to_string_lossy().to_string()),
        segmenter_model_label,
        default_segmenter_model_path: default_segmenter.map(|p| p.to_string_lossy().to_string()),
        default_segmenter_model_label,
        is_custom_segmenter: state.has_custom_segmenter(),
        segmenter_models_ready,
        face_parser_model_path: face_parser.map(|p| p.to_string_lossy().to_string()),
        face_parser_model_label,
        is_custom_face_parser: state.has_custom_face_parser(),
        face_parser_ready,
        lama_model_path: lama.map(|p| p.to_string_lossy().to_string()),
        lama_model_label,
        is_custom_lama: state.has_custom_lama(),
        lama_ready,
        supported_raw_count: supported_raw_extensions().len(),
        backend_label,
        develop_gpu_backend_label,
        develop_gpu_adapter_name,
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        gpu_available,
        gpu_acceleration_enabled,
    }
}

#[tauri::command]
pub fn check_runtime_status(state: State<'_, AppState>) -> RuntimeStatusDto {
    build_runtime_status_dto(&state)
}

#[tauri::command]
pub async fn check_models_manifest(
    state: State<'_, AppState>,
) -> Result<ModelManifestStatusDto, String> {
    let managed_dir = state.resolve_managed_models_dir();
    let search_dirs = state.model_search_dirs();
    tauri::async_runtime::spawn_blocking(move || {
        let manifest = load_models_manifest(&search_dirs)?;
        inspect_manifest_status(&manifest, &managed_dir, &search_dirs)
    })
    .await
    .map_err(|err| format!("check models manifest worker failed: {err}"))?
}

#[tauri::command]
pub async fn download_missing_models(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<ModelDownloadOutcomeDto, String> {
    let managed_dir = state.resolve_managed_models_dir();
    let search_dirs = state.model_search_dirs();
    let cancel_flag = Arc::clone(&state.model_download_cancel);
    let app_handle = app.clone();

    let manifest_status = tauri::async_runtime::spawn_blocking(move || {
        let manifest = load_models_manifest(&search_dirs)?;
        download_pending_models(
            &manifest,
            &managed_dir,
            &search_dirs,
            &cancel_flag,
            |progress| {
                let _ = app_handle.emit(MODEL_DOWNLOAD_PROGRESS_EVENT, progress);
            },
        )
    })
    .await
    .map_err(|err| format!("model download worker failed: {err}"))??;

    // Clear warmed engine caches so newly downloaded models are picked up immediately
    if let Ok(mut guard) = state.cached_ocr_engine.lock() {
        *guard = None;
    }
    if let Ok(mut guard) = state.cached_segmenter_engine.lock() {
        *guard = None;
    }
    if let Ok(mut guard) = state.cached_face_parser_engine.lock() {
        *guard = None;
    }
    if let Ok(mut guard) = state.cached_lama_engine.lock() {
        *guard = None;
    }
    if let Ok(mut guard) = state.cached_yunet_engine.lock() {
        *guard = None;
    }

    let runtime = build_runtime_status_dto(&state);
    Ok(ModelDownloadOutcomeDto {
        manifest: manifest_status,
        runtime,
    })
}

#[tauri::command]
pub async fn download_optional_model(
    model_id: String,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<ModelDownloadOutcomeDto, String> {
    let managed_dir = state.resolve_managed_models_dir();
    let search_dirs = state.model_search_dirs();
    let cancel_flag = Arc::clone(&state.model_download_cancel);
    let app_handle = app.clone();

    let manifest_status = tauri::async_runtime::spawn_blocking(move || {
        let manifest = load_models_manifest(&search_dirs)?;
        download_optional_model_by_id(
            &manifest,
            &model_id,
            &managed_dir,
            &search_dirs,
            &cancel_flag,
            |progress| {
                let _ = app_handle.emit(MODEL_DOWNLOAD_PROGRESS_EVENT, progress);
            },
        )
    })
    .await
    .map_err(|err| format!("optional model download worker failed: {err}"))??;

    if let Ok(mut guard) = state.cached_face_parser_engine.lock() {
        *guard = None;
    }
    if let Ok(mut guard) = state.cached_lama_engine.lock() {
        *guard = None;
    }

    let runtime = build_runtime_status_dto(&state);
    Ok(ModelDownloadOutcomeDto {
        manifest: manifest_status,
        runtime,
    })
}

#[tauri::command]
pub fn cancel_model_downloads(state: State<'_, AppState>) {
    state.model_download_cancel.store(true, Ordering::Relaxed);
}

#[tauri::command]
pub fn set_gpu_acceleration(enabled: bool, state: State<'_, AppState>) -> RuntimeStatusDto {
    set_gpu_acceleration_enabled(enabled);
    if let Ok(mut guard) = state.cached_ocr_engine.lock() {
        *guard = None;
    }
    if let Ok(mut guard) = state.cached_segmenter_engine.lock() {
        *guard = None;
    }
    if let Ok(mut guard) = state.cached_face_parser_engine.lock() {
        *guard = None;
    }
    if let Ok(mut guard) = state.cached_lama_engine.lock() {
        *guard = None;
    }
    build_runtime_status_dto(&state)
}

fn is_preview_cache_file(file_name: &str) -> bool {
    file_name.ends_with("_preview.jpg") || file_name.ends_with("_preview.kr16")
}

fn is_face_cache_file(file_name: &str) -> bool {
    file_name.starts_with("face_cache_") && file_name.ends_with(".json")
}

fn summarize_cache_dir(dir: &Path, is_custom_cache_dir: bool) -> StorageCacheSummaryDto {
    let mut preview_cache_bytes = 0u64;
    let mut preview_file_count = 0usize;
    let mut face_cache_bytes = 0u64;
    let mut face_cache_file_count = 0usize;

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let Ok(meta) = entry.metadata() else {
                continue;
            };
            if !meta.is_file() {
                continue;
            }
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if is_preview_cache_file(&name_str) {
                preview_cache_bytes = preview_cache_bytes.saturating_add(meta.len());
                preview_file_count += 1;
            } else if is_face_cache_file(&name_str) {
                face_cache_bytes = face_cache_bytes.saturating_add(meta.len());
                face_cache_file_count += 1;
            }
        }
    }

    StorageCacheSummaryDto {
        cache_dir_path: dir.to_string_lossy().to_string(),
        is_custom_cache_dir,
        preview_cache_bytes,
        preview_file_count,
        face_cache_bytes,
        face_cache_file_count,
    }
}

fn clear_cache_dir_entries(dir: &Path, target: &str) -> ClearCacheResultDto {
    let clear_previews = target == "previews" || target == "all";
    let clear_faces = target == "face_cache" || target == "all";
    let mut removed_files = 0usize;
    let mut freed_bytes = 0u64;

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let Ok(meta) = entry.metadata() else {
                continue;
            };
            if !meta.is_file() {
                continue;
            }
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            let should_remove = (clear_previews && is_preview_cache_file(&name_str))
                || (clear_faces && is_face_cache_file(&name_str));
            if should_remove && fs::remove_file(entry.path()).is_ok() {
                removed_files += 1;
                freed_bytes = freed_bytes.saturating_add(meta.len());
            }
        }
    }

    ClearCacheResultDto {
        removed_files,
        freed_bytes,
    }
}

#[tauri::command]
pub async fn get_storage_cache_summary(
    state: State<'_, AppState>,
) -> Result<StorageCacheSummaryDto, String> {
    let dir = state.resolve_cache_dir();
    let is_custom = state.is_custom_cache_dir();
    tauri::async_runtime::spawn_blocking(move || summarize_cache_dir(&dir, is_custom))
        .await
        .map_err(|err| format!("storage summary worker failed: {err}"))
}

#[tauri::command]
pub async fn set_storage_cache_dir(
    cache_dir: Option<String>,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<StorageCacheSummaryDto, String> {
    let next_path = cache_dir
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from);
    let resolved = state.set_custom_cache_dir(next_path)?;
    let is_custom = state.is_custom_cache_dir();
    invalidate_thumbnail_renders(&state, Some(&app), "edited_32bit", false);
    state.clear_develop_buffer_cache();
    if let Ok(mut guard) = state.active_shoot.lock()
        && let Some(ctx) = guard.as_mut()
    {
        ctx.rendered_preview_paths.clear();
    }
    tauri::async_runtime::spawn_blocking(move || summarize_cache_dir(&resolved, is_custom))
        .await
        .map_err(|err| format!("set cache dir worker failed: {err}"))
}

#[tauri::command]
pub async fn clear_storage_cache(
    target: String,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<ClearCacheResultDto, String> {
    let normalized = target.trim().to_ascii_lowercase();
    if normalized != "previews" && normalized != "face_cache" && normalized != "all" {
        return Err(format!("unknown cache target '{target}'"));
    }

    if normalized == "previews" || normalized == "all" {
        invalidate_thumbnail_renders(&state, Some(&app), "edited_32bit", false);
        state.clear_develop_buffer_cache();
        if let Ok(mut guard) = state.active_shoot.lock()
            && let Some(ctx) = guard.as_mut()
        {
            ctx.rendered_preview_paths.clear();
        }
    }
    if (normalized == "face_cache" || normalized == "all")
        && let Ok(mut guard) = state.active_shoot.lock()
        && let Some(ctx) = guard.as_mut()
    {
        ctx.cached_face_scans = None;
    }

    let dir = state.resolve_cache_dir();
    tauri::async_runtime::spawn_blocking(move || clear_cache_dir_entries(&dir, &normalized))
        .await
        .map_err(|err| format!("clear cache worker failed: {err}"))
}

#[tauri::command]
pub async fn inspect_onnx_model(
    model_path: String,
    role: String,
) -> Result<ModelInspectionDto, String> {
    let parsed_role = match role.to_ascii_lowercase().as_str() {
        "detector" => ModelRole::Detector,
        "recognizer" => ModelRole::Recognizer,
        "segmenter" => ModelRole::Segmenter,
        "face_parser" | "faceparser" => ModelRole::FaceParser,
        "inpainter" | "lama" => ModelRole::Inpainter,
        other => return Err(format!("unknown model role '{other}'")),
    };

    tauri::async_runtime::spawn_blocking(move || {
        let report = inspect_onnx_model_file(Path::new(&model_path), parsed_role);
        ModelInspectionDto {
            compatibility: report.compatibility.as_str(),
            summary: report.summary,
            warnings: report.warnings,
            detected_label: report.detected_label,
            input_shape: report.input_shape,
            output_shape: report.output_shape,
            embedding_dim: report.embedding_dim,
            file_size_bytes: report.file_size_bytes,
        }
    })
    .await
    .map_err(|err| format!("model inspection worker failed: {err}"))
}

#[tauri::command]
pub async fn set_custom_ai_models(
    yunet_path: Option<String>,
    recognizer_path: Option<String>,
    segmenter_path: Option<String>,
    face_parser_path: Option<String>,
    lama_path: Option<String>,
    state: State<'_, AppState>,
) -> Result<RuntimeStatusDto, String> {
    let next_yunet = yunet_path
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from);
    let next_recognizer = recognizer_path
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from);
    let next_segmenter = segmenter_path
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from);
    let next_face_parser = face_parser_path
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from);
    let next_lama = lama_path
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from);

    // Backend defense-in-depth: block structurally incompatible ONNX files from activating
    let yunet_check = next_yunet.clone();
    let rec_check = next_recognizer.clone();
    let seg_check = next_segmenter.clone();
    let fp_check = next_face_parser.clone();
    let lama_check = next_lama.clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        if let Some(ref p) = yunet_check {
            let report = inspect_onnx_model_file(p, ModelRole::Detector);
            if report.compatibility == ModelCompatibility::Incompatible {
                return Err(format!(
                    "Incompatible face detector: {}",
                    report.warnings.first().cloned().unwrap_or(report.summary)
                ));
            }
        }
        if let Some(ref p) = rec_check {
            let report = inspect_onnx_model_file(p, ModelRole::Recognizer);
            if report.compatibility == ModelCompatibility::Incompatible {
                return Err(format!(
                    "Incompatible face recognizer: {}",
                    report.warnings.first().cloned().unwrap_or(report.summary)
                ));
            }
        }
        if let Some(ref p) = seg_check {
            let report = inspect_onnx_model_file(p, ModelRole::Segmenter);
            if report.compatibility == ModelCompatibility::Incompatible {
                return Err(format!(
                    "Incompatible subject segmenter: {}",
                    report.warnings.first().cloned().unwrap_or(report.summary)
                ));
            }
        }
        if let Some(ref p) = fp_check {
            let report = inspect_onnx_model_file(p, ModelRole::FaceParser);
            if report.compatibility == ModelCompatibility::Incompatible {
                return Err(format!(
                    "Incompatible face parser: {}",
                    report.warnings.first().cloned().unwrap_or(report.summary)
                ));
            }
        }
        if let Some(ref p) = lama_check {
            let report = inspect_onnx_model_file(p, ModelRole::Inpainter);
            if report.compatibility == ModelCompatibility::Incompatible {
                return Err(format!(
                    "Incompatible inpainting model: {}",
                    report.warnings.first().cloned().unwrap_or(report.summary)
                ));
            }
        }
        Ok(())
    })
    .await
    .map_err(|err| format!("model validation worker failed: {err}"))??;

    {
        let mut guard = state
            .custom_yunet_path
            .lock()
            .map_err(|_| "custom_yunet_path mutex poisoned".to_string())?;
        *guard = next_yunet;
    }
    {
        let mut guard = state
            .custom_recognizer_path
            .lock()
            .map_err(|_| "custom_recognizer_path mutex poisoned".to_string())?;
        *guard = next_recognizer;
    }
    {
        let mut guard = state
            .custom_segmenter_path
            .lock()
            .map_err(|_| "custom_segmenter_path mutex poisoned".to_string())?;
        *guard = next_segmenter;
    }
    {
        let mut guard = state
            .custom_face_parser_path
            .lock()
            .map_err(|_| "custom_face_parser_path mutex poisoned".to_string())?;
        *guard = next_face_parser;
    }
    {
        let mut guard = state
            .custom_lama_path
            .lock()
            .map_err(|_| "custom_lama_path mutex poisoned".to_string())?;
        *guard = next_lama;
    }
    if let Ok(mut seg_cache) = state.cached_segmenter_engine.lock() {
        *seg_cache = None;
    }
    if let Ok(mut fp_cache) = state.cached_face_parser_engine.lock() {
        *fp_cache = None;
    }
    if let Ok(mut lama_cache) = state.cached_lama_engine.lock() {
        *lama_cache = None;
    }
    if let Ok(mut yunet_cache) = state.cached_yunet_engine.lock() {
        *yunet_cache = None;
    }
    if let Ok(mut shoot_guard) = state.active_shoot.lock()
        && let Some(ctx) = shoot_guard.as_mut()
    {
        ctx.cached_face_scans = None;
        ctx.detected_face_parse_maps.clear();
    }

    Ok(build_runtime_status_dto(&state))
}

/// Fast Step 1: Opens folders of 2,000+ PNG/JPEG or camera RAW photos nearly instantaneously.
/// - Small web-native files (<300 KB) render directly in WebView2 with zero proxy encode.
/// - Camera RAWs, TIFFs, and large camera JPEGs/PNGs (>=300 KB) build compact cached `1024px`
///   thumbnails for the first 36 uncached photos immediately and stream the rest in the background.
#[tauri::command]
pub async fn load_shoot_folder(
    folder_path: String,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<ShootSessionDto, String> {
    state.clear_develop_buffer_cache();
    invalidate_thumbnail_renders(&state, Some(&app), "edited_32bit", false);
    state.export_estimate_generation.fetch_add(1, Ordering::SeqCst);
    let preview_cache_dir = state.resolve_cache_dir();
    let single_sidecars_dir = state.resolve_managed_single_sidecars_dir();
    let active_shoot = state.active_shoot.clone();
    let load_generation = state.load_generation.clone();
    let my_gen = load_generation.fetch_add(1, Ordering::SeqCst) + 1;

    tauri::async_runtime::spawn_blocking(move || {
        let (ctx, pending_bg_jobs) = run_instant_folder_load(
            folder_path,
            &preview_cache_dir,
            &single_sidecars_dir,
            &app,
        )?;
        let dto = build_session_dto(&ctx);
        {
            let mut guard = active_shoot
                .lock()
                .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
            *guard = Some(ctx);
        }

        if !pending_bg_jobs.is_empty() {
            spawn_background_preview_worker(
                pending_bg_jobs,
                active_shoot,
                load_generation,
                my_gen,
                app,
            );
        }

        Ok(dto)
    })
    .await
    .map_err(|err| format!("folder loader worker failed: {err}"))?
}

/// Opens 1 specific photo (entering distraction-free Single Photo Mode) or a curated selection
/// of specific photo files without scanning the rest of their parent folder.
/// Sidecar `.kroma.json` recipes still load and auto-save inside the parent folder's `.kroma_sidecars/`.
#[tauri::command]
pub async fn load_photo_files(
    file_paths: Vec<String>,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<ShootSessionDto, String> {
    state.clear_develop_buffer_cache();
    invalidate_thumbnail_renders(&state, Some(&app), "edited_32bit", false);
    state.export_estimate_generation.fetch_add(1, Ordering::SeqCst);
    let preview_cache_dir = state.resolve_cache_dir();
    let single_sidecars_dir = state.resolve_managed_single_sidecars_dir();
    let active_shoot = state.active_shoot.clone();
    let load_generation = state.load_generation.clone();
    let my_gen = load_generation.fetch_add(1, Ordering::SeqCst) + 1;

    tauri::async_runtime::spawn_blocking(move || {
        let (ctx, pending_bg_jobs) = run_instant_files_load(
            file_paths,
            &preview_cache_dir,
            &single_sidecars_dir,
            &app,
        )?;
        let dto = build_session_dto(&ctx);
        {
            let mut guard = active_shoot
                .lock()
                .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
            *guard = Some(ctx);
        }

        if !pending_bg_jobs.is_empty() {
            spawn_background_preview_worker(
                pending_bg_jobs,
                active_shoot,
                load_generation,
                my_gen,
                app,
            );
        }

        Ok(dto)
    })
    .await
    .map_err(|err| format!("photo file loader worker failed: {err}"))?
}

/// Step 2 (On-Demand): Runs parallel YuNet 2026 + batched ArcFace 512-D (DirectML GPU / CPU)
/// clustering with disk & RAM embedding caching so re-clustering at new Match thresholds takes <0.1s.
#[tauri::command]
pub async fn run_face_clustering(
    sface_threshold: f32,
    use_temporal_assist: Option<bool>,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<ShootSessionDto, String> {
    let (
        folder_path,
        scan_targets,
        source_to_preview,
        cached_scans_if_match,
    ) = {
        let guard = state
            .active_shoot
            .lock()
            .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
        let ctx = guard
            .as_ref()
            .ok_or_else(|| "no active shoot folder loaded".to_string())?;
        (
            ctx.folder_path.clone(),
            ctx.scan_targets.clone(),
            ctx.source_to_preview.clone(),
            ctx.cached_face_scans.clone(),
        )
    };

    let (yunet_path, recognizer_path) = state.resolve_model_paths();
    let preview_cache_dir = state.resolve_cache_dir();
    let active_shoot = state.active_shoot.clone();
    let temporal_assist = use_temporal_assist.unwrap_or(true);

    tauri::async_runtime::spawn_blocking(move || {
        let started = Instant::now();
        let yunet_path = yunet_path.ok_or_else(|| {
            "YuNet ONNX model not found in ./models/ (expected face_detection_yunet_*.onnx)"
                .to_string()
        })?;
        let recognizer_path = recognizer_path.ok_or_else(|| {
            "Face recognition ONNX model not found in ./models/ (expected w600k_r50.onnx or face_recognition_sface_*.onnx)"
                .to_string()
        })?;

        let model_key = format!(
            "{}|{}",
            yunet_path.to_string_lossy(),
            recognizer_path.to_string_lossy()
        );
        let total_photos = scan_targets.len();

        let (extracted_scans, scan_failures, provider_note) = match cached_scans_if_match {
            Some((cached_key, cached)) if cached_key == model_key && cached.len() == total_photos => {
                emit_progress(
                    &app,
                    "clustering",
                    total_photos,
                    total_photos,
                    format!("Re-clustering {total_photos} photo(s) from RAM cache..."),
                );
                (cached, Vec::new(), "RAM Cache".to_string())
            }
            _ => {
                let outcome = face_cache::load_or_extract_face_scans(
                    &folder_path,
                    &scan_targets,
                    &source_to_preview,
                    &yunet_path,
                    &recognizer_path,
                    &preview_cache_dir,
                    0.35,
                    |done, total, msg| emit_progress(&app, "clustering", done, total, msg),
                )?;
                let note = if outcome.cache_hits > 0 {
                    format!("{}, {} cached", outcome.provider_label, outcome.cache_hits)
                } else {
                    outcome.provider_label.to_string()
                };
                (outcome.scans, outcome.failures, note)
            }
        };

        emit_progress(
            &app,
            "clustering",
            total_photos,
            total_photos,
            format!("Grouping {total_photos} face(s) ({provider_note})..."),
        );

        let verified = sface_threshold.clamp(0.25, 0.90);
        let review = (verified - 0.09).clamp(0.0, verified);
        let thresholds = SimilarityThresholds::new(verified, review).map_err(|e| e.to_string())?;

        let clustered = cluster_preextracted_scans(
            extracted_scans.clone(),
            total_photos,
            scan_failures,
            thresholds,
            temporal_assist,
        )
        .map_err(|e| e.to_string())?;

        let elapsed_ms = started.elapsed().as_millis() as u64;

        let mut guard = active_shoot
            .lock()
            .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
        let ctx = guard
            .as_mut()
            .ok_or_else(|| "no active shoot folder loaded".to_string())?;
        let preserved_virtual_folders = ctx.plan.virtual_folders.clone();
        let preserved_photo_folders = ctx.plan.photo_folders.clone();
        if ctx.is_clustered {
            let prev = ctx.plan.clone();
            ctx.history.record(prev, "Re-cluster faces");
        } else {
            ctx.history.clear();
        }
        ctx.is_clustered = true;
        ctx.total_photos = clustered.scanned_photo_count;
        ctx.scan_elapsed_ms = elapsed_ms;
        ctx.scan_error_count = clustered.failures.len();
        let mut next_plan = clustered.plan;
        next_plan.virtual_folders = preserved_virtual_folders;
        next_plan.photo_folders = preserved_photo_folders;
        ctx.plan = next_plan;
        ctx.cached_face_scans = Some((model_key, extracted_scans));
        persist_organization_plan(ctx)?;

        Ok(build_session_dto(ctx))
    })
    .await
    .map_err(|err| format!("face clustering worker failed: {err}"))?
}

static OCR_SCAN_CANCEL: AtomicBool = AtomicBool::new(false);

fn get_or_init_ocr_engine(
    cache: &CachedOcrEngine,
    det_path: &Path,
    rec_path: &Path,
    dict_path: &Path,
) -> Result<Arc<PpOcrV5Engine>, String> {
    let key = format!(
        "{}|{}|{}",
        det_path.display(),
        rec_path.display(),
        dict_path.display()
    );
    if let Ok(guard) = cache.lock()
        && let Some((cached_key, engine)) = guard.as_ref()
        && *cached_key == key
    {
        return Ok(Arc::clone(engine));
    }

    let threads = std::thread::available_parallelism()
        .map(|n| n.get().clamp(2, 8))
        .unwrap_or(4);
    let config = PpOcrV5Config {
        intra_threads: threads,
        ..PpOcrV5Config::default()
    };
    let engine = Arc::new(
        PpOcrV5Engine::from_model_files(det_path, rec_path, dict_path, config)
            .map_err(|e| e.to_string())?,
    );
    if let Ok(mut guard) = cache.lock() {
        *guard = Some((key, Arc::clone(&engine)));
    }
    Ok(engine)
}

fn map_ocr_lines_dto(lines: Vec<kroma_domain::slate::OcrTextLine>) -> Vec<OcrTextLineDto> {
    lines
        .into_iter()
        .map(|l| OcrTextLineDto {
            text: l.text,
            confidence: l.confidence,
            left: l.region.left,
            top: l.region.top,
            width: l.region.width,
            height: l.region.height,
        })
        .collect()
}

fn file_name_from_path(path: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path)
        .to_string()
}

#[tauri::command]
pub fn cancel_slate_ocr_scan() {
    OCR_SCAN_CANCEL.store(true, Ordering::SeqCst);
}

/// Dedicated Slate OCR scanner powered by local ONNX PP-OCRv5 (`PP-OCRv5_server_det` +
/// `en_PP-OCRv5_mobile_rec`), intentionally decoupled from `run_face_clustering`.
///
/// Scans each subject cluster's frames (prioritizing the earliest capture frame first) and returns
fn ocr_candidate_is_better(
    next_line_count: usize,
    next_conf: Option<f32>,
    prev_line_count: usize,
    prev_conf: Option<f32>,
) -> bool {
    match (next_conf, prev_conf) {
        (Some(_), None) => true,
        (Some(next), Some(prev)) => next > prev,
        (None, None) => next_line_count > 0 && prev_line_count == 0,
        (None, Some(_)) => false,
    }
}

fn should_retry_ocr_with_source(
    scan_path: &str,
    source_path: &str,
    detected_line_count: usize,
    parsed_confidence: Option<f32>,
) -> bool {
    scan_path != source_path
        && detected_line_count > 0
        && parsed_confidence.is_none_or(|c| c < OCR_HIGH_RES_RETRY_CONFIDENCE)
}

fn read_slate_with_raw_fallback(
    engine: &PpOcrV5Engine,
    scan_path: &str,
    source_path: &str,
) -> Result<
    (
        Vec<kroma_domain::slate::OcrTextLine>,
        Option<kroma_domain::slate::SlateCandidateParse>,
    ),
    String,
> {
    let (lines, parsed) = engine
        .read_slate_from_path(scan_path)
        .map_err(|e| e.to_string())?;

    let fast_conf = parsed.as_ref().map(|p| p.confidence);
    if should_retry_ocr_with_source(scan_path, source_path, lines.len(), fast_conf)
        && Path::new(source_path).is_file()
        && let Ok((hi_lines, hi_parsed)) = engine.read_slate_from_path(source_path)
    {
        let hi_conf = hi_parsed.as_ref().map(|p| p.confidence);
        if ocr_candidate_is_better(hi_lines.len(), hi_conf, lines.len(), fast_conf) {
            return Ok((hi_lines, hi_parsed));
        }
    }

    Ok((lines, parsed))
}

/// structured whiteboard proposals for visual review before any subject folder is renamed.
#[tauri::command]
pub async fn scan_project_slates_ocr(
    scan_all_frames: Option<bool>,
    cluster_ids: Option<Vec<String>>,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<Vec<ClusterSlateOcrResultDto>, String> {
    let (det_path, rec_path, dict_path) = state.resolve_ocr_model_paths();
    let (Some(det_path), Some(rec_path), Some(dict_path)) = (det_path, rec_path, dict_path) else {
        return Err(
            "OCR models not found in ./models/ocr/ (expected ppocrv5_mobile_det.onnx, en_ppocrv5_rec.onnx, and en_ppocrv5_dict.txt)."
                .to_string(),
        );
    };

    let target_filter: Option<std::collections::HashSet<String>> = cluster_ids
        .map(|ids| ids.into_iter().filter(|s| !s.trim().is_empty()).collect::<std::collections::HashSet<_>>())
        .filter(|set| !set.is_empty());

    struct OcrFrameCandidate {
        photo_id: String,
        file_name: String,
        source_path: String,
        scan_path: String,
        preview_path: String,
    }

    struct ClusterScanJob {
        cluster_id: String,
        current_name: String,
        first_timestamp: Option<i64>,
        frames: Vec<OcrFrameCandidate>,
    }

    struct NoFaceCandidate {
        photo_id: String,
        file_name: String,
        source_path: String,
        scan_path: String,
        preview_path: String,
        timestamp: Option<i64>,
    }

    let (jobs, no_face_candidates) = {
        let guard = state
            .active_shoot
            .lock()
            .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
        let ctx = guard
            .as_ref()
            .ok_or_else(|| "no active shoot loaded".to_string())?;

        let mut jobs = Vec::new();
        if !ctx.plan.clusters.is_empty() {
            for cluster in &ctx.plan.clusters {
                if let Some(ref allowed) = target_filter
                    && !allowed.contains(cluster.id.as_str())
                {
                    continue;
                }
                let mut entries: Vec<(Option<i64>, OcrFrameCandidate)> = cluster
                    .entries
                    .iter()
                    .map(|e| {
                        let pid = e.photo_id.as_str().to_string();
                        let ts = ctx.photo_capture_timestamps.get(&pid).copied().flatten();
                        let preview = ctx
                            .preview_paths
                            .get(&pid)
                            .cloned()
                            .unwrap_or_else(|| e.source_path.clone());
                        // Start with fast cached preview; automatically escalate to 1600px source RAW only on fuzzy/missed whiteboards
                        let scan_path = if Path::new(&preview).is_file() {
                            preview.clone()
                        } else {
                            e.source_path.clone()
                        };
                        let fname = file_name_from_path(&e.source_path);
                        (
                            ts,
                            OcrFrameCandidate {
                                photo_id: pid,
                                file_name: fname,
                                source_path: e.source_path.clone(),
                                scan_path,
                                preview_path: preview,
                            },
                        )
                    })
                    .collect();

                // Sort earliest capture first (most photographers shoot the whiteboard frame first)
                entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.file_name.cmp(&b.1.file_name)));
                let first_timestamp = entries.first().and_then(|e| e.0);
                // Reorder so frame 0 is checked first, then last frame (for end-of-set slates), then middle
                if entries.len() >= 3 {
                    let last = entries.pop().unwrap();
                    entries.insert(1, last);
                }

                jobs.push(ClusterScanJob {
                    cluster_id: cluster.id.as_str().to_string(),
                    current_name: cluster.display_name.clone(),
                    first_timestamp,
                    frames: entries.into_iter().map(|(_, frame)| frame).collect(),
                });
            }
        } else {
            // Unclustered fallback: allow scanning selected or all photos individually for whiteboards
            for target in &ctx.scan_targets {
                let pid = target.photo_id.as_str().to_string();
                if let Some(ref allowed) = target_filter
                    && !allowed.contains(&pid)
                {
                    continue;
                }
                let preview = ctx
                    .preview_paths
                    .get(&pid)
                    .cloned()
                    .unwrap_or_else(|| target.source_path.clone());
                let scan_path = if Path::new(&preview).is_file() {
                    preview.clone()
                } else {
                    target.source_path.clone()
                };
                let fname = file_name_from_path(&target.source_path);
                jobs.push(ClusterScanJob {
                    cluster_id: pid.clone(),
                    current_name: fname.clone(),
                    first_timestamp: Some(target.capture_timestamp_epoch_sec),
                    frames: vec![OcrFrameCandidate {
                        photo_id: pid,
                        file_name: fname,
                        source_path: target.source_path.clone(),
                        scan_path,
                        preview_path: preview,
                    }],
                });
            }
        }

        // Only pre-scan unassigned `_no_face` frames that sit within 45s before a targeted job
        let job_timestamps: Vec<i64> = jobs.iter().filter_map(|j| j.first_timestamp).collect();
        let no_face_candidates: Vec<NoFaceCandidate> = ctx
            .plan
            .no_face_photos
            .iter()
            .filter_map(|nf| {
                let pid = nf.photo_id.as_str().to_string();
                let ts = ctx.photo_capture_timestamps.get(&pid).copied().flatten()?;
                let near_target = job_timestamps
                    .iter()
                    .any(|&cluster_ts| (0..=45).contains(&(cluster_ts - ts)));
                if !near_target {
                    return None;
                }
                let preview = ctx
                    .preview_paths
                    .get(&pid)
                    .cloned()
                    .unwrap_or_else(|| nf.source_path.clone());
                let scan_path = if Path::new(&preview).is_file() {
                    preview.clone()
                } else {
                    nf.source_path.clone()
                };
                Some(NoFaceCandidate {
                    file_name: file_name_from_path(&nf.source_path),
                    timestamp: Some(ts),
                    photo_id: pid,
                    source_path: nf.source_path.clone(),
                    scan_path,
                    preview_path: preview,
                })
            })
            .collect();

        (jobs, no_face_candidates)
    };

    let scan_every_frame = scan_all_frames.unwrap_or(false);
    let ocr_cache = state.cached_ocr_engine.clone();
    OCR_SCAN_CANCEL.store(false, Ordering::SeqCst);

    tauri::async_runtime::spawn_blocking(move || {
        let total = jobs.len();
        emit_progress(
            &app,
            "ocr",
            0,
            total.max(1),
            "Preparing whiteboard scan…".to_string(),
        );
        let engine = get_or_init_ocr_engine(&ocr_cache, &det_path, &rec_path, &dict_path)?;

        // Pre-scan any standalone no-face whiteboard frames once if present and timestamped
        let mut no_face_slates: Vec<(
            NoFaceCandidate,
            Vec<kroma_domain::slate::OcrTextLine>,
            kroma_domain::slate::SlateCandidateParse,
        )> = Vec::new();
        for nf in no_face_candidates {
            if OCR_SCAN_CANCEL.load(Ordering::SeqCst) {
                break;
            }
            if nf.timestamp.is_none() {
                continue;
            }
            if let Ok((lines, Some(parsed))) =
                read_slate_with_raw_fallback(&engine, &nf.scan_path, &nf.source_path)
                && parsed.confidence >= 0.60
            {
                no_face_slates.push((nf, lines, parsed));
            }
        }

        let mut results = Vec::with_capacity(total);
        for (idx, job) in jobs.into_iter().enumerate() {
            if OCR_SCAN_CANCEL.load(Ordering::SeqCst) {
                break;
            }

            let mut best_photo: Option<(String, String, String)> = None;
            let mut best_lines: Vec<kroma_domain::slate::OcrTextLine> = Vec::new();
            let mut best_parse: Option<kroma_domain::slate::SlateCandidateParse> = None;
            let mut scanned_frames = 0usize;

            for frame in &job.frames {
                if OCR_SCAN_CANCEL.load(Ordering::SeqCst) {
                    break;
                }
                scanned_frames += 1;
                let Ok((lines, parsed)) =
                    read_slate_with_raw_fallback(&engine, &frame.scan_path, &frame.source_path)
                else {
                    continue;
                };

                let is_better = ocr_candidate_is_better(
                    lines.len(),
                    parsed.as_ref().map(|p| p.confidence),
                    best_lines.len(),
                    best_parse.as_ref().map(|p| p.confidence),
                );

                if is_better {
                    best_photo = Some((
                        frame.photo_id.clone(),
                        frame.file_name.clone(),
                        frame.preview_path.clone(),
                    ));
                    best_lines = lines;
                    best_parse = parsed;
                }

                // Fast-path: stop scanning remaining frames in this cluster once a confident slate is read
                if !scan_every_frame
                    && best_parse
                        .as_ref()
                        .is_some_and(|p| p.confidence >= 0.72)
                {
                    break;
                }
            }

            // Distant-board RAW rescue: if no text at all was spotted at 1024px across the cluster,
            // try at most the first frame at 1600px from its embedded RAW preview.
            if best_parse.is_none()
                && best_lines.is_empty()
                && !OCR_SCAN_CANCEL.load(Ordering::SeqCst)
                && let Some(first) = job.frames.first()
                && first.scan_path != first.source_path
                && Path::new(&first.source_path).is_file()
                && let Ok((hi_lines, hi_parsed)) = engine.read_slate_from_path(&first.source_path)
                && (hi_parsed.is_some() || !hi_lines.is_empty())
            {
                best_photo = Some((
                    first.photo_id.clone(),
                    first.file_name.clone(),
                    first.preview_path.clone(),
                ));
                best_lines = hi_lines;
                best_parse = hi_parsed;
            }

            // If no whiteboard was visible inside the subject's face frames, check if a standalone
            // whiteboard-over-face frame was captured right before this subject (within 45 seconds)
            if best_parse.is_none()
                && let Some(cluster_ts) = job.first_timestamp
                && let Some((nf, nf_lines, nf_parsed)) = no_face_slates
                    .iter()
                    .filter_map(|entry| {
                        let ts = entry.0.timestamp?;
                        let delta = cluster_ts - ts;
                        if (0..=45).contains(&delta) {
                            Some((delta, entry))
                        } else {
                            None
                        }
                    })
                    .min_by_key(|(delta, _)| *delta)
                    .map(|(_, entry)| entry)
            {
                best_photo = Some((
                    nf.photo_id.clone(),
                    nf.file_name.clone(),
                    nf.preview_path.clone(),
                ));
                best_lines = nf_lines.clone();
                best_parse = Some(nf_parsed.clone());
            }

            let fallback_first = job.frames.first();
            let (slate_photo_id, slate_file_name, slate_preview_path) = match best_photo {
                Some((pid, fname, preview)) => (Some(pid), Some(fname), Some(preview)),
                None => match fallback_first {
                    Some(first) => (
                        Some(first.photo_id.clone()),
                        Some(first.file_name.clone()),
                        Some(first.preview_path.clone()),
                    ),
                    None => (None, None, None),
                },
            };

            let (proposed_name, proposed_code, confidence) = match best_parse {
                Some(p) => (Some(p.subject_name), p.subject_code, p.confidence),
                None => (None, None, 0.0),
            };

            results.push(ClusterSlateOcrResultDto {
                cluster_id: job.cluster_id,
                current_name: job.current_name.clone(),
                slate_photo_id,
                slate_file_name,
                slate_preview_path,
                proposed_name,
                proposed_code,
                confidence,
                scanned_frame_count: scanned_frames,
                lines: map_ocr_lines_dto(best_lines),
            });

            emit_progress(
                &app,
                "ocr",
                idx + 1,
                total.max(1),
                format!(
                    "Reading whiteboards ({}/{}) — {}",
                    idx + 1,
                    total,
                    job.current_name
                ),
            );
        }

        Ok(results)
    })
    .await
    .map_err(|err| format!("OCR scan worker failed: {err}"))?
}

/// Runs whiteboard OCR on a single photo (reusing the cached warm engine) so photographers can test
/// or re-scan any alternate frame in a subject cluster using the higher-res embedded RAW preview.
#[tauri::command]
pub async fn scan_single_photo_ocr(
    photo_id: String,
    state: State<'_, AppState>,
) -> Result<PhotoOcrResultDto, String> {
    let (det_path, rec_path, dict_path) = state.resolve_ocr_model_paths();
    let (Some(det_path), Some(rec_path), Some(dict_path)) = (det_path, rec_path, dict_path) else {
        return Err(
            "OCR models not found in ./models/ocr/ (expected ppocrv5_det.onnx, en_ppocrv5_rec.onnx, and en_ppocrv5_dict.txt)."
                .to_string(),
        );
    };

    let (source_path, preview_path) = {
        let guard = state
            .active_shoot
            .lock()
            .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
        let ctx = guard
            .as_ref()
            .ok_or_else(|| "no active shoot loaded".to_string())?;
        let src = ctx
            .photo_sources
            .get(&photo_id)
            .cloned()
            .ok_or_else(|| format!("photo '{photo_id}' not found in active shoot"))?;
        let prev = ctx
            .preview_paths
            .get(&photo_id)
            .cloned()
            .unwrap_or_else(|| src.clone());
        (src, prev)
    };

    let ocr_cache = state.cached_ocr_engine.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let engine = get_or_init_ocr_engine(&ocr_cache, &det_path, &rec_path, &dict_path)?;
        // Manual single-frame scan prefers the original source (1600px embedded preview for RAW)
        // for maximum readability on small/faint handwriting, falling back to cached preview if offline.
        let scan_path = if Path::new(&source_path).is_file() {
            source_path.as_str()
        } else {
            preview_path.as_str()
        };
        let (lines, parsed) = engine
            .read_slate_from_path(scan_path)
            .map_err(|e| e.to_string())?;
        let (proposed_name, proposed_code, confidence) = match parsed {
            Some(p) => (Some(p.subject_name), p.subject_code, p.confidence),
            None => (None, None, 0.0),
        };
        Ok(PhotoOcrResultDto {
            photo_id,
            file_name: file_name_from_path(&source_path),
            preview_path,
            proposed_name,
            proposed_code,
            confidence,
            lines: map_ocr_lines_dto(lines),
        })
    })
    .await
    .map_err(|err| format!("single-photo OCR worker failed: {err}"))?
}

/// Applies one or many approved Slate OCR renames in a single undoable OrganizationHistory step,
/// attaches OCR metadata + optional slate-frame exclusion flags, and persists the project plan.
#[tauri::command]
pub fn apply_slate_ocr_renames(
    items: Vec<SlateRenameApplyItemDto>,
    state: State<'_, AppState>,
) -> Result<ShootSessionDto, String> {
    if items.is_empty() {
        return Err("no OCR rename items provided".to_string());
    }

    let mut guard = state
        .active_shoot
        .lock()
        .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
    let ctx = guard
        .as_mut()
        .ok_or_else(|| "no active shoot loaded".to_string())?;

    let prev_plan = ctx.plan.clone();
    let mut applied = 0usize;

    for item in items {
        let trimmed_name = item.next_name.trim();
        if trimmed_name.is_empty() {
            continue;
        }
        let Some(cluster) = ctx
            .plan
            .clusters
            .iter_mut()
            .find(|c| c.id.as_str() == item.cluster_id)
        else {
            continue;
        };

        let slate_pid = item
            .slate_photo_id
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .and_then(|s| PhotoId::new(s).ok());
        let raw_text = item
            .ocr_raw_text
            .unwrap_or_else(|| trimmed_name.to_string());
        let conf = item.ocr_confidence.unwrap_or(1.0);

        cluster
            .apply_slate_ocr_result(
                trimmed_name,
                item.subject_code,
                slate_pid,
                raw_text,
                conf,
                item.mark_as_slate_photo,
            )
            .map_err(|e| e.to_string())?;
        applied += 1;
    }

    if applied > 0 {
        let label = if applied == 1 {
            "Apply Slate OCR name".to_string()
        } else {
            format!("Apply Slate OCR names ({applied} subjects)")
        };
        ctx.history.record(prev_plan, label);
        persist_organization_plan(ctx)?;
    }

    Ok(build_session_dto(ctx))
}

#[tauri::command]
pub fn rename_cluster(
    cluster_id: String,
    next_name: String,
    state: State<'_, AppState>,
) -> Result<ShootSessionDto, String> {
    let mut guard = state
        .active_shoot
        .lock()
        .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
    let ctx = guard
        .as_mut()
        .ok_or_else(|| "no active shoot loaded".to_string())?;

    let prev_plan = ctx.plan.clone();
    let cluster = ctx
        .plan
        .clusters
        .iter_mut()
        .find(|c| c.id.as_str() == cluster_id)
        .ok_or_else(|| format!("cluster '{cluster_id}' not found"))?;
    cluster.rename(next_name).map_err(|e| e.to_string())?;
    ctx.history.record(prev_plan, "Rename subject");
    persist_organization_plan(ctx)?;

    Ok(build_session_dto(ctx))
}

#[tauri::command]
pub fn merge_clusters(
    source_cluster_id: String,
    target_cluster_id: String,
    state: State<'_, AppState>,
) -> Result<ShootSessionDto, String> {
    let mut guard = state
        .active_shoot
        .lock()
        .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
    let ctx = guard
        .as_mut()
        .ok_or_else(|| "no active shoot loaded".to_string())?;

    let src = ClusterId::new(source_cluster_id).map_err(|e| e.to_string())?;
    let dst = ClusterId::new(target_cluster_id).map_err(|e| e.to_string())?;
    let prev_plan = ctx.plan.clone();
    ctx.plan.merge_clusters(&src, &dst).map_err(|e| e.to_string())?;
    ctx.history.record(prev_plan, "Merge subjects");
    persist_organization_plan(ctx)?;

    Ok(build_session_dto(ctx))
}

#[tauri::command]
pub fn split_photos_to_new_cluster(
    photo_ids: Vec<String>,
    state: State<'_, AppState>,
) -> Result<ShootSessionDto, String> {
    let mut guard = state
        .active_shoot
        .lock()
        .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
    let ctx = guard
        .as_mut()
        .ok_or_else(|| "no active shoot loaded".to_string())?;

    let ids = photo_ids
        .into_iter()
        .map(|id| PhotoId::new(id).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let prev_plan = ctx.plan.clone();
    ctx.plan
        .split_photos_to_new_cluster(&ids)
        .map_err(|e| e.to_string())?;
    ctx.history
        .record(prev_plan, format!("Split {} photo(s)", ids.len()));
    persist_organization_plan(ctx)?;

    Ok(build_session_dto(ctx))
}

#[tauri::command]
pub fn move_photos_to_cluster(
    photo_ids: Vec<String>,
    target_cluster_id: String,
    state: State<'_, AppState>,
) -> Result<ShootSessionDto, String> {
    let mut guard = state
        .active_shoot
        .lock()
        .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
    let ctx = guard
        .as_mut()
        .ok_or_else(|| "no active shoot loaded".to_string())?;

    let ids = photo_ids
        .into_iter()
        .map(|id| PhotoId::new(id).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let target = ClusterId::new(target_cluster_id).map_err(|e| e.to_string())?;
    let prev_plan = ctx.plan.clone();
    ctx.plan
        .move_photos_to_cluster(&ids, &target)
        .map_err(|e| e.to_string())?;
    ctx.history
        .record(prev_plan, format!("Match {} photo(s)", ids.len()));
    persist_organization_plan(ctx)?;

    Ok(build_session_dto(ctx))
}

#[tauri::command]
pub fn create_virtual_folder(
    name: String,
    parent_id: Option<String>,
    state: State<'_, AppState>,
) -> Result<ShootSessionDto, String> {
    let mut guard = state
        .active_shoot
        .lock()
        .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
    let ctx = guard
        .as_mut()
        .ok_or_else(|| "no active shoot loaded".to_string())?;

    let parent = match parent_id.filter(|s| !s.trim().is_empty()) {
        Some(pid) => Some(VirtualFolderId::new(pid).map_err(|e| e.to_string())?),
        None => None,
    };
    let prev_plan = ctx.plan.clone();
    ctx.plan
        .create_virtual_folder(name.clone(), parent)
        .map_err(|e| e.to_string())?;
    ctx.history
        .record(prev_plan, format!("Create folder '{}'", name.trim()));
    persist_organization_plan(ctx)?;

    Ok(build_session_dto(ctx))
}

#[tauri::command]
pub fn rename_virtual_folder(
    folder_id: String,
    next_name: String,
    state: State<'_, AppState>,
) -> Result<ShootSessionDto, String> {
    let mut guard = state
        .active_shoot
        .lock()
        .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
    let ctx = guard
        .as_mut()
        .ok_or_else(|| "no active shoot loaded".to_string())?;

    let fid = VirtualFolderId::new(folder_id).map_err(|e| e.to_string())?;
    let prev_plan = ctx.plan.clone();
    ctx.plan
        .rename_virtual_folder(&fid, next_name.clone())
        .map_err(|e| e.to_string())?;
    ctx.history
        .record(prev_plan, format!("Rename folder '{}'", next_name.trim()));
    persist_organization_plan(ctx)?;

    Ok(build_session_dto(ctx))
}

#[tauri::command]
pub fn move_virtual_folder(
    folder_id: String,
    next_parent_id: Option<String>,
    state: State<'_, AppState>,
) -> Result<ShootSessionDto, String> {
    let mut guard = state
        .active_shoot
        .lock()
        .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
    let ctx = guard
        .as_mut()
        .ok_or_else(|| "no active shoot loaded".to_string())?;

    let fid = VirtualFolderId::new(folder_id).map_err(|e| e.to_string())?;
    let next_parent = match next_parent_id.filter(|s| !s.trim().is_empty()) {
        Some(pid) => Some(VirtualFolderId::new(pid).map_err(|e| e.to_string())?),
        None => None,
    };
    let prev_plan = ctx.plan.clone();
    ctx.plan
        .move_virtual_folder(&fid, next_parent)
        .map_err(|e| e.to_string())?;
    ctx.history.record(prev_plan, "Move folder");
    persist_organization_plan(ctx)?;

    Ok(build_session_dto(ctx))
}

#[tauri::command]
pub fn delete_virtual_folder(
    folder_id: String,
    state: State<'_, AppState>,
) -> Result<ShootSessionDto, String> {
    let mut guard = state
        .active_shoot
        .lock()
        .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
    let ctx = guard
        .as_mut()
        .ok_or_else(|| "no active shoot loaded".to_string())?;

    let fid = VirtualFolderId::new(folder_id).map_err(|e| e.to_string())?;
    let prev_plan = ctx.plan.clone();
    ctx.plan
        .delete_virtual_folder(&fid)
        .map_err(|e| e.to_string())?;
    ctx.history.record(prev_plan, "Delete folder");
    persist_organization_plan(ctx)?;

    Ok(build_session_dto(ctx))
}

#[tauri::command]
pub fn assign_photos_to_virtual_folder(
    photo_ids: Vec<String>,
    folder_id: Option<String>,
    state: State<'_, AppState>,
) -> Result<ShootSessionDto, String> {
    let mut guard = state
        .active_shoot
        .lock()
        .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
    let ctx = guard
        .as_mut()
        .ok_or_else(|| "no active shoot loaded".to_string())?;

    let ids = photo_ids
        .into_iter()
        .map(|id| PhotoId::new(id).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let target_fid = match folder_id.filter(|s| !s.trim().is_empty()) {
        Some(fid) => Some(VirtualFolderId::new(fid).map_err(|e| e.to_string())?),
        None => None,
    };

    let target_label = match target_fid.as_ref() {
        Some(fid) => ctx
            .plan
            .virtual_folders
            .iter()
            .find(|f| &f.id == fid)
            .map(|f| f.name.clone())
            .unwrap_or_else(|| "folder".to_string()),
        None => "Unassigned".to_string(),
    };

    let prev_plan = ctx.plan.clone();
    ctx.plan
        .assign_photos_to_folder(&ids, target_fid.as_ref())
        .map_err(|e| e.to_string())?;
    ctx.history.record(
        prev_plan,
        format!("Move {} photo(s) to {target_label}", ids.len()),
    );
    persist_organization_plan(ctx)?;

    Ok(build_session_dto(ctx))
}

#[tauri::command]
pub fn resolve_ambiguous_match(
    photo_id: String,
    decision: String,
    state: State<'_, AppState>,
) -> Result<ShootSessionDto, String> {
    let mut guard = state
        .active_shoot
        .lock()
        .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
    let ctx = guard
        .as_mut()
        .ok_or_else(|| "no active shoot loaded".to_string())?;

    let pid = PhotoId::new(&photo_id).map_err(|e| e.to_string())?;
    let prev_plan = ctx.plan.clone();
    let action_label = match decision.as_str() {
        "approve" => {
            ctx.plan
                .approve_ambiguous_match(&pid)
                .map_err(|e| e.to_string())?;
            "Approve match"
        }
        "split" => {
            ctx.plan
                .split_ambiguous_to_new_cluster(&pid)
                .map_err(|e| e.to_string())?;
            "Split review photo"
        }
        other => {
            return Err(format!(
                "unsupported ambiguous review decision '{other}' (expected 'approve' or 'split')"
            ));
        }
    };
    ctx.history.record(prev_plan, action_label);
    persist_organization_plan(ctx)?;

    Ok(build_session_dto(ctx))
}

#[tauri::command]
pub fn undo_organization_action(state: State<'_, AppState>) -> Result<ShootSessionDto, String> {
    let mut guard = state
        .active_shoot
        .lock()
        .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
    let ctx = guard
        .as_mut()
        .ok_or_else(|| "no active shoot loaded".to_string())?;

    let current = ctx.plan.clone();
    let Some((restored, _label)) = ctx.history.undo(current) else {
        return Err("nothing to undo".to_string());
    };
    ctx.plan = restored;
    persist_organization_plan(ctx)?;

    Ok(build_session_dto(ctx))
}

#[tauri::command]
pub fn redo_organization_action(state: State<'_, AppState>) -> Result<ShootSessionDto, String> {
    let mut guard = state
        .active_shoot
        .lock()
        .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
    let ctx = guard
        .as_mut()
        .ok_or_else(|| "no active shoot loaded".to_string())?;

    let current = ctx.plan.clone();
    let Some((restored, _label)) = ctx.history.redo(current) else {
        return Err("nothing to redo".to_string());
    };
    ctx.plan = restored;
    persist_organization_plan(ctx)?;

    Ok(build_session_dto(ctx))
}

#[tauri::command]
pub async fn auto_level_cluster_eyes(
    cluster_id: String,
    state: State<'_, AppState>,
) -> Result<ShootSessionDto, String> {
    let active_shoot = state.active_shoot.clone();

    tauri::async_runtime::spawn_blocking(move || {
        let (sidecars_dir, targets): (PathBuf, Vec<(PhotoId, f32)>) = {
            let guard = active_shoot
                .lock()
                .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
            let ctx = guard
                .as_ref()
                .ok_or_else(|| "no active shoot loaded".to_string())?;
            if !ctx.is_clustered {
                return Err(
                    "run AI Face Organizer first to detect eye landmarks for Auto-Level".to_string(),
                );
            }
            let cluster = ctx
                .plan
                .clusters
                .iter()
                .find(|c| c.id.as_str() == cluster_id)
                .ok_or_else(|| format!("cluster '{cluster_id}' not found"))?;
            let list = cluster
                .entries
                .iter()
                .map(|entry| {
                    let level_deg = (-entry.landmarks.eye_horizon_angle_deg()).clamp(-15.0, 15.0);
                    (entry.photo_id.clone(), (level_deg * 10.0).round() / 10.0)
                })
                .collect();
            (ctx.sidecars_dir.clone(), list)
        };

        let store = JsonSidecarRecipeStore::new(&sidecars_dir);
        let save_uc = SavePhotoDevelopRecipeUseCase::new(&store);
        let mut updated_recipes = Vec::with_capacity(targets.len());

        for (pid, angle_deg) in targets {
            let mut recipe = store.load_recipe(&pid).ok().flatten().unwrap_or_default();
            recipe.crop_transform.rotation_deg = angle_deg;
            recipe.crop_transform.auto_eye_level_locked = true;
            let saved = save_uc
                .execute(SavePhotoDevelopRecipeInput {
                    photo_id: pid.clone(),
                    recipe,
                })
                .map_err(|e| e.to_string())?;
            updated_recipes.push((pid.as_str().to_string(), saved.clamped_recipe));
        }

        let mut guard = active_shoot
            .lock()
            .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
        let ctx = guard
            .as_mut()
            .ok_or_else(|| "no active shoot loaded".to_string())?;
        for (pid, recipe) in updated_recipes {
            ctx.recipes.insert(pid, recipe);
        }
        Ok(build_session_dto(ctx))
    })
    .await
    .map_err(|err| format!("auto-level worker failed: {err}"))?
}

fn compute_working_buffer_cache_path(
    cache_dir: &Path,
    source_path: &str,
    file_size_bytes: u64,
    max_dim: u32,
) -> PathBuf {
    let mut h1 = DefaultHasher::new();
    "kroma-work-kr16-v1-a".hash(&mut h1);
    source_path.hash(&mut h1);
    file_size_bytes.hash(&mut h1);
    max_dim.hash(&mut h1);

    let mut h2 = DefaultHasher::new();
    "kroma-work-kr16-v1-b".hash(&mut h2);
    max_dim.hash(&mut h2);
    file_size_bytes.hash(&mut h2);
    source_path.hash(&mut h2);

    cache_dir.join(format!(
        "v1_work_{:016x}{:016x}_preview.kr16",
        h1.finish(),
        h2.finish()
    ))
}

fn persist_working_kr16_cache_bounded(cache_dir: &Path, dest: &Path, bytes: &[u8]) {
    if dest.is_file() {
        return;
    }
    if fs::create_dir_all(cache_dir).is_err() {
        return;
    }
    let unique_suffix = format!(
        "kr16.{:?}.{}.tmp",
        std::thread::current().id(),
        std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    )
    .replace(|c: char| !c.is_ascii_alphanumeric() && c != '.', "_");
    let tmp = dest.with_extension(unique_suffix);
    if fs::write(&tmp, bytes).is_ok() {
        if fs::rename(&tmp, dest).is_err() {
            let _ = fs::remove_file(&tmp);
        }
    } else {
        let _ = fs::remove_file(&tmp);
    }

    // Keep disk usage strictly bounded (at most MAX_DISK_WORK_CACHE_FILES recent working buffers)
    let Ok(entries) = fs::read_dir(cache_dir) else {
        return;
    };
    let mut work_files: Vec<(std::time::SystemTime, PathBuf)> = entries
        .flatten()
        .filter_map(|e| {
            let name = e.file_name();
            if !name.to_string_lossy().ends_with("_preview.kr16") {
                return None;
            }
            let modified = e
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .unwrap_or(UNIX_EPOCH);
            Some((modified, e.path()))
        })
        .collect();
    if work_files.len() > MAX_DISK_WORK_CACHE_FILES {
        work_files.sort_by_key(|(mtime, _)| *mtime);
        let to_evict = work_files.len() - MAX_DISK_WORK_CACHE_FILES;
        for (_, path) in work_files.into_iter().take(to_evict) {
            if path != dest {
                let _ = fs::remove_file(path);
            }
        }
    }
}

/// 4-byte sentinel payload (`"KRSK"`) returned when a RAW working-buffer decode is preempted by a
/// newer active photo selection. Kept distinct from empty `Vec::new()` (8-bit fallback) so the
/// frontend never poisons its LRU cache with `null` for skipped photos.
const SUPERSEDED_WORKING_BUFFER_MAGIC: &[u8; 4] = b"KRSK";

enum DevelopDecodeOutcome {
    Superseded,
    NotHighBitDepth,
    Ready {
        wire_bytes: Vec<u8>,
        proxies: Option<(DecodedDevelopBuffer, DecodedDevelopBuffer)>,
    },
}

/// Decodes the active Camera RAW / 16-bit TIFF / 16-bit PNG directly into a screen-sized
/// 16-bit half-float `KR16` (`RGBA16F` Linear Rec.2020 — lossless for 12/14-bit camera sensors,
/// 2× faster IPC & GPU upload, fed straight into 32-bit `rgba32float` WGSL compute shaders)
/// or 32-bit float `KR32` binary buffer for Studio Develop editing.
/// Checks in-memory LRU first, then bounded SSD cache (~10ms), falling back to latest-wins
/// scheduled parallel RAW demosaic so rapid subject switching never starves the selected photo.
#[tauri::command]
pub async fn load_develop_working_buffer(
    photo_id: String,
    max_dim: Option<u32>,
    precision: Option<String>,
    priority: Option<String>,
    state: State<'_, AppState>,
) -> Result<tauri::ipc::Response, String> {
    let decode_priority = DevelopDecodePriority::parse(priority.as_deref());
    let (source_path, file_size_bytes) = {
        let guard = state
            .active_shoot
            .lock()
            .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
        let ctx = guard
            .as_ref()
            .ok_or_else(|| "no active shoot loaded".to_string())?;
        let src = ctx
            .photo_sources
            .get(&photo_id)
            .cloned()
            .ok_or_else(|| format!("photo '{photo_id}' not found in active shoot"))?;
        let size = ctx.photo_file_sizes.get(&photo_id).copied().unwrap_or(0);
        (src, size)
    };

    let ticket = state.develop_decode_scheduler.begin_request(decode_priority);

    let use_f32 = matches!(precision.as_deref(), Some("f32"));
    // `max_dim == Some(0)` requests full native sensor resolution (capped to 8192 for GPU safety).
    // Standard Fit/scrub buffers (`<= 2560`) use the 12-entry LRU + bounded SSD cache;
    // on-demand 1:1 loupe zoom buffers (`> 2560`) use a dedicated 2-entry RAM-only cache so
    // zooming in never bloats RAM or writes 150-250 MB files to disk.
    let clamped_max_dim = match max_dim {
        Some(0) => 8192,
        Some(d) => d.clamp(256, 8192),
        None => 1600,
    };
    let is_hi_res = clamped_max_dim > 2560;
    let cache_key = format!(
        "{source_path}@{clamped_max_dim}@{}",
        if use_f32 { "f32" } else { "f16" }
    );
    // RAM LRU hit bypasses demosaic scheduling completely (~0.1ms instant) while still having
    // bumped `active_gen` above so superseded background decodes yield immediately.
    if is_hi_res {
        if let Some(cached_bytes) = state.get_cached_hi_res_develop_buffer(&cache_key) {
            return Ok(tauri::ipc::Response::new((*cached_bytes).clone()));
        }
    } else if let Some(cached_bytes) = state.get_cached_develop_buffer(&cache_key) {
        return Ok(tauri::ipc::Response::new((*cached_bytes).clone()));
    }

    if state.develop_decode_scheduler.is_superseded(&ticket) {
        return Ok(tauri::ipc::Response::new(
            SUPERSEDED_WORKING_BUFFER_MAGIC.to_vec(),
        ));
    }

    let cache_dir = state.resolve_cache_dir();
    let disk_cache_path = if !use_f32 && !is_hi_res {
        Some(compute_working_buffer_cache_path(
            &cache_dir,
            &source_path,
            file_size_bytes,
            clamped_max_dim,
        ))
    } else {
        None
    };

    let scheduler = Arc::clone(&state.develop_decode_scheduler);
    let source_for_worker = source_path.clone();
    let decoded_result = tauri::async_runtime::spawn_blocking(move || {
        if scheduler.is_superseded(&ticket) {
            return Ok::<DevelopDecodeOutcome, String>(DevelopDecodeOutcome::Superseded);
        }

        // Fast SSD cache hit for compact KR16 working buffers (~8-10ms instead of 250-600ms RAW
        // demosaic). Intentionally bypasses the heavy demosaic permit so previously viewed photos
        // load immediately even while an older uncached RAW demosaic is unwinding.
        if let Some(ref disk_path) = disk_cache_path
            && disk_path.is_file()
            && let Ok(cached_kr16) = fs::read(disk_path)
        {
            if scheduler.is_superseded(&ticket) {
                return Ok(DevelopDecodeOutcome::Superseded);
            }
            if let Some(thumb_buffer) =
                decode_kr16_downscaled_rgb32f(&cached_kr16, THUMBNAIL_32BIT_MAX_DIM)
            {
                if scheduler.is_superseded(&ticket) {
                    return Ok(DevelopDecodeOutcome::Superseded);
                }
                let hist_image = downscale_rgb32f_unclamped(&thumb_buffer.image, 384);
                let hist_buffer = DecodedDevelopBuffer {
                    image: hist_image,
                    is_linear: thumb_buffer.is_linear,
                    is_wide_gamut: thumb_buffer.is_wide_gamut,
                    bit_depth: thumb_buffer.bit_depth,
                    is_raw_demosaic: thumb_buffer.is_raw_demosaic,
                };
                return Ok(DevelopDecodeOutcome::Ready {
                    wire_bytes: cached_kr16,
                    proxies: Some((hist_buffer, thumb_buffer)),
                });
            }
        }

        // Uncached heavy RAW Bayer demosaic: acquire the single latest-wins slot so rapid subject
        // switching never launches competing Rayon decodes that starve the selected photo or FPS.
        let Some(demosaic_permit) = scheduler.acquire_demosaic_slot(ticket) else {
            return Ok(DevelopDecodeOutcome::Superseded);
        };

        let decoded_opt =
            decode_high_bit_depth_working_buffer(&source_for_worker, clamped_max_dim)
                .map_err(|e| e.to_string())?;
        drop(demosaic_permit);

        let Some(decoded) = decoded_opt else {
            return Ok(DevelopDecodeOutcome::NotHighBitDepth);
        };

        if scheduler.is_superseded(&ticket) {
            return Ok(DevelopDecodeOutcome::Superseded);
        }

        let wire_bytes = if use_f32 {
            encode_kr32_rgba32f(&decoded)
        } else {
            encode_kr16_rgba16f(&decoded)
        };
        if let Some(ref disk_path) = disk_cache_path {
            persist_working_kr16_cache_bounded(&cache_dir, disk_path, &wire_bytes);
        }

        if is_hi_res {
            return Ok(DevelopDecodeOutcome::Ready {
                wire_bytes,
                proxies: None,
            });
        }

        let thumb_image = downscale_rgb32f_unclamped(&decoded.image, THUMBNAIL_32BIT_MAX_DIM);
        let hist_image = downscale_rgb32f_unclamped(&thumb_image, 384);
        let hist_buffer = DecodedDevelopBuffer {
            image: hist_image,
            is_linear: decoded.is_linear,
            is_wide_gamut: decoded.is_wide_gamut,
            bit_depth: decoded.bit_depth,
            is_raw_demosaic: decoded.is_raw_demosaic,
        };
        let thumb_buffer = DecodedDevelopBuffer {
            image: thumb_image,
            is_linear: decoded.is_linear,
            is_wide_gamut: decoded.is_wide_gamut,
            bit_depth: decoded.bit_depth,
            is_raw_demosaic: decoded.is_raw_demosaic,
        };
        Ok(DevelopDecodeOutcome::Ready {
            wire_bytes,
            proxies: Some((hist_buffer, thumb_buffer)),
        })
    })
    .await
    .map_err(|err| format!("high-bit-depth develop decode worker failed: {err}"))??;

    let (wire_bytes, proxies) = match decoded_result {
        DevelopDecodeOutcome::Superseded => {
            return Ok(tauri::ipc::Response::new(
                SUPERSEDED_WORKING_BUFFER_MAGIC.to_vec(),
            ));
        }
        DevelopDecodeOutcome::NotHighBitDepth => {
            return Ok(tauri::ipc::Response::new(Vec::new()));
        }
        DevelopDecodeOutcome::Ready {
            wire_bytes,
            proxies,
        } => (wire_bytes, proxies),
    };

    let shared_bytes = Arc::new(wire_bytes);
    if is_hi_res {
        state.insert_hi_res_develop_buffer(cache_key, Arc::clone(&shared_bytes));
    } else if let Some((hist_buffer, thumb_buffer)) = proxies {
        state.insert_develop_buffer(CachedDevelopBuffer {
            cache_key,
            source_path,
            kr16_bytes: Arc::clone(&shared_bytes),
            hist_buffer: Arc::new(hist_buffer),
            thumb_buffer: Arc::new(thumb_buffer),
        });
    }

    // If superseded during the final ~8ms encode/proxy window, keep the completed buffer warm in
    // Rust's RAM LRU while skipping the 15 MB IPC transfer back to WebView2.
    if state.develop_decode_scheduler.is_superseded(&ticket) {
        return Ok(tauri::ipc::Response::new(
            SUPERSEDED_WORKING_BUFFER_MAGIC.to_vec(),
        ));
    }

    Ok(tauri::ipc::Response::new((*shared_bytes).clone()))
}

/// Returns the EXIF-oriented native pixel dimensions `(width, height)` of a photo using fast
/// header-only metadata inspection (<1ms, cached in `AppState`) so Studio's `100%` (1:1) zoom
/// always reflects real sensor pixels even before decoding a full-resolution working buffer.
#[tauri::command]
pub async fn get_photo_native_dimensions(
    photo_id: String,
    state: State<'_, AppState>,
) -> Result<Option<(u32, u32)>, String> {
    let source_path = {
        let guard = state
            .active_shoot
            .lock()
            .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
        let Some(ctx) = guard.as_ref() else {
            return Ok(None);
        };
        match ctx.photo_sources.get(&photo_id) {
            Some(src) => src.clone(),
            None => return Ok(None),
        }
    };

    if let Some(dims) = state.get_cached_native_dims(&source_path) {
        return Ok(Some(dims));
    }

    let source_for_worker = source_path.clone();
    let dims = tauri::async_runtime::spawn_blocking(move || {
        read_native_photo_dimensions(&source_for_worker).ok()
    })
    .await
    .map_err(|err| format!("native dimension probe failed: {err}"))?;

    if let Some(d) = dims {
        state.insert_cached_native_dims(source_path, d);
    }
    Ok(dims)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThumbnailPreviewRenderMode {
    Fast,
    Edited32Bit,
    All32Bit,
}

impl ThumbnailPreviewRenderMode {
    fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "fast" => Self::Fast,
            "all_32bit" | "all" => Self::All32Bit,
            _ => Self::Edited32Bit,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Edited32Bit => "edited_32bit",
            Self::All32Bit => "all_32bit",
        }
    }
}

fn emit_thumbnail_progress(app: &AppHandle, snapshot: ThumbnailProgressSnapshot) {
    let _ = app.emit(
        THUMBNAIL_RENDER_PROGRESS_EVENT,
        ThumbnailRenderProgressDto {
            active: snapshot.active,
            completed: snapshot.completed,
            total: snapshot.total,
            mode: snapshot.mode,
        },
    );
}

fn invalidate_thumbnail_renders(
    state: &AppState,
    app: Option<&AppHandle>,
    mode: &str,
    force_emit_inactive: bool,
) -> u64 {
    let mut tracker = state
        .thumbnail_progress_tracker
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (next_gen, had_active, inactive) =
        tracker.invalidate_and_bump(&state.thumbnail_render_generation, mode);
    if let Some(app_handle) = app
        && (force_emit_inactive || had_active)
    {
        emit_thumbnail_progress(app_handle, inactive);
    }
    next_gen
}

fn emit_inactive_if_thumbnail_idle(state: &AppState, app: &AppHandle, mode: &str) {
    let tracker = state
        .thumbnail_progress_tracker
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if !tracker.has_active_workers() {
        emit_thumbnail_progress(
            app,
            ThumbnailProgressSnapshot {
                active: false,
                completed: 0,
                total: 0,
                mode: mode.to_string(),
            },
        );
    }
}

/// RAII progress guard for each spawned 32-bit thumbnail worker thread.
/// Guarantees linearizable event ordering under `thumbnail_progress_tracker` and ensures
/// `active: false` (or remaining concurrent workers' progress) is always emitted on normal
/// completion, generation cancellation, early return, or panic unwind.
struct ThumbnailWorkerProgressGuard {
    worker_id: u64,
    mode: String,
    tracker: Arc<Mutex<ThumbnailProgressTracker>>,
    app: AppHandle,
}

impl ThumbnailWorkerProgressGuard {
    #[allow(clippy::too_many_arguments)]
    fn try_start(
        tracker: Arc<Mutex<ThumbnailProgressTracker>>,
        thumb_gen: &AtomicU64,
        load_gen: &AtomicU64,
        expected_thumb_gen: u64,
        expected_load_gen: u64,
        total: usize,
        mode: String,
        app: AppHandle,
    ) -> Option<Self> {
        let mut guard = tracker.lock().unwrap_or_else(|e| e.into_inner());
        let (worker_id, snapshot) = guard.try_register_worker(
            thumb_gen,
            load_gen,
            expected_thumb_gen,
            expected_load_gen,
            total,
            mode.clone(),
        )?;
        emit_thumbnail_progress(&app, snapshot);
        drop(guard);
        Some(Self {
            worker_id,
            mode,
            tracker,
            app,
        })
    }

    fn advance(&self, completed: usize) {
        let mut guard = self.tracker.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(snapshot) = guard.update_worker(self.worker_id, completed) {
            emit_thumbnail_progress(&self.app, snapshot);
        }
    }
}

impl Drop for ThumbnailWorkerProgressGuard {
    fn drop(&mut self) {
        let mut guard = self.tracker.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(snapshot) = guard.finish_worker(self.worker_id, &self.mode) {
            emit_thumbnail_progress(&self.app, snapshot);
        }
    }
}

fn hash_portrait_retouch_context<H: Hasher>(
    state: &mut H,
    face_guides: &[PortraitFaceGuide],
    face_parse_map: Option<&FaceParseMap>,
) {
    face_guides.len().min(4).hash(state);
    for g in face_guides.iter().take(4) {
        for v in [
            g.box_left,
            g.box_top,
            g.box_width,
            g.box_height,
            g.right_eye[0],
            g.right_eye[1],
            g.left_eye[0],
            g.left_eye[1],
        ] {
            (v * 1000.0).round().to_bits().hash(state);
        }
    }
    if let Some(fp) = face_parse_map.filter(|m| m.is_valid()) {
        true.hash(state);
        fp.width.hash(state);
        fp.height.hash(state);
        let step = (fp.rgba.len() / 64).max(1);
        for &b in fp.rgba.iter().step_by(step) {
            b.hash(state);
        }
    } else {
        false.hash(state);
    }
}

#[cfg(test)]
fn compute_32bit_thumbnail_hash(
    source_path: &str,
    file_size_bytes: u64,
    recipe: &DevelopRecipe,
) -> (u64, u64) {
    compute_32bit_thumbnail_hash_with_retouch(source_path, file_size_bytes, recipe, &[], None)
}

fn compute_32bit_thumbnail_hash_with_retouch(
    source_path: &str,
    file_size_bytes: u64,
    recipe: &DevelopRecipe,
    face_guides: &[PortraitFaceGuide],
    face_parse_map: Option<&FaceParseMap>,
) -> (u64, u64) {
    let recipe_json = serde_json::to_string(&DevelopRecipeDto::from(recipe)).unwrap_or_default();
    let portrait_active = recipe.retouch.is_portrait_active();
    let mut h1 = DefaultHasher::new();
    "kroma-32bit-thumb-v1-a".hash(&mut h1);
    THUMBNAIL_32BIT_MAX_DIM.hash(&mut h1);
    source_path.hash(&mut h1);
    file_size_bytes.hash(&mut h1);
    recipe_json.hash(&mut h1);
    if portrait_active {
        hash_portrait_retouch_context(&mut h1, face_guides, face_parse_map);
    }

    let mut h2 = DefaultHasher::new();
    "kroma-32bit-thumb-v1-b".hash(&mut h2);
    recipe_json.hash(&mut h2);
    file_size_bytes.hash(&mut h2);
    source_path.hash(&mut h2);
    if portrait_active {
        hash_portrait_retouch_context(&mut h2, face_guides, face_parse_map);
    }

    (h1.finish(), h2.finish())
}

fn should_render_32bit_thumbnail(
    mode: ThumbnailPreviewRenderMode,
    source_path: &str,
    recipe: &DevelopRecipe,
) -> bool {
    match mode {
        ThumbnailPreviewRenderMode::Fast => false,
        ThumbnailPreviewRenderMode::Edited32Bit => {
            if !recipe.is_modified() {
                return false;
            }
            let p = Path::new(source_path);
            let is_high_bit_depth =
                is_raw_photo_path(p) || is_tiff_photo_path(p) || is_16bit_png_path(p);
            is_high_bit_depth || recipe.has_non_svg_thumbnail_effects()
        }
        ThumbnailPreviewRenderMode::All32Bit => {
            let p = Path::new(source_path);
            let is_high_bit_depth =
                is_raw_photo_path(p) || is_tiff_photo_path(p) || is_16bit_png_path(p);
            is_high_bit_depth || recipe.has_non_svg_thumbnail_effects()
        }
    }
}

/// Renders compact 640px 32-bit linear-developed JPEG thumbnails in the background for Face
/// Organizer, Filmstrip, Lightbox, and Export previews so edited cards & RAW grids match
/// Studio's 32-bit WebGPU / `wgpu` color, backdrops, masks, HSL, and LUTs 1:1 without consuming
/// extra WebGPU contexts or VRAM.
#[tauri::command]
pub fn render_32bit_thumbnails(
    mode: String,
    photo_ids: Option<Vec<String>>,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let parsed_mode = ThumbnailPreviewRenderMode::parse(&mode);
    let mode_str = parsed_mode.as_str().to_string();

    if parsed_mode == ThumbnailPreviewRenderMode::Fast {
        invalidate_thumbnail_renders(&state, Some(&app), &mode_str, true);
        let cleared_ids: Vec<String> = {
            let mut guard = state
                .active_shoot
                .lock()
                .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
            if let Some(ctx) = guard.as_mut() {
                ctx.rendered_preview_paths
                    .drain()
                    .map(|(pid, _)| pid)
                    .collect()
            } else {
                Vec::new()
            }
        };
        if !cleared_ids.is_empty() {
            let total = cleared_ids.len();
            let updates = cleared_ids
                .into_iter()
                .map(|photo_id| PreviewPatchDto {
                    photo_id,
                    preview_path: None,
                    rendered_preview_path: None,
                    clear_rendered_preview: true,
                    captured_at_epoch_sec: None,
                    captured_at_label: None,
                })
                .collect();
            let _ = app.emit(
                PREVIEWS_READY_EVENT,
                PreviewsReadyBatchDto {
                    updates,
                    completed: total,
                    total,
                },
            );
        }
        return Ok(());
    }

    let is_full_pass = photo_ids.as_ref().is_none_or(|ids| ids.is_empty());
    let expected_thumb_gen = if is_full_pass {
        invalidate_thumbnail_renders(&state, None, &mode_str, false)
    } else {
        state.thumbnail_render_generation.load(Ordering::SeqCst)
    };
    let expected_load_gen = state.load_generation.load(Ordering::SeqCst);
    let preview_cache_dir = state.resolve_cache_dir();

    struct ThumbRenderJob {
        photo_id: String,
        source_path: String,
        recipe: DevelopRecipe,
        face_guides: Vec<PortraitFaceGuide>,
        face_parse_map: Option<FaceParseMap>,
        hash: (u64, u64),
        dest_path: PathBuf,
    }

    let (cached_hits, needs_render, cleared_ids) = {
        let mut guard = state
            .active_shoot
            .lock()
            .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
        let Some(ctx) = guard.as_mut() else {
            if is_full_pass {
                emit_inactive_if_thumbnail_idle(&state, &app, &mode_str);
            }
            return Ok(());
        };
        let default_recipe = DevelopRecipe::default();
        let target_ids: Vec<String> = match photo_ids {
            Some(ids) if !ids.is_empty() => ids,
            _ => ctx
                .scan_targets
                .iter()
                .map(|t| t.photo_id.as_str().to_string())
                .collect(),
        };

        let mut cached_hits: Vec<(String, String)> = Vec::new();
        let mut needs_render: Vec<ThumbRenderJob> = Vec::new();
        let mut cleared_ids: Vec<String> = Vec::new();
        let mut seen = HashSet::with_capacity(target_ids.len());

        for pid in target_ids {
            if !seen.insert(pid.clone()) {
                continue;
            }
            let Some(source_path) = ctx.photo_sources.get(&pid).cloned() else {
                continue;
            };
            let recipe = ctx.recipes.get(&pid).cloned().unwrap_or_else(|| default_recipe.clone());
            if !should_render_32bit_thumbnail(parsed_mode, &source_path, &recipe) {
                if ctx.rendered_preview_paths.remove(&pid).is_some() {
                    cleared_ids.push(pid);
                }
                continue;
            }
            let portrait_active = recipe.retouch.is_portrait_active();
            let face_guides = if portrait_active {
                get_photo_face_guides(ctx, &pid)
            } else {
                Vec::new()
            };
            let face_parse_map = if portrait_active {
                get_photo_face_parse_map(ctx, &pid, recipe.crop_transform)
            } else {
                None
            };
            let file_size_bytes = ctx.photo_file_sizes.get(&pid).copied().unwrap_or(0);
            let hash = compute_32bit_thumbnail_hash_with_retouch(
                &source_path,
                file_size_bytes,
                &recipe,
                &face_guides,
                face_parse_map.as_ref(),
            );
            let dest_path = preview_cache_dir
                .join(format!("v1_32bit_{:016x}{:016x}_preview.jpg", hash.0, hash.1));
            let dest_str = dest_path.to_string_lossy().to_string();

            if dest_path.is_file() {
                if ctx.rendered_preview_paths.get(&pid).map(String::as_str) != Some(dest_str.as_str())
                {
                    ctx.rendered_preview_paths.insert(pid.clone(), dest_str.clone());
                    cached_hits.push((pid, dest_str));
                }
            } else {
                needs_render.push(ThumbRenderJob {
                    photo_id: pid,
                    source_path,
                    recipe,
                    face_guides,
                    face_parse_map,
                    hash,
                    dest_path,
                });
            }
        }

        (cached_hits, needs_render, cleared_ids)
    };

    if !cleared_ids.is_empty() || !cached_hits.is_empty() {
        let mut updates = Vec::with_capacity(cleared_ids.len() + cached_hits.len());
        for photo_id in cleared_ids {
            updates.push(PreviewPatchDto {
                photo_id,
                preview_path: None,
                rendered_preview_path: None,
                clear_rendered_preview: true,
                captured_at_epoch_sec: None,
                captured_at_label: None,
            });
        }
        for (photo_id, rendered_path) in cached_hits {
            updates.push(PreviewPatchDto {
                photo_id,
                preview_path: None,
                rendered_preview_path: Some(rendered_path),
                clear_rendered_preview: false,
                captured_at_epoch_sec: None,
                captured_at_label: None,
            });
        }
        let count = updates.len();
        let _ = app.emit(
            PREVIEWS_READY_EVENT,
            PreviewsReadyBatchDto {
                updates,
                completed: count,
                total: count,
            },
        );
    }

    if needs_render.is_empty() {
        if is_full_pass {
            emit_inactive_if_thumbnail_idle(&state, &app, &mode_str);
        }
        return Ok(());
    }

    let active_shoot = state.active_shoot.clone();
    let load_generation = state.load_generation.clone();
    let thumb_generation = state.thumbnail_render_generation.clone();
    let progress_tracker = state.thumbnail_progress_tracker.clone();
    let develop_buffer_cache = state.develop_buffer_cache.clone();

    std::thread::spawn(move || {
        let _ = fs::create_dir_all(&preview_cache_dir);
        let engine = RayonImageDevelopEngine::new();
        let total = needs_render.len();
        let mut completed = 0usize;

        let is_still_valid = || {
            load_generation.load(Ordering::SeqCst) == expected_load_gen
                && thumb_generation.load(Ordering::SeqCst) == expected_thumb_gen
        };

        let Some(progress_guard) = ThumbnailWorkerProgressGuard::try_start(
            progress_tracker,
            &thumb_generation,
            &load_generation,
            expected_thumb_gen,
            expected_load_gen,
            total,
            mode_str,
            app.clone(),
        ) else {
            return;
        };

        for chunk in needs_render.chunks(THUMBNAIL_32BIT_BATCH_SIZE) {
            if !is_still_valid() {
                return;
            }

            let rendered: Vec<(String, String, (u64, u64))> = chunk
                .par_iter()
                .filter_map(|job| {
                    if !is_still_valid() {
                        return None;
                    }
                    if !job.dest_path.is_file() {
                        let cached_buf = develop_buffer_cache
                            .lock()
                            .ok()
                            .and_then(|guard| {
                                guard
                                    .iter()
                                    .rev()
                                    .find(|entry| entry.source_path == job.source_path)
                                    .map(|entry| Arc::clone(&entry.thumb_buffer))
                            });
                        let render_result = if let Some(hot_buf) = cached_buf {
                            engine.render_buffer_to_jpeg_with_retouch(
                                &hot_buf,
                                &job.recipe,
                                &job.face_guides,
                                job.face_parse_map.as_ref(),
                                &job.dest_path,
                                THUMBNAIL_32BIT_JPEG_QUALITY,
                            )
                        } else {
                            decode_develop_buffer(&job.source_path, THUMBNAIL_32BIT_MAX_DIM)
                                .and_then(|decoded| {
                                    engine.render_buffer_to_jpeg_with_retouch(
                                        &decoded,
                                        &job.recipe,
                                        &job.face_guides,
                                        job.face_parse_map.as_ref(),
                                        &job.dest_path,
                                        THUMBNAIL_32BIT_JPEG_QUALITY,
                                    )
                                })
                        };
                        if render_result.is_err() || !job.dest_path.is_file() {
                            return None;
                        }
                    }
                    Some((
                        job.photo_id.clone(),
                        job.dest_path.to_string_lossy().to_string(),
                        job.hash,
                    ))
                })
                .collect();

            if !is_still_valid() {
                return;
            }

            completed += chunk.len();
            let mut patches = Vec::with_capacity(rendered.len());

            if let Ok(mut guard) = active_shoot.lock()
                && let Some(ctx) = guard.as_mut()
            {
                let default_recipe = DevelopRecipe::default();
                for (pid, rendered_path, expected_hash) in rendered {
                    let Some(source_path) = ctx.photo_sources.get(&pid) else {
                        continue;
                    };
                    let current_size = ctx.photo_file_sizes.get(&pid).copied().unwrap_or(0);
                    let current_recipe = ctx.recipes.get(&pid).unwrap_or(&default_recipe);
                    let portrait_active = current_recipe.retouch.is_portrait_active();
                    let current_guides = if portrait_active {
                        get_photo_face_guides(ctx, &pid)
                    } else {
                        Vec::new()
                    };
                    let current_parse = if portrait_active {
                        get_photo_face_parse_map(ctx, &pid, current_recipe.crop_transform)
                    } else {
                        None
                    };
                    let current_hash = compute_32bit_thumbnail_hash_with_retouch(
                        source_path,
                        current_size,
                        current_recipe,
                        &current_guides,
                        current_parse.as_ref(),
                    );
                    if current_hash != expected_hash {
                        continue;
                    }
                    ctx.rendered_preview_paths
                        .insert(pid.clone(), rendered_path.clone());
                    patches.push(PreviewPatchDto {
                        photo_id: pid,
                        preview_path: None,
                        rendered_preview_path: Some(rendered_path),
                        clear_rendered_preview: false,
                        captured_at_epoch_sec: None,
                        captured_at_label: None,
                    });
                }
            }

            if !patches.is_empty() {
                let _ = app.emit(
                    PREVIEWS_READY_EVENT,
                    PreviewsReadyBatchDto {
                        updates: patches,
                        completed,
                        total,
                    },
                );
            }

            progress_guard.advance(completed);
        }

        // `progress_guard` drops here (or on any early return / panic unwind), removing this
        // worker from `thumbnail_progress_tracker` and emitting `active: false` when idle.
        drop(progress_guard);
    });

    Ok(())
}

/// Generates a non-destructive AI alpha matte (`subject`, `background`, `face_skin`, `sky`)
/// aligned to the photo's current orientation (quarter-turns + flips).
/// Uses an installed ONNX segmentation model (`RMBG`, `MODNet`, `BiRefNet`, `IS-Net`, `U2-Net`)
/// on DirectML GPU / CPU when configured, and falls back to Kroma's built-in YuNet + edge-matte AI.
fn entry_to_face_guide(bbox: BoundingBox, lm: YuNetLandmarks) -> PortraitFaceGuide {
    PortraitFaceGuide::from_normalized(
        bbox.x,
        bbox.y,
        bbox.width,
        bbox.height,
        [lm.right_eye.x, lm.right_eye.y],
        [lm.left_eye.x, lm.left_eye.y],
        [lm.nose_tip.x, lm.nose_tip.y],
        [lm.right_mouth_corner.x, lm.right_mouth_corner.y],
        [lm.left_mouth_corner.x, lm.left_mouth_corner.y],
    )
}

fn get_photo_face_guides(ctx: &ActiveShootContext, photo_id: &str) -> Vec<PortraitFaceGuide> {
    if let Some(guides) = ctx.detected_face_guides.get(photo_id)
        && !guides.is_empty()
    {
        return guides.clone();
    }
    if ctx.is_clustered {
        for cluster in &ctx.plan.clusters {
            if let Some(entry) = cluster.entries.iter().find(|e| e.photo_id.as_str() == photo_id)
                && entry.face_bounding_box.area() > 0.0
            {
                return vec![entry_to_face_guide(entry.face_bounding_box, entry.landmarks)];
            }
        }
    }
    Vec::new()
}

fn get_photo_face_parse_map(
    ctx: &ActiveShootContext,
    photo_id: &str,
    crop: CropTransform,
) -> Option<FaceParseMap> {
    let current_ori = format!(
        "{}:{}:{}",
        crop.quarter_turns % 4,
        crop.flip_horizontal as u8,
        crop.flip_vertical as u8
    );
    ctx.detected_face_parse_maps
        .get(photo_id)
        .filter(|(ori, map)| ori == &current_ori && map.is_valid())
        .map(|(_, map)| map.clone())
}

fn collect_shoot_face_guides_by_source(
    ctx: &ActiveShootContext,
) -> HashMap<String, Vec<PortraitFaceGuide>> {
    let mut map: HashMap<String, Vec<PortraitFaceGuide>> = HashMap::new();
    if ctx.is_clustered {
        for cluster in &ctx.plan.clusters {
            for entry in &cluster.entries {
                if entry.face_bounding_box.area() > 0.0 {
                    map.insert(
                        entry.source_path.clone(),
                        vec![entry_to_face_guide(entry.face_bounding_box, entry.landmarks)],
                    );
                }
            }
        }
    }
    for (pid, guides) in &ctx.detected_face_guides {
        if !guides.is_empty()
            && let Some(source_path) = ctx.photo_sources.get(pid)
        {
            map.insert(source_path.clone(), guides.clone());
        }
    }
    map
}

fn collect_shoot_face_parse_maps_by_source(
    ctx: &ActiveShootContext,
) -> HashMap<String, FaceParseMap> {
    let mut map: HashMap<String, FaceParseMap> = HashMap::new();
    for (pid, (cached_ori, parse_map)) in &ctx.detected_face_parse_maps {
        if !parse_map.is_valid() {
            continue;
        }
        let crop = ctx
            .recipes
            .get(pid)
            .map(|r| r.crop_transform)
            .unwrap_or_default();
        let current_ori = format!(
            "{}:{}:{}",
            crop.quarter_turns % 4,
            crop.flip_horizontal as u8,
            crop.flip_vertical as u8
        );
        if cached_ori == &current_ori
            && let Some(source_path) = ctx.photo_sources.get(pid)
        {
            map.insert(source_path.clone(), parse_map.clone());
        }
    }
    map
}

/// Fast on-demand YuNet 5-point face landmark detection for a single photo (used by Portrait
/// Retouching in Single-Photo Mode or before full-shoot Face Clustering has been run).
#[tauri::command]
pub async fn detect_photo_face_overlay(
    photo_id: String,
    state: State<'_, AppState>,
) -> Result<Option<FaceOverlayDto>, String> {
    let (existing_overlay, preview_path) = {
        let guard = state
            .active_shoot
            .lock()
            .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
        let ctx = guard
            .as_ref()
            .ok_or_else(|| "no active shoot loaded".to_string())?;

        if let Some(ov) = ctx.detected_face_overlays.get(&photo_id) {
            return Ok(Some(ov.clone()));
        }
        if ctx.is_clustered {
            for cluster in &ctx.plan.clusters {
                if let Some(entry) = cluster
                    .entries
                    .iter()
                    .find(|e| e.photo_id.as_str() == photo_id.as_str())
                    && entry.face_bounding_box.area() > 0.0
                {
                    return Ok(Some(build_face_overlay(
                        entry.face_bounding_box,
                        entry.landmarks,
                    )));
                }
            }
        }
        let source = ctx
            .photo_sources
            .get(&photo_id)
            .cloned()
            .ok_or_else(|| format!("photo '{photo_id}' not found in active shoot"))?;
        let preview = ctx.preview_paths.get(&photo_id).cloned().unwrap_or(source);
        (None::<FaceOverlayDto>, preview)
    };
    if let Some(ov) = existing_overlay {
        return Ok(Some(ov));
    }

    let (yunet_path, _) = state.resolve_model_paths();
    let Some(yunet_model_path) = yunet_path else {
        return Ok(None);
    };
    let cached_yunet = state.cached_yunet_engine.clone();
    let active_shoot = state.active_shoot.clone();
    let pid_clone = photo_id.clone();

    tauri::async_runtime::spawn_blocking(move || {
        let key = yunet_model_path.to_string_lossy().to_string();
        let detector_arc = {
            let existing = cached_yunet
                .lock()
                .ok()
                .and_then(|g| g.as_ref().filter(|(k, _)| *k == key).map(|(_, e)| Arc::clone(e)));
            if let Some(det) = existing {
                det
            } else {
                let det = YuNetOnnxDetector::from_model_file(&yunet_model_path, YuNetConfig::default())
                    .map_err(|e| format!("failed to load YuNet detector: {e}"))?;
                let arc = Arc::new(det);
                if let Ok(mut g) = cached_yunet.lock() {
                    *g = Some((key, Arc::clone(&arc)));
                }
                arc
            }
        };

        let rgb = decode_fast_preview_max_dim(&preview_path, 1024)
            .map_err(|e| format!("failed to decode preview for face detection: {e}"))?;
        let detections = detector_arc
            .detect_faces_in_rgb(&rgb)
            .map_err(|e| format!("YuNet face detection failed: {e}"))?;

        if detections.is_empty() {
            return Ok(None);
        }

        // Sort largest primary face first
        let mut sorted = detections;
        sorted.sort_by(|a, b| {
            b.bounding_box
                .area()
                .partial_cmp(&a.bounding_box.area())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let primary = &sorted[0];
        let overlay = build_face_overlay(primary.bounding_box, primary.landmarks);
        let guides: Vec<PortraitFaceGuide> = sorted
            .iter()
            .take(4)
            .map(|d| entry_to_face_guide(d.bounding_box, d.landmarks))
            .collect();

        if let Ok(mut guard) = active_shoot.lock()
            && let Some(ctx) = guard.as_mut()
        {
            ctx.detected_face_guides.insert(pid_clone.clone(), guides);
            ctx.detected_face_overlays
                .insert(pid_clone, overlay.clone());
        }

        Ok(Some(overlay))
    })
    .await
    .map_err(|err| format!("face detection worker failed: {err}"))?
}

/// On-demand BiSeNet 19-class neural face parsing for Portrait Retouching (Phase 3 opt-in).
/// Returns `Ok(None)` cleanly when `parsing_resnet18.onnx` is not installed so Studio stays on
/// built-in YuNet 5-point landmark masks without errors.
#[tauri::command]
pub async fn parse_photo_face_retouch_map(
    photo_id: String,
    state: State<'_, AppState>,
) -> Result<Option<FaceParseMapDto>, String> {
    let Some(parser_path) = state.resolve_face_parser_model_path() else {
        return Ok(None);
    };

    let (cached_hit, preview_path, crop_transform, existing_guides) = {
        let guard = state
            .active_shoot
            .lock()
            .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
        let ctx = guard
            .as_ref()
            .ok_or_else(|| "no active shoot loaded".to_string())?;
        let source = ctx
            .photo_sources
            .get(&photo_id)
            .cloned()
            .ok_or_else(|| format!("photo '{photo_id}' not found in active shoot"))?;
        let preview = ctx.preview_paths.get(&photo_id).cloned().unwrap_or(source);
        let crop = ctx
            .recipes
            .get(&photo_id)
            .map(|r| r.crop_transform)
            .unwrap_or_default();
        let ori_key = format!(
            "{}:{}:{}",
            crop.quarter_turns % 4,
            crop.flip_horizontal as u8,
            crop.flip_vertical as u8
        );
        let hit = ctx
            .detected_face_parse_maps
            .get(&photo_id)
            .filter(|(k, map)| k == &ori_key && map.is_valid())
            .map(|(_, map)| map.clone());
        let guides = get_photo_face_guides(ctx, &photo_id);
        (hit, preview, crop, guides)
    };

    if let Some(map) = cached_hit {
        let label = AppState::face_parser_model_label(Some(&parser_path));
        return Ok(Some(FaceParseMapDto::from_map(&map, label, 0)));
    }

    let (yunet_path, _) = state.resolve_model_paths();
    let cached_parser = state.cached_face_parser_engine.clone();
    let cached_yunet = state.cached_yunet_engine.clone();
    let active_shoot = state.active_shoot.clone();
    let pid_clone = photo_id.clone();

    tauri::async_runtime::spawn_blocking(move || {
        let started = Instant::now();
        let mut rgb = decode_fast_preview_max_dim(&preview_path, 1024)
            .map_err(|e| format!("failed to decode preview for face parsing: {e}"))?;

        let qt = crop_transform.quarter_turns % 4;
        rgb = match qt {
            1 => image::imageops::rotate90(&rgb),
            2 => image::imageops::rotate180(&rgb),
            3 => image::imageops::rotate270(&rgb),
            _ => rgb,
        };
        if crop_transform.flip_horizontal {
            rgb = image::imageops::flip_horizontal(&rgb);
        }
        if crop_transform.flip_vertical {
            rgb = image::imageops::flip_vertical(&rgb);
        }

        let (ow, oh) = rgb.dimensions();
        let mut oriented_guides: Vec<PortraitFaceGuide> = existing_guides
            .iter()
            .take(4)
            .map(|g| g.into_oriented_uv(crop_transform, ow, oh, false))
            .collect();

        if oriented_guides.is_empty()
            && let Some(ref yp) = yunet_path
        {
            let ykey = yp.to_string_lossy().to_string();
            let det_arc = {
                let existing = cached_yunet
                    .lock()
                    .ok()
                    .and_then(|g| g.as_ref().filter(|(k, _)| *k == ykey).map(|(_, e)| Arc::clone(e)));
                if let Some(det) = existing {
                    Some(det)
                } else {
                    YuNetOnnxDetector::from_model_file(yp, YuNetConfig::default())
                        .ok()
                        .map(|det| {
                            let arc = Arc::new(det);
                            if let Ok(mut g) = cached_yunet.lock() {
                                *g = Some((ykey, Arc::clone(&arc)));
                            }
                            arc
                        })
                }
            };
            if let Some(det) = det_arc
                && let Ok(mut faces) = det.detect_faces_in_rgb(&rgb)
            {
                faces.sort_by(|a, b| {
                    b.bounding_box
                        .area()
                        .partial_cmp(&a.bounding_box.area())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                oriented_guides = faces
                    .iter()
                    .take(4)
                    .map(|f| entry_to_face_guide(f.bounding_box, f.landmarks))
                    .collect();
            }
        }

        let pkey = parser_path.to_string_lossy().to_string();
        let parser_arc = {
            let existing = cached_parser
                .lock()
                .ok()
                .and_then(|g| g.as_ref().filter(|(k, _)| *k == pkey).map(|(_, e)| Arc::clone(e)));
            if let Some(eng) = existing {
                eng
            } else {
                let eng = OnnxBiSeNetFaceParser::from_model_file(&parser_path, 4)
                    .map_err(|e| format!("failed to load BiSeNet face parser: {e}"))?;
                let arc = Arc::new(eng);
                if let Ok(mut g) = cached_parser.lock() {
                    *g = Some((pkey, Arc::clone(&arc)));
                }
                arc
            }
        };

        let map = parser_arc
            .parse_portrait_rgba(&rgb, &oriented_guides)
            .map_err(|e| format!("BiSeNet face parsing failed: {e}"))?;
        let elapsed_ms = started.elapsed().as_millis() as u64;
        let ori_key = format!(
            "{}:{}:{}",
            qt, crop_transform.flip_horizontal as u8, crop_transform.flip_vertical as u8
        );

        if let Ok(mut guard) = active_shoot.lock()
            && let Some(ctx) = guard.as_mut()
        {
            ctx.detected_face_parse_maps
                .insert(pid_clone, (ori_key, map.clone()));
        }

        Ok(Some(FaceParseMapDto::from_map(
            &map,
            parser_arc.engine_label(),
            elapsed_ms,
        )))
    })
    .await
    .map_err(|err| format!("face parsing worker failed: {err}"))?
}

/// Generates a non-destructive Phase 3 `InpaintPatch` for painted brush strokes using LaMa ONNX
/// (`lama_fp32.onnx`) when installed, or Kroma's fast built-in Pyramidal Harmonic Fill fallback.
#[tauri::command]
pub async fn inpaint_photo_region(
    photo_id: String,
    strokes: Vec<InpaintStrokeDto>,
    state: State<'_, AppState>,
) -> Result<GeneratedInpaintPatchDto, String> {
    if strokes.is_empty() {
        return Err("paint over an object before running AI Remove".to_string());
    }

    let (source_path, preview_path, recipe) = {
        let guard = state
            .active_shoot
            .lock()
            .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
        let ctx = guard
            .as_ref()
            .ok_or_else(|| "no active shoot loaded".to_string())?;
        let source = ctx
            .photo_sources
            .get(&photo_id)
            .cloned()
            .ok_or_else(|| format!("photo '{photo_id}' not found in active shoot"))?;
        let preview = ctx
            .preview_paths
            .get(&photo_id)
            .cloned()
            .unwrap_or_else(|| source.clone());
        let rec = ctx.recipes.get(&photo_id).cloned().unwrap_or_default();
        (source, preview, rec)
    };

    let cached_wire = state.get_cached_develop_wire_for_source(&source_path);
    let lama_path = state.resolve_lama_model_path();
    let cached_lama = state.cached_lama_engine.clone();
    let stroke_inputs: Vec<InpaintStrokeInput> = strokes
        .into_iter()
        .map(|s| InpaintStrokeInput {
            radius: s.radius,
            points: s.points,
        })
        .collect();

    tauri::async_runtime::spawn_blocking(move || {
        // Grade from the actual 32-bit scene-linear RAW working buffer (reusing Studio's hot
        // KR32/KR16 RAM cache when available, or decoding `source_path` directly at <=1600px) so
        // exposure & white-balance edits on `.ARW`/`.CR3`/`.NEF` match Studio's live canvas 1:1
        // instead of double-exposing the embedded 8-bit camera JPEG preview.
        let develop_buf = if let Some(wire) = cached_wire
            && let Some(decoded) = decode_working_wire_downscaled_rgb32f(&wire, 1600)
        {
            decoded
        } else {
            match decode_develop_buffer(&source_path, 1600) {
                Ok(buf) => buf,
                Err(_) => {
                    let rgb = decode_fast_preview_max_dim(&preview_path, 1600)
                        .map_err(|e| format!("failed to decode photo for AI Remove: {e}"))?;
                    DecodedDevelopBuffer {
                        image: image::DynamicImage::ImageRgb8(rgb).to_rgb32f(),
                        is_linear: false,
                        is_wide_gamut: false,
                        bit_depth: 8,
                        is_raw_demosaic: false,
                    }
                }
            }
        };

        // Grade in full-frame oriented space (preserving quarter-turns & flips + live color grade
        // + existing patches/spots, while deferring fine crop, masks, portrait retouch & sharpening
        // to the live non-destructive pipeline).
        let mut full_frame_recipe = recipe;
        full_frame_recipe.crop_transform.rect = NormalizedRect::full_frame();
        full_frame_recipe.crop_transform.rotation_deg = 0.0;
        full_frame_recipe.masks.clear();
        full_frame_recipe.background_removal.enabled = false;
        full_frame_recipe.retouch.portrait.enabled = false;
        full_frame_recipe.detail_optics.sharpening_amount = 0;
        full_frame_recipe.detail_optics.grain_amount = 0;

        let engine = RayonImageDevelopEngine::new();
        let graded_oriented = engine
            .process_buffer_to_rgb8(&develop_buf, &full_frame_recipe)
            .map_err(|e| format!("failed to prepare graded frame for AI Remove: {e}"))?;

        let lama_arc: Option<Arc<OnnxLamaInpainter>> = if let Some(ref p) = lama_path {
            let key = p.to_string_lossy().to_string();
            let existing = cached_lama
                .lock()
                .ok()
                .and_then(|g| g.as_ref().filter(|(k, _)| *k == key).map(|(_, e)| Arc::clone(e)));
            if let Some(eng) = existing {
                Some(eng)
            } else {
                match OnnxLamaInpainter::from_model_file(p, 4) {
                    Ok(eng) => {
                        let arc = Arc::new(eng);
                        if let Ok(mut g) = cached_lama.lock() {
                            *g = Some((key, Arc::clone(&arc)));
                        }
                        Some(arc)
                    }
                    Err(err) => {
                        eprintln!(
                            "[kroma] LaMa ONNX load failed ({}), using Harmonic Fill fallback: {err}",
                            p.display()
                        );
                        None
                    }
                }
            }
        } else {
            None
        };

        let generated = generate_inpaint_patch(&graded_oriented, &stroke_inputs, lama_arc.as_deref())
            .map_err(|e| format!("AI Remove failed: {e}"))?;

        Ok(GeneratedInpaintPatchDto {
            patch: (&generated.patch).into(),
            elapsed_ms: generated.elapsed_ms,
        })
    })
    .await
    .map_err(|err| format!("AI Remove worker failed: {err}"))?
}

#[tauri::command]
pub async fn generate_ai_mask(
    photo_id: String,
    kind: String,
    state: State<'_, AppState>,
) -> Result<GeneratedAiMaskDto, String> {
    let parsed_kind = AiMaskKind::parse(&kind);
    let (preview_path, quarter_turns, flip_h, flip_v) = {
        let guard = state
            .active_shoot
            .lock()
            .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
        let ctx = guard
            .as_ref()
            .ok_or_else(|| "no active shoot loaded".to_string())?;
        let source = ctx
            .photo_sources
            .get(&photo_id)
            .cloned()
            .ok_or_else(|| format!("photo '{photo_id}' not found in active shoot"))?;
        let preview = ctx
            .preview_paths
            .get(&photo_id)
            .cloned()
            .unwrap_or(source);
        let crop = ctx
            .recipes
            .get(&photo_id)
            .map(|r| r.crop_transform)
            .unwrap_or_default();
        (preview, crop.quarter_turns % 4, crop.flip_horizontal, crop.flip_vertical)
    };

    let seg_path = state.resolve_segmenter_model_path();
    let (yunet_path, _) = state.resolve_model_paths();
    let face_parser_path = state.resolve_face_parser_model_path();
    let cached_seg = state.cached_segmenter_engine.clone();
    let cached_yunet = state.cached_yunet_engine.clone();
    let cached_face_parser = state.cached_face_parser_engine.clone();

    tauri::async_runtime::spawn_blocking(move || {
        let mut rgb = decode_fast_preview_max_dim(&preview_path, 1024)
            .map_err(|e| format!("failed to decode preview for AI masking: {e}"))?;

        // Match exact Develop oriented image space (quarter_turns then flip_h / flip_v)
        rgb = match quarter_turns {
            1 => image::imageops::rotate90(&rgb),
            2 => image::imageops::rotate180(&rgb),
            3 => image::imageops::rotate270(&rgb),
            _ => rgb,
        };
        if flip_h {
            rgb = image::imageops::flip_horizontal(&rgb);
        }
        if flip_v {
            rgb = image::imageops::flip_vertical(&rgb);
        }

        // Load or reuse cached ONNX segmenter if configured (used directly for Subject/Background
        // and as an exact subject/background gate for Face/Skin and Sky masks)
        let segmenter_arc: Option<Arc<OnnxSubjectSegmenter>> = if let Some(ref p) = seg_path {
            let key = p.to_string_lossy().to_string();
            let existing = cached_seg
                .lock()
                .ok()
                .and_then(|g| g.as_ref().filter(|(k, _)| *k == key).map(|(_, e)| Arc::clone(e)));
            if let Some(eng) = existing {
                Some(eng)
            } else {
                match OnnxSubjectSegmenter::from_model_file(p, 4) {
                    Ok(eng) => {
                        let arc = Arc::new(eng);
                        if let Ok(mut g) = cached_seg.lock() {
                            *g = Some((key, Arc::clone(&arc)));
                        }
                        Some(arc)
                    }
                    Err(err) => {
                        eprintln!("[kroma] ONNX segmenter load failed ({}): {err}", p.display());
                        None
                    }
                }
            }
        } else {
            None
        };

        // Load or reuse cached bundled YuNet detector for Portrait Face/Skin & fallback Subject AI
        let yunet_arc: Option<Arc<YuNetOnnxDetector>> = if matches!(parsed_kind, AiMaskKind::FaceSkin)
            || (segmenter_arc.is_none() && !matches!(parsed_kind, AiMaskKind::Sky))
        {
            if let Some(ref p) = yunet_path {
                let key = p.to_string_lossy().to_string();
                let existing = cached_yunet
                    .lock()
                    .ok()
                    .and_then(|g| g.as_ref().filter(|(k, _)| *k == key).map(|(_, e)| Arc::clone(e)));
                if let Some(det) = existing {
                    Some(det)
                } else {
                    match YuNetOnnxDetector::from_model_file(p, YuNetConfig::default()) {
                        Ok(det) => {
                            let arc = Arc::new(det);
                            if let Ok(mut g) = cached_yunet.lock() {
                                *g = Some((key, Arc::clone(&arc)));
                            }
                            Some(arc)
                        }
                        Err(_) => None,
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        // Optional Phase 3 BiSeNet face parser automatically upgrades AI Face & Skin masks
        let face_parser_arc: Option<Arc<OnnxBiSeNetFaceParser>> =
            if matches!(parsed_kind, AiMaskKind::FaceSkin) {
                if let Some(ref p) = face_parser_path {
                    let key = p.to_string_lossy().to_string();
                    let existing = cached_face_parser.lock().ok().and_then(|g| {
                        g.as_ref()
                            .filter(|(k, _)| *k == key)
                            .map(|(_, e)| Arc::clone(e))
                    });
                    if let Some(eng) = existing {
                        Some(eng)
                    } else {
                        OnnxBiSeNetFaceParser::from_model_file(p, 4)
                            .ok()
                            .map(|eng| {
                                let arc = Arc::new(eng);
                                if let Ok(mut g) = cached_face_parser.lock() {
                                    *g = Some((key, Arc::clone(&arc)));
                                }
                                arc
                            })
                    }
                } else {
                    None
                }
            } else {
                None
            };

        let result = generate_ai_mask_for_rgb_with_face_parser(
            &rgb,
            parsed_kind,
            segmenter_arc.as_deref(),
            yunet_arc.as_deref(),
            face_parser_arc.as_deref(),
        )
        .map_err(|e| format!("AI mask generation failed: {e}"))?;

        Ok(GeneratedAiMaskDto {
            kind: result.kind.as_str().to_string(),
            matte_width: result.matte_width,
            matte_height: result.matte_height,
            matte_rle_b64: result.matte_rle_b64,
            engine_label: result.engine_label,
            elapsed_ms: result.elapsed_ms,
        })
    })
    .await
    .map_err(|err| format!("AI mask worker failed: {err}"))?
}

/// Decodes a compact 384px sRGB `f32` histogram proxy buffer for standard photos (JPEG/PNG/WebP)
/// once and returns it alongside the computed histogram so subsequent slider pauses compute live
/// histograms in ~1–2ms straight from RAM with zero disk I/O.
fn compute_histogram_for_photo(
    engine: &RayonImageDevelopEngine,
    cached_hist_buf: Option<Arc<DecodedDevelopBuffer>>,
    preview_path: &str,
    recipe: &DevelopRecipe,
) -> Result<(kroma_domain::develop::RgbHistogram, Option<Arc<DecodedDevelopBuffer>>), String> {
    if let Some(hist_buf) = cached_hist_buf {
        let hist = engine
            .compute_histogram_from_buffer(&hist_buf, recipe)
            .map_err(|e| e.to_string())?;
        return Ok((hist, None));
    }

    if let Ok(rgb) = decode_fast_preview_max_dim(preview_path, 384) {
        let fresh_buf = Arc::new(DecodedDevelopBuffer {
            image: image::DynamicImage::ImageRgb8(rgb).to_rgb32f(),
            is_linear: false,
            is_wide_gamut: false,
            bit_depth: 8,
            is_raw_demosaic: false,
        });
        let hist = engine
            .compute_histogram_from_buffer(&fresh_buf, recipe)
            .map_err(|e| e.to_string())?;
        return Ok((hist, Some(fresh_buf)));
    }

    let hist = engine
        .compute_histogram(preview_path, recipe)
        .map_err(|e| e.to_string())?;
    Ok((hist, None))
}

#[tauri::command]
pub async fn save_photo_develop_recipe(
    photo_id: String,
    recipe: DevelopRecipeDto,
    state: State<'_, AppState>,
) -> Result<HistogramDto, String> {
    let active_shoot = state.active_shoot.clone();
    let (sidecars_dir, preview_path, source_path) = {
        let guard = active_shoot
            .lock()
            .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
        let ctx = guard
            .as_ref()
            .ok_or_else(|| "no active shoot loaded".to_string())?;
        let source = ctx
            .photo_sources
            .get(&photo_id)
            .cloned()
            .ok_or_else(|| format!("photo '{photo_id}' not found in active shoot"))?;
        let preview = ctx
            .preview_paths
            .get(&photo_id)
            .cloned()
            .unwrap_or_else(|| source.clone());
        (ctx.sidecars_dir.clone(), preview, source)
    };
    let cached_hist_buf = state.get_cached_hist_buffer(&source_path);
    let source_path_for_cache = source_path.clone();

    let (hist_dto, new_std_hist_buf) = tauri::async_runtime::spawn_blocking(move || {
        let pid = PhotoId::new(&photo_id).map_err(|e| e.to_string())?;
        let store = JsonSidecarRecipeStore::new(&sidecars_dir);
        let save_uc = SavePhotoDevelopRecipeUseCase::new(&store);
        let saved = save_uc
            .execute(SavePhotoDevelopRecipeInput {
                photo_id: pid,
                recipe: recipe.into_domain(),
            })
            .map_err(|e| e.to_string())?;

        if let Ok(mut guard) = active_shoot.lock()
            && let Some(ctx) = guard.as_mut()
        {
            ctx.recipes
                .insert(photo_id.clone(), saved.clamped_recipe.clone());
        }

        let engine = RayonImageDevelopEngine::new();
        let (histogram, fresh_buf) = compute_histogram_for_photo(
            &engine,
            cached_hist_buf,
            &preview_path,
            &saved.clamped_recipe,
        )?;
        Ok::<(HistogramDto, Option<Arc<DecodedDevelopBuffer>>), String>((
            HistogramDto::from(&histogram),
            fresh_buf,
        ))
    })
    .await
    .map_err(|err| format!("recipe save worker failed: {err}"))??;

    if let Some(fresh_buf) = new_std_hist_buf {
        state.insert_std_hist_buffer(source_path_for_cache, fresh_buf);
    }

    Ok(hist_dto)
}

/// Coarse-grained batch persistence for Develop Undo/Redo, Paste, and pre-Export flushes:
/// writes all `.kroma.json` sidecars inside a single blocking worker without spawning per-photo
/// image decode tasks, and computes at most ONE histogram (only for `active_photo_id`).
#[tauri::command]
pub async fn save_photo_develop_recipes_batch(
    items: Vec<PhotoRecipeSaveItemDto>,
    active_photo_id: Option<String>,
    state: State<'_, AppState>,
) -> Result<Option<HistogramDto>, String> {
    if items.is_empty() {
        return Ok(None);
    }

    let active_shoot = state.active_shoot.clone();
    let (sidecars_dir, active_preview_path, active_source_path) = {
        let guard = active_shoot
            .lock()
            .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
        let ctx = guard
            .as_ref()
            .ok_or_else(|| "no active shoot loaded".to_string())?;
        let source = active_photo_id
            .as_deref()
            .and_then(|pid| ctx.photo_sources.get(pid).cloned());
        let preview = active_photo_id.as_deref().and_then(|pid| {
            ctx.preview_paths
                .get(pid)
                .or_else(|| ctx.photo_sources.get(pid))
                .cloned()
        });
        (ctx.sidecars_dir.clone(), preview, source)
    };
    let cached_hist_buf = active_source_path
        .as_deref()
        .and_then(|src| state.get_cached_hist_buffer(src));

    let (histogram, new_std_hist_buf) = tauri::async_runtime::spawn_blocking(move || {
        let store = JsonSidecarRecipeStore::new(&sidecars_dir);
        let save_uc = SavePhotoDevelopRecipeUseCase::new(&store);
        let mut saved_pairs: Vec<(String, DevelopRecipe)> = Vec::with_capacity(items.len());
        let mut active_saved_recipe: Option<DevelopRecipe> = None;

        for item in items {
            let Ok(pid) = PhotoId::new(&item.photo_id) else {
                continue;
            };
            let saved = save_uc
                .execute(SavePhotoDevelopRecipeInput {
                    photo_id: pid,
                    recipe: item.recipe.into_domain(),
                })
                .map_err(|e| e.to_string())?;
            if active_photo_id.as_deref() == Some(item.photo_id.as_str()) {
                active_saved_recipe = Some(saved.clamped_recipe.clone());
            }
            saved_pairs.push((item.photo_id, saved.clamped_recipe));
        }

        if let Ok(mut guard) = active_shoot.lock()
            && let Some(ctx) = guard.as_mut()
        {
            for (pid, recipe) in saved_pairs {
                ctx.recipes.insert(pid, recipe);
            }
        }

        let (histogram, fresh_buf) = match (active_saved_recipe, active_preview_path) {
            (Some(recipe), Some(preview_path)) => {
                let engine = RayonImageDevelopEngine::new();
                match compute_histogram_for_photo(&engine, cached_hist_buf, &preview_path, &recipe) {
                    Ok((h, buf)) => (Some(HistogramDto::from(&h)), buf),
                    Err(_) => (None, None),
                }
            }
            (Some(recipe), None) => {
                let engine = RayonImageDevelopEngine::new();
                let h = cached_hist_buf
                    .as_ref()
                    .and_then(|buf| engine.compute_histogram_from_buffer(buf, &recipe).ok())
                    .map(|h| HistogramDto::from(&h));
                (h, None)
            }
            _ => (None, None),
        };

        Ok::<(Option<HistogramDto>, Option<Arc<DecodedDevelopBuffer>>), String>((
            histogram,
            fresh_buf,
        ))
    })
    .await
    .map_err(|err| format!("batch recipe save worker failed: {err}"))??;

    if let (Some(src), Some(fresh_buf)) = (active_source_path, new_std_hist_buf) {
        state.insert_std_hist_buffer(src, fresh_buf);
    }

    Ok(histogram)
}

#[tauri::command]
pub async fn sync_cluster_develop_recipe(
    source_photo_id: String,
    target_photo_ids: Vec<String>,
    mask: DevelopSyncMaskDto,
    state: State<'_, AppState>,
) -> Result<BatchSyncResultDto, String> {
    let active_shoot = state.active_shoot.clone();
    let sidecars_dir = {
        let guard = active_shoot
            .lock()
            .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
        guard
            .as_ref()
            .ok_or_else(|| "no active shoot loaded".to_string())?
            .sidecars_dir
            .clone()
    };

    tauri::async_runtime::spawn_blocking(move || {
        let src_id = PhotoId::new(source_photo_id).map_err(|e| e.to_string())?;
        let targets = target_photo_ids
            .into_iter()
            .map(|id| PhotoId::new(id).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?;

        let store = JsonSidecarRecipeStore::new(&sidecars_dir);
        let sync_uc = BatchSyncDevelopUseCase::new(&store);
        let out = sync_uc
            .execute(BatchSyncDevelopInput {
                source_photo_id: src_id,
                target_photo_ids: targets,
                sync_mask: mask.into(),
            })
            .map_err(|e| e.to_string())?;

        if let Ok(mut guard) = active_shoot.lock()
            && let Some(ctx) = guard.as_mut()
        {
            for pid in &out.synced_photo_ids {
                if let Ok(Some(r)) = store.load_recipe(pid) {
                    ctx.recipes.insert(pid.as_str().to_string(), r);
                }
            }
        }

        Ok(BatchSyncResultDto {
            synced_count: out.synced_photo_ids.len(),
            failed_count: out.failures.len(),
        })
    })
    .await
    .map_err(|err| format!("sync worker failed: {err}"))?
}

#[tauri::command]
pub async fn apply_relative_develop_nudge(
    photo_ids: Vec<String>,
    delta: RelativeDevelopDeltaDto,
    state: State<'_, AppState>,
) -> Result<BatchSyncResultDto, String> {
    let active_shoot = state.active_shoot.clone();
    let sidecars_dir = {
        let guard = active_shoot
            .lock()
            .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
        guard
            .as_ref()
            .ok_or_else(|| "no active shoot loaded".to_string())?
            .sidecars_dir
            .clone()
    };

    tauri::async_runtime::spawn_blocking(move || {
        let targets = photo_ids
            .into_iter()
            .map(|id| PhotoId::new(id).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?;

        let store = JsonSidecarRecipeStore::new(&sidecars_dir);
        let nudge_uc = BatchRelativeNudgeUseCase::new(&store);
        let out = nudge_uc
            .execute(BatchRelativeNudgeInput {
                target_photo_ids: targets.clone(),
                delta: delta.into(),
            })
            .map_err(|e| e.to_string())?;

        if let Ok(mut guard) = active_shoot.lock()
            && let Some(ctx) = guard.as_mut()
        {
            for pid in &targets {
                if let Ok(Some(r)) = store.load_recipe(pid) {
                    ctx.recipes.insert(pid.as_str().to_string(), r);
                }
            }
        }

        Ok(BatchSyncResultDto {
            synced_count: out.updated_count,
            failed_count: out.failed_ids.len(),
        })
    })
    .await
    .map_err(|err| format!("nudge worker failed: {err}"))?
}

#[tauri::command]
pub async fn calibrate_wb_from_slate(
    slate_photo_id: String,
    target_photo_ids: Vec<String>,
    state: State<'_, AppState>,
) -> Result<WhiteBalanceDto, String> {
    let active_shoot = state.active_shoot.clone();
    let (sidecars_dir, slate_path) = {
        let guard = active_shoot
            .lock()
            .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
        let ctx = guard
            .as_ref()
            .ok_or_else(|| "no active shoot loaded".to_string())?;
        let path = ctx
            .preview_paths
            .get(&slate_photo_id)
            .or_else(|| ctx.photo_sources.get(&slate_photo_id))
            .cloned()
            .ok_or_else(|| format!("reference photo '{slate_photo_id}' not found"))?;
        (ctx.sidecars_dir.clone(), path)
    };

    tauri::async_runtime::spawn_blocking(move || {
        let targets = target_photo_ids
            .into_iter()
            .map(|id| PhotoId::new(id).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?;

        let engine = RayonImageDevelopEngine::new();
        let store = JsonSidecarRecipeStore::new(&sidecars_dir);
        let wb_uc = CalibrateWhiteBalanceFromSlateUseCase::new(&engine, &store);
        let out = wb_uc
            .execute(CalibrateWhiteBalanceFromSlateInput {
                slate_source_path: slate_path,
                whiteboard_region: NormalizedRect::new(0.35, 0.55, 0.30, 0.20)
                    .map_err(|e| e.to_string())?,
                target_photo_ids: targets.clone(),
            })
            .map_err(|e| e.to_string())?;

        if let Ok(mut guard) = active_shoot.lock()
            && let Some(ctx) = guard.as_mut()
        {
            for pid in &targets {
                if let Ok(Some(r)) = store.load_recipe(pid) {
                    ctx.recipes.insert(pid.as_str().to_string(), r);
                }
            }
        }

        Ok(WhiteBalanceDto {
            temperature_kelvin: out.calibrated_wb.temperature_kelvin,
            tint: out.calibrated_wb.tint,
        })
    })
    .await
    .map_err(|err| format!("wb calibration worker failed: {err}"))?
}

fn pick_evenly_spaced_indices(len: usize, max_samples: usize) -> Vec<usize> {
    if len == 0 || max_samples == 0 {
        return Vec::new();
    }
    if len <= max_samples {
        return (0..len).collect();
    }
    if max_samples == 1 {
        return vec![len / 2];
    }
    let mut indices = Vec::with_capacity(max_samples);
    for i in 0..max_samples {
        let idx = (i * (len - 1)) / (max_samples - 1);
        if indices.last().copied() != Some(idx) {
            indices.push(idx);
        }
    }
    indices
}

fn parse_selected_photo_ids(photo_ids: Option<&[String]>) -> Option<HashSet<PhotoId>> {
    let raw = photo_ids?;
    if raw.is_empty() {
        return None;
    }
    let set: HashSet<PhotoId> = raw
        .iter()
        .filter_map(|id| PhotoId::new(id.clone()).ok())
        .collect();
    if set.is_empty() {
        None
    } else {
        Some(set)
    }
}

#[tauri::command]
pub async fn estimate_export_size(
    request: EstimateExportSizeRequestDto,
    state: State<'_, AppState>,
) -> Result<Option<EstimateExportSizeResponseDto>, String> {
    let estimate_gen = state.export_estimate_generation.clone();
    let my_gen = estimate_gen.fetch_add(1, Ordering::SeqCst) + 1;

    struct SampleJob {
        source_path: String,
        preview_path: Option<String>,
        recipe: DevelopRecipe,
    }

    let selected_ids = parse_selected_photo_ids(request.photo_ids.as_deref());
    let mode = request.mode.to_ascii_lowercase();
    let needs_render_sampling = mode != "move" && mode != "hardlink" && mode != "copy";

    let (
        total_photos,
        mut total_source_bytes,
        mut companion_raw_bytes,
        companion_count,
        missing_size_paths,
        samples,
    ) = {
        let guard = state
            .active_shoot
            .lock()
            .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
        let ctx = guard
            .as_ref()
            .ok_or_else(|| "no active shoot loaded".to_string())?;

        struct CandidateRef<'a> {
            photo_id: &'a str,
            source_path: &'a str,
            has_raw_companion: bool,
        }

        let mut candidates: Vec<CandidateRef<'_>> = Vec::new();
        for cluster in &ctx.plan.clusters {
            for entry in &cluster.entries {
                if entry.is_slate_photo && request.exclude_slate_photo {
                    if request.archive_excluded_slates {
                        candidates.push(CandidateRef {
                            photo_id: entry.photo_id.as_str(),
                            source_path: entry.source_path.as_str(),
                            has_raw_companion: false,
                        });
                    }
                    continue;
                }
                candidates.push(CandidateRef {
                    photo_id: entry.photo_id.as_str(),
                    source_path: entry.source_path.as_str(),
                    has_raw_companion: true,
                });
            }
        }
        for group in &ctx.plan.group_photos {
            candidates.push(CandidateRef {
                photo_id: group.photo_id.as_str(),
                source_path: group.source_path.as_str(),
                has_raw_companion: true,
            });
        }
        for nf in &ctx.plan.no_face_photos {
            candidates.push(CandidateRef {
                photo_id: nf.photo_id.as_str(),
                source_path: nf.source_path.as_str(),
                has_raw_companion: true,
            });
        }

        if let Some(ref filter) = selected_ids {
            candidates.retain(|c| {
                PhotoId::new(c.photo_id.to_string())
                    .map(|id| filter.contains(&id))
                    .unwrap_or(false)
            });
        }

        let total_photos = candidates.len();
        let mut total_source_bytes: u64 = 0;
        let mut companion_raw_bytes: u64 = 0;
        let mut companion_count: usize = 0;
        let mut missing_size_paths: Vec<(String, bool)> = Vec::new();

        for item in &candidates {
            if item.has_raw_companion {
                companion_count += 1;
            }
            match ctx.photo_file_sizes.get(item.photo_id).copied() {
                Some(bytes) if bytes > 0 => {
                    total_source_bytes = total_source_bytes.saturating_add(bytes);
                    if item.has_raw_companion {
                        companion_raw_bytes = companion_raw_bytes.saturating_add(bytes);
                    }
                }
                _ => {
                    missing_size_paths.push((item.source_path.to_string(), item.has_raw_companion));
                }
            }
        }

        // Only clone the <=5 sampled recipes and preview paths instead of cloning the full shoot.
        let samples: Vec<SampleJob> = if needs_render_sampling && total_photos > 0 {
            let sample_indices = pick_evenly_spaced_indices(total_photos, 5);
            sample_indices
                .into_iter()
                .map(|idx| {
                    let item = &candidates[idx];
                    let preview_path = ctx
                        .source_to_preview
                        .get(item.source_path)
                        .filter(|p| p.as_str() != item.source_path)
                        .cloned();
                    let recipe = ctx
                        .recipes
                        .get(item.photo_id)
                        .cloned()
                        .unwrap_or_default();
                    SampleJob {
                        source_path: item.source_path.to_string(),
                        preview_path,
                        recipe,
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        (
            total_photos,
            total_source_bytes,
            companion_raw_bytes,
            companion_count,
            missing_size_paths,
            samples,
        )
    };

    if total_photos == 0 {
        return Ok(Some(EstimateExportSizeResponseDto {
            extra_disk_bytes: 0,
            edited_bytes: 0,
            originals_bytes: 0,
            total_source_bytes: 0,
            file_count: 0,
            sampled_photo_count: 0,
            is_exact: true,
        }));
    }

    tauri::async_runtime::spawn_blocking(move || {
        for (path, has_companion) in missing_size_paths {
            if estimate_gen.load(Ordering::SeqCst) != my_gen {
                return Ok(None);
            }
            let bytes = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            total_source_bytes = total_source_bytes.saturating_add(bytes);
            if has_companion {
                companion_raw_bytes = companion_raw_bytes.saturating_add(bytes);
            }
        }

        if mode == "move" || mode == "hardlink" {
            return Ok(Some(EstimateExportSizeResponseDto {
                extra_disk_bytes: 0,
                edited_bytes: 0,
                originals_bytes: total_source_bytes,
                total_source_bytes,
                file_count: total_photos,
                sampled_photo_count: 0,
                is_exact: true,
            }));
        }

        if mode == "copy" {
            return Ok(Some(EstimateExportSizeResponseDto {
                extra_disk_bytes: total_source_bytes,
                edited_bytes: 0,
                originals_bytes: total_source_bytes,
                total_source_bytes,
                file_count: total_photos,
                sampled_photo_count: 0,
                is_exact: true,
            }));
        }

        let format = match request.format.to_ascii_lowercase().as_str() {
            "png" => OutputImageFormat::Png,
            "tiff" | "tif" => OutputImageFormat::Tiff,
            _ => OutputImageFormat::Jpeg {
                quality: request.quality.clamp(50, 100),
            },
        };

        let engine = RayonImageDevelopEngine::new();
        let mut measured: Vec<u64> = Vec::with_capacity(samples.len());

        // Measure samples sequentially (1 compact preview at a time) and bail immediately if a newer
        // settings change superseded this request, keeping RAM <~25 MB and CPU burst minimal.
        for sample in &samples {
            if estimate_gen.load(Ordering::SeqCst) != my_gen {
                return Ok(None);
            }
            if let Ok(bytes) = engine.measure_rendered_bytes_with_preview(
                &sample.source_path,
                sample.preview_path.as_deref(),
                &sample.recipe,
                format,
            ) {
                measured.push(bytes);
            }
        }

        if estimate_gen.load(Ordering::SeqCst) != my_gen {
            return Ok(None);
        }

        let sampled_photo_count = measured.len();
        let edited_bytes = if sampled_photo_count > 0 {
            let sum: u64 = measured.iter().sum();
            let avg = (sum as f64) / (sampled_photo_count as f64);
            (avg * (total_photos as f64)).round() as u64
        } else {
            total_source_bytes
        };

        let is_dual = mode == "develop_and_raw";
        let extra_companion_count = if is_dual { companion_count } else { 0 };
        let originals_bytes = if is_dual { companion_raw_bytes } else { 0 };
        let extra_disk_bytes = edited_bytes.saturating_add(originals_bytes);

        Ok(Some(EstimateExportSizeResponseDto {
            extra_disk_bytes,
            edited_bytes,
            originals_bytes,
            total_source_bytes,
            file_count: total_photos + extra_companion_count,
            sampled_photo_count,
            is_exact: sampled_photo_count == total_photos
                && sampled_photo_count > 0
                && matches!(format, OutputImageFormat::Tiff),
        }))
    })
    .await
    .map_err(|err| format!("export size estimate worker failed: {err}"))?
}

#[tauri::command]
pub async fn organize_and_export_shoot(
    app: AppHandle,
    request: ExportShootRequestDto,
    state: State<'_, AppState>,
) -> Result<ExportShootResultDto, String> {
    let (plan, sidecars_dir, face_guides_by_source, face_parse_maps_by_source) = {
        let guard = state
            .active_shoot
            .lock()
            .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
        let ctx = guard
            .as_ref()
            .ok_or_else(|| "no active shoot loaded".to_string())?;
        (
            ctx.plan.clone(),
            ctx.sidecars_dir.clone(),
            collect_shoot_face_guides_by_source(ctx),
            collect_shoot_face_parse_maps_by_source(ctx),
        )
    };
    let (yunet_path, _) = state.resolve_model_paths();
    let face_parser_path = state.resolve_face_parser_model_path();
    let cached_yunet = state.cached_yunet_engine.clone();
    let cached_face_parser = state.cached_face_parser_engine.clone();

    let app_handle = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let template =
            OutputNamingTemplate::new(request.naming_pattern, request.exclude_slate_photo)
                .map_err(|e| e.to_string())?;

        let format = match request.format.to_ascii_lowercase().as_str() {
            "png" => OutputImageFormat::Png,
            "tiff" | "tif" => OutputImageFormat::Tiff,
            _ => OutputImageFormat::Jpeg {
                quality: request.quality.clamp(50, 100),
            },
        };

        let execution_mode = match request.mode.to_ascii_lowercase().as_str() {
            "develop_and_raw" => OrganizeExecutionMode::DevelopAndKeepRaw {
                format,
                raw_transfer_mode: OrganizeTransferMode::Copy,
            },
            "move" => OrganizeExecutionMode::TransferOriginals(OrganizeTransferMode::Move),
            "hardlink" => OrganizeExecutionMode::TransferOriginals(OrganizeTransferMode::HardLink),
            "develop" => OrganizeExecutionMode::DevelopAndExport { format },
            _ => OrganizeExecutionMode::TransferOriginals(OrganizeTransferMode::Copy),
        };

        let fallback_yunet: Option<Arc<YuNetOnnxDetector>> = if let Some(ref p) = yunet_path {
            let key = p.to_string_lossy().to_string();
            let existing = cached_yunet
                .lock()
                .ok()
                .and_then(|g| g.as_ref().filter(|(k, _)| *k == key).map(|(_, e)| Arc::clone(e)));
            if let Some(det) = existing {
                Some(det)
            } else {
                YuNetOnnxDetector::from_model_file(p, YuNetConfig::default())
                    .ok()
                    .map(|det| {
                        let arc = Arc::new(det);
                        if let Ok(mut g) = cached_yunet.lock() {
                            *g = Some((key, Arc::clone(&arc)));
                        }
                        arc
                    })
            }
        } else {
            None
        };

        let fallback_face_parser: Option<Arc<OnnxBiSeNetFaceParser>> =
            if let Some(ref p) = face_parser_path {
                let key = p.to_string_lossy().to_string();
                let existing = cached_face_parser.lock().ok().and_then(|g| {
                    g.as_ref()
                        .filter(|(k, _)| *k == key)
                        .map(|(_, e)| Arc::clone(e))
                });
                if let Some(eng) = existing {
                    Some(eng)
                } else {
                    OnnxBiSeNetFaceParser::from_model_file(p, 4)
                        .ok()
                        .map(|eng| {
                            let arc = Arc::new(eng);
                            if let Ok(mut g) = cached_face_parser.lock() {
                                *g = Some((key, Arc::clone(&arc)));
                            }
                            arc
                        })
                }
            } else {
                None
            };

        let organizer = LocalFileOrganizer::new();
        let organize_uc = OrganizeShootFoldersUseCase::new(&organizer);
        let engine = RayonImageDevelopEngine::with_retouch_context(
            face_guides_by_source,
            fallback_yunet,
            face_parse_maps_by_source,
            fallback_face_parser,
        );
        let store = JsonSidecarRecipeStore::new(&sidecars_dir);

        let started = Instant::now();
        let last_emit_ms = AtomicU64::new(0);
        emit_progress(&app_handle, "export", 0, 1, "Preparing export…".to_string());

        let selected_photo_ids = parse_selected_photo_ids(request.photo_ids.as_deref());
        let write_manifest_json = selected_photo_ids.is_none();

        let out = organize_uc
            .execute_with_progress(
                OrganizeShootInput {
                    plan,
                    output_root: request.output_root,
                    naming_template: template,
                    execution_mode,
                    archive_excluded_slates: request.archive_excluded_slates,
                    write_manifest_json,
                    selected_photo_ids,
                },
                Some(&engine),
                Some(&store),
                &|done, total, label| {
                    let now_ms = started.elapsed().as_millis() as u64;
                    let prev = last_emit_ms.load(Ordering::Relaxed);
                    let is_boundary = done == 0 || done == 1 || done >= total;
                    if is_boundary
                        || (now_ms.saturating_sub(prev) >= 40
                            && last_emit_ms
                                .compare_exchange(prev, now_ms, Ordering::Relaxed, Ordering::Relaxed)
                                .is_ok())
                    {
                        if is_boundary {
                            last_emit_ms.store(now_ms, Ordering::Relaxed);
                        }
                        let msg = if label.is_empty() {
                            format!("Exporting ({done}/{total})…")
                        } else if done == 0 || label.ends_with('…') {
                            label.to_string()
                        } else {
                            format!("Exporting {done}/{total} • {label}")
                        };
                        emit_progress(&app_handle, "export", done, total.max(1), msg);
                    }
                },
            )
            .map_err(|e| e.to_string())?;

        let final_total = out.organized_files.len() + out.failures.len();
        emit_progress(
            &app_handle,
            "export",
            final_total.max(1),
            final_total.max(1),
            format!("Exported {} file(s)", out.organized_files.len()),
        );

        Ok(ExportShootResultDto {
            organized_count: out.organized_files.len(),
            failed_count: out.failures.len(),
            manifest_path: out.manifest_path,
        })
    })
    .await
    .map_err(|err| format!("export worker failed: {err}"))?
}

struct BackgroundPreviewJob {
    photo_id: String,
    source_path: String,
    preview_dest: PathBuf,
    needs_exif: bool,
}

fn should_generate_preview_proxy(path: &Path, file_size_bytes: u64) -> bool {
    if !is_web_native_image_path(path) {
        return true;
    }
    file_size_bytes >= WEB_NATIVE_PREVIEW_PROXY_MIN_BYTES
}

fn build_preview_cache_path(
    preview_cache_dir: &Path,
    path: &Path,
    stem: &str,
    source_str: &str,
    file_size_bytes: u64,
    mtime_sec: i64,
) -> PathBuf {
    let mut hasher = DefaultHasher::new();
    source_str.hash(&mut hasher);
    if is_web_native_image_path(path) {
        file_size_bytes.hash(&mut hasher);
        mtime_sec.hash(&mut hasher);
        let path_hash = hasher.finish();
        preview_cache_dir.join(format!("v3_web_{stem}_{path_hash:016x}_preview.jpg"))
    } else {
        let path_hash = hasher.finish();
        preview_cache_dir.join(format!("v3_{stem}_{path_hash:016x}_preview.jpg"))
    }
}

fn resolve_common_parent_dir(files: &[PathBuf]) -> Option<PathBuf> {
    let first_parent = files.first()?.parent()?.to_path_buf();
    let mut prefix: Vec<std::path::Component<'_>> = first_parent.components().collect();
    for path in files.iter().skip(1) {
        let Some(parent) = path.parent() else {
            return Some(first_parent);
        };
        let next_components: Vec<std::path::Component<'_>> = parent.components().collect();
        let common_len = prefix
            .iter()
            .zip(next_components.iter())
            .take_while(|(a, b)| a == b)
            .count();
        prefix.truncate(common_len);
        if prefix.is_empty() {
            return Some(first_parent);
        }
    }
    let candidate: PathBuf = prefix.into_iter().collect();
    if candidate.is_dir() {
        Some(candidate)
    } else {
        Some(first_parent)
    }
}

fn filter_supported_photo_file_paths(raw_paths: Vec<String>) -> Vec<PathBuf> {
    let mut seen = HashSet::new();
    let mut files = Vec::new();
    for raw in raw_paths {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let path = PathBuf::from(trimmed);
        if !path.is_file() || !is_supported_photo_path(&path) {
            continue;
        }
        let dedup_key = path.to_string_lossy().replace('\\', "/").to_ascii_lowercase();
        if seen.insert(dedup_key) {
            files.push(path);
        }
    }
    files
}

fn managed_single_sidecars_bucket(root: &Path, managed_single_sidecars_dir: &Path) -> PathBuf {
    let normalized = root
        .to_string_lossy()
        .replace('\\', "/")
        .trim_end_matches('/')
        .to_ascii_lowercase();
    let mut hasher = DefaultHasher::new();
    normalized.hash(&mut hasher);
    managed_single_sidecars_dir.join(format!("dir_{:016x}", hasher.finish()))
}

/// Chooses where `.kroma.json` Develop sidecars are stored for a session:
/// - Full folder shoots always store portable sidecars in `<folder>/.kroma_sidecars`.
/// - Single-photo / selected-file sessions reuse `<parent>/.kroma_sidecars` ONLY if that shoot
///   directory already exists; otherwise they store sidecars in Kroma's managed app-data directory
///   so opening a photo from `Downloads` or `Desktop` never leaves visible stray folders behind.
fn resolve_sidecars_dir(
    root: &Path,
    session_mode: ShootSessionMode,
    managed_single_sidecars_dir: &Path,
) -> PathBuf {
    let local_sidecars = root.join(".kroma_sidecars");
    if session_mode == ShootSessionMode::Folder || local_sidecars.is_dir() {
        local_sidecars
    } else {
        managed_single_sidecars_bucket(root, managed_single_sidecars_dir)
    }
}

fn run_instant_files_load(
    file_paths: Vec<String>,
    preview_cache_dir: &Path,
    managed_single_sidecars_dir: &Path,
    app: &AppHandle,
) -> Result<(ActiveShootContext, Vec<BackgroundPreviewJob>), String> {
    let files = filter_supported_photo_file_paths(file_paths);
    if files.is_empty() {
        return Err("no supported RAW or standard image files selected".to_string());
    }

    let root = resolve_common_parent_dir(&files)
        .ok_or_else(|| "could not resolve parent directory for selected photo(s)".to_string())?;
    let folder_path = root.to_string_lossy().to_string();

    let (session_mode, project_name) = if files.len() == 1 {
        let name = files[0]
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("Photo")
            .to_string();
        (ShootSessionMode::SinglePhoto, name)
    } else {
        let parent_name = root
            .file_name()
            .and_then(|n| n.to_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("Selected Photos");
        (
            ShootSessionMode::Files,
            format!("{parent_name} ({} selected)", files.len()),
        )
    };

    run_instant_paths_load(
        folder_path,
        project_name,
        files,
        session_mode,
        preview_cache_dir,
        managed_single_sidecars_dir,
        app,
    )
}

fn run_instant_folder_load(
    folder_path: String,
    preview_cache_dir: &Path,
    managed_single_sidecars_dir: &Path,
    app: &AppHandle,
) -> Result<(ActiveShootContext, Vec<BackgroundPreviewJob>), String> {
    let root = Path::new(&folder_path);
    if !root.is_dir() {
        return Err(format!("directory '{}' does not exist", root.display()));
    }

    let mut files = Vec::new();
    collect_supported_photos(root, 0, 3, &mut files);
    files.sort();

    if files.is_empty() {
        return Err(format!(
            "no supported RAW or standard images found in '{}'",
            root.display()
        ));
    }

    let project_name = root
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("Shoot")
        .to_string();

    run_instant_paths_load(
        folder_path,
        project_name,
        files,
        ShootSessionMode::Folder,
        preview_cache_dir,
        managed_single_sidecars_dir,
        app,
    )
}

fn run_instant_paths_load(
    folder_path: String,
    project_name: String,
    files: Vec<PathBuf>,
    session_mode: ShootSessionMode,
    preview_cache_dir: &Path,
    managed_single_sidecars_dir: &Path,
    app: &AppHandle,
) -> Result<(ActiveShootContext, Vec<BackgroundPreviewJob>), String> {
    let started = Instant::now();
    let root = Path::new(&folder_path);

    fs::create_dir_all(preview_cache_dir)
        .map_err(|err| format!("failed to create preview cache dir: {err}"))?;

    let root_name = root
        .file_name()
        .and_then(|n| n.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("Shoot")
        .to_string();

    let total_files = files.len();
    emit_progress(
        app,
        "loading",
        0,
        total_files,
        format!("Indexing {total_files} photo(s)..."),
    );

    // Lightweight parallel metadata pass: ZERO full PNG decoding!
    struct IndexedFile {
        raw_id: String,
        dir_name: String,
        source_str: String,
        file_size_bytes: u64,
        preview_dest: Option<PathBuf>,
        already_cached: bool,
        needs_exif: bool,
        fallback_ts: i64,
        exact_ts: Option<i64>,
    }

    let indexed: Vec<IndexedFile> = files
        .par_iter()
        .filter_map(|path| {
            let source_str = path.to_string_lossy().to_string();
            let rel_parent = path
                .parent()
                .and_then(|p| p.strip_prefix(root).ok())
                .map(|p| p.to_string_lossy().replace('\\', "/"))
                .unwrap_or_default();
            let dir_name = if rel_parent.is_empty() {
                root_name.clone()
            } else {
                rel_parent
            };

            let stem = path.file_stem()?.to_str()?;
            let raw_id = if dir_name == root_name {
                stem.to_string()
            } else {
                format!("{}_{}", dir_name.replace('/', "_"), stem)
            };

            let file_meta = fs::metadata(path).ok();
            let file_size_bytes = file_meta.as_ref().map(|m| m.len()).unwrap_or(0);
            let fallback_ts = file_meta
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);

            // RAW/TIFF files always use a 1024px cached JPEG proxy; large web-native files
            // (e.g. 24–60 MP camera JPEGs/PNGs >= 300 KB) also get cached 1024px UI thumbnails
            // so switching tabs never triggers bursts of full-resolution image decodes in WebView2.
            let needs_proxy = should_generate_preview_proxy(path, file_size_bytes);
            let (preview_dest, already_cached) = if needs_proxy {
                let dest = build_preview_cache_path(
                    preview_cache_dir,
                    path,
                    stem,
                    &source_str,
                    file_size_bytes,
                    fallback_ts,
                );
                let exists = dest.exists();
                (Some(dest), exists)
            } else {
                (None, true)
            };

            // Fast shutter time: filename timestamp immediately + EXIF for JPEGs & already-cached RAWs
            // (uncached RAW EXIF is extracted alongside its background preview so open stays <0.5s!)
            let is_raw = is_raw_photo_path(path);
            let filename_ts = parse_filename_capture_hints(&source_str).filename_timestamp_epoch_sec;
            let exact_ts = filename_ts.or_else(|| {
                if !is_raw || already_cached {
                    extract_exif_capture_timestamp_epoch_sec(&source_str)
                } else {
                    None
                }
            });
            let needs_exif = is_raw && exact_ts.is_none();

            Some(IndexedFile {
                raw_id,
                dir_name,
                source_str,
                file_size_bytes,
                preview_dest,
                already_cached,
                needs_exif,
                fallback_ts,
                exact_ts,
            })
        })
        .collect();

    // Assign deterministic unique IDs and split initial immediate previews (first 36) from background stream
    let mut photo_sources = HashMap::with_capacity(indexed.len());
    let mut photo_file_sizes = HashMap::with_capacity(indexed.len());
    let mut preview_paths = HashMap::with_capacity(indexed.len());
    let mut source_to_preview = HashMap::with_capacity(indexed.len());
    let mut photo_directories = HashMap::with_capacity(indexed.len());
    let mut photo_capture_timestamps = HashMap::with_capacity(indexed.len());
    let mut dir_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut scan_targets = Vec::with_capacity(indexed.len());
    let mut immediate_preview_jobs: Vec<BackgroundPreviewJob> = Vec::new();
    let mut background_preview_jobs: Vec<BackgroundPreviewJob> = Vec::new();
    let mut used_safe_sidecar_ids: HashSet<String> = HashSet::with_capacity(indexed.len());

    for item in indexed {
        let base_id = item.raw_id;
        let mut unique_id = base_id.clone();
        let mut suffix = 2_usize;
        while photo_sources.contains_key(&unique_id)
            || used_safe_sidecar_ids.contains(&JsonSidecarRecipeStore::sanitize_photo_id(&unique_id))
        {
            unique_id = format!("{base_id}_{suffix}");
            suffix += 1;
        }
        used_safe_sidecar_ids.insert(JsonSidecarRecipeStore::sanitize_photo_id(&unique_id));
        let dir_id = format!(
            "dir-{}",
            item.dir_name.to_ascii_lowercase().replace(['/', ' '], "-")
        );
        *dir_counts.entry(item.dir_name).or_insert(0) += 1;

        let preview_str = match item.preview_dest {
            Some(dest) if item.already_cached => dest.to_string_lossy().to_string(),
            Some(dest) => {
                let job = BackgroundPreviewJob {
                    photo_id: unique_id.clone(),
                    source_path: item.source_str.clone(),
                    preview_dest: dest,
                    needs_exif: item.needs_exif,
                };
                if immediate_preview_jobs.len() < INITIAL_SYNC_RAW_PREVIEWS {
                    immediate_preview_jobs.push(job);
                } else {
                    background_preview_jobs.push(job);
                }
                item.source_str.clone()
            }
            None => item.source_str.clone(),
        };

        let photo_id = PhotoId::new(&unique_id).map_err(|e| e.to_string())?;
        let order_ts = item.exact_ts.unwrap_or(item.fallback_ts);
        source_to_preview.insert(item.source_str.clone(), preview_str.clone());
        photo_sources.insert(unique_id.clone(), item.source_str.clone());
        photo_file_sizes.insert(unique_id.clone(), item.file_size_bytes);
        preview_paths.insert(unique_id.clone(), preview_str);
        photo_directories.insert(unique_id.clone(), dir_id);
        photo_capture_timestamps.insert(unique_id, item.exact_ts);
        scan_targets.push(PhotoScanTarget {
            photo_id,
            source_path: item.source_str,
            capture_timestamp_epoch_sec: order_ts,
            exact_capture_timestamp_epoch_sec: item.exact_ts,
        });
    }

    // Generate the first screenful (up to 36) of uncached previews synchronously in parallel (~0.3s)
    if !immediate_preview_jobs.is_empty() {
        emit_progress(
            app,
            "loading",
            0,
            immediate_preview_jobs.len(),
            format!(
                "Preparing initial {} photo preview(s)...",
                immediate_preview_jobs.len()
            ),
        );
        let ready: Vec<(String, String, String, Option<i64>)> = immediate_preview_jobs
            .into_par_iter()
            .filter_map(|job| {
                let ts = if job.needs_exif {
                    extract_exif_capture_timestamp_epoch_sec(&job.source_path)
                } else {
                    None
                };
                let _ = generate_cached_preview(&job.source_path, &job.preview_dest);
                if job.preview_dest.exists() {
                    Some((
                        job.photo_id,
                        job.source_path,
                        job.preview_dest.to_string_lossy().to_string(),
                        ts,
                    ))
                } else {
                    None
                }
            })
            .collect();

        for (pid, src, preview, ts) in ready {
            preview_paths.insert(pid.clone(), preview.clone());
            source_to_preview.insert(src, preview);
            if let Some(exact) = ts {
                photo_capture_timestamps.insert(pid.clone(), Some(exact));
                if let Some(target) = scan_targets.iter_mut().find(|t| t.photo_id.as_str() == pid) {
                    target.capture_timestamp_epoch_sec = exact;
                    target.exact_capture_timestamp_epoch_sec = Some(exact);
                }
            }
        }
    }

    let directory_summaries = dir_counts
        .into_iter()
        .map(|(name, count)| {
            let id = format!("dir-{}", name.to_ascii_lowercase().replace(['/', ' '], "-"));
            (id, name, count)
        })
        .collect();

    let sidecars_dir = resolve_sidecars_dir(root, session_mode, managed_single_sidecars_dir);
    let mut recipes = load_existing_recipes_once(&sidecars_dir, &scan_targets);
    // If opening a full folder after previously grading an individual photo in Single Photo Mode,
    // inherit those managed sidecars seamlessly when no local `.kroma_sidecars` override exists yet.
    if session_mode == ShootSessionMode::Folder {
        let managed_bucket = managed_single_sidecars_bucket(root, managed_single_sidecars_dir);
        if managed_bucket.is_dir() {
            for (pid, recipe) in load_existing_recipes_once(&managed_bucket, &scan_targets) {
                recipes.entry(pid).or_insert(recipe);
            }
        }
    }

    let discovered_pairs: Vec<(PhotoId, String)> = scan_targets
        .iter()
        .map(|t| (t.photo_id.clone(), t.source_path.clone()))
        .collect();
    let restored_plan = if session_mode == ShootSessionMode::Folder {
        JsonOrganizationPlanStore::new(&sidecars_dir)
            .load_plan()
            .ok()
            .flatten()
            .and_then(|saved| saved.reconcile_with_disk(&discovered_pairs))
    } else {
        None
    };

    let (is_clustered, plan) = if let Some(reconciled) = restored_plan {
        let is_unclustered_placeholder = reconciled.clusters.len() == 1
            && reconciled.clusters[0].id.as_str() == "all_photos"
            && reconciled.group_photos.is_empty()
            && reconciled.no_face_photos.is_empty()
            && reconciled.ambiguous_review_queue.is_empty();
        (!is_unclustered_placeholder, reconciled)
    } else {
        let neutral_box = BoundingBox {
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
        };
        let p = Point2D { x: 0.5, y: 0.5 };
        let neutral_landmarks = YuNetLandmarks {
            right_eye: p,
            left_eye: p,
            nose_tip: p,
            right_mouth_corner: p,
            left_mouth_corner: p,
        };
        let mut raw_vec = [0.0_f32; SFACE_EMBEDDING_DIM];
        raw_vec[0] = 1.0;
        let placeholder_emb = FaceEmbedding::from_raw(raw_vec).map_err(|e| e.to_string())?;

        let mut targets_iter = scan_targets.iter();
        let mut clusters = Vec::new();
        if let Some(first) = targets_iter.next() {
            let mut all_cluster = SubjectCluster::new_from_face(
                ClusterId::new("all_photos").map_err(|e| e.to_string())?,
                "All Photos",
                None,
                DetectedFace {
                    photo_id: first.photo_id.clone(),
                    bounding_box: neutral_box,
                    landmarks: neutral_landmarks,
                    detection_confidence: 1.0,
                    embedding: placeholder_emb,
                },
                first.source_path.clone(),
                None,
            )
            .map_err(|e| e.to_string())?;

            for target in targets_iter {
                all_cluster.entries.push(ClusterPhotoEntry {
                    photo_id: target.photo_id.clone(),
                    source_path: target.source_path.clone(),
                    cosine_similarity: 1.0,
                    confidence: MatchConfidence::Verified,
                    face_bounding_box: neutral_box,
                    landmarks: neutral_landmarks,
                    is_slate_photo: false,
                });
            }
            clusters.push(all_cluster);
        }

        (
            false,
            ShootOrganizationPlan {
                clusters,
                ambiguous_review_queue: Vec::new(),
                group_photos: Vec::new(),
                no_face_photos: Vec::new(),
                virtual_folders: Vec::new(),
                photo_folders: BTreeMap::new(),
            },
        )
    };

    let total = scan_targets.len();
    let ctx = ActiveShootContext {
        session_mode,
        folder_path,
        project_name,
        sidecars_dir,
        is_clustered,
        total_photos: total,
        scan_elapsed_ms: started.elapsed().as_millis() as u64,
        scan_error_count: 0,
        plan,
        history: OrganizationHistory::default(),
        photo_sources,
        photo_file_sizes,
        preview_paths,
        rendered_preview_paths: HashMap::new(),
        source_to_preview,
        photo_directories,
        photo_capture_timestamps,
        recipes,
        scan_targets,
        cached_face_scans: None,
        detected_face_guides: HashMap::new(),
        detected_face_overlays: HashMap::new(),
        detected_face_parse_maps: HashMap::new(),
        directory_summaries,
    };

    Ok((ctx, background_preview_jobs))
}

fn spawn_background_preview_worker(
    jobs: Vec<BackgroundPreviewJob>,
    active_shoot: Arc<Mutex<Option<ActiveShootContext>>>,
    load_generation: Arc<AtomicU64>,
    expected_gen: u64,
    app: AppHandle,
) {
    let total = jobs.len();
    std::thread::spawn(move || {
        let mut completed = 0usize;
        for chunk in jobs.chunks(BACKGROUND_PREVIEW_BATCH_SIZE) {
            if load_generation.load(Ordering::SeqCst) != expected_gen {
                return;
            }

            let ready: Vec<(String, String, String, Option<i64>)> = chunk
                .par_iter()
                .filter_map(|job| {
                    if load_generation.load(Ordering::Relaxed) != expected_gen {
                        return None;
                    }
                    let ts = if job.needs_exif {
                        extract_exif_capture_timestamp_epoch_sec(&job.source_path)
                    } else {
                        None
                    };
                    let _ = generate_cached_preview(&job.source_path, &job.preview_dest);
                    if job.preview_dest.exists() {
                        Some((
                            job.photo_id.clone(),
                            job.source_path.clone(),
                            job.preview_dest.to_string_lossy().to_string(),
                            ts,
                        ))
                    } else {
                        None
                    }
                })
                .collect();

            if load_generation.load(Ordering::SeqCst) != expected_gen {
                return;
            }

            completed += chunk.len();
            let mut patches = Vec::with_capacity(ready.len());

            if let Ok(mut guard) = active_shoot.lock()
                && let Some(ctx) = guard.as_mut()
            {
                for (pid, src, preview, ts) in ready {
                    ctx.preview_paths.insert(pid.clone(), preview.clone());
                    ctx.source_to_preview.insert(src, preview.clone());
                    if let Some(exact) = ts {
                        ctx.photo_capture_timestamps.insert(pid.clone(), Some(exact));
                        if let Some(target) =
                            ctx.scan_targets.iter_mut().find(|t| t.photo_id.as_str() == pid)
                        {
                            target.capture_timestamp_epoch_sec = exact;
                            target.exact_capture_timestamp_epoch_sec = Some(exact);
                        }
                    }
                    patches.push(PreviewPatchDto {
                        photo_id: pid,
                        preview_path: Some(preview),
                        rendered_preview_path: None,
                        clear_rendered_preview: false,
                        captured_at_epoch_sec: ts,
                        captured_at_label: format_capture_time_label(ts),
                    });
                }
            }

            if !patches.is_empty() {
                let _ = app.emit(
                    PREVIEWS_READY_EVENT,
                    PreviewsReadyBatchDto {
                        updates: patches,
                        completed,
                        total,
                    },
                );
            }
        }
    });
}

/// Loads ONLY `.kroma.json` sidecar files that actually exist in `sidecars_dir` once on folder open,
/// eliminating 2,000+ redundant `fs::stat` calls on every UI mutation.
/// Maps each sanitized `.kroma.json` filename back to its original `PhotoId` (which may contain
/// spaces, commas, or periods when photos are inside nested subfolders like `2 PINNING 01/AJIAS, ...`).
fn load_existing_recipes_once(
    sidecars_dir: &Path,
    scan_targets: &[PhotoScanTarget],
) -> HashMap<String, DevelopRecipe> {
    let mut map = HashMap::new();
    let Ok(entries) = fs::read_dir(sidecars_dir) else {
        return map;
    };
    let mut existing_safe_stems = HashSet::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if let Some(stem) = name.strip_suffix(".kroma.json") {
            existing_safe_stems.insert(stem.to_string());
        }
    }
    if existing_safe_stems.is_empty() {
        return map;
    }

    let store = JsonSidecarRecipeStore::new(sidecars_dir);
    for target in scan_targets {
        let pid_str = target.photo_id.as_str();
        let safe_id = JsonSidecarRecipeStore::sanitize_photo_id(pid_str);
        if existing_safe_stems.contains(&safe_id)
            && let Ok(Some(recipe)) = store.load_recipe(&target.photo_id)
        {
            map.insert(pid_str.to_string(), recipe);
        }
    }
    map
}

fn collect_supported_photos(dir: &Path, depth: usize, max_depth: usize, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            if is_supported_photo_path(&path) {
                out.push(path);
            }
        } else if path.is_dir() && depth < max_depth {
            let skip = path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|name| name.starts_with('.') || name.starts_with('_'));
            if !skip {
                collect_supported_photos(&path, depth + 1, max_depth, out);
            }
        }
    }
}

fn generate_cached_preview(source_path: &str, preview_dest: &Path) -> Result<(), String> {
    let rgb = decode_fast_preview_max_dim(source_path, PREVIEW_MAX_DIM).map_err(|e| e.to_string())?;
    let file = File::create(preview_dest).map_err(|e| e.to_string())?;
    let writer = BufWriter::new(file);
    let mut encoder = JpegEncoder::new_with_quality(writer, 78);
    encoder.encode_image(&rgb).map_err(|e| e.to_string())
}

fn format_capture_time_label(epoch_sec: Option<i64>) -> Option<String> {
    let ts = epoch_sec?;
    let sod = ts.rem_euclid(86_400);
    let h = sod / 3_600;
    let m = (sod % 3_600) / 60;
    let s = sod % 60;
    Some(format!("{h:02}:{m:02}:{s:02}"))
}

fn build_session_dto(ctx: &ActiveShootContext) -> ShootSessionDto {
    let default_recipe = DevelopRecipe::default();
    let default_dir_id = ctx
        .directory_summaries
        .first()
        .map(|(id, _, _)| id.clone())
        .unwrap_or_else(|| "dir-root".to_string());

    let mut clusters: Vec<ClusterDto> = ctx
        .plan
        .clusters
        .iter()
        .map(|cluster| {
            let rep_preview = ctx
                .preview_paths
                .get(cluster.representative_photo_id.as_str())
                .cloned()
                .unwrap_or_else(|| cluster.representative_source_path.clone());

            let photos = cluster
                .entries
                .iter()
                .map(|entry| {
                    let pid_str = entry.photo_id.as_str();
                    let preview_path = ctx
                        .preview_paths
                        .get(pid_str)
                        .cloned()
                        .unwrap_or_else(|| entry.source_path.clone());
                    let file_name = Path::new(&entry.source_path)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or(pid_str)
                        .to_string();
                    let recipe = ctx.recipes.get(pid_str).unwrap_or(&default_recipe);

                    let overlay = if ctx.is_clustered && entry.face_bounding_box.area() > 0.0 {
                        Some(build_face_overlay(entry.face_bounding_box, entry.landmarks))
                    } else {
                        ctx.detected_face_overlays.get(pid_str).cloned()
                    };

                    let captured_at_epoch_sec = ctx
                        .photo_capture_timestamps
                        .get(pid_str)
                        .copied()
                        .flatten();

                    PhotoDto {
                        id: pid_str.to_string(),
                        file_name,
                        directory_id: ctx
                            .photo_directories
                            .get(pid_str)
                            .cloned()
                            .unwrap_or_else(|| default_dir_id.clone()),
                        virtual_folder_id: ctx
                            .plan
                            .photo_folders
                            .get(&entry.photo_id)
                            .map(|f| f.as_str().to_string()),
                        source_path: entry.source_path.clone(),
                        preview_path,
                        rendered_preview_path: ctx.rendered_preview_paths.get(pid_str).cloned(),
                        file_size_bytes: ctx.photo_file_sizes.get(pid_str).copied().unwrap_or(0),
                        is_slate_anchor: entry.is_slate_photo,
                        cluster_id: Some(cluster.id.as_str().to_string()),
                        sface_cosine_similarity: entry.cosine_similarity,
                        confidence: confidence_label(entry.confidence),
                        captured_at_epoch_sec,
                        captured_at_label: format_capture_time_label(captured_at_epoch_sec),
                        face_overlay: overlay,
                        develop_recipe: DevelopRecipeDto::from(recipe),
                    }
                })
                .collect();

            ClusterDto {
                id: cluster.id.as_str().to_string(),
                display_name: cluster.display_name.clone(),
                subject_code: cluster.subject_code.clone(),
                has_slate: cluster.has_slate(),
                representative_photo_id: cluster.representative_photo_id.as_str().to_string(),
                representative_preview_path: rep_preview,
                photos,
            }
        })
        .collect();

    if !ctx.plan.group_photos.is_empty() {
        let first = &ctx.plan.group_photos[0];
        let rep_preview = ctx
            .preview_paths
            .get(first.photo_id.as_str())
            .cloned()
            .unwrap_or_else(|| first.source_path.clone());
        let photos = ctx
            .plan
            .group_photos
            .iter()
            .map(|g| {
                let pid = g.photo_id.as_str();
                let recipe = ctx.recipes.get(pid).unwrap_or(&default_recipe);
                let captured_at_epoch_sec = ctx
                    .photo_capture_timestamps
                    .get(pid)
                    .copied()
                    .flatten();
                PhotoDto {
                    id: pid.to_string(),
                    file_name: Path::new(&g.source_path)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or(pid)
                        .to_string(),
                    directory_id: ctx
                        .photo_directories
                        .get(pid)
                        .cloned()
                        .unwrap_or_else(|| default_dir_id.clone()),
                    virtual_folder_id: ctx
                        .plan
                        .photo_folders
                        .get(&g.photo_id)
                        .map(|f| f.as_str().to_string()),
                    source_path: g.source_path.clone(),
                    preview_path: ctx
                        .preview_paths
                        .get(pid)
                        .cloned()
                        .unwrap_or_else(|| g.source_path.clone()),
                    rendered_preview_path: ctx.rendered_preview_paths.get(pid).cloned(),
                    file_size_bytes: ctx.photo_file_sizes.get(pid).copied().unwrap_or(0),
                    is_slate_anchor: false,
                    cluster_id: Some("_group_photos".to_string()),
                    sface_cosine_similarity: 1.0,
                    confidence: "verified",
                    captured_at_epoch_sec,
                    captured_at_label: format_capture_time_label(captured_at_epoch_sec),
                    face_overlay: None,
                    develop_recipe: DevelopRecipeDto::from(recipe),
                }
            })
            .collect();
        clusters.push(ClusterDto {
            id: "_group_photos".to_string(),
            display_name: "Group Photos".to_string(),
            subject_code: None,
            has_slate: false,
            representative_photo_id: first.photo_id.as_str().to_string(),
            representative_preview_path: rep_preview,
            photos,
        });
    }

    if !ctx.plan.no_face_photos.is_empty() {
        let first = &ctx.plan.no_face_photos[0];
        let rep_preview = ctx
            .preview_paths
            .get(first.photo_id.as_str())
            .cloned()
            .unwrap_or_else(|| first.source_path.clone());
        let photos = ctx
            .plan
            .no_face_photos
            .iter()
            .map(|nf| {
                let pid = nf.photo_id.as_str();
                let recipe = ctx.recipes.get(pid).unwrap_or(&default_recipe);
                let captured_at_epoch_sec = ctx
                    .photo_capture_timestamps
                    .get(pid)
                    .copied()
                    .flatten();
                PhotoDto {
                    id: pid.to_string(),
                    file_name: Path::new(&nf.source_path)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or(pid)
                        .to_string(),
                    directory_id: ctx
                        .photo_directories
                        .get(pid)
                        .cloned()
                        .unwrap_or_else(|| default_dir_id.clone()),
                    virtual_folder_id: ctx
                        .plan
                        .photo_folders
                        .get(&nf.photo_id)
                        .map(|f| f.as_str().to_string()),
                    source_path: nf.source_path.clone(),
                    preview_path: ctx
                        .preview_paths
                        .get(pid)
                        .cloned()
                        .unwrap_or_else(|| nf.source_path.clone()),
                    rendered_preview_path: ctx.rendered_preview_paths.get(pid).cloned(),
                    file_size_bytes: ctx.photo_file_sizes.get(pid).copied().unwrap_or(0),
                    is_slate_anchor: false,
                    cluster_id: Some("_no_face_photos".to_string()),
                    sface_cosine_similarity: 1.0,
                    confidence: "verified",
                    captured_at_epoch_sec,
                    captured_at_label: format_capture_time_label(captured_at_epoch_sec),
                    face_overlay: None,
                    develop_recipe: DevelopRecipeDto::from(recipe),
                }
            })
            .collect();
        clusters.push(ClusterDto {
            id: "_no_face_photos".to_string(),
            display_name: "No Face / Scenic".to_string(),
            subject_code: None,
            has_slate: false,
            representative_photo_id: first.photo_id.as_str().to_string(),
            representative_preview_path: rep_preview,
            photos,
        });
    }

    let ambiguous_review = ctx
        .plan
        .ambiguous_review_queue
        .iter()
        .map(|a| {
            let file_name = Path::new(&a.source_path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(a.photo_id.as_str())
                .to_string();
            AmbiguousCandidateDto {
                photo_id: a.photo_id.as_str().to_string(),
                file_name,
                source_path: a.source_path.clone(),
                preview_path: ctx
                    .preview_paths
                    .get(a.photo_id.as_str())
                    .cloned()
                    .unwrap_or_else(|| a.source_path.clone()),
                candidate_cluster_id: a.candidate_cluster_id.as_str().to_string(),
                cosine_similarity: a.cosine_similarity,
                face_overlay: build_face_overlay(a.face_bounding_box, a.landmarks),
            }
        })
        .collect();

    let directories = ctx
        .directory_summaries
        .iter()
        .map(|(id, name, count)| DirectorySummaryDto {
            id: id.clone(),
            name: name.clone(),
            photo_count: *count,
        })
        .collect();

    let virtual_folders = ctx
        .plan
        .virtual_folders
        .iter()
        .map(|f| VirtualFolderDto {
            id: f.id.as_str().to_string(),
            name: f.name.clone(),
            parent_id: f.parent_id.as_ref().map(|p| p.as_str().to_string()),
        })
        .collect();

    ShootSessionDto {
        session_mode: ctx.session_mode.as_str().to_string(),
        folder_path: ctx.folder_path.clone(),
        project_name: ctx.project_name.clone(),
        is_clustered: ctx.is_clustered,
        total_photos: ctx.total_photos,
        scan_elapsed_ms: ctx.scan_elapsed_ms,
        scan_error_count: ctx.scan_error_count,
        directories,
        virtual_folders,
        clusters,
        ambiguous_review,
        group_photo_count: ctx.plan.group_photos.len(),
        no_face_photo_count: ctx.plan.no_face_photos.len(),
        can_undo: ctx.history.can_undo(),
        can_redo: ctx.history.can_redo(),
        undo_label: ctx.history.undo_label().map(str::to_string),
        redo_label: ctx.history.redo_label().map(str::to_string),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_evenly_spaced_indices_covers_shoot_span_without_duplicates() {
        assert_eq!(pick_evenly_spaced_indices(0, 5), Vec::<usize>::new());
        assert_eq!(pick_evenly_spaced_indices(3, 5), vec![0, 1, 2]);
        assert_eq!(pick_evenly_spaced_indices(5, 5), vec![0, 1, 2, 3, 4]);
        assert_eq!(pick_evenly_spaced_indices(2090, 5), vec![0, 522, 1044, 1566, 2089]);
    }

    #[test]
    fn parse_selected_photo_ids_treats_none_and_empty_as_full_export() {
        assert!(parse_selected_photo_ids(None).is_none());
        assert!(parse_selected_photo_ids(Some(&[])).is_none());
        let ids = vec!["photo-a".to_string(), "photo-b".to_string()];
        let parsed = parse_selected_photo_ids(Some(&ids)).expect("non-empty selection");
        assert_eq!(parsed.len(), 2);
        assert!(parsed.contains(&PhotoId::new("photo-a").unwrap()));
        assert!(parsed.contains(&PhotoId::new("photo-b").unwrap()));
    }

    #[test]
    fn raw_ocr_fallback_only_retries_fuzzy_or_unparsed_frames_with_text() {
        // Same path (PNG/JPEG): never retries
        assert!(!should_retry_ocr_with_source("/a/img.png", "/a/img.png", 3, Some(0.50)));
        // Portrait without text: never opens RAW
        assert!(!should_retry_ocr_with_source("/cache/p.jpg", "/raw/p.arw", 0, None));
        // Confident preview read: stays fast, never opens RAW
        assert!(!should_retry_ocr_with_source("/cache/p.jpg", "/raw/p.arw", 2, Some(0.92)));
        // Fuzzy or unparsed whiteboard text on a RAW preview: escalates to 1600px source
        assert!(should_retry_ocr_with_source("/cache/p.jpg", "/raw/p.arw", 2, Some(0.74)));
        assert!(should_retry_ocr_with_source("/cache/p.jpg", "/raw/p.arw", 1, None));

        assert!(ocr_candidate_is_better(2, Some(0.91), 2, Some(0.74)));
        assert!(!ocr_candidate_is_better(2, Some(0.65), 2, Some(0.74)));
        assert!(ocr_candidate_is_better(1, Some(0.70), 3, None));
        assert!(ocr_candidate_is_better(2, None, 0, None));
    }

    #[test]
    fn storage_cache_summary_and_selective_clear_preserve_unrelated_files() {
        let dir = std::env::temp_dir().join(format!(
            "kroma_cache_test_{}",
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("v3_DSC0001_0001_preview.jpg"), vec![1u8; 120]).unwrap();
        fs::write(dir.join("v3_DSC0002_0002_preview.jpg"), vec![2u8; 80]).unwrap();
        fs::write(dir.join("face_cache_v1_abcd.json"), vec![3u8; 64]).unwrap();
        fs::write(dir.join("keep_me.txt"), b"hello").unwrap();

        let summary = summarize_cache_dir(&dir, true);
        assert!(summary.is_custom_cache_dir);
        assert_eq!(summary.preview_file_count, 2);
        assert_eq!(summary.preview_cache_bytes, 200);
        assert_eq!(summary.face_cache_file_count, 1);
        assert_eq!(summary.face_cache_bytes, 64);

        let cleared_previews = clear_cache_dir_entries(&dir, "previews");
        assert_eq!(cleared_previews.removed_files, 2);
        assert_eq!(cleared_previews.freed_bytes, 200);
        assert!(dir.join("face_cache_v1_abcd.json").exists());
        assert!(dir.join("keep_me.txt").exists());

        let cleared_faces = clear_cache_dir_entries(&dir, "face_cache");
        assert_eq!(cleared_faces.removed_files, 1);
        assert_eq!(cleared_faces.freed_bytes, 64);
        assert!(dir.join("keep_me.txt").exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn thumbnail_32bit_mode_selection_and_hash_invalidation_behave_accurately() {
        let default_recipe = DevelopRecipe::default();
        let mut edited_recipe = DevelopRecipe::default();
        edited_recipe.basic_tone.exposure_ev = 0.75;

        assert!(!default_recipe.is_modified());
        assert!(edited_recipe.is_modified());

        assert!(!should_render_32bit_thumbnail(
            ThumbnailPreviewRenderMode::Fast,
            "DSC0001.ARW",
            &edited_recipe
        ));
        assert!(!should_render_32bit_thumbnail(
            ThumbnailPreviewRenderMode::Edited32Bit,
            "DSC0001.ARW",
            &default_recipe
        ));
        assert!(should_render_32bit_thumbnail(
            ThumbnailPreviewRenderMode::Edited32Bit,
            "DSC0001.ARW",
            &edited_recipe
        ));
        assert!(should_render_32bit_thumbnail(
            ThumbnailPreviewRenderMode::All32Bit,
            "DSC0001.ARW",
            &default_recipe
        ));
        assert!(!should_render_32bit_thumbnail(
            ThumbnailPreviewRenderMode::All32Bit,
            "IMG_0001.JPG",
            &default_recipe
        ));
        // 8-bit JPG/PNG with only basic tone/WB sliders uses the 0ms GPU/SVG LUT instead of queuing CPU re-decodes
        assert!(!should_render_32bit_thumbnail(
            ThumbnailPreviewRenderMode::Edited32Bit,
            "IMG_0001.JPG",
            &edited_recipe
        ));
        assert!(!should_render_32bit_thumbnail(
            ThumbnailPreviewRenderMode::All32Bit,
            "IMG_0001.JPG",
            &edited_recipe
        ));
        // 8-bit JPG/PNG with non-SVG effects (HSL, masks, AI backdrop, crop) renders a baked thumbnail
        let mut hsl_recipe = edited_recipe.clone();
        hsl_recipe.hsl_mixer.red.saturation = 25;
        assert!(should_render_32bit_thumbnail(
            ThumbnailPreviewRenderMode::Edited32Bit,
            "IMG_0001.JPG",
            &hsl_recipe
        ));

        let h_default = compute_32bit_thumbnail_hash("DSC0001.ARW", 24_000_000, &default_recipe);
        let h_edited = compute_32bit_thumbnail_hash("DSC0001.ARW", 24_000_000, &edited_recipe);
        assert_ne!(h_default, h_edited);

        // Portrait retouching triggers 32-bit previews on JPGs too and invalidates cache hash when
        // YuNet face guides or BiSeNet FaceParseMap become available
        let mut portrait_recipe = DevelopRecipe::default();
        portrait_recipe.retouch.portrait.enabled = true;
        portrait_recipe.retouch.portrait.skin_smoothing = 45;
        assert!(should_render_32bit_thumbnail(
            ThumbnailPreviewRenderMode::Edited32Bit,
            "IMG_0001.JPG",
            &portrait_recipe
        ));
        let h_no_face = compute_32bit_thumbnail_hash_with_retouch(
            "IMG_0001.JPG",
            4_000_000,
            &portrait_recipe,
            &[],
            None,
        );
        let guide = PortraitFaceGuide::from_normalized(
            0.25,
            0.2,
            0.5,
            0.55,
            [0.4, 0.4],
            [0.6, 0.4],
            [0.5, 0.5],
            [0.42, 0.62],
            [0.58, 0.62],
        );
        let h_with_guide = compute_32bit_thumbnail_hash_with_retouch(
            "IMG_0001.JPG",
            4_000_000,
            &portrait_recipe,
            &[guide],
            None,
        );
        let parse_map = FaceParseMap {
            width: 8,
            height: 8,
            rgba: vec![200u8; 8 * 8 * 4],
        };
        let h_with_parse = compute_32bit_thumbnail_hash_with_retouch(
            "IMG_0001.JPG",
            4_000_000,
            &portrait_recipe,
            &[guide],
            Some(&parse_map),
        );
        assert_ne!(h_no_face, h_with_guide);
        assert_ne!(h_with_guide, h_with_parse);
    }

    #[test]
    fn load_existing_recipes_once_restores_sidecars_for_subfolders_with_spaces_and_commas() {
        let dir = std::env::temp_dir().join(format!(
            "kroma_sidecar_restore_test_{}",
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let store = JsonSidecarRecipeStore::new(&dir);

        // Subfolder PhotoId containing spaces, commas, and periods (e.g. "2 PINNING 01_AJIAS, RICHMOND D._DSC06182")
        let subfolder_pid =
            PhotoId::new("2 PINNING 01_AJIAS, RICHMOND D._DSC06182").expect("valid photo id");
        let mut recipe = DevelopRecipe::default();
        recipe.basic_tone.exposure_ev = 1.25;
        recipe.white_balance.temperature_kelvin = 6100;
        store.save_recipe(&subfolder_pid, &recipe).expect("saved sidecar");

        let targets = vec![PhotoScanTarget {
            photo_id: subfolder_pid.clone(),
            source_path: "D:/Shoot/2 PINNING 01/AJIAS, RICHMOND D./DSC06182.ARW".to_string(),
            capture_timestamp_epoch_sec: 100,
            exact_capture_timestamp_epoch_sec: Some(100),
        }];

        let restored = load_existing_recipes_once(&dir, &targets);
        let loaded = restored
            .get(subfolder_pid.as_str())
            .expect("must map sanitized .kroma.json filename back to original subfolder PhotoId");
        assert_eq!(loaded.basic_tone.exposure_ev, 1.25);
        assert_eq!(loaded.white_balance.temperature_kelvin, 6100);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn filter_supported_photo_files_and_common_parent_handles_single_and_multi_files() {
        let base = std::env::temp_dir().join(format!(
            "kroma_open_files_test_{}",
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let sub_a = base.join("day1");
        let sub_b = base.join("day2");
        fs::create_dir_all(&sub_a).unwrap();
        fs::create_dir_all(&sub_b).unwrap();

        let p1 = sub_a.join("IMG_001.jpg");
        let p2 = sub_b.join("IMG_002.CR2");
        let ignored = sub_a.join("notes.txt");
        fs::write(&p1, b"fake").unwrap();
        fs::write(&p2, b"fake").unwrap();
        fs::write(&ignored, b"text").unwrap();

        let single = filter_supported_photo_file_paths(vec![
            p1.to_string_lossy().to_string(),
            p1.to_string_lossy().to_string(),
            ignored.to_string_lossy().to_string(),
        ]);
        assert_eq!(single.len(), 1);
        assert_eq!(resolve_common_parent_dir(&single), Some(sub_a.clone()));

        let multi = filter_supported_photo_file_paths(vec![
            p1.to_string_lossy().to_string(),
            p2.to_string_lossy().to_string(),
        ]);
        assert_eq!(multi.len(), 2);
        assert_eq!(resolve_common_parent_dir(&multi), Some(base.clone()));
        assert_eq!(ShootSessionMode::SinglePhoto.as_str(), "single");
        assert_eq!(ShootSessionMode::Files.as_str(), "files");
        assert_eq!(ShootSessionMode::Folder.as_str(), "folder");

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn resolve_sidecars_dir_keeps_downloads_clean_until_local_shoot_sidecars_exist() {
        let base = std::env::temp_dir().join(format!(
            "kroma_sidecars_dir_test_{}",
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let downloads_dir = base.join("Downloads");
        let managed_dir = base.join("managed_single_sidecars");
        fs::create_dir_all(&downloads_dir).unwrap();
        fs::create_dir_all(&managed_dir).unwrap();

        // Single photo in Downloads with no existing .kroma_sidecars -> managed app-data bucket
        let single_dir =
            resolve_sidecars_dir(&downloads_dir, ShootSessionMode::SinglePhoto, &managed_dir);
        assert!(single_dir.starts_with(&managed_dir));
        assert!(!downloads_dir.join(".kroma_sidecars").exists());

        // Full folder shoot always uses local .kroma_sidecars
        let folder_dir =
            resolve_sidecars_dir(&downloads_dir, ShootSessionMode::Folder, &managed_dir);
        assert_eq!(folder_dir, downloads_dir.join(".kroma_sidecars"));

        // If .kroma_sidecars already exists in the parent shoot folder, SinglePhoto shares it
        fs::create_dir_all(&folder_dir).unwrap();
        let shared_single =
            resolve_sidecars_dir(&downloads_dir, ShootSessionMode::SinglePhoto, &managed_dir);
        assert_eq!(shared_single, folder_dir);

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn preview_proxy_policy_handles_raw_and_large_web_images() {
        let cache_dir = Path::new("cache_root");
        let raw_path = Path::new("D:/Shoot/DSC0001.ARW");
        let tiff_path = Path::new("D:/Shoot/scan.tiff");
        let small_jpg = Path::new("D:/Shoot/web_small.jpg");
        let camera_jpg = Path::new("D:/Shoot/IMG_4821.JPG");
        let large_png = Path::new("D:/Shoot/render.png");

        // RAW and TIFF always get cached previews regardless of size
        assert!(should_generate_preview_proxy(raw_path, 10_000));
        assert!(should_generate_preview_proxy(tiff_path, 10_000));

        // Small web-native files (<300 KB) skip proxy encoding; large camera JPEGs/PNGs get proxied
        assert!(!should_generate_preview_proxy(small_jpg, 120_000));
        assert!(should_generate_preview_proxy(
            camera_jpg,
            WEB_NATIVE_PREVIEW_PROXY_MIN_BYTES
        ));
        assert!(should_generate_preview_proxy(large_png, 8_500_000));

        // RAW cache path keeps its stable v3_ key; web-native cache path uses v3_web_ and
        // invalidates automatically if file size or mtime changes while staying recognized by cache cleanup
        let raw_cache_a = build_preview_cache_path(
            cache_dir,
            raw_path,
            "DSC0001",
            "D:/Shoot/DSC0001.ARW",
            25_000_000,
            1000,
        );
        let raw_cache_b = build_preview_cache_path(
            cache_dir,
            raw_path,
            "DSC0001",
            "D:/Shoot/DSC0001.ARW",
            25_000_000,
            2000,
        );
        assert_eq!(raw_cache_a, raw_cache_b);
        assert!(is_preview_cache_file(
            raw_cache_a.file_name().unwrap().to_str().unwrap()
        ));

        let jpg_cache_a = build_preview_cache_path(
            cache_dir,
            camera_jpg,
            "IMG_4821",
            "D:/Shoot/IMG_4821.JPG",
            6_400_000,
            1000,
        );
        let jpg_cache_b = build_preview_cache_path(
            cache_dir,
            camera_jpg,
            "IMG_4821",
            "D:/Shoot/IMG_4821.JPG",
            6_400_000,
            2000,
        );
        let jpg_cache_c = build_preview_cache_path(
            cache_dir,
            camera_jpg,
            "IMG_4821",
            "D:/Shoot/IMG_4821.JPG",
            6_500_000,
            1000,
        );
        assert_ne!(jpg_cache_a, jpg_cache_b);
        assert_ne!(jpg_cache_a, jpg_cache_c);
        assert!(is_preview_cache_file(
            jpg_cache_a.file_name().unwrap().to_str().unwrap()
        ));
    }
}
