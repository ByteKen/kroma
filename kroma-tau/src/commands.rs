use crate::dto::{
    build_face_overlay, confidence_label, AmbiguousCandidateDto, BatchSyncResultDto,
    ClearCacheResultDto, ClusterDto, ClusterSlateOcrResultDto, DevelopRecipeDto,
    DevelopSyncMaskDto, DirectorySummaryDto, EstimateExportSizeRequestDto,
    EstimateExportSizeResponseDto, ExportShootRequestDto, ExportShootResultDto, GeneratedAiMaskDto,
    HistogramDto, ModelDownloadOutcomeDto, ModelInspectionDto, ModelManifestStatusDto,
    OcrTextLineDto, PhotoDto, PhotoOcrResultDto,
    PhotoRecipeSaveItemDto, PreviewPatchDto, PreviewsReadyBatchDto, RelativeDevelopDeltaDto,
    RuntimeStatusDto, ScanProgressDto, ShootSessionDto, SlateRenameApplyItemDto,
    StorageCacheSummaryDto, ThumbnailRenderProgressDto, VirtualFolderDto, WhiteBalanceDto,
};
use crate::face_cache;
use crate::model_manager::{
    download_pending_models, inspect_manifest_status, load_models_manifest,
    MODEL_DOWNLOAD_PROGRESS_EVENT,
};
use crate::state::{
    ActiveShootContext, AppState, CachedDevelopBuffer, CachedOcrEngine, OrganizationHistory,
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
use kroma_domain::develop::{DevelopRecipe, NormalizedRect};
use kroma_domain::masking::AiMaskKind;
use kroma_domain::identity::{
    parse_filename_capture_hints, BoundingBox, ClusterId, ClusterPhotoEntry, DetectedFace,
    FaceEmbedding, MatchConfidence, Point2D, ShootOrganizationPlan, SimilarityThresholds,
    SubjectCluster, VirtualFolderId, YuNetLandmarks, SFACE_EMBEDDING_DIM,
};
use kroma_domain::organization::OutputNamingTemplate;
use kroma_domain::photo::PhotoId;
use kroma_infra::develop::image_decoder::{
    decode_develop_buffer, decode_fast_preview_max_dim, decode_high_bit_depth_working_buffer,
    decode_kr16_downscaled_rgb32f, downscale_rgb32f_unclamped, encode_kr16_rgba16f,
    encode_kr32_rgba32f, extract_exif_capture_timestamp_epoch_sec, is_16bit_png_path,
    is_raw_photo_path, is_supported_photo_path, is_tiff_photo_path, is_web_native_image_path,
    supported_raw_extensions, DecodedDevelopBuffer,
};
use kroma_infra::develop::pixel_engine::RayonImageDevelopEngine;
use kroma_infra::develop::wgpu_engine::{
    is_wgpu_develop_available, wgpu_develop_adapter_name, wgpu_develop_backend_label,
};
use kroma_infra::organizer::local_file_organizer::LocalFileOrganizer;
use kroma_infra::storage::json_organization_plan_store::JsonOrganizationPlanStore;
use kroma_infra::storage::json_sidecar_store::JsonSidecarRecipeStore;
use kroma_infra::vision::model_inspector::{
    inspect_onnx_model as inspect_onnx_model_file, ModelCompatibility, ModelRole,
};
use kroma_infra::vision::onnx_runtime::{
    gpu_execution_provider_label, is_gpu_acceleration_enabled,
    is_gpu_execution_provider_available, set_gpu_acceleration_enabled,
};
use kroma_infra::vision::ppocr_v5::{PpOcrV5Config, PpOcrV5Engine};
use kroma_infra::vision::subject_segmenter::{generate_ai_mask_for_rgb, OnnxSubjectSegmenter};
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
use tauri::{AppHandle, Emitter, State};

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
    if !ctx.is_clustered {
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
pub fn reveal_main_window(window: tauri::WebviewWindow) {
    if !window.is_visible().unwrap_or(true) {
        let _ = window.show();
        let _ = window.set_focus();
    }
}

fn build_runtime_status_dto(state: &AppState) -> RuntimeStatusDto {
    let (yunet, recognizer) = state.resolve_model_paths();
    let (default_yunet, default_recognizer) = state.resolve_default_model_paths();
    let (ocr_det, ocr_rec, ocr_dict) = state.resolve_ocr_model_paths();
    let segmenter = state.resolve_segmenter_model_path();
    let default_segmenter = state.resolve_default_segmenter_model_path();
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
    let models_ready = yunet.is_some() && recognizer.is_some();
    let ocr_models_ready = ocr_det.is_some() && ocr_rec.is_some() && ocr_dict.is_some();
    let segmenter_models_ready = segmenter.is_some();
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
    state: State<'_, AppState>,
) -> Result<StorageCacheSummaryDto, String> {
    let next_path = cache_dir
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from);
    let resolved = state.set_custom_cache_dir(next_path)?;
    let is_custom = state.is_custom_cache_dir();
    state.thumbnail_render_generation.fetch_add(1, Ordering::SeqCst);
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
    state: State<'_, AppState>,
) -> Result<ClearCacheResultDto, String> {
    let normalized = target.trim().to_ascii_lowercase();
    if normalized != "previews" && normalized != "face_cache" && normalized != "all" {
        return Err(format!("unknown cache target '{target}'"));
    }

    if normalized == "previews" || normalized == "all" {
        state.thumbnail_render_generation.fetch_add(1, Ordering::SeqCst);
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

    // Backend defense-in-depth: block structurally incompatible ONNX files from activating
    let yunet_check = next_yunet.clone();
    let rec_check = next_recognizer.clone();
    let seg_check = next_segmenter.clone();
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
    if let Ok(mut seg_cache) = state.cached_segmenter_engine.lock() {
        *seg_cache = None;
    }
    if let Ok(mut yunet_cache) = state.cached_yunet_engine.lock() {
        *yunet_cache = None;
    }
    if let Ok(mut shoot_guard) = state.active_shoot.lock()
        && let Some(ctx) = shoot_guard.as_mut()
    {
        ctx.cached_face_scans = None;
    }

    Ok(build_runtime_status_dto(&state))
}

/// Fast Step 1: Opens folders of 2,000+ PNG/JPEG or camera RAW photos nearly instantaneously.
/// - PNG/JPEG/WebP render directly in WebView2 with zero upfront re-encoding.
/// - Camera RAWs/TIFFs render the first 36 uncached previews immediately and stream the rest in background.
#[tauri::command]
pub async fn load_shoot_folder(
    folder_path: String,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<ShootSessionDto, String> {
    state.clear_develop_buffer_cache();
    state.thumbnail_render_generation.fetch_add(1, Ordering::SeqCst);
    let preview_cache_dir = state.resolve_cache_dir();
    let active_shoot = state.active_shoot.clone();
    let load_generation = state.load_generation.clone();
    let my_gen = load_generation.fetch_add(1, Ordering::SeqCst) + 1;

    tauri::async_runtime::spawn_blocking(move || {
        let (ctx, pending_bg_jobs) =
            run_instant_folder_load(folder_path, &preview_cache_dir, &app)?;
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

/// Decodes the active Camera RAW / 16-bit TIFF / 16-bit PNG directly into a screen-sized
/// 16-bit half-float `KR16` (`RGBA16F` Linear Rec.2020 — lossless for 12/14-bit camera sensors,
/// 2× faster IPC & GPU upload, fed straight into 32-bit `rgba32float` WGSL compute shaders)
/// or 32-bit float `KR32` binary buffer for Studio Develop editing.
/// Checks in-memory LRU first, then bounded SSD cache (~10ms), falling back to full parallel RAW demosaic.
#[tauri::command]
pub async fn load_develop_working_buffer(
    photo_id: String,
    max_dim: Option<u32>,
    precision: Option<String>,
    state: State<'_, AppState>,
) -> Result<tauri::ipc::Response, String> {
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

    let use_f32 = matches!(precision.as_deref(), Some("f32"));
    let clamped_max_dim = max_dim.unwrap_or(1600).clamp(256, 2560);
    let cache_key = format!(
        "{source_path}@{clamped_max_dim}@{}",
        if use_f32 { "f32" } else { "f16" }
    );
    if let Some(cached_bytes) = state.get_cached_develop_buffer(&cache_key) {
        return Ok(tauri::ipc::Response::new((*cached_bytes).clone()));
    }

    let cache_dir = state.resolve_cache_dir();
    let disk_cache_path = if !use_f32 {
        Some(compute_working_buffer_cache_path(
            &cache_dir,
            &source_path,
            file_size_bytes,
            clamped_max_dim,
        ))
    } else {
        None
    };

    let source_for_worker = source_path.clone();
    let decoded_result = tauri::async_runtime::spawn_blocking(move || {
        // Fast SSD cache hit for compact KR16 working buffers (~10ms instead of 250-600ms RAW demosaic)
        if let Some(ref disk_path) = disk_cache_path
            && disk_path.is_file()
            && let Ok(cached_kr16) = fs::read(disk_path)
            && let Some(thumb_buffer) =
                decode_kr16_downscaled_rgb32f(&cached_kr16, THUMBNAIL_32BIT_MAX_DIM)
        {
            let hist_image = downscale_rgb32f_unclamped(&thumb_buffer.image, 384);
            let hist_buffer = DecodedDevelopBuffer {
                image: hist_image,
                is_linear: thumb_buffer.is_linear,
                is_wide_gamut: thumb_buffer.is_wide_gamut,
                bit_depth: thumb_buffer.bit_depth,
                is_raw_demosaic: thumb_buffer.is_raw_demosaic,
            };
            return Ok::<Option<(Vec<u8>, DecodedDevelopBuffer, DecodedDevelopBuffer)>, String>(
                Some((cached_kr16, hist_buffer, thumb_buffer)),
            );
        }

        let Some(decoded) =
            decode_high_bit_depth_working_buffer(&source_for_worker, clamped_max_dim)
                .map_err(|e| e.to_string())?
        else {
            return Ok(None);
        };

        let wire_bytes = if use_f32 {
            encode_kr32_rgba32f(&decoded)
        } else {
            encode_kr16_rgba16f(&decoded)
        };
        if let Some(ref disk_path) = disk_cache_path {
            persist_working_kr16_cache_bounded(&cache_dir, disk_path, &wire_bytes);
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
        Ok(Some((wire_bytes, hist_buffer, thumb_buffer)))
    })
    .await
    .map_err(|err| format!("high-bit-depth develop decode worker failed: {err}"))??;

    let Some((wire_bytes, hist_buffer, thumb_buffer)) = decoded_result else {
        return Ok(tauri::ipc::Response::new(Vec::new()));
    };

    let shared_bytes = Arc::new(wire_bytes);
    let response_bytes = (*shared_bytes).clone();
    state.insert_develop_buffer(CachedDevelopBuffer {
        cache_key,
        source_path,
        kr16_bytes: shared_bytes,
        hist_buffer: Arc::new(hist_buffer),
        thumb_buffer: Arc::new(thumb_buffer),
    });

    Ok(tauri::ipc::Response::new(response_bytes))
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

fn compute_32bit_thumbnail_hash(
    source_path: &str,
    file_size_bytes: u64,
    recipe: &DevelopRecipe,
) -> (u64, u64) {
    let recipe_json = serde_json::to_string(&DevelopRecipeDto::from(recipe)).unwrap_or_default();
    let mut h1 = DefaultHasher::new();
    "kroma-32bit-thumb-v1-a".hash(&mut h1);
    THUMBNAIL_32BIT_MAX_DIM.hash(&mut h1);
    source_path.hash(&mut h1);
    file_size_bytes.hash(&mut h1);
    recipe_json.hash(&mut h1);

    let mut h2 = DefaultHasher::new();
    "kroma-32bit-thumb-v1-b".hash(&mut h2);
    recipe_json.hash(&mut h2);
    file_size_bytes.hash(&mut h2);
    source_path.hash(&mut h2);

    (h1.finish(), h2.finish())
}

fn should_render_32bit_thumbnail(
    mode: ThumbnailPreviewRenderMode,
    source_path: &str,
    recipe: &DevelopRecipe,
) -> bool {
    match mode {
        ThumbnailPreviewRenderMode::Fast => false,
        ThumbnailPreviewRenderMode::Edited32Bit => recipe.is_modified(),
        ThumbnailPreviewRenderMode::All32Bit => {
            if recipe.is_modified() {
                return true;
            }
            let p = Path::new(source_path);
            is_raw_photo_path(p) || is_tiff_photo_path(p) || is_16bit_png_path(p)
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
        state
            .thumbnail_render_generation
            .fetch_add(1, Ordering::SeqCst);
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
        let _ = app.emit(
            THUMBNAIL_RENDER_PROGRESS_EVENT,
            ThumbnailRenderProgressDto {
                active: false,
                completed: 0,
                total: 0,
                mode: mode_str,
            },
        );
        return Ok(());
    }

    let is_full_pass = photo_ids.as_ref().is_none_or(|ids| ids.is_empty());
    let expected_thumb_gen = if is_full_pass {
        state
            .thumbnail_render_generation
            .fetch_add(1, Ordering::SeqCst)
            + 1
    } else {
        state.thumbnail_render_generation.load(Ordering::SeqCst)
    };
    let expected_load_gen = state.load_generation.load(Ordering::SeqCst);
    let preview_cache_dir = state.resolve_cache_dir();

    struct ThumbRenderJob {
        photo_id: String,
        source_path: String,
        recipe: DevelopRecipe,
        hash: (u64, u64),
        dest_path: PathBuf,
    }

    let (cached_hits, needs_render, cleared_ids) = {
        let mut guard = state
            .active_shoot
            .lock()
            .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
        let Some(ctx) = guard.as_mut() else {
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
            let file_size_bytes = ctx.photo_file_sizes.get(&pid).copied().unwrap_or(0);
            let hash = compute_32bit_thumbnail_hash(&source_path, file_size_bytes, &recipe);
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
            let _ = app.emit(
                THUMBNAIL_RENDER_PROGRESS_EVENT,
                ThumbnailRenderProgressDto {
                    active: false,
                    completed: 0,
                    total: 0,
                    mode: mode_str,
                },
            );
        }
        return Ok(());
    }

    let active_shoot = state.active_shoot.clone();
    let load_generation = state.load_generation.clone();
    let thumb_generation = state.thumbnail_render_generation.clone();
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

        if !is_still_valid() {
            return;
        }

        let _ = app.emit(
            THUMBNAIL_RENDER_PROGRESS_EVENT,
            ThumbnailRenderProgressDto {
                active: true,
                completed: 0,
                total,
                mode: mode_str.clone(),
            },
        );

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
                            engine.render_buffer_to_jpeg(
                                &hot_buf,
                                &job.recipe,
                                &job.dest_path,
                                THUMBNAIL_32BIT_JPEG_QUALITY,
                            )
                        } else {
                            decode_develop_buffer(&job.source_path, THUMBNAIL_32BIT_MAX_DIM)
                                .and_then(|decoded| {
                                    engine.render_buffer_to_jpeg(
                                        &decoded,
                                        &job.recipe,
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
                    let current_hash =
                        compute_32bit_thumbnail_hash(source_path, current_size, current_recipe);
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

            let _ = app.emit(
                THUMBNAIL_RENDER_PROGRESS_EVENT,
                ThumbnailRenderProgressDto {
                    active: completed < total,
                    completed,
                    total,
                    mode: mode_str.clone(),
                },
            );
        }

        if is_still_valid() {
            let _ = app.emit(
                THUMBNAIL_RENDER_PROGRESS_EVENT,
                ThumbnailRenderProgressDto {
                    active: false,
                    completed: total,
                    total,
                    mode: mode_str,
                },
            );
        }
    });

    Ok(())
}

/// Generates a non-destructive AI alpha matte (`subject`, `background`, `face_skin`, `sky`)
/// aligned to the photo's current orientation (quarter-turns + flips).
/// Uses an installed ONNX segmentation model (`RMBG`, `MODNet`, `BiRefNet`, `IS-Net`, `U2-Net`)
/// on DirectML GPU / CPU when configured, and falls back to Kroma's built-in YuNet + edge-matte AI.
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
    let cached_seg = state.cached_segmenter_engine.clone();
    let cached_yunet = state.cached_yunet_engine.clone();

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

        let result = generate_ai_mask_for_rgb(
            &rgb,
            parsed_kind,
            segmenter_arc.as_deref(),
            yunet_arc.as_deref(),
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

    tauri::async_runtime::spawn_blocking(move || {
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
        let histogram = if let Some(ref hist_buf) = cached_hist_buf {
            engine
                .compute_histogram_from_buffer(hist_buf, &saved.clamped_recipe)
                .map_err(|e| e.to_string())?
        } else {
            engine
                .compute_histogram(&preview_path, &saved.clamped_recipe)
                .map_err(|e| e.to_string())?
        };
        Ok(HistogramDto::from(&histogram))
    })
    .await
    .map_err(|err| format!("recipe save worker failed: {err}"))?
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

    tauri::async_runtime::spawn_blocking(move || {
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

        let histogram = match active_saved_recipe {
            Some(recipe) => {
                let engine = RayonImageDevelopEngine::new();
                if let Some(ref hist_buf) = cached_hist_buf {
                    engine
                        .compute_histogram_from_buffer(hist_buf, &recipe)
                        .ok()
                        .map(|h| HistogramDto::from(&h))
                } else if let Some(preview_path) = active_preview_path {
                    engine
                        .compute_histogram(&preview_path, &recipe)
                        .ok()
                        .map(|h| HistogramDto::from(&h))
                } else {
                    None
                }
            }
            None => None,
        };

        Ok(histogram)
    })
    .await
    .map_err(|err| format!("batch recipe save worker failed: {err}"))?
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
) -> Result<EstimateExportSizeResponseDto, String> {
    let (plan, photo_file_sizes, recipes) = {
        let guard = state
            .active_shoot
            .lock()
            .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
        let ctx = guard
            .as_ref()
            .ok_or_else(|| "no active shoot loaded".to_string())?;
        (
            ctx.plan.clone(),
            ctx.photo_file_sizes.clone(),
            ctx.recipes.clone(),
        )
    };

    tauri::async_runtime::spawn_blocking(move || {
        struct ExportCandidate {
            photo_id: String,
            source_path: String,
            file_size_bytes: u64,
            has_raw_companion: bool,
        }

        let lookup_size = |photo_id: &str, source_path: &str| -> u64 {
            photo_file_sizes
                .get(photo_id)
                .copied()
                .filter(|&bytes| bytes > 0)
                .unwrap_or_else(|| fs::metadata(source_path).map(|m| m.len()).unwrap_or(0))
        };

        let mut candidates = Vec::new();
        for cluster in &plan.clusters {
            for entry in &cluster.entries {
                if entry.is_slate_photo && request.exclude_slate_photo {
                    if request.archive_excluded_slates {
                        let pid = entry.photo_id.as_str();
                        candidates.push(ExportCandidate {
                            photo_id: pid.to_string(),
                            source_path: entry.source_path.clone(),
                            file_size_bytes: lookup_size(pid, &entry.source_path),
                            has_raw_companion: false,
                        });
                    }
                    continue;
                }
                let pid = entry.photo_id.as_str();
                candidates.push(ExportCandidate {
                    photo_id: pid.to_string(),
                    source_path: entry.source_path.clone(),
                    file_size_bytes: lookup_size(pid, &entry.source_path),
                    has_raw_companion: true,
                });
            }
        }
        for group in &plan.group_photos {
            let pid = group.photo_id.as_str();
            candidates.push(ExportCandidate {
                photo_id: pid.to_string(),
                source_path: group.source_path.clone(),
                file_size_bytes: lookup_size(pid, &group.source_path),
                has_raw_companion: true,
            });
        }
        for nf in &plan.no_face_photos {
            let pid = nf.photo_id.as_str();
            candidates.push(ExportCandidate {
                photo_id: pid.to_string(),
                source_path: nf.source_path.clone(),
                file_size_bytes: lookup_size(pid, &nf.source_path),
                has_raw_companion: true,
            });
        }

        if let Some(selected_ids) = parse_selected_photo_ids(request.photo_ids.as_deref()) {
            candidates.retain(|c| {
                PhotoId::new(c.photo_id.clone())
                    .map(|id| selected_ids.contains(&id))
                    .unwrap_or(false)
            });
        }

        let total_photos = candidates.len();
        let total_source_bytes: u64 = candidates.iter().map(|c| c.file_size_bytes).sum();
        let companion_raw_bytes: u64 = candidates
            .iter()
            .filter(|c| c.has_raw_companion)
            .map(|c| c.file_size_bytes)
            .sum();

        let mode = request.mode.to_ascii_lowercase();
        if total_photos == 0 {
            return Ok(EstimateExportSizeResponseDto {
                extra_disk_bytes: 0,
                edited_bytes: 0,
                originals_bytes: 0,
                total_source_bytes: 0,
                file_count: 0,
                sampled_photo_count: 0,
                is_exact: true,
            });
        }

        if mode == "move" || mode == "hardlink" {
            return Ok(EstimateExportSizeResponseDto {
                extra_disk_bytes: 0,
                edited_bytes: 0,
                originals_bytes: total_source_bytes,
                total_source_bytes,
                file_count: total_photos,
                sampled_photo_count: 0,
                is_exact: true,
            });
        }

        if mode == "copy" {
            return Ok(EstimateExportSizeResponseDto {
                extra_disk_bytes: total_source_bytes,
                edited_bytes: 0,
                originals_bytes: total_source_bytes,
                total_source_bytes,
                file_count: total_photos,
                sampled_photo_count: 0,
                is_exact: true,
            });
        }

        let format = match request.format.to_ascii_lowercase().as_str() {
            "png" => OutputImageFormat::Png,
            "tiff" | "tif" => OutputImageFormat::Tiff,
            _ => OutputImageFormat::Jpeg {
                quality: request.quality.clamp(50, 100),
            },
        };

        let sample_indices = pick_evenly_spaced_indices(total_photos, 5);
        let engine = RayonImageDevelopEngine::new();
        let default_recipe = DevelopRecipe::default();

        let measured: Vec<u64> = sample_indices
            .par_iter()
            .filter_map(|&idx| {
                let item = &candidates[idx];
                let recipe = recipes.get(&item.photo_id).unwrap_or(&default_recipe);
                engine
                    .measure_rendered_bytes(&item.source_path, recipe, format)
                    .ok()
            })
            .collect();

        let sampled_photo_count = measured.len();
        let edited_bytes = if sampled_photo_count > 0 {
            let sum: u64 = measured.iter().sum();
            let avg = (sum as f64) / (sampled_photo_count as f64);
            (avg * (total_photos as f64)).round() as u64
        } else {
            total_source_bytes
        };

        let is_dual = mode == "develop_and_raw";
        let companion_count = if is_dual {
            candidates.iter().filter(|c| c.has_raw_companion).count()
        } else {
            0
        };
        let originals_bytes = if is_dual { companion_raw_bytes } else { 0 };
        let extra_disk_bytes = edited_bytes.saturating_add(originals_bytes);

        Ok(EstimateExportSizeResponseDto {
            extra_disk_bytes,
            edited_bytes,
            originals_bytes,
            total_source_bytes,
            file_count: total_photos + companion_count,
            sampled_photo_count,
            is_exact: sampled_photo_count == total_photos && sampled_photo_count > 0,
        })
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
    let (plan, sidecars_dir) = {
        let guard = state
            .active_shoot
            .lock()
            .map_err(|_| "active_shoot state mutex poisoned".to_string())?;
        let ctx = guard
            .as_ref()
            .ok_or_else(|| "no active shoot loaded".to_string())?;
        (ctx.plan.clone(), ctx.sidecars_dir.clone())
    };

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

        let organizer = LocalFileOrganizer::new();
        let organize_uc = OrganizeShootFoldersUseCase::new(&organizer);
        let engine = RayonImageDevelopEngine::new();
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
}

fn run_instant_folder_load(
    folder_path: String,
    preview_cache_dir: &Path,
    app: &AppHandle,
) -> Result<(ActiveShootContext, Vec<BackgroundPreviewJob>), String> {
    let started = Instant::now();
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

    fs::create_dir_all(preview_cache_dir)
        .map_err(|err| format!("failed to create preview cache dir: {err}"))?;

    let root_name = root
        .file_name()
        .and_then(|n| n.to_str())
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

            // Web-native images (PNG/JPEG/WebP) render directly via WebView2 with zero proxy encode!
            let needs_proxy = !is_web_native_image_path(path);
            let (preview_dest, already_cached) = if needs_proxy {
                let mut hasher = DefaultHasher::new();
                source_str.hash(&mut hasher);
                let path_hash = hasher.finish();
                let dest =
                    preview_cache_dir.join(format!("v3_{stem}_{path_hash:016x}_preview.jpg"));
                let exists = dest.exists();
                (Some(dest), exists)
            } else {
                (None, true)
            };

            // Fast shutter time: filename timestamp immediately + EXIF for JPEGs & already-cached RAWs
            // (uncached RAW EXIF is extracted alongside its background preview so open stays <0.5s!)
            let filename_ts = parse_filename_capture_hints(&source_str).filename_timestamp_epoch_sec;
            let exact_ts = filename_ts.or_else(|| {
                if !is_raw_photo_path(path) || already_cached {
                    extract_exif_capture_timestamp_epoch_sec(&source_str)
                } else {
                    None
                }
            });

            Some(IndexedFile {
                raw_id,
                dir_name,
                source_str,
                file_size_bytes,
                preview_dest,
                already_cached,
                fallback_ts,
                exact_ts,
            })
        })
        .collect();

    // Assign deterministic unique IDs and split initial immediate RAW previews (first 36) from background stream
    let mut photo_sources = HashMap::with_capacity(indexed.len());
    let mut photo_file_sizes = HashMap::with_capacity(indexed.len());
    let mut preview_paths = HashMap::with_capacity(indexed.len());
    let mut source_to_preview = HashMap::with_capacity(indexed.len());
    let mut photo_directories = HashMap::with_capacity(indexed.len());
    let mut photo_capture_timestamps = HashMap::with_capacity(indexed.len());
    let mut dir_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut scan_targets = Vec::with_capacity(indexed.len());
    let mut immediate_raw_jobs: Vec<BackgroundPreviewJob> = Vec::new();
    let mut background_raw_jobs: Vec<BackgroundPreviewJob> = Vec::new();
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
                };
                if immediate_raw_jobs.len() < INITIAL_SYNC_RAW_PREVIEWS {
                    immediate_raw_jobs.push(job);
                } else {
                    background_raw_jobs.push(job);
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

    // Generate the first screenful (up to 36) of uncached RAW previews synchronously in parallel (~0.3s)
    if !immediate_raw_jobs.is_empty() {
        emit_progress(
            app,
            "loading",
            0,
            immediate_raw_jobs.len(),
            format!(
                "Preparing initial {} RAW preview(s)...",
                immediate_raw_jobs.len()
            ),
        );
        let ready: Vec<(String, String, String, Option<i64>)> = immediate_raw_jobs
            .into_par_iter()
            .filter_map(|job| {
                let ts = extract_exif_capture_timestamp_epoch_sec(&job.source_path);
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

    let sidecars_dir = root.join(".kroma_sidecars");
    let recipes = load_existing_recipes_once(&sidecars_dir, &scan_targets);

    let discovered_pairs: Vec<(PhotoId, String)> = scan_targets
        .iter()
        .map(|t| (t.photo_id.clone(), t.source_path.clone()))
        .collect();
    let restored_plan = JsonOrganizationPlanStore::new(&sidecars_dir)
        .load_plan()
        .ok()
        .flatten()
        .and_then(|saved| saved.reconcile_with_disk(&discovered_pairs));

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
        folder_path,
        project_name: root_name,
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
        directory_summaries,
    };

    Ok((ctx, background_raw_jobs))
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
                    let ts = extract_exif_capture_timestamp_epoch_sec(&job.source_path);
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
                        None
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
        assert!(should_render_32bit_thumbnail(
            ThumbnailPreviewRenderMode::All32Bit,
            "IMG_0001.JPG",
            &edited_recipe
        ));

        let h_default = compute_32bit_thumbnail_hash("DSC0001.ARW", 24_000_000, &default_recipe);
        let h_edited = compute_32bit_thumbnail_hash("DSC0001.ARW", 24_000_000, &edited_recipe);
        assert_ne!(h_default, h_edited);
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
}
