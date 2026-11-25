use crate::dto::FaceOverlayDto;
use kroma_app::use_cases::cluster_shoot::{PhotoScanTarget, PreExtractedPhotoScan};
use kroma_domain::develop::DevelopRecipe;
use kroma_domain::identity::ShootOrganizationPlan;
use kroma_domain::retouch::{FaceParseMap, PortraitFaceGuide};
use kroma_infra::develop::image_decoder::DecodedDevelopBuffer;
use kroma_infra::vision::face_parser::OnnxBiSeNetFaceParser;
use kroma_infra::vision::lama_inpainter::OnnxLamaInpainter;
use kroma_infra::vision::ppocr_v5::PpOcrV5Engine;
use kroma_infra::vision::subject_segmenter::OnnxSubjectSegmenter;
use kroma_infra::vision::yunet_detector::YuNetOnnxDetector;
use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

const MAX_ORGANIZATION_HISTORY: usize = 50;
const MAX_DEVELOP_BUFFER_CACHE: usize = 12;
const MAX_HI_RES_DEVELOP_BUFFER_CACHE: usize = 2;

/// Cached 16-bit half-float `KR16` / 32-bit float `KR32` working buffer + compact linear
/// proxies (`384px` histogram proxy + `640px` thumbnail proxy) for instant Studio Develop photo
/// switching, zero-I/O live RAW histograms, and ~4ms 32-bit thumbnail updates on slider edits.
#[derive(Debug, Clone)]
pub struct CachedDevelopBuffer {
    pub cache_key: String,
    pub source_path: String,
    pub kr16_bytes: Arc<Vec<u8>>,
    pub hist_buffer: Arc<DecodedDevelopBuffer>,
    pub thumb_buffer: Arc<DecodedDevelopBuffer>,
}

#[derive(Debug, Clone)]
pub struct PlanHistoryEntry {
    pub plan: ShootOrganizationPlan,
    pub label: String,
}

/// Bounded in-memory undo/redo stack for non-destructive subject organization edits
/// (Split, Merge, Manual Match, Rename, Review Queue, Re-cluster).
#[derive(Debug, Clone, Default)]
pub struct OrganizationHistory {
    undo_stack: Vec<PlanHistoryEntry>,
    redo_stack: Vec<PlanHistoryEntry>,
}

impl OrganizationHistory {
    pub fn record(&mut self, previous_plan: ShootOrganizationPlan, label: impl Into<String>) {
        if self.undo_stack.len() >= MAX_ORGANIZATION_HISTORY {
            self.undo_stack.remove(0);
        }
        self.undo_stack.push(PlanHistoryEntry {
            plan: previous_plan,
            label: label.into(),
        });
        self.redo_stack.clear();
    }

    pub fn undo(
        &mut self,
        current_plan: ShootOrganizationPlan,
    ) -> Option<(ShootOrganizationPlan, String)> {
        let entry = self.undo_stack.pop()?;
        let label = entry.label.clone();
        self.redo_stack.push(PlanHistoryEntry {
            plan: current_plan,
            label: entry.label,
        });
        Some((entry.plan, label))
    }

    pub fn redo(
        &mut self,
        current_plan: ShootOrganizationPlan,
    ) -> Option<(ShootOrganizationPlan, String)> {
        let entry = self.redo_stack.pop()?;
        let label = entry.label.clone();
        self.undo_stack.push(PlanHistoryEntry {
            plan: current_plan,
            label: entry.label,
        });
        Some((entry.plan, label))
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    pub fn undo_label(&self) -> Option<&str> {
        self.undo_stack.last().map(|e| e.label.as_str())
    }

    pub fn redo_label(&self) -> Option<&str> {
        self.redo_stack.last().map(|e| e.label.as_str())
    }

    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShootSessionMode {
    #[default]
    Folder,
    Files,
    SinglePhoto,
}

impl ShootSessionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Folder => "folder",
            Self::Files => "files",
            Self::SinglePhoto => "single",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ActiveShootContext {
    pub session_mode: ShootSessionMode,
    pub folder_path: String,
    pub project_name: String,
    pub sidecars_dir: PathBuf,
    pub is_clustered: bool,
    pub total_photos: usize,
    pub scan_elapsed_ms: u64,
    pub scan_error_count: usize,
    pub plan: ShootOrganizationPlan,
    /// In-memory undo/redo stack for cluster organization actions
    pub history: OrganizationHistory,
    /// Maps photo_id -> original RAW/JPEG/PNG source path
    pub photo_sources: HashMap<String, String>,
    /// Maps photo_id -> exact on-disk source file size in bytes
    pub photo_file_sizes: HashMap<String, u64>,
    /// Maps photo_id -> cached fast JPEG preview path (or web-native source path) for WebView rendering
    pub preview_paths: HashMap<String, String>,
    /// Maps photo_id -> optional 32-bit rendered JPEG thumbnail path matching its current recipe
    pub rendered_preview_paths: HashMap<String, String>,
    /// Maps original source_path -> fast preview path for AI scanning
    pub source_to_preview: HashMap<String, String>,
    /// Maps photo_id -> relative directory bucket id
    pub photo_directories: HashMap<String, String>,
    /// Maps photo_id -> exact EXIF / filename shutter capture timestamp (epoch sec, camera local time)
    pub photo_capture_timestamps: HashMap<String, Option<i64>>,
    /// In-memory DevelopRecipe cache so build_session_dto performs zero per-photo disk I/O
    pub recipes: HashMap<String, DevelopRecipe>,
    /// Discovered scan targets ready for instant clustering without re-walking disk
    pub scan_targets: Vec<PhotoScanTarget>,
    /// In-memory cached face detections & embeddings keyed by recognition model path
    /// so adjusting Match strictness or Time Assist re-clusters in <0.1s!
    pub cached_face_scans: Option<(String, Vec<PreExtractedPhotoScan>)>,
    /// Per-photo YuNet face guides detected on-demand for Portrait Retouching (Single-Photo or un-clustered shoots)
    pub detected_face_guides: HashMap<String, Vec<PortraitFaceGuide>>,
    /// Per-photo UI face overlay DTOs detected on-demand for live Portrait Retouching
    pub detected_face_overlays: HashMap<String, FaceOverlayDto>,
    /// Per-photo BiSeNet 19-class semantic FaceParseMap cached keyed by `(photo_id, orientation_key)`
    pub detected_face_parse_maps: HashMap<String, (String, FaceParseMap)>,
    /// Maps directory bucket id -> (display name, photo count)
    pub directory_summaries: Vec<(String, String, usize)>,
}

pub type CachedOcrEngine = Arc<Mutex<Option<(String, Arc<PpOcrV5Engine>)>>>;
pub type CachedSegmenterEngine = Arc<Mutex<Option<(String, Arc<OnnxSubjectSegmenter>)>>>;
pub type CachedFaceParserEngine = Arc<Mutex<Option<(String, Arc<OnnxBiSeNetFaceParser>)>>>;
pub type CachedLamaEngine = Arc<Mutex<Option<(String, Arc<OnnxLamaInpainter>)>>>;
pub type CachedYuNetEngine = Arc<Mutex<Option<(String, Arc<YuNetOnnxDetector>)>>>;
pub type CachedStdHistBuffers = Arc<Mutex<Vec<(String, Arc<DecodedDevelopBuffer>)>>>;
pub type CachedHiResBuffers = Arc<Mutex<Vec<(String, Arc<Vec<u8>>)>>>;
pub type CachedNativeDims = Arc<Mutex<HashMap<String, (u32, u32)>>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThumbnailProgressSnapshot {
    pub active: bool,
    pub completed: usize,
    pub total: usize,
    pub mode: String,
}

#[derive(Debug, Clone)]
struct ActiveThumbnailWorker {
    completed: usize,
    total: usize,
    mode: String,
}

/// Coordinates concurrent full-shoot and targeted single-photo 32-bit thumbnail workers so the
/// bottom status pill can never remain stuck on `0/1` due to generation-invalidation races,
/// overlapping edits, or background worker panics.
#[derive(Debug, Default)]
pub struct ThumbnailProgressTracker {
    active_workers: HashMap<u64, ActiveThumbnailWorker>,
    next_worker_id: u64,
}

impl ThumbnailProgressTracker {
    pub fn try_register_worker(
        &mut self,
        thumb_gen: &AtomicU64,
        load_gen: &AtomicU64,
        expected_thumb_gen: u64,
        expected_load_gen: u64,
        total: usize,
        mode: String,
    ) -> Option<(u64, ThumbnailProgressSnapshot)> {
        if total == 0
            || thumb_gen.load(Ordering::SeqCst) != expected_thumb_gen
            || load_gen.load(Ordering::SeqCst) != expected_load_gen
        {
            return None;
        }
        self.next_worker_id = self.next_worker_id.wrapping_add(1).max(1);
        let worker_id = self.next_worker_id;
        self.active_workers.insert(
            worker_id,
            ActiveThumbnailWorker {
                completed: 0,
                total,
                mode: mode.clone(),
            },
        );
        let snapshot = self.aggregated_snapshot(&mode);
        Some((worker_id, snapshot))
    }

    pub fn update_worker(
        &mut self,
        worker_id: u64,
        completed: usize,
    ) -> Option<ThumbnailProgressSnapshot> {
        let mode = {
            let worker = self.active_workers.get_mut(&worker_id)?;
            worker.completed = completed.min(worker.total);
            worker.mode.clone()
        };
        Some(self.aggregated_snapshot(&mode))
    }

    pub fn finish_worker(
        &mut self,
        worker_id: u64,
        fallback_mode: &str,
    ) -> Option<ThumbnailProgressSnapshot> {
        let removed = self.active_workers.remove(&worker_id)?;
        if self.active_workers.is_empty() {
            Some(ThumbnailProgressSnapshot {
                active: false,
                completed: removed.total,
                total: removed.total,
                mode: if removed.mode.is_empty() {
                    fallback_mode.to_string()
                } else {
                    removed.mode
                },
            })
        } else {
            Some(self.aggregated_snapshot(fallback_mode))
        }
    }

    pub fn invalidate_and_bump(
        &mut self,
        thumb_gen: &AtomicU64,
        mode: &str,
    ) -> (u64, bool, ThumbnailProgressSnapshot) {
        let next_gen = thumb_gen.fetch_add(1, Ordering::SeqCst) + 1;
        let had_active = !self.active_workers.is_empty();
        self.active_workers.clear();
        (
            next_gen,
            had_active,
            ThumbnailProgressSnapshot {
                active: false,
                completed: 0,
                total: 0,
                mode: mode.to_string(),
            },
        )
    }

    pub fn has_active_workers(&self) -> bool {
        !self.active_workers.is_empty()
    }

    fn aggregated_snapshot(&self, fallback_mode: &str) -> ThumbnailProgressSnapshot {
        let mut completed = 0usize;
        let mut total = 0usize;
        let mut mode: Option<&str> = None;
        for worker in self.active_workers.values() {
            completed = completed.saturating_add(worker.completed);
            total = total.saturating_add(worker.total);
            if mode.is_none() && !worker.mode.is_empty() {
                mode = Some(worker.mode.as_str());
            }
        }
        ThumbnailProgressSnapshot {
            active: total > 0 && completed < total,
            completed,
            total,
            mode: mode.unwrap_or(fallback_mode).to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevelopDecodePriority {
    Active,
    Prefetch,
}

impl DevelopDecodePriority {
    pub fn parse(raw: Option<&str>) -> Self {
        match raw {
            Some("prefetch") => Self::Prefetch,
            _ => Self::Active,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DevelopDecodeTicket {
    pub priority: DevelopDecodePriority,
    pub generation: u64,
}

#[derive(Debug, Default)]
struct DevelopDecodeQueueState {
    active_gen: u64,
    running: bool,
    waiting_active: usize,
}

/// Latest-wins priority scheduler for heavy Camera RAW working-buffer decodes.
///
/// Guarantees at most one heavy multi-threaded Bayer demosaic runs at a time so rapid subject or
/// photo switching never stacks competing Rayon decodes that starve the selected frame or drop
/// UI FPS. RAM LRU hits and SSD `.kr16` cache hits bypass `acquire_demosaic_slot` completely.
#[derive(Debug, Default)]
pub struct DevelopDecodeScheduler {
    queue: Mutex<DevelopDecodeQueueState>,
    cv: Condvar,
    active_gen_fast: AtomicU64,
}

pub struct DevelopDecodePermit {
    scheduler: Arc<DevelopDecodeScheduler>,
}

impl Drop for DevelopDecodePermit {
    fn drop(&mut self) {
        let mut guard = self
            .scheduler
            .queue
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        guard.running = false;
        drop(guard);
        self.scheduler.cv.notify_all();
    }
}

impl DevelopDecodeScheduler {
    pub fn begin_request(&self, priority: DevelopDecodePriority) -> DevelopDecodeTicket {
        match priority {
            DevelopDecodePriority::Active => {
                let mut guard = self.queue.lock().unwrap_or_else(|e| e.into_inner());
                guard.active_gen = guard.active_gen.wrapping_add(1);
                let generation = guard.active_gen;
                self.active_gen_fast.store(generation, Ordering::Release);
                drop(guard);
                self.cv.notify_all();
                DevelopDecodeTicket {
                    priority,
                    generation,
                }
            }
            DevelopDecodePriority::Prefetch => {
                let generation = self.active_gen_fast.load(Ordering::Acquire);
                DevelopDecodeTicket {
                    priority,
                    generation,
                }
            }
        }
    }

    /// Invalidates all in-flight or queued decode tickets (e.g., on folder switch or cache clear)
    /// and wakes waiting workers immediately.
    pub fn invalidate(&self) {
        let _ = self.begin_request(DevelopDecodePriority::Active);
    }

    pub fn is_superseded(&self, ticket: &DevelopDecodeTicket) -> bool {
        self.active_gen_fast.load(Ordering::Acquire) != ticket.generation
    }

    /// Acquires the single heavy RAW demosaic slot if `ticket` is still the latest winner.
    /// Returns `None` immediately when superseded by a newer `Active` selection.
    pub fn acquire_demosaic_slot(
        self: &Arc<Self>,
        ticket: DevelopDecodeTicket,
    ) -> Option<DevelopDecodePermit> {
        let mut guard = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        match ticket.priority {
            DevelopDecodePriority::Active => {
                if guard.active_gen != ticket.generation {
                    return None;
                }
                guard.waiting_active += 1;
                while guard.running && guard.active_gen == ticket.generation {
                    guard = self.cv.wait(guard).unwrap_or_else(|e| e.into_inner());
                }
                guard.waiting_active = guard.waiting_active.saturating_sub(1);
                if guard.active_gen != ticket.generation {
                    return None;
                }
                guard.running = true;
                Some(DevelopDecodePermit {
                    scheduler: Arc::clone(self),
                })
            }
            DevelopDecodePriority::Prefetch => {
                while guard.running || guard.waiting_active > 0 {
                    if guard.active_gen != ticket.generation {
                        return None;
                    }
                    guard = self.cv.wait(guard).unwrap_or_else(|e| e.into_inner());
                }
                if guard.active_gen != ticket.generation {
                    return None;
                }
                guard.running = true;
                Some(DevelopDecodePermit {
                    scheduler: Arc::clone(self),
                })
            }
        }
    }

    #[cfg(test)]
    fn waiting_active_count(&self) -> usize {
        self.queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .waiting_active
    }
}

pub struct AppState {
    pub default_preview_cache_dir: PathBuf,
    pub custom_cache_dir: Arc<Mutex<Option<PathBuf>>>,
    pub managed_models_dir: Arc<Mutex<PathBuf>>,
    pub managed_single_sidecars_dir: Arc<Mutex<PathBuf>>,
    pub model_download_cancel: Arc<AtomicBool>,
    pub active_shoot: Arc<Mutex<Option<ActiveShootContext>>>,
    pub load_generation: Arc<AtomicU64>,
    pub thumbnail_render_generation: Arc<AtomicU64>,
    pub thumbnail_progress_tracker: Arc<Mutex<ThumbnailProgressTracker>>,
    pub export_estimate_generation: Arc<AtomicU64>,
    pub custom_yunet_path: Arc<Mutex<Option<PathBuf>>>,
    pub custom_recognizer_path: Arc<Mutex<Option<PathBuf>>>,
    pub custom_segmenter_path: Arc<Mutex<Option<PathBuf>>>,
    pub custom_face_parser_path: Arc<Mutex<Option<PathBuf>>>,
    pub custom_lama_path: Arc<Mutex<Option<PathBuf>>>,
    /// Cached warmed-up PP-OCRv5 engine keyed by model paths so single-photo re-scans run in <100ms
    pub cached_ocr_engine: CachedOcrEngine,
    /// Cached warmed-up ONNX subject segmenter & YuNet detector for instant AI Masking (<200ms)
    pub cached_segmenter_engine: CachedSegmenterEngine,
    /// Cached warmed-up opt-in BiSeNet Face Parsing ONNX engine (Phase 3)
    pub cached_face_parser_engine: CachedFaceParserEngine,
    /// Cached warmed-up opt-in LaMa Generative Inpainting ONNX engine (Phase 3)
    pub cached_lama_engine: CachedLamaEngine,
    pub cached_yunet_engine: CachedYuNetEngine,
    /// Bounded LRU cache (max 12 entries) of decoded 16-bit half-float `KR16` Develop
    /// buffers + compact 384px linear histogram proxies for instant RAW switching & live histograms.
    pub develop_buffer_cache: Arc<Mutex<Vec<CachedDevelopBuffer>>>,
    /// Bounded LRU cache (max 12 entries, ~14 MB total) of 384px sRGB `f32` histogram proxies for
    /// standard JPEG/PNG/WebP photos so repeated slider adjustments compute live histograms from RAM
    /// in ~1–2ms without re-decoding disk previews.
    pub std_hist_buffer_cache: CachedStdHistBuffers,
    /// Dedicated 2-entry RAM-only LRU for on-demand native 1:1 zoom buffers (`max_dim > 2560`),
    /// kept separate from `develop_buffer_cache` so high-res loupe inspection never evicts Fit
    /// browsing buffers or bloats memory/SSD caches.
    pub hi_res_develop_buffer_cache: CachedHiResBuffers,
    /// Lightweight header-only `(width, height)` cache for true `1:1` zoom calculations.
    pub photo_native_dims_cache: CachedNativeDims,
    /// Latest-wins priority scheduler preventing competing concurrent RAW Bayer demosaics.
    pub develop_decode_scheduler: Arc<DevelopDecodeScheduler>,
}

impl AppState {
    pub fn new() -> Self {
        let default_preview_cache_dir = env::temp_dir().join("kroma_preview_cache");
        let default_managed_models_dir = env::temp_dir().join("kroma_models");
        let default_single_sidecars_dir = env::temp_dir().join("kroma_single_sidecars");
        Self {
            default_preview_cache_dir,
            custom_cache_dir: Arc::new(Mutex::new(None)),
            managed_models_dir: Arc::new(Mutex::new(default_managed_models_dir)),
            managed_single_sidecars_dir: Arc::new(Mutex::new(default_single_sidecars_dir)),
            model_download_cancel: Arc::new(AtomicBool::new(false)),
            active_shoot: Arc::new(Mutex::new(None)),
            load_generation: Arc::new(AtomicU64::new(0)),
            thumbnail_render_generation: Arc::new(AtomicU64::new(0)),
            thumbnail_progress_tracker: Arc::new(Mutex::new(ThumbnailProgressTracker::default())),
            export_estimate_generation: Arc::new(AtomicU64::new(0)),
            custom_yunet_path: Arc::new(Mutex::new(None)),
            custom_recognizer_path: Arc::new(Mutex::new(None)),
            custom_segmenter_path: Arc::new(Mutex::new(None)),
            custom_face_parser_path: Arc::new(Mutex::new(None)),
            custom_lama_path: Arc::new(Mutex::new(None)),
            cached_ocr_engine: Arc::new(Mutex::new(None)),
            cached_segmenter_engine: Arc::new(Mutex::new(None)),
            cached_face_parser_engine: Arc::new(Mutex::new(None)),
            cached_lama_engine: Arc::new(Mutex::new(None)),
            cached_yunet_engine: Arc::new(Mutex::new(None)),
            develop_buffer_cache: Arc::new(Mutex::new(Vec::with_capacity(
                MAX_DEVELOP_BUFFER_CACHE,
            ))),
            std_hist_buffer_cache: Arc::new(Mutex::new(Vec::with_capacity(
                MAX_DEVELOP_BUFFER_CACHE,
            ))),
            hi_res_develop_buffer_cache: Arc::new(Mutex::new(Vec::with_capacity(
                MAX_HI_RES_DEVELOP_BUFFER_CACHE,
            ))),
            photo_native_dims_cache: Arc::new(Mutex::new(HashMap::new())),
            develop_decode_scheduler: Arc::new(DevelopDecodeScheduler::default()),
        }
    }

    pub fn set_managed_models_dir(&self, dir: PathBuf) {
        if let Ok(mut guard) = self.managed_models_dir.lock() {
            *guard = dir;
        }
    }

    pub fn resolve_managed_models_dir(&self) -> PathBuf {
        self.managed_models_dir
            .lock()
            .map(|g| g.clone())
            .unwrap_or_else(|_| env::temp_dir().join("kroma_models"))
    }

    pub fn set_managed_single_sidecars_dir(&self, dir: PathBuf) {
        if let Ok(mut guard) = self.managed_single_sidecars_dir.lock() {
            *guard = dir;
        }
    }

    pub fn resolve_managed_single_sidecars_dir(&self) -> PathBuf {
        self.managed_single_sidecars_dir
            .lock()
            .map(|g| g.clone())
            .unwrap_or_else(|_| env::temp_dir().join("kroma_single_sidecars"))
    }

    /// Ordered root search directories for bundled or manifest-downloaded AI models.
    /// Managed app-data models take priority so downloaded updates override older copies.
    pub fn model_search_dirs(&self) -> Vec<PathBuf> {
        let mut search_dirs = Vec::new();
        let managed = self.resolve_managed_models_dir();
        if !managed.as_os_str().is_empty() {
            search_dirs.push(managed);
        }
        if let Ok(cwd) = env::current_dir() {
            search_dirs.push(cwd.join("models"));
            if let Some(p1) = cwd.parent() {
                search_dirs.push(p1.join("models"));
                if let Some(p2) = p1.parent() {
                    search_dirs.push(p2.join("models"));
                }
            }
        }
        if let Ok(exe) = env::current_exe()
            && let Some(exe_dir) = exe.parent()
        {
            search_dirs.push(exe_dir.join("models"));
        }
        search_dirs
    }

    pub fn resolve_cache_dir(&self) -> PathBuf {
        if let Ok(guard) = self.custom_cache_dir.lock()
            && let Some(ref custom) = *guard
        {
            return custom.clone();
        }
        self.default_preview_cache_dir.clone()
    }

    pub fn is_custom_cache_dir(&self) -> bool {
        self.custom_cache_dir
            .lock()
            .ok()
            .is_some_and(|guard| guard.is_some())
    }

    pub fn set_custom_cache_dir(&self, next: Option<PathBuf>) -> Result<PathBuf, String> {
        let validated = match next {
            Some(path) => {
                if path.as_os_str().is_empty() {
                    None
                } else {
                    std::fs::create_dir_all(&path).map_err(|err| {
                        format!(
                            "Could not create or access cache folder '{}': {err}",
                            path.display()
                        )
                    })?;
                    Some(path)
                }
            }
            None => None,
        };
        let mut guard = self
            .custom_cache_dir
            .lock()
            .map_err(|_| "custom_cache_dir mutex poisoned".to_string())?;
        *guard = validated;
        drop(guard);
        Ok(self.resolve_cache_dir())
    }

    pub fn get_cached_develop_buffer(&self, cache_key: &str) -> Option<Arc<Vec<u8>>> {
        let mut guard = self.develop_buffer_cache.lock().ok()?;
        let pos = guard.iter().position(|entry| entry.cache_key == cache_key)?;
        let entry = guard.remove(pos);
        let bytes = Arc::clone(&entry.kr16_bytes);
        guard.push(entry);
        Some(bytes)
    }

    /// Returns the most recently cached `KR32`/`KR16` high-bit-depth working wire buffer for
    /// `source_path` (if Studio already decoded this RAW/TIFF/16-bit photo into RAM).
    pub fn get_cached_develop_wire_for_source(&self, source_path: &str) -> Option<Arc<Vec<u8>>> {
        let guard = self.develop_buffer_cache.lock().ok()?;
        guard
            .iter()
            .rev()
            .find(|entry| entry.source_path == source_path)
            .map(|entry| Arc::clone(&entry.kr16_bytes))
    }

    pub fn get_cached_hist_buffer(&self, source_path: &str) -> Option<Arc<DecodedDevelopBuffer>> {
        if let Ok(guard) = self.develop_buffer_cache.lock()
            && let Some(entry) = guard
                .iter()
                .rev()
                .find(|entry| entry.source_path == source_path)
        {
            return Some(Arc::clone(&entry.hist_buffer));
        }
        let mut std_guard = self.std_hist_buffer_cache.lock().ok()?;
        let pos = std_guard
            .iter()
            .position(|(src, _)| src == source_path)?;
        let (src, buf) = std_guard.remove(pos);
        let cloned = Arc::clone(&buf);
        std_guard.push((src, buf));
        Some(cloned)
    }

    pub fn insert_std_hist_buffer(&self, source_path: String, buf: Arc<DecodedDevelopBuffer>) {
        let Ok(mut guard) = self.std_hist_buffer_cache.lock() else {
            return;
        };
        guard.retain(|(existing, _)| existing != &source_path);
        while guard.len() >= MAX_DEVELOP_BUFFER_CACHE {
            guard.remove(0);
        }
        guard.push((source_path, buf));
    }

    pub fn insert_develop_buffer(&self, entry: CachedDevelopBuffer) {
        let Ok(mut guard) = self.develop_buffer_cache.lock() else {
            return;
        };
        guard.retain(|existing| existing.cache_key != entry.cache_key);
        while guard.len() >= MAX_DEVELOP_BUFFER_CACHE {
            guard.remove(0);
        }
        guard.push(entry);
    }

    pub fn get_cached_hi_res_develop_buffer(&self, cache_key: &str) -> Option<Arc<Vec<u8>>> {
        let mut guard = self.hi_res_develop_buffer_cache.lock().ok()?;
        let pos = guard.iter().position(|(k, _)| k == cache_key)?;
        let (k, bytes) = guard.remove(pos);
        let cloned = Arc::clone(&bytes);
        guard.push((k, bytes));
        Some(cloned)
    }

    pub fn insert_hi_res_develop_buffer(&self, cache_key: String, bytes: Arc<Vec<u8>>) {
        let Ok(mut guard) = self.hi_res_develop_buffer_cache.lock() else {
            return;
        };
        guard.retain(|(existing, _)| existing != &cache_key);
        while guard.len() >= MAX_HI_RES_DEVELOP_BUFFER_CACHE {
            guard.remove(0);
        }
        guard.push((cache_key, bytes));
    }

    pub fn get_cached_native_dims(&self, source_path: &str) -> Option<(u32, u32)> {
        self.photo_native_dims_cache
            .lock()
            .ok()
            .and_then(|guard| guard.get(source_path).copied())
    }

    pub fn insert_cached_native_dims(&self, source_path: String, dims: (u32, u32)) {
        if let Ok(mut guard) = self.photo_native_dims_cache.lock() {
            guard.insert(source_path, dims);
        }
    }

    pub fn clear_develop_buffer_cache(&self) {
        self.develop_decode_scheduler.invalidate();
        if let Ok(mut guard) = self.develop_buffer_cache.lock() {
            guard.clear();
        }
        if let Ok(mut guard) = self.std_hist_buffer_cache.lock() {
            guard.clear();
        }
        if let Ok(mut guard) = self.hi_res_develop_buffer_cache.lock() {
            guard.clear();
        }
        if let Ok(mut guard) = self.photo_native_dims_cache.lock() {
            guard.clear();
        }
    }

    /// Resolves active detector & recognizer paths, preferring user-configured custom `.onnx`
    /// overrides (when still present on disk) and falling back to bundled default models.
    pub fn resolve_model_paths(&self) -> (Option<PathBuf>, Option<PathBuf>) {
        let (default_yunet, default_rec) = self.resolve_default_model_paths();
        let custom_yunet = self
            .custom_yunet_path
            .lock()
            .ok()
            .and_then(|g| g.clone())
            .filter(|p| p.is_file());
        let custom_rec = self
            .custom_recognizer_path
            .lock()
            .ok()
            .and_then(|g| g.clone())
            .filter(|p| p.is_file());
        (custom_yunet.or(default_yunet), custom_rec.or(default_rec))
    }

    pub fn has_custom_yunet(&self) -> bool {
        self.custom_yunet_path
            .lock()
            .ok()
            .and_then(|g| g.clone())
            .is_some_and(|p| p.is_file())
    }

    pub fn has_custom_recognizer(&self) -> bool {
        self.custom_recognizer_path
            .lock()
            .ok()
            .and_then(|g| g.clone())
            .is_some_and(|p| p.is_file())
    }

    pub fn has_custom_segmenter(&self) -> bool {
        self.custom_segmenter_path
            .lock()
            .ok()
            .and_then(|g| g.clone())
            .is_some_and(|p| p.is_file())
    }

    pub fn has_custom_face_parser(&self) -> bool {
        self.custom_face_parser_path
            .lock()
            .ok()
            .and_then(|g| g.clone())
            .is_some_and(|p| p.is_file())
    }

    pub fn has_custom_lama(&self) -> bool {
        self.custom_lama_path
            .lock()
            .ok()
            .and_then(|g| g.clone())
            .is_some_and(|p| p.is_file())
    }

    /// Resolves active BiSeNet Face Parsing ONNX model (custom override -> `models/retouch/`).
    pub fn resolve_face_parser_model_path(&self) -> Option<PathBuf> {
        let custom = self
            .custom_face_parser_path
            .lock()
            .ok()
            .and_then(|g| g.clone())
            .filter(|p| p.is_file());
        custom.or_else(|| self.resolve_default_face_parser_model_path())
    }

    pub fn resolve_default_face_parser_model_path(&self) -> Option<PathBuf> {
        let mut search_dirs = Vec::new();
        for base in self.model_search_dirs() {
            search_dirs.push(base.join("retouch"));
            search_dirs.push(base);
        }
        for dir in search_dirs {
            if !dir.is_dir() {
                continue;
            }
            if let Some(found) = find_matching_onnx(&dir, "parsing")
                .or_else(|| find_matching_onnx(&dir, "bisenet"))
            {
                return Some(found);
            }
        }
        None
    }

    pub fn face_parser_model_label(model_path: Option<&Path>) -> String {
        let Some(raw_name) = model_path.and_then(|p| p.file_name()).and_then(|n| n.to_str()) else {
            return "YuNet 5-Point (Built-in)".to_string();
        };
        let lower = raw_name.to_ascii_lowercase();
        if lower.contains("resnet34") {
            "BiSeNet ResNet-34 Face Parsing".to_string()
        } else if lower.contains("resnet18") || lower.contains("parsing") || lower.contains("bisenet") {
            "BiSeNet Face Parsing".to_string()
        } else {
            raw_name.to_string()
        }
    }

    /// Resolves active LaMa Generative Inpainting ONNX model (custom override -> `models/retouch/`).
    pub fn resolve_lama_model_path(&self) -> Option<PathBuf> {
        let custom = self
            .custom_lama_path
            .lock()
            .ok()
            .and_then(|g| g.clone())
            .filter(|p| p.is_file());
        custom.or_else(|| self.resolve_default_lama_model_path())
    }

    pub fn resolve_default_lama_model_path(&self) -> Option<PathBuf> {
        let mut search_dirs = Vec::new();
        for base in self.model_search_dirs() {
            search_dirs.push(base.join("retouch"));
            search_dirs.push(base);
        }
        for dir in search_dirs {
            if !dir.is_dir() {
                continue;
            }
            if let Some(found) = find_matching_onnx(&dir, "lama") {
                return Some(found);
            }
        }
        None
    }

    pub fn lama_model_label(model_path: Option<&Path>) -> String {
        let Some(raw_name) = model_path.and_then(|p| p.file_name()).and_then(|n| n.to_str()) else {
            return "Harmonic Fill (Built-in)".to_string();
        };
        let lower = raw_name.to_ascii_lowercase();
        if lower.contains("lama") {
            "LaMa Neural Inpainter".to_string()
        } else {
            raw_name.to_string()
        }
    }

    /// Resolves active subject segmentation ONNX model (custom override -> bundled `./models/`).
    pub fn resolve_segmenter_model_path(&self) -> Option<PathBuf> {
        let custom = self
            .custom_segmenter_path
            .lock()
            .ok()
            .and_then(|g| g.clone())
            .filter(|p| p.is_file());
        custom.or_else(|| self.resolve_default_segmenter_model_path())
    }

    pub fn resolve_default_segmenter_model_path(&self) -> Option<PathBuf> {
        let mut search_dirs = Vec::new();
        for base in self.model_search_dirs() {
            search_dirs.push(base.clone());
            search_dirs.push(base.join("masking"));
        }

        for dir in search_dirs {
            if !dir.is_dir() {
                continue;
            }
            if let Some(found) = find_matching_onnx(&dir, "rmbg")
                .or_else(|| find_matching_onnx(&dir, "birefnet"))
                .or_else(|| find_matching_onnx(&dir, "modnet"))
                .or_else(|| find_matching_onnx(&dir, "isnet"))
                .or_else(|| find_matching_onnx(&dir, "u2net"))
                .or_else(|| find_matching_onnx(&dir, "silueta"))
            {
                return Some(found);
            }
        }
        None
    }

    pub fn segmenter_model_label(model_path: Option<&Path>) -> String {
        let Some(raw_name) = model_path.and_then(|p| p.file_name()).and_then(|n| n.to_str()) else {
            return "Built-in Portrait AI (YuNet)".to_string();
        };
        let lower = raw_name.to_ascii_lowercase();
        if lower.contains("rmbg-2") || lower.contains("rmbg_2") || lower.contains("rmbg2") {
            "RMBG-2.0 Subject Matte".to_string()
        } else if lower.contains("rmbg") {
            "RMBG-1.4 Subject Matte".to_string()
        } else if lower.contains("birefnet") {
            "BiRefNet High-Res Matte".to_string()
        } else if lower.contains("modnet") {
            "MODNet Portrait Matte".to_string()
        } else if lower.contains("isnet") {
            "IS-Net Salient Matte".to_string()
        } else if lower.contains("u2net") {
            "U2-Net Subject Matte".to_string()
        } else if lower.contains("silueta") {
            "Silueta Subject Matte".to_string()
        } else {
            Path::new(raw_name)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(raw_name)
                .to_string()
        }
    }

    /// Resolves bundled default `face_detection_yunet_*.onnx` and the best available face
    /// recognition ONNX model (preferring InsightFace ArcFace ResNet-50 `w600k_r50.onnx`,
    /// falling back to OpenCV `sface`) from `./models/`, `../models/`, or beside the binary.
    pub fn resolve_default_model_paths(&self) -> (Option<PathBuf>, Option<PathBuf>) {
        let search_dirs = self.model_search_dirs();

        let mut yunet = None;
        let mut arcface = None;
        let mut sface = None;

        for dir in search_dirs {
            if !dir.is_dir() {
                continue;
            }
            if yunet.is_none() {
                yunet = find_matching_onnx(&dir, "yunet");
            }
            if arcface.is_none() {
                arcface = find_matching_onnx(&dir, "w600k_r50")
                    .or_else(|| find_matching_onnx(&dir, "arcface"))
                    .or_else(|| find_matching_onnx(&dir, "glintr"));
            }
            if sface.is_none() {
                sface = find_matching_onnx(&dir, "sface");
            }
            if yunet.is_some() && arcface.is_some() {
                break;
            }
        }

        (yunet, arcface.or(sface))
    }

    pub fn recognition_model_label(model_path: Option<&Path>) -> String {
        let Some(raw_name) = model_path.and_then(|p| p.file_name()).and_then(|n| n.to_str()) else {
            return "Missing".to_string();
        };
        let name = raw_name.to_ascii_lowercase();
        if name.contains("w600k_r50") {
            "ArcFace R50 512-D".to_string()
        } else if name.contains("w600k_mbf") || name.contains("mobilefacenet") {
            "ArcFace MobileFaceNet 512-D".to_string()
        } else if name.contains("glintr100") {
            "GlintR100 512-D".to_string()
        } else if name.contains("glintr") {
            "GlintR 512-D".to_string()
        } else if name.contains("adaface") {
            "AdaFace 512-D".to_string()
        } else if name.contains("arcface") || name.contains("w600k") {
            "ArcFace 512-D".to_string()
        } else if name.contains("sface") {
            "SFace 128-D".to_string()
        } else {
            Path::new(raw_name)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(raw_name)
                .to_string()
        }
    }

    pub fn detector_model_label(model_path: Option<&Path>) -> String {
        let Some(raw_name) = model_path.and_then(|p| p.file_name()).and_then(|n| n.to_str()) else {
            return "Missing".to_string();
        };
        let name = raw_name.to_ascii_lowercase();
        if name.contains("2026") {
            "YuNet 2026".to_string()
        } else if name.contains("2023") {
            "YuNet 2023".to_string()
        } else if name.contains("yunet") {
            "YuNet".to_string()
        } else {
            Path::new(raw_name)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(raw_name)
                .to_string()
        }
    }

    /// Resolves bundled PP-OCRv5 text detector (`*det*.onnx`), recognizer (`*rec*.onnx`),
    /// and dictionary (`*dict*.txt`) from `./models/ocr/`, `./models/`, `../models/ocr/`, or beside the exe.
    pub fn resolve_ocr_model_paths(&self) -> (Option<PathBuf>, Option<PathBuf>, Option<PathBuf>) {
        let mut search_dirs = Vec::new();
        for base in self.model_search_dirs() {
            search_dirs.push(base.join("ocr"));
            search_dirs.push(base);
        }

        let mut det = None;
        let mut rec = None;
        let mut dict = None;

        for dir in search_dirs {
            if !dir.is_dir() {
                continue;
            }
            if det.is_none() {
                det = find_matching_onnx(&dir, "ppocrv5_mobile_det")
                    .or_else(|| find_matching_onnx(&dir, "ppocrv5_det"))
                    .or_else(|| find_matching_onnx(&dir, "ocrv5_det"))
                    .or_else(|| find_matching_onnx(&dir, "det"));
            }
            if rec.is_none() {
                rec = find_matching_onnx(&dir, "en_ppocrv5_rec")
                    .or_else(|| find_matching_onnx(&dir, "latin_ppocrv5_rec"))
                    .or_else(|| find_matching_onnx(&dir, "ppocrv5_rec"))
                    .or_else(|| find_matching_onnx(&dir, "rec"));
            }
            if dict.is_none() {
                dict = find_matching_file_with_ext(&dir, "en_ppocrv5_dict", "txt")
                    .or_else(|| find_matching_file_with_ext(&dir, "ppocrv5", "txt"))
                    .or_else(|| find_matching_file_with_ext(&dir, "dict", "txt"));
            }
            if det.is_some() && rec.is_some() && dict.is_some() {
                break;
            }
        }

        (det, rec, dict)
    }

    pub fn ocr_det_model_label(model_path: Option<&Path>) -> String {
        let Some(raw_name) = model_path.and_then(|p| p.file_name()).and_then(|n| n.to_str()) else {
            return "Not installed".to_string();
        };
        let lower = raw_name.to_ascii_lowercase();
        if lower.contains("v5") && lower.contains("mobile") {
            "PP-OCRv5 Mobile Det".to_string()
        } else if lower.contains("v5") {
            "PP-OCRv5 Server Det".to_string()
        } else {
            Path::new(raw_name)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(raw_name)
                .to_string()
        }
    }

    pub fn ocr_rec_model_label(model_path: Option<&Path>) -> String {
        let Some(raw_name) = model_path.and_then(|p| p.file_name()).and_then(|n| n.to_str()) else {
            return "Not installed".to_string();
        };
        let lower = raw_name.to_ascii_lowercase();
        if lower.contains("en") && lower.contains("v5") {
            "PP-OCRv5 English + Latin Rec".to_string()
        } else if lower.contains("latin") && lower.contains("v5") {
            "PP-OCRv5 Latin Rec".to_string()
        } else if lower.contains("v5") {
            "PP-OCRv5 Mobile Rec".to_string()
        } else {
            Path::new(raw_name)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(raw_name)
                .to_string()
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

fn find_matching_onnx(dir: &Path, needle: &str) -> Option<PathBuf> {
    find_matching_file_with_ext(dir, needle, "onnx")
}

fn find_matching_file_with_ext(dir: &Path, needle: &str, ext_target: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut matches: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|ext| ext.eq_ignore_ascii_case(ext_target))
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|name| name.to_ascii_lowercase().contains(needle))
        })
        .collect();
    // Sort descending so newer dated exports (e.g. 2026may over 2023mar) win automatically
    matches.sort();
    matches.pop()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_plan(group_count: usize) -> ShootOrganizationPlan {
        let mut plan = ShootOrganizationPlan {
            clusters: Vec::new(),
            ambiguous_review_queue: Vec::new(),
            group_photos: Vec::new(),
            no_face_photos: Vec::new(),
            virtual_folders: Vec::new(),
            photo_folders: std::collections::BTreeMap::new(),
        };
        for i in 0..group_count {
            plan.no_face_photos
                .push(kroma_domain::identity::NoFacePhotoEntry {
                    photo_id: kroma_domain::photo::PhotoId::new(format!("p_{i}")).unwrap(),
                    source_path: format!("p_{i}.jpg"),
                });
        }
        plan
    }

    #[test]
    fn organization_history_supports_undo_redo_and_clears_redo_on_branch() {
        let mut hist = OrganizationHistory::default();
        assert!(!hist.can_undo());
        assert!(!hist.can_redo());

        let p0 = empty_plan(0);
        let p1 = empty_plan(1);
        let p2 = empty_plan(2);

        hist.record(p0.clone(), "Split 2 photo(s)");
        assert!(hist.can_undo());
        assert_eq!(hist.undo_label(), Some("Split 2 photo(s)"));

        hist.record(p1.clone(), "Merge subjects");
        assert_eq!(hist.undo_label(), Some("Merge subjects"));

        let (restored_p1, label) = hist.undo(p2.clone()).expect("undo 1");
        assert_eq!(label, "Merge subjects");
        assert_eq!(restored_p1.no_face_photos.len(), 1);
        assert!(hist.can_redo());
        assert_eq!(hist.redo_label(), Some("Merge subjects"));

        let (redone_p2, rlabel) = hist.redo(restored_p1).expect("redo 1");
        assert_eq!(rlabel, "Merge subjects");
        assert_eq!(redone_p2.no_face_photos.len(), 2);
        assert!(!hist.can_redo());

        // Undo then branch with a new edit -> redo stack must clear
        let (back_to_p1, _) = hist.undo(redone_p2).expect("undo 2");
        assert!(hist.can_redo());
        hist.record(back_to_p1, "Rename subject");
        assert!(!hist.can_redo());
        assert_eq!(hist.undo_label(), Some("Rename subject"));
    }

    #[test]
    fn organization_history_enforces_bounded_depth() {
        let mut hist = OrganizationHistory::default();
        for i in 0..(MAX_ORGANIZATION_HISTORY + 15) {
            hist.record(empty_plan(i % 3), format!("Action {i}"));
        }
        let mut steps = 0;
        let mut current = empty_plan(0);
        while let Some((prev, _)) = hist.undo(current) {
            current = prev;
            steps += 1;
        }
        assert_eq!(steps, MAX_ORGANIZATION_HISTORY);
    }

    #[test]
    fn std_hist_buffer_cache_enforces_lru_and_clears() {
        let state = AppState::new();
        let make_buf = || {
            Arc::new(DecodedDevelopBuffer {
                image: image::Rgb32FImage::new(2, 2),
                is_linear: false,
                is_wide_gamut: false,
                bit_depth: 8,
                is_raw_demosaic: false,
            })
        };

        let first = make_buf();
        state.insert_std_hist_buffer("p0.jpg".to_string(), Arc::clone(&first));
        for i in 1..MAX_DEVELOP_BUFFER_CACHE {
            state.insert_std_hist_buffer(format!("p{i}.jpg"), make_buf());
        }

        // Touch p0.jpg so it becomes MRU, making p1.jpg the LRU entry
        let hit = state.get_cached_hist_buffer("p0.jpg").expect("p0 hit");
        assert!(Arc::ptr_eq(&first, &hit));

        // Inserting one more entry must evict p1.jpg while retaining p0.jpg
        state.insert_std_hist_buffer("p_extra.jpg".to_string(), make_buf());
        assert!(state.get_cached_hist_buffer("p0.jpg").is_some());
        assert!(state.get_cached_hist_buffer("p1.jpg").is_none());
        assert!(state.get_cached_hist_buffer("p_extra.jpg").is_some());

        state.clear_develop_buffer_cache();
        assert!(state.get_cached_hist_buffer("p0.jpg").is_none());
        assert!(state.get_cached_hist_buffer("p_extra.jpg").is_none());
    }

    #[test]
    fn hi_res_develop_buffer_cache_bounds_to_two_entries_and_clears() {
        let state = AppState::new();
        let a = Arc::new(vec![1_u8, 2, 3]);
        let b = Arc::new(vec![4_u8, 5, 6]);
        let c = Arc::new(vec![7_u8, 8, 9]);

        state.insert_hi_res_develop_buffer("a@6144".to_string(), Arc::clone(&a));
        state.insert_hi_res_develop_buffer("b@6144".to_string(), Arc::clone(&b));
        // Touch `a` so `b` becomes LRU
        let hit = state.get_cached_hi_res_develop_buffer("a@6144").expect("a hit");
        assert!(Arc::ptr_eq(&a, &hit));

        state.insert_hi_res_develop_buffer("c@6144".to_string(), Arc::clone(&c));
        assert!(state.get_cached_hi_res_develop_buffer("a@6144").is_some());
        assert!(state.get_cached_hi_res_develop_buffer("b@6144").is_none());
        assert!(state.get_cached_hi_res_develop_buffer("c@6144").is_some());

        state.insert_cached_native_dims("raw.arw".to_string(), (6000, 4000));
        assert_eq!(state.get_cached_native_dims("raw.arw"), Some((6000, 4000)));

        state.clear_develop_buffer_cache();
        assert!(state.get_cached_hi_res_develop_buffer("a@6144").is_none());
        assert!(state.get_cached_native_dims("raw.arw").is_none());
    }

    #[test]
    fn thumbnail_progress_tracker_prevents_stale_0_of_1_sticking_across_cancels_overlaps_and_panics() {
        let thumb_gen = AtomicU64::new(1);
        let load_gen = AtomicU64::new(1);
        let mut tracker = ThumbnailProgressTracker::default();

        // 1. Overlapping single-photo edits aggregate smoothly (0/1 -> 0/2 -> 1/2 -> inactive)
        let (w1, s1) = tracker
            .try_register_worker(&thumb_gen, &load_gen, 1, 1, 1, "edited_32bit".to_string())
            .expect("worker 1 registers");
        assert_eq!((s1.active, s1.completed, s1.total), (true, 0, 1));

        let (w2, s2) = tracker
            .try_register_worker(&thumb_gen, &load_gen, 1, 1, 1, "edited_32bit".to_string())
            .expect("worker 2 registers");
        assert_eq!((s2.active, s2.completed, s2.total), (true, 0, 2));

        // Finishing worker 1 keeps progress active for remaining worker 2
        let after_w1 = tracker
            .finish_worker(w1, "edited_32bit")
            .expect("worker 1 finish emits remaining progress");
        assert_eq!((after_w1.active, after_w1.completed, after_w1.total), (true, 0, 1));

        // Finishing worker 2 clears pill cleanly
        let after_w2 = tracker
            .finish_worker(w2, "edited_32bit")
            .expect("worker 2 finish clears pill");
        assert!(!after_w2.active);
        assert!(!tracker.has_active_workers());

        // 2. Generation invalidation (e.g. open folder / clear cache / full pass) rejects stale
        //    registration (eliminating TOCTOU 0/1 leak) and ignores stale worker finish events
        let (w3, s3) = tracker
            .try_register_worker(&thumb_gen, &load_gen, 1, 1, 1, "edited_32bit".to_string())
            .expect("worker 3 registers");
        assert!(s3.active);

        let (next_gen, had_active, cleared) =
            tracker.invalidate_and_bump(&thumb_gen, "edited_32bit");
        assert_eq!(next_gen, 2);
        assert!(had_active);
        assert!(!cleared.active);

        // Stale gen-1 worker that wakes up late cannot register `0/1`
        assert!(tracker
            .try_register_worker(&thumb_gen, &load_gen, 1, 1, 1, "edited_32bit".to_string())
            .is_none());

        // Start a fresh gen-2 worker, then let cancelled gen-1 worker w3 unwind/finish:
        // w3 must be a strict no-op and NEVER clobber gen-2 progress!
        let (w4, s4) = tracker
            .try_register_worker(&thumb_gen, &load_gen, 2, 1, 4, "all_32bit".to_string())
            .expect("gen-2 worker registers");
        assert_eq!((s4.active, s4.completed, s4.total), (true, 0, 4));
        assert!(tracker.update_worker(w3, 1).is_none());
        assert!(tracker.finish_worker(w3, "edited_32bit").is_none());
        assert!(tracker.has_active_workers());

        // Advance and finish gen-2 worker cleanly
        let mid = tracker.update_worker(w4, 2).expect("gen-2 update");
        assert_eq!((mid.active, mid.completed, mid.total), (true, 2, 4));
        let done = tracker.finish_worker(w4, "all_32bit").expect("gen-2 done");
        assert!(!done.active);

        // 3. Load generation mismatch also rejects stale worker registration
        load_gen.fetch_add(1, Ordering::SeqCst);
        assert!(tracker
            .try_register_worker(&thumb_gen, &load_gen, 2, 1, 1, "edited_32bit".to_string())
            .is_none());
    }

    #[test]
    fn develop_decode_scheduler_latest_active_wins_and_preempts_stale_waiters() {
        let scheduler = Arc::new(DevelopDecodeScheduler::default());
        assert_eq!(
            DevelopDecodePriority::parse(Some("prefetch")),
            DevelopDecodePriority::Prefetch
        );
        assert_eq!(
            DevelopDecodePriority::parse(Some("active")),
            DevelopDecodePriority::Active
        );
        assert_eq!(
            DevelopDecodePriority::parse(None),
            DevelopDecodePriority::Active
        );

        // Worker 1 holds the single heavy Bayer demosaic slot.
        let t1 = scheduler.begin_request(DevelopDecodePriority::Active);
        let permit1 = scheduler
            .acquire_demosaic_slot(t1)
            .expect("initial active request acquires demosaic slot immediately");

        // User rapidly clicks photo 2 (`t2`) while a background prefetch (`p2`) also queues.
        let t2 = scheduler.begin_request(DevelopDecodePriority::Active);
        let p2 = scheduler.begin_request(DevelopDecodePriority::Prefetch);
        assert!(scheduler.is_superseded(&t1));
        assert!(!scheduler.is_superseded(&t2));
        assert!(!scheduler.is_superseded(&p2));

        let s_t2 = Arc::clone(&scheduler);
        let h_t2 = std::thread::spawn(move || s_t2.acquire_demosaic_slot(t2).is_some());
        let s_p2 = Arc::clone(&scheduler);
        let h_p2 = std::thread::spawn(move || s_p2.acquire_demosaic_slot(p2).is_some());

        while scheduler.waiting_active_count() < 1 {
            std::thread::yield_now();
        }

        // User clicks photo 3 (`t3`) before worker 1 finishes: both `t2` and `p2` must wake up
        // and abort immediately without waiting for worker 1 to finish.
        let t3 = scheduler.begin_request(DevelopDecodePriority::Active);
        assert!(!h_t2.join().expect("t2 thread joined"));
        assert!(!h_p2.join().expect("p2 thread joined"));
        assert!(scheduler.is_superseded(&t2));
        assert!(scheduler.is_superseded(&p2));
        assert!(!scheduler.is_superseded(&t3));

        // Queue `t3`, then release `permit1`: `t3` must immediately acquire the demosaic slot.
        let s_t3 = Arc::clone(&scheduler);
        let h_t3 = std::thread::spawn(move || {
            let permit3 = s_t3.acquire_demosaic_slot(t3);
            permit3.is_some()
        });
        while scheduler.waiting_active_count() < 1 {
            std::thread::yield_now();
        }
        drop(permit1);
        assert!(h_t3.join().expect("t3 thread joined"));
    }

    #[test]
    fn develop_decode_scheduler_active_outranks_same_gen_prefetch_and_recovers_after_panic() {
        let scheduler = Arc::new(DevelopDecodeScheduler::default());
        let order = Arc::new(Mutex::new(Vec::<&'static str>::new()));

        let t1 = scheduler.begin_request(DevelopDecodePriority::Active);
        let permit1 = scheduler.acquire_demosaic_slot(t1).expect("t1 permit");

        // Same generation `t2` (Active) and `p2` (Prefetch) both wait behind `permit1`.
        let t2 = scheduler.begin_request(DevelopDecodePriority::Active);
        let p2 = scheduler.begin_request(DevelopDecodePriority::Prefetch);

        let s_t2 = Arc::clone(&scheduler);
        let o_t2 = Arc::clone(&order);
        let h_t2 = std::thread::spawn(move || {
            let permit2 = s_t2
                .acquire_demosaic_slot(t2)
                .expect("active t2 must acquire slot ahead of same-gen prefetch");
            o_t2.lock().unwrap().push("active");
            drop(permit2);
        });
        while scheduler.waiting_active_count() < 1 {
            std::thread::yield_now();
        }

        let s_p2 = Arc::clone(&scheduler);
        let o_p2 = Arc::clone(&order);
        let h_p2 = std::thread::spawn(move || {
            let permit_p2 = s_p2
                .acquire_demosaic_slot(p2)
                .expect("prefetch runs cleanly once active finishes");
            o_p2.lock().unwrap().push("prefetch");
            drop(permit_p2);
        });

        // Give prefetch a brief moment to enter the wait queue behind active t2, then release t1.
        std::thread::sleep(std::time::Duration::from_millis(10));
        drop(permit1);

        h_t2.join().expect("t2 finished");
        h_p2.join().expect("p2 finished");
        assert_eq!(*order.lock().unwrap(), vec!["active", "prefetch"]);

        // Panic while holding a permit must still release the slot cleanly via Drop.
        let s_panic = Arc::clone(&scheduler);
        let panicked = std::thread::spawn(move || {
            let t_bad = s_panic.begin_request(DevelopDecodePriority::Active);
            let _permit = s_panic.acquire_demosaic_slot(t_bad).expect("t_bad permit");
            panic!("simulated demosaic panic");
        })
        .join();
        assert!(panicked.is_err());

        let t_next = scheduler.begin_request(DevelopDecodePriority::Active);
        let permit_next = scheduler.acquire_demosaic_slot(t_next);
        assert!(permit_next.is_some());
    }
}
