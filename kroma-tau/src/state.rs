use kroma_app::use_cases::cluster_shoot::{PhotoScanTarget, PreExtractedPhotoScan};
use kroma_domain::develop::DevelopRecipe;
use kroma_domain::identity::ShootOrganizationPlan;
use kroma_infra::develop::image_decoder::DecodedDevelopBuffer;
use kroma_infra::vision::ppocr_v5::PpOcrV5Engine;
use kroma_infra::vision::subject_segmenter::OnnxSubjectSegmenter;
use kroma_infra::vision::yunet_detector::YuNetOnnxDetector;
use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, Mutex};

const MAX_ORGANIZATION_HISTORY: usize = 50;
const MAX_DEVELOP_BUFFER_CACHE: usize = 4;

/// Cached 16-bit half-float `KR16` working buffer + compact 384px linear histogram proxy
/// for instant Studio Develop photo switching and zero-I/O live RAW histograms.
#[derive(Debug, Clone)]
pub struct CachedDevelopBuffer {
    pub cache_key: String,
    pub source_path: String,
    pub kr16_bytes: Arc<Vec<u8>>,
    pub hist_buffer: Arc<DecodedDevelopBuffer>,
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

#[derive(Debug, Clone)]
pub struct ActiveShootContext {
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
    /// Maps directory bucket id -> (display name, photo count)
    pub directory_summaries: Vec<(String, String, usize)>,
}

pub type CachedOcrEngine = Arc<Mutex<Option<(String, Arc<PpOcrV5Engine>)>>>;
pub type CachedSegmenterEngine = Arc<Mutex<Option<(String, Arc<OnnxSubjectSegmenter>)>>>;
pub type CachedYuNetEngine = Arc<Mutex<Option<(String, Arc<YuNetOnnxDetector>)>>>;

pub struct AppState {
    pub default_preview_cache_dir: PathBuf,
    pub custom_cache_dir: Arc<Mutex<Option<PathBuf>>>,
    pub managed_models_dir: Arc<Mutex<PathBuf>>,
    pub model_download_cancel: Arc<AtomicBool>,
    pub active_shoot: Arc<Mutex<Option<ActiveShootContext>>>,
    pub load_generation: Arc<AtomicU64>,
    pub custom_yunet_path: Arc<Mutex<Option<PathBuf>>>,
    pub custom_recognizer_path: Arc<Mutex<Option<PathBuf>>>,
    pub custom_segmenter_path: Arc<Mutex<Option<PathBuf>>>,
    /// Cached warmed-up PP-OCRv5 engine keyed by model paths so single-photo re-scans run in <100ms
    pub cached_ocr_engine: CachedOcrEngine,
    /// Cached warmed-up ONNX subject segmenter & YuNet detector for instant AI Masking (<200ms)
    pub cached_segmenter_engine: CachedSegmenterEngine,
    pub cached_yunet_engine: CachedYuNetEngine,
    /// Bounded LRU cache (max 4 entries, ~50 MB) of decoded 16-bit half-float `KR16` Develop
    /// buffers + compact 384px linear histogram proxies for instant RAW switching & live histograms.
    pub develop_buffer_cache: Arc<Mutex<Vec<CachedDevelopBuffer>>>,
}

impl AppState {
    pub fn new() -> Self {
        let default_preview_cache_dir = env::temp_dir().join("kroma_preview_cache");
        let default_managed_models_dir = env::temp_dir().join("kroma_models");
        Self {
            default_preview_cache_dir,
            custom_cache_dir: Arc::new(Mutex::new(None)),
            managed_models_dir: Arc::new(Mutex::new(default_managed_models_dir)),
            model_download_cancel: Arc::new(AtomicBool::new(false)),
            active_shoot: Arc::new(Mutex::new(None)),
            load_generation: Arc::new(AtomicU64::new(0)),
            custom_yunet_path: Arc::new(Mutex::new(None)),
            custom_recognizer_path: Arc::new(Mutex::new(None)),
            custom_segmenter_path: Arc::new(Mutex::new(None)),
            cached_ocr_engine: Arc::new(Mutex::new(None)),
            cached_segmenter_engine: Arc::new(Mutex::new(None)),
            cached_yunet_engine: Arc::new(Mutex::new(None)),
            develop_buffer_cache: Arc::new(Mutex::new(Vec::with_capacity(
                MAX_DEVELOP_BUFFER_CACHE,
            ))),
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

    pub fn get_cached_hist_buffer(&self, source_path: &str) -> Option<Arc<DecodedDevelopBuffer>> {
        let guard = self.develop_buffer_cache.lock().ok()?;
        guard
            .iter()
            .rev()
            .find(|entry| entry.source_path == source_path)
            .map(|entry| Arc::clone(&entry.hist_buffer))
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

    pub fn clear_develop_buffer_cache(&self) {
        if let Ok(mut guard) = self.develop_buffer_cache.lock() {
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
}
