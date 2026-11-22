use crate::error::DomainError;
use crate::photo::PhotoId;
use std::collections::{BTreeMap, HashSet};

pub const SFACE_EMBEDDING_DIM: usize = 128;
pub const ARCFACE_EMBEDDING_DIM: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl BoundingBox {
    pub fn area(&self) -> f32 {
        self.width.max(0.0) * self.height.max(0.0)
    }
}

/// YuNet 5-point facial landmarks (right eye, left eye, nose tip, right mouth, left mouth).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YuNetLandmarks {
    pub right_eye: Point2D,
    pub left_eye: Point2D,
    pub nose_tip: Point2D,
    pub right_mouth_corner: Point2D,
    pub left_mouth_corner: Point2D,
}

impl YuNetLandmarks {
    /// Calculates head tilt angle in degrees based on eye horizon line for batch auto-leveling.
    pub fn eye_horizon_angle_deg(&self) -> f32 {
        let dy = self.left_eye.y - self.right_eye.y;
        let dx = self.left_eye.x - self.right_eye.x;
        dy.atan2(dx).to_degrees()
    }
}

/// L2-normalized face feature vector produced by ArcFace (512-D) or SFace (128-D).
#[derive(Debug, Clone, PartialEq)]
pub struct FaceEmbedding {
    vector: Vec<f32>,
}

impl FaceEmbedding {
    pub fn from_vec(raw: Vec<f32>) -> Result<Self, DomainError> {
        if raw.is_empty() {
            return Err(DomainError::ZeroMagnitudeEmbedding);
        }
        let norm = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm <= f32::EPSILON || !norm.is_finite() {
            return Err(DomainError::ZeroMagnitudeEmbedding);
        }
        let normalized = raw.into_iter().map(|val| val / norm).collect();
        Ok(Self { vector: normalized })
    }

    pub fn from_raw(raw: [f32; SFACE_EMBEDDING_DIM]) -> Result<Self, DomainError> {
        Self::from_vec(raw.to_vec())
    }

    pub fn dim(&self) -> usize {
        self.vector.len()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.vector
    }

    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        if self.vector.len() != other.vector.len() {
            return 0.0;
        }
        self.vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>()
            .clamp(-1.0, 1.0)
    }

    /// Computes an updated L2-normalized running centroid from `count` existing samples + `next`.
    pub fn merge_into_centroid(&self, existing_count: usize, next: &Self) -> Self {
        if self.vector.len() != next.vector.len() {
            return self.clone();
        }
        let w = existing_count.max(1) as f32;
        let combined: Vec<f32> = self
            .vector
            .iter()
            .zip(next.vector.iter())
            .map(|(&current, &incoming)| (current * w + incoming) / (w + 1.0))
            .collect();
        Self::from_vec(combined).unwrap_or_else(|_| self.clone())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimilarityThresholds {
    pub verified_match_min: f32,
    pub ambiguous_review_min: f32,
}

impl SimilarityThresholds {
    pub fn new(verified_match_min: f32, ambiguous_review_min: f32) -> Result<Self, DomainError> {
        if !(0.0..=1.0).contains(&verified_match_min) {
            return Err(DomainError::InvalidSimilarityThreshold(verified_match_min));
        }
        if !(0.0..=1.0).contains(&ambiguous_review_min)
            || ambiguous_review_min > verified_match_min
        {
            return Err(DomainError::InvalidSimilarityThreshold(ambiguous_review_min));
        }
        Ok(Self {
            verified_match_min,
            ambiguous_review_min,
        })
    }

    pub fn classify(&self, cosine_score: f32) -> MatchConfidence {
        if cosine_score >= self.verified_match_min {
            MatchConfidence::Verified
        } else if cosine_score >= self.ambiguous_review_min {
            MatchConfidence::AmbiguousNeedsReview
        } else {
            MatchConfidence::Unmatched
        }
    }
}

impl Default for SimilarityThresholds {
    fn default() -> Self {
        Self {
            verified_match_min: 0.44,
            ambiguous_review_min: 0.35,
        }
    }
}

/// Pure capture-identity hints extracted from a photo's filename (no filesystem or EXIF crate I/O).
/// Supports camera sequence matching (`DSC03512` -> `DSC03513`), RAW+Edited stem pairing
/// (`DSC03440.ARW` <-> `DSC03440_edited.jpg`), and embedded filename timestamps (`20260314_101402`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureIdentityHints {
    pub canonical_stem: String,
    pub camera_prefix: Option<String>,
    pub camera_sequence: Option<u64>,
    pub filename_timestamp_epoch_sec: Option<i64>,
}

/// Extracts canonical stem, camera shutter sequence, and any embedded filename timestamp.
pub fn parse_filename_capture_hints(source_path: &str) -> CaptureIdentityHints {
    let file_name = source_path
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(source_path);
    let raw_stem = match file_name.rfind('.') {
        Some(dot) if dot > 0 => &file_name[..dot],
        _ => file_name,
    };

    let canonical_stem = normalize_canonical_stem(raw_stem);
    let filename_timestamp_epoch_sec = parse_filename_timestamp_epoch_sec(&canonical_stem);
    let (camera_prefix, camera_sequence) = if filename_timestamp_epoch_sec.is_some() {
        (None, None)
    } else {
        parse_camera_sequence(&canonical_stem)
    };

    CaptureIdentityHints {
        canonical_stem,
        camera_prefix,
        camera_sequence,
        filename_timestamp_epoch_sec,
    }
}

/// Parses standard EXIF `DateTimeOriginal` strings (`"YYYY:MM:DD HH:MM:SS"` or ISO-like `"YYYY-MM-DD HH:MM:SS"`)
/// into Unix epoch seconds (treated consistently in camera local time for shoot-relative intervals).
pub fn parse_exif_datetime_epoch_sec(raw: &str) -> Option<i64> {
    let digits: Vec<u8> = raw
        .bytes()
        .filter(|b| b.is_ascii_digit())
        .collect();
    if digits.len() < 14 {
        return None;
    }
    let parse_slice = |start: usize, len: usize| -> Option<u32> {
        std::str::from_utf8(&digits[start..start + len])
            .ok()?
            .parse::<u32>()
            .ok()
    };
    let year = parse_slice(0, 4)? as i64;
    let month = parse_slice(4, 2)?;
    let day = parse_slice(6, 2)?;
    let hour = parse_slice(8, 2)?;
    let minute = parse_slice(10, 2)?;
    let second = parse_slice(12, 2)?;
    civil_datetime_to_epoch_sec(year, month, day, hour, minute, second)
}

fn normalize_canonical_stem(raw_stem: &str) -> String {
    let mut s = raw_stem.trim().to_ascii_lowercase();
    // Strip trailing duplicate copy markers like " (1)" or " (2)"
    if let Some(open) = s.rfind(" (")
        && s.ends_with(')')
        && s[open + 2..s.len() - 1]
            .bytes()
            .all(|b| b.is_ascii_digit())
    {
        s.truncate(open);
    }
    const EDIT_SUFFIXES: &[&str] = &[
        "_edited",
        "-edited",
        "_edit",
        "-edit",
        "_retouched",
        "-retouched",
        "_graded",
        "-graded",
        "_final",
        "-final",
        "_export",
        "-export",
        "_web",
        "-web",
    ];
    loop {
        let mut stripped = false;
        for suffix in EDIT_SUFFIXES {
            if let Some(trimmed) = s.strip_suffix(suffix)
                && !trimmed.is_empty()
            {
                s = trimmed.to_string();
                stripped = true;
                break;
            }
        }
        if !stripped {
            break;
        }
    }
    s
}

fn parse_filename_timestamp_epoch_sec(stem: &str) -> Option<i64> {
    let bytes = stem.as_bytes();
    // Pattern 1: YYYYMMDD[_-]?HHMMSS (e.g. IMG_20260314_101402, PXL_20260314_101402123, 20260314101402)
    for i in 0..bytes.len() {
        if i + 14 > bytes.len() {
            break;
        }
        if !bytes[i..i + 8].iter().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let year = std::str::from_utf8(&bytes[i..i + 4]).ok()?.parse::<i64>().ok()?;
        let month = std::str::from_utf8(&bytes[i + 4..i + 6]).ok()?.parse::<u32>().ok()?;
        let day = std::str::from_utf8(&bytes[i + 6..i + 8]).ok()?.parse::<u32>().ok()?;
        let time_start = if i + 8 < bytes.len()
            && matches!(bytes[i + 8], b'_' | b'-' | b' ' | b't' | b'T')
        {
            i + 9
        } else {
            i + 8
        };
        if time_start + 6 <= bytes.len()
            && bytes[time_start..time_start + 6]
                .iter()
                .all(|b| b.is_ascii_digit())
        {
            let hour = std::str::from_utf8(&bytes[time_start..time_start + 2])
                .ok()?
                .parse::<u32>()
                .ok()?;
            let minute = std::str::from_utf8(&bytes[time_start + 2..time_start + 4])
                .ok()?
                .parse::<u32>()
                .ok()?;
            let second = std::str::from_utf8(&bytes[time_start + 4..time_start + 6])
                .ok()?
                .parse::<u32>()
                .ok()?;
            if let Some(ts) = civil_datetime_to_epoch_sec(year, month, day, hour, minute, second) {
                return Some(ts);
            }
        }
    }

    // Pattern 2: YYYY-MM-DD[ _T]HH[-_.:]MM[-_.:]SS
    for i in 0..bytes.len() {
        if i + 19 > bytes.len() {
            break;
        }
        let window = &bytes[i..i + 19];
        if window[0..4].iter().all(|b| b.is_ascii_digit())
            && matches!(window[4], b'-' | b'_' | b'.')
            && window[5..7].iter().all(|b| b.is_ascii_digit())
            && matches!(window[7], b'-' | b'_' | b'.')
            && window[8..10].iter().all(|b| b.is_ascii_digit())
            && matches!(window[10], b'_' | b'-' | b' ' | b't' | b'T')
            && window[11..13].iter().all(|b| b.is_ascii_digit())
            && matches!(window[13], b'-' | b'_' | b'.' | b':')
            && window[14..16].iter().all(|b| b.is_ascii_digit())
            && matches!(window[16], b'-' | b'_' | b'.' | b':')
            && window[17..19].iter().all(|b| b.is_ascii_digit())
        {
            let year = std::str::from_utf8(&window[0..4]).ok()?.parse::<i64>().ok()?;
            let month = std::str::from_utf8(&window[5..7]).ok()?.parse::<u32>().ok()?;
            let day = std::str::from_utf8(&window[8..10]).ok()?.parse::<u32>().ok()?;
            let hour = std::str::from_utf8(&window[11..13]).ok()?.parse::<u32>().ok()?;
            let minute = std::str::from_utf8(&window[14..16]).ok()?.parse::<u32>().ok()?;
            let second = std::str::from_utf8(&window[17..19]).ok()?.parse::<u32>().ok()?;
            if let Some(ts) = civil_datetime_to_epoch_sec(year, month, day, hour, minute, second) {
                return Some(ts);
            }
        }
    }

    None
}

fn parse_camera_sequence(stem: &str) -> (Option<String>, Option<u64>) {
    let bytes = stem.as_bytes();
    let mut end = bytes.len();
    while end > 0 && !bytes[end - 1].is_ascii_digit() {
        end -= 1;
    }
    let mut start = end;
    while start > 0 && bytes[start - 1].is_ascii_digit() {
        start -= 1;
    }
    let digit_len = end.saturating_sub(start);
    if !(3..=7).contains(&digit_len) {
        return (None, None);
    }
    let Ok(seq) = stem[start..end].parse::<u64>() else {
        return (None, None);
    };
    let prefix = stem[..start]
        .trim_end_matches(['_', '-', ' '])
        .to_string();
    let prefix_opt = if prefix.is_empty() { None } else { Some(prefix) };
    (prefix_opt, Some(seq))
}

fn civil_datetime_to_epoch_sec(
    year: i64,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
) -> Option<i64> {
    if !(1990..=2100).contains(&year)
        || !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || second > 59
    {
        return None;
    }
    // Howard Hinnant's exact civil-from-days inverse algorithm
    let y = year - i64::from(month <= 2);
    let era = y.div_euclid(400);
    let yoe = (y - era * 400) as u64;
    let m = u64::from(month);
    let d = u64::from(day);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days_since_epoch = era * 146_097 + (doe as i64) - 719_468;
    Some(
        days_since_epoch * 86_400
            + i64::from(hour) * 3_600
            + i64::from(minute) * 60
            + i64::from(second),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchConfidence {
    Verified,
    AmbiguousNeedsReview,
    Unmatched,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DetectedFace {
    pub photo_id: PhotoId,
    pub bounding_box: BoundingBox,
    pub landmarks: YuNetLandmarks,
    pub detection_confidence: f32,
    pub embedding: FaceEmbedding,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClusterId(String);

impl ClusterId {
    pub fn new(id: impl Into<String>) -> Result<Self, DomainError> {
        let val = id.into();
        if val.trim().is_empty() {
            return Err(DomainError::EmptyIdentifier("ClusterId"));
        }
        Ok(Self(val))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Optional whiteboard/placard OCR metadata attached to a face cluster when a slate photo exists.
#[derive(Debug, Clone, PartialEq)]
pub struct SlateMetadata {
    pub slate_photo_id: PhotoId,
    pub ocr_raw_text: String,
    pub ocr_confidence: f32,
    pub calibrated_white_balance_kelvin: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClusterPhotoEntry {
    pub photo_id: PhotoId,
    pub source_path: String,
    pub cosine_similarity: f32,
    pub confidence: MatchConfidence,
    pub face_bounding_box: BoundingBox,
    pub landmarks: YuNetLandmarks,
    pub is_slate_photo: bool,
}

/// Face-first Subject Cluster. Works cleanly both WITH a whiteboard slate (`slate: Some(...)`)
/// and WITHOUT a slate (`slate: None`, auto-named `Person_001` with a representative preview photo).
#[derive(Debug, Clone, PartialEq)]
pub struct SubjectCluster {
    pub id: ClusterId,
    pub display_name: String,
    pub subject_code: Option<String>,
    pub representative_photo_id: PhotoId,
    pub representative_source_path: String,
    pub centroid_embedding: FaceEmbedding,
    pub slate: Option<SlateMetadata>,
    pub entries: Vec<ClusterPhotoEntry>,
    verified_embedding_count: usize,
}

impl SubjectCluster {
    pub fn new_from_face(
        id: ClusterId,
        display_name: impl Into<String>,
        subject_code: Option<String>,
        face: DetectedFace,
        source_path: impl Into<String>,
        slate: Option<SlateMetadata>,
    ) -> Result<Self, DomainError> {
        let display_name = display_name.into();
        if display_name.trim().is_empty() {
            return Err(DomainError::EmptyIdentifier("display_name"));
        }
        let source_path = source_path.into();
        let is_slate_photo = slate.is_some();
        let entry = ClusterPhotoEntry {
            photo_id: face.photo_id.clone(),
            source_path: source_path.clone(),
            cosine_similarity: 1.0,
            confidence: MatchConfidence::Verified,
            face_bounding_box: face.bounding_box,
            landmarks: face.landmarks,
            is_slate_photo,
        };

        Ok(Self {
            id,
            display_name,
            subject_code,
            representative_photo_id: face.photo_id,
            representative_source_path: source_path,
            centroid_embedding: face.embedding,
            slate,
            entries: vec![entry],
            verified_embedding_count: 1,
        })
    }

    pub fn add_verified_match(
        &mut self,
        face: DetectedFace,
        source_path: impl Into<String>,
        cosine_similarity: f32,
        is_slate_photo: bool,
    ) {
        self.centroid_embedding = self
            .centroid_embedding
            .merge_into_centroid(self.verified_embedding_count, &face.embedding);
        self.verified_embedding_count += 1;

        self.entries.push(ClusterPhotoEntry {
            photo_id: face.photo_id,
            source_path: source_path.into(),
            cosine_similarity,
            confidence: MatchConfidence::Verified,
            face_bounding_box: face.bounding_box,
            landmarks: face.landmarks,
            is_slate_photo,
        });
    }

    pub fn rename(&mut self, next_display_name: impl Into<String>) -> Result<(), DomainError> {
        let trimmed = next_display_name.into().trim().to_string();
        if trimmed.is_empty() {
            return Err(DomainError::EmptyIdentifier("display_name"));
        }
        self.display_name = trimmed;
        Ok(())
    }

    /// Applies an approved Slate OCR rename to this subject cluster, optionally recording the
    /// detected subject code, OCR metadata, and marking the whiteboard frame so Export excludes it.
    pub fn apply_slate_ocr_result(
        &mut self,
        next_display_name: impl Into<String>,
        subject_code: Option<String>,
        slate_photo_id: Option<PhotoId>,
        ocr_raw_text: String,
        ocr_confidence: f32,
        mark_as_slate_photo: bool,
    ) -> Result<(), DomainError> {
        self.rename(next_display_name)?;
        if let Some(code) = subject_code.map(|c| c.trim().to_string()).filter(|c| !c.is_empty()) {
            self.subject_code = Some(code);
        }
        if let Some(ref pid) = slate_photo_id {
            let calibrated_wb = self
                .slate
                .as_ref()
                .and_then(|s| s.calibrated_white_balance_kelvin);
            self.slate = Some(SlateMetadata {
                slate_photo_id: pid.clone(),
                ocr_raw_text,
                ocr_confidence: ocr_confidence.clamp(0.0, 1.0),
                calibrated_white_balance_kelvin: calibrated_wb,
            });
            if mark_as_slate_photo {
                for entry in &mut self.entries {
                    entry.is_slate_photo = entry.photo_id == *pid;
                }
            }
        }
        Ok(())
    }

    pub fn has_slate(&self) -> bool {
        self.slate.is_some()
    }

    pub fn verified_embedding_count(&self) -> usize {
        self.verified_embedding_count
    }

    /// Reconstructs a previously persisted `SubjectCluster` while enforcing core domain invariants.
    #[allow(clippy::too_many_arguments)]
    pub fn restore(
        id: ClusterId,
        display_name: impl Into<String>,
        subject_code: Option<String>,
        representative_photo_id: PhotoId,
        representative_source_path: impl Into<String>,
        centroid_embedding: FaceEmbedding,
        slate: Option<SlateMetadata>,
        entries: Vec<ClusterPhotoEntry>,
        verified_embedding_count: usize,
    ) -> Result<Self, DomainError> {
        let display_name = display_name.into();
        if display_name.trim().is_empty() {
            return Err(DomainError::EmptyIdentifier("display_name"));
        }
        if entries.is_empty() {
            return Err(DomainError::EmptyIdentifier("entries"));
        }
        let (rep_id, rep_path) = if let Some(found) =
            entries.iter().find(|e| e.photo_id == representative_photo_id)
        {
            (found.photo_id.clone(), found.source_path.clone())
        } else {
            let first = &entries[0];
            (first.photo_id.clone(), first.source_path.clone())
        };
        let rep_source_fallback = representative_source_path.into();
        let representative_source_path = if rep_path.is_empty() {
            rep_source_fallback
        } else {
            rep_path
        };

        Ok(Self {
            id,
            display_name,
            subject_code,
            representative_photo_id: rep_id,
            representative_source_path,
            centroid_embedding,
            slate,
            entries,
            verified_embedding_count: verified_embedding_count.max(1),
        })
    }
}

/// Borderline SFace match awaiting 1-click human approval or split into a new cluster.
#[derive(Debug, Clone, PartialEq)]
pub struct AmbiguousMatchCandidate {
    pub photo_id: PhotoId,
    pub source_path: String,
    pub candidate_cluster_id: ClusterId,
    pub cosine_similarity: f32,
    pub face_bounding_box: BoundingBox,
    pub landmarks: YuNetLandmarks,
}

/// Photo with 2+ distinct faces (e.g., team/family/couple portraits) linked to matched clusters.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupPhotoEntry {
    pub photo_id: PhotoId,
    pub source_path: String,
    pub matched_cluster_ids: Vec<ClusterId>,
    pub detected_face_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NoFacePhotoEntry {
    pub photo_id: PhotoId,
    pub source_path: String,
}

/// Stable identifier for a non-destructive virtual organization folder (e.g. a Department or College).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VirtualFolderId(String);

impl VirtualFolderId {
    pub fn new(id: impl Into<String>) -> Result<Self, DomainError> {
        let val = id.into();
        if val.trim().is_empty() {
            return Err(DomainError::EmptyIdentifier("VirtualFolderId"));
        }
        Ok(Self(val))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Non-destructive virtual folder node forming an arbitrary-depth department/category tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualFolder {
    pub id: VirtualFolderId,
    pub name: String,
    pub parent_id: Option<VirtualFolderId>,
}

/// Complete identity clustering & virtual folder organization plan for a shoot.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ShootOrganizationPlan {
    pub clusters: Vec<SubjectCluster>,
    pub ambiguous_review_queue: Vec<AmbiguousMatchCandidate>,
    pub group_photos: Vec<GroupPhotoEntry>,
    pub no_face_photos: Vec<NoFacePhotoEntry>,
    pub virtual_folders: Vec<VirtualFolder>,
    /// Per-photo virtual folder assignments (single source of truth surviving merges/splits/re-clustering).
    pub photo_folders: BTreeMap<PhotoId, VirtualFolderId>,
}

impl ShootOrganizationPlan {
    /// Creates a new virtual folder (optionally nested inside `parent_id`), enforcing non-empty
    /// and unique sibling names within the same parent.
    pub fn create_virtual_folder(
        &mut self,
        name: impl Into<String>,
        parent_id: Option<VirtualFolderId>,
    ) -> Result<VirtualFolderId, DomainError> {
        let trimmed = name.into().trim().to_string();
        if trimmed.is_empty() {
            return Err(DomainError::EmptyIdentifier("virtual folder name"));
        }
        if let Some(ref pid) = parent_id
            && !self.virtual_folders.iter().any(|f| &f.id == pid)
        {
            return Err(DomainError::EmptyIdentifier("parent virtual folder not found"));
        }
        if self.has_sibling_folder_name(&trimmed, parent_id.as_ref(), None) {
            return Err(DomainError::EmptyIdentifier(
                "a folder with that name already exists here",
            ));
        }

        let slug: String = trimmed
            .to_ascii_lowercase()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect::<String>()
            .trim_matches('_')
            .to_string();
        let base_slug = if slug.is_empty() {
            "folder".to_string()
        } else {
            slug.chars().take(24).collect()
        };

        let mut suffix = 1usize;
        let folder_id = loop {
            let candidate = VirtualFolderId::new(format!("vf_{base_slug}_{suffix}"))?;
            if !self.virtual_folders.iter().any(|f| f.id == candidate) {
                break candidate;
            }
            suffix += 1;
        };

        self.virtual_folders.push(VirtualFolder {
            id: folder_id.clone(),
            name: trimmed,
            parent_id,
        });
        Ok(folder_id)
    }

    /// Renames an existing virtual folder while preserving sibling uniqueness.
    pub fn rename_virtual_folder(
        &mut self,
        folder_id: &VirtualFolderId,
        next_name: impl Into<String>,
    ) -> Result<(), DomainError> {
        let trimmed = next_name.into().trim().to_string();
        if trimmed.is_empty() {
            return Err(DomainError::EmptyIdentifier("virtual folder name"));
        }
        let parent_id = self
            .virtual_folders
            .iter()
            .find(|f| &f.id == folder_id)
            .ok_or(DomainError::EmptyIdentifier("virtual folder not found"))?
            .parent_id
            .clone();

        if self.has_sibling_folder_name(&trimmed, parent_id.as_ref(), Some(folder_id)) {
            return Err(DomainError::EmptyIdentifier(
                "a folder with that name already exists here",
            ));
        }

        let folder = self
            .virtual_folders
            .iter_mut()
            .find(|f| &f.id == folder_id)
            .expect("folder verified above");
        folder.name = trimmed;
        Ok(())
    }

    /// Re-parents a virtual folder inside another folder (or back to root when `None`),
    /// preventing circular nesting into itself or its own descendants.
    pub fn move_virtual_folder(
        &mut self,
        folder_id: &VirtualFolderId,
        next_parent_id: Option<VirtualFolderId>,
    ) -> Result<(), DomainError> {
        let current_name = self
            .virtual_folders
            .iter()
            .find(|f| &f.id == folder_id)
            .ok_or(DomainError::EmptyIdentifier("virtual folder not found"))?
            .name
            .clone();

        if let Some(ref next_parent) = next_parent_id {
            if next_parent == folder_id {
                return Err(DomainError::EmptyIdentifier(
                    "cannot move a folder inside itself",
                ));
            }
            if !self.virtual_folders.iter().any(|f| &f.id == next_parent) {
                return Err(DomainError::EmptyIdentifier("target parent folder not found"));
            }
            if self.is_descendant_folder(next_parent, folder_id) {
                return Err(DomainError::EmptyIdentifier(
                    "cannot move a folder inside one of its own subfolders",
                ));
            }
        }

        if self.has_sibling_folder_name(&current_name, next_parent_id.as_ref(), Some(folder_id)) {
            return Err(DomainError::EmptyIdentifier(
                "a folder with that name already exists in the destination",
            ));
        }

        let folder = self
            .virtual_folders
            .iter_mut()
            .find(|f| &f.id == folder_id)
            .expect("folder verified above");
        folder.parent_id = next_parent_id;
        Ok(())
    }

    /// Deletes a virtual folder and all nested subfolders, returning any photos inside them to
    /// Unassigned (never deleting actual photo files). Returns the number of unassigned photos.
    pub fn delete_virtual_folder(
        &mut self,
        folder_id: &VirtualFolderId,
    ) -> Result<usize, DomainError> {
        if !self.virtual_folders.iter().any(|f| &f.id == folder_id) {
            return Err(DomainError::EmptyIdentifier("virtual folder not found"));
        }

        let mut removed: HashSet<VirtualFolderId> = HashSet::new();
        removed.insert(folder_id.clone());
        loop {
            let before = removed.len();
            for folder in &self.virtual_folders {
                if let Some(ref pid) = folder.parent_id
                    && removed.contains(pid)
                {
                    removed.insert(folder.id.clone());
                }
            }
            if removed.len() == before {
                break;
            }
        }

        self.virtual_folders.retain(|f| !removed.contains(&f.id));
        let before_photos = self.photo_folders.len();
        self.photo_folders.retain(|_, fid| !removed.contains(fid));
        Ok(before_photos - self.photo_folders.len())
    }

    /// Assigns ONLY the specified `photo_ids` to `folder_id` (or removes their virtual folder
    /// assignment when `folder_id` is `None`). Returns the number of photos updated.
    pub fn assign_photos_to_folder(
        &mut self,
        photo_ids: &[PhotoId],
        folder_id: Option<&VirtualFolderId>,
    ) -> Result<usize, DomainError> {
        if photo_ids.is_empty() {
            return Err(DomainError::EmptyIdentifier("photo_ids"));
        }
        if let Some(fid) = folder_id
            && !self.virtual_folders.iter().any(|f| &f.id == fid)
        {
            return Err(DomainError::EmptyIdentifier("target virtual folder not found"));
        }

        let mut touched = 0usize;
        match folder_id {
            Some(fid) => {
                for pid in photo_ids {
                    self.photo_folders.insert(pid.clone(), fid.clone());
                    touched += 1;
                }
            }
            None => {
                for pid in photo_ids {
                    if self.photo_folders.remove(pid).is_some() {
                        touched += 1;
                    }
                }
            }
        }
        Ok(touched)
    }

    /// Returns `true` if `candidate_child` sits anywhere below `ancestor` in the folder tree.
    pub fn is_descendant_folder(
        &self,
        candidate_child: &VirtualFolderId,
        ancestor: &VirtualFolderId,
    ) -> bool {
        let mut cursor = Some(candidate_child);
        let mut hops = 0usize;
        while let Some(id) = cursor {
            if hops > self.virtual_folders.len() {
                break;
            }
            hops += 1;
            let Some(folder) = self.virtual_folders.iter().find(|f| &f.id == id) else {
                break;
            };
            match folder.parent_id.as_ref() {
                Some(parent) if parent == ancestor => return true,
                Some(parent) => cursor = Some(parent),
                None => break,
            }
        }
        false
    }

    /// Resolves ordered path segments from root down to `folder_id` (e.g. `["Engineering", "BSARC"]`).
    pub fn resolve_folder_path_segments(&self, folder_id: &VirtualFolderId) -> Vec<String> {
        let mut segments = Vec::new();
        let mut cursor = Some(folder_id);
        let mut seen: HashSet<&VirtualFolderId> = HashSet::new();
        while let Some(id) = cursor {
            if !seen.insert(id) {
                break;
            }
            let Some(folder) = self.virtual_folders.iter().find(|f| &f.id == id) else {
                break;
            };
            segments.push(folder.name.clone());
            cursor = folder.parent_id.as_ref();
        }
        segments.reverse();
        segments
    }

    /// Resolves the virtual folder path segments assigned to `photo_id` (empty when unassigned).
    pub fn photo_folder_segments(&self, photo_id: &PhotoId) -> Vec<String> {
        match self.photo_folders.get(photo_id) {
            Some(fid) => self.resolve_folder_path_segments(fid),
            None => Vec::new(),
        }
    }

    fn has_sibling_folder_name(
        &self,
        name: &str,
        parent_id: Option<&VirtualFolderId>,
        exclude_id: Option<&VirtualFolderId>,
    ) -> bool {
        let needle = name.to_ascii_lowercase();
        self.virtual_folders.iter().any(|f| {
            if exclude_id == Some(&f.id) {
                return false;
            }
            f.parent_id.as_ref() == parent_id && f.name.to_ascii_lowercase() == needle
        })
    }
    /// Approves a borderline match from `ambiguous_review_queue` and appends it to its candidate cluster.
    pub fn approve_ambiguous_match(&mut self, photo_id: &PhotoId) -> Result<(), DomainError> {
        let Some(pos) = self
            .ambiguous_review_queue
            .iter()
            .position(|c| &c.photo_id == photo_id)
        else {
            return Err(DomainError::EmptyIdentifier("ambiguous photo_id not found"));
        };

        let candidate = self.ambiguous_review_queue.remove(pos);
        let Some(cluster) = self
            .clusters
            .iter_mut()
            .find(|c| c.id == candidate.candidate_cluster_id)
        else {
            self.ambiguous_review_queue.insert(pos, candidate);
            return Err(DomainError::EmptyIdentifier("candidate cluster not found"));
        };

        cluster.entries.push(ClusterPhotoEntry {
            photo_id: candidate.photo_id,
            source_path: candidate.source_path,
            cosine_similarity: candidate.cosine_similarity,
            confidence: MatchConfidence::Verified,
            face_bounding_box: candidate.face_bounding_box,
            landmarks: candidate.landmarks,
            is_slate_photo: false,
        });
        Ok(())
    }

    /// Splits a borderline match out of `ambiguous_review_queue` into a new Face-Only subject cluster.
    pub fn split_ambiguous_to_new_cluster(
        &mut self,
        photo_id: &PhotoId,
    ) -> Result<ClusterId, DomainError> {
        let Some(pos) = self
            .ambiguous_review_queue
            .iter()
            .position(|c| &c.photo_id == photo_id)
        else {
            return Err(DomainError::EmptyIdentifier("ambiguous photo_id not found"));
        };

        let candidate = self.ambiguous_review_queue.remove(pos);
        let mut next_number = self.clusters.len() + 1;
        while self
            .clusters
            .iter()
            .any(|c| c.id.as_str() == format!("person_{next_number:03}"))
        {
            next_number += 1;
        }
        let cluster_id = ClusterId::new(format!("person_{next_number:03}"))?;
        let display_name = format!("Person {next_number:03}");
        let subject_code = Some(self.next_available_subject_code());

        let mut placeholder_raw = [0.0_f32; SFACE_EMBEDDING_DIM];
        placeholder_raw[0] = 1.0;
        let placeholder_embedding = FaceEmbedding::from_raw(placeholder_raw)?;

        let face = DetectedFace {
            photo_id: candidate.photo_id,
            bounding_box: candidate.face_bounding_box,
            landmarks: candidate.landmarks,
            detection_confidence: 1.0,
            embedding: placeholder_embedding,
        };

        let new_cluster = SubjectCluster::new_from_face(
            cluster_id.clone(),
            display_name,
            subject_code,
            face,
            candidate.source_path,
            None,
        )?;
        self.clusters.push(new_cluster);
        Ok(cluster_id)
    }

    /// Merges all photos from `source_id` into `target_id` and removes `source_id`.
    pub fn merge_clusters(
        &mut self,
        source_id: &ClusterId,
        target_id: &ClusterId,
    ) -> Result<(), DomainError> {
        if source_id == target_id {
            return Err(DomainError::EmptyIdentifier(
                "cannot merge a cluster into itself",
            ));
        }

        let Some(src_pos) = self.clusters.iter().position(|c| &c.id == source_id) else {
            return Err(DomainError::EmptyIdentifier("source cluster not found"));
        };
        if !self.clusters.iter().any(|c| &c.id == target_id) {
            return Err(DomainError::EmptyIdentifier("target cluster not found"));
        }

        let source = self.clusters.remove(src_pos);
        let target = self
            .clusters
            .iter_mut()
            .find(|c| &c.id == target_id)
            .expect("target cluster verified above");

        target.entries.extend(source.entries);
        for candidate in &mut self.ambiguous_review_queue {
            if &candidate.candidate_cluster_id == source_id {
                candidate.candidate_cluster_id = target_id.clone();
            }
        }
        Ok(())
    }

    /// Splits the selected `photo_ids` out of their current cluster(s) into a new Face-Only cluster.
    pub fn split_photos_to_new_cluster(
        &mut self,
        photo_ids: &[PhotoId],
    ) -> Result<ClusterId, DomainError> {
        if photo_ids.is_empty() {
            return Err(DomainError::EmptyIdentifier("photo_ids"));
        }

        let mut extracted: Vec<ClusterPhotoEntry> = Vec::new();
        for cluster in &mut self.clusters {
            let mut retained = Vec::with_capacity(cluster.entries.len());
            for entry in cluster.entries.drain(..) {
                if photo_ids.contains(&entry.photo_id) {
                    extracted.push(entry);
                } else {
                    retained.push(entry);
                }
            }
            cluster.entries = retained;
            if let Some(first) = cluster.entries.first()
                && !cluster
                    .entries
                    .iter()
                    .any(|e| e.photo_id == cluster.representative_photo_id)
            {
                cluster.representative_photo_id = first.photo_id.clone();
                cluster.representative_source_path = first.source_path.clone();
            }
        }

        // Also extract any selected photos sitting in `ambiguous_review_queue` so multi-selecting
        // review candidates can split all of them together into one single new subject cluster.
        let mut kept_ambiguous = Vec::with_capacity(self.ambiguous_review_queue.len());
        for cand in self.ambiguous_review_queue.drain(..) {
            if photo_ids.contains(&cand.photo_id) {
                extracted.push(ClusterPhotoEntry {
                    photo_id: cand.photo_id,
                    source_path: cand.source_path,
                    cosine_similarity: 1.0,
                    confidence: MatchConfidence::Verified,
                    face_bounding_box: cand.face_bounding_box,
                    landmarks: cand.landmarks,
                    is_slate_photo: false,
                });
            } else {
                kept_ambiguous.push(cand);
            }
        }
        self.ambiguous_review_queue = kept_ambiguous;

        if extracted.is_empty() {
            return Err(DomainError::EmptyIdentifier(
                "none of the selected photos exist in active clusters or review queue",
            ));
        }

        self.clusters.retain(|c| !c.entries.is_empty());

        let mut next_number = self.clusters.len() + 1;
        while self
            .clusters
            .iter()
            .any(|c| c.id.as_str() == format!("person_{next_number:03}"))
        {
            next_number += 1;
        }
        let cluster_id = ClusterId::new(format!("person_{next_number:03}"))?;
        let display_name = format!("Person {next_number:03}");
        let subject_code = Some(self.next_available_subject_code());

        let first = extracted.remove(0);
        let mut placeholder_raw = [0.0_f32; SFACE_EMBEDDING_DIM];
        placeholder_raw[0] = 1.0;
        let placeholder_embedding = FaceEmbedding::from_raw(placeholder_raw)?;

        let first_face = DetectedFace {
            photo_id: first.photo_id,
            bounding_box: first.face_bounding_box,
            landmarks: first.landmarks,
            detection_confidence: 1.0,
            embedding: placeholder_embedding,
        };

        let mut new_cluster = SubjectCluster::new_from_face(
            cluster_id.clone(),
            display_name,
            subject_code,
            first_face,
            first.source_path,
            None,
        )?;
        for mut entry in extracted {
            entry.is_slate_photo = false;
            entry.cosine_similarity = 1.0;
            entry.confidence = MatchConfidence::Verified;
            new_cluster.entries.push(entry);
        }

        self.clusters.push(new_cluster);
        Ok(cluster_id)
    }

    /// Manually moves/matches one or more photos from any source cluster, review queue,
    /// group bucket, or no-face bucket directly into `target_cluster_id` as verified matches.
    /// Empty source clusters are pruned automatically.
    pub fn move_photos_to_cluster(
        &mut self,
        photo_ids: &[PhotoId],
        target_cluster_id: &ClusterId,
    ) -> Result<usize, DomainError> {
        if photo_ids.is_empty() {
            return Err(DomainError::EmptyIdentifier("photo_ids"));
        }
        if !self.clusters.iter().any(|c| &c.id == target_cluster_id) {
            return Err(DomainError::EmptyIdentifier("target cluster not found"));
        }

        let mut moved_entries: Vec<ClusterPhotoEntry> = Vec::new();
        let mut removed_cluster_ids: Vec<ClusterId> = Vec::new();

        for cluster in &mut self.clusters {
            if &cluster.id == target_cluster_id {
                continue;
            }
            let mut retained = Vec::with_capacity(cluster.entries.len());
            for mut entry in cluster.entries.drain(..) {
                if photo_ids.contains(&entry.photo_id) {
                    entry.cosine_similarity = 1.0;
                    entry.confidence = MatchConfidence::Verified;
                    entry.is_slate_photo = false;
                    moved_entries.push(entry);
                } else {
                    retained.push(entry);
                }
            }
            cluster.entries = retained;
            if let Some(first) = cluster.entries.first()
                && !cluster
                    .entries
                    .iter()
                    .any(|e| e.photo_id == cluster.representative_photo_id)
            {
                cluster.representative_photo_id = first.photo_id.clone();
                cluster.representative_source_path = first.source_path.clone();
            }
            if cluster.entries.is_empty() {
                removed_cluster_ids.push(cluster.id.clone());
            }
        }

        // Also pull from ambiguous review queue if any selected photo lives there
        let mut kept_ambiguous = Vec::with_capacity(self.ambiguous_review_queue.len());
        for cand in self.ambiguous_review_queue.drain(..) {
            if photo_ids.contains(&cand.photo_id) {
                moved_entries.push(ClusterPhotoEntry {
                    photo_id: cand.photo_id,
                    source_path: cand.source_path,
                    cosine_similarity: 1.0,
                    confidence: MatchConfidence::Verified,
                    face_bounding_box: cand.face_bounding_box,
                    landmarks: cand.landmarks,
                    is_slate_photo: false,
                });
            } else {
                kept_ambiguous.push(cand);
            }
        }
        self.ambiguous_review_queue = kept_ambiguous;

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

        // Also support manually assigning from Group Photos or No-Face buckets
        let mut kept_groups = Vec::with_capacity(self.group_photos.len());
        for group in self.group_photos.drain(..) {
            if photo_ids.contains(&group.photo_id) {
                moved_entries.push(ClusterPhotoEntry {
                    photo_id: group.photo_id,
                    source_path: group.source_path,
                    cosine_similarity: 1.0,
                    confidence: MatchConfidence::Verified,
                    face_bounding_box: neutral_box,
                    landmarks: neutral_landmarks,
                    is_slate_photo: false,
                });
            } else {
                kept_groups.push(group);
            }
        }
        self.group_photos = kept_groups;

        let mut kept_no_face = Vec::with_capacity(self.no_face_photos.len());
        for nf in self.no_face_photos.drain(..) {
            if photo_ids.contains(&nf.photo_id) {
                moved_entries.push(ClusterPhotoEntry {
                    photo_id: nf.photo_id,
                    source_path: nf.source_path,
                    cosine_similarity: 1.0,
                    confidence: MatchConfidence::Verified,
                    face_bounding_box: neutral_box,
                    landmarks: neutral_landmarks,
                    is_slate_photo: false,
                });
            } else {
                kept_no_face.push(nf);
            }
        }
        self.no_face_photos = kept_no_face;

        if moved_entries.is_empty() {
            return Err(DomainError::EmptyIdentifier(
                "no movable photos found outside the target cluster",
            ));
        }

        let moved_count = moved_entries.len();
        self.clusters.retain(|c| !c.entries.is_empty());

        for cand in &mut self.ambiguous_review_queue {
            if removed_cluster_ids.contains(&cand.candidate_cluster_id) {
                cand.candidate_cluster_id = target_cluster_id.clone();
            }
        }

        let target = self
            .clusters
            .iter_mut()
            .find(|c| &c.id == target_cluster_id)
            .expect("target cluster verified above");
        target.entries.extend(moved_entries);

        Ok(moved_count)
    }

    /// Reconciles a restored organization plan against the photos currently present on disk.
    /// - Drops photos deleted from disk and prunes clusters that become empty.
    /// - Updates `source_path` on every surviving entry (handles renamed or moved project folders).
    /// - Re-points deleted representative photos to the first surviving entry in the cluster.
    /// - Appends newly added photos on disk into `no_face_photos` (`_Unassigned`) so nothing is lost.
    /// - Returns `None` if no meaningful organized structure survived reconciliation.
    pub fn reconcile_with_disk(
        self,
        discovered_photos: &[(PhotoId, String)],
    ) -> Option<ShootOrganizationPlan> {
        use std::collections::{HashMap, HashSet};

        if discovered_photos.is_empty() {
            return None;
        }

        let current_paths: HashMap<&PhotoId, &str> = discovered_photos
            .iter()
            .map(|(id, path)| (id, path.as_str()))
            .collect();
        let mut accounted: HashSet<PhotoId> = HashSet::with_capacity(discovered_photos.len());

        let mut surviving_clusters: Vec<SubjectCluster> = Vec::with_capacity(self.clusters.len());
        let mut surviving_cluster_ids: HashSet<ClusterId> =
            HashSet::with_capacity(self.clusters.len());

        for mut cluster in self.clusters {
            let mut retained_entries = Vec::with_capacity(cluster.entries.len());
            for mut entry in cluster.entries {
                if let Some(&fresh_path) = current_paths.get(&entry.photo_id)
                    && accounted.insert(entry.photo_id.clone())
                {
                    entry.source_path = fresh_path.to_string();
                    retained_entries.push(entry);
                }
            }
            if retained_entries.is_empty() {
                continue;
            }
            if let Some(rep) = retained_entries
                .iter()
                .find(|e| e.photo_id == cluster.representative_photo_id)
            {
                cluster.representative_source_path = rep.source_path.clone();
            } else {
                cluster.representative_photo_id = retained_entries[0].photo_id.clone();
                cluster.representative_source_path = retained_entries[0].source_path.clone();
            }
            cluster.entries = retained_entries;
            surviving_cluster_ids.insert(cluster.id.clone());
            surviving_clusters.push(cluster);
        }

        let mut surviving_no_face: Vec<NoFacePhotoEntry> = Vec::new();

        let mut surviving_ambiguous: Vec<AmbiguousMatchCandidate> =
            Vec::with_capacity(self.ambiguous_review_queue.len());
        for mut cand in self.ambiguous_review_queue {
            if let Some(&fresh_path) = current_paths.get(&cand.photo_id)
                && accounted.insert(cand.photo_id.clone())
            {
                cand.source_path = fresh_path.to_string();
                if surviving_cluster_ids.contains(&cand.candidate_cluster_id) {
                    surviving_ambiguous.push(cand);
                } else {
                    surviving_no_face.push(NoFacePhotoEntry {
                        photo_id: cand.photo_id,
                        source_path: cand.source_path,
                    });
                }
            }
        }

        let mut surviving_groups: Vec<GroupPhotoEntry> =
            Vec::with_capacity(self.group_photos.len());
        for mut group in self.group_photos {
            if let Some(&fresh_path) = current_paths.get(&group.photo_id)
                && accounted.insert(group.photo_id.clone())
            {
                group.source_path = fresh_path.to_string();
                group
                    .matched_cluster_ids
                    .retain(|cid| surviving_cluster_ids.contains(cid));
                surviving_groups.push(group);
            }
        }

        for mut nf in self.no_face_photos {
            if let Some(&fresh_path) = current_paths.get(&nf.photo_id)
                && accounted.insert(nf.photo_id.clone())
            {
                nf.source_path = fresh_path.to_string();
                surviving_no_face.push(nf);
            }
        }

        if surviving_clusters.is_empty()
            && surviving_ambiguous.is_empty()
            && surviving_groups.is_empty()
        {
            return None;
        }

        // Any newly added photos on disk since last save go cleanly into `_Unassigned`
        for (pid, path) in discovered_photos {
            if accounted.insert(pid.clone()) {
                surviving_no_face.push(NoFacePhotoEntry {
                    photo_id: pid.clone(),
                    source_path: path.clone(),
                });
            }
        }

        let surviving_folder_ids: HashSet<VirtualFolderId> = self
            .virtual_folders
            .iter()
            .map(|f| f.id.clone())
            .collect();
        let surviving_photo_folders: BTreeMap<PhotoId, VirtualFolderId> = self
            .photo_folders
            .into_iter()
            .filter(|(pid, fid)| accounted.contains(pid) && surviving_folder_ids.contains(fid))
            .collect();

        let mut reconciled = ShootOrganizationPlan {
            clusters: surviving_clusters,
            ambiguous_review_queue: surviving_ambiguous,
            group_photos: surviving_groups,
            no_face_photos: surviving_no_face,
            virtual_folders: self.virtual_folders,
            photo_folders: surviving_photo_folders,
        };
        reconciled.ensure_generated_subject_codes();
        Some(reconciled)
    }

    /// Returns the next unused sequential `#001`, `#002`, ... slate/subject code in this plan.
    pub fn next_available_subject_code(&self) -> String {
        let mut used_numbers: HashSet<usize> = HashSet::new();
        for cluster in &self.clusters {
            if let Some(code) = cluster.subject_code.as_deref()
                && let Some(num) = parse_generated_subject_number(code)
            {
                used_numbers.insert(num);
            }
        }
        let mut candidate = 1usize;
        while used_numbers.contains(&candidate) {
            candidate += 1;
        }
        format!("#{candidate:03}")
    }

    /// Ensures every subject cluster has a unique sequential `#001`, `#002`, ... code without
    /// overwriting existing generated or manually customized codes.
    pub fn ensure_generated_subject_codes(&mut self) {
        let mut used_numbers: HashSet<usize> = HashSet::new();
        for cluster in &self.clusters {
            if let Some(code) = cluster.subject_code.as_deref()
                && let Some(num) = parse_generated_subject_number(code)
            {
                used_numbers.insert(num);
            }
        }

        let mut next = 1usize;
        for cluster in &mut self.clusters {
            let has_real_code = cluster
                .subject_code
                .as_deref()
                .map(|s| {
                    let trimmed = s.trim();
                    !trimmed.is_empty() && trimmed != "#000" && trimmed != "000"
                })
                .unwrap_or(false);
            if has_real_code {
                continue;
            }
            while used_numbers.contains(&next) {
                next += 1;
            }
            used_numbers.insert(next);
            cluster.subject_code = Some(format!("#{next:03}"));
            next += 1;
        }
    }
}

fn parse_generated_subject_number(code: &str) -> Option<usize> {
    let digits = code.trim().trim_start_matches('#');
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    digits.parse::<usize>().ok().filter(|&n| n > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_landmarks() -> YuNetLandmarks {
        YuNetLandmarks {
            right_eye: Point2D { x: 0.35, y: 0.4 },
            left_eye: Point2D { x: 0.65, y: 0.4 },
            nose_tip: Point2D { x: 0.5, y: 0.55 },
            right_mouth_corner: Point2D { x: 0.4, y: 0.7 },
            left_mouth_corner: Point2D { x: 0.6, y: 0.7 },
        }
    }

    fn sample_box() -> BoundingBox {
        BoundingBox {
            x: 0.2,
            y: 0.2,
            width: 0.4,
            height: 0.4,
        }
    }

    #[test]
    fn ambiguous_matches_can_be_approved_or_split_into_face_only_clusters() {
        let mut raw = [0.0_f32; SFACE_EMBEDDING_DIM];
        raw[0] = 1.0;
        let anchor_face = DetectedFace {
            photo_id: PhotoId::new("p1").unwrap(),
            bounding_box: sample_box(),
            landmarks: sample_landmarks(),
            detection_confidence: 0.99,
            embedding: FaceEmbedding::from_raw(raw).unwrap(),
        };
        let cluster = SubjectCluster::new_from_face(
            ClusterId::new("person_001").unwrap(),
            "Person 001",
            None,
            anchor_face,
            "/shoot/p1.arw",
            None,
        )
        .unwrap();

        let mut plan = ShootOrganizationPlan {
            clusters: vec![cluster],
            ambiguous_review_queue: vec![
                AmbiguousMatchCandidate {
                    photo_id: PhotoId::new("p2").unwrap(),
                    source_path: "/shoot/p2.arw".to_string(),
                    candidate_cluster_id: ClusterId::new("person_001").unwrap(),
                    cosine_similarity: 0.41,
                    face_bounding_box: sample_box(),
                    landmarks: sample_landmarks(),
                },
                AmbiguousMatchCandidate {
                    photo_id: PhotoId::new("p3").unwrap(),
                    source_path: "/shoot/p3.arw".to_string(),
                    candidate_cluster_id: ClusterId::new("person_001").unwrap(),
                    cosine_similarity: 0.38,
                    face_bounding_box: sample_box(),
                    landmarks: sample_landmarks(),
                },
            ],
            group_photos: vec![],
            no_face_photos: vec![],
            virtual_folders: vec![],
            photo_folders: BTreeMap::new(),
        };

        plan.approve_ambiguous_match(&PhotoId::new("p2").unwrap())
            .unwrap();
        assert_eq!(plan.clusters[0].entries.len(), 2);
        assert_eq!(plan.ambiguous_review_queue.len(), 1);

        let split_id = plan
            .split_ambiguous_to_new_cluster(&PhotoId::new("p3").unwrap())
            .unwrap();
        assert_eq!(split_id.as_str(), "person_002");
        assert_eq!(plan.clusters.len(), 2);
        assert_eq!(plan.clusters[1].display_name, "Person 002");
        assert!(plan.ambiguous_review_queue.is_empty());

        // Merge person_002 back into person_001, then split p2 out into a standalone cluster
        plan.merge_clusters(
            &ClusterId::new("person_002").unwrap(),
            &ClusterId::new("person_001").unwrap(),
        )
        .unwrap();
        assert_eq!(plan.clusters.len(), 1);
        assert_eq!(plan.clusters[0].entries.len(), 3);

        let newly_split = plan
            .split_photos_to_new_cluster(&[PhotoId::new("p2").unwrap()])
            .unwrap();
        assert_eq!(plan.clusters.len(), 2);
        assert_eq!(plan.clusters[0].entries.len(), 2);
        assert_eq!(plan.clusters[1].id, newly_split);
        assert_eq!(plan.clusters[1].entries.len(), 1);

        // Manually move p2 back into person_001 and verify empty source cluster is pruned!
        let moved = plan
            .move_photos_to_cluster(
                &[PhotoId::new("p2").unwrap()],
                &ClusterId::new("person_001").unwrap(),
            )
            .unwrap();
        assert_eq!(moved, 1);
        assert_eq!(plan.clusters.len(), 1);
        assert_eq!(plan.clusters[0].entries.len(), 3);
    }

    #[test]
    fn parses_filename_stems_camera_sequences_and_embedded_or_exif_timestamps() {
        let raw = parse_filename_capture_hints(r"D:\Shoot\RAW\DSC03512.ARW");
        let edited = parse_filename_capture_hints(r"D:\Shoot\EDITED\DSC03512_edited.jpg");
        assert_eq!(raw.canonical_stem, "dsc03512");
        assert_eq!(edited.canonical_stem, "dsc03512");
        assert_eq!(raw.camera_prefix.as_deref(), Some("dsc"));
        assert_eq!(raw.camera_sequence, Some(3512));
        assert_eq!(raw.filename_timestamp_epoch_sec, None);

        let ts_file = parse_filename_capture_hints("IMG_20260314_101402.jpg");
        let exif_ts = parse_exif_datetime_epoch_sec("2026:03:14 10:14:02");
        assert!(ts_file.filename_timestamp_epoch_sec.is_some());
        assert_eq!(ts_file.filename_timestamp_epoch_sec, exif_ts);

        let ts_dashed = parse_filename_capture_hints("2026-03-14_10-14-07_portrait.jpg");
        assert_eq!(
            ts_dashed.filename_timestamp_epoch_sec.unwrap() - exif_ts.unwrap(),
            5
        );
    }

    #[test]
    fn reconcile_with_disk_refreshes_moved_paths_heals_deletions_and_appends_new_photos() {
        let mut raw = [0.0_f32; SFACE_EMBEDDING_DIM];
        raw[0] = 1.0;
        let emb = FaceEmbedding::from_raw(raw).unwrap();

        let mut c1 = SubjectCluster::new_from_face(
            ClusterId::new("person_001").unwrap(),
            "Salas",
            None,
            DetectedFace {
                photo_id: PhotoId::new("p1").unwrap(),
                bounding_box: sample_box(),
                landmarks: sample_landmarks(),
                detection_confidence: 0.99,
                embedding: emb.clone(),
            },
            "D:\\Old\\p1.png",
            None,
        )
        .unwrap();
        c1.add_verified_match(
            DetectedFace {
                photo_id: PhotoId::new("p2").unwrap(),
                bounding_box: sample_box(),
                landmarks: sample_landmarks(),
                detection_confidence: 0.99,
                embedding: emb.clone(),
            },
            "D:\\Old\\p2.png",
            0.91,
            false,
        );

        let c2 = SubjectCluster::new_from_face(
            ClusterId::new("person_002").unwrap(),
            "Deleted Subject",
            None,
            DetectedFace {
                photo_id: PhotoId::new("p3").unwrap(),
                bounding_box: sample_box(),
                landmarks: sample_landmarks(),
                detection_confidence: 0.98,
                embedding: emb,
            },
            "D:\\Old\\p3.png",
            None,
        )
        .unwrap();

        let plan = ShootOrganizationPlan {
            clusters: vec![c1, c2],
            ambiguous_review_queue: vec![AmbiguousMatchCandidate {
                photo_id: PhotoId::new("p4").unwrap(),
                source_path: "D:\\Old\\p4.png".to_string(),
                candidate_cluster_id: ClusterId::new("person_002").unwrap(),
                cosine_similarity: 0.42,
                face_bounding_box: sample_box(),
                landmarks: sample_landmarks(),
            }],
            group_photos: vec![],
            no_face_photos: vec![],
            virtual_folders: vec![],
            photo_folders: BTreeMap::new(),
        };

        // Folder moved to E:\Moved; p1 (rep of c1) & p3 (all of c2) deleted; p5 added on disk!
        let discovered = vec![
            (PhotoId::new("p2").unwrap(), "E:\\Moved\\p2.png".to_string()),
            (PhotoId::new("p4").unwrap(), "E:\\Moved\\p4.png".to_string()),
            (PhotoId::new("p5").unwrap(), "E:\\Moved\\p5.png".to_string()),
        ];

        let reconciled = plan
            .reconcile_with_disk(&discovered)
            .expect("should keep surviving cluster");
        assert_eq!(reconciled.clusters.len(), 1);
        assert_eq!(reconciled.clusters[0].display_name, "Salas");
        assert_eq!(reconciled.clusters[0].representative_photo_id.as_str(), "p2");
        assert_eq!(
            reconciled.clusters[0].representative_source_path,
            "E:\\Moved\\p2.png"
        );
        // p4's candidate cluster (person_002) was deleted, so p4 + new p5 both land in no_face_photos
        assert!(reconciled.ambiguous_review_queue.is_empty());
        let unassigned_ids: Vec<&str> = reconciled
            .no_face_photos
            .iter()
            .map(|n| n.photo_id.as_str())
            .collect();
        assert_eq!(unassigned_ids, vec!["p4", "p5"]);

        // Completely mismatched folder yields None so loader falls back cleanly to unclustered
        let unmatched = reconciled.reconcile_with_disk(&[(
            PhotoId::new("other").unwrap(),
            "E:\\Other\\other.png".to_string(),
        )]);
        assert!(unmatched.is_none());
    }

    #[test]
    fn virtual_folder_tree_supports_nesting_cycle_guards_and_selective_photo_assignment() {
        let mut plan = ShootOrganizationPlan::default();
        let eng = plan
            .create_virtual_folder("College of Engineering", None)
            .expect("create root folder");
        let bsarc = plan
            .create_virtual_folder("BSARC", Some(eng.clone()))
            .expect("create nested department");
        let hardhat = plan
            .create_virtual_folder("Hardhat", Some(bsarc.clone()))
            .expect("create nested prop subfolder");

        // Duplicate sibling name rejected (case-insensitive)
        assert!(plan.create_virtual_folder("bsarc", Some(eng.clone())).is_err());

        // Circular parent move rejected
        assert!(plan.move_virtual_folder(&eng, Some(hardhat.clone())).is_err());
        assert!(plan.move_virtual_folder(&bsarc, Some(bsarc.clone())).is_err());

        let p1 = PhotoId::new("DSC001").unwrap();
        let p2 = PhotoId::new("DSC002").unwrap();
        plan.assign_photos_to_folder(std::slice::from_ref(&p1), Some(&hardhat))
            .expect("assign p1 only");
        assert_eq!(
            plan.photo_folder_segments(&p1),
            vec!["College of Engineering", "BSARC", "Hardhat"]
        );
        assert!(plan.photo_folder_segments(&p2).is_empty());

        // Renaming parent updates resolved path immediately
        plan.rename_virtual_folder(&eng, "Engineering").unwrap();
        assert_eq!(
            plan.photo_folder_segments(&p1),
            vec!["Engineering", "BSARC", "Hardhat"]
        );

        // Deleting a parent unassigns descendant photos safely without losing photo IDs
        let unassigned = plan.delete_virtual_folder(&bsarc).unwrap();
        assert_eq!(unassigned, 1);
        assert_eq!(plan.virtual_folders.len(), 1);
        assert!(plan.photo_folder_segments(&p1).is_empty());
    }

    #[test]
    fn split_photos_to_new_cluster_combines_multiple_selected_ambiguous_queue_items() {
        let mut raw = [0.0_f32; SFACE_EMBEDDING_DIM];
        raw[0] = 1.0;
        let anchor_face = DetectedFace {
            photo_id: PhotoId::new("p1").unwrap(),
            bounding_box: sample_box(),
            landmarks: sample_landmarks(),
            detection_confidence: 0.99,
            embedding: FaceEmbedding::from_raw(raw).unwrap(),
        };
        let cluster = SubjectCluster::new_from_face(
            ClusterId::new("person_001").unwrap(),
            "Person 001",
            None,
            anchor_face,
            "/shoot/p1.arw",
            None,
        )
        .unwrap();

        let mut plan = ShootOrganizationPlan {
            clusters: vec![cluster],
            ambiguous_review_queue: vec![
                AmbiguousMatchCandidate {
                    photo_id: PhotoId::new("p2").unwrap(),
                    source_path: "/shoot/p2.arw".to_string(),
                    candidate_cluster_id: ClusterId::new("person_001").unwrap(),
                    cosine_similarity: 0.39,
                    face_bounding_box: sample_box(),
                    landmarks: sample_landmarks(),
                },
                AmbiguousMatchCandidate {
                    photo_id: PhotoId::new("p3").unwrap(),
                    source_path: "/shoot/p3.arw".to_string(),
                    candidate_cluster_id: ClusterId::new("person_001").unwrap(),
                    cosine_similarity: 0.40,
                    face_bounding_box: sample_box(),
                    landmarks: sample_landmarks(),
                },
            ],
            group_photos: vec![],
            no_face_photos: vec![],
            virtual_folders: vec![],
            photo_folders: BTreeMap::new(),
        };

        let new_id = plan
            .split_photos_to_new_cluster(&[
                PhotoId::new("p2").unwrap(),
                PhotoId::new("p3").unwrap(),
            ])
            .expect("batch split ambiguous items");
        assert!(plan.ambiguous_review_queue.is_empty());
        let created = plan.clusters.iter().find(|c| c.id == new_id).expect("new cluster");
        assert_eq!(created.entries.len(), 2);
        assert!(created
            .entries
            .iter()
            .all(|e| (e.cosine_similarity - 1.0).abs() < 1e-6 && e.confidence == MatchConfidence::Verified));

        // Also ensure low-confidence entries already inside a cluster reset cosine_similarity to 1.0 when split together
        let cluster_one = plan.clusters.first_mut().expect("person_001");
        cluster_one.entries.push(ClusterPhotoEntry {
            photo_id: PhotoId::new("p4").unwrap(),
            source_path: "/shoot/p4.arw".to_string(),
            cosine_similarity: 0.66,
            confidence: MatchConfidence::Verified,
            face_bounding_box: sample_box(),
            landmarks: sample_landmarks(),
            is_slate_photo: false,
        });
        cluster_one.entries.push(ClusterPhotoEntry {
            photo_id: PhotoId::new("p5").unwrap(),
            source_path: "/shoot/p5.arw".to_string(),
            cosine_similarity: 0.68,
            confidence: MatchConfidence::Verified,
            face_bounding_box: sample_box(),
            landmarks: sample_landmarks(),
            is_slate_photo: false,
        });
        let split_id = plan
            .split_photos_to_new_cluster(&[
                PhotoId::new("p4").unwrap(),
                PhotoId::new("p5").unwrap(),
            ])
            .expect("batch split low-confidence cluster entries");
        let split_cluster = plan.clusters.iter().find(|c| c.id == split_id).expect("split cluster");
        assert_eq!(split_cluster.entries.len(), 2);
        assert!(split_cluster
            .entries
            .iter()
            .all(|e| (e.cosine_similarity - 1.0).abs() < 1e-6));
        assert!(split_cluster.subject_code.is_some());

        // Ensure generated codes replace missing or placeholder "#000" codes without duplicating existing ones
        plan.clusters[0].subject_code = Some("#000".to_string());
        plan.clusters[1].subject_code = Some("#002".to_string());
        plan.clusters[2].subject_code = None;
        plan.ensure_generated_subject_codes();
        assert_eq!(plan.clusters[0].subject_code.as_deref(), Some("#001"));
        assert_eq!(plan.clusters[1].subject_code.as_deref(), Some("#002"));
        assert_eq!(plan.clusters[2].subject_code.as_deref(), Some("#003"));
    }
}
