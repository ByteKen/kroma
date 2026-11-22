use kroma_domain::error::DomainError;
use kroma_domain::identity::{
    AmbiguousMatchCandidate, BoundingBox, ClusterId, ClusterPhotoEntry, FaceEmbedding,
    GroupPhotoEntry, MatchConfidence, NoFacePhotoEntry, Point2D, ShootOrganizationPlan,
    SlateMetadata, SubjectCluster, VirtualFolder, VirtualFolderId, YuNetLandmarks,
};
use kroma_domain::photo::PhotoId;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const CURRENT_ORGANIZATION_PLAN_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualFolderFileDto {
    pub id: String,
    pub name: String,
    pub parent_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganizationPlanFileDto {
    pub schema_version: u32,
    pub clusters: Vec<SubjectClusterDto>,
    pub ambiguous_review_queue: Vec<AmbiguousMatchCandidateDto>,
    pub group_photos: Vec<GroupPhotoEntryDto>,
    pub no_face_photos: Vec<NoFacePhotoEntryDto>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub virtual_folders: Vec<VirtualFolderFileDto>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub photo_folders: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point2DDto {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBoxDto {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YuNetLandmarksDto {
    pub right_eye: Point2DDto,
    pub left_eye: Point2DDto,
    pub nose_tip: Point2DDto,
    pub right_mouth_corner: Point2DDto,
    pub left_mouth_corner: Point2DDto,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlateMetadataDto {
    pub slate_photo_id: String,
    pub ocr_raw_text: String,
    pub ocr_confidence: f32,
    pub calibrated_white_balance_kelvin: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterPhotoEntryDto {
    pub photo_id: String,
    pub source_path: String,
    pub cosine_similarity: f32,
    pub confidence: String,
    pub face_bounding_box: BoundingBoxDto,
    pub landmarks: YuNetLandmarksDto,
    pub is_slate_photo: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectClusterDto {
    pub id: String,
    pub display_name: String,
    pub subject_code: Option<String>,
    pub representative_photo_id: String,
    pub representative_source_path: String,
    pub centroid_embedding: Vec<f32>,
    pub slate: Option<SlateMetadataDto>,
    pub entries: Vec<ClusterPhotoEntryDto>,
    pub verified_embedding_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbiguousMatchCandidateDto {
    pub photo_id: String,
    pub source_path: String,
    pub candidate_cluster_id: String,
    pub cosine_similarity: f32,
    pub face_bounding_box: BoundingBoxDto,
    pub landmarks: YuNetLandmarksDto,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupPhotoEntryDto {
    pub photo_id: String,
    pub source_path: String,
    pub matched_cluster_ids: Vec<String>,
    pub detected_face_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoFacePhotoEntryDto {
    pub photo_id: String,
    pub source_path: String,
}

impl From<Point2D> for Point2DDto {
    fn from(p: Point2D) -> Self {
        Self { x: p.x, y: p.y }
    }
}

impl From<Point2DDto> for Point2D {
    fn from(p: Point2DDto) -> Self {
        Self { x: p.x, y: p.y }
    }
}

impl From<BoundingBox> for BoundingBoxDto {
    fn from(b: BoundingBox) -> Self {
        Self {
            x: b.x,
            y: b.y,
            width: b.width,
            height: b.height,
        }
    }
}

impl From<BoundingBoxDto> for BoundingBox {
    fn from(b: BoundingBoxDto) -> Self {
        Self {
            x: b.x,
            y: b.y,
            width: b.width,
            height: b.height,
        }
    }
}

impl From<YuNetLandmarks> for YuNetLandmarksDto {
    fn from(l: YuNetLandmarks) -> Self {
        Self {
            right_eye: l.right_eye.into(),
            left_eye: l.left_eye.into(),
            nose_tip: l.nose_tip.into(),
            right_mouth_corner: l.right_mouth_corner.into(),
            left_mouth_corner: l.left_mouth_corner.into(),
        }
    }
}

impl From<YuNetLandmarksDto> for YuNetLandmarks {
    fn from(l: YuNetLandmarksDto) -> Self {
        Self {
            right_eye: l.right_eye.into(),
            left_eye: l.left_eye.into(),
            nose_tip: l.nose_tip.into(),
            right_mouth_corner: l.right_mouth_corner.into(),
            left_mouth_corner: l.left_mouth_corner.into(),
        }
    }
}

fn confidence_to_str(c: MatchConfidence) -> &'static str {
    match c {
        MatchConfidence::Verified => "verified",
        MatchConfidence::AmbiguousNeedsReview => "needs_review",
        MatchConfidence::Unmatched => "unmatched",
    }
}

fn str_to_confidence(s: &str) -> MatchConfidence {
    match s {
        "needs_review" => MatchConfidence::AmbiguousNeedsReview,
        "unmatched" => MatchConfidence::Unmatched,
        _ => MatchConfidence::Verified,
    }
}

impl From<&ShootOrganizationPlan> for OrganizationPlanFileDto {
    fn from(plan: &ShootOrganizationPlan) -> Self {
        Self {
            schema_version: CURRENT_ORGANIZATION_PLAN_SCHEMA_VERSION,
            clusters: plan
                .clusters
                .iter()
                .map(|c| SubjectClusterDto {
                    id: c.id.as_str().to_string(),
                    display_name: c.display_name.clone(),
                    subject_code: c.subject_code.clone(),
                    representative_photo_id: c.representative_photo_id.as_str().to_string(),
                    representative_source_path: c.representative_source_path.clone(),
                    centroid_embedding: c.centroid_embedding.as_slice().to_vec(),
                    slate: c.slate.as_ref().map(|s| SlateMetadataDto {
                        slate_photo_id: s.slate_photo_id.as_str().to_string(),
                        ocr_raw_text: s.ocr_raw_text.clone(),
                        ocr_confidence: s.ocr_confidence,
                        calibrated_white_balance_kelvin: s.calibrated_white_balance_kelvin,
                    }),
                    entries: c
                        .entries
                        .iter()
                        .map(|e| ClusterPhotoEntryDto {
                            photo_id: e.photo_id.as_str().to_string(),
                            source_path: e.source_path.clone(),
                            cosine_similarity: e.cosine_similarity,
                            confidence: confidence_to_str(e.confidence).to_string(),
                            face_bounding_box: e.face_bounding_box.into(),
                            landmarks: e.landmarks.into(),
                            is_slate_photo: e.is_slate_photo,
                        })
                        .collect(),
                    verified_embedding_count: c.verified_embedding_count(),
                })
                .collect(),
            ambiguous_review_queue: plan
                .ambiguous_review_queue
                .iter()
                .map(|a| AmbiguousMatchCandidateDto {
                    photo_id: a.photo_id.as_str().to_string(),
                    source_path: a.source_path.clone(),
                    candidate_cluster_id: a.candidate_cluster_id.as_str().to_string(),
                    cosine_similarity: a.cosine_similarity,
                    face_bounding_box: a.face_bounding_box.into(),
                    landmarks: a.landmarks.into(),
                })
                .collect(),
            group_photos: plan
                .group_photos
                .iter()
                .map(|g| GroupPhotoEntryDto {
                    photo_id: g.photo_id.as_str().to_string(),
                    source_path: g.source_path.clone(),
                    matched_cluster_ids: g
                        .matched_cluster_ids
                        .iter()
                        .map(|cid| cid.as_str().to_string())
                        .collect(),
                    detected_face_count: g.detected_face_count,
                })
                .collect(),
            no_face_photos: plan
                .no_face_photos
                .iter()
                .map(|n| NoFacePhotoEntryDto {
                    photo_id: n.photo_id.as_str().to_string(),
                    source_path: n.source_path.clone(),
                })
                .collect(),
            virtual_folders: plan
                .virtual_folders
                .iter()
                .map(|f| VirtualFolderFileDto {
                    id: f.id.as_str().to_string(),
                    name: f.name.clone(),
                    parent_id: f.parent_id.as_ref().map(|p| p.as_str().to_string()),
                })
                .collect(),
            photo_folders: plan
                .photo_folders
                .iter()
                .map(|(pid, fid)| (pid.as_str().to_string(), fid.as_str().to_string()))
                .collect(),
        }
    }
}

impl OrganizationPlanFileDto {
    pub fn into_domain(self) -> Result<ShootOrganizationPlan, DomainError> {
        let mut clusters = Vec::with_capacity(self.clusters.len());
        for c in self.clusters {
            let entries: Vec<ClusterPhotoEntry> = c
                .entries
                .into_iter()
                .map(|e| -> Result<ClusterPhotoEntry, DomainError> {
                    Ok(ClusterPhotoEntry {
                        photo_id: PhotoId::new(e.photo_id)?,
                        source_path: e.source_path,
                        cosine_similarity: e.cosine_similarity,
                        confidence: str_to_confidence(&e.confidence),
                        face_bounding_box: e.face_bounding_box.into(),
                        landmarks: e.landmarks.into(),
                        is_slate_photo: e.is_slate_photo,
                    })
                })
                .collect::<Result<_, _>>()?;
            if entries.is_empty() {
                continue;
            }
            let slate = match c.slate {
                Some(s) => Some(SlateMetadata {
                    slate_photo_id: PhotoId::new(s.slate_photo_id)?,
                    ocr_raw_text: s.ocr_raw_text,
                    ocr_confidence: s.ocr_confidence,
                    calibrated_white_balance_kelvin: s.calibrated_white_balance_kelvin,
                }),
                None => None,
            };
            let centroid_embedding = FaceEmbedding::from_vec(c.centroid_embedding)?;
            let cluster = SubjectCluster::restore(
                ClusterId::new(c.id)?,
                c.display_name,
                c.subject_code,
                PhotoId::new(c.representative_photo_id)?,
                c.representative_source_path,
                centroid_embedding,
                slate,
                entries,
                c.verified_embedding_count,
            )?;
            clusters.push(cluster);
        }

        let ambiguous_review_queue = self
            .ambiguous_review_queue
            .into_iter()
            .map(|a| -> Result<AmbiguousMatchCandidate, DomainError> {
                Ok(AmbiguousMatchCandidate {
                    photo_id: PhotoId::new(a.photo_id)?,
                    source_path: a.source_path,
                    candidate_cluster_id: ClusterId::new(a.candidate_cluster_id)?,
                    cosine_similarity: a.cosine_similarity,
                    face_bounding_box: a.face_bounding_box.into(),
                    landmarks: a.landmarks.into(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let group_photos = self
            .group_photos
            .into_iter()
            .map(|g| -> Result<GroupPhotoEntry, DomainError> {
                let matched_cluster_ids = g
                    .matched_cluster_ids
                    .into_iter()
                    .map(ClusterId::new)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(GroupPhotoEntry {
                    photo_id: PhotoId::new(g.photo_id)?,
                    source_path: g.source_path,
                    matched_cluster_ids,
                    detected_face_count: g.detected_face_count,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let no_face_photos = self
            .no_face_photos
            .into_iter()
            .map(|n| -> Result<NoFacePhotoEntry, DomainError> {
                Ok(NoFacePhotoEntry {
                    photo_id: PhotoId::new(n.photo_id)?,
                    source_path: n.source_path,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let virtual_folders = self
            .virtual_folders
            .into_iter()
            .map(|f| -> Result<VirtualFolder, DomainError> {
                let parent_id = match f.parent_id {
                    Some(pid) => Some(VirtualFolderId::new(pid)?),
                    None => None,
                };
                Ok(VirtualFolder {
                    id: VirtualFolderId::new(f.id)?,
                    name: f.name,
                    parent_id,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut photo_folders = BTreeMap::new();
        for (pid, fid) in self.photo_folders {
            photo_folders.insert(PhotoId::new(pid)?, VirtualFolderId::new(fid)?);
        }

        Ok(ShootOrganizationPlan {
            clusters,
            ambiguous_review_queue,
            group_photos,
            no_face_photos,
            virtual_folders,
            photo_folders,
        })
    }
}
