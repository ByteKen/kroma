use kroma_domain::identity::{MatchConfidence, ShootOrganizationPlan};
use serde::Serialize;

pub const CURRENT_MANIFEST_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Serialize)]
pub struct ShootManifestDto {
    pub schema_version: u32,
    pub total_clusters: usize,
    pub ambiguous_review_count: usize,
    pub group_photo_count: usize,
    pub no_face_photo_count: usize,
    pub clusters: Vec<ClusterManifestDto>,
    pub ambiguous_review_queue: Vec<AmbiguousManifestDto>,
    pub group_photos: Vec<GroupManifestDto>,
    pub no_face_photos: Vec<NoFaceManifestDto>,
}

#[derive(Debug, Serialize)]
pub struct ClusterManifestDto {
    pub id: String,
    pub display_name: String,
    pub subject_code: Option<String>,
    pub has_slate: bool,
    pub representative_photo_id: String,
    pub representative_source_path: String,
    pub photo_count: usize,
    pub entries: Vec<ClusterEntryManifestDto>,
}

#[derive(Debug, Serialize)]
pub struct ClusterEntryManifestDto {
    pub photo_id: String,
    pub source_path: String,
    pub cosine_similarity: f32,
    pub confidence: &'static str,
    pub is_slate_photo: bool,
}

#[derive(Debug, Serialize)]
pub struct AmbiguousManifestDto {
    pub photo_id: String,
    pub source_path: String,
    pub candidate_cluster_id: String,
    pub cosine_similarity: f32,
}

#[derive(Debug, Serialize)]
pub struct GroupManifestDto {
    pub photo_id: String,
    pub source_path: String,
    pub matched_cluster_ids: Vec<String>,
    pub detected_face_count: usize,
}

#[derive(Debug, Serialize)]
pub struct NoFaceManifestDto {
    pub photo_id: String,
    pub source_path: String,
}

impl From<&ShootOrganizationPlan> for ShootManifestDto {
    fn from(plan: &ShootOrganizationPlan) -> Self {
        Self {
            schema_version: CURRENT_MANIFEST_SCHEMA_VERSION,
            total_clusters: plan.clusters.len(),
            ambiguous_review_count: plan.ambiguous_review_queue.len(),
            group_photo_count: plan.group_photos.len(),
            no_face_photo_count: plan.no_face_photos.len(),
            clusters: plan
                .clusters
                .iter()
                .map(|c| ClusterManifestDto {
                    id: c.id.as_str().to_string(),
                    display_name: c.display_name.clone(),
                    subject_code: c.subject_code.clone(),
                    has_slate: c.has_slate(),
                    representative_photo_id: c.representative_photo_id.as_str().to_string(),
                    representative_source_path: c.representative_source_path.clone(),
                    photo_count: c.entries.len(),
                    entries: c
                        .entries
                        .iter()
                        .map(|e| ClusterEntryManifestDto {
                            photo_id: e.photo_id.as_str().to_string(),
                            source_path: e.source_path.clone(),
                            cosine_similarity: e.cosine_similarity,
                            confidence: match e.confidence {
                                MatchConfidence::Verified => "verified",
                                MatchConfidence::AmbiguousNeedsReview => "needs_review",
                                MatchConfidence::Unmatched => "unmatched",
                            },
                            is_slate_photo: e.is_slate_photo,
                        })
                        .collect(),
                })
                .collect(),
            ambiguous_review_queue: plan
                .ambiguous_review_queue
                .iter()
                .map(|a| AmbiguousManifestDto {
                    photo_id: a.photo_id.as_str().to_string(),
                    source_path: a.source_path.clone(),
                    candidate_cluster_id: a.candidate_cluster_id.as_str().to_string(),
                    cosine_similarity: a.cosine_similarity,
                })
                .collect(),
            group_photos: plan
                .group_photos
                .iter()
                .map(|g| GroupManifestDto {
                    photo_id: g.photo_id.as_str().to_string(),
                    source_path: g.source_path.clone(),
                    matched_cluster_ids: g
                        .matched_cluster_ids
                        .iter()
                        .map(|id| id.as_str().to_string())
                        .collect(),
                    detected_face_count: g.detected_face_count,
                })
                .collect(),
            no_face_photos: plan
                .no_face_photos
                .iter()
                .map(|n| NoFaceManifestDto {
                    photo_id: n.photo_id.as_str().to_string(),
                    source_path: n.source_path.clone(),
                })
                .collect(),
        }
    }
}
