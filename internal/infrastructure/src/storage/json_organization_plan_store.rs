use crate::storage::organization_plan_dto::OrganizationPlanFileDto;
use kroma_app::error::AppError;
use kroma_app::ports::organization_plan_store::OrganizationPlanStorePort;
use kroma_domain::identity::ShootOrganizationPlan;
use std::fs;
use std::path::PathBuf;

const PLAN_FILENAME: &str = "_kroma_project_plan.json";
const PLAN_TEMP_FILENAME: &str = "_kroma_project_plan.json.tmp";

/// Stores a project's non-destructive `ShootOrganizationPlan` inside `<project>/.kroma_sidecars/`
/// using write-temp + rename so sudden quits or switching projects never leave corrupted JSON.
#[derive(Debug, Clone)]
pub struct JsonOrganizationPlanStore {
    root_dir: PathBuf,
}

impl JsonOrganizationPlanStore {
    pub fn new(root_dir: impl Into<PathBuf>) -> Self {
        Self {
            root_dir: root_dir.into(),
        }
    }

    pub fn plan_path(&self) -> PathBuf {
        self.root_dir.join(PLAN_FILENAME)
    }

    fn temp_path(&self) -> PathBuf {
        self.root_dir.join(PLAN_TEMP_FILENAME)
    }
}

impl OrganizationPlanStorePort for JsonOrganizationPlanStore {
    fn load_plan(&self) -> Result<Option<ShootOrganizationPlan>, AppError> {
        let path = self.plan_path();
        if !path.exists() {
            return Ok(None);
        }

        let raw = fs::read_to_string(&path).map_err(|err| {
            AppError::StorageFailure(format!(
                "failed to read project organization plan '{}': {err}",
                path.display()
            ))
        })?;
        let dto: OrganizationPlanFileDto = serde_json::from_str(&raw).map_err(|err| {
            AppError::StorageFailure(format!(
                "failed to parse project organization plan '{}': {err}",
                path.display()
            ))
        })?;
        let plan = dto.into_domain().map_err(AppError::from)?;
        Ok(Some(plan))
    }

    fn save_plan(&self, plan: &ShootOrganizationPlan) -> Result<(), AppError> {
        fs::create_dir_all(&self.root_dir).map_err(|err| {
            AppError::StorageFailure(format!(
                "failed to create sidecar directory '{}': {err}",
                self.root_dir.display()
            ))
        })?;

        let target = self.plan_path();
        let temp = self.temp_path();
        let dto = OrganizationPlanFileDto::from(plan);
        let payload = serde_json::to_string_pretty(&dto).map_err(|err| {
            AppError::StorageFailure(format!("failed to serialize project plan: {err}"))
        })?;

        fs::write(&temp, payload).map_err(|err| {
            AppError::StorageFailure(format!(
                "failed to write temp project plan '{}': {err}",
                temp.display()
            ))
        })?;
        fs::rename(&temp, &target).map_err(|err| {
            AppError::StorageFailure(format!(
                "failed to commit project plan '{}': {err}",
                target.display()
            ))
        })?;
        Ok(())
    }

    fn clear_plan(&self) -> Result<(), AppError> {
        let path = self.plan_path();
        if path.exists() {
            fs::remove_file(&path).map_err(|err| {
                AppError::StorageFailure(format!(
                    "failed to remove project plan '{}': {err}",
                    path.display()
                ))
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kroma_domain::identity::{
        AmbiguousMatchCandidate, BoundingBox, ClusterId, DetectedFace, FaceEmbedding,
        GroupPhotoEntry, NoFacePhotoEntry, Point2D, SlateMetadata, SubjectCluster, YuNetLandmarks,
        SFACE_EMBEDDING_DIM,
    };
    use kroma_domain::photo::PhotoId;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("kroma_{label}_{nanos}"))
    }

    fn sample_landmarks() -> YuNetLandmarks {
        YuNetLandmarks {
            right_eye: Point2D { x: 0.35, y: 0.4 },
            left_eye: Point2D { x: 0.65, y: 0.4 },
            nose_tip: Point2D { x: 0.5, y: 0.55 },
            right_mouth_corner: Point2D { x: 0.4, y: 0.7 },
            left_mouth_corner: Point2D { x: 0.6, y: 0.7 },
        }
    }

    #[test]
    fn saves_reloads_overwrites_and_clears_organization_plan_atomically() {
        let dir = unique_temp_dir("org_plan_store");
        let store = JsonOrganizationPlanStore::new(&dir);
        assert!(store.load_plan().unwrap().is_none());

        let mut raw = [0.0_f32; SFACE_EMBEDDING_DIM];
        raw[0] = 1.0;
        let emb = FaceEmbedding::from_raw(raw).unwrap();
        let face = DetectedFace {
            photo_id: PhotoId::new("DSC0001").unwrap(),
            bounding_box: BoundingBox {
                x: 0.2,
                y: 0.2,
                width: 0.4,
                height: 0.4,
            },
            landmarks: sample_landmarks(),
            detection_confidence: 0.98,
            embedding: emb.clone(),
        };
        let mut cluster = SubjectCluster::new_from_face(
            ClusterId::new("person_001").unwrap(),
            "Salas, Khnt Jr.",
            Some("#001".to_string()),
            face,
            "D:\\Projects\\Grad\\DSC0001.png",
            Some(SlateMetadata {
                slate_photo_id: PhotoId::new("DSC0001").unwrap(),
                ocr_raw_text: "SALAS, KHNT JR.".to_string(),
                ocr_confidence: 0.96,
                calibrated_white_balance_kelvin: Some(5400),
            }),
        )
        .unwrap();
        cluster.add_verified_match(
            DetectedFace {
                photo_id: PhotoId::new("DSC0002").unwrap(),
                bounding_box: BoundingBox {
                    x: 0.22,
                    y: 0.21,
                    width: 0.39,
                    height: 0.41,
                },
                landmarks: sample_landmarks(),
                detection_confidence: 0.97,
                embedding: emb,
            },
            "D:\\Projects\\Grad\\DSC0002.png",
            0.87,
            false,
        );

        let plan = ShootOrganizationPlan {
            clusters: vec![cluster],
            ambiguous_review_queue: vec![AmbiguousMatchCandidate {
                photo_id: PhotoId::new("DSC0003").unwrap(),
                source_path: "D:\\Projects\\Grad\\DSC0003.png".to_string(),
                candidate_cluster_id: ClusterId::new("person_001").unwrap(),
                cosine_similarity: 0.41,
                face_bounding_box: BoundingBox {
                    x: 0.1,
                    y: 0.1,
                    width: 0.3,
                    height: 0.3,
                },
                landmarks: sample_landmarks(),
            }],
            group_photos: vec![GroupPhotoEntry {
                photo_id: PhotoId::new("DSC0004").unwrap(),
                source_path: "D:\\Projects\\Grad\\DSC0004.png".to_string(),
                matched_cluster_ids: vec![ClusterId::new("person_001").unwrap()],
                detected_face_count: 2,
            }],
            no_face_photos: vec![NoFacePhotoEntry {
                photo_id: PhotoId::new("DSC0005").unwrap(),
                source_path: "D:\\Projects\\Grad\\DSC0005.png".to_string(),
            }],
            virtual_folders: Vec::new(),
            photo_folders: std::collections::BTreeMap::new(),
        };

        store.save_plan(&plan).unwrap();
        let reloaded = store.load_plan().unwrap().expect("plan must reload");
        assert_eq!(reloaded, plan);

        // Overwrite existing file atomically (important on Windows)
        let mut updated = reloaded;
        updated.clusters[0].rename("Bryle R.").unwrap();
        store.save_plan(&updated).unwrap();
        let reloaded_again = store.load_plan().unwrap().unwrap();
        assert_eq!(reloaded_again.clusters[0].display_name, "Bryle R.");

        store.clear_plan().unwrap();
        assert!(store.load_plan().unwrap().is_none());
        let _ = fs::remove_dir_all(&dir);
    }
}
