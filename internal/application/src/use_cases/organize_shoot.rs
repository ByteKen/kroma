use crate::error::AppError;
use crate::ports::develop_engine::{
    BatchRenderItemOutcome, BatchRenderJob, ImageDevelopPort, OutputImageFormat,
};
use crate::ports::organizer::{FileOrganizerPort, OrganizeTransferMode};
use crate::ports::recipe_store::DevelopRecipeStorePort;
use kroma_domain::identity::ShootOrganizationPlan;
use kroma_domain::organization::{sanitize_path_component, OutputNamingTemplate};
use kroma_domain::photo::PhotoId;
use std::collections::HashSet;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrganizeExecutionMode {
    /// Copies, moves, or hard-links the original RAW/JPEG files directly into subject folders.
    TransferOriginals(OrganizeTransferMode),
    /// Renders each photo through its saved Lightroom DevelopRecipe straight into organized subject folders.
    DevelopAndExport { format: OutputImageFormat },
    /// Saves BOTH the rendered edited photo (`Edited/`) AND the untouched original source file (`RAW/`)
    /// inside each organized subject folder.
    DevelopAndKeepRaw {
        format: OutputImageFormat,
        raw_transfer_mode: OrganizeTransferMode,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct OrganizeShootInput {
    pub plan: ShootOrganizationPlan,
    pub output_root: String,
    pub naming_template: OutputNamingTemplate,
    pub execution_mode: OrganizeExecutionMode,
    pub archive_excluded_slates: bool,
    pub write_manifest_json: bool,
    /// Optional selective export filter. When `Some` and non-empty, only matching photos are
    /// exported while preserving their canonical folder names and sequence numbers from `plan`.
    pub selected_photo_ids: Option<HashSet<PhotoId>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrganizedFileItem {
    pub photo_id: PhotoId,
    pub destination_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrganizeFileFailure {
    pub photo_id: PhotoId,
    pub source_path: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrganizeShootOutput {
    pub organized_files: Vec<OrganizedFileItem>,
    pub failures: Vec<OrganizeFileFailure>,
    pub manifest_path: Option<String>,
}

struct PlannedDeliveryItem {
    photo_id: PhotoId,
    source_path: String,
    destination_path: String,
    raw_companion_destination_path: Option<String>,
}

/// Organizes a clustered shoot into clean named subject folders (supporting both slate-named
/// and slate-free `Person_001` clusters), either by transferring originals OR by running the
/// Lightroom Batch Develop pipeline directly into each subject's delivery folder.
pub struct OrganizeShootFoldersUseCase<'a, O: FileOrganizerPort> {
    organizer: &'a O,
}

impl<'a, O: FileOrganizerPort> OrganizeShootFoldersUseCase<'a, O> {
    pub fn new(organizer: &'a O) -> Self {
        Self { organizer }
    }

    pub fn execute_with_optional_develop<E: ImageDevelopPort, S: DevelopRecipeStorePort>(
        &self,
        input: OrganizeShootInput,
        develop_engine: Option<&E>,
        recipe_store: Option<&S>,
    ) -> Result<OrganizeShootOutput, AppError> {
        self.execute_with_progress(input, develop_engine, recipe_store, &|_, _, _| {})
    }

    /// Executes folder organization + optional develop rendering while reporting live per-file
    /// completion `(completed_ops, total_ops, current_label)` to the caller.
    pub fn execute_with_progress<E: ImageDevelopPort, S: DevelopRecipeStorePort>(
        &self,
        input: OrganizeShootInput,
        develop_engine: Option<&E>,
        recipe_store: Option<&S>,
        on_progress: &(dyn Fn(usize, usize, &str) + Sync),
    ) -> Result<OrganizeShootOutput, AppError> {
        if input.output_root.trim().is_empty() {
            return Err(AppError::InvalidInput(
                "output_root directory cannot be empty".to_string(),
            ));
        }

        let deliveries = build_delivery_plan(&input);
        let mut organized_files = Vec::new();
        let mut failures = Vec::new();

        match input.execution_mode {
            OrganizeExecutionMode::TransferOriginals(transfer_mode) => {
                let total_ops = deliveries.len();
                on_progress(0, total_ops, "Preparing file transfers…");
                for (idx, item) in deliveries.into_iter().enumerate() {
                    let label = Path::new(&item.destination_path)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or(item.photo_id.as_str())
                        .to_string();
                    match self.organizer.transfer_file(
                        &item.source_path,
                        &item.destination_path,
                        transfer_mode,
                    ) {
                        Ok(()) => organized_files.push(OrganizedFileItem {
                            photo_id: item.photo_id,
                            destination_path: item.destination_path,
                        }),
                        Err(err) => failures.push(OrganizeFileFailure {
                            photo_id: item.photo_id,
                            source_path: item.source_path,
                            reason: err.to_string(),
                        }),
                    }
                    on_progress(idx + 1, total_ops, &label);
                }
            }
            OrganizeExecutionMode::DevelopAndExport { format } => {
                let engine = develop_engine.ok_or_else(|| {
                    AppError::InvalidInput(
                        "DevelopAndExport mode requires an ImageDevelopPort implementation"
                            .to_string(),
                    )
                })?;

                let mut jobs = Vec::with_capacity(deliveries.len());
                for item in deliveries {
                    let recipe = match recipe_store {
                        Some(store) => store.load_recipe(&item.photo_id)?.unwrap_or_default(),
                        None => Default::default(),
                    };
                    jobs.push(BatchRenderJob {
                        photo_id: item.photo_id,
                        source_path: item.source_path,
                        output_path: item.destination_path,
                        recipe,
                        format,
                    });
                }

                let total_ops = jobs.len();
                on_progress(0, total_ops, "Starting parallel photo export…");

                for outcome in engine.render_batch_with_progress(&jobs, &|done, total, label| {
                    on_progress(done, total, label);
                }) {
                    match outcome {
                        BatchRenderItemOutcome::Succeeded {
                            photo_id,
                            output_path,
                            ..
                        } => organized_files.push(OrganizedFileItem {
                            photo_id,
                            destination_path: output_path,
                        }),
                        BatchRenderItemOutcome::Failed {
                            photo_id,
                            source_path,
                            reason,
                        } => failures.push(OrganizeFileFailure {
                            photo_id,
                            source_path,
                            reason,
                        }),
                    }
                }
            }
            OrganizeExecutionMode::DevelopAndKeepRaw {
                format,
                raw_transfer_mode,
            } => {
                let engine = develop_engine.ok_or_else(|| {
                    AppError::InvalidInput(
                        "DevelopAndKeepRaw mode requires an ImageDevelopPort implementation"
                            .to_string(),
                    )
                })?;

                let mut jobs = Vec::with_capacity(deliveries.len());
                let mut raw_transfers = Vec::with_capacity(deliveries.len());
                for item in deliveries {
                    let recipe = match recipe_store {
                        Some(store) => store.load_recipe(&item.photo_id)?.unwrap_or_default(),
                        None => Default::default(),
                    };
                    if let Some(raw_dest) = item.raw_companion_destination_path {
                        raw_transfers.push((
                            item.photo_id.clone(),
                            item.source_path.clone(),
                            raw_dest,
                        ));
                    }
                    jobs.push(BatchRenderJob {
                        photo_id: item.photo_id,
                        source_path: item.source_path,
                        output_path: item.destination_path,
                        recipe,
                        format,
                    });
                }

                let render_count = jobs.len();
                let total_ops = render_count + raw_transfers.len();
                on_progress(0, total_ops, "Starting Edited + RAW export…");

                for outcome in engine.render_batch_with_progress(&jobs, &|done, _, label| {
                    on_progress(done, total_ops, label);
                }) {
                    match outcome {
                        BatchRenderItemOutcome::Succeeded {
                            photo_id,
                            output_path,
                            ..
                        } => organized_files.push(OrganizedFileItem {
                            photo_id,
                            destination_path: output_path,
                        }),
                        BatchRenderItemOutcome::Failed {
                            photo_id,
                            source_path,
                            reason,
                        } => failures.push(OrganizeFileFailure {
                            photo_id,
                            source_path,
                            reason,
                        }),
                    }
                }

                for (idx, (photo_id, source_path, raw_dest)) in raw_transfers.into_iter().enumerate()
                {
                    let label = Path::new(&raw_dest)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or(photo_id.as_str())
                        .to_string();
                    match self
                        .organizer
                        .transfer_file(&source_path, &raw_dest, raw_transfer_mode)
                    {
                        Ok(()) => organized_files.push(OrganizedFileItem {
                            photo_id,
                            destination_path: raw_dest,
                        }),
                        Err(err) => failures.push(OrganizeFileFailure {
                            photo_id,
                            source_path,
                            reason: err.to_string(),
                        }),
                    }
                    on_progress(render_count + idx + 1, total_ops, &label);
                }
            }
        }

        let manifest_path = if input.write_manifest_json {
            let total_done = organized_files.len() + failures.len();
            on_progress(total_done, total_done.max(1), "Writing shoot manifest…");
            let path = join_clean(&input.output_root, "kroma_shoot_manifest.json");
            self.organizer.write_plan_manifest(&path, &input.plan)?;
            Some(path)
        } else {
            None
        };

        Ok(OrganizeShootOutput {
            organized_files,
            failures,
            manifest_path,
        })
    }
}

fn build_delivery_plan(input: &OrganizeShootInput) -> Vec<PlannedDeliveryItem> {
    use std::collections::HashMap;

    let mut items = Vec::new();
    let target_ext = output_extension(input.execution_mode);
    let dual_raw_and_edited = matches!(
        input.execution_mode,
        OrganizeExecutionMode::DevelopAndKeepRaw { .. }
    );

    let template_includes_id = input.naming_template.pattern().contains("[ID]");
    let mut name_counts: HashMap<String, usize> = HashMap::new();
    for cluster in &input.plan.clusters {
        let key = sanitize_path_component(&cluster.display_name).to_ascii_lowercase();
        *name_counts.entry(key).or_insert(0) += 1;
    }

    for (cluster_idx, cluster) in input.plan.clusters.iter().enumerate() {
        let mut seq_by_folder: HashMap<String, usize> = HashMap::new();
        let name_key = sanitize_path_component(&cluster.display_name).to_ascii_lowercase();
        let needs_collision_suffix =
            !template_includes_id && name_counts.get(&name_key).copied().unwrap_or(0) > 1;
        let effective_subject_name = if needs_collision_suffix {
            let suffix = cluster
                .subject_code
                .as_deref()
                .map(sanitize_path_component)
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| format!("{:03}", cluster_idx + 1));
            format!("{}_{}", cluster.display_name, suffix)
        } else {
            cluster.display_name.clone()
        };

        for entry in &cluster.entries {
            if entry.is_slate_photo && input.naming_template.exclude_slate_photo_from_client_folder()
            {
                if input.archive_excluded_slates {
                    let ext = target_ext
                        .map(str::to_string)
                        .unwrap_or_else(|| source_extension_or(&entry.source_path, "jpg"));
                    let dest = join_clean(
                        &input.output_root,
                        &format!("_Slates_Archive/{}.{}", entry.photo_id.as_str(), ext),
                    );
                    items.push(PlannedDeliveryItem {
                        photo_id: entry.photo_id.clone(),
                        source_path: entry.source_path.clone(),
                        destination_path: dest,
                        raw_companion_destination_path: None,
                    });
                }
                continue;
            }

            let folder_segments = input.plan.photo_folder_segments(&entry.photo_id);
            let folder_key = folder_segments.join("/");
            let seq_entry = seq_by_folder.entry(folder_key).or_insert(0);
            *seq_entry += 1;
            let client_seq = *seq_entry;

            let orig_stem = Path::new(&entry.source_path)
                .file_stem()
                .and_then(|s| s.to_str());
            let raw_rel = input.naming_template.format_relative_path_for_file(
                &folder_segments,
                &effective_subject_name,
                cluster.subject_code.as_deref(),
                client_seq,
                orig_stem,
            );

            let ext = target_ext
                .map(str::to_string)
                .unwrap_or_else(|| source_extension_or(&entry.source_path, "jpg"));
            let adjusted_rel = replace_extension(raw_rel.trim_start_matches(['/', '\\']), &ext);

            if dual_raw_and_edited {
                let (edited_rel, raw_rel_dest) =
                    split_into_edited_and_raw_subdirs(&adjusted_rel, &entry.source_path);
                items.push(PlannedDeliveryItem {
                    photo_id: entry.photo_id.clone(),
                    source_path: entry.source_path.clone(),
                    destination_path: join_clean(&input.output_root, &edited_rel),
                    raw_companion_destination_path: Some(join_clean(
                        &input.output_root,
                        &raw_rel_dest,
                    )),
                });
            } else {
                items.push(PlannedDeliveryItem {
                    photo_id: entry.photo_id.clone(),
                    source_path: entry.source_path.clone(),
                    destination_path: join_clean(&input.output_root, &adjusted_rel),
                    raw_companion_destination_path: None,
                });
            }
        }
    }

    let mut group_seq_by_folder: HashMap<String, usize> = HashMap::new();
    for group in &input.plan.group_photos {
        let ext = target_ext
            .map(str::to_string)
            .unwrap_or_else(|| source_extension_or(&group.source_path, "jpg"));
        let folder_prefix = sanitized_folder_prefix(&input.plan.photo_folder_segments(&group.photo_id));
        let seq = group_seq_by_folder.entry(folder_prefix.clone()).or_insert(0);
        *seq += 1;
        let rel = if folder_prefix.is_empty() {
            format!("_Group_Shots/GROUP_{:03}.{}", *seq, ext)
        } else {
            format!("{folder_prefix}/_Group_Shots/GROUP_{:03}.{}", *seq, ext)
        };
        if dual_raw_and_edited {
            let (edited_rel, raw_rel_dest) =
                split_into_edited_and_raw_subdirs(&rel, &group.source_path);
            items.push(PlannedDeliveryItem {
                photo_id: group.photo_id.clone(),
                source_path: group.source_path.clone(),
                destination_path: join_clean(&input.output_root, &edited_rel),
                raw_companion_destination_path: Some(join_clean(
                    &input.output_root,
                    &raw_rel_dest,
                )),
            });
        } else {
            items.push(PlannedDeliveryItem {
                photo_id: group.photo_id.clone(),
                source_path: group.source_path.clone(),
                destination_path: join_clean(&input.output_root, &rel),
                raw_companion_destination_path: None,
            });
        }
    }

    let mut no_face_seq_by_folder: HashMap<String, usize> = HashMap::new();
    for item in &input.plan.no_face_photos {
        let ext = target_ext
            .map(str::to_string)
            .unwrap_or_else(|| source_extension_or(&item.source_path, "jpg"));
        let folder_prefix = sanitized_folder_prefix(&input.plan.photo_folder_segments(&item.photo_id));
        let seq = no_face_seq_by_folder.entry(folder_prefix.clone()).or_insert(0);
        *seq += 1;
        let rel = if folder_prefix.is_empty() {
            format!("_No_Face_Detected/IMG_{:03}.{}", *seq, ext)
        } else {
            format!("{folder_prefix}/_No_Face_Detected/IMG_{:03}.{}", *seq, ext)
        };
        if dual_raw_and_edited {
            let (edited_rel, raw_rel_dest) =
                split_into_edited_and_raw_subdirs(&rel, &item.source_path);
            items.push(PlannedDeliveryItem {
                photo_id: item.photo_id.clone(),
                source_path: item.source_path.clone(),
                destination_path: join_clean(&input.output_root, &edited_rel),
                raw_companion_destination_path: Some(join_clean(
                    &input.output_root,
                    &raw_rel_dest,
                )),
            });
        } else {
            items.push(PlannedDeliveryItem {
                photo_id: item.photo_id.clone(),
                source_path: item.source_path.clone(),
                destination_path: join_clean(&input.output_root, &rel),
                raw_companion_destination_path: None,
            });
        }
    }

    if let Some(ref selected) = input.selected_photo_ids
        && !selected.is_empty()
    {
        items.retain(|item| selected.contains(&item.photo_id));
    }

    items
}

fn sanitized_folder_prefix(segments: &[String]) -> String {
    segments
        .iter()
        .map(|seg| {
            seg.trim()
                .chars()
                .map(|c| {
                    if c.is_alphanumeric() || c == '-' || c == '_' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect::<String>()
                .trim_matches('_')
                .to_string()
        })
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("/")
}

fn split_into_edited_and_raw_subdirs(relative_path: &str, source_path: &str) -> (String, String) {
    let orig_name = Path::new(source_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("original.raw");
    let clean = relative_path.trim_start_matches(['/', '\\']);
    match clean.rsplit_once('/') {
        Some((dir, file)) => (format!("{dir}/Edited/{file}"), format!("{dir}/RAW/{orig_name}")),
        None => (format!("Edited/{clean}"), format!("RAW/{orig_name}")),
    }
}

fn output_extension(mode: OrganizeExecutionMode) -> Option<&'static str> {
    match mode {
        OrganizeExecutionMode::TransferOriginals(_) => None,
        OrganizeExecutionMode::DevelopAndExport { format }
        | OrganizeExecutionMode::DevelopAndKeepRaw { format, .. } => Some(match format {
            OutputImageFormat::Jpeg { .. } => "jpg",
            OutputImageFormat::Png => "png",
            OutputImageFormat::Tiff => "tiff",
        }),
    }
}

fn source_extension_or(source_path: &str, fallback: &str) -> String {
    Path::new(source_path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_else(|| fallback.to_string())
}

fn replace_extension(relative_path: &str, ext: &str) -> String {
    let p = Path::new(relative_path);
    match p.with_extension(ext).to_str() {
        Some(updated) => updated.replace('\\', "/"),
        None => relative_path.to_string(),
    }
}

fn join_clean(root: &str, relative: &str) -> String {
    let clean_root = root.trim_end_matches(['/', '\\']);
    let clean_rel = relative.trim_start_matches(['/', '\\']);
    format!("{clean_root}/{clean_rel}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ports::develop_engine::RenderMetrics;
    use crate::ports::vision::{
        FaceDetectorPort, FaceRecognizerPort, RawFaceDetection, SlateOcrReading, SlateReaderPort,
    };
    use crate::use_cases::cluster_shoot::{
        ClusterShootFacesUseCase, ClusterShootInput, PhotoScanTarget,
    };
    use kroma_domain::develop::{DevelopRecipe, NormalizedRect, RgbHistogram};
    use kroma_domain::identity::{
        BoundingBox, FaceEmbedding, Point2D, YuNetLandmarks, SFACE_EMBEDDING_DIM,
    };
    use std::collections::HashMap;
    use std::sync::Mutex;

    fn sample_landmarks() -> YuNetLandmarks {
        let p = Point2D { x: 0.5, y: 0.5 };
        YuNetLandmarks {
            right_eye: p,
            left_eye: p,
            nose_tip: p,
            right_mouth_corner: p,
            left_mouth_corner: p,
        }
    }

    fn face_box(size: f32) -> BoundingBox {
        BoundingBox {
            x: 0.2,
            y: 0.2,
            width: size,
            height: size,
        }
    }

    fn basis_embedding(axis: usize, noise_axis: usize, noise: f32) -> FaceEmbedding {
        let mut v = [0.0_f32; SFACE_EMBEDDING_DIM];
        v[axis] = 1.0;
        v[noise_axis] = noise;
        FaceEmbedding::from_raw(v).unwrap()
    }

    struct StubVision {
        detections: HashMap<String, Vec<RawFaceDetection>>,
        embeddings: HashMap<String, Vec<FaceEmbedding>>,
        slates: HashMap<String, SlateOcrReading>,
    }

    impl FaceDetectorPort for StubVision {
        fn detect_faces(&self, source_path: &str) -> Result<Vec<RawFaceDetection>, AppError> {
            Ok(self.detections.get(source_path).cloned().unwrap_or_default())
        }
    }

    impl FaceRecognizerPort for StubVision {
        fn extract_embedding(
            &self,
            source_path: &str,
            detection: &RawFaceDetection,
        ) -> Result<FaceEmbedding, AppError> {
            let list = self
                .embeddings
                .get(source_path)
                .ok_or_else(|| AppError::NotFound(source_path.to_string()))?;
            let idx = (detection.bounding_box.x * 10.0).round() as usize;
            Ok(list.get(idx).cloned().unwrap_or_else(|| list[0].clone()))
        }
    }

    impl SlateReaderPort for StubVision {
        fn read_slate(&self, source_path: &str) -> Result<Option<SlateOcrReading>, AppError> {
            Ok(self.slates.get(source_path).cloned())
        }
    }

    #[derive(Default)]
    struct RecordingOrganizer {
        transfers: Mutex<Vec<(String, String, OrganizeTransferMode)>>,
        manifests: Mutex<Vec<String>>,
    }

    impl FileOrganizerPort for RecordingOrganizer {
        fn transfer_file(
            &self,
            source_path: &str,
            destination_path: &str,
            mode: OrganizeTransferMode,
        ) -> Result<(), AppError> {
            self.transfers.lock().unwrap().push((
                source_path.to_string(),
                destination_path.to_string(),
                mode,
            ));
            Ok(())
        }

        fn write_plan_manifest(
            &self,
            destination_path: &str,
            _plan: &ShootOrganizationPlan,
        ) -> Result<(), AppError> {
            self.manifests.lock().unwrap().push(destination_path.to_string());
            Ok(())
        }
    }

    struct DummyEngine;
    impl ImageDevelopPort for DummyEngine {
        fn render_photo(
            &self,
            _s: &str,
            _o: &str,
            _r: &DevelopRecipe,
            _f: OutputImageFormat,
        ) -> Result<RenderMetrics, AppError> {
            unimplemented!()
        }
        fn render_batch(&self, jobs: &[BatchRenderJob]) -> Vec<BatchRenderItemOutcome> {
            jobs.iter()
                .map(|j| BatchRenderItemOutcome::Succeeded {
                    photo_id: j.photo_id.clone(),
                    output_path: j.output_path.clone(),
                    metrics: RenderMetrics {
                        output_width: 100,
                        output_height: 100,
                        elapsed_ms: 1,
                    },
                })
                .collect()
        }
        fn sample_average_rgb(&self, _s: &str, _r: NormalizedRect) -> Result<(f32, f32, f32), AppError> {
            Ok((0.9, 0.9, 0.9))
        }
        fn compute_histogram(&self, _s: &str, _r: &DevelopRecipe) -> Result<RgbHistogram, AppError> {
            unimplemented!()
        }
    }

    struct EmptyStore;
    impl DevelopRecipeStorePort for EmptyStore {
        fn load_recipe(&self, _id: &PhotoId) -> Result<Option<DevelopRecipe>, AppError> {
            Ok(None)
        }
        fn save_recipe(&self, _id: &PhotoId, _r: &DevelopRecipe) -> Result<(), AppError> {
            Ok(())
        }
        fn reset_recipe(&self, _id: &PhotoId) -> Result<(), AppError> {
            Ok(())
        }
    }

    #[test]
    fn clusters_and_organizes_purely_by_face_without_any_slates() {
        let mut detections = HashMap::new();
        let mut embeddings = HashMap::new();

        // Person A (2 shots), Person B (1 shot), 1 venue shot with no faces
        for path in ["p1.arw", "p2.arw", "p3.arw"] {
            detections.insert(
                path.to_string(),
                vec![RawFaceDetection {
                    bounding_box: face_box(0.4),
                    landmarks: sample_landmarks(),
                    detection_confidence: 0.99,
                }],
            );
        }
        detections.insert("venue.arw".to_string(), vec![]);

        embeddings.insert("p1.arw".to_string(), vec![basis_embedding(0, 1, 0.0)]);
        embeddings.insert("p2.arw".to_string(), vec![basis_embedding(0, 1, 0.08)]);
        embeddings.insert("p3.arw".to_string(), vec![basis_embedding(5, 6, 0.0)]);

        let vision = StubVision {
            detections,
            embeddings,
            slates: HashMap::new(),
        };

        // Slate reader is explicitly None!
        let cluster_uc = ClusterShootFacesUseCase::new(&vision, &vision, None);
        let clustered = cluster_uc
            .execute(ClusterShootInput::with_defaults(vec![
                PhotoScanTarget {
                    photo_id: PhotoId::new("DSC_0001").unwrap(),
                    source_path: "p1.arw".to_string(),
                    capture_timestamp_epoch_sec: 100,
                    exact_capture_timestamp_epoch_sec: None,
                },
                PhotoScanTarget {
                    photo_id: PhotoId::new("DSC_0002").unwrap(),
                    source_path: "p2.arw".to_string(),
                    capture_timestamp_epoch_sec: 101,
                    exact_capture_timestamp_epoch_sec: None,
                },
                PhotoScanTarget {
                    photo_id: PhotoId::new("DSC_0003").unwrap(),
                    source_path: "p3.arw".to_string(),
                    capture_timestamp_epoch_sec: 102,
                    exact_capture_timestamp_epoch_sec: None,
                },
                PhotoScanTarget {
                    photo_id: PhotoId::new("DSC_0004").unwrap(),
                    source_path: "venue.arw".to_string(),
                    capture_timestamp_epoch_sec: 103,
                    exact_capture_timestamp_epoch_sec: None,
                },
            ]))
            .unwrap();

        assert_eq!(clustered.plan.clusters.len(), 2);
        assert_eq!(clustered.plan.clusters[0].display_name, "Person 001");
        assert_eq!(clustered.plan.clusters[0].entries.len(), 2);
        assert!(!clustered.plan.clusters[0].has_slate());
        assert_eq!(clustered.plan.clusters[1].display_name, "Person 002");
        assert_eq!(clustered.plan.no_face_photos.len(), 1);

        let organizer = RecordingOrganizer::default();
        let org_uc = OrganizeShootFoldersUseCase::new(&organizer);
        let org_out = org_uc
            .execute_with_optional_develop::<DummyEngine, EmptyStore>(
                OrganizeShootInput {
                    plan: clustered.plan,
                    output_root: "/deliveries".to_string(),
                    naming_template: OutputNamingTemplate::default_portrait_template(),
                    execution_mode: OrganizeExecutionMode::TransferOriginals(
                        OrganizeTransferMode::Copy,
                    ),
                    archive_excluded_slates: true,
                    write_manifest_json: true,
                    selected_photo_ids: None,
                },
                None,
                None,
            )
            .unwrap();

        assert_eq!(org_out.organized_files.len(), 4);
        assert_eq!(
            org_out.organized_files[0].destination_path,
            "/deliveries/Person_001/p1.arw"
        );
        assert_eq!(
            org_out.organized_files[1].destination_path,
            "/deliveries/Person_001/p2.arw"
        );
        assert_eq!(
            org_out.organized_files[2].destination_path,
            "/deliveries/Person_002/p3.arw"
        );
        assert_eq!(
            org_out.organized_files[3].destination_path,
            "/deliveries/_No_Face_Detected/IMG_001.arw"
        );
    }

    #[test]
    fn hybrid_mode_uses_optional_slate_names_and_excludes_slate_from_client_pack() {
        let mut detections = HashMap::new();
        let mut embeddings = HashMap::new();
        let mut slates = HashMap::new();

        for path in ["slate_sophia.arw", "sophia_shot.arw", "walkin_guest.arw"] {
            detections.insert(
                path.to_string(),
                vec![RawFaceDetection {
                    bounding_box: face_box(0.45),
                    landmarks: sample_landmarks(),
                    detection_confidence: 0.98,
                }],
            );
        }

        embeddings.insert("slate_sophia.arw".to_string(), vec![basis_embedding(0, 1, 0.0)]);
        embeddings.insert("sophia_shot.arw".to_string(), vec![basis_embedding(0, 1, 0.05)]);
        embeddings.insert("walkin_guest.arw".to_string(), vec![basis_embedding(9, 10, 0.0)]);

        slates.insert(
            "slate_sophia.arw".to_string(),
            SlateOcrReading {
                subject_name: "Sophia Martinez".to_string(),
                subject_code: Some("#104".to_string()),
                ocr_confidence: 0.994,
                whiteboard_region: None,
            },
        );

        let vision = StubVision {
            detections,
            embeddings,
            slates,
        };

        let cluster_uc = ClusterShootFacesUseCase::new(&vision, &vision, Some(&vision));
        let clustered = cluster_uc
            .execute(ClusterShootInput::with_defaults(vec![
                PhotoScanTarget {
                    photo_id: PhotoId::new("DSC_0401").unwrap(),
                    source_path: "slate_sophia.arw".to_string(),
                    capture_timestamp_epoch_sec: 100,
                    exact_capture_timestamp_epoch_sec: None,
                },
                PhotoScanTarget {
                    photo_id: PhotoId::new("DSC_0402").unwrap(),
                    source_path: "sophia_shot.arw".to_string(),
                    capture_timestamp_epoch_sec: 101,
                    exact_capture_timestamp_epoch_sec: None,
                },
                PhotoScanTarget {
                    photo_id: PhotoId::new("DSC_0403").unwrap(),
                    source_path: "walkin_guest.arw".to_string(),
                    capture_timestamp_epoch_sec: 102,
                    exact_capture_timestamp_epoch_sec: None,
                },
            ]))
            .unwrap();

        assert_eq!(clustered.plan.clusters.len(), 2);
        assert_eq!(clustered.plan.clusters[0].display_name, "Sophia Martinez");
        assert_eq!(clustered.plan.clusters[0].subject_code.as_deref(), Some("#001"));
        assert!(clustered.plan.clusters[0].has_slate());
        assert_eq!(clustered.plan.clusters[1].display_name, "Person 002");
        assert_eq!(clustered.plan.clusters[1].subject_code.as_deref(), Some("#002"));
        assert!(!clustered.plan.clusters[1].has_slate());

        let organizer = RecordingOrganizer::default();
        let org_uc = OrganizeShootFoldersUseCase::new(&organizer);
        let org_out = org_uc
            .execute_with_optional_develop(
                OrganizeShootInput {
                    plan: clustered.plan,
                    output_root: "/client_pack".to_string(),
                    naming_template: OutputNamingTemplate::default_portrait_template(),
                    execution_mode: OrganizeExecutionMode::DevelopAndExport {
                        format: OutputImageFormat::Jpeg { quality: 95 },
                    },
                    archive_excluded_slates: true,
                    write_manifest_json: false,
                    selected_photo_ids: None,
                },
                Some(&DummyEngine),
                Some(&EmptyStore),
            )
            .unwrap();

        let paths: Vec<&str> = org_out
            .organized_files
            .iter()
            .map(|f| f.destination_path.as_str())
            .collect();
        assert_eq!(
            paths,
            vec![
                "/client_pack/_Slates_Archive/DSC_0401.jpg",
                "/client_pack/Sophia_Martinez/sophia_shot.jpg",
                "/client_pack/Person_002/walkin_guest.jpg",
            ]
        );
    }

    #[test]
    fn duplicate_subject_names_automatically_disambiguate_with_generated_codes() {
        let mut detections = HashMap::new();
        let mut embeddings = HashMap::new();
        for path in ["/shoot/DSC_001.ARW", "/shoot/DSC_002.ARW"] {
            detections.insert(
                path.to_string(),
                vec![RawFaceDetection {
                    bounding_box: face_box(0.45),
                    landmarks: sample_landmarks(),
                    detection_confidence: 0.99,
                }],
            );
        }
        embeddings.insert("/shoot/DSC_001.ARW".to_string(), vec![basis_embedding(0, 1, 0.0)]);
        embeddings.insert("/shoot/DSC_002.ARW".to_string(), vec![basis_embedding(6, 7, 0.0)]);

        let vision = StubVision {
            detections,
            embeddings,
            slates: HashMap::new(),
        };
        let cluster_uc = ClusterShootFacesUseCase::new(&vision, &vision, None);
        let mut clustered = cluster_uc
            .execute(ClusterShootInput::with_defaults(vec![
                PhotoScanTarget {
                    photo_id: PhotoId::new("DSC_001").unwrap(),
                    source_path: "/shoot/DSC_001.ARW".to_string(),
                    capture_timestamp_epoch_sec: 100,
                    exact_capture_timestamp_epoch_sec: None,
                },
                PhotoScanTarget {
                    photo_id: PhotoId::new("DSC_002").unwrap(),
                    source_path: "/shoot/DSC_002.ARW".to_string(),
                    capture_timestamp_epoch_sec: 101,
                    exact_capture_timestamp_epoch_sec: None,
                },
            ]))
            .unwrap();

        // Both people share the exact same name -> folders must not collide
        clustered.plan.clusters[0].rename("Santos, Maria").unwrap();
        clustered.plan.clusters[1].rename("Santos, Maria").unwrap();

        let organizer = RecordingOrganizer::default();
        let org_uc = OrganizeShootFoldersUseCase::new(&organizer);
        let out = org_uc
            .execute_with_optional_develop(
                OrganizeShootInput {
                    plan: clustered.plan,
                    output_root: "/deliveries".to_string(),
                    naming_template: OutputNamingTemplate::default_portrait_template(),
                    execution_mode: OrganizeExecutionMode::DevelopAndExport {
                        format: OutputImageFormat::Jpeg { quality: 94 },
                    },
                    archive_excluded_slates: false,
                    write_manifest_json: false,
                    selected_photo_ids: None,
                },
                Some(&DummyEngine),
                Some(&EmptyStore),
            )
            .unwrap();

        let paths: Vec<&str> = out
            .organized_files
            .iter()
            .map(|f| f.destination_path.as_str())
            .collect();
        assert_eq!(
            paths,
            vec![
                "/deliveries/Santos__Maria_001/DSC_001.jpg",
                "/deliveries/Santos__Maria_002/DSC_002.jpg",
            ]
        );
    }

    #[test]
    fn develop_and_keep_raw_outputs_both_edited_and_untouched_raw_subfolders() {
        let mut detections = HashMap::new();
        let mut embeddings = HashMap::new();
        detections.insert(
            "/shoot/DSC_0100.ARW".to_string(),
            vec![RawFaceDetection {
                bounding_box: face_box(0.45),
                landmarks: sample_landmarks(),
                detection_confidence: 0.99,
            }],
        );
        embeddings.insert(
            "/shoot/DSC_0100.ARW".to_string(),
            vec![basis_embedding(0, 1, 0.0)],
        );

        let vision = StubVision {
            detections,
            embeddings,
            slates: HashMap::new(),
        };
        let cluster_uc = ClusterShootFacesUseCase::new(&vision, &vision, None);
        let clustered = cluster_uc
            .execute(ClusterShootInput::with_defaults(vec![PhotoScanTarget {
                photo_id: PhotoId::new("DSC_0100").unwrap(),
                source_path: "/shoot/DSC_0100.ARW".to_string(),
                capture_timestamp_epoch_sec: 100,
                exact_capture_timestamp_epoch_sec: None,
            }]))
            .unwrap();

        let organizer = RecordingOrganizer::default();
        let org_uc = OrganizeShootFoldersUseCase::new(&organizer);
        let out = org_uc
            .execute_with_optional_develop(
                OrganizeShootInput {
                    plan: clustered.plan,
                    output_root: "/deliveries".to_string(),
                    naming_template: OutputNamingTemplate::default_portrait_template(),
                    execution_mode: OrganizeExecutionMode::DevelopAndKeepRaw {
                        format: OutputImageFormat::Jpeg { quality: 94 },
                        raw_transfer_mode: OrganizeTransferMode::Copy,
                    },
                    archive_excluded_slates: false,
                    write_manifest_json: false,
                    selected_photo_ids: None,
                },
                Some(&DummyEngine),
                Some(&EmptyStore),
            )
            .unwrap();

        let paths: Vec<&str> = out
            .organized_files
            .iter()
            .map(|f| f.destination_path.as_str())
            .collect();
        assert_eq!(
            paths,
            vec![
                "/deliveries/Person_001/Edited/DSC_0100.jpg",
                "/deliveries/Person_001/RAW/DSC_0100.ARW",
            ]
        );
    }

    #[test]
    fn execute_with_progress_reports_monotonic_export_counts() {
        use std::sync::Mutex;

        let mut detections = HashMap::new();
        let mut embeddings = HashMap::new();
        for path in ["/shoot/A.ARW", "/shoot/B.ARW"] {
            detections.insert(
                path.to_string(),
                vec![RawFaceDetection {
                    bounding_box: face_box(0.45),
                    landmarks: sample_landmarks(),
                    detection_confidence: 0.99,
                }],
            );
        }
        embeddings.insert("/shoot/A.ARW".to_string(), vec![basis_embedding(0, 1, 0.0)]);
        embeddings.insert("/shoot/B.ARW".to_string(), vec![basis_embedding(5, 6, 0.0)]);

        let vision = StubVision {
            detections,
            embeddings,
            slates: HashMap::new(),
        };
        let cluster_uc = ClusterShootFacesUseCase::new(&vision, &vision, None);
        let clustered = cluster_uc
            .execute(ClusterShootInput::with_defaults(vec![
                PhotoScanTarget {
                    photo_id: PhotoId::new("A").unwrap(),
                    source_path: "/shoot/A.ARW".to_string(),
                    capture_timestamp_epoch_sec: 100,
                    exact_capture_timestamp_epoch_sec: None,
                },
                PhotoScanTarget {
                    photo_id: PhotoId::new("B").unwrap(),
                    source_path: "/shoot/B.ARW".to_string(),
                    capture_timestamp_epoch_sec: 101,
                    exact_capture_timestamp_epoch_sec: None,
                },
            ]))
            .unwrap();

        let organizer = RecordingOrganizer::default();
        let org_uc = OrganizeShootFoldersUseCase::new(&organizer);
        let events: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

        org_uc
            .execute_with_progress(
                OrganizeShootInput {
                    plan: clustered.plan,
                    output_root: "/deliveries".to_string(),
                    naming_template: OutputNamingTemplate::default_portrait_template(),
                    execution_mode: OrganizeExecutionMode::TransferOriginals(
                        OrganizeTransferMode::Copy,
                    ),
                    archive_excluded_slates: false,
                    write_manifest_json: false,
                    selected_photo_ids: None,
                },
                Option::<&DummyEngine>::None,
                Option::<&EmptyStore>::None,
                &|done, total, _| {
                    events.lock().unwrap().push((done, total));
                },
            )
            .unwrap();

        let log = events.into_inner().unwrap();
        assert_eq!(log.first().copied(), Some((0, 2)));
        assert_eq!(log.last().copied(), Some((2, 2)));
        assert_eq!(log.len(), 3);
    }

    #[test]
    fn selective_photo_ids_exports_exact_subset_preserving_canonical_paths() {
        let mut detections = HashMap::new();
        let mut embeddings = HashMap::new();
        for path in [
            "/shoot/DSC_001.ARW",
            "/shoot/DSC_002.ARW",
            "/shoot/DSC_003.ARW",
        ] {
            detections.insert(
                path.to_string(),
                vec![RawFaceDetection {
                    bounding_box: face_box(0.45),
                    landmarks: sample_landmarks(),
                    detection_confidence: 0.99,
                }],
            );
        }
        // Person 1 has 2 photos (DSC_001, DSC_002); Person 2 has 1 photo (DSC_003).
        embeddings.insert(
            "/shoot/DSC_001.ARW".to_string(),
            vec![basis_embedding(0, 1, 0.0)],
        );
        embeddings.insert(
            "/shoot/DSC_002.ARW".to_string(),
            vec![basis_embedding(0, 1, 0.04)],
        );
        embeddings.insert(
            "/shoot/DSC_003.ARW".to_string(),
            vec![basis_embedding(6, 7, 0.0)],
        );

        let vision = StubVision {
            detections,
            embeddings,
            slates: HashMap::new(),
        };
        let cluster_uc = ClusterShootFacesUseCase::new(&vision, &vision, None);
        let mut clustered = cluster_uc
            .execute(ClusterShootInput::with_defaults(vec![
                PhotoScanTarget {
                    photo_id: PhotoId::new("DSC_001").unwrap(),
                    source_path: "/shoot/DSC_001.ARW".to_string(),
                    capture_timestamp_epoch_sec: 100,
                    exact_capture_timestamp_epoch_sec: None,
                },
                PhotoScanTarget {
                    photo_id: PhotoId::new("DSC_002").unwrap(),
                    source_path: "/shoot/DSC_002.ARW".to_string(),
                    capture_timestamp_epoch_sec: 101,
                    exact_capture_timestamp_epoch_sec: None,
                },
                PhotoScanTarget {
                    photo_id: PhotoId::new("DSC_003").unwrap(),
                    source_path: "/shoot/DSC_003.ARW".to_string(),
                    capture_timestamp_epoch_sec: 102,
                    exact_capture_timestamp_epoch_sec: None,
                },
            ]))
            .unwrap();

        // Give both clusters the same name so collision disambiguation must stay consistent.
        clustered.plan.clusters[0].rename("Santos, Maria").unwrap();
        clustered.plan.clusters[1].rename("Santos, Maria").unwrap();

        let organizer = RecordingOrganizer::default();
        let org_uc = OrganizeShootFoldersUseCase::new(&organizer);
        let mut only_second_photo = HashSet::new();
        only_second_photo.insert(PhotoId::new("DSC_002").unwrap());

        let out = org_uc
            .execute_with_optional_develop(
                OrganizeShootInput {
                    plan: clustered.plan,
                    output_root: "/deliveries".to_string(),
                    naming_template: OutputNamingTemplate::new("/[Slate_Name]/IMG_001.jpg", true)
                        .unwrap(),
                    execution_mode: OrganizeExecutionMode::DevelopAndExport {
                        format: OutputImageFormat::Jpeg { quality: 94 },
                    },
                    archive_excluded_slates: false,
                    write_manifest_json: false,
                    selected_photo_ids: Some(only_second_photo),
                },
                Some(&DummyEngine),
                Some(&EmptyStore),
            )
            .unwrap();

        let paths: Vec<&str> = out
            .organized_files
            .iter()
            .map(|f| f.destination_path.as_str())
            .collect();
        assert_eq!(paths, vec!["/deliveries/Santos__Maria_001/IMG_002.jpg"]);
        assert!(out.manifest_path.is_none());
    }
}
