use crate::error::AppError;
use crate::ports::develop_engine::{
    BatchRenderItemOutcome, BatchRenderJob, ImageDevelopPort, OutputImageFormat,
};
use crate::ports::recipe_store::DevelopRecipeStorePort;
use kroma_domain::photo::PhotoId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchRenderPhotoTarget {
    pub photo_id: PhotoId,
    pub source_path: String,
    pub output_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchRenderPhotosInput {
    pub targets: Vec<BatchRenderPhotoTarget>,
    pub format: OutputImageFormat,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BatchRenderPhotosOutput {
    pub outcomes: Vec<BatchRenderItemOutcome>,
    pub succeeded_count: usize,
    pub failed_count: usize,
}

/// Resolves each target photo's stored non-destructive DevelopRecipe and delegates
/// parallel rendering to `ImageDevelopPort`, preserving per-photo partial-failure reporting.
pub struct BatchRenderPhotosUseCase<'a, E: ImageDevelopPort, S: DevelopRecipeStorePort> {
    engine: &'a E,
    store: &'a S,
}

impl<'a, E: ImageDevelopPort, S: DevelopRecipeStorePort> BatchRenderPhotosUseCase<'a, E, S> {
    pub fn new(engine: &'a E, store: &'a S) -> Self {
        Self { engine, store }
    }

    pub fn execute(
        &self,
        input: BatchRenderPhotosInput,
    ) -> Result<BatchRenderPhotosOutput, AppError> {
        if input.targets.is_empty() {
            return Err(AppError::InvalidInput(
                "at least one target photo is required for batch render".to_string(),
            ));
        }

        let mut jobs = Vec::with_capacity(input.targets.len());
        let mut pre_failures = Vec::new();

        for target in input.targets {
            match self.store.load_recipe(&target.photo_id) {
                Ok(maybe_recipe) => jobs.push(BatchRenderJob {
                    photo_id: target.photo_id,
                    source_path: target.source_path,
                    output_path: target.output_path,
                    recipe: maybe_recipe.unwrap_or_default(),
                    format: input.format,
                }),
                Err(err) => pre_failures.push(BatchRenderItemOutcome::Failed {
                    photo_id: target.photo_id,
                    source_path: target.source_path,
                    reason: err.to_string(),
                }),
            }
        }

        let mut outcomes = self.engine.render_batch(&jobs);
        outcomes.extend(pre_failures);

        let succeeded_count = outcomes
            .iter()
            .filter(|o| matches!(o, BatchRenderItemOutcome::Succeeded { .. }))
            .count();
        let failed_count = outcomes.len() - succeeded_count;

        Ok(BatchRenderPhotosOutput {
            outcomes,
            succeeded_count,
            failed_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ports::develop_engine::RenderMetrics;
    use crate::use_cases::auto_wb_from_slate::{
        CalibrateWhiteBalanceFromSlateInput, CalibrateWhiteBalanceFromSlateUseCase,
    };
    use crate::use_cases::batch_nudge_develop::{
        BatchRelativeNudgeInput, BatchRelativeNudgeUseCase,
    };
    use crate::use_cases::batch_sync_develop::{BatchSyncDevelopInput, BatchSyncDevelopUseCase};
    use crate::use_cases::edit_photo::{
        SavePhotoDevelopRecipeInput, SavePhotoDevelopRecipeUseCase,
    };
    use kroma_domain::develop::{
        DevelopRecipe, DevelopSyncMask, NormalizedRect, RelativeDevelopDelta, RgbHistogram,
    };
    use std::collections::HashMap;
    use std::sync::Mutex;

    #[derive(Default)]
    struct InMemoryRecipeStore {
        recipes: Mutex<HashMap<PhotoId, DevelopRecipe>>,
    }

    impl DevelopRecipeStorePort for InMemoryRecipeStore {
        fn load_recipe(&self, photo_id: &PhotoId) -> Result<Option<DevelopRecipe>, AppError> {
            Ok(self.recipes.lock().expect("lock").get(photo_id).cloned())
        }

        fn save_recipe(&self, photo_id: &PhotoId, recipe: &DevelopRecipe) -> Result<(), AppError> {
            self.recipes
                .lock()
                .expect("lock")
                .insert(photo_id.clone(), recipe.clone());
            Ok(())
        }

        fn reset_recipe(&self, photo_id: &PhotoId) -> Result<(), AppError> {
            self.recipes.lock().expect("lock").remove(photo_id);
            Ok(())
        }
    }

    struct FakeDevelopEngine {
        sample_rgb: (f32, f32, f32),
        fail_photo_id: Option<PhotoId>,
    }

    impl ImageDevelopPort for FakeDevelopEngine {
        fn render_photo(
            &self,
            _source_path: &str,
            _output_path: &str,
            _recipe: &DevelopRecipe,
            _format: OutputImageFormat,
        ) -> Result<RenderMetrics, AppError> {
            Ok(RenderMetrics {
                output_width: 4000,
                output_height: 6000,
                elapsed_ms: 12,
            })
        }

        fn render_batch(&self, jobs: &[BatchRenderJob]) -> Vec<BatchRenderItemOutcome> {
            jobs.iter()
                .map(|job| {
                    if self.fail_photo_id.as_ref() == Some(&job.photo_id) {
                        BatchRenderItemOutcome::Failed {
                            photo_id: job.photo_id.clone(),
                            source_path: job.source_path.clone(),
                            reason: "simulated corrupt file".to_string(),
                        }
                    } else {
                        BatchRenderItemOutcome::Succeeded {
                            photo_id: job.photo_id.clone(),
                            output_path: job.output_path.clone(),
                            metrics: RenderMetrics {
                                output_width: 4000,
                                output_height: 6000,
                                elapsed_ms: 10,
                            },
                        }
                    }
                })
                .collect()
        }

        fn sample_average_rgb(
            &self,
            _source_path: &str,
            _region: NormalizedRect,
        ) -> Result<(f32, f32, f32), AppError> {
            Ok(self.sample_rgb)
        }

        fn compute_histogram(
            &self,
            _source_path: &str,
            _recipe: &DevelopRecipe,
        ) -> Result<RgbHistogram, AppError> {
            Ok(RgbHistogram {
                red: [0; 256],
                green: [0; 256],
                blue: [0; 256],
                luminance: [0; 256],
                total_pixels: 0,
            })
        }
    }

    #[test]
    fn batch_sync_copies_tone_and_preserves_target_crop_then_nudges_relatively() {
        let store = InMemoryRecipeStore::default();
        let save_uc = SavePhotoDevelopRecipeUseCase::new(&store);

        let src_id = PhotoId::new("DSC_0401").unwrap();
        let tgt_id = PhotoId::new("DSC_0402").unwrap();

        let mut src_recipe = DevelopRecipe::default();
        src_recipe.basic_tone.exposure_ev = 0.6;
        src_recipe.basic_tone.contrast = 25;
        src_recipe.crop_transform.rotation_deg = 5.0;
        save_uc
            .execute(SavePhotoDevelopRecipeInput {
                photo_id: src_id.clone(),
                recipe: src_recipe,
            })
            .unwrap();

        let mut tgt_initial = DevelopRecipe::default();
        tgt_initial.crop_transform.rotation_deg = -2.0;
        save_uc
            .execute(SavePhotoDevelopRecipeInput {
                photo_id: tgt_id.clone(),
                recipe: tgt_initial,
            })
            .unwrap();

        let sync_uc = BatchSyncDevelopUseCase::new(&store);
        let sync_out = sync_uc
            .execute(BatchSyncDevelopInput {
                source_photo_id: src_id.clone(),
                target_photo_ids: vec![tgt_id.clone()],
                sync_mask: DevelopSyncMask::color_and_tone_only(),
            })
            .unwrap();

        assert_eq!(sync_out.synced_photo_ids.len(), 1);
        assert!(sync_out.failures.is_empty());

        let synced_tgt = store.load_recipe(&tgt_id).unwrap().unwrap();
        assert_eq!(synced_tgt.basic_tone.exposure_ev, 0.6);
        assert_eq!(synced_tgt.basic_tone.contrast, 25);
        assert_eq!(synced_tgt.crop_transform.rotation_deg, -2.0);

        // Relative Quick Develop nudge (+0.25 EV across both photos)
        let nudge_uc = BatchRelativeNudgeUseCase::new(&store);
        let nudge_out = nudge_uc
            .execute(BatchRelativeNudgeInput {
                target_photo_ids: vec![src_id.clone(), tgt_id.clone()],
                delta: RelativeDevelopDelta {
                    exposure_delta_ev: 0.25,
                    ..Default::default()
                },
            })
            .unwrap();

        assert_eq!(nudge_out.updated_count, 2);
        assert!(
            (store.load_recipe(&tgt_id).unwrap().unwrap().basic_tone.exposure_ev - 0.85).abs()
                < 1e-4
        );
    }

    #[test]
    fn slate_white_balance_calibration_updates_entire_subject_cluster() {
        let store = InMemoryRecipeStore::default();
        let engine = FakeDevelopEngine {
            sample_rgb: (0.72, 0.84, 0.96),
            fail_photo_id: None,
        };
        let wb_uc = CalibrateWhiteBalanceFromSlateUseCase::new(&engine, &store);

        let p1 = PhotoId::new("DSC_0402").unwrap();
        let p2 = PhotoId::new("DSC_0405").unwrap();
        let out = wb_uc
            .execute(CalibrateWhiteBalanceFromSlateInput {
                slate_source_path: "/shoot/DSC_0401.ARW".to_string(),
                whiteboard_region: NormalizedRect::new(0.3, 0.55, 0.4, 0.25).unwrap(),
                target_photo_ids: vec![p1.clone(), p2.clone()],
            })
            .unwrap();

        assert_eq!(out.applied_photo_count, 2);
        assert!(out.calibrated_wb.temperature_kelvin > 5500);
        assert_eq!(
            store.load_recipe(&p1).unwrap().unwrap().white_balance,
            out.calibrated_wb
        );
        assert_eq!(
            store.load_recipe(&p2).unwrap().unwrap().white_balance,
            out.calibrated_wb
        );
    }

    #[test]
    fn batch_render_reports_partial_failures_without_aborting_batch() {
        let store = InMemoryRecipeStore::default();
        let ok_id = PhotoId::new("DSC_0402").unwrap();
        let bad_id = PhotoId::new("DSC_0403").unwrap();
        let engine = FakeDevelopEngine {
            sample_rgb: (0.9, 0.9, 0.9),
            fail_photo_id: Some(bad_id.clone()),
        };

        let render_uc = BatchRenderPhotosUseCase::new(&engine, &store);
        let out = render_uc
            .execute(BatchRenderPhotosInput {
                targets: vec![
                    BatchRenderPhotoTarget {
                        photo_id: ok_id,
                        source_path: "/in/DSC_0402.jpg".to_string(),
                        output_path: "/out/DSC_0402.jpg".to_string(),
                    },
                    BatchRenderPhotoTarget {
                        photo_id: bad_id,
                        source_path: "/in/DSC_0403.jpg".to_string(),
                        output_path: "/out/DSC_0403.jpg".to_string(),
                    },
                ],
                format: OutputImageFormat::default(),
            })
            .unwrap();

        assert_eq!(out.succeeded_count, 1);
        assert_eq!(out.failed_count, 1);
    }
}
