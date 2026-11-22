use crate::error::AppError;
use crate::ports::develop_engine::ImageDevelopPort;
use crate::ports::recipe_store::DevelopRecipeStorePort;
use kroma_domain::develop::{NormalizedRect, WhiteBalance};
use kroma_domain::photo::PhotoId;

#[derive(Debug, Clone, PartialEq)]
pub struct CalibrateWhiteBalanceFromSlateInput {
    pub slate_source_path: String,
    pub whiteboard_region: NormalizedRect,
    pub target_photo_ids: Vec<PhotoId>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CalibrateWhiteBalanceFromSlateOutput {
    pub calibrated_wb: WhiteBalance,
    pub applied_photo_count: usize,
}

/// Samples the subject's physical whiteboard in the slate photo as a neutral gray/white card
/// and syncs the calibrated Kelvin + Tint across all target photos in the subject's cluster.
pub struct CalibrateWhiteBalanceFromSlateUseCase<'a, E: ImageDevelopPort, S: DevelopRecipeStorePort>
{
    engine: &'a E,
    store: &'a S,
}

impl<'a, E: ImageDevelopPort, S: DevelopRecipeStorePort>
    CalibrateWhiteBalanceFromSlateUseCase<'a, E, S>
{
    pub fn new(engine: &'a E, store: &'a S) -> Self {
        Self { engine, store }
    }

    pub fn execute(
        &self,
        input: CalibrateWhiteBalanceFromSlateInput,
    ) -> Result<CalibrateWhiteBalanceFromSlateOutput, AppError> {
        if input.slate_source_path.trim().is_empty() {
            return Err(AppError::InvalidInput(
                "slate_source_path cannot be empty".to_string(),
            ));
        }

        let (r, g, b) = self
            .engine
            .sample_average_rgb(&input.slate_source_path, input.whiteboard_region)?;
        let calibrated_wb = WhiteBalance::calibrated_from_neutral_rgb(r, g, b);

        let mut applied_photo_count = 0usize;
        for photo_id in &input.target_photo_ids {
            let mut recipe = self.store.load_recipe(photo_id)?.unwrap_or_default();
            recipe.white_balance = calibrated_wb;
            self.store.save_recipe(photo_id, &recipe)?;
            applied_photo_count += 1;
        }

        Ok(CalibrateWhiteBalanceFromSlateOutput {
            calibrated_wb,
            applied_photo_count,
        })
    }
}
