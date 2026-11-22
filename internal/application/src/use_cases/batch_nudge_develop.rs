use crate::error::AppError;
use crate::ports::recipe_store::DevelopRecipeStorePort;
use kroma_domain::develop::RelativeDevelopDelta;
use kroma_domain::photo::PhotoId;

#[derive(Debug, Clone, PartialEq)]
pub struct BatchRelativeNudgeInput {
    pub target_photo_ids: Vec<PhotoId>,
    pub delta: RelativeDevelopDelta,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BatchRelativeNudgeOutput {
    pub updated_count: usize,
    pub failed_ids: Vec<(PhotoId, String)>,
}

/// Lightroom "Quick Develop" relative adjustment across a multi-selection
/// (e.g. nudging +0.25 EV or +200K across photos that already have individual tweaks).
pub struct BatchRelativeNudgeUseCase<'a, S: DevelopRecipeStorePort> {
    store: &'a S,
}

impl<'a, S: DevelopRecipeStorePort> BatchRelativeNudgeUseCase<'a, S> {
    pub fn new(store: &'a S) -> Self {
        Self { store }
    }

    pub fn execute(
        &self,
        input: BatchRelativeNudgeInput,
    ) -> Result<BatchRelativeNudgeOutput, AppError> {
        if input.target_photo_ids.is_empty() {
            return Err(AppError::InvalidInput(
                "at least one target photo is required for relative batch nudge".to_string(),
            ));
        }

        let mut updated_count = 0usize;
        let mut failed_ids = Vec::new();

        for photo_id in input.target_photo_ids {
            match self.nudge_single(&photo_id, input.delta) {
                Ok(()) => updated_count += 1,
                Err(err) => failed_ids.push((photo_id, err.to_string())),
            }
        }

        Ok(BatchRelativeNudgeOutput {
            updated_count,
            failed_ids,
        })
    }

    fn nudge_single(
        &self,
        photo_id: &PhotoId,
        delta: RelativeDevelopDelta,
    ) -> Result<(), AppError> {
        let mut recipe = self.store.load_recipe(photo_id)?.unwrap_or_default();
        recipe.apply_relative_delta(delta);
        self.store.save_recipe(photo_id, &recipe)
    }
}
