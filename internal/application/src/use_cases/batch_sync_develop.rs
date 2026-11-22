use crate::error::AppError;
use crate::ports::recipe_store::DevelopRecipeStorePort;
use kroma_domain::develop::{DevelopRecipe, DevelopSyncMask};
use kroma_domain::photo::PhotoId;

#[derive(Debug, Clone, PartialEq)]
pub struct BatchSyncDevelopInput {
    pub source_photo_id: PhotoId,
    pub target_photo_ids: Vec<PhotoId>,
    pub sync_mask: DevelopSyncMask,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BatchSyncFailure {
    pub photo_id: PhotoId,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BatchSyncDevelopOutput {
    pub synced_photo_ids: Vec<PhotoId>,
    pub failures: Vec<BatchSyncFailure>,
}

/// Lightroom-style `Sync Settings` (`Ctrl+Shift+S`): copies selected develop modules
/// from a reference photo onto a batch of target photos while preserving unmasked edits.
pub struct BatchSyncDevelopUseCase<'a, S: DevelopRecipeStorePort> {
    store: &'a S,
}

impl<'a, S: DevelopRecipeStorePort> BatchSyncDevelopUseCase<'a, S> {
    pub fn new(store: &'a S) -> Self {
        Self { store }
    }

    pub fn execute(
        &self,
        input: BatchSyncDevelopInput,
    ) -> Result<BatchSyncDevelopOutput, AppError> {
        if input.target_photo_ids.is_empty() {
            return Err(AppError::InvalidInput(
                "at least one target photo is required for batch sync".to_string(),
            ));
        }

        let mut source_recipe = self
            .store
            .load_recipe(&input.source_photo_id)?
            .unwrap_or_default();
        source_recipe.clamp_all_invariants();

        let mut synced_photo_ids = Vec::with_capacity(input.target_photo_ids.len());
        let mut failures = Vec::new();

        for target_id in input.target_photo_ids {
            if target_id == input.source_photo_id {
                synced_photo_ids.push(target_id);
                continue;
            }

            let target_result = self.sync_single_target(&target_id, &source_recipe, input.sync_mask);
            match target_result {
                Ok(()) => synced_photo_ids.push(target_id),
                Err(err) => failures.push(BatchSyncFailure {
                    photo_id: target_id,
                    reason: err.to_string(),
                }),
            }
        }

        Ok(BatchSyncDevelopOutput {
            synced_photo_ids,
            failures,
        })
    }

    fn sync_single_target(
        &self,
        target_id: &PhotoId,
        source_recipe: &DevelopRecipe,
        mask: DevelopSyncMask,
    ) -> Result<(), AppError> {
        let mut target_recipe = self.store.load_recipe(target_id)?.unwrap_or_default();
        target_recipe.sync_from(source_recipe, mask);
        self.store.save_recipe(target_id, &target_recipe)
    }
}
