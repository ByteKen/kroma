use crate::error::AppError;
use crate::ports::recipe_store::DevelopRecipeStorePort;
use kroma_domain::develop::DevelopRecipe;
use kroma_domain::photo::PhotoId;

#[derive(Debug, Clone, PartialEq)]
pub struct SavePhotoDevelopRecipeInput {
    pub photo_id: PhotoId,
    pub recipe: DevelopRecipe,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SavePhotoDevelopRecipeOutput {
    pub photo_id: PhotoId,
    pub clamped_recipe: DevelopRecipe,
}

pub struct SavePhotoDevelopRecipeUseCase<'a, S: DevelopRecipeStorePort> {
    store: &'a S,
}

impl<'a, S: DevelopRecipeStorePort> SavePhotoDevelopRecipeUseCase<'a, S> {
    pub fn new(store: &'a S) -> Self {
        Self { store }
    }

    pub fn execute(
        &self,
        mut input: SavePhotoDevelopRecipeInput,
    ) -> Result<SavePhotoDevelopRecipeOutput, AppError> {
        input.recipe.clamp_all_invariants();
        self.store.save_recipe(&input.photo_id, &input.recipe)?;
        Ok(SavePhotoDevelopRecipeOutput {
            photo_id: input.photo_id,
            clamped_recipe: input.recipe,
        })
    }
}
