use crate::error::AppError;
use kroma_domain::develop::DevelopRecipe;
use kroma_domain::photo::PhotoId;

/// Inward-owned port for persisting non-destructive DevelopRecipes (e.g. `.kroma.json` sidecars or SQLite).
pub trait DevelopRecipeStorePort: Send + Sync {
    fn load_recipe(&self, photo_id: &PhotoId) -> Result<Option<DevelopRecipe>, AppError>;
    fn save_recipe(&self, photo_id: &PhotoId, recipe: &DevelopRecipe) -> Result<(), AppError>;
    fn reset_recipe(&self, photo_id: &PhotoId) -> Result<(), AppError>;
}
