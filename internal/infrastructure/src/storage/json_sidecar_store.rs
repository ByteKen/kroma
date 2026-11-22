use crate::storage::recipe_dto::RecipeSidecarDto;
use kroma_app::error::AppError;
use kroma_app::ports::recipe_store::DevelopRecipeStorePort;
use kroma_domain::develop::DevelopRecipe;
use kroma_domain::photo::PhotoId;
use std::fs;
use std::path::PathBuf;

/// Stores non-destructive `.kroma.json` sidecar files inside a dedicated catalog directory.
/// Uses write-temp + rename for atomic persistence so concurrent UI/batch reads never observe torn JSON.
#[derive(Debug, Clone)]
pub struct JsonSidecarRecipeStore {
    root_dir: PathBuf,
}

impl JsonSidecarRecipeStore {
    pub fn new(root_dir: impl Into<PathBuf>) -> Self {
        Self {
            root_dir: root_dir.into(),
        }
    }

    fn sidecar_path(&self, photo_id: &PhotoId) -> PathBuf {
        let safe_id: String = photo_id
            .as_str()
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        self.root_dir.join(format!("{safe_id}.kroma.json"))
    }
}

impl DevelopRecipeStorePort for JsonSidecarRecipeStore {
    fn load_recipe(&self, photo_id: &PhotoId) -> Result<Option<DevelopRecipe>, AppError> {
        let path = self.sidecar_path(photo_id);
        if !path.exists() {
            return Ok(None);
        }

        let raw = fs::read_to_string(&path).map_err(|err| {
            AppError::StorageFailure(format!("failed to read sidecar '{}': {err}", path.display()))
        })?;
        let dto: RecipeSidecarDto = serde_json::from_str(&raw).map_err(|err| {
            AppError::StorageFailure(format!(
                "failed to parse sidecar '{}': {err}",
                path.display()
            ))
        })?;
        Ok(Some(dto.into_domain()))
    }

    fn save_recipe(&self, photo_id: &PhotoId, recipe: &DevelopRecipe) -> Result<(), AppError> {
        fs::create_dir_all(&self.root_dir).map_err(|err| {
            AppError::StorageFailure(format!(
                "failed to create sidecar directory '{}': {err}",
                self.root_dir.display()
            ))
        })?;

        let target = self.sidecar_path(photo_id);
        let temp = target.with_extension("kroma.json.tmp");
        let dto = RecipeSidecarDto::from(recipe);
        let payload = serde_json::to_string_pretty(&dto).map_err(|err| {
            AppError::StorageFailure(format!("failed to serialize recipe sidecar: {err}"))
        })?;

        fs::write(&temp, payload).map_err(|err| {
            AppError::StorageFailure(format!(
                "failed to write temp sidecar '{}': {err}",
                temp.display()
            ))
        })?;
        fs::rename(&temp, &target).map_err(|err| {
            AppError::StorageFailure(format!(
                "failed to commit sidecar '{}': {err}",
                target.display()
            ))
        })?;
        Ok(())
    }

    fn reset_recipe(&self, photo_id: &PhotoId) -> Result<(), AppError> {
        let path = self.sidecar_path(photo_id);
        if path.exists() {
            fs::remove_file(&path).map_err(|err| {
                AppError::StorageFailure(format!(
                    "failed to remove sidecar '{}': {err}",
                    path.display()
                ))
            })?;
        }
        Ok(())
    }
}
