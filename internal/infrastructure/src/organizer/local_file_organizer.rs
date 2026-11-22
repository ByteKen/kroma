use crate::organizer::manifest_dto::ShootManifestDto;
use kroma_app::error::AppError;
use kroma_app::ports::organizer::{FileOrganizerPort, OrganizeTransferMode};
use kroma_domain::identity::ShootOrganizationPlan;
use std::fs;
use std::path::Path;

/// Concrete filesystem organizer adapter implementing `FileOrganizerPort`.
/// Supports Copy, Move (with automatic cross-volume copy+delete fallback), HardLink,
/// and atomic JSON manifest writing.
#[derive(Debug, Default, Clone)]
pub struct LocalFileOrganizer;

impl LocalFileOrganizer {
    pub fn new() -> Self {
        Self
    }
}

impl FileOrganizerPort for LocalFileOrganizer {
    fn transfer_file(
        &self,
        source_path: &str,
        destination_path: &str,
        mode: OrganizeTransferMode,
    ) -> Result<(), AppError> {
        let src = Path::new(source_path);
        let dst = Path::new(destination_path);

        if let Some(parent) = dst.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|err| {
                AppError::StorageFailure(format!(
                    "failed to create directory '{}': {err}",
                    parent.display()
                ))
            })?;
        }

        match mode {
            OrganizeTransferMode::Copy => {
                fs::copy(src, dst).map_err(|err| {
                    AppError::StorageFailure(format!(
                        "failed to copy '{}' -> '{}': {err}",
                        src.display(),
                        dst.display()
                    ))
                })?;
            }
            OrganizeTransferMode::HardLink => {
                if dst.exists() {
                    fs::remove_file(dst).map_err(|err| {
                        AppError::StorageFailure(format!(
                            "failed to replace existing '{}': {err}",
                            dst.display()
                        ))
                    })?;
                }
                fs::hard_link(src, dst).map_err(|err| {
                    AppError::StorageFailure(format!(
                        "failed to hard-link '{}' -> '{}': {err}",
                        src.display(),
                        dst.display()
                    ))
                })?;
            }
            OrganizeTransferMode::Move => {
                if fs::rename(src, dst).is_err() {
                    // Cross-drive move fallback (e.g. SD Card E:\ -> D:\Deliveries)
                    fs::copy(src, dst).map_err(|err| {
                        AppError::StorageFailure(format!(
                            "failed cross-volume copy '{}' -> '{}': {err}",
                            src.display(),
                            dst.display()
                        ))
                    })?;
                    fs::remove_file(src).map_err(|err| {
                        AppError::StorageFailure(format!(
                            "copied '{}' but failed to delete original: {err}",
                            src.display()
                        ))
                    })?;
                }
            }
        }

        Ok(())
    }

    fn write_plan_manifest(
        &self,
        destination_path: &str,
        plan: &ShootOrganizationPlan,
    ) -> Result<(), AppError> {
        let dst = Path::new(destination_path);
        if let Some(parent) = dst.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|err| {
                AppError::StorageFailure(format!(
                    "failed to create manifest directory '{}': {err}",
                    parent.display()
                ))
            })?;
        }

        let dto = ShootManifestDto::from(plan);
        let json = serde_json::to_string_pretty(&dto).map_err(|err| {
            AppError::StorageFailure(format!("failed to serialize shoot manifest: {err}"))
        })?;
        let tmp = dst.with_extension("json.tmp");
        fs::write(&tmp, json).map_err(|err| {
            AppError::StorageFailure(format!("failed to write temp manifest: {err}"))
        })?;
        fs::rename(&tmp, dst).map_err(|err| {
            AppError::StorageFailure(format!("failed to commit manifest '{}': {err}", dst.display()))
        })?;
        Ok(())
    }
}
