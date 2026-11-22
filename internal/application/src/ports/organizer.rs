use crate::error::AppError;
use kroma_domain::identity::ShootOrganizationPlan;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrganizeTransferMode {
    Copy,
    Move,
    HardLink,
}

/// Inward-owned port for writing organized folder hierarchies and shoot manifests to disk.
pub trait FileOrganizerPort: Send + Sync {
    fn transfer_file(
        &self,
        source_path: &str,
        destination_path: &str,
        mode: OrganizeTransferMode,
    ) -> Result<(), AppError>;

    fn write_plan_manifest(
        &self,
        destination_path: &str,
        plan: &ShootOrganizationPlan,
    ) -> Result<(), AppError>;
}
