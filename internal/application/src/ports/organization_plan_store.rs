use crate::error::AppError;
use kroma_domain::identity::ShootOrganizationPlan;

/// Inward-owned port for persisting and restoring non-destructive subject organization plans
/// (clusters, subject renames, splits, merges, manual matches, and review queues).
pub trait OrganizationPlanStorePort: Send + Sync {
    fn load_plan(&self) -> Result<Option<ShootOrganizationPlan>, AppError>;
    fn save_plan(&self, plan: &ShootOrganizationPlan) -> Result<(), AppError>;
    fn clear_plan(&self) -> Result<(), AppError>;
}
