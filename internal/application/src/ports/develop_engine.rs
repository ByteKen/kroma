use crate::error::AppError;
use kroma_domain::develop::{DevelopRecipe, NormalizedRect, RgbHistogram};
use kroma_domain::photo::PhotoId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputImageFormat {
    Jpeg { quality: u8 },
    Png,
    Tiff,
}

impl Default for OutputImageFormat {
    fn default() -> Self {
        Self::Jpeg { quality: 92 }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BatchRenderJob {
    pub photo_id: PhotoId,
    pub source_path: String,
    pub output_path: String,
    pub recipe: DevelopRecipe,
    pub format: OutputImageFormat,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderMetrics {
    pub output_width: u32,
    pub output_height: u32,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BatchRenderItemOutcome {
    Succeeded {
        photo_id: PhotoId,
        output_path: String,
        metrics: RenderMetrics,
    },
    Failed {
        photo_id: PhotoId,
        source_path: String,
        reason: String,
    },
}

/// Inward-owned port for parametric image rendering, histogram computation, and WB sampling.
/// Concrete adapters (e.g. Rayon + `image` crate, or future `wgpu` GPU shader pipeline) live in `kroma-infra`.
pub trait ImageDevelopPort: Send + Sync {
    fn render_photo(
        &self,
        source_path: &str,
        output_path: &str,
        recipe: &DevelopRecipe,
        format: OutputImageFormat,
    ) -> Result<RenderMetrics, AppError>;

    fn render_batch(&self, jobs: &[BatchRenderJob]) -> Vec<BatchRenderItemOutcome>;

    /// Parallel batch render that reports incremental completion `(completed, total, label)` as each
    /// job finishes. Default implementation delegates to `render_batch` for lightweight test doubles.
    fn render_batch_with_progress(
        &self,
        jobs: &[BatchRenderJob],
        on_item_complete: &(dyn Fn(usize, usize, &str) + Sync),
    ) -> Vec<BatchRenderItemOutcome> {
        let outcomes = self.render_batch(jobs);
        on_item_complete(jobs.len(), jobs.len(), "");
        outcomes
    }

    fn sample_average_rgb(
        &self,
        source_path: &str,
        region: NormalizedRect,
    ) -> Result<(f32, f32, f32), AppError>;

    fn compute_histogram(
        &self,
        source_path: &str,
        recipe: &DevelopRecipe,
    ) -> Result<RgbHistogram, AppError>;
}
