use kroma_app::error::AppError;
use ort::environment::Environment;
use ort::session::Session;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

static ORT_ENV: OnceLock<Result<Environment, String>> = OnceLock::new();
static GPU_ACCELERATION_ENABLED: AtomicBool = AtomicBool::new(true);

pub fn is_gpu_acceleration_enabled() -> bool {
    GPU_ACCELERATION_ENABLED.load(Ordering::Relaxed)
}

pub fn set_gpu_acceleration_enabled(enabled: bool) {
    GPU_ACCELERATION_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Returns the human-readable hardware acceleration backend label for the current OS.
pub fn gpu_execution_provider_label() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "DirectML GPU"
    }
    #[cfg(target_os = "macos")]
    {
        "CoreML GPU"
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        "CPU"
    }
}

/// Checks whether a native GPU/NPU ONNX Execution Provider (DirectML on Windows,
/// CoreML / Apple Neural Engine on macOS) is available on this machine.
pub fn is_gpu_execution_provider_available() -> bool {
    #[cfg(target_os = "windows")]
    {
        use ort::ep::{DirectML, ExecutionProvider};
        DirectML::is_available().unwrap_or(false)
    }
    #[cfg(target_os = "macos")]
    {
        use ort::ep::{CoreML, ExecutionProvider};
        CoreML::is_available().unwrap_or(false)
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        false
    }
}

/// Attempts to create a hardware-accelerated ONNX Runtime `Session` using DirectML (Windows)
/// or CoreML (macOS Apple Silicon / Metal), returning `None` when GPU acceleration is disabled,
/// unavailable, or incompatible with the model so callers fall back cleanly to CPU.
pub fn try_build_gpu_session(env: &Environment, model_path: &Path) -> Option<Session> {
    if !is_gpu_acceleration_enabled() {
        return None;
    }

    #[cfg(target_os = "windows")]
    {
        use ort::ep::directml::PerformancePreference;
        use ort::ep::{DirectML, ExecutionProvider};
        use ort::session::builder::GraphOptimizationLevel;

        if !DirectML::is_available().unwrap_or(false) {
            return None;
        }
        // DirectML requires sequential execution & disabled mem pattern per ONNX Runtime spec
        let try_create = || -> ort::Result<Session> {
            Session::builder(env)?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_parallel_execution(false)?
                .with_memory_pattern(false)?
                .with_execution_providers([DirectML::default()
                    .with_performance_preference(PerformancePreference::HighPerformance)
                    .build()])?
                .commit_from_file(model_path)
        };
        try_create().ok()
    }

    #[cfg(target_os = "macos")]
    {
        use ort::ep::{CoreML, ExecutionProvider};
        use ort::session::builder::GraphOptimizationLevel;

        if !CoreML::is_available().unwrap_or(false) {
            return None;
        }
        let try_create = || -> ort::Result<Session> {
            Session::builder(env)?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_execution_providers([CoreML::default().build()])?
                .commit_from_file(model_path)
        };
        try_create().ok()
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        let _ = (env, model_path);
        None
    }
}

/// Initializes (once per process) and returns a shared ONNX Runtime `Environment`.
pub fn shared_ort_environment() -> Result<&'static Environment, AppError> {
    let result = ORT_ENV.get_or_init(|| {
        ort::init()
            .with_name("kroma-vision")
            .build()
            .map_err(|err| format!("failed to initialize ONNX Runtime environment: {err}"))
    });

    match result {
        Ok(env) => Ok(env),
        Err(msg) => Err(AppError::ImageProcessingFailure(msg.clone())),
    }
}
