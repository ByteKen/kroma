use kroma_app::ports::develop_engine::{BatchRenderItemOutcome, OutputImageFormat};
use kroma_app::ports::organizer::OrganizeTransferMode;
use kroma_app::use_cases::auto_wb_from_slate::{
    CalibrateWhiteBalanceFromSlateInput, CalibrateWhiteBalanceFromSlateUseCase,
};
use kroma_app::use_cases::batch_nudge_develop::{
    BatchRelativeNudgeInput, BatchRelativeNudgeUseCase,
};
use kroma_app::use_cases::batch_render::{
    BatchRenderPhotoTarget, BatchRenderPhotosInput, BatchRenderPhotosUseCase,
};
use kroma_app::use_cases::cluster_shoot::{
    ClusterShootFacesUseCase, ClusterShootInput, PhotoScanTarget,
};
use kroma_app::use_cases::edit_photo::{
    SavePhotoDevelopRecipeInput, SavePhotoDevelopRecipeUseCase,
};
use kroma_app::use_cases::organize_shoot::{
    OrganizeExecutionMode, OrganizeShootFoldersUseCase, OrganizeShootInput,
};
use kroma_domain::develop::{
    DevelopRecipe, Lut3DReference, NormalizedRect, RelativeDevelopDelta, WhiteBalance,
};
use kroma_domain::identity::{parse_filename_capture_hints, SimilarityThresholds};
use kroma_domain::organization::OutputNamingTemplate;
use kroma_domain::photo::PhotoId;
use kroma_infra::develop::image_decoder::{
    extract_exif_capture_timestamp_epoch_sec, is_supported_photo_path,
};
use kroma_infra::develop::pixel_engine::RayonImageDevelopEngine;
use kroma_infra::organizer::local_file_organizer::LocalFileOrganizer;
use kroma_infra::storage::json_sidecar_store::JsonSidecarRecipeStore;
use kroma_infra::vision::sface_recognizer::SFaceOnnxRecognizer;
use kroma_infra::vision::yunet_detector::{YuNetConfig, YuNetOnnxDetector};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::UNIX_EPOCH;

/// Humble CLI controller: parses arguments and invokes application use cases without owning domain rules.
fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || matches!(args[0].as_str(), "-h" | "--help" | "help") {
        print_usage();
        return ExitCode::SUCCESS;
    }

    let result = match args[0].as_str() {
        "batch-edit" => run_batch_edit(&args[1..]),
        "wb-from-slate" => run_wb_from_slate(&args[1..]),
        "batch-nudge" => run_batch_nudge(&args[1..]),
        "organize" => run_organize(&args[1..]),
        other => Err(format!(
            "unknown subcommand '{other}'. Run `kroma --help` for usage."
        )),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run_batch_edit(args: &[String]) -> Result<(), String> {
    let input_path = required_flag(args, "--input")?;
    let output_dir = required_flag(args, "--output")?;
    let sidecars_dir = optional_flag(args, "--sidecars")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(&output_dir).join(".kroma_sidecars"));

    let recipe = parse_recipe_flags(args)?;
    let format = parse_output_format(args)?;
    let out_ext = format_extension(format);

    let source_files = collect_supported_images(Path::new(&input_path))?;
    if source_files.is_empty() {
        return Err(format!(
            "no supported RAW or standard images found in '{input_path}'"
        ));
    }

    let store = JsonSidecarRecipeStore::new(&sidecars_dir);
    let engine = RayonImageDevelopEngine::new();
    let save_uc = SavePhotoDevelopRecipeUseCase::new(&store);

    let mut targets = Vec::with_capacity(source_files.len());
    for file_path in source_files {
        let stem = file_stem_str(&file_path)?;
        let photo_id = PhotoId::new(&stem).map_err(|e| e.to_string())?;

        save_uc
            .execute(SavePhotoDevelopRecipeInput {
                photo_id: photo_id.clone(),
                recipe: recipe.clone(),
            })
            .map_err(|e| e.to_string())?;

        let out_file = Path::new(&output_dir).join(format!("{stem}.{out_ext}"));
        targets.push(BatchRenderPhotoTarget {
            photo_id,
            source_path: file_path.to_string_lossy().to_string(),
            output_path: out_file.to_string_lossy().to_string(),
        });
    }

    let render_uc = BatchRenderPhotosUseCase::new(&engine, &store);
    let result = render_uc
        .execute(BatchRenderPhotosInput { targets, format })
        .map_err(|e| e.to_string())?;

    print_render_summary(&result.outcomes, result.succeeded_count, result.failed_count);
    Ok(())
}

fn run_organize(args: &[String]) -> Result<(), String> {
    let input_path = required_flag(args, "--input")?;
    let output_dir = required_flag(args, "--output")?;
    let yunet_path = required_flag(args, "--yunet")?;
    let sface_path = required_flag(args, "--sface")?;

    let verified_threshold = optional_f32(args, "--threshold")?.unwrap_or(0.44);
    let review_threshold = (verified_threshold - 0.09).clamp(0.0, verified_threshold);
    let thresholds =
        SimilarityThresholds::new(verified_threshold, review_threshold).map_err(|e| e.to_string())?;

    let mode = match optional_flag(args, "--mode")
        .as_deref()
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        None | Some("copy") => OrganizeExecutionMode::TransferOriginals(OrganizeTransferMode::Copy),
        Some("move") => OrganizeExecutionMode::TransferOriginals(OrganizeTransferMode::Move),
        Some("hardlink") => {
            OrganizeExecutionMode::TransferOriginals(OrganizeTransferMode::HardLink)
        }
        Some("develop") => OrganizeExecutionMode::DevelopAndExport {
            format: parse_output_format(args)?,
        },
        Some(other) => {
            return Err(format!(
                "unsupported --mode '{other}' (expected copy, move, hardlink, or develop)"
            ));
        }
    };

    let source_files = collect_supported_images(Path::new(&input_path))?;
    if source_files.is_empty() {
        return Err(format!(
            "no supported RAW or standard images found in '{input_path}'"
        ));
    }

    println!(
        "Loading ONNX models (YuNet + SFace) and scanning {} photos...",
        source_files.len()
    );
    let detector =
        YuNetOnnxDetector::from_model_file(&yunet_path, YuNetConfig::default()).map_err(|e| e.to_string())?;
    let recognizer =
        SFaceOnnxRecognizer::from_model_file(&sface_path, 4).map_err(|e| e.to_string())?;

    let mut scan_targets = Vec::with_capacity(source_files.len());
    for file_path in &source_files {
        let stem = file_stem_str(file_path)?;
        let photo_id = PhotoId::new(&stem).map_err(|e| e.to_string())?;
        let source_str = file_path.to_string_lossy().to_string();
        let exact_ts = extract_exif_capture_timestamp_epoch_sec(&source_str)
            .or_else(|| parse_filename_capture_hints(&source_str).filename_timestamp_epoch_sec);
        let ts = exact_ts.unwrap_or_else(|| {
            fs::metadata(file_path)
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0)
        });
        scan_targets.push(PhotoScanTarget {
            photo_id,
            source_path: source_str,
            capture_timestamp_epoch_sec: ts,
            exact_capture_timestamp_epoch_sec: exact_ts,
        });
    }

    // Face-first clustering (slate OCR is optional; None runs pure YuNet + ArcFace/SFace clustering!)
    let cluster_uc = ClusterShootFacesUseCase::new(&detector, &recognizer, None);
    let clustered = cluster_uc
        .execute(ClusterShootInput {
            photos: scan_targets,
            thresholds,
            group_face_min_area_ratio: 0.35,
            use_temporal_sequence_assist: true,
        })
        .map_err(|e| e.to_string())?;

    println!(
        "Clustered {} photos -> {} people, {} group shots, {} no-face, {} ambiguous review, {} scan errors",
        clustered.scanned_photo_count,
        clustered.plan.clusters.len(),
        clustered.plan.group_photos.len(),
        clustered.plan.no_face_photos.len(),
        clustered.plan.ambiguous_review_queue.len(),
        clustered.failures.len()
    );

    let naming_template = optional_flag(args, "--pattern")
        .map(|p| OutputNamingTemplate::new(p, true))
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or_else(OutputNamingTemplate::default_portrait_template);

    let organizer = LocalFileOrganizer::new();
    let organize_uc = OrganizeShootFoldersUseCase::new(&organizer);
    let develop_engine = RayonImageDevelopEngine::new();
    let sidecars_dir = optional_flag(args, "--sidecars")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(&output_dir).join(".kroma_sidecars"));
    let recipe_store = JsonSidecarRecipeStore::new(&sidecars_dir);

    if matches!(mode, OrganizeExecutionMode::DevelopAndExport { .. }) {
        let shared_recipe = parse_recipe_flags(args)?;
        let save_uc = SavePhotoDevelopRecipeUseCase::new(&recipe_store);
        for file_path in &source_files {
            let stem = file_stem_str(file_path)?;
            let photo_id = PhotoId::new(&stem).map_err(|e| e.to_string())?;
            save_uc
                .execute(SavePhotoDevelopRecipeInput {
                    photo_id,
                    recipe: shared_recipe.clone(),
                })
                .map_err(|e| e.to_string())?;
        }
    }

    let organized = organize_uc
        .execute_with_optional_develop(
            OrganizeShootInput {
                plan: clustered.plan,
                output_root: output_dir,
                naming_template,
                execution_mode: mode,
                archive_excluded_slates: true,
                write_manifest_json: true,
                selected_photo_ids: None,
            },
            Some(&develop_engine),
            Some(&recipe_store),
        )
        .map_err(|e| e.to_string())?;

    println!(
        "Organized {} files ({} failures).",
        organized.organized_files.len(),
        organized.failures.len()
    );
    if let Some(manifest) = organized.manifest_path {
        println!("Manifest written: {manifest}");
    }
    Ok(())
}

fn run_wb_from_slate(args: &[String]) -> Result<(), String> {
    let slate_path = required_flag(args, "--slate")?;
    let input_path = required_flag(args, "--input")?;
    let output_dir = required_flag(args, "--output")?;
    let sidecars_dir = Path::new(&output_dir).join(".kroma_sidecars");

    let source_files = collect_supported_images(Path::new(&input_path))?;
    if source_files.is_empty() {
        return Err(format!(
            "no supported RAW or standard images found in '{input_path}'"
        ));
    }

    let store = JsonSidecarRecipeStore::new(&sidecars_dir);
    let engine = RayonImageDevelopEngine::new();

    let mut photo_ids = Vec::with_capacity(source_files.len());
    let mut targets = Vec::with_capacity(source_files.len());

    for file_path in source_files {
        let stem = file_stem_str(&file_path)?;
        let photo_id = PhotoId::new(&stem).map_err(|e| e.to_string())?;
        photo_ids.push(photo_id.clone());

        let out_file = Path::new(&output_dir).join(format!("{stem}.jpg"));
        targets.push(BatchRenderPhotoTarget {
            photo_id,
            source_path: file_path.to_string_lossy().to_string(),
            output_path: out_file.to_string_lossy().to_string(),
        });
    }

    let region = NormalizedRect::new(0.35, 0.55, 0.30, 0.20).map_err(|e| e.to_string())?;
    let wb_uc = CalibrateWhiteBalanceFromSlateUseCase::new(&engine, &store);
    let wb_out = wb_uc
        .execute(CalibrateWhiteBalanceFromSlateInput {
            slate_source_path: slate_path,
            whiteboard_region: region,
            target_photo_ids: photo_ids,
        })
        .map_err(|e| e.to_string())?;

    println!(
        "Calibrated White Balance from Slate: {}K (Tint {:+}) -> applied to {} photos",
        wb_out.calibrated_wb.temperature_kelvin,
        wb_out.calibrated_wb.tint,
        wb_out.applied_photo_count
    );

    let render_uc = BatchRenderPhotosUseCase::new(&engine, &store);
    let rendered = render_uc
        .execute(BatchRenderPhotosInput {
            targets,
            format: OutputImageFormat::default(),
        })
        .map_err(|e| e.to_string())?;

    print_render_summary(
        &rendered.outcomes,
        rendered.succeeded_count,
        rendered.failed_count,
    );
    Ok(())
}

fn run_batch_nudge(args: &[String]) -> Result<(), String> {
    let sidecars_dir = required_flag(args, "--sidecars")?;
    let delta = RelativeDevelopDelta {
        exposure_delta_ev: optional_f32(args, "--exposure-delta")?.unwrap_or(0.0),
        temperature_delta_kelvin: optional_i32(args, "--temp-delta")?.unwrap_or(0),
        tint_delta: optional_i16(args, "--tint-delta")?.unwrap_or(0),
        contrast_delta: optional_i16(args, "--contrast-delta")?.unwrap_or(0),
        highlights_delta: optional_i16(args, "--highlights-delta")?.unwrap_or(0),
        shadows_delta: optional_i16(args, "--shadows-delta")?.unwrap_or(0),
        vibrance_delta: optional_i16(args, "--vibrance-delta")?.unwrap_or(0),
        saturation_delta: optional_i16(args, "--saturation-delta")?.unwrap_or(0),
    };

    let mut photo_ids = Vec::new();
    let entries = fs::read_dir(&sidecars_dir)
        .map_err(|err| format!("failed to read sidecars directory '{sidecars_dir}': {err}"))?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if let Some(stem) = name.strip_suffix(".kroma.json") {
            photo_ids.push(PhotoId::new(stem).map_err(|e| e.to_string())?);
        }
    }

    let store = JsonSidecarRecipeStore::new(&sidecars_dir);
    let nudge_uc = BatchRelativeNudgeUseCase::new(&store);
    let out = nudge_uc
        .execute(BatchRelativeNudgeInput {
            target_photo_ids: photo_ids,
            delta,
        })
        .map_err(|e| e.to_string())?;

    println!(
        "Relative batch nudge applied to {} sidecars ({} failures)",
        out.updated_count,
        out.failed_ids.len()
    );
    Ok(())
}

fn parse_recipe_flags(args: &[String]) -> Result<DevelopRecipe, String> {
    let mut recipe = DevelopRecipe::default();
    if let Some(ev) = optional_f32(args, "--exposure")? {
        recipe.basic_tone.exposure_ev = ev;
    }
    if let Some(temp) = optional_u32(args, "--temp")? {
        recipe.white_balance = WhiteBalance::new(temp, recipe.white_balance.tint)
            .map_err(|e| e.to_string())?;
    }
    if let Some(tint) = optional_i16(args, "--tint")? {
        recipe.white_balance.tint = tint;
    }
    if let Some(val) = optional_i16(args, "--contrast")? {
        recipe.basic_tone.contrast = val;
    }
    if let Some(val) = optional_i16(args, "--highlights")? {
        recipe.basic_tone.highlights = val;
    }
    if let Some(val) = optional_i16(args, "--shadows")? {
        recipe.basic_tone.shadows = val;
    }
    if let Some(val) = optional_i16(args, "--whites")? {
        recipe.basic_tone.whites = val;
    }
    if let Some(val) = optional_i16(args, "--blacks")? {
        recipe.basic_tone.blacks = val;
    }
    if let Some(val) = optional_i16(args, "--vibrance")? {
        recipe.basic_tone.vibrance = val;
    }
    if let Some(val) = optional_i16(args, "--saturation")? {
        recipe.basic_tone.saturation = val;
    }
    if let Some(val) = optional_i16(args, "--vignette")? {
        recipe.detail_optics.vignette_amount = val;
    }
    if let Some(val) = optional_u8(args, "--sharpen")? {
        recipe.detail_optics.sharpening_amount = val;
    }
    if let Some(cube_path) = optional_flag(args, "--lut") {
        let intensity = optional_f32(args, "--lut-intensity")?.unwrap_or(1.0);
        let lut_name = Path::new(&cube_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("CustomLUT")
            .to_string();
        recipe.lut_3d = Some(
            Lut3DReference::new(lut_name, cube_path, intensity).map_err(|e| e.to_string())?,
        );
    }
    Ok(recipe)
}

fn parse_output_format(args: &[String]) -> Result<OutputImageFormat, String> {
    let quality = optional_u8(args, "--quality")?.unwrap_or(92);
    match optional_flag(args, "--format")
        .as_deref()
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        None | Some("jpg" | "jpeg") => Ok(OutputImageFormat::Jpeg { quality }),
        Some("png") => Ok(OutputImageFormat::Png),
        Some("tif" | "tiff") => Ok(OutputImageFormat::Tiff),
        Some(other) => Err(format!(
            "unsupported --format '{other}' (expected jpeg, png, or tiff)"
        )),
    }
}

fn format_extension(format: OutputImageFormat) -> &'static str {
    match format {
        OutputImageFormat::Jpeg { .. } => "jpg",
        OutputImageFormat::Png => "png",
        OutputImageFormat::Tiff => "tiff",
    }
}

fn file_stem_str(path: &Path) -> Result<String, String> {
    path.file_stem()
        .and_then(|s| s.to_str())
        .map(str::to_string)
        .ok_or_else(|| format!("invalid file name '{}'", path.display()))
}

fn print_render_summary(outcomes: &[BatchRenderItemOutcome], succeeded: usize, failed: usize) {
    for outcome in outcomes {
        match outcome {
            BatchRenderItemOutcome::Succeeded {
                photo_id,
                output_path,
                metrics,
            } => println!(
                "  [OK] {} -> {} ({}x{}, {} ms)",
                photo_id.as_str(),
                output_path,
                metrics.output_width,
                metrics.output_height,
                metrics.elapsed_ms
            ),
            BatchRenderItemOutcome::Failed {
                photo_id,
                source_path,
                reason,
            } => eprintln!(
                "  [ERR] {} ({}): {}",
                photo_id.as_str(),
                source_path,
                reason
            ),
        }
    }
    println!("Batch complete: {succeeded} succeeded, {failed} failed.");
}

fn collect_supported_images(path: &Path) -> Result<Vec<PathBuf>, String> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    if !path.is_dir() {
        return Err(format!("input path '{}' does not exist", path.display()));
    }

    let mut files = Vec::new();
    let entries = fs::read_dir(path)
        .map_err(|err| format!("failed to read directory '{}': {err}", path.display()))?;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_file() && is_supported_photo_path(&p) {
            files.push(p);
        }
    }
    files.sort();
    Ok(files)
}

fn required_flag(args: &[String], name: &str) -> Result<String, String> {
    optional_flag(args, name).ok_or_else(|| format!("missing required flag {name}"))
}

fn optional_flag(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].clone())
}

fn optional_f32(args: &[String], name: &str) -> Result<Option<f32>, String> {
    optional_flag(args, name)
        .map(|v| {
            v.parse::<f32>()
                .map_err(|e| format!("invalid {name} '{v}': {e}"))
        })
        .transpose()
}

fn optional_i16(args: &[String], name: &str) -> Result<Option<i16>, String> {
    optional_flag(args, name)
        .map(|v| {
            v.parse::<i16>()
                .map_err(|e| format!("invalid {name} '{v}': {e}"))
        })
        .transpose()
}

fn optional_i32(args: &[String], name: &str) -> Result<Option<i32>, String> {
    optional_flag(args, name)
        .map(|v| {
            v.parse::<i32>()
                .map_err(|e| format!("invalid {name} '{v}': {e}"))
        })
        .transpose()
}

fn optional_u32(args: &[String], name: &str) -> Result<Option<u32>, String> {
    optional_flag(args, name)
        .map(|v| {
            v.parse::<u32>()
                .map_err(|e| format!("invalid {name} '{v}': {e}"))
        })
        .transpose()
}

fn optional_u8(args: &[String], name: &str) -> Result<Option<u8>, String> {
    optional_flag(args, name)
        .map(|v| {
            v.parse::<u8>()
                .map_err(|e| format!("invalid {name} '{v}': {e}"))
        })
        .transpose()
}

fn print_usage() {
    println!(
        "\
Kroma — Rust RAW/Photo Batch Develop & AI Organizer CLI

USAGE:
  kroma batch-edit --input <file|dir> --output <dir> [develop options]
      Supports 29 camera RAW formats (.ARW, .CR3, .CR2, .NEF, .DNG, .RAF, ...) + JPG/PNG/TIFF/WebP.

  kroma organize --input <dir> --output <dir> --yunet <yunet.onnx> --sface <sface.onnx> [options]
      --threshold <0.30..0.85>  SFace cosine match threshold (default 0.45)
      --mode <copy|move|hardlink|develop>  Transfer originals OR batch-develop into subject folders
      --pattern <template>      Output folder template (default: [Slate_Name]_[ID]/IMG_001.jpg)
      [plus any develop options when --mode develop is used]

  kroma wb-from-slate --slate <photo> --input <dir> --output <dir>
      Samples neutral white balance from the slate whiteboard and renders the batch.

  kroma batch-nudge --sidecars <dir> [--exposure-delta <EV>] [--temp-delta <K>]
      Applies Lightroom Quick-Develop relative nudges to saved .kroma.json sidecars."
    );
}
