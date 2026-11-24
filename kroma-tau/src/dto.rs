use kroma_domain::develop::{
    BasicTone, CameraCalibration, ColorGrading, ColorGradingWheel, CropAspectRatio, CropTransform,
    DetailAndOptics, DevelopRecipe, DevelopSyncMask, HslBandShift, HslColorMixer, Lut3DReference,
    NormalizedRect, ParametricToneCurve, RelativeDevelopDelta, RgbHistogram, ToneCurvePoint,
    WhiteBalance,
};
use kroma_domain::identity::{BoundingBox, MatchConfidence, Point2D, YuNetLandmarks};
use kroma_domain::masking::{
    AiMaskKind, BackgroundRemoval, BackgroundReplaceMode, BrushStroke, LocalAdjustments,
    MaskCombineOp, MaskComponent, MaskLayer,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScanProgressDto {
    pub phase: &'static str,
    pub completed: usize,
    pub total: usize,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PreviewPatchDto {
    pub photo_id: String,
    pub preview_path: Option<String>,
    pub rendered_preview_path: Option<String>,
    pub clear_rendered_preview: bool,
    pub captured_at_epoch_sec: Option<i64>,
    pub captured_at_label: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PreviewsReadyBatchDto {
    pub updates: Vec<PreviewPatchDto>,
    pub completed: usize,
    pub total: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ThumbnailRenderProgressDto {
    pub active: bool,
    pub completed: usize,
    pub total: usize,
    pub mode: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeStatusDto {
    pub yunet_model_path: Option<String>,
    pub sface_model_path: Option<String>,
    pub detector_model_label: String,
    pub recognition_model_label: String,
    pub default_yunet_model_path: Option<String>,
    pub default_sface_model_path: Option<String>,
    pub default_detector_model_label: String,
    pub default_recognition_model_label: String,
    pub is_custom_yunet: bool,
    pub is_custom_sface: bool,
    pub models_ready: bool,
    pub ocr_models_ready: bool,
    pub ocr_det_model_label: String,
    pub ocr_rec_model_label: String,
    pub ocr_det_model_path: Option<String>,
    pub ocr_rec_model_path: Option<String>,
    pub segmenter_model_path: Option<String>,
    pub segmenter_model_label: String,
    pub default_segmenter_model_path: Option<String>,
    pub default_segmenter_model_label: String,
    pub is_custom_segmenter: bool,
    pub segmenter_models_ready: bool,
    pub supported_raw_count: usize,
    pub backend_label: String,
    pub develop_gpu_backend_label: String,
    pub develop_gpu_adapter_name: Option<String>,
    pub app_version: String,
    pub gpu_available: bool,
    pub gpu_acceleration_enabled: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StorageCacheSummaryDto {
    pub cache_dir_path: String,
    pub is_custom_cache_dir: bool,
    pub preview_cache_bytes: u64,
    pub preview_file_count: usize,
    pub face_cache_bytes: u64,
    pub face_cache_file_count: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClearCacheResultDto {
    pub removed_files: usize,
    pub freed_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelManifestEntryStatusDto {
    pub id: String,
    pub name: String,
    pub category: String,
    pub version: String,
    pub installed_version: Option<String>,
    pub relative_path: String,
    pub size_bytes: u64,
    pub required: bool,
    pub status: &'static str,
    pub resolved_path: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelManifestStatusDto {
    pub schema_version: u32,
    pub release_tag: String,
    pub models_dir: String,
    pub all_ready: bool,
    pub missing_count: usize,
    pub outdated_count: usize,
    pub pending_download_bytes: u64,
    pub entries: Vec<ModelManifestEntryStatusDto>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelDownloadProgressDto {
    pub phase: &'static str,
    pub model_id: String,
    pub model_name: String,
    pub index: usize,
    pub total_count: usize,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub overall_downloaded_bytes: u64,
    pub overall_total_bytes: u64,
    pub bytes_per_second: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelDownloadOutcomeDto {
    pub manifest: ModelManifestStatusDto,
    pub runtime: RuntimeStatusDto,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppUpdateDownloadProgressDto {
    pub phase: &'static str,
    pub asset_name: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub bytes_per_second: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportedCubeLutDto {
    pub name: String,
    pub cube_path: String,
    pub grid_size: usize,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GeneratedAiMaskDto {
    pub kind: String,
    pub matte_width: u16,
    pub matte_height: u16,
    pub matte_rle_b64: String,
    pub engine_label: String,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrTextLineDto {
    pub text: String,
    pub confidence: f32,
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PhotoOcrResultDto {
    pub photo_id: String,
    pub file_name: String,
    pub preview_path: String,
    pub proposed_name: Option<String>,
    pub proposed_code: Option<String>,
    pub confidence: f32,
    pub lines: Vec<OcrTextLineDto>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClusterSlateOcrResultDto {
    pub cluster_id: String,
    pub current_name: String,
    pub slate_photo_id: Option<String>,
    pub slate_file_name: Option<String>,
    pub slate_preview_path: Option<String>,
    pub proposed_name: Option<String>,
    pub proposed_code: Option<String>,
    pub confidence: f32,
    pub scanned_frame_count: usize,
    pub lines: Vec<OcrTextLineDto>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SlateRenameApplyItemDto {
    pub cluster_id: String,
    pub next_name: String,
    pub subject_code: Option<String>,
    pub slate_photo_id: Option<String>,
    pub ocr_raw_text: Option<String>,
    pub ocr_confidence: Option<f32>,
    pub mark_as_slate_photo: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInspectionDto {
    pub compatibility: &'static str, // "compatible" | "warning" | "incompatible"
    pub summary: String,
    pub warnings: Vec<String>,
    pub detected_label: String,
    pub input_shape: String,
    pub output_shape: String,
    pub embedding_dim: Option<usize>,
    pub file_size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WhiteBalanceDto {
    pub temperature_kelvin: u32,
    pub tint: i16,
}

fn default_fifty_u8() -> u8 {
    50
}

fn default_twenty_five_u8() -> u8 {
    25
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BasicToneDto {
    #[serde(default)]
    pub black_and_white: bool,
    pub exposure_ev: f32,
    pub contrast: i16,
    pub highlights: i16,
    pub shadows: i16,
    pub whites: i16,
    pub blacks: i16,
    pub texture: i16,
    pub clarity: i16,
    pub dehaze: i16,
    pub vibrance: i16,
    pub saturation: i16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToneCurvePointDto {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ParametricToneCurveDto {
    pub highlights: i16,
    pub lights: i16,
    pub darks: i16,
    pub shadows: i16,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub points: Vec<ToneCurvePointDto>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub red_points: Vec<ToneCurvePointDto>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub green_points: Vec<ToneCurvePointDto>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blue_points: Vec<ToneCurvePointDto>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HslChannelShiftDto {
    pub hue: i16,
    pub saturation: i16,
    pub luminance: i16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HslColorMixerDto {
    pub red: HslChannelShiftDto,
    pub orange: HslChannelShiftDto,
    pub yellow: HslChannelShiftDto,
    pub green: HslChannelShiftDto,
    pub aqua: HslChannelShiftDto,
    pub blue: HslChannelShiftDto,
    pub purple: HslChannelShiftDto,
    pub magenta: HslChannelShiftDto,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ColorGradingWheelDto {
    #[serde(default)]
    pub hue_deg: f32,
    #[serde(default)]
    pub saturation: u8,
    #[serde(default)]
    pub luminance: i16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ColorGradingDto {
    #[serde(default)]
    pub shadows: ColorGradingWheelDto,
    #[serde(default)]
    pub midtones: ColorGradingWheelDto,
    #[serde(default)]
    pub highlights: ColorGradingWheelDto,
    #[serde(default = "default_fifty_u8")]
    pub blending: u8,
    #[serde(default)]
    pub balance: i16,
}

impl Default for ColorGradingDto {
    fn default() -> Self {
        Self {
            shadows: ColorGradingWheelDto::default(),
            midtones: ColorGradingWheelDto::default(),
            highlights: ColorGradingWheelDto::default(),
            blending: 50,
            balance: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CameraCalibrationDto {
    #[serde(default)]
    pub shadows_tint: i16,
    #[serde(default)]
    pub red_hue: i16,
    #[serde(default)]
    pub red_saturation: i16,
    #[serde(default)]
    pub green_hue: i16,
    #[serde(default)]
    pub green_saturation: i16,
    #[serde(default)]
    pub blue_hue: i16,
    #[serde(default)]
    pub blue_saturation: i16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Lut3DDto {
    pub name: String,
    pub cube_path: String,
    pub intensity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DetailOpticsDto {
    pub sharpening_amount: u8,
    pub luminance_noise_reduction: u8,
    #[serde(default)]
    pub color_noise_reduction: u8,
    pub vignette_amount: i16,
    #[serde(default = "default_fifty_u8")]
    pub vignette_midpoint: u8,
    #[serde(default = "default_fifty_u8")]
    pub vignette_feather: u8,
    #[serde(default)]
    pub grain_amount: u8,
    #[serde(default = "default_twenty_five_u8")]
    pub grain_size: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CropRectDto {
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
}

impl Default for CropRectDto {
    fn default() -> Self {
        Self {
            left: 0.0,
            top: 0.0,
            width: 1.0,
            height: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CropTransformDto {
    #[serde(default)]
    pub rect: CropRectDto,
    pub rotation_deg: f32,
    pub aspect_ratio: String,
    #[serde(default)]
    pub flip_horizontal: bool,
    #[serde(default)]
    pub flip_vertical: bool,
    #[serde(default)]
    pub quarter_turns: u8,
    pub eye_level_auto_locked: bool,
}

fn default_hundred_u8() -> u8 {
    100
}

fn default_fifty_f32() -> f32 {
    50.0
}

fn default_twenty_five_f32() -> f32 {
    25.0
}

fn default_op_add() -> String {
    "add".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrushStrokeDto {
    pub size: f32,
    pub feather: f32,
    pub flow: f32,
    #[serde(default)]
    pub points: Vec<[f32; 2]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MaskComponentDto {
    #[serde(rename_all = "camelCase")]
    LinearGradient {
        id: String,
        #[serde(default = "default_op_add")]
        op: String,
        #[serde(default)]
        inverted: bool,
        start_x: f32,
        start_y: f32,
        end_x: f32,
        end_y: f32,
        #[serde(default = "default_fifty_f32")]
        feather: f32,
    },
    #[serde(rename_all = "camelCase")]
    RadialGradient {
        id: String,
        #[serde(default = "default_op_add")]
        op: String,
        #[serde(default)]
        inverted: bool,
        center_x: f32,
        center_y: f32,
        radius_x: f32,
        radius_y: f32,
        #[serde(default)]
        rotation_deg: f32,
        #[serde(default = "default_fifty_f32")]
        feather: f32,
    },
    #[serde(rename_all = "camelCase")]
    Brush {
        id: String,
        #[serde(default = "default_op_add")]
        op: String,
        #[serde(default)]
        inverted: bool,
        #[serde(default)]
        strokes: Vec<BrushStrokeDto>,
    },
    #[serde(rename_all = "camelCase")]
    LuminanceRange {
        id: String,
        #[serde(default = "default_op_add")]
        op: String,
        #[serde(default)]
        inverted: bool,
        min_luma: f32,
        max_luma: f32,
        feather: f32,
    },
    #[serde(rename_all = "camelCase")]
    ColorRange {
        id: String,
        #[serde(default = "default_op_add")]
        op: String,
        #[serde(default)]
        inverted: bool,
        target_hue_deg: f32,
        target_sat: f32,
        target_luma: f32,
        tolerance: f32,
        feather: f32,
    },
    #[serde(rename_all = "camelCase")]
    AiMatte {
        id: String,
        #[serde(default = "default_op_add")]
        op: String,
        #[serde(default)]
        inverted: bool,
        kind: String,
        #[serde(default = "default_twenty_five_f32")]
        feather: f32,
        matte_width: u16,
        matte_height: u16,
        #[serde(default)]
        matte_rle_b64: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct LocalAdjustmentsDto {
    #[serde(default)]
    pub temperature_shift: i16,
    #[serde(default)]
    pub tint_shift: i16,
    #[serde(default)]
    pub exposure_ev: f32,
    #[serde(default)]
    pub contrast: i16,
    #[serde(default)]
    pub highlights: i16,
    #[serde(default)]
    pub shadows: i16,
    #[serde(default)]
    pub whites: i16,
    #[serde(default)]
    pub blacks: i16,
    #[serde(default)]
    pub texture: i16,
    #[serde(default)]
    pub clarity: i16,
    #[serde(default)]
    pub dehaze: i16,
    #[serde(default)]
    pub saturation: i16,
    #[serde(default)]
    pub sharpness: i16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MaskLayerDto {
    pub id: String,
    pub name: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub inverted: bool,
    #[serde(default = "default_hundred_u8")]
    pub opacity: u8,
    #[serde(default)]
    pub components: Vec<MaskComponentDto>,
    #[serde(default)]
    pub adjustments: LocalAdjustmentsDto,
}

fn default_white_rgb() -> [u8; 3] {
    [255, 255, 255]
}

fn default_dark_charcoal_rgb() -> [u8; 3] {
    [20, 22, 27]
}

fn default_bg_mode() -> String {
    "original".to_string()
}

fn default_grad_style() -> String {
    "radial".to_string()
}

fn default_grad_angle_f32() -> f32 {
    180.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BackgroundRemovalDto {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_bg_mode")]
    pub mode: String,
    #[serde(default = "default_white_rgb")]
    pub color_rgb: [u8; 3],
    #[serde(default = "default_dark_charcoal_rgb")]
    pub color2_rgb: [u8; 3],
    #[serde(default = "default_grad_style")]
    pub gradient_style: String,
    #[serde(default = "default_grad_angle_f32")]
    pub gradient_angle_deg: f32,
    #[serde(default = "default_twenty_five_f32")]
    pub feather: f32,
    #[serde(default)]
    pub matte_width: u16,
    #[serde(default)]
    pub matte_height: u16,
    #[serde(default)]
    pub matte_rle_b64: String,
}

impl Default for BackgroundRemovalDto {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: default_bg_mode(),
            color_rgb: default_white_rgb(),
            color2_rgb: default_dark_charcoal_rgb(),
            gradient_style: default_grad_style(),
            gradient_angle_deg: default_grad_angle_f32(),
            feather: 25.0,
            matte_width: 0,
            matte_height: 0,
            matte_rle_b64: String::new(),
        }
    }
}

impl From<&BackgroundRemoval> for BackgroundRemovalDto {
    fn from(b: &BackgroundRemoval) -> Self {
        Self {
            enabled: b.enabled,
            mode: b.mode.as_str().to_string(),
            color_rgb: b.color_rgb,
            color2_rgb: b.color2_rgb,
            gradient_style: b.gradient_style.as_str().to_string(),
            gradient_angle_deg: b.gradient_angle_deg,
            feather: b.feather,
            matte_width: b.matte_width,
            matte_height: b.matte_height,
            matte_rle_b64: b.matte_rle_b64.clone(),
        }
    }
}

impl From<BackgroundRemovalDto> for BackgroundRemoval {
    fn from(d: BackgroundRemovalDto) -> Self {
        Self {
            enabled: d.enabled,
            mode: BackgroundReplaceMode::parse(&d.mode),
            color_rgb: d.color_rgb,
            color2_rgb: d.color2_rgb,
            gradient_style: kroma_domain::masking::BackgroundGradientStyle::parse(
                &d.gradient_style,
            ),
            gradient_angle_deg: d.gradient_angle_deg,
            feather: d.feather,
            matte_width: d.matte_width,
            matte_height: d.matte_height,
            matte_rle_b64: d.matte_rle_b64,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DevelopRecipeDto {
    pub white_balance: WhiteBalanceDto,
    pub basic_tone: BasicToneDto,
    pub tone_curve: ParametricToneCurveDto,
    pub hsl_mixer: HslColorMixerDto,
    #[serde(default)]
    pub color_grading: ColorGradingDto,
    #[serde(default)]
    pub camera_calibration: CameraCalibrationDto,
    #[serde(rename = "lut3d")]
    pub lut_3d: Option<Lut3DDto>,
    pub detail_optics: DetailOpticsDto,
    pub crop_transform: CropTransformDto,
    #[serde(default)]
    pub masks: Vec<MaskLayerDto>,
    #[serde(default)]
    pub background_removal: BackgroundRemovalDto,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DevelopSyncMaskDto {
    pub white_balance: bool,
    pub basic_tone: bool,
    pub tone_curve: bool,
    pub hsl_mixer: bool,
    #[serde(default = "default_true")]
    pub color_grading: bool,
    #[serde(default = "default_true")]
    pub camera_calibration: bool,
    #[serde(rename = "lut3d")]
    pub lut_3d: bool,
    pub detail_optics: bool,
    pub crop_transform: bool,
    #[serde(default)]
    pub masks: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelativeDevelopDeltaDto {
    pub exposure_delta_ev: Option<f32>,
    pub temperature_delta_kelvin: Option<i32>,
    pub tint_delta: Option<i16>,
    pub contrast_delta: Option<i16>,
    pub highlights_delta: Option<i16>,
    pub shadows_delta: Option<i16>,
    pub vibrance_delta: Option<i16>,
    pub saturation_delta: Option<i16>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Point2DPctDto {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FaceOverlayDto {
    pub top_pct: f32,
    pub left_pct: f32,
    pub width_pct: f32,
    pub height_pct: f32,
    pub right_eye: Point2DPctDto,
    pub left_eye: Point2DPctDto,
    pub nose_tip: Point2DPctDto,
    pub right_mouth: Point2DPctDto,
    pub left_mouth: Point2DPctDto,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DirectorySummaryDto {
    pub id: String,
    pub name: String,
    pub photo_count: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VirtualFolderDto {
    pub id: String,
    pub name: String,
    pub parent_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PhotoDto {
    pub id: String,
    pub file_name: String,
    pub directory_id: String,
    pub virtual_folder_id: Option<String>,
    pub source_path: String,
    pub preview_path: String,
    pub rendered_preview_path: Option<String>,
    pub file_size_bytes: u64,
    pub is_slate_anchor: bool,
    pub cluster_id: Option<String>,
    pub sface_cosine_similarity: f32,
    pub confidence: &'static str,
    pub captured_at_epoch_sec: Option<i64>,
    pub captured_at_label: Option<String>,
    pub face_overlay: Option<FaceOverlayDto>,
    pub develop_recipe: DevelopRecipeDto,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClusterDto {
    pub id: String,
    pub display_name: String,
    pub subject_code: Option<String>,
    pub has_slate: bool,
    pub representative_photo_id: String,
    pub representative_preview_path: String,
    pub photos: Vec<PhotoDto>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AmbiguousCandidateDto {
    pub photo_id: String,
    pub file_name: String,
    pub source_path: String,
    pub preview_path: String,
    pub candidate_cluster_id: String,
    pub cosine_similarity: f32,
    pub face_overlay: FaceOverlayDto,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ShootSessionDto {
    pub session_mode: String,
    pub folder_path: String,
    pub project_name: String,
    pub is_clustered: bool,
    pub total_photos: usize,
    pub scan_elapsed_ms: u64,
    pub scan_error_count: usize,
    pub directories: Vec<DirectorySummaryDto>,
    pub virtual_folders: Vec<VirtualFolderDto>,
    pub clusters: Vec<ClusterDto>,
    pub ambiguous_review: Vec<AmbiguousCandidateDto>,
    pub group_photo_count: usize,
    pub no_face_photo_count: usize,
    pub can_undo: bool,
    pub can_redo: bool,
    pub undo_label: Option<String>,
    pub redo_label: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HistogramDto {
    pub red: Vec<u32>,
    pub green: Vec<u32>,
    pub blue: Vec<u32>,
    pub luminance: Vec<u32>,
    pub total_pixels: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhotoRecipeSaveItemDto {
    pub photo_id: String,
    pub recipe: DevelopRecipeDto,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchSyncResultDto {
    pub synced_count: usize,
    pub failed_count: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportShootRequestDto {
    pub output_root: String,
    pub naming_pattern: String,
    pub exclude_slate_photo: bool,
    pub archive_excluded_slates: bool,
    pub mode: String,   // "develop_and_raw" | "develop" | "copy" | "move" | "hardlink"
    pub format: String, // "jpeg" | "png" | "tiff"
    pub quality: u8,
    #[serde(default)]
    pub photo_ids: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportShootResultDto {
    pub organized_count: usize,
    pub failed_count: usize,
    pub manifest_path: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EstimateExportSizeRequestDto {
    pub exclude_slate_photo: bool,
    pub archive_excluded_slates: bool,
    pub mode: String,   // "develop_and_raw" | "develop" | "copy" | "move" | "hardlink"
    pub format: String, // "jpeg" | "png" | "tiff"
    pub quality: u8,
    #[serde(default)]
    pub photo_ids: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EstimateExportSizeResponseDto {
    pub extra_disk_bytes: u64,
    pub edited_bytes: u64,
    pub originals_bytes: u64,
    pub total_source_bytes: u64,
    pub file_count: usize,
    pub sampled_photo_count: usize,
    pub is_exact: bool,
}

pub fn build_face_overlay(bbox: BoundingBox, landmarks: YuNetLandmarks) -> FaceOverlayDto {
    let bw = bbox.width.max(1e-4);
    let bh = bbox.height.max(1e-4);
    let rel = |pt: Point2D| Point2DPctDto {
        x: (((pt.x - bbox.x) / bw) * 100.0).clamp(0.0, 100.0),
        y: (((pt.y - bbox.y) / bh) * 100.0).clamp(0.0, 100.0),
    };

    FaceOverlayDto {
        top_pct: (bbox.y * 100.0).clamp(0.0, 100.0),
        left_pct: (bbox.x * 100.0).clamp(0.0, 100.0),
        width_pct: (bbox.width * 100.0).clamp(0.0, 100.0),
        height_pct: (bbox.height * 100.0).clamp(0.0, 100.0),
        right_eye: rel(landmarks.right_eye),
        left_eye: rel(landmarks.left_eye),
        nose_tip: rel(landmarks.nose_tip),
        right_mouth: rel(landmarks.right_mouth_corner),
        left_mouth: rel(landmarks.left_mouth_corner),
    }
}

pub fn confidence_label(conf: MatchConfidence) -> &'static str {
    match conf {
        MatchConfidence::Verified => "verified",
        MatchConfidence::AmbiguousNeedsReview => "needs_review",
        MatchConfidence::Unmatched => "unmatched",
    }
}

impl From<&RgbHistogram> for HistogramDto {
    fn from(h: &RgbHistogram) -> Self {
        Self {
            red: h.red.to_vec(),
            green: h.green.to_vec(),
            blue: h.blue.to_vec(),
            luminance: h.luminance.to_vec(),
            total_pixels: h.total_pixels,
        }
    }
}

impl From<&HslBandShift> for HslChannelShiftDto {
    fn from(b: &HslBandShift) -> Self {
        Self {
            hue: b.hue,
            saturation: b.saturation,
            luminance: b.luminance,
        }
    }
}

impl From<HslChannelShiftDto> for HslBandShift {
    fn from(d: HslChannelShiftDto) -> Self {
        Self {
            hue: d.hue,
            saturation: d.saturation,
            luminance: d.luminance,
        }
    }
}

impl From<&ColorGradingWheel> for ColorGradingWheelDto {
    fn from(w: &ColorGradingWheel) -> Self {
        Self {
            hue_deg: w.hue_deg,
            saturation: w.saturation,
            luminance: w.luminance,
        }
    }
}

impl From<ColorGradingWheelDto> for ColorGradingWheel {
    fn from(d: ColorGradingWheelDto) -> Self {
        Self {
            hue_deg: d.hue_deg,
            saturation: d.saturation,
            luminance: d.luminance,
        }
    }
}

fn map_points_to_dto(pts: &[ToneCurvePoint]) -> Vec<ToneCurvePointDto> {
    pts.iter()
        .map(|p| ToneCurvePointDto { x: p.x, y: p.y })
        .collect()
}

fn map_points_from_dto(pts: Vec<ToneCurvePointDto>) -> Vec<ToneCurvePoint> {
    pts.into_iter()
        .map(|p| ToneCurvePoint { x: p.x, y: p.y })
        .collect()
}

impl From<&MaskComponent> for MaskComponentDto {
    fn from(comp: &MaskComponent) -> Self {
        match comp {
            MaskComponent::LinearGradient {
                id,
                op,
                inverted,
                start_x,
                start_y,
                end_x,
                end_y,
                feather,
            } => Self::LinearGradient {
                id: id.clone(),
                op: op.as_str().to_string(),
                inverted: *inverted,
                start_x: *start_x,
                start_y: *start_y,
                end_x: *end_x,
                end_y: *end_y,
                feather: *feather,
            },
            MaskComponent::RadialGradient {
                id,
                op,
                inverted,
                center_x,
                center_y,
                radius_x,
                radius_y,
                rotation_deg,
                feather,
            } => Self::RadialGradient {
                id: id.clone(),
                op: op.as_str().to_string(),
                inverted: *inverted,
                center_x: *center_x,
                center_y: *center_y,
                radius_x: *radius_x,
                radius_y: *radius_y,
                rotation_deg: *rotation_deg,
                feather: *feather,
            },
            MaskComponent::Brush {
                id,
                op,
                inverted,
                strokes,
            } => Self::Brush {
                id: id.clone(),
                op: op.as_str().to_string(),
                inverted: *inverted,
                strokes: strokes
                    .iter()
                    .map(|s| BrushStrokeDto {
                        size: s.size,
                        feather: s.feather,
                        flow: s.flow,
                        points: s.points.clone(),
                    })
                    .collect(),
            },
            MaskComponent::LuminanceRange {
                id,
                op,
                inverted,
                min_luma,
                max_luma,
                feather,
            } => Self::LuminanceRange {
                id: id.clone(),
                op: op.as_str().to_string(),
                inverted: *inverted,
                min_luma: *min_luma,
                max_luma: *max_luma,
                feather: *feather,
            },
            MaskComponent::ColorRange {
                id,
                op,
                inverted,
                target_hue_deg,
                target_sat,
                target_luma,
                tolerance,
                feather,
            } => Self::ColorRange {
                id: id.clone(),
                op: op.as_str().to_string(),
                inverted: *inverted,
                target_hue_deg: *target_hue_deg,
                target_sat: *target_sat,
                target_luma: *target_luma,
                tolerance: *tolerance,
                feather: *feather,
            },
            MaskComponent::AiMatte {
                id,
                op,
                inverted,
                kind,
                feather,
                matte_width,
                matte_height,
                matte_rle_b64,
            } => Self::AiMatte {
                id: id.clone(),
                op: op.as_str().to_string(),
                inverted: *inverted,
                kind: kind.as_str().to_string(),
                feather: *feather,
                matte_width: *matte_width,
                matte_height: *matte_height,
                matte_rle_b64: matte_rle_b64.clone(),
            },
        }
    }
}

impl From<MaskComponentDto> for MaskComponent {
    fn from(dto: MaskComponentDto) -> Self {
        match dto {
            MaskComponentDto::LinearGradient {
                id,
                op,
                inverted,
                start_x,
                start_y,
                end_x,
                end_y,
                feather,
            } => Self::LinearGradient {
                id,
                op: MaskCombineOp::parse(&op),
                inverted,
                start_x,
                start_y,
                end_x,
                end_y,
                feather,
            },
            MaskComponentDto::RadialGradient {
                id,
                op,
                inverted,
                center_x,
                center_y,
                radius_x,
                radius_y,
                rotation_deg,
                feather,
            } => Self::RadialGradient {
                id,
                op: MaskCombineOp::parse(&op),
                inverted,
                center_x,
                center_y,
                radius_x,
                radius_y,
                rotation_deg,
                feather,
            },
            MaskComponentDto::Brush {
                id,
                op,
                inverted,
                strokes,
            } => Self::Brush {
                id,
                op: MaskCombineOp::parse(&op),
                inverted,
                strokes: strokes
                    .into_iter()
                    .map(|s| BrushStroke {
                        size: s.size,
                        feather: s.feather,
                        flow: s.flow,
                        points: s.points,
                    })
                    .collect(),
            },
            MaskComponentDto::LuminanceRange {
                id,
                op,
                inverted,
                min_luma,
                max_luma,
                feather,
            } => Self::LuminanceRange {
                id,
                op: MaskCombineOp::parse(&op),
                inverted,
                min_luma,
                max_luma,
                feather,
            },
            MaskComponentDto::ColorRange {
                id,
                op,
                inverted,
                target_hue_deg,
                target_sat,
                target_luma,
                tolerance,
                feather,
            } => Self::ColorRange {
                id,
                op: MaskCombineOp::parse(&op),
                inverted,
                target_hue_deg,
                target_sat,
                target_luma,
                tolerance,
                feather,
            },
            MaskComponentDto::AiMatte {
                id,
                op,
                inverted,
                kind,
                feather,
                matte_width,
                matte_height,
                matte_rle_b64,
            } => Self::AiMatte {
                id,
                op: MaskCombineOp::parse(&op),
                inverted,
                kind: AiMaskKind::parse(&kind),
                feather,
                matte_width,
                matte_height,
                matte_rle_b64,
            },
        }
    }
}

impl From<&LocalAdjustments> for LocalAdjustmentsDto {
    fn from(a: &LocalAdjustments) -> Self {
        Self {
            temperature_shift: a.temperature_shift,
            tint_shift: a.tint_shift,
            exposure_ev: a.exposure_ev,
            contrast: a.contrast,
            highlights: a.highlights,
            shadows: a.shadows,
            whites: a.whites,
            blacks: a.blacks,
            texture: a.texture,
            clarity: a.clarity,
            dehaze: a.dehaze,
            saturation: a.saturation,
            sharpness: a.sharpness,
        }
    }
}

impl From<LocalAdjustmentsDto> for LocalAdjustments {
    fn from(d: LocalAdjustmentsDto) -> Self {
        Self {
            temperature_shift: d.temperature_shift,
            tint_shift: d.tint_shift,
            exposure_ev: d.exposure_ev,
            contrast: d.contrast,
            highlights: d.highlights,
            shadows: d.shadows,
            whites: d.whites,
            blacks: d.blacks,
            texture: d.texture,
            clarity: d.clarity,
            dehaze: d.dehaze,
            saturation: d.saturation,
            sharpness: d.sharpness,
        }
    }
}

impl From<&MaskLayer> for MaskLayerDto {
    fn from(m: &MaskLayer) -> Self {
        Self {
            id: m.id.clone(),
            name: m.name.clone(),
            enabled: m.enabled,
            inverted: m.inverted,
            opacity: m.opacity,
            components: m.components.iter().map(Into::into).collect(),
            adjustments: (&m.adjustments).into(),
        }
    }
}

impl From<MaskLayerDto> for MaskLayer {
    fn from(d: MaskLayerDto) -> Self {
        Self {
            id: d.id,
            name: d.name,
            enabled: d.enabled,
            inverted: d.inverted,
            opacity: d.opacity,
            components: d.components.into_iter().map(Into::into).collect(),
            adjustments: d.adjustments.into(),
        }
    }
}

impl From<&DevelopRecipe> for DevelopRecipeDto {
    fn from(r: &DevelopRecipe) -> Self {
        Self {
            white_balance: WhiteBalanceDto {
                temperature_kelvin: r.white_balance.temperature_kelvin,
                tint: r.white_balance.tint,
            },
            basic_tone: BasicToneDto {
                black_and_white: r.basic_tone.black_and_white,
                exposure_ev: r.basic_tone.exposure_ev,
                contrast: r.basic_tone.contrast,
                highlights: r.basic_tone.highlights,
                shadows: r.basic_tone.shadows,
                whites: r.basic_tone.whites,
                blacks: r.basic_tone.blacks,
                texture: r.basic_tone.texture,
                clarity: r.basic_tone.clarity,
                dehaze: r.basic_tone.dehaze,
                vibrance: r.basic_tone.vibrance,
                saturation: r.basic_tone.saturation,
            },
            tone_curve: ParametricToneCurveDto {
                highlights: r.tone_curve.highlights,
                lights: r.tone_curve.lights,
                darks: r.tone_curve.darks,
                shadows: r.tone_curve.shadows,
                points: map_points_to_dto(&r.tone_curve.points),
                red_points: map_points_to_dto(&r.tone_curve.red_points),
                green_points: map_points_to_dto(&r.tone_curve.green_points),
                blue_points: map_points_to_dto(&r.tone_curve.blue_points),
            },
            hsl_mixer: HslColorMixerDto {
                red: (&r.hsl_mixer.red).into(),
                orange: (&r.hsl_mixer.orange).into(),
                yellow: (&r.hsl_mixer.yellow).into(),
                green: (&r.hsl_mixer.green).into(),
                aqua: (&r.hsl_mixer.aqua).into(),
                blue: (&r.hsl_mixer.blue).into(),
                purple: (&r.hsl_mixer.purple).into(),
                magenta: (&r.hsl_mixer.magenta).into(),
            },
            color_grading: ColorGradingDto {
                shadows: (&r.color_grading.shadows).into(),
                midtones: (&r.color_grading.midtones).into(),
                highlights: (&r.color_grading.highlights).into(),
                blending: r.color_grading.blending,
                balance: r.color_grading.balance,
            },
            camera_calibration: CameraCalibrationDto {
                shadows_tint: r.camera_calibration.shadows_tint,
                red_hue: r.camera_calibration.red_hue,
                red_saturation: r.camera_calibration.red_saturation,
                green_hue: r.camera_calibration.green_hue,
                green_saturation: r.camera_calibration.green_saturation,
                blue_hue: r.camera_calibration.blue_hue,
                blue_saturation: r.camera_calibration.blue_saturation,
            },
            lut_3d: r.lut_3d.as_ref().map(|l| Lut3DDto {
                name: l.lut_name.clone(),
                cube_path: l.cube_path.clone(),
                intensity: l.intensity * 100.0,
            }),
            detail_optics: DetailOpticsDto {
                sharpening_amount: r.detail_optics.sharpening_amount,
                luminance_noise_reduction: r.detail_optics.luminance_noise_reduction,
                color_noise_reduction: r.detail_optics.color_noise_reduction,
                vignette_amount: r.detail_optics.vignette_amount,
                vignette_midpoint: r.detail_optics.vignette_midpoint,
                vignette_feather: r.detail_optics.vignette_feather,
                grain_amount: r.detail_optics.grain_amount,
                grain_size: r.detail_optics.grain_size,
            },
            crop_transform: CropTransformDto {
                rect: CropRectDto {
                    left: r.crop_transform.rect.left,
                    top: r.crop_transform.rect.top,
                    width: r.crop_transform.rect.width,
                    height: r.crop_transform.rect.height,
                },
                rotation_deg: r.crop_transform.rotation_deg,
                aspect_ratio: r.crop_transform.aspect_ratio.as_str().to_string(),
                flip_horizontal: r.crop_transform.flip_horizontal,
                flip_vertical: r.crop_transform.flip_vertical,
                quarter_turns: r.crop_transform.quarter_turns,
                eye_level_auto_locked: r.crop_transform.auto_eye_level_locked,
            },
            masks: r.masks.iter().map(Into::into).collect(),
            background_removal: (&r.background_removal).into(),
        }
    }
}

impl DevelopRecipeDto {
    pub fn into_domain(self) -> DevelopRecipe {
        let rect = NormalizedRect::new(
            self.crop_transform.rect.left,
            self.crop_transform.rect.top,
            self.crop_transform.rect.width,
            self.crop_transform.rect.height,
        )
        .unwrap_or_else(|_| NormalizedRect::full_frame());

        let mut recipe = DevelopRecipe {
            white_balance: WhiteBalance {
                temperature_kelvin: self.white_balance.temperature_kelvin,
                tint: self.white_balance.tint,
            },
            basic_tone: BasicTone {
                black_and_white: self.basic_tone.black_and_white,
                exposure_ev: self.basic_tone.exposure_ev,
                contrast: self.basic_tone.contrast,
                highlights: self.basic_tone.highlights,
                shadows: self.basic_tone.shadows,
                whites: self.basic_tone.whites,
                blacks: self.basic_tone.blacks,
                texture: self.basic_tone.texture,
                clarity: self.basic_tone.clarity,
                dehaze: self.basic_tone.dehaze,
                vibrance: self.basic_tone.vibrance,
                saturation: self.basic_tone.saturation,
            },
            tone_curve: ParametricToneCurve {
                highlights: self.tone_curve.highlights,
                lights: self.tone_curve.lights,
                darks: self.tone_curve.darks,
                shadows: self.tone_curve.shadows,
                points: map_points_from_dto(self.tone_curve.points),
                red_points: map_points_from_dto(self.tone_curve.red_points),
                green_points: map_points_from_dto(self.tone_curve.green_points),
                blue_points: map_points_from_dto(self.tone_curve.blue_points),
            },
            hsl_mixer: HslColorMixer {
                red: self.hsl_mixer.red.into(),
                orange: self.hsl_mixer.orange.into(),
                yellow: self.hsl_mixer.yellow.into(),
                green: self.hsl_mixer.green.into(),
                aqua: self.hsl_mixer.aqua.into(),
                blue: self.hsl_mixer.blue.into(),
                purple: self.hsl_mixer.purple.into(),
                magenta: self.hsl_mixer.magenta.into(),
            },
            color_grading: ColorGrading {
                shadows: self.color_grading.shadows.into(),
                midtones: self.color_grading.midtones.into(),
                highlights: self.color_grading.highlights.into(),
                blending: self.color_grading.blending,
                balance: self.color_grading.balance,
            },
            camera_calibration: CameraCalibration {
                shadows_tint: self.camera_calibration.shadows_tint,
                red_hue: self.camera_calibration.red_hue,
                red_saturation: self.camera_calibration.red_saturation,
                green_hue: self.camera_calibration.green_hue,
                green_saturation: self.camera_calibration.green_saturation,
                blue_hue: self.camera_calibration.blue_hue,
                blue_saturation: self.camera_calibration.blue_saturation,
            },
            lut_3d: self.lut_3d.and_then(|l| {
                Lut3DReference::new(l.name, l.cube_path, (l.intensity / 100.0).clamp(0.0, 1.0)).ok()
            }),
            detail_optics: DetailAndOptics {
                sharpening_amount: self.detail_optics.sharpening_amount,
                luminance_noise_reduction: self.detail_optics.luminance_noise_reduction,
                color_noise_reduction: self.detail_optics.color_noise_reduction,
                vignette_amount: self.detail_optics.vignette_amount,
                vignette_midpoint: self.detail_optics.vignette_midpoint,
                vignette_feather: self.detail_optics.vignette_feather,
                grain_amount: self.detail_optics.grain_amount,
                grain_size: self.detail_optics.grain_size,
            },
            crop_transform: CropTransform {
                rect,
                rotation_deg: self.crop_transform.rotation_deg,
                aspect_ratio: CropAspectRatio::parse_label(&self.crop_transform.aspect_ratio),
                flip_horizontal: self.crop_transform.flip_horizontal,
                flip_vertical: self.crop_transform.flip_vertical,
                quarter_turns: self.crop_transform.quarter_turns,
                auto_eye_level_locked: self.crop_transform.eye_level_auto_locked,
            },
            masks: self.masks.into_iter().map(Into::into).collect(),
            background_removal: self.background_removal.into(),
        };
        recipe.clamp_all_invariants();
        recipe
    }
}

impl From<DevelopSyncMaskDto> for DevelopSyncMask {
    fn from(m: DevelopSyncMaskDto) -> Self {
        Self {
            sync_white_balance: m.white_balance,
            sync_basic_tone: m.basic_tone,
            sync_tone_curve: m.tone_curve,
            sync_hsl_mixer: m.hsl_mixer,
            sync_color_grading: m.color_grading,
            sync_camera_calibration: m.camera_calibration,
            sync_lut_3d: m.lut_3d,
            sync_detail_optics: m.detail_optics,
            sync_crop_transform: m.crop_transform,
            sync_masks: m.masks,
        }
    }
}

impl From<RelativeDevelopDeltaDto> for RelativeDevelopDelta {
    fn from(d: RelativeDevelopDeltaDto) -> Self {
        Self {
            exposure_delta_ev: d.exposure_delta_ev.unwrap_or(0.0),
            temperature_delta_kelvin: d.temperature_delta_kelvin.unwrap_or(0),
            tint_delta: d.tint_delta.unwrap_or(0),
            contrast_delta: d.contrast_delta.unwrap_or(0),
            highlights_delta: d.highlights_delta.unwrap_or(0),
            shadows_delta: d.shadows_delta.unwrap_or(0),
            vibrance_delta: d.vibrance_delta.unwrap_or(0),
            saturation_delta: d.saturation_delta.unwrap_or(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_dto_serializes_lut3d_matching_typescript_contract() {
        let recipe = DevelopRecipe::default();
        let dto = DevelopRecipeDto::from(&recipe);
        let json = serde_json::to_string(&dto).expect("serialize DevelopRecipeDto");
        assert!(json.contains("\"lut3d\":null"), "actual json: {json}");
        assert!(json.contains("\"temperatureKelvin\":5500"));

        let round_trip: DevelopRecipeDto =
            serde_json::from_str(&json).expect("deserialize DevelopRecipeDto");
        let domain = round_trip.into_domain();
        assert_eq!(domain.white_balance.temperature_kelvin, 5500);
    }
}
