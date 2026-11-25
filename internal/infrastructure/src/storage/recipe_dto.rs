use kroma_domain::develop::{
    BasicTone, CameraCalibration, ColorGrading, ColorGradingWheel, CropAspectRatio, CropTransform,
    DetailAndOptics, DevelopRecipe, HslBandShift, HslColorMixer, Lut3DReference, NormalizedRect,
    ParametricToneCurve, ToneCurvePoint, WhiteBalance,
};
use kroma_domain::masking::{
    AiMaskKind, BackgroundRemoval, BackgroundReplaceMode, BrushStroke, LocalAdjustments,
    MaskCombineOp, MaskComponent, MaskLayer,
};
use kroma_domain::retouch::{HealSpot, HealSpotMode, PortraitRetouch, RetouchSettings};
use serde::{Deserialize, Serialize};

pub const CURRENT_SIDECAR_SCHEMA_VERSION: u32 = 3;

fn default_true() -> bool {
    true
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

fn default_fifty() -> u8 {
    50
}

fn default_twenty_five() -> u8 {
    25
}

fn default_aspect_original() -> String {
    "original".to_string()
}

/// Explicit outer-layer Data Transfer Object (PoEAA Data Mapper pattern).
/// Keeps `serde` serialization completely outside `kroma-domain`.
/// Every Phase 1/2 field carries `#[serde(default)]` so legacy v1/v2 `.kroma.json` files load cleanly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecipeSidecarDto {
    pub schema_version: u32,
    pub white_balance: WhiteBalanceDto,
    pub basic_tone: BasicToneDto,
    pub tone_curve: ParametricToneCurveDto,
    pub hsl_mixer: HslColorMixerDto,
    #[serde(default)]
    pub color_grading: ColorGradingDto,
    #[serde(default)]
    pub camera_calibration: CameraCalibrationDto,
    pub lut_3d: Option<Lut3DReferenceDto>,
    pub detail_optics: DetailAndOpticsDto,
    pub crop_transform: CropTransformDto,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub masks: Vec<MaskLayerDto>,
    #[serde(default)]
    pub background_removal: BackgroundRemovalDto,
    #[serde(default)]
    pub retouch: RetouchSettingsDto,
}

fn default_heal_mode() -> String {
    "heal".to_string()
}

fn default_heal_radius() -> f32 {
    0.018
}

fn default_heal_feather() -> f32 {
    0.55
}

fn default_one_f32() -> f32 {
    1.0
}

fn default_seventy_u8() -> u8 {
    70
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealSpotDto {
    pub id: String,
    #[serde(default = "default_heal_mode")]
    pub mode: String,
    pub target_x: f32,
    pub target_y: f32,
    pub source_x: f32,
    pub source_y: f32,
    #[serde(default = "default_heal_radius")]
    pub radius: f32,
    #[serde(default = "default_heal_feather")]
    pub feather: f32,
    #[serde(default = "default_one_f32")]
    pub opacity: f32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub points: Vec<[f32; 2]>,
}

impl From<&HealSpot> for HealSpotDto {
    fn from(s: &HealSpot) -> Self {
        Self {
            id: s.id.clone(),
            mode: s.mode.as_str().to_string(),
            target_x: s.target_x,
            target_y: s.target_y,
            source_x: s.source_x,
            source_y: s.source_y,
            radius: s.radius,
            feather: s.feather,
            opacity: s.opacity,
            points: s.points.clone(),
        }
    }
}

impl From<HealSpotDto> for HealSpot {
    fn from(d: HealSpotDto) -> Self {
        Self {
            id: d.id,
            mode: HealSpotMode::parse(&d.mode),
            target_x: d.target_x,
            target_y: d.target_y,
            source_x: d.source_x,
            source_y: d.source_y,
            radius: d.radius,
            feather: d.feather,
            opacity: d.opacity,
            points: d.points,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortraitRetouchDto {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub skin_smoothing: u8,
    #[serde(default = "default_seventy_u8")]
    pub skin_texture_preserve: u8,
    #[serde(default)]
    pub skin_tone_evenness: u8,
    #[serde(default)]
    pub shine_removal: u8,
    #[serde(default)]
    pub under_eye_brighten: u8,
    #[serde(default)]
    pub eye_enhance: u8,
    #[serde(default)]
    pub teeth_whiten: u8,
    #[serde(default)]
    pub lip_enhance: u8,
}

impl Default for PortraitRetouchDto {
    fn default() -> Self {
        Self {
            enabled: true,
            skin_smoothing: 0,
            skin_texture_preserve: 70,
            skin_tone_evenness: 0,
            shine_removal: 0,
            under_eye_brighten: 0,
            eye_enhance: 0,
            teeth_whiten: 0,
            lip_enhance: 0,
        }
    }
}

impl From<&PortraitRetouch> for PortraitRetouchDto {
    fn from(p: &PortraitRetouch) -> Self {
        Self {
            enabled: p.enabled,
            skin_smoothing: p.skin_smoothing,
            skin_texture_preserve: p.skin_texture_preserve,
            skin_tone_evenness: p.skin_tone_evenness,
            shine_removal: p.shine_removal,
            under_eye_brighten: p.under_eye_brighten,
            eye_enhance: p.eye_enhance,
            teeth_whiten: p.teeth_whiten,
            lip_enhance: p.lip_enhance,
        }
    }
}

impl From<PortraitRetouchDto> for PortraitRetouch {
    fn from(d: PortraitRetouchDto) -> Self {
        Self {
            enabled: d.enabled,
            skin_smoothing: d.skin_smoothing,
            skin_texture_preserve: d.skin_texture_preserve,
            skin_tone_evenness: d.skin_tone_evenness,
            shine_removal: d.shine_removal,
            under_eye_brighten: d.under_eye_brighten,
            eye_enhance: d.eye_enhance,
            teeth_whiten: d.teeth_whiten,
            lip_enhance: d.lip_enhance,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchBoundsDto {
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InpaintPatchDto {
    pub id: String,
    pub bounds: PatchBoundsDto,
    pub patch_width: u16,
    pub patch_height: u16,
    #[serde(default = "default_one_f32")]
    pub opacity: f32,
    #[serde(default)]
    pub mask_rle_b64: String,
    #[serde(default)]
    pub patch_image_b64: String,
    #[serde(default)]
    pub engine_label: String,
}

impl From<&kroma_domain::retouch::InpaintPatch> for InpaintPatchDto {
    fn from(p: &kroma_domain::retouch::InpaintPatch) -> Self {
        Self {
            id: p.id.clone(),
            bounds: PatchBoundsDto {
                left: p.bounds.left,
                top: p.bounds.top,
                width: p.bounds.width,
                height: p.bounds.height,
            },
            patch_width: p.patch_width,
            patch_height: p.patch_height,
            opacity: p.opacity,
            mask_rle_b64: p.mask_rle_b64.clone(),
            patch_image_b64: p.patch_image_b64.clone(),
            engine_label: p.engine_label.clone(),
        }
    }
}

impl From<InpaintPatchDto> for kroma_domain::retouch::InpaintPatch {
    fn from(d: InpaintPatchDto) -> Self {
        let mut out = Self {
            id: d.id,
            bounds: NormalizedRect {
                left: d.bounds.left,
                top: d.bounds.top,
                width: d.bounds.width,
                height: d.bounds.height,
            },
            patch_width: d.patch_width,
            patch_height: d.patch_height,
            opacity: d.opacity,
            mask_rle_b64: d.mask_rle_b64,
            patch_image_b64: d.patch_image_b64,
            engine_label: d.engine_label,
        };
        out.clamp_invariants();
        out
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RetouchSettingsDto {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub heal_spots: Vec<HealSpotDto>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inpaint_patches: Vec<InpaintPatchDto>,
    #[serde(default)]
    pub portrait: PortraitRetouchDto,
}

impl From<&RetouchSettings> for RetouchSettingsDto {
    fn from(r: &RetouchSettings) -> Self {
        Self {
            heal_spots: r.heal_spots.iter().map(Into::into).collect(),
            inpaint_patches: r.inpaint_patches.iter().map(Into::into).collect(),
            portrait: (&r.portrait).into(),
        }
    }
}

impl From<RetouchSettingsDto> for RetouchSettings {
    fn from(d: RetouchSettingsDto) -> Self {
        Self {
            heal_spots: d.heal_spots.into_iter().map(Into::into).collect(),
            inpaint_patches: d.inpaint_patches.into_iter().map(Into::into).collect(),
            portrait: d.portrait.into(),
        }
    }
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
pub struct WhiteBalanceDto {
    pub temperature_kelvin: u32,
    pub tint: i16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
pub struct ToneCurvePointDto {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
pub struct HslBandShiftDto {
    pub hue: i16,
    pub saturation: i16,
    pub luminance: i16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HslColorMixerDto {
    pub red: HslBandShiftDto,
    pub orange: HslBandShiftDto,
    pub yellow: HslBandShiftDto,
    pub green: HslBandShiftDto,
    pub aqua: HslBandShiftDto,
    pub blue: HslBandShiftDto,
    pub purple: HslBandShiftDto,
    pub magenta: HslBandShiftDto,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ColorGradingWheelDto {
    #[serde(default)]
    pub hue_deg: f32,
    #[serde(default)]
    pub saturation: u8,
    #[serde(default)]
    pub luminance: i16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorGradingDto {
    #[serde(default)]
    pub shadows: ColorGradingWheelDto,
    #[serde(default)]
    pub midtones: ColorGradingWheelDto,
    #[serde(default)]
    pub highlights: ColorGradingWheelDto,
    #[serde(default = "default_fifty")]
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
pub struct Lut3DReferenceDto {
    pub lut_name: String,
    pub cube_path: String,
    pub intensity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailAndOpticsDto {
    pub sharpening_amount: u8,
    pub luminance_noise_reduction: u8,
    pub color_noise_reduction: u8,
    pub vignette_amount: i16,
    #[serde(default = "default_fifty")]
    pub vignette_midpoint: u8,
    #[serde(default = "default_fifty")]
    pub vignette_feather: u8,
    #[serde(default)]
    pub grain_amount: u8,
    #[serde(default = "default_twenty_five")]
    pub grain_size: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CropTransformDto {
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
    pub rotation_deg: f32,
    #[serde(default = "default_aspect_original")]
    pub aspect_ratio: String,
    #[serde(default)]
    pub flip_horizontal: bool,
    #[serde(default)]
    pub flip_vertical: bool,
    #[serde(default)]
    pub quarter_turns: u8,
    pub auto_eye_level_locked: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    Brush {
        id: String,
        #[serde(default = "default_op_add")]
        op: String,
        #[serde(default)]
        inverted: bool,
        #[serde(default)]
        strokes: Vec<BrushStrokeDto>,
    },
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

fn points_to_dto(pts: &[ToneCurvePoint]) -> Vec<ToneCurvePointDto> {
    pts.iter()
        .map(|p| ToneCurvePointDto { x: p.x, y: p.y })
        .collect()
}

fn points_from_dto(pts: Vec<ToneCurvePointDto>) -> Vec<ToneCurvePoint> {
    pts.into_iter()
        .map(|p| ToneCurvePoint { x: p.x, y: p.y })
        .collect()
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

impl From<&DevelopRecipe> for RecipeSidecarDto {
    fn from(recipe: &DevelopRecipe) -> Self {
        Self {
            schema_version: CURRENT_SIDECAR_SCHEMA_VERSION,
            white_balance: WhiteBalanceDto {
                temperature_kelvin: recipe.white_balance.temperature_kelvin,
                tint: recipe.white_balance.tint,
            },
            basic_tone: BasicToneDto {
                black_and_white: recipe.basic_tone.black_and_white,
                exposure_ev: recipe.basic_tone.exposure_ev,
                contrast: recipe.basic_tone.contrast,
                highlights: recipe.basic_tone.highlights,
                shadows: recipe.basic_tone.shadows,
                whites: recipe.basic_tone.whites,
                blacks: recipe.basic_tone.blacks,
                texture: recipe.basic_tone.texture,
                clarity: recipe.basic_tone.clarity,
                dehaze: recipe.basic_tone.dehaze,
                vibrance: recipe.basic_tone.vibrance,
                saturation: recipe.basic_tone.saturation,
            },
            tone_curve: ParametricToneCurveDto {
                highlights: recipe.tone_curve.highlights,
                lights: recipe.tone_curve.lights,
                darks: recipe.tone_curve.darks,
                shadows: recipe.tone_curve.shadows,
                points: points_to_dto(&recipe.tone_curve.points),
                red_points: points_to_dto(&recipe.tone_curve.red_points),
                green_points: points_to_dto(&recipe.tone_curve.green_points),
                blue_points: points_to_dto(&recipe.tone_curve.blue_points),
            },
            hsl_mixer: HslColorMixerDto {
                red: (&recipe.hsl_mixer.red).into(),
                orange: (&recipe.hsl_mixer.orange).into(),
                yellow: (&recipe.hsl_mixer.yellow).into(),
                green: (&recipe.hsl_mixer.green).into(),
                aqua: (&recipe.hsl_mixer.aqua).into(),
                blue: (&recipe.hsl_mixer.blue).into(),
                purple: (&recipe.hsl_mixer.purple).into(),
                magenta: (&recipe.hsl_mixer.magenta).into(),
            },
            color_grading: ColorGradingDto {
                shadows: (&recipe.color_grading.shadows).into(),
                midtones: (&recipe.color_grading.midtones).into(),
                highlights: (&recipe.color_grading.highlights).into(),
                blending: recipe.color_grading.blending,
                balance: recipe.color_grading.balance,
            },
            camera_calibration: CameraCalibrationDto {
                shadows_tint: recipe.camera_calibration.shadows_tint,
                red_hue: recipe.camera_calibration.red_hue,
                red_saturation: recipe.camera_calibration.red_saturation,
                green_hue: recipe.camera_calibration.green_hue,
                green_saturation: recipe.camera_calibration.green_saturation,
                blue_hue: recipe.camera_calibration.blue_hue,
                blue_saturation: recipe.camera_calibration.blue_saturation,
            },
            lut_3d: recipe.lut_3d.as_ref().map(|lut| Lut3DReferenceDto {
                lut_name: lut.lut_name.clone(),
                cube_path: lut.cube_path.clone(),
                intensity: lut.intensity,
            }),
            detail_optics: DetailAndOpticsDto {
                sharpening_amount: recipe.detail_optics.sharpening_amount,
                luminance_noise_reduction: recipe.detail_optics.luminance_noise_reduction,
                color_noise_reduction: recipe.detail_optics.color_noise_reduction,
                vignette_amount: recipe.detail_optics.vignette_amount,
                vignette_midpoint: recipe.detail_optics.vignette_midpoint,
                vignette_feather: recipe.detail_optics.vignette_feather,
                grain_amount: recipe.detail_optics.grain_amount,
                grain_size: recipe.detail_optics.grain_size,
            },
            crop_transform: CropTransformDto {
                left: recipe.crop_transform.rect.left,
                top: recipe.crop_transform.rect.top,
                width: recipe.crop_transform.rect.width,
                height: recipe.crop_transform.rect.height,
                rotation_deg: recipe.crop_transform.rotation_deg,
                aspect_ratio: recipe.crop_transform.aspect_ratio.as_str().to_string(),
                flip_horizontal: recipe.crop_transform.flip_horizontal,
                flip_vertical: recipe.crop_transform.flip_vertical,
                quarter_turns: recipe.crop_transform.quarter_turns,
                auto_eye_level_locked: recipe.crop_transform.auto_eye_level_locked,
            },
            masks: recipe.masks.iter().map(Into::into).collect(),
            background_removal: (&recipe.background_removal).into(),
            retouch: (&recipe.retouch).into(),
        }
    }
}

impl From<&HslBandShift> for HslBandShiftDto {
    fn from(band: &HslBandShift) -> Self {
        Self {
            hue: band.hue,
            saturation: band.saturation,
            luminance: band.luminance,
        }
    }
}

impl From<HslBandShiftDto> for HslBandShift {
    fn from(dto: HslBandShiftDto) -> Self {
        Self {
            hue: dto.hue,
            saturation: dto.saturation,
            luminance: dto.luminance,
        }
    }
}

impl RecipeSidecarDto {
    pub fn into_domain(self) -> DevelopRecipe {
        let rect = NormalizedRect::new(
            self.crop_transform.left,
            self.crop_transform.top,
            self.crop_transform.width,
            self.crop_transform.height,
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
                points: points_from_dto(self.tone_curve.points),
                red_points: points_from_dto(self.tone_curve.red_points),
                green_points: points_from_dto(self.tone_curve.green_points),
                blue_points: points_from_dto(self.tone_curve.blue_points),
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
            lut_3d: self.lut_3d.and_then(|lut| {
                Lut3DReference::new(lut.lut_name, lut.cube_path, lut.intensity).ok()
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
                auto_eye_level_locked: self.crop_transform.auto_eye_level_locked,
            },
            masks: self.masks.into_iter().map(Into::into).collect(),
            background_removal: self.background_removal.into(),
            retouch: self.retouch.into(),
        };
        recipe.clamp_all_invariants();
        recipe
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kroma_domain::masking::encode_matte_rle_b64;

    #[test]
    fn roundtrips_v3_sidecar_with_composable_manual_and_ai_masks() {
        let mut recipe = DevelopRecipe::default();
        let matte = vec![255u8; 16];
        recipe.masks.push(MaskLayer {
            id: "mask-1".to_string(),
            name: "Subject Pop".to_string(),
            enabled: true,
            inverted: false,
            opacity: 85,
            components: vec![
                MaskComponent::AiMatte {
                    id: "c-ai".to_string(),
                    op: MaskCombineOp::Add,
                    inverted: false,
                    kind: AiMaskKind::Subject,
                    feather: 30.0,
                    matte_width: 4,
                    matte_height: 4,
                    matte_rle_b64: encode_matte_rle_b64(&matte),
                },
                MaskComponent::LinearGradient {
                    id: "c-lin".to_string(),
                    op: MaskCombineOp::Intersect,
                    inverted: false,
                    start_x: 0.5,
                    start_y: 0.0,
                    end_x: 0.5,
                    end_y: 1.0,
                    feather: 60.0,
                },
                MaskComponent::LuminanceRange {
                    id: "c-lum".to_string(),
                    op: MaskCombineOp::Subtract,
                    inverted: false,
                    min_luma: 0.0,
                    max_luma: 0.15,
                    feather: 0.08,
                },
            ],
            adjustments: LocalAdjustments {
                exposure_ev: 0.45,
                clarity: 18,
                sharpness: -25,
                ..LocalAdjustments::default()
            },
        });

        let dto = RecipeSidecarDto::from(&recipe);
        let json = serde_json::to_string(&dto).expect("serialize v3 sidecar");
        let parsed: RecipeSidecarDto = serde_json::from_str(&json).expect("deserialize v3 sidecar");
        let restored = parsed.into_domain();
        assert_eq!(restored.masks.len(), 1);
        assert_eq!(restored.masks[0], recipe.masks[0]);
    }

    #[test]
    fn deserializes_legacy_v1_sidecar_json_with_safe_defaults() {
        let legacy_json = r#"{
            "schema_version": 1,
            "white_balance": { "temperature_kelvin": 4900, "tint": 4 },
            "basic_tone": {
                "exposure_ev": 0.5, "contrast": 12, "highlights": -20, "shadows": 15,
                "whites": 0, "blacks": 0, "texture": 5, "clarity": 8, "dehaze": 0,
                "vibrance": 10, "saturation": 0
            },
            "tone_curve": { "highlights": 0, "lights": 0, "darks": 0, "shadows": 0 },
            "hsl_mixer": {
                "red": { "hue": 0, "saturation": 0, "luminance": 0 },
                "orange": { "hue": 0, "saturation": 0, "luminance": 0 },
                "yellow": { "hue": 0, "saturation": 0, "luminance": 0 },
                "green": { "hue": 0, "saturation": 0, "luminance": 0 },
                "aqua": { "hue": 0, "saturation": 0, "luminance": 0 },
                "blue": { "hue": 0, "saturation": 0, "luminance": 0 },
                "purple": { "hue": 0, "saturation": 0, "luminance": 0 },
                "magenta": { "hue": 0, "saturation": 0, "luminance": 0 }
            },
            "lut_3d": null,
            "detail_optics": {
                "sharpening_amount": 40, "luminance_noise_reduction": 10,
                "color_noise_reduction": 0, "vignette_amount": -15, "vignette_midpoint": 50
            },
            "crop_transform": {
                "left": 0.0, "top": 0.0, "width": 1.0, "height": 1.0,
                "rotation_deg": 0.0, "auto_eye_level_locked": false
            }
        }"#;

        let dto: RecipeSidecarDto =
            serde_json::from_str(legacy_json).expect("legacy v1 sidecar must deserialize");
        let domain = dto.into_domain();
        assert_eq!(domain.white_balance.temperature_kelvin, 4900);
        assert!(!domain.basic_tone.black_and_white);
        assert!(domain.color_grading.is_neutral());
        assert_eq!(domain.color_grading.blending, 50);
        assert!(domain.camera_calibration.is_neutral());
        assert_eq!(domain.detail_optics.vignette_feather, 50);
        assert_eq!(domain.detail_optics.grain_amount, 0);
        assert_eq!(domain.detail_optics.grain_size, 25);
        assert!(domain.crop_transform.is_identity());
        assert!(!domain.retouch.is_active());
        assert_eq!(domain.retouch.portrait.skin_texture_preserve, 70);
    }

    #[test]
    fn roundtrips_sidecar_with_heal_spots_and_portrait_retouch() {
        let mut recipe = DevelopRecipe::default();
        recipe.retouch.heal_spots.push(HealSpot {
            id: "h1".to_string(),
            mode: HealSpotMode::Heal,
            target_x: 0.45,
            target_y: 0.35,
            source_x: 0.52,
            source_y: 0.36,
            radius: 0.02,
            feather: 0.6,
            opacity: 0.9,
            points: vec![[0.45, 0.35], [0.47, 0.36]],
        });
        recipe.retouch.portrait = PortraitRetouch {
            enabled: true,
            skin_smoothing: 48,
            skin_texture_preserve: 75,
            skin_tone_evenness: 35,
            shine_removal: 30,
            under_eye_brighten: 28,
            eye_enhance: 24,
            teeth_whiten: 32,
            lip_enhance: 16,
        };

        let dto = RecipeSidecarDto::from(&recipe);
        let json = serde_json::to_string(&dto).expect("serialize retouch sidecar");
        let parsed: RecipeSidecarDto =
            serde_json::from_str(&json).expect("deserialize retouch sidecar");
        let restored = parsed.into_domain();

        assert_eq!(restored.retouch, recipe.retouch);
    }
}
