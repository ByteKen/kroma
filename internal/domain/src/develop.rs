use crate::error::DomainError;

pub const MIN_KELVIN: u32 = 2_000;
pub const MAX_KELVIN: u32 = 50_000;
pub const NEUTRAL_KELVIN: u32 = 5_500;

/// White Balance (Temperature in Kelvin + Green/Magenta Tint).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WhiteBalance {
    pub temperature_kelvin: u32,
    pub tint: i16,
}

impl WhiteBalance {
    pub fn new(temperature_kelvin: u32, tint: i16) -> Result<Self, DomainError> {
        if !(MIN_KELVIN..=MAX_KELVIN).contains(&temperature_kelvin) {
            return Err(DomainError::InvalidColorTemperature(temperature_kelvin));
        }
        Ok(Self {
            temperature_kelvin,
            tint: tint.clamp(-150, 150),
        })
    }

    /// Computes neutral white balance correction from a sampled RGB gray/whiteboard average
    /// (such as a subject holding a physical white slate placard).
    pub fn calibrated_from_neutral_rgb(r: f32, g: f32, b: f32) -> Self {
        let safe_r = r.max(1e-3);
        let safe_g = g.max(1e-3);
        let safe_b = b.max(1e-3);

        // If the sampled white looks bluish (b > r), warm up Kelvin; if yellowish (r > b), cool down.
        let rb_ratio = (safe_b / safe_r).clamp(0.5, 2.0);
        let kelvin = ((NEUTRAL_KELVIN as f32) * rb_ratio)
            .round()
            .clamp(MIN_KELVIN as f32, MAX_KELVIN as f32) as u32;

        // If green dominates average of R & B, shift tint toward magenta (+); otherwise green (-).
        let rb_avg = 0.5 * (safe_r + safe_b);
        let green_bias = ((safe_g / rb_avg) - 1.0) * 180.0;
        let tint = green_bias.round().clamp(-150.0, 150.0) as i16;

        Self {
            temperature_kelvin: kelvin,
            tint,
        }
    }
}

impl Default for WhiteBalance {
    fn default() -> Self {
        Self {
            temperature_kelvin: NEUTRAL_KELVIN,
            tint: 0,
        }
    }
}

/// Lightroom-style Basic Tone & Presence controls.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BasicTone {
    /// Black & White treatment mode (uses HSL Luminance sliders as B&W Color Mix)
    pub black_and_white: bool,
    /// Exposure compensation in EV (-5.0 to +5.0)
    pub exposure_ev: f32,
    /// Contrast (-100 to +100)
    pub contrast: i16,
    /// Highlights recovery (-100 to +100)
    pub highlights: i16,
    /// Shadows lift (-100 to +100)
    pub shadows: i16,
    /// Whites clipping point (-100 to +100)
    pub whites: i16,
    /// Blacks clipping point (-100 to +100)
    pub blacks: i16,
    /// Midtone micro-contrast texture (-100 to +100)
    pub texture: i16,
    /// Local contrast clarity (-100 to +100)
    pub clarity: i16,
    /// Atmospheric dehaze (-100 to +100)
    pub dehaze: i16,
    /// Skin-tone-protected vibrance (-100 to +100)
    pub vibrance: i16,
    /// Global saturation (-100 to +100)
    pub saturation: i16,
}

impl BasicTone {
    pub fn clamp_invariants(&mut self) {
        self.exposure_ev = self.exposure_ev.clamp(-5.0, 5.0);
        self.contrast = self.contrast.clamp(-100, 100);
        self.highlights = self.highlights.clamp(-100, 100);
        self.shadows = self.shadows.clamp(-100, 100);
        self.whites = self.whites.clamp(-100, 100);
        self.blacks = self.blacks.clamp(-100, 100);
        self.texture = self.texture.clamp(-100, 100);
        self.clarity = self.clarity.clamp(-100, 100);
        self.dehaze = self.dehaze.clamp(-100, 100);
        self.vibrance = self.vibrance.clamp(-100, 100);
        self.saturation = self.saturation.clamp(-100, 100);
    }
}

/// Control point on a Lightroom/Photoshop-style tone curve, normalized in `[0.0, 1.0]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToneCurvePoint {
    pub x: f32,
    pub y: f32,
}

/// Combined Parametric + Point Tone Curve (`-100..=100` regions + optional master & R/G/B points).
/// Empty point vectors represent the identity linear diagonal `[(0, 0), (1, 1)]`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ParametricToneCurve {
    pub highlights: i16,
    pub lights: i16,
    pub darks: i16,
    pub shadows: i16,
    pub points: Vec<ToneCurvePoint>,
    pub red_points: Vec<ToneCurvePoint>,
    pub green_points: Vec<ToneCurvePoint>,
    pub blue_points: Vec<ToneCurvePoint>,
}

fn clamp_curve_points(points: &mut Vec<ToneCurvePoint>) {
    if points.is_empty() {
        return;
    }
    for pt in points.iter_mut() {
        pt.x = if pt.x.is_finite() { pt.x.clamp(0.0, 1.0) } else { 0.0 };
        pt.y = if pt.y.is_finite() { pt.y.clamp(0.0, 1.0) } else { 0.0 };
    }
    points.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
    if points.len() > 16 {
        points.truncate(16);
    }
}

fn has_custom_point_slice(points: &[ToneCurvePoint]) -> bool {
    if points.len() < 2 {
        return false;
    }
    if points.len() == 2 {
        let p0 = points[0];
        let p1 = points[1];
        return p0.x.abs() > 1e-4
            || p0.y.abs() > 1e-4
            || (p1.x - 1.0).abs() > 1e-4
            || (p1.y - 1.0).abs() > 1e-4;
    }
    true
}

impl ParametricToneCurve {
    pub fn clamp_invariants(&mut self) {
        self.highlights = self.highlights.clamp(-100, 100);
        self.lights = self.lights.clamp(-100, 100);
        self.darks = self.darks.clamp(-100, 100);
        self.shadows = self.shadows.clamp(-100, 100);
        clamp_curve_points(&mut self.points);
        clamp_curve_points(&mut self.red_points);
        clamp_curve_points(&mut self.green_points);
        clamp_curve_points(&mut self.blue_points);
    }

    pub fn has_custom_points(&self) -> bool {
        has_custom_point_slice(&self.points)
    }

    pub fn has_custom_channel_points(&self) -> bool {
        has_custom_point_slice(&self.red_points)
            || has_custom_point_slice(&self.green_points)
            || has_custom_point_slice(&self.blue_points)
    }

    pub fn is_neutral(&self) -> bool {
        self.highlights == 0
            && self.lights == 0
            && self.darks == 0
            && self.shadows == 0
            && !self.has_custom_points()
            && !self.has_custom_channel_points()
    }

    /// Evaluates the master point tone curve at normalized luminance `x` in `[0.0, 1.0]`
    /// using Fritsch-Carlson monotone cubic Hermite interpolation (no ringing/overshoot).
    pub fn evaluate_point_curve(&self, x: f32) -> f32 {
        Self::evaluate_points_slice(&self.points, x)
    }

    /// Evaluates an arbitrary point-curve slice (Master, Red, Green, or Blue) at `x` in `[0.0, 1.0]`.
    pub fn evaluate_points_slice(pts: &[ToneCurvePoint], x: f32) -> f32 {
        let x = x.clamp(0.0, 1.0);
        if !has_custom_point_slice(pts) {
            return x;
        }
        let n = pts.len();
        if x <= pts[0].x {
            return pts[0].y.clamp(0.0, 1.0);
        }
        if x >= pts[n - 1].x {
            return pts[n - 1].y.clamp(0.0, 1.0);
        }

        // Secant slopes between consecutive control points
        let mut h = Vec::with_capacity(n - 1);
        let mut delta = Vec::with_capacity(n - 1);
        for i in 0..(n - 1) {
            let dx = (pts[i + 1].x - pts[i].x).max(1e-5);
            h.push(dx);
            delta.push((pts[i + 1].y - pts[i].y) / dx);
        }

        // Initial endpoint & interior tangents
        let mut m = vec![0.0_f32; n];
        m[0] = delta[0];
        m[n - 1] = delta[n - 2];
        for i in 1..(n - 1) {
            if delta[i - 1] * delta[i] <= 0.0 {
                m[i] = 0.0;
            } else {
                m[i] = 0.5 * (delta[i - 1] + delta[i]);
            }
        }

        // Fritsch-Carlson monotonicity constraint (prevents cubic overshoot)
        for i in 0..(n - 1) {
            if delta[i].abs() < 1e-6 {
                m[i] = 0.0;
                m[i + 1] = 0.0;
            } else {
                let alpha = m[i] / delta[i];
                let beta = m[i + 1] / delta[i];
                let sq = alpha * alpha + beta * beta;
                if sq > 9.0 {
                    let tau = 3.0 / sq.sqrt();
                    m[i] = tau * alpha * delta[i];
                    m[i + 1] = tau * beta * delta[i];
                }
            }
        }

        // Locate segment containing x
        let mut seg = n - 2;
        for i in 0..(n - 1) {
            if x <= pts[i + 1].x {
                seg = i;
                break;
            }
        }

        let t = ((x - pts[seg].x) / h[seg]).clamp(0.0, 1.0);
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        let y = h00 * pts[seg].y
            + h10 * h[seg] * m[seg]
            + h01 * pts[seg + 1].y
            + h11 * h[seg] * m[seg + 1];
        y.clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HslBandShift {
    pub hue: i16,
    pub saturation: i16,
    pub luminance: i16,
}

impl HslBandShift {
    pub fn clamp_invariants(&mut self) {
        self.hue = self.hue.clamp(-100, 100);
        self.saturation = self.saturation.clamp(-100, 100);
        self.luminance = self.luminance.clamp(-100, 100);
    }

    pub fn is_neutral(&self) -> bool {
        self.hue == 0 && self.saturation == 0 && self.luminance == 0
    }
}

/// 8-Channel Lightroom HSL Color Mixer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HslColorMixer {
    pub red: HslBandShift,
    pub orange: HslBandShift,
    pub yellow: HslBandShift,
    pub green: HslBandShift,
    pub aqua: HslBandShift,
    pub blue: HslBandShift,
    pub purple: HslBandShift,
    pub magenta: HslBandShift,
}

impl HslColorMixer {
    pub fn clamp_invariants(&mut self) {
        self.red.clamp_invariants();
        self.orange.clamp_invariants();
        self.yellow.clamp_invariants();
        self.green.clamp_invariants();
        self.aqua.clamp_invariants();
        self.blue.clamp_invariants();
        self.purple.clamp_invariants();
        self.magenta.clamp_invariants();
    }

    pub fn is_neutral(&self) -> bool {
        self.red.is_neutral()
            && self.orange.is_neutral()
            && self.yellow.is_neutral()
            && self.green.is_neutral()
            && self.aqua.is_neutral()
            && self.blue.is_neutral()
            && self.purple.is_neutral()
            && self.magenta.is_neutral()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Lut3DReference {
    pub lut_name: String,
    pub cube_path: String,
    /// Blend intensity in [0.0, 1.0]
    pub intensity: f32,
}

impl Lut3DReference {
    pub fn new(
        lut_name: impl Into<String>,
        cube_path: impl Into<String>,
        intensity: f32,
    ) -> Result<Self, DomainError> {
        let lut_name = lut_name.into();
        if lut_name.trim().is_empty() {
            return Err(DomainError::EmptyIdentifier("lut_name"));
        }
        let cube_path = cube_path.into();
        if cube_path.trim().is_empty() {
            return Err(DomainError::EmptyIdentifier("cube_path"));
        }
        if !(0.0..=1.0).contains(&intensity) {
            return Err(DomainError::InvalidLutIntensity(intensity));
        }
        Ok(Self {
            lut_name,
            cube_path,
            intensity,
        })
    }
}

/// Single Lightroom 3-Way Color Grading wheel (`hue_deg` 0..=360, `saturation` 0..=100, `luminance` -100..=100).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ColorGradingWheel {
    pub hue_deg: f32,
    pub saturation: u8,
    pub luminance: i16,
}

impl ColorGradingWheel {
    pub fn clamp_invariants(&mut self) {
        self.hue_deg = if self.hue_deg.is_finite() {
            self.hue_deg.rem_euclid(360.0)
        } else {
            0.0
        };
        self.saturation = self.saturation.min(100);
        self.luminance = self.luminance.clamp(-100, 100);
    }

    pub fn is_neutral(&self) -> bool {
        self.saturation == 0 && self.luminance == 0
    }
}

/// Lightroom 3-Way Color Grading (`Shadows`, `Midtones`, `Highlights` + `Blending` & `Balance`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorGrading {
    pub shadows: ColorGradingWheel,
    pub midtones: ColorGradingWheel,
    pub highlights: ColorGradingWheel,
    /// Tonal transition overlap (`0..=100`, default `50`)
    pub blending: u8,
    /// Shadow/Highlight split bias (`-100..=100`, default `0`)
    pub balance: i16,
}

impl Default for ColorGrading {
    fn default() -> Self {
        Self {
            shadows: ColorGradingWheel::default(),
            midtones: ColorGradingWheel::default(),
            highlights: ColorGradingWheel::default(),
            blending: 50,
            balance: 0,
        }
    }
}

impl ColorGrading {
    pub fn clamp_invariants(&mut self) {
        self.shadows.clamp_invariants();
        self.midtones.clamp_invariants();
        self.highlights.clamp_invariants();
        self.blending = self.blending.min(100);
        self.balance = self.balance.clamp(-100, 100);
    }

    pub fn is_neutral(&self) -> bool {
        self.shadows.is_neutral() && self.midtones.is_neutral() && self.highlights.is_neutral()
    }
}

/// Lightroom Camera Calibration (`Shadows Tint` + `Red / Green / Blue` primary hue & saturation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CameraCalibration {
    pub shadows_tint: i16,
    pub red_hue: i16,
    pub red_saturation: i16,
    pub green_hue: i16,
    pub green_saturation: i16,
    pub blue_hue: i16,
    pub blue_saturation: i16,
}

impl CameraCalibration {
    pub fn clamp_invariants(&mut self) {
        self.shadows_tint = self.shadows_tint.clamp(-100, 100);
        self.red_hue = self.red_hue.clamp(-100, 100);
        self.red_saturation = self.red_saturation.clamp(-100, 100);
        self.green_hue = self.green_hue.clamp(-100, 100);
        self.green_saturation = self.green_saturation.clamp(-100, 100);
        self.blue_hue = self.blue_hue.clamp(-100, 100);
        self.blue_saturation = self.blue_saturation.clamp(-100, 100);
    }

    pub fn is_neutral(&self) -> bool {
        self.shadows_tint == 0
            && self.red_hue == 0
            && self.red_saturation == 0
            && self.green_hue == 0
            && self.green_saturation == 0
            && self.blue_hue == 0
            && self.blue_saturation == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DetailAndOptics {
    /// Unsharp mask strength (0..=150)
    pub sharpening_amount: u8,
    /// Luminance noise reduction (0..=100)
    pub luminance_noise_reduction: u8,
    /// Color noise reduction (0..=100)
    pub color_noise_reduction: u8,
    /// Post-crop vignette (-100 darkens edges, +100 brightens edges)
    pub vignette_amount: i16,
    /// Vignette midpoint (0..=100, default 50)
    pub vignette_midpoint: u8,
    /// Vignette feather softness (0..=100, default 50)
    pub vignette_feather: u8,
    /// Film grain amount (0..=100, default 0)
    pub grain_amount: u8,
    /// Film grain particle size (0..=100, default 25)
    pub grain_size: u8,
}

impl DetailAndOptics {
    pub fn clamp_invariants(&mut self) {
        self.sharpening_amount = self.sharpening_amount.min(150);
        self.luminance_noise_reduction = self.luminance_noise_reduction.min(100);
        self.color_noise_reduction = self.color_noise_reduction.min(100);
        self.vignette_amount = self.vignette_amount.clamp(-100, 100);
        self.vignette_midpoint = self.vignette_midpoint.min(100);
        self.vignette_feather = self.vignette_feather.min(100);
        self.grain_amount = self.grain_amount.min(100);
        self.grain_size = self.grain_size.min(100);
    }
}

impl Default for DetailAndOptics {
    fn default() -> Self {
        Self {
            sharpening_amount: 0,
            luminance_noise_reduction: 0,
            color_noise_reduction: 0,
            vignette_amount: 0,
            vignette_midpoint: 50,
            vignette_feather: 50,
            grain_amount: 0,
            grain_size: 25,
        }
    }
}

/// Normalized [0.0, 1.0] rectangle inside an image (used for crop or whiteboard WB sampling).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizedRect {
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
}

impl NormalizedRect {
    pub fn new(left: f32, top: f32, width: f32, height: f32) -> Result<Self, DomainError> {
        let valid = (0.0..=1.0).contains(&left)
            && (0.0..=1.0).contains(&top)
            && width > 0.0
            && height > 0.0
            && left + width <= 1.0001
            && top + height <= 1.0001;
        if !valid {
            return Err(DomainError::InvalidCropRect {
                left,
                top,
                width,
                height,
            });
        }
        Ok(Self {
            left,
            top,
            width: width.min(1.0 - left),
            height: height.min(1.0 - top),
        })
    }

    pub fn full_frame() -> Self {
        Self {
            left: 0.0,
            top: 0.0,
            width: 1.0,
            height: 1.0,
        }
    }

    pub fn is_full_frame(&self) -> bool {
        self.left <= f32::EPSILON
            && self.top <= f32::EPSILON
            && (self.width - 1.0).abs() <= 1e-4
            && (self.height - 1.0).abs() <= 1e-4
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CropAspectRatio {
    #[default]
    Original,
    Free,
    Ratio4x5,
    Ratio2x3,
    Ratio5x7,
    Ratio1x1,
    Ratio16x9,
}

impl CropAspectRatio {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Original => "original",
            Self::Free => "free",
            Self::Ratio4x5 => "4:5",
            Self::Ratio2x3 => "2:3",
            Self::Ratio5x7 => "5:7",
            Self::Ratio1x1 => "1:1",
            Self::Ratio16x9 => "16:9",
        }
    }

    pub fn parse_label(s: &str) -> Self {
        match s {
            "free" => Self::Free,
            "4:5" => Self::Ratio4x5,
            "2:3" => Self::Ratio2x3,
            "5:7" => Self::Ratio5x7,
            "1:1" => Self::Ratio1x1,
            "16:9" => Self::Ratio16x9,
            _ => Self::Original,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CropTransform {
    pub rect: NormalizedRect,
    /// Rotation angle in degrees (-45.0 to +45.0)
    pub rotation_deg: f32,
    pub aspect_ratio: CropAspectRatio,
    pub flip_horizontal: bool,
    pub flip_vertical: bool,
    /// 90-degree clockwise steps (`0..=3`)
    pub quarter_turns: u8,
    pub auto_eye_level_locked: bool,
}

impl CropTransform {
    pub fn clamp_invariants(&mut self) {
        self.rotation_deg = self.rotation_deg.clamp(-45.0, 45.0);
        self.quarter_turns %= 4;
    }

    pub fn is_identity(&self) -> bool {
        self.rect.is_full_frame()
            && self.rotation_deg.abs() <= 1e-4
            && !self.flip_horizontal
            && !self.flip_vertical
            && self.quarter_turns.is_multiple_of(4)
    }
}

impl Default for CropTransform {
    fn default() -> Self {
        Self {
            rect: NormalizedRect::full_frame(),
            rotation_deg: 0.0,
            aspect_ratio: CropAspectRatio::Original,
            flip_horizontal: false,
            flip_vertical: false,
            quarter_turns: 0,
            auto_eye_level_locked: false,
        }
    }
}

use crate::masking::{BackgroundRemoval, MaskLayer};

/// Non-destructive Lightroom-class Parametric Develop Recipe.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DevelopRecipe {
    pub white_balance: WhiteBalance,
    pub basic_tone: BasicTone,
    pub tone_curve: ParametricToneCurve,
    pub hsl_mixer: HslColorMixer,
    pub color_grading: ColorGrading,
    pub camera_calibration: CameraCalibration,
    pub lut_3d: Option<Lut3DReference>,
    pub detail_optics: DetailAndOptics,
    pub crop_transform: CropTransform,
    pub masks: Vec<MaskLayer>,
    pub background_removal: BackgroundRemoval,
}

/// Selective checklist mask for Lightroom-style `Sync Settings` (`Ctrl+Shift+S`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DevelopSyncMask {
    pub sync_white_balance: bool,
    pub sync_basic_tone: bool,
    pub sync_tone_curve: bool,
    pub sync_hsl_mixer: bool,
    pub sync_color_grading: bool,
    pub sync_camera_calibration: bool,
    pub sync_lut_3d: bool,
    pub sync_detail_optics: bool,
    pub sync_crop_transform: bool,
    pub sync_masks: bool,
}

impl DevelopSyncMask {
    /// Default Lightroom batch sync: syncs all color/tone grading, preserves per-photo crops & masks.
    pub fn color_and_tone_only() -> Self {
        Self {
            sync_white_balance: true,
            sync_basic_tone: true,
            sync_tone_curve: true,
            sync_hsl_mixer: true,
            sync_color_grading: true,
            sync_camera_calibration: true,
            sync_lut_3d: true,
            sync_detail_optics: true,
            sync_crop_transform: false,
            sync_masks: false,
        }
    }

    pub fn all() -> Self {
        Self {
            sync_white_balance: true,
            sync_basic_tone: true,
            sync_tone_curve: true,
            sync_hsl_mixer: true,
            sync_color_grading: true,
            sync_camera_calibration: true,
            sync_lut_3d: true,
            sync_detail_optics: true,
            sync_crop_transform: true,
            sync_masks: true,
        }
    }
}

/// Relative delta nudge for Lightroom "Quick Develop" multi-photo batch adjustments.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RelativeDevelopDelta {
    pub exposure_delta_ev: f32,
    pub temperature_delta_kelvin: i32,
    pub tint_delta: i16,
    pub contrast_delta: i16,
    pub highlights_delta: i16,
    pub shadows_delta: i16,
    pub vibrance_delta: i16,
    pub saturation_delta: i16,
}

impl DevelopRecipe {
    pub fn clamp_all_invariants(&mut self) {
        self.white_balance.temperature_kelvin = self
            .white_balance
            .temperature_kelvin
            .clamp(MIN_KELVIN, MAX_KELVIN);
        self.white_balance.tint = self.white_balance.tint.clamp(-150, 150);
        self.basic_tone.clamp_invariants();
        self.tone_curve.clamp_invariants();
        self.hsl_mixer.clamp_invariants();
        self.color_grading.clamp_invariants();
        self.camera_calibration.clamp_invariants();
        if let Some(lut) = &mut self.lut_3d {
            lut.intensity = lut.intensity.clamp(0.0, 1.0);
        }
        self.detail_optics.clamp_invariants();
        self.crop_transform.clamp_invariants();
        for mask in &mut self.masks {
            mask.clamp_invariants();
        }
        self.background_removal.clamp_invariants();
    }

    /// Applies selected modules from `source` onto `self` according to `mask`.
    pub fn sync_from(&mut self, source: &DevelopRecipe, mask: DevelopSyncMask) {
        if mask.sync_white_balance {
            self.white_balance = source.white_balance;
        }
        if mask.sync_basic_tone {
            self.basic_tone = source.basic_tone;
        }
        if mask.sync_tone_curve {
            self.tone_curve = source.tone_curve.clone();
        }
        if mask.sync_hsl_mixer {
            self.hsl_mixer = source.hsl_mixer;
        }
        if mask.sync_color_grading {
            self.color_grading = source.color_grading;
        }
        if mask.sync_camera_calibration {
            self.camera_calibration = source.camera_calibration;
        }
        if mask.sync_lut_3d {
            self.lut_3d = source.lut_3d.clone();
        }
        if mask.sync_detail_optics {
            self.detail_optics = source.detail_optics;
        }
        if mask.sync_crop_transform {
            self.crop_transform = source.crop_transform;
        }
        if mask.sync_masks {
            self.masks = source.masks.clone();
            self.background_removal = source.background_removal.clone();
        }
        self.clamp_all_invariants();
    }

    /// Shifts this recipe's parameters relatively while enforcing safe domain bounds.
    pub fn apply_relative_delta(&mut self, delta: RelativeDevelopDelta) {
        let new_temp = (self.white_balance.temperature_kelvin as i64
            + delta.temperature_delta_kelvin as i64)
            .clamp(MIN_KELVIN as i64, MAX_KELVIN as i64) as u32;
        self.white_balance.temperature_kelvin = new_temp;
        self.white_balance.tint = (self.white_balance.tint + delta.tint_delta).clamp(-150, 150);

        self.basic_tone.exposure_ev += delta.exposure_delta_ev;
        self.basic_tone.contrast += delta.contrast_delta;
        self.basic_tone.highlights += delta.highlights_delta;
        self.basic_tone.shadows += delta.shadows_delta;
        self.basic_tone.vibrance += delta.vibrance_delta;
        self.basic_tone.saturation += delta.saturation_delta;
        self.clamp_all_invariants();
    }
}

/// Non-destructive Lightroom-style undo/redo history stack for a single photo's recipe.
#[derive(Debug, Clone, PartialEq)]
pub struct DevelopHistory {
    entries: Vec<(String, DevelopRecipe)>,
    cursor: usize,
}

impl DevelopHistory {
    pub fn new(initial: DevelopRecipe) -> Self {
        Self {
            entries: vec![("Import Original".to_string(), initial)],
            cursor: 0,
        }
    }

    pub fn current(&self) -> &DevelopRecipe {
        &self.entries[self.cursor].1
    }

    pub fn current_label(&self) -> &str {
        &self.entries[self.cursor].0
    }

    pub fn push(&mut self, step_label: impl Into<String>, recipe: DevelopRecipe) {
        self.entries.truncate(self.cursor + 1);
        self.entries.push((step_label.into(), recipe));
        self.cursor = self.entries.len() - 1;
    }

    pub fn undo(&mut self) -> bool {
        if self.cursor == 0 {
            return false;
        }
        self.cursor -= 1;
        true
    }

    pub fn redo(&mut self) -> bool {
        if self.cursor + 1 >= self.entries.len() {
            return false;
        }
        self.cursor += 1;
        true
    }

    pub fn step_count(&self) -> usize {
        self.entries.len()
    }
}

/// 256-bin R, G, B, and Luminance histogram for darkroom telemetry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbHistogram {
    pub red: [u32; 256],
    pub green: [u32; 256],
    pub blue: [u32; 256],
    pub luminance: [u32; 256],
    pub total_pixels: u64,
}
