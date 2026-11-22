//! Non-destructive Local & AI Masking domain model (Phase 2).
//!
//! Pure framework-free domain entities and invariants shared by WebGL2 real-time shading,
//! Rust 32-bit float export, `.kroma.json` sidecar persistence, and Batch Sync.

/// Boolean composition operator combining a component onto its parent mask layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MaskCombineOp {
    #[default]
    Add,
    Subtract,
    Intersect,
}

impl MaskCombineOp {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Subtract => "subtract",
            Self::Intersect => "intersect",
        }
    }

    pub fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "subtract" => Self::Subtract,
            "intersect" => Self::Intersect,
            _ => Self::Add,
        }
    }
}

/// Supported Kroma-native AI mask kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AiMaskKind {
    #[default]
    Subject,
    Background,
    FaceSkin,
    Sky,
}

impl AiMaskKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Subject => "subject",
            Self::Background => "background",
            Self::FaceSkin => "face_skin",
            Self::Sky => "sky",
        }
    }

    pub fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "background" | "bg" => Self::Background,
            "face_skin" | "skin" | "face" => Self::FaceSkin,
            "sky" => Self::Sky,
            _ => Self::Subject,
        }
    }
}

/// A single continuous freehand brush stroke in oriented full-frame normalized space `[0..1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct BrushStroke {
    /// Brush radius as a fraction of the shorter image dimension (`0.004..=0.35`).
    pub size: f32,
    /// Soft edge feathering (`0.0..=1.0`, where `0` is hard edge and `1` is full soft falloff).
    pub feather: f32,
    /// Per-stroke opacity cap (`0.05..=1.0`).
    pub flow: f32,
    /// Ordered polyline vertices `(u, v)` in `[0.0..=1.0]`.
    pub points: Vec<[f32; 2]>,
}

impl BrushStroke {
    pub fn clamp_invariants(&mut self) {
        self.size = if self.size.is_finite() {
            self.size.clamp(0.004, 0.35)
        } else {
            0.06
        };
        self.feather = if self.feather.is_finite() {
            self.feather.clamp(0.0, 1.0)
        } else {
            0.5
        };
        self.flow = if self.flow.is_finite() {
            self.flow.clamp(0.05, 1.0)
        } else {
            1.0
        };
        for pt in &mut self.points {
            pt[0] = if pt[0].is_finite() {
                pt[0].clamp(-0.25, 1.25)
            } else {
                0.5
            };
            pt[1] = if pt[1].is_finite() {
                pt[1].clamp(-0.25, 1.25)
            } else {
                0.5
            };
        }
    }
}

/// Composable manual or AI mask primitive inside a `MaskLayer`.
#[derive(Debug, Clone, PartialEq)]
pub enum MaskComponent {
    LinearGradient {
        id: String,
        op: MaskCombineOp,
        inverted: bool,
        start_x: f32,
        start_y: f32,
        end_x: f32,
        end_y: f32,
        /// Feather multiplier (`0..=100`, default 50).
        feather: f32,
    },
    RadialGradient {
        id: String,
        op: MaskCombineOp,
        inverted: bool,
        center_x: f32,
        center_y: f32,
        radius_x: f32,
        radius_y: f32,
        rotation_deg: f32,
        /// Soft inner feathering (`0..=100`, default 50).
        feather: f32,
    },
    Brush {
        id: String,
        op: MaskCombineOp,
        inverted: bool,
        strokes: Vec<BrushStroke>,
    },
    LuminanceRange {
        id: String,
        op: MaskCombineOp,
        inverted: bool,
        /// Selected lower luminance bound (`0.0..=1.0`).
        min_luma: f32,
        /// Selected upper luminance bound (`0.0..=1.0`).
        max_luma: f32,
        /// Smooth transition width (`0.0..=0.5`).
        feather: f32,
    },
    ColorRange {
        id: String,
        op: MaskCombineOp,
        inverted: bool,
        /// Target hue in degrees (`0.0..360.0`).
        target_hue_deg: f32,
        /// Target HSL saturation (`0.0..=1.0`).
        target_sat: f32,
        /// Target HSL lightness (`0.0..=1.0`).
        target_luma: f32,
        /// Chromatic tolerance (`0.02..=1.0`).
        tolerance: f32,
        /// Soft rolloff (`0.0..=1.0`).
        feather: f32,
    },
    AiMatte {
        id: String,
        op: MaskCombineOp,
        inverted: bool,
        kind: AiMaskKind,
        /// Edge softness adjustment (`0..=100`, default 25).
        feather: f32,
        matte_width: u16,
        matte_height: u16,
        /// Compact PackBits-RLE + Base64 encoded 8-bit grayscale alpha matte.
        matte_rle_b64: String,
    },
}

impl MaskComponent {
    pub fn clamp_invariants(&mut self) {
        match self {
            Self::LinearGradient {
                start_x,
                start_y,
                end_x,
                end_y,
                feather,
                ..
            } => {
                *start_x = clamp_finite(*start_x, -0.5, 1.5, 0.5);
                *start_y = clamp_finite(*start_y, -0.5, 1.5, 0.2);
                *end_x = clamp_finite(*end_x, -0.5, 1.5, 0.5);
                *end_y = clamp_finite(*end_y, -0.5, 1.5, 0.8);
                *feather = clamp_finite(*feather, 0.0, 100.0, 50.0);
            }
            Self::RadialGradient {
                center_x,
                center_y,
                radius_x,
                radius_y,
                rotation_deg,
                feather,
                ..
            } => {
                *center_x = clamp_finite(*center_x, -0.5, 1.5, 0.5);
                *center_y = clamp_finite(*center_y, -0.5, 1.5, 0.5);
                *radius_x = clamp_finite(*radius_x, 0.02, 1.5, 0.30);
                *radius_y = clamp_finite(*radius_y, 0.02, 1.5, 0.35);
                *rotation_deg = clamp_finite(*rotation_deg, -180.0, 180.0, 0.0);
                *feather = clamp_finite(*feather, 0.0, 100.0, 50.0);
            }
            Self::Brush { strokes, .. } => {
                for stroke in strokes {
                    stroke.clamp_invariants();
                }
            }
            Self::LuminanceRange {
                min_luma,
                max_luma,
                feather,
                ..
            } => {
                *min_luma = clamp_finite(*min_luma, 0.0, 1.0, 0.35);
                *max_luma = clamp_finite(*max_luma, 0.0, 1.0, 0.85);
                if *min_luma > *max_luma {
                    std::mem::swap(min_luma, max_luma);
                }
                *feather = clamp_finite(*feather, 0.0, 0.5, 0.12);
            }
            Self::ColorRange {
                target_hue_deg,
                target_sat,
                target_luma,
                tolerance,
                feather,
                ..
            } => {
                *target_hue_deg = if target_hue_deg.is_finite() {
                    target_hue_deg.rem_euclid(360.0)
                } else {
                    0.0
                };
                *target_sat = clamp_finite(*target_sat, 0.0, 1.0, 0.5);
                *target_luma = clamp_finite(*target_luma, 0.0, 1.0, 0.5);
                *tolerance = clamp_finite(*tolerance, 0.02, 1.0, 0.25);
                *feather = clamp_finite(*feather, 0.0, 1.0, 0.35);
            }
            Self::AiMatte {
                feather,
                matte_width,
                matte_height,
                ..
            } => {
                *feather = clamp_finite(*feather, 0.0, 100.0, 25.0);
                *matte_width = (*matte_width).min(1024);
                *matte_height = (*matte_height).min(1024);
            }
        }
    }

    pub fn is_range_component(&self) -> bool {
        matches!(self, Self::LuminanceRange { .. } | Self::ColorRange { .. })
    }
}

/// Per-mask non-destructive local adjustments (matching Lightroom's Local Adjustment stack).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LocalAdjustments {
    /// Relative warmth/coolness shift (`-100..=100`).
    pub temperature_shift: i16,
    /// Relative green/magenta shift (`-100..=100`).
    pub tint_shift: i16,
    /// Local exposure in EV stops (`-4.0..=4.0`).
    pub exposure_ev: f32,
    pub contrast: i16,
    pub highlights: i16,
    pub shadows: i16,
    pub whites: i16,
    pub blacks: i16,
    pub texture: i16,
    pub clarity: i16,
    pub dehaze: i16,
    pub saturation: i16,
    /// Local sharpness (`-100..=100`; negative values smoothly soften skin).
    pub sharpness: i16,
}

impl LocalAdjustments {
    pub fn clamp_invariants(&mut self) {
        self.temperature_shift = self.temperature_shift.clamp(-100, 100);
        self.tint_shift = self.tint_shift.clamp(-100, 100);
        self.exposure_ev = clamp_finite(self.exposure_ev, -4.0, 4.0, 0.0);
        self.contrast = self.contrast.clamp(-100, 100);
        self.highlights = self.highlights.clamp(-100, 100);
        self.shadows = self.shadows.clamp(-100, 100);
        self.whites = self.whites.clamp(-100, 100);
        self.blacks = self.blacks.clamp(-100, 100);
        self.texture = self.texture.clamp(-100, 100);
        self.clarity = self.clarity.clamp(-100, 100);
        self.dehaze = self.dehaze.clamp(-100, 100);
        self.saturation = self.saturation.clamp(-100, 100);
        self.sharpness = self.sharpness.clamp(-100, 100);
    }

    pub fn is_neutral(&self) -> bool {
        self.temperature_shift == 0
            && self.tint_shift == 0
            && self.exposure_ev.abs() <= 1e-4
            && self.contrast == 0
            && self.highlights == 0
            && self.shadows == 0
            && self.whites == 0
            && self.blacks == 0
            && self.texture == 0
            && self.clarity == 0
            && self.dehaze == 0
            && self.saturation == 0
            && self.sharpness == 0
    }

    pub fn has_spatial_adjustments(&self) -> bool {
        self.texture != 0 || self.clarity != 0 || self.sharpness != 0
    }
}

/// Named non-destructive mask layer containing composable sub-masks and local adjustments.
#[derive(Debug, Clone, PartialEq)]
pub struct MaskLayer {
    pub id: String,
    pub name: String,
    pub enabled: bool,
    pub inverted: bool,
    /// Global mask opacity (`0..=100`, default `100`).
    pub opacity: u8,
    pub components: Vec<MaskComponent>,
    pub adjustments: LocalAdjustments,
}

impl MaskLayer {
    pub fn clamp_invariants(&mut self) {
        if self.name.trim().is_empty() {
            self.name = "Mask".to_string();
        }
        self.opacity = self.opacity.min(100);
        for comp in &mut self.components {
            comp.clamp_invariants();
        }
        self.adjustments.clamp_invariants();
    }

    /// True when this layer will visibly change rendered pixels.
    pub fn is_effective(&self) -> bool {
        self.enabled
            && self.opacity > 0
            && !self.components.is_empty()
            && !self.adjustments.is_neutral()
    }
}

/// Non-destructive AI Background Removal / Studio Backdrop Replacement mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackgroundReplaceMode {
    #[default]
    Original,
    Transparent,
    SolidColor,
    Gradient,
}

impl BackgroundReplaceMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Original => "original",
            Self::Transparent => "transparent",
            Self::SolidColor => "solid_color",
            Self::Gradient => "gradient",
        }
    }

    pub fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "transparent" | "cutout" => Self::Transparent,
            "solid_color" | "solid" | "color" => Self::SolidColor,
            "gradient" | "grad" => Self::Gradient,
            _ => Self::Original,
        }
    }
}

/// Studio backdrop gradient geometry when `BackgroundReplaceMode::Gradient` is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackgroundGradientStyle {
    /// Classic portrait studio spotlight centered behind the subject (`0.5, 0.42`).
    #[default]
    Radial,
    /// Directional linear gradient matching CSS `linear-gradient(angle deg, color1, color2)`.
    Linear,
}

impl BackgroundGradientStyle {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Radial => "radial",
            Self::Linear => "linear",
        }
    }

    pub fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "linear" => Self::Linear,
            _ => Self::Radial,
        }
    }
}

/// Non-destructive AI background cutout / solid or gradient studio backdrop backed by `RMBG-1.4`.
#[derive(Debug, Clone, PartialEq)]
pub struct BackgroundRemoval {
    pub enabled: bool,
    pub mode: BackgroundReplaceMode,
    /// Solid studio backdrop RGB `[0..=255]`, or Gradient Color 1 (center / start).
    pub color_rgb: [u8; 3],
    /// Gradient Color 2 (outer / end) RGB `[0..=255]` used when `mode == Gradient`.
    pub color2_rgb: [u8; 3],
    /// Gradient geometry (`Radial` studio spotlight or `Linear` angle wash).
    pub gradient_style: BackgroundGradientStyle,
    /// Linear gradient direction in CSS degrees (`0..=360`, default `180` = top to bottom).
    pub gradient_angle_deg: f32,
    /// Edge softness refinement (`0..=100`, default `25` = neutral neural matte).
    pub feather: f32,
    pub matte_width: u16,
    pub matte_height: u16,
    /// Compact PackBits-RLE + Base64 subject foreground alpha matte.
    pub matte_rle_b64: String,
}

impl Default for BackgroundRemoval {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: BackgroundReplaceMode::Original,
            color_rgb: [74, 79, 90],
            color2_rgb: [20, 22, 27],
            gradient_style: BackgroundGradientStyle::Radial,
            gradient_angle_deg: 180.0,
            feather: 25.0,
            matte_width: 0,
            matte_height: 0,
            matte_rle_b64: String::new(),
        }
    }
}

impl BackgroundRemoval {
    pub fn clamp_invariants(&mut self) {
        self.gradient_angle_deg = if self.gradient_angle_deg.is_finite() {
            self.gradient_angle_deg.rem_euclid(360.0)
        } else {
            180.0
        };
        self.feather = clamp_finite(self.feather, 0.0, 100.0, 25.0);
        self.matte_width = self.matte_width.min(1024);
        self.matte_height = self.matte_height.min(1024);
    }

    pub fn is_active(&self) -> bool {
        self.enabled
            && self.mode != BackgroundReplaceMode::Original
            && self.matte_width > 0
            && self.matte_height > 0
            && !self.matte_rle_b64.is_empty()
    }
}

fn clamp_finite(value: f32, min: f32, max: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        fallback
    }
}

const B64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encodes an 8-bit grayscale alpha matte with PackBits RLE followed by standard Base64.
/// Shrinks solid foreground/background regions by ~85–95% with zero external dependencies.
pub fn encode_matte_rle_b64(bytes: &[u8]) -> String {
    let rle = packbits_encode(bytes);
    base64_encode(&rle)
}

/// Decodes a PackBits-RLE + Base64 grayscale matte produced by `encode_matte_rle_b64`.
/// Returns `None` if malformed or if decoded byte count mismatches `expected_len`.
pub fn decode_matte_rle_b64(encoded: &str, expected_len: usize) -> Option<Vec<u8>> {
    if expected_len == 0 {
        return Some(Vec::new());
    }
    let rle = base64_decode(encoded)?;
    let decoded = packbits_decode(&rle, expected_len)?;
    if decoded.len() == expected_len {
        Some(decoded)
    } else {
        None
    }
}

fn packbits_encode(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() / 2 + 8);
    let n = input.len();
    let mut i = 0;

    while i < n {
        // Count identical byte run (min 3 to justify repeat packet)
        let mut run = 1;
        while i + run < n && run < 128 && input[i + run] == input[i] {
            run += 1;
        }
        if run >= 3 {
            out.push((257 - run) as u8);
            out.push(input[i]);
            i += run;
        } else {
            let start = i;
            let mut lit = 0;
            while i < n && lit < 128 {
                if i + 2 < n && input[i] == input[i + 1] && input[i] == input[i + 2] {
                    break;
                }
                i += 1;
                lit += 1;
            }
            if lit > 0 {
                out.push((lit - 1) as u8);
                out.extend_from_slice(&input[start..start + lit]);
            }
        }
    }

    out
}

fn packbits_decode(rle: &[u8], max_out: usize) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(max_out);
    let mut i = 0;
    while i < rle.len() {
        let header = rle[i];
        i += 1;
        if header == 128 {
            continue;
        } else if header < 128 {
            let count = (header as usize) + 1;
            if i + count > rle.len() || out.len() + count > max_out {
                return None;
            }
            out.extend_from_slice(&rle[i..i + count]);
            i += count;
        } else {
            let count = 257 - (header as usize);
            if i >= rle.len() || out.len() + count > max_out {
                return None;
            }
            let val = rle[i];
            i += 1;
            out.resize(out.len() + count, val);
        }
    }
    Some(out)
}

fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    let mut chunks = bytes.chunks_exact(3);
    for chunk in &mut chunks {
        let n = ((chunk[0] as u32) << 16) | ((chunk[1] as u32) << 8) | (chunk[2] as u32);
        out.push(B64_ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(B64_ALPHABET[((n >> 12) & 63) as usize] as char);
        out.push(B64_ALPHABET[((n >> 6) & 63) as usize] as char);
        out.push(B64_ALPHABET[(n & 63) as usize] as char);
    }
    let rem = chunks.remainder();
    if rem.len() == 1 {
        let n = (rem[0] as u32) << 16;
        out.push(B64_ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(B64_ALPHABET[((n >> 12) & 63) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rem.len() == 2 {
        let n = ((rem[0] as u32) << 16) | ((rem[1] as u32) << 8);
        out.push(B64_ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(B64_ALPHABET[((n >> 12) & 63) as usize] as char);
        out.push(B64_ALPHABET[((n >> 6) & 63) as usize] as char);
        out.push('=');
    }
    out
}

fn b64_val(b: u8) -> Option<u8> {
    match b {
        b'A'..=b'Z' => Some(b - b'A'),
        b'a'..=b'z' => Some(b - b'a' + 26),
        b'0'..=b'9' => Some(b - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let clean: Vec<u8> = s
        .bytes()
        .filter(|b| !b.is_ascii_whitespace())
        .collect();
    if !clean.len().is_multiple_of(4) {
        return None;
    }
    let mut out = Vec::with_capacity((clean.len() / 4) * 3);
    for chunk in clean.chunks_exact(4) {
        let a = b64_val(chunk[0])?;
        let b = b64_val(chunk[1])?;
        out.push((a << 2) | (b >> 4));
        if chunk[2] != b'=' {
            let c = b64_val(chunk[2])?;
            out.push((b << 4) | (c >> 2));
            if chunk[3] != b'=' {
                let d = b64_val(chunk[3])?;
                out.push((c << 6) | d);
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matte_rle_base64_roundtrip_compresses_smooth_portrait_mask() {
        let mut matte = vec![0u8; 160 * 120];
        for y in 20..100 {
            for x in 40..120 {
                let edge = ((x - 40).min(119 - x).min(y - 20).min(99 - y)) as u8;
                matte[y * 160 + x] = edge.saturating_mul(24);
            }
        }
        let encoded = encode_matte_rle_b64(&matte);
        assert!(
            encoded.len() < matte.len() / 4,
            "expected >4x RLE compression on portrait matte, got {} vs {} bytes",
            encoded.len(),
            matte.len()
        );
        let decoded = decode_matte_rle_b64(&encoded, matte.len()).expect("valid RLE+b64 decode");
        assert_eq!(decoded, matte);
    }

    #[test]
    fn matte_rle_base64_handles_short_and_random_patterns() {
        let sample: Vec<u8> = (0..257).map(|i| ((i * 73 + 19) % 256) as u8).collect();
        let encoded = encode_matte_rle_b64(&sample);
        let decoded = decode_matte_rle_b64(&encoded, sample.len()).expect("roundtrip");
        assert_eq!(decoded, sample);
    }

    #[test]
    fn mask_layer_clamps_invariants_and_checks_effectiveness() {
        let mut layer = MaskLayer {
            id: "m1".to_string(),
            name: "   ".to_string(),
            enabled: true,
            inverted: false,
            opacity: 150,
            components: vec![MaskComponent::LuminanceRange {
                id: "c1".to_string(),
                op: MaskCombineOp::Add,
                inverted: false,
                min_luma: 0.9,
                max_luma: 0.2,
                feather: 2.0,
            }],
            adjustments: LocalAdjustments {
                exposure_ev: 9.0,
                sharpness: -140,
                ..LocalAdjustments::default()
            },
        };
        layer.clamp_invariants();
        assert_eq!(layer.name, "Mask");
        assert_eq!(layer.opacity, 100);
        assert_eq!(layer.adjustments.exposure_ev, 4.0);
        assert_eq!(layer.adjustments.sharpness, -100);
        assert!(layer.is_effective());
        if let MaskComponent::LuminanceRange {
            min_luma,
            max_luma,
            feather,
            ..
        } = &layer.components[0]
        {
            assert!((*min_luma - 0.2).abs() < 1e-6);
            assert!((*max_luma - 0.9).abs() < 1e-6);
            assert!((*feather - 0.5).abs() < 1e-6);
        } else {
            panic!("expected LuminanceRange");
        }
    }
}
