//! Non-destructive Portrait Retouching & Spot Heal / Clone domain model (v0.3.0).
//!
//! Pure framework-free domain entities and invariants shared by:
//! - Shared WGSL compute shaders (`shaders/develop/{heal_pass,portrait_retouch_pass}.wgsl`)
//! - Rust `wgpu` DX12/Metal export and Rayon 32-bit CPU fallback
//! - `.kroma.json` sidecar persistence and Lightroom-style Batch Sync.

use crate::develop::{CropTransform, NormalizedRect};
use crate::identity::{BoundingBox, Point2D, YuNetLandmarks};

/// Maximum number of non-destructive heal/clone spots per photo recipe.
pub const MAX_HEAL_SPOTS: usize = 256;
/// Maximum number of non-destructive AI inpaint patches per photo recipe.
pub const MAX_INPAINT_PATCHES: usize = 32;
/// Maximum number of vertices in a single continuous heal/clone stroke.
pub const MAX_HEAL_STROKE_POINTS: usize = 96;
/// Maximum number of faces packed into a single GPU portrait-retouch dispatch.
pub const MAX_PORTRAIT_FACE_GUIDES: usize = 4;

/// Pixel transfer mode for a non-destructive spot/stroke retouch patch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HealSpotMode {
    /// Seamless ring-matched harmonic tone/luminance healing that transfers donor pore texture
    /// while smoothly matching destination lighting gradients.
    #[default]
    Heal,
    /// Direct feathered pixel copy from the donor location.
    Clone,
}

impl HealSpotMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Heal => "heal",
            Self::Clone => "clone",
        }
    }

    pub fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "clone" => Self::Clone,
            _ => Self::Heal,
        }
    }
}

/// Non-destructive Spot Heal or Clone patch (single-click circle or continuous brush stroke)
/// in oriented full-frame normalized coordinates `[0..1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct HealSpot {
    pub id: String,
    pub mode: HealSpotMode,
    /// Destination blemish center `(u, v)` in oriented full-frame `[0..1]`.
    pub target_x: f32,
    pub target_y: f32,
    /// Donor clean patch center `(u, v)` in oriented full-frame `[0..1]`.
    pub source_x: f32,
    pub source_y: f32,
    /// Brush radius as a fraction of the shorter oriented image dimension (`0.002..=0.20`).
    pub radius: f32,
    /// Edge feather softness (`0.0..=1.0`, default `0.55`).
    pub feather: f32,
    /// Blend opacity (`0.05..=1.0`, default `1.0`).
    pub opacity: f32,
    /// Optional polyline vertices `(u, v)` for continuous stray-hair / wrinkle strokes.
    /// When empty or length 1, the spot renders as a single circle at `(target_x, target_y)`.
    pub points: Vec<[f32; 2]>,
}

impl HealSpot {
    pub fn clamp_invariants(&mut self) {
        self.target_x = sanitize_coord(self.target_x, 0.5);
        self.target_y = sanitize_coord(self.target_y, 0.5);
        self.source_x = sanitize_coord(self.source_x, (self.target_x + 0.04).clamp(0.0, 1.0));
        self.source_y = sanitize_coord(self.source_y, self.target_y);
        self.radius = if self.radius.is_finite() {
            self.radius.clamp(0.002, 0.20)
        } else {
            0.018
        };
        self.feather = if self.feather.is_finite() {
            self.feather.clamp(0.0, 1.0)
        } else {
            0.55
        };
        self.opacity = if self.opacity.is_finite() {
            self.opacity.clamp(0.05, 1.0)
        } else {
            1.0
        };
        if self.points.len() > MAX_HEAL_STROKE_POINTS {
            self.points.truncate(MAX_HEAL_STROKE_POINTS);
        }
        for pt in &mut self.points {
            pt[0] = sanitize_coord(pt[0], self.target_x);
            pt[1] = sanitize_coord(pt[1], self.target_y);
        }
    }

    pub fn is_effective(&self) -> bool {
        self.radius >= 0.002 && self.opacity >= 0.05
    }

    /// Donor offset `(du, dv)` applied uniformly across the circular spot or polyline stroke.
    pub fn donor_delta(&self) -> (f32, f32) {
        (self.source_x - self.target_x, self.source_y - self.target_y)
    }
}

/// Non-destructive Frequency-Separation & Landmark-Guided Portrait Retouching parameters.
/// Fully batch-syncable across an entire shoot because facial coordinates are resolved per-photo
/// from each photo's own `PortraitFaceGuide` rather than stored inside the recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PortraitRetouch {
    /// Master bypass switch allowing instant before/after toggling of all portrait sliders.
    pub enabled: bool,
    /// 3-band frequency-separation skin smoothing (`0..=100`): softens mid-frequency blotchiness
    /// while preserving high-frequency real pores.
    pub skin_smoothing: u8,
    /// High-frequency pore preservation (`0..=100`, default `70`).
    pub skin_texture_preserve: u8,
    /// Skin chroma/redness blotch evening (`0..=100`).
    pub skin_tone_evenness: u8,
    /// Specular skin glare & oil control (`0..=100`).
    pub shine_removal: u8,
    /// Under-eye shadow & dark-circle brightening (`0..=100`), guided by YuNet eye landmarks.
    pub under_eye_brighten: u8,
    /// Eye clarity, subtle iris pop & sclera brightening (`0..=100`), guided by YuNet eye landmarks.
    pub eye_enhance: u8,
    /// Natural teeth whitening & yellow-cast neutralization (`0..=100`), guided by mouth landmarks.
    pub teeth_whiten: u8,
    /// Subtle lip vibrancy & tone definition (`0..=100`), guided by mouth landmarks.
    pub lip_enhance: u8,
}

impl Default for PortraitRetouch {
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

impl PortraitRetouch {
    pub fn clamp_invariants(&mut self) {
        self.skin_smoothing = self.skin_smoothing.min(100);
        self.skin_texture_preserve = self.skin_texture_preserve.min(100);
        self.skin_tone_evenness = self.skin_tone_evenness.min(100);
        self.shine_removal = self.shine_removal.min(100);
        self.under_eye_brighten = self.under_eye_brighten.min(100);
        self.eye_enhance = self.eye_enhance.min(100);
        self.teeth_whiten = self.teeth_whiten.min(100);
        self.lip_enhance = self.lip_enhance.min(100);
    }

    /// Returns `true` when at least one portrait retouch slider has a non-zero visible effect.
    pub fn has_any_slider_active(&self) -> bool {
        self.skin_smoothing > 0
            || self.skin_tone_evenness > 0
            || self.shine_removal > 0
            || self.under_eye_brighten > 0
            || self.eye_enhance > 0
            || self.teeth_whiten > 0
            || self.lip_enhance > 0
    }

    pub fn is_active(&self) -> bool {
        self.enabled && self.has_any_slider_active()
    }

    /// Returns `true` when any landmark-targeted facial feature slider is active.
    pub fn needs_face_landmarks(&self) -> bool {
        self.enabled
            && (self.under_eye_brighten > 0
                || self.eye_enhance > 0
                || self.teeth_whiten > 0
                || self.lip_enhance > 0
                || self.skin_smoothing > 0
                || self.skin_tone_evenness > 0
                || self.shine_removal > 0)
    }
}

/// Non-destructive generative/harmonic inpaint patch stored in oriented full-frame UV space `[0..1]`.
/// Produced on-demand by LaMa ONNX (or built-in harmonic inpaint fallback) and composited seamlessly
/// with perimeter harmonic tone matching across WebGPU preview, WebGL2 fallback, and full-res export.
#[derive(Debug, Clone, PartialEq)]
pub struct InpaintPatch {
    pub id: String,
    /// Tight bounding region in oriented full-frame UV `[0..1]` where the patch applies.
    pub bounds: NormalizedRect,
    pub patch_width: u16,
    pub patch_height: u16,
    /// Blend opacity (`0.05..=1.0`, default `1.0`).
    pub opacity: f32,
    /// PackBits RLE Base64 soft alpha feather mask (`patch_width * patch_height`).
    pub mask_rle_b64: String,
    /// Compact Base64 encoded PNG/JPEG RGB image for the patch (`patch_width x patch_height`).
    pub patch_image_b64: String,
    /// Short engine tag (`"LaMa AI"` or `"Harmonic Fill"`).
    pub engine_label: String,
}

impl InpaintPatch {
    pub fn clamp_invariants(&mut self) {
        let left = sanitize_unit(self.bounds.left, 0.0).clamp(0.0, 0.99);
        let top = sanitize_unit(self.bounds.top, 0.0).clamp(0.0, 0.99);
        let width = sanitize_unit(self.bounds.width, 0.1).clamp(0.005, 1.0 - left);
        let height = sanitize_unit(self.bounds.height, 0.1).clamp(0.005, 1.0 - top);
        self.bounds = NormalizedRect {
            left,
            top,
            width,
            height,
        };
        self.patch_width = self.patch_width.min(1024);
        self.patch_height = self.patch_height.min(1024);
        self.opacity = if self.opacity.is_finite() {
            self.opacity.clamp(0.05, 1.0)
        } else {
            1.0
        };
    }

    pub fn is_effective(&self) -> bool {
        self.patch_width > 0
            && self.patch_height > 0
            && self.opacity >= 0.05
            && !self.mask_rle_b64.is_empty()
            && !self.patch_image_b64.is_empty()
    }
}

/// Transient per-photo 4-channel semantic portrait parsing map generated by opt-in BiSeNet ONNX.
/// Cached per photo (never persisted into batch-syncable recipes) so Portrait Retouch sliders stay
/// 100% batch-syncable across a shoot while gaining pixel-exact skin, eye, teeth, and lip isolation.
///
/// Channel packing (`width * height * 4` bytes, `[0..=255]`):
/// - `R`: skin (face + nose + neck + ears, excluding eyes, eyebrows, eyeglasses, hair, mouth, lips)
/// - `G`: eyes (`l_eye + r_eye`)
/// - `B`: inner mouth / teeth
/// - `A`: lips (`u_lip + l_lip`)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FaceParseMap {
    pub width: u16,
    pub height: u16,
    pub rgba: Vec<u8>,
}

impl FaceParseMap {
    pub fn is_valid(&self) -> bool {
        let expected = (self.width as usize)
            .saturating_mul(self.height as usize)
            .saturating_mul(4);
        self.width > 0 && self.height > 0 && expected > 0 && self.rgba.len() == expected
    }
}

/// Complete non-destructive retouching block stored inside `DevelopRecipe`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RetouchSettings {
    pub heal_spots: Vec<HealSpot>,
    pub inpaint_patches: Vec<InpaintPatch>,
    pub portrait: PortraitRetouch,
}

impl RetouchSettings {
    pub fn clamp_invariants(&mut self) {
        if self.heal_spots.len() > MAX_HEAL_SPOTS {
            self.heal_spots.truncate(MAX_HEAL_SPOTS);
        }
        for spot in &mut self.heal_spots {
            spot.clamp_invariants();
        }
        if self.inpaint_patches.len() > MAX_INPAINT_PATCHES {
            self.inpaint_patches.truncate(MAX_INPAINT_PATCHES);
        }
        for patch in &mut self.inpaint_patches {
            patch.clamp_invariants();
        }
        self.portrait.clamp_invariants();
    }

    pub fn has_heal_spots(&self) -> bool {
        self.heal_spots.iter().any(HealSpot::is_effective)
    }

    pub fn has_inpaint_patches(&self) -> bool {
        self.inpaint_patches.iter().any(InpaintPatch::is_effective)
    }

    pub fn is_portrait_active(&self) -> bool {
        self.portrait.is_active()
    }

    pub fn is_active(&self) -> bool {
        self.has_heal_spots() || self.has_inpaint_patches() || self.is_portrait_active()
    }
}

fn sanitize_unit(v: f32, fallback: f32) -> f32 {
    if v.is_finite() { v } else { fallback }
}

/// Per-photo face box and 5-point YuNet landmarks in normalized `[0..1]` image coordinates.
/// Derived from each photo's own detection (never baked into a batch-synced recipe).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PortraitFaceGuide {
    pub box_left: f32,
    pub box_top: f32,
    pub box_width: f32,
    pub box_height: f32,
    pub right_eye: [f32; 2],
    pub left_eye: [f32; 2],
    pub nose_tip: [f32; 2],
    pub right_mouth: [f32; 2],
    pub left_mouth: [f32; 2],
}

impl PortraitFaceGuide {
    /// Builds a normalized `[0..1]` guide from YuNet pixel-space detection & image dimensions.
    pub fn from_pixel_detection(
        bbox: BoundingBox,
        landmarks: YuNetLandmarks,
        image_width: u32,
        image_height: u32,
    ) -> Self {
        let w = (image_width as f32).max(1.0);
        let h = (image_height as f32).max(1.0);
        Self::from_normalized(
            bbox.x / w,
            bbox.y / h,
            bbox.width / w,
            bbox.height / h,
            [landmarks.right_eye.x / w, landmarks.right_eye.y / h],
            [landmarks.left_eye.x / w, landmarks.left_eye.y / h],
            [landmarks.nose_tip.x / w, landmarks.nose_tip.y / h],
            [
                landmarks.right_mouth_corner.x / w,
                landmarks.right_mouth_corner.y / h,
            ],
            [
                landmarks.left_mouth_corner.x / w,
                landmarks.left_mouth_corner.y / h,
            ],
        )
    }

    /// Builds a normalized `[0..1]` guide from already-normalized coordinates (`[0..1]`).
    #[allow(clippy::too_many_arguments)]
    pub fn from_normalized(
        box_left: f32,
        box_top: f32,
        box_width: f32,
        box_height: f32,
        right_eye: [f32; 2],
        left_eye: [f32; 2],
        nose_tip: [f32; 2],
        right_mouth: [f32; 2],
        left_mouth: [f32; 2],
    ) -> Self {
        Self {
            box_left: box_left.clamp(0.0, 1.0),
            box_top: box_top.clamp(0.0, 1.0),
            box_width: box_width.clamp(0.01, 1.0),
            box_height: box_height.clamp(0.01, 1.0),
            right_eye: [right_eye[0].clamp(0.0, 1.0), right_eye[1].clamp(0.0, 1.0)],
            left_eye: [left_eye[0].clamp(0.0, 1.0), left_eye[1].clamp(0.0, 1.0)],
            nose_tip: [nose_tip[0].clamp(0.0, 1.0), nose_tip[1].clamp(0.0, 1.0)],
            right_mouth: [
                right_mouth[0].clamp(0.0, 1.0),
                right_mouth[1].clamp(0.0, 1.0),
            ],
            left_mouth: [left_mouth[0].clamp(0.0, 1.0), left_mouth[1].clamp(0.0, 1.0)],
        }
    }

    /// Transforms this raw-source `[0..1]` face guide into oriented full-frame UV space `[0..1]`
    /// matching `grade_pass.wgsl` (`quarter_turns`, `flip_horizontal`, `flip_vertical`, `rotation_deg`).
    pub fn into_oriented_uv(
        self,
        crop: CropTransform,
        ori_width: u32,
        ori_height: u32,
        include_fine_rotation: bool,
    ) -> Self {
        let map_pt = |p: [f32; 2]| -> [f32; 2] {
            source_uv_to_oriented_uv(p[0], p[1], crop, ori_width, ori_height, include_fine_rotation)
        };

        let corners = [
            map_pt([self.box_left, self.box_top]),
            map_pt([self.box_left + self.box_width, self.box_top]),
            map_pt([self.box_left, self.box_top + self.box_height]),
            map_pt([
                self.box_left + self.box_width,
                self.box_top + self.box_height,
            ]),
        ];
        let mut min_u = f32::INFINITY;
        let mut min_v = f32::INFINITY;
        let mut max_u = f32::NEG_INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        for c in corners {
            min_u = min_u.min(c[0]);
            min_v = min_v.min(c[1]);
            max_u = max_u.max(c[0]);
            max_v = max_v.max(c[1]);
        }

        Self {
            box_left: min_u,
            box_top: min_v,
            box_width: (max_u - min_u).max(0.01),
            box_height: (max_v - min_v).max(0.01),
            right_eye: map_pt(self.right_eye),
            left_eye: map_pt(self.left_eye),
            nose_tip: map_pt(self.nose_tip),
            right_mouth: map_pt(self.right_mouth),
            left_mouth: map_pt(self.left_mouth),
        }
    }

    /// Convenience helper building from a `Point2D` landmark set in percentage `[0..100]` units.
    #[allow(clippy::too_many_arguments)]
    pub fn from_percentages(
        left_pct: f32,
        top_pct: f32,
        width_pct: f32,
        height_pct: f32,
        right_eye: Point2D,
        left_eye: Point2D,
        nose_tip: Point2D,
        right_mouth: Point2D,
        left_mouth: Point2D,
    ) -> Self {
        Self::from_normalized(
            left_pct / 100.0,
            top_pct / 100.0,
            width_pct / 100.0,
            height_pct / 100.0,
            [right_eye.x / 100.0, right_eye.y / 100.0],
            [left_eye.x / 100.0, left_eye.y / 100.0],
            [nose_tip.x / 100.0, nose_tip.y / 100.0],
            [right_mouth.x / 100.0, right_mouth.y / 100.0],
            [left_mouth.x / 100.0, left_mouth.y / 100.0],
        )
    }
}

/// Exact inverse of `grade_pass.wgsl::orientSourceUv` + forward straighten rotation.
pub fn source_uv_to_oriented_uv(
    su: f32,
    sv: f32,
    crop: CropTransform,
    ori_width: u32,
    ori_height: u32,
    include_fine_rotation: bool,
) -> [f32; 2] {
    let qt = crop.quarter_turns % 4;
    let (px, py) = match qt {
        1 => (1.0 - sv, su),
        2 => (1.0 - su, 1.0 - sv),
        3 => (sv, 1.0 - su),
        _ => (su, sv),
    };
    let mut u = if crop.flip_horizontal { 1.0 - px } else { px };
    let mut v = if crop.flip_vertical { 1.0 - py } else { py };

    if include_fine_rotation && crop.rotation_deg.abs() > 1e-3 {
        let ori_w = (ori_width as f32).max(1.0);
        let ori_h = (ori_height as f32).max(1.0);
        let cx = (ori_w - 1.0) * 0.5;
        let cy = (ori_h - 1.0) * 0.5;
        let dx = u * ori_w - 0.5 - cx;
        let dy = v * ori_h - 0.5 - cy;
        let rad = crop.rotation_deg.to_radians();
        let cos_a = rad.cos();
        let sin_a = rad.sin();
        let rx = dx * cos_a - dy * sin_a + cx;
        let ry = dx * sin_a + dy * cos_a + cy;
        u = (rx + 0.5) / ori_w;
        v = (ry + 0.5) / ori_h;
    }

    [u, v]
}

fn sanitize_coord(v: f32, fallback: f32) -> f32 {
    if v.is_finite() {
        v.clamp(-0.25, 1.25)
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heal_spot_clamps_bounds_and_preserves_donor_delta() {
        let mut spot = HealSpot {
            id: "s1".to_string(),
            mode: HealSpotMode::Heal,
            target_x: 0.42,
            target_y: 0.35,
            source_x: 0.48,
            source_y: 0.37,
            radius: 0.9,
            feather: -0.2,
            opacity: 2.5,
            points: vec![[0.42, 0.35], [f32::NAN, 0.4]],
        };
        spot.clamp_invariants();
        assert!((spot.radius - 0.20).abs() < 1e-6);
        assert!((spot.feather - 0.0).abs() < 1e-6);
        assert!((spot.opacity - 1.0).abs() < 1e-6);
        assert!((spot.points[1][0] - 0.42).abs() < 1e-6);
        let (du, dv) = spot.donor_delta();
        assert!((du - 0.06).abs() < 1e-5);
        assert!((dv - 0.02).abs() < 1e-5);
    }

    #[test]
    fn portrait_retouch_respects_enabled_bypass_and_defaults() {
        let mut p = PortraitRetouch::default();
        assert_eq!(p.skin_texture_preserve, 70);
        assert!(!p.is_active());

        p.skin_smoothing = 140;
        p.clamp_invariants();
        assert_eq!(p.skin_smoothing, 100);
        assert!(p.is_active());

        p.enabled = false;
        assert!(!p.is_active());
    }

    #[test]
    fn source_uv_to_oriented_uv_inverts_quarter_turns_and_flips_exactly() {
        for qt in 0..4u8 {
            for flip_h in [false, true] {
                for flip_v in [false, true] {
                    let crop = CropTransform {
                        quarter_turns: qt,
                        flip_horizontal: flip_h,
                        flip_vertical: flip_v,
                        ..CropTransform::default()
                    };
                    let u = 0.23_f32;
                    let v = 0.71_f32;
                    // Forward shader mapping (oriented -> source)
                    let px = if flip_h { 1.0 - u } else { u };
                    let py = if flip_v { 1.0 - v } else { v };
                    let (su, sv) = match qt {
                        1 => (py, 1.0 - px),
                        2 => (1.0 - px, 1.0 - py),
                        3 => (1.0 - py, px),
                        _ => (px, py),
                    };
                    let [ru, rv] = source_uv_to_oriented_uv(su, sv, crop, 1000, 1500, false);
                    assert!(
                        (ru - u).abs() < 1e-5 && (rv - v).abs() < 1e-5,
                        "mismatch for qt={qt} fh={flip_h} fv={flip_v}: got ({ru},{rv}) expected ({u},{v})"
                    );
                }
            }
        }
    }
}
