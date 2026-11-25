//! 32-bit float parallel Spot Heal / Clone & Frequency-Separation Portrait Retouching engine.
//!
//! Mirrors `shaders/develop/heal_pass.wgsl` and `shaders/develop/portrait_retouch_pass.wgsl`
//! formula-for-formula so Rust Rayon CPU fallback and Rust `wgpu` / WebGPU GPU passes maintain
//! `< 1/255` pixel parity across live preview and full-resolution batch export.

use std::f32::consts::FRAC_1_SQRT_2;

use bytemuck::{Pod, Zeroable};
use image::Rgb32FImage;
use kroma_domain::develop::DevelopRecipe;
use kroma_domain::retouch::{
    FaceParseMap, HealSpotMode, PortraitFaceGuide, PortraitRetouch, MAX_HEAL_SPOTS,
    MAX_INPAINT_PATCHES, MAX_PORTRAIT_FACE_GUIDES,
};
use rayon::prelude::*;

use crate::vision::lama_inpainter::decode_inpaint_patch_rgba;

const BORDER_TAPS: [[f32; 2]; 8] = [
    [0.02, 0.02],
    [0.50, 0.02],
    [0.98, 0.02],
    [0.98, 0.50],
    [0.98, 0.98],
    [0.50, 0.98],
    [0.02, 0.98],
    [0.02, 0.50],
];

const RING8: [[f32; 2]; 8] = [
    [1.0, 0.0],
    [FRAC_1_SQRT_2, FRAC_1_SQRT_2],
    [0.0, 1.0],
    [-FRAC_1_SQRT_2, FRAC_1_SQRT_2],
    [-1.0, 0.0],
    [-FRAC_1_SQRT_2, -FRAC_1_SQRT_2],
    [0.0, -1.0],
    [FRAC_1_SQRT_2, -FRAC_1_SQRT_2],
];

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct InpaintUniforms {
    pub dims: [f32; 4],
    pub crop_rect: [f32; 4],
    pub bounds: [f32; 4],
    pub params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct HealUniforms {
    pub dims: [f32; 4],
    pub crop_rect: [f32; 4],
    pub meta: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct HealSpotGpu {
    pub geom: [f32; 4],
    pub params: [f32; 4],
    pub bounds: [f32; 4],
    pub center: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PortraitFaceGpu {
    pub box_rect: [f32; 4],
    pub eyes: [f32; 4],
    pub nose: [f32; 4],
    pub mouth: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PortraitRetouchUniforms {
    pub dims: [f32; 4],
    pub crop_rect: [f32; 4],
    pub meta: [f32; 4],
    pub skin_params: [f32; 4],
    pub feature_params: [f32; 4],
    pub faces: [PortraitFaceGpu; 4],
}

/// Compiles `DevelopRecipe::retouch.heal_spots` into GPU storage buffers matching `heal_pass.wgsl`.
pub fn pack_heal_spots_for_gpu(
    recipe: &DevelopRecipe,
    ori_w: u32,
    ori_h: u32,
) -> (Vec<HealSpotGpu>, Vec<[f32; 4]>) {
    let ow = (ori_w as f32).max(1.0);
    let oh = (ori_h as f32).max(1.0);
    let min_side = ow.min(oh);

    let mut spots_gpu = Vec::new();
    let mut segs_gpu = Vec::new();

    for spot in recipe
        .retouch
        .heal_spots
        .iter()
        .filter(|s| s.is_effective())
        .take(MAX_HEAL_SPOTS)
    {
        let seg_start = segs_gpu.len() as u32;
        let mut min_u = f32::INFINITY;
        let mut min_v = f32::INFINITY;
        let mut max_u = f32::NEG_INFINITY;
        let mut max_v = f32::NEG_INFINITY;

        if spot.points.len() <= 1 {
            let p = spot
                .points
                .first()
                .copied()
                .unwrap_or([spot.target_x, spot.target_y]);
            segs_gpu.push([p[0], p[1], p[0], p[1]]);
            min_u = p[0];
            max_u = p[0];
            min_v = p[1];
            max_v = p[1];
        } else {
            for win in spot.points.windows(2) {
                let a = win[0];
                let b = win[1];
                segs_gpu.push([a[0], a[1], b[0], b[1]]);
                min_u = min_u.min(a[0]).min(b[0]);
                min_v = min_v.min(a[1]).min(b[1]);
                max_u = max_u.max(a[0]).max(b[0]);
                max_v = max_v.max(a[1]).max(b[1]);
            }
        }

        let seg_count = (segs_gpu.len() as u32).saturating_sub(seg_start).max(1);
        let margin_u = (spot.radius * min_side) / ow + 2.0 / ow;
        let margin_v = (spot.radius * min_side) / oh + 2.0 / oh;
        let (du, dv) = spot.donor_delta();

        spots_gpu.push(HealSpotGpu {
            geom: [du, dv, spot.radius, spot.feather],
            params: [
                spot.opacity,
                if spot.mode == HealSpotMode::Clone {
                    1.0
                } else {
                    0.0
                },
                seg_start as f32,
                seg_count as f32,
            ],
            bounds: [
                min_u - margin_u,
                min_v - margin_v,
                max_u + margin_u,
                max_v + margin_v,
            ],
            center: [spot.target_x, spot.target_y, 0.0, 0.0],
        });
    }

    (spots_gpu, segs_gpu)
}

pub fn build_heal_uniforms(
    out_w: u32,
    out_h: u32,
    ori_w: u32,
    ori_h: u32,
    crop_rect: [f32; 4],
    spot_count: u32,
) -> HealUniforms {
    let spatial_scale = ((ori_w.max(ori_h) as f32) / 1024.0).max(1.0);
    HealUniforms {
        dims: [out_w as f32, out_h as f32, ori_w as f32, ori_h as f32],
        crop_rect,
        meta: [spot_count as f32, spatial_scale, 0.0, 0.0],
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_portrait_retouch_uniforms(
    out_w: u32,
    out_h: u32,
    ori_w: u32,
    ori_h: u32,
    crop_rect: [f32; 4],
    portrait: PortraitRetouch,
    oriented_face_guides: &[PortraitFaceGuide],
    has_face_parse_map: bool,
) -> PortraitRetouchUniforms {
    let ow = (ori_w as f32).max(1.0);
    let oh = (ori_h as f32).max(1.0);
    let spatial_scale = (ow.max(oh) / 1024.0).clamp(0.75, 4.0);
    let face_count = oriented_face_guides.len().min(MAX_PORTRAIT_FACE_GUIDES);

    let mut faces = [PortraitFaceGpu::zeroed(); 4];
    for (dst, src) in faces.iter_mut().zip(oriented_face_guides.iter().take(4)) {
        *dst = PortraitFaceGpu {
            box_rect: [src.box_left, src.box_top, src.box_width, src.box_height],
            eyes: [
                src.right_eye[0],
                src.right_eye[1],
                src.left_eye[0],
                src.left_eye[1],
            ],
            nose: [src.nose_tip[0], src.nose_tip[1], 0.0, 0.0],
            mouth: [
                src.right_mouth[0],
                src.right_mouth[1],
                src.left_mouth[0],
                src.left_mouth[1],
            ],
        };
    }

    PortraitRetouchUniforms {
        dims: [out_w as f32, out_h as f32, ow, oh],
        crop_rect,
        meta: [
            spatial_scale,
            face_count as f32,
            ow / oh,
            if has_face_parse_map { 1.0 } else { 0.0 },
        ],
        skin_params: [
            portrait.skin_smoothing as f32 / 100.0,
            portrait.skin_texture_preserve as f32 / 100.0,
            portrait.skin_tone_evenness as f32 / 100.0,
            portrait.shine_removal as f32 / 100.0,
        ],
        feature_params: [
            portrait.under_eye_brighten as f32 / 100.0,
            portrait.eye_enhance as f32 / 100.0,
            portrait.teeth_whiten as f32 / 100.0,
            portrait.lip_enhance as f32 / 100.0,
        ],
        faces,
    }
}

#[inline]
fn k_smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn luma709(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

#[inline]
fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

#[inline]
fn sample_bilinear_px(src: &[f32], w: usize, h: usize, fx: f32, fy: f32) -> [f32; 3] {
    let max_x = (w.saturating_sub(1)) as f32;
    let max_y = (h.saturating_sub(1)) as f32;
    let cx = fx.clamp(0.0, max_x);
    let cy = fy.clamp(0.0, max_y);
    let x0 = cx.floor() as usize;
    let y0 = cy.floor() as usize;
    let x1 = (x0 + 1).min(w.saturating_sub(1));
    let y1 = (y0 + 1).min(h.saturating_sub(1));
    let tx = cx - x0 as f32;
    let ty = cy - y0 as f32;

    let read = |x: usize, y: usize| -> [f32; 3] {
        let i = (y * w + x) * 3;
        [src[i], src[i + 1], src[i + 2]]
    };
    let p00 = read(x0, y0);
    let p10 = read(x1, y0);
    let p01 = read(x0, y1);
    let p11 = read(x1, y1);
    mix3(mix3(p00, p10, tx), mix3(p01, p11, tx), ty)
}

#[inline]
fn sample_bilinear_uv(
    src: &[f32],
    w: usize,
    h: usize,
    crop_rect: [f32; 4],
    uv: [f32; 2],
) -> [f32; 3] {
    let fx = ((uv[0] - crop_rect[0]) / crop_rect[2].max(1e-6)) * (w as f32) - 0.5;
    let fy = ((uv[1] - crop_rect[1]) / crop_rect[3].max(1e-6)) * (h as f32) - 0.5;
    sample_bilinear_px(src, w, h, fx, fy)
}

#[inline]
fn point_to_segment_info(
    uv: [f32; 2],
    a: [f32; 2],
    b: [f32; 2],
    scale_uv: [f32; 2],
) -> (f32, [f32; 2]) {
    let ps = [uv[0] * scale_uv[0], uv[1] * scale_uv[1]];
    let a_s = [a[0] * scale_uv[0], a[1] * scale_uv[1]];
    let b_s = [b[0] * scale_uv[0], b[1] * scale_uv[1]];
    let ab = [b_s[0] - a_s[0], b_s[1] - a_s[1]];
    let len_sq = ab[0] * ab[0] + ab[1] * ab[1];
    let t = if len_sq > 1e-10 {
        (((ps[0] - a_s[0]) * ab[0] + (ps[1] - a_s[1]) * ab[1]) / len_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let q = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
    let dx = (uv[0] - q[0]) * scale_uv[0];
    let dy = (uv[1] - q[1]) * scale_uv[1];
    ((dx * dx + dy * dy).sqrt(), q)
}

fn min_dist_to_stroke(
    uv: [f32; 2],
    segs: &[[f32; 4]],
    seg_start: usize,
    seg_count: usize,
    scale_uv: [f32; 2],
) -> f32 {
    let mut best = 1e6_f32;
    for k in 0..seg_count {
        let seg = segs[seg_start + k];
        let (d, _) = point_to_segment_info(uv, [seg[0], seg[1]], [seg[2], seg[3]], scale_uv);
        best = best.min(d);
    }
    best
}

/// Applies non-destructive Spot Heal & Clone strokes on a 32-bit float image in-place (CPU parity
/// mirror of `shaders/develop/heal_pass.wgsl`).
pub fn apply_heal_spots_f32(
    working: &mut Rgb32FImage,
    recipe: &DevelopRecipe,
    ori_w: u32,
    ori_h: u32,
    crop_rect: [f32; 4],
) {
    if !recipe.retouch.has_heal_spots() {
        return;
    }
    let (spots_gpu, segs_gpu) = pack_heal_spots_for_gpu(recipe, ori_w, ori_h);
    if spots_gpu.is_empty() {
        return;
    }

    let w = working.width() as usize;
    let h = working.height() as usize;
    if w == 0 || h == 0 {
        return;
    }

    let snapshot = working.as_raw().to_vec();
    let ow = (ori_w as f32).max(1.0);
    let oh = (ori_h as f32).max(1.0);
    let min_side = ow.min(oh);
    let scale_uv = [ow / min_side, oh / min_side];
    let inv_scale_uv = [min_side / ow, min_side / oh];
    let row_stride = w * 3;

    working
        .as_mut()
        .par_chunks_exact_mut(row_stride)
        .enumerate()
        .for_each(|(y, row)| {
            let v_crop = (y as f32 + 0.5) / h as f32;
            let uv_y = crop_rect[1] + v_crop * crop_rect[3];
            for x in 0..w {
                let u_crop = (x as f32 + 0.5) / w as f32;
                let uv = [crop_rect[0] + u_crop * crop_rect[2], uv_y];
                let idx = x * 3;
                let mut rgb = [snapshot[y * row_stride + idx], snapshot[y * row_stride + idx + 1], snapshot[y * row_stride + idx + 2]];

                for spot in &spots_gpu {
                    if uv[0] < spot.bounds[0]
                        || uv[0] > spot.bounds[2]
                        || uv[1] < spot.bounds[1]
                        || uv[1] > spot.bounds[3]
                    {
                        continue;
                    }

                    let radius = spot.geom[2];
                    let feather = spot.geom[3];
                    let opacity = spot.params[0];
                    let is_clone = spot.params[1] > 0.5;
                    let seg_start = spot.params[2] as usize;
                    let seg_count = (spot.params[3] as usize).max(1);

                    let mut best_dist = 1e6_f32;
                    let mut closest_uv = [spot.center[0], spot.center[1]];
                    for k in 0..seg_count {
                        let seg = segs_gpu[seg_start + k];
                        let (d, q) = point_to_segment_info(
                            uv,
                            [seg[0], seg[1]],
                            [seg[2], seg[3]],
                            scale_uv,
                        );
                        if d < best_dist {
                            best_dist = d;
                            closest_uv = q;
                        }
                    }

                    if best_dist >= radius {
                        continue;
                    }

                    let mask = if feather > 1e-4 {
                        let inner_r = radius * (1.0 - feather * 0.85);
                        1.0 - k_smoothstep(inner_r, radius, best_dist)
                    } else {
                        1.0
                    };
                    let alpha = (mask * opacity).clamp(0.0, 1.0);
                    if alpha <= 1e-5 {
                        continue;
                    }

                    let delta = [spot.geom[0], spot.geom[1]];
                    let donor_rgb = sample_bilinear_uv(
                        &snapshot,
                        w,
                        h,
                        crop_rect,
                        [uv[0] + delta[0], uv[1] + delta[1]],
                    );

                    let patch_rgb = if is_clone {
                        donor_rgb
                    } else {
                        let ring_r = radius * 1.28;
                        let reg_eps = 0.18 * radius * radius;
                        let mut shift_sum = [0.0_f32; 3];
                        let mut w_sum = 0.0_f32;

                        for dir in RING8 {
                            let ring_dst_uv = [
                                closest_uv[0] + dir[0] * ring_r * inv_scale_uv[0],
                                closest_uv[1] + dir[1] * ring_r * inv_scale_uv[1],
                            ];
                            let outside_w = if seg_count > 1 {
                                let sd = min_dist_to_stroke(
                                    ring_dst_uv,
                                    &segs_gpu,
                                    seg_start,
                                    seg_count,
                                    scale_uv,
                                );
                                k_smoothstep(radius * 0.82, radius * 1.12, sd)
                            } else {
                                1.0
                            };
                            if outside_w <= 1e-4 {
                                continue;
                            }
                            let ring_src_uv = [ring_dst_uv[0] + delta[0], ring_dst_uv[1] + delta[1]];
                            let dst_col = sample_bilinear_uv(&snapshot, w, h, crop_rect, ring_dst_uv);
                            let src_col = sample_bilinear_uv(&snapshot, w, h, crop_rect, ring_src_uv);
                            let pdx = (uv[0] - ring_dst_uv[0]) * scale_uv[0];
                            let pdy = (uv[1] - ring_dst_uv[1]) * scale_uv[1];
                            let inv_w = outside_w / (pdx * pdx + pdy * pdy + reg_eps);
                            shift_sum[0] += (dst_col[0] - src_col[0]) * inv_w;
                            shift_sum[1] += (dst_col[1] - src_col[1]) * inv_w;
                            shift_sum[2] += (dst_col[2] - src_col[2]) * inv_w;
                            w_sum += inv_w;
                        }

                        if w_sum > 1e-6 {
                            let s0 = (shift_sum[0] / w_sum).clamp(-0.35, 0.35);
                            let s1 = (shift_sum[1] / w_sum).clamp(-0.35, 0.35);
                            let s2 = (shift_sum[2] / w_sum).clamp(-0.35, 0.35);
                            [
                                (donor_rgb[0] + s0).clamp(0.0, 1.0),
                                (donor_rgb[1] + s1).clamp(0.0, 1.0),
                                (donor_rgb[2] + s2).clamp(0.0, 1.0),
                            ]
                        } else {
                            donor_rgb
                        }
                    };

                    rgb = mix3(rgb, patch_rgb, alpha);
                }

                row[idx] = rgb[0].clamp(0.0, 1.0);
                row[idx + 1] = rgb[1].clamp(0.0, 1.0);
                row[idx + 2] = rgb[2].clamp(0.0, 1.0);
            }
        });
}

#[inline]
fn ycbcr_skin_probability(rgb: [f32; 3]) -> f32 {
    let y = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
    let cb = 0.5 - 0.168736 * rgb[0] - 0.331264 * rgb[1] + 0.5 * rgb[2];
    let cr = 0.5 + 0.5 * rgb[0] - 0.418688 * rgb[1] - 0.081312 * rgb[2];

    let y_gate = k_smoothstep(0.06, 0.16, y) * (1.0 - k_smoothstep(0.95, 0.995, y));
    let cb_gate = k_smoothstep(0.24, 0.32, cb) * (1.0 - k_smoothstep(0.49, 0.55, cb));
    let cr_gate = k_smoothstep(0.505, 0.54, cr) * (1.0 - k_smoothstep(0.70, 0.76, cr));
    let warmth_gate = k_smoothstep(-0.015, 0.035, rgb[0] - rgb[2]);
    (y_gate * cb_gate * cr_gate * warmth_gate).clamp(0.0, 1.0)
}

#[inline]
fn rotated_ellipse_mask(
    p: [f32; 2],
    center: [f32; 2],
    ax: [f32; 2],
    ay: [f32; 2],
    rx: f32,
    ry: f32,
    inner_ratio: f32,
) -> f32 {
    let dx = p[0] - center[0];
    let dy = p[1] - center[1];
    let lx = (dx * ax[0] + dy * ax[1]) / rx.max(1e-5);
    let ly = (dx * ay[0] + dy * ay[1]) / ry.max(1e-5);
    let r = (lx * lx + ly * ly).sqrt();
    1.0 - k_smoothstep(inner_ratio, 1.0, r)
}

#[inline]
fn sample_rgba8_bilinear(rgba: &[u8], w: usize, h: usize, u: f32, v: f32) -> [f32; 4] {
    let max_x = (w.saturating_sub(1)) as f32;
    let max_y = (h.saturating_sub(1)) as f32;
    let fx = (u * w as f32 - 0.5).clamp(0.0, max_x);
    let fy = (v * h as f32 - 0.5).clamp(0.0, max_y);
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(w.saturating_sub(1));
    let y1 = (y0 + 1).min(h.saturating_sub(1));
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;

    let load = |x: usize, y: usize| -> [f32; 4] {
        let i = (y * w + x) * 4;
        [
            rgba[i] as f32 / 255.0,
            rgba[i + 1] as f32 / 255.0,
            rgba[i + 2] as f32 / 255.0,
            rgba[i + 3] as f32 / 255.0,
        ]
    };
    let p00 = load(x0, y0);
    let p10 = load(x1, y0);
    let p01 = load(x0, y1);
    let p11 = load(x1, y1);
    let mut out = [0.0_f32; 4];
    for c in 0..4 {
        let top = p00[c] + (p10[c] - p00[c]) * tx;
        let bot = p01[c] + (p11[c] - p01[c]) * tx;
        out[c] = top + (bot - top) * ty;
    }
    out
}

/// Composites non-destructive LaMa / Harmonic `InpaintPatch` entries sequentially on a 32-bit float
/// image in-place (CPU parity mirror of `shaders/develop/inpaint_pass.wgsl`).
pub fn apply_inpaint_patches_f32(
    working: &mut Rgb32FImage,
    recipe: &DevelopRecipe,
    crop_rect: [f32; 4],
) {
    if !recipe.retouch.has_inpaint_patches() {
        return;
    }
    let w = working.width() as usize;
    let h = working.height() as usize;
    if w == 0 || h == 0 {
        return;
    }

    for patch in recipe
        .retouch
        .inpaint_patches
        .iter()
        .filter(|p| p.is_effective())
        .take(MAX_INPAINT_PATCHES)
    {
        let Some((pw, ph, rgba)) = decode_inpaint_patch_rgba(patch) else {
            continue;
        };
        let pw_u = pw as usize;
        let ph_u = ph as usize;
        let bounds = [
            patch.bounds.left,
            patch.bounds.top,
            patch.bounds.width.max(1e-6),
            patch.bounds.height.max(1e-6),
        ];
        let opacity = patch.opacity.clamp(0.05, 1.0);
        let snapshot = working.as_raw().to_vec();
        let row_stride = w * 3;

        working
            .as_mut()
            .par_chunks_exact_mut(row_stride)
            .enumerate()
            .for_each(|(y, row)| {
                let v_crop = (y as f32 + 0.5) / h as f32;
                let uv_y = crop_rect[1] + v_crop * crop_rect[3];
                let lv = (uv_y - bounds[1]) / bounds[3];
                if !(0.0..=1.0).contains(&lv) {
                    return;
                }
                for x in 0..w {
                    let u_crop = (x as f32 + 0.5) / w as f32;
                    let uv_x = crop_rect[0] + u_crop * crop_rect[2];
                    let lu = (uv_x - bounds[0]) / bounds[2];
                    if !(0.0..=1.0).contains(&lu) {
                        continue;
                    }
                    let ps = sample_rgba8_bilinear(&rgba, pw_u, ph_u, lu, lv);
                    let alpha = (ps[3] * opacity).clamp(0.0, 1.0);
                    if alpha <= 1e-5 {
                        continue;
                    }

                    let mut shift_sum = [0.0_f32; 3];
                    let mut w_sum = 0.0_f32;
                    for tap in BORDER_TAPS {
                        let p_tap = sample_rgba8_bilinear(&rgba, pw_u, ph_u, tap[0], tap[1]);
                        let border_w = 1.0 - (p_tap[3] * 4.0).clamp(0.0, 1.0);
                        if border_w > 0.05 {
                            let tap_u = bounds[0] + tap[0] * bounds[2];
                            let tap_v = bounds[1] + tap[1] * bounds[3];
                            let live = sample_bilinear_uv(&snapshot, w, h, crop_rect, [tap_u, tap_v]);
                            let dx = lu - tap[0];
                            let dy = lv - tap[1];
                            let inv_w = border_w / (dx * dx + dy * dy + 0.12);
                            shift_sum[0] += (live[0] - p_tap[0]) * inv_w;
                            shift_sum[1] += (live[1] - p_tap[1]) * inv_w;
                            shift_sum[2] += (live[2] - p_tap[2]) * inv_w;
                            w_sum += inv_w;
                        }
                    }

                    let mut adapted = [ps[0], ps[1], ps[2]];
                    if w_sum > 1e-5 {
                        adapted[0] = (adapted[0] + (shift_sum[0] / w_sum).clamp(-0.30, 0.30)).clamp(0.0, 1.0);
                        adapted[1] = (adapted[1] + (shift_sum[1] / w_sum).clamp(-0.30, 0.30)).clamp(0.0, 1.0);
                        adapted[2] = (adapted[2] + (shift_sum[2] / w_sum).clamp(-0.30, 0.30)).clamp(0.0, 1.0);
                    }

                    let idx = x * 3;
                    let src_rgb = [row[idx], row[idx + 1], row[idx + 2]];
                    let out_rgb = mix3(src_rgb, adapted, alpha);
                    row[idx] = out_rgb[0].clamp(0.0, 1.0);
                    row[idx + 1] = out_rgb[1].clamp(0.0, 1.0);
                    row[idx + 2] = out_rgb[2].clamp(0.0, 1.0);
                }
            });
    }
}

/// Applies 3-band Frequency-Separation skin retouching + YuNet landmark-guided portrait
/// enhancements on a 32-bit float image in-place (CPU parity mirror of `portrait_retouch_pass.wgsl`).
pub fn apply_portrait_retouch_f32(
    working: &mut Rgb32FImage,
    recipe: &DevelopRecipe,
    ori_w: u32,
    ori_h: u32,
    crop_rect: [f32; 4],
    oriented_face_guides: &[PortraitFaceGuide],
) {
    apply_portrait_retouch_with_parse_f32(
        working,
        recipe,
        ori_w,
        ori_h,
        crop_rect,
        oriented_face_guides,
        None,
    );
}

/// Applies 3-band Frequency-Separation skin retouching + optional BiSeNet 19-class neural
/// `FaceParseMap` + YuNet landmark-guided portrait enhancements in-place.
pub fn apply_portrait_retouch_with_parse_f32(
    working: &mut Rgb32FImage,
    recipe: &DevelopRecipe,
    ori_w: u32,
    ori_h: u32,
    crop_rect: [f32; 4],
    oriented_face_guides: &[PortraitFaceGuide],
    face_parse_map: Option<&FaceParseMap>,
) {
    if !recipe.retouch.is_portrait_active() {
        return;
    }

    let w = working.width() as usize;
    let h = working.height() as usize;
    if w == 0 || h == 0 {
        return;
    }

    let valid_parse = face_parse_map.filter(|m| m.is_valid());
    let has_face_parse = valid_parse.is_some();

    let u = build_portrait_retouch_uniforms(
        w as u32,
        h as u32,
        ori_w,
        ori_h,
        crop_rect,
        recipe.retouch.portrait,
        oriented_face_guides,
        has_face_parse,
    );

    let skin_smooth = u.skin_params[0];
    let skin_tex_preserve = u.skin_params[1];
    let skin_even = u.skin_params[2];
    let shine_amt = u.skin_params[3];

    let under_eye_amt = u.feature_params[0];
    let eye_amt = u.feature_params[1];
    let teeth_amt = u.feature_params[2];
    let lip_amt = u.feature_params[3];

    let any_skin = skin_smooth > 1e-4 || skin_even > 1e-4 || shine_amt > 1e-4;
    let any_feature = under_eye_amt > 1e-4 || eye_amt > 1e-4 || teeth_amt > 1e-4 || lip_amt > 1e-4;
    if !any_skin && !any_feature {
        return;
    }

    let snapshot = working.as_raw().to_vec();
    let ow = u.dims[2];
    let oh = u.dims[3];
    let min_side = ow.min(oh);
    let scale_uv = [ow / min_side, oh / min_side];
    let scale = u.meta[0];
    let face_count = (u.meta[1] as usize).min(4);
    let row_stride = w * 3;

    working
        .as_mut()
        .par_chunks_exact_mut(row_stride)
        .enumerate()
        .for_each(|(y, row)| {
            let fy = y as f32;
            let v_crop = (fy + 0.5) / h as f32;
            let uv_y = crop_rect[1] + v_crop * crop_rect[3];

            for x in 0..w {
                let fx = x as f32;
                let u_crop = (fx + 0.5) / w as f32;
                let uv_x = crop_rect[0] + u_crop * crop_rect[2];
                let ps = [uv_x * scale_uv[0], uv_y * scale_uv[1]];
                let idx = x * 3;
                let base_idx = y * row_stride + idx;
                let mut rgb = [snapshot[base_idx], snapshot[base_idx + 1], snapshot[base_idx + 2]];

                let mut face_env = 0.0_f32;
                let mut feature_protect = 0.0_f32;
                let mut under_eye_mask = 0.0_f32;
                let mut eye_mask = 0.0_f32;
                let mut mouth_mask = 0.0_f32;

                for face in &u.faces[..face_count] {
                    let r_eye = [face.eyes[0] * scale_uv[0], face.eyes[1] * scale_uv[1]];
                    let l_eye = [face.eyes[2] * scale_uv[0], face.eyes[3] * scale_uv[1]];
                    let nose = [face.nose[0] * scale_uv[0], face.nose[1] * scale_uv[1]];
                    let r_mouth = [face.mouth[0] * scale_uv[0], face.mouth[1] * scale_uv[1]];
                    let l_mouth = [face.mouth[2] * scale_uv[0], face.mouth[3] * scale_uv[1]];

                    let e_vec = [l_eye[0] - r_eye[0], l_eye[1] - r_eye[1]];
                    let e_len = (e_vec[0] * e_vec[0] + e_vec[1] * e_vec[1]).sqrt();
                    let eye_dist = e_len.max(0.02);
                    let ax = if e_len > 1e-4 {
                        [e_vec[0] / e_len, e_vec[1] / e_len]
                    } else {
                        [1.0, 0.0]
                    };
                    let mut ay = [-ax[1], ax[0]];
                    let eye_mid = [0.5 * (r_eye[0] + l_eye[0]), 0.5 * (r_eye[1] + l_eye[1])];
                    let mouth_mid = [0.5 * (r_mouth[0] + l_mouth[0]), 0.5 * (r_mouth[1] + l_mouth[1])];
                    if (mouth_mid[0] - eye_mid[0]) * ay[0] + (mouth_mid[1] - eye_mid[1]) * ay[1] < 0.0
                    {
                        ay = [-ay[0], -ay[1]];
                    }

                    let face_center = [
                        eye_mid[0] + ay[0] * (0.28 * eye_dist),
                        eye_mid[1] + ay[1] * (0.28 * eye_dist),
                    ];
                    let f_head = rotated_ellipse_mask(
                        ps,
                        face_center,
                        ax,
                        ay,
                        1.32 * eye_dist,
                        1.72 * eye_dist,
                        0.52,
                    );
                    let neck_center = [
                        eye_mid[0] + ay[0] * (1.65 * eye_dist),
                        eye_mid[1] + ay[1] * (1.65 * eye_dist),
                    ];
                    let f_neck = rotated_ellipse_mask(
                        ps,
                        neck_center,
                        ax,
                        ay,
                        1.02 * eye_dist,
                        1.05 * eye_dist,
                        0.35,
                    );
                    face_env = face_env.max(f_head.max(f_neck * 0.85));

                    let r_eye_prot = rotated_ellipse_mask(
                        ps,
                        r_eye,
                        ax,
                        ay,
                        0.28 * eye_dist,
                        0.16 * eye_dist,
                        0.45,
                    );
                    let l_eye_prot = rotated_ellipse_mask(
                        ps,
                        l_eye,
                        ax,
                        ay,
                        0.28 * eye_dist,
                        0.16 * eye_dist,
                        0.45,
                    );
                    let r_brow_prot = rotated_ellipse_mask(
                        ps,
                        [
                            r_eye[0] - ay[0] * (0.24 * eye_dist),
                            r_eye[1] - ay[1] * (0.24 * eye_dist),
                        ],
                        ax,
                        ay,
                        0.32 * eye_dist,
                        0.12 * eye_dist,
                        0.4,
                    );
                    let l_brow_prot = rotated_ellipse_mask(
                        ps,
                        [
                            l_eye[0] - ay[0] * (0.24 * eye_dist),
                            l_eye[1] - ay[1] * (0.24 * eye_dist),
                        ],
                        ax,
                        ay,
                        0.32 * eye_dist,
                        0.12 * eye_dist,
                        0.4,
                    );
                    let nose_prot = rotated_ellipse_mask(
                        ps,
                        nose,
                        ax,
                        ay,
                        0.15 * eye_dist,
                        0.11 * eye_dist,
                        0.4,
                    );

                    let m_vec = [l_mouth[0] - r_mouth[0], l_mouth[1] - r_mouth[1]];
                    let m_len = (m_vec[0] * m_vec[0] + m_vec[1] * m_vec[1]).sqrt();
                    let mouth_w = m_len.max(eye_dist * 0.55);
                    let m_ax = if m_len > 1e-4 {
                        [m_vec[0] / m_len, m_vec[1] / m_len]
                    } else {
                        ax
                    };
                    let mut m_ay = [-m_ax[1], m_ax[0]];
                    if m_ay[0] * ay[0] + m_ay[1] * ay[1] < 0.0 {
                        m_ay = [-m_ay[0], -m_ay[1]];
                    }
                    let m_prot = rotated_ellipse_mask(
                        ps,
                        mouth_mid,
                        m_ax,
                        m_ay,
                        0.58 * mouth_w,
                        0.25 * mouth_w,
                        0.45,
                    );
                    feature_protect = feature_protect.max(
                        r_eye_prot
                            .max(l_eye_prot)
                            .max(r_brow_prot)
                            .max(l_brow_prot)
                            .max(nose_prot)
                            .max(m_prot),
                    );

                    let r_eye_m = rotated_ellipse_mask(
                        ps,
                        r_eye,
                        ax,
                        ay,
                        0.24 * eye_dist,
                        0.12 * eye_dist,
                        0.35,
                    );
                    let l_eye_m = rotated_ellipse_mask(
                        ps,
                        l_eye,
                        ax,
                        ay,
                        0.24 * eye_dist,
                        0.12 * eye_dist,
                        0.35,
                    );
                    eye_mask = eye_mask.max(r_eye_m.max(l_eye_m));

                    let r_ue = rotated_ellipse_mask(
                        ps,
                        [
                            r_eye[0] + ay[0] * (0.22 * eye_dist),
                            r_eye[1] + ay[1] * (0.22 * eye_dist),
                        ],
                        ax,
                        ay,
                        0.36 * eye_dist,
                        0.14 * eye_dist,
                        0.25,
                    ) * (1.0 - r_eye_prot);
                    let l_ue = rotated_ellipse_mask(
                        ps,
                        [
                            l_eye[0] + ay[0] * (0.22 * eye_dist),
                            l_eye[1] + ay[1] * (0.22 * eye_dist),
                        ],
                        ax,
                        ay,
                        0.36 * eye_dist,
                        0.14 * eye_dist,
                        0.25,
                    ) * (1.0 - l_eye_prot);
                    under_eye_mask = under_eye_mask.max(r_ue.max(l_ue));

                    let m_mask = rotated_ellipse_mask(
                        ps,
                        mouth_mid,
                        m_ax,
                        m_ay,
                        0.50 * mouth_w,
                        0.22 * mouth_w,
                        0.35,
                    );
                    mouth_mask = mouth_mask.max(m_mask);
                }

                let skin_prob = ycbcr_skin_probability(rgb);
                let mut skin_mask = if face_count > 0 {
                    skin_prob * face_env * (1.0 - feature_protect * 0.92)
                } else {
                    skin_prob * 0.85
                };
                let mut teeth_mask = mouth_mask;
                let mut lip_mask = mouth_mask;

                if let Some(fp_map) = valid_parse {
                    let fp = sample_rgba8_bilinear(
                        &fp_map.rgba,
                        fp_map.width as usize,
                        fp_map.height as usize,
                        uv_x,
                        uv_y,
                    );
                    skin_mask = fp[0].clamp(0.0, 1.0);
                    eye_mask = fp[1].clamp(0.0, 1.0);
                    teeth_mask = fp[2].clamp(0.0, 1.0);
                    lip_mask = fp[3].clamp(0.0, 1.0);
                    under_eye_mask =
                        under_eye_mask * (fp[0] * 1.15).clamp(0.0, 1.0) * (1.0 - eye_mask);
                }

                let need_spatial_skin = (any_skin && skin_mask > 0.005)
                    || (under_eye_amt > 1e-4 && under_eye_mask > 0.005);
                let need_fine_detail = need_spatial_skin
                    || (eye_amt > 1e-4 && eye_mask > 0.005)
                    || (lip_amt > 1e-4 && lip_mask > 0.005);

                if !need_fine_detail && !(teeth_amt > 1e-4 && teeth_mask > 0.005) {
                    continue;
                }

                let fine_low = if need_fine_detail {
                    let r_fine = 1.15 * scale;
                    let f1 = sample_bilinear_px(&snapshot, w, h, fx - r_fine, fy);
                    let f2 = sample_bilinear_px(&snapshot, w, h, fx + r_fine, fy);
                    let f3 = sample_bilinear_px(&snapshot, w, h, fx, fy - r_fine);
                    let f4 = sample_bilinear_px(&snapshot, w, h, fx, fy + r_fine);
                    [
                        (rgb[0] * 2.0 + f1[0] + f2[0] + f3[0] + f4[0]) / 6.0,
                        (rgb[1] * 2.0 + f1[1] + f2[1] + f3[1] + f4[1]) / 6.0,
                        (rgb[2] * 2.0 + f1[2] + f2[2] + f3[2] + f4[2]) / 6.0,
                    ]
                } else {
                    rgb
                };
                let high_freq_pores = [
                    rgb[0] - fine_low[0],
                    rgb[1] - fine_low[1],
                    rgb[2] - fine_low[2],
                ];

                let low_freq = if need_spatial_skin {
                    let r1 = 2.6 * scale;
                    let r2 = 5.6 * scale;
                    let sigma_color = 0.095 + 0.085 * skin_smooth.max(skin_even.max(under_eye_amt));
                    let inv2_sig2 = -0.5 / (sigma_color * sigma_color);

                    let mut acc = [rgb[0] * 1.4, rgb[1] * 1.4, rgb[2] * 1.4];
                    let mut w_sum = 1.4_f32;

                    for dir in RING8 {
                        let s1 =
                            sample_bilinear_px(&snapshot, w, h, fx + dir[0] * r1, fy + dir[1] * r1);
                        let d1 = [s1[0] - rgb[0], s1[1] - rgb[1], s1[2] - rgb[2]];
                        let w1 = 0.85 * ((d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2]) * inv2_sig2).exp();
                        acc[0] += s1[0] * w1;
                        acc[1] += s1[1] * w1;
                        acc[2] += s1[2] * w1;
                        w_sum += w1;

                        let s2 =
                            sample_bilinear_px(&snapshot, w, h, fx + dir[0] * r2, fy + dir[1] * r2);
                        let d2 = [s2[0] - rgb[0], s2[1] - rgb[1], s2[2] - rgb[2]];
                        let w2 = 0.55 * ((d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2]) * inv2_sig2).exp();
                        acc[0] += s2[0] * w2;
                        acc[1] += s2[1] * w2;
                        acc[2] += s2[2] * w2;
                        w_sum += w2;
                    }
                    let inv = 1.0 / w_sum.max(1e-5);
                    [acc[0] * inv, acc[1] * inv, acc[2] * inv]
                } else {
                    rgb
                };

                let mid_freq_blotch = [
                    fine_low[0] - low_freq[0],
                    fine_low[1] - low_freq[1],
                    fine_low[2] - low_freq[2],
                ];

                // 1. 3-Band Frequency-Separation Skin Smoothing
                if skin_smooth > 1e-4 && skin_mask > 0.005 {
                    let pore_keep = (1.0 - skin_smooth * 0.55)
                        + (1.0 - (1.0 - skin_smooth * 0.55)) * skin_tex_preserve;
                    let mid_scale = 1.0 - skin_smooth * 0.88;
                    let fs_rgb = [
                        (low_freq[0] + mid_freq_blotch[0] * mid_scale + high_freq_pores[0] * pore_keep)
                            .clamp(0.0, 1.0),
                        (low_freq[1] + mid_freq_blotch[1] * mid_scale + high_freq_pores[1] * pore_keep)
                            .clamp(0.0, 1.0),
                        (low_freq[2] + mid_freq_blotch[2] * mid_scale + high_freq_pores[2] * pore_keep)
                            .clamp(0.0, 1.0),
                    ];
                    let blend_w = (skin_mask * skin_smooth * 1.05).clamp(0.0, 1.0);
                    rgb = mix3(rgb, fs_rgb, blend_w);
                }

                // 2. Skin Tone Evenness
                if skin_even > 1e-4 && skin_mask > 0.005 {
                    let cur_l = luma709(rgb);
                    let low_l = luma709(low_freq);
                    let cur_c = [rgb[0] - cur_l, rgb[1] - cur_l, rgb[2] - cur_l];
                    let low_c = [low_freq[0] - low_l, low_freq[1] - low_l, low_freq[2] - low_l];
                    let even_c = mix3(cur_c, low_c, skin_mask * skin_even * 0.65);
                    let red_spike = ((rgb[0] - rgb[1]) - (low_freq[0] - low_freq[1]) - 0.015).max(0.0);
                    rgb = [
                        (cur_l + even_c[0] - red_spike * skin_mask * skin_even * 0.45).clamp(0.0, 1.0),
                        (cur_l + even_c[1] + red_spike * skin_mask * skin_even * 0.14).clamp(0.0, 1.0),
                        (cur_l + even_c[2]).clamp(0.0, 1.0),
                    ];
                }

                // 3. Shine / Oil Control
                if shine_amt > 1e-4 && skin_mask > 0.005 {
                    let cur_l = luma709(rgb);
                    let max_c = rgb[0].max(rgb[1]).max(rgb[2]);
                    let min_c = rgb[0].min(rgb[1]).min(rgb[2]);
                    let sat = (max_c - min_c) / max_c.max(1e-4);
                    let spec_w = k_smoothstep(0.60, 0.86, cur_l) * (1.0 - k_smoothstep(0.14, 0.38, sat));
                    let low_l = luma709(low_freq);
                    let matte = [
                        (low_l * 0.92 + (low_freq[0] - low_l) * 1.18 + high_freq_pores[0] * 0.5)
                            .clamp(0.0, 1.0),
                        (low_l * 0.92 + (low_freq[1] - low_l) * 1.18 + high_freq_pores[1] * 0.5)
                            .clamp(0.0, 1.0),
                        (low_l * 0.92 + (low_freq[2] - low_l) * 1.18 + high_freq_pores[2] * 0.5)
                            .clamp(0.0, 1.0),
                    ];
                    rgb = mix3(rgb, matte, (skin_mask * spec_w * shine_amt * 0.78).clamp(0.0, 0.85));
                }

                // 4. Under-Eye Brighten
                if under_eye_amt > 1e-4 && under_eye_mask > 0.005 {
                    let ue_w = under_eye_mask * (skin_prob * 1.15).clamp(0.0, 1.0) * under_eye_amt;
                    if ue_w > 1e-4 {
                        let cur_l = luma709(rgb);
                        let shadow_lift = 0.115 * ue_w * (1.0 - k_smoothstep(0.45, 0.88, cur_l));
                        let soft_target = [
                            low_freq[0] + high_freq_pores[0] * 0.75,
                            low_freq[1] + high_freq_pores[1] * 0.75,
                            low_freq[2] + high_freq_pores[2] * 0.75,
                        ];
                        let softened = mix3(rgb, soft_target, (ue_w * 0.45).clamp(0.0, 0.6));
                        rgb = [
                            (softened[0] + shadow_lift * 1.06).clamp(0.0, 1.0),
                            (softened[1] + shadow_lift * 1.0).clamp(0.0, 1.0),
                            (softened[2] + shadow_lift * 0.86).clamp(0.0, 1.0),
                        ];
                    }
                }

                // 5. Eye Enhance
                if eye_amt > 1e-4 && eye_mask > 0.005 {
                    let e_w = eye_mask * eye_amt;
                    let cur_l = luma709(rgb);
                    let max_c = rgb[0].max(rgb[1]).max(rgb[2]);
                    let min_c = rgb[0].min(rgb[1]).min(rgb[2]);
                    let sat = (max_c - min_c) / max_c.max(1e-4);
                    let sclera_w = k_smoothstep(0.42, 0.72, cur_l) * (1.0 - k_smoothstep(0.10, 0.28, sat));
                    let sclera_lift = 0.065 * sclera_w * e_w * (1.0 - cur_l * 0.45);
                    let nw = mix3(rgb, [cur_l, cur_l, cur_l], sclera_w * e_w * 0.48);
                    let iris_w = 1.0 - sclera_w;
                    let iris_lift = 0.042 * iris_w * e_w * k_smoothstep(0.08, 0.45, cur_l);
                    rgb = [
                        (nw[0] + sclera_lift + high_freq_pores[0] * (e_w * 1.15) + iris_lift)
                            .clamp(0.0, 1.0),
                        (nw[1] + sclera_lift + high_freq_pores[1] * (e_w * 1.15) + iris_lift)
                            .clamp(0.0, 1.0),
                        (nw[2] + sclera_lift + high_freq_pores[2] * (e_w * 1.15) + iris_lift)
                            .clamp(0.0, 1.0),
                    ];
                }

                // 6. Teeth Whitening
                if teeth_amt > 1e-4 && teeth_mask > 0.005 {
                    let cur_l = luma709(rgb);
                    let max_c = rgb[0].max(rgb[1]).max(rgb[2]);
                    let min_c = rgb[0].min(rgb[1]).min(rgb[2]);
                    let sat = (max_c - min_c) / max_c.max(1e-4);
                    let non_lip = if has_face_parse {
                        1.0 - k_smoothstep(0.10, 0.22, rgb[0] - rgb[1])
                    } else {
                        1.0 - k_smoothstep(0.06, 0.15, rgb[0] - rgb[1])
                    };
                    let bright = k_smoothstep(0.32, 0.58, cur_l);
                    let low_sat = 1.0 - k_smoothstep(0.24, 0.46, sat);
                    let teeth_w = teeth_mask * non_lip * bright * low_sat * teeth_amt;
                    if teeth_w > 1e-4 {
                        let desat = mix3(rgb, [cur_l, cur_l, cur_l], (teeth_w * 0.72).clamp(0.0, 0.85));
                        let lift = 0.085 * teeth_w * (1.0 - cur_l * 0.55);
                        rgb = [
                            (desat[0] + lift).clamp(0.0, 1.0),
                            (desat[1] + lift).clamp(0.0, 1.0),
                            (desat[2] + lift).clamp(0.0, 1.0),
                        ];
                    }
                }

                // 7. Lip Enhance
                if lip_amt > 1e-4 && lip_mask > 0.005 {
                    let cur_l = luma709(rgb);
                    let max_c = rgb[0].max(rgb[1]).max(rgb[2]);
                    let min_c = rgb[0].min(rgb[1]).min(rgb[2]);
                    let sat = (max_c - min_c) / max_c.max(1e-4);
                    let lip_hue = if has_face_parse {
                        1.0
                    } else {
                        k_smoothstep(0.055, 0.17, rgb[0] - rgb[1])
                    };
                    let lip_sat = if has_face_parse {
                        k_smoothstep(0.04, 0.14, sat)
                    } else {
                        k_smoothstep(0.10, 0.28, sat)
                    };
                    let lip_w = lip_mask * lip_hue * lip_sat * lip_amt;
                    if lip_w > 1e-4 {
                        let c_scale = 1.0 + 0.30 * lip_w;
                        let base_l = cur_l * (1.0 - 0.04 * lip_w);
                        rgb = [
                            (base_l + (rgb[0] - cur_l) * c_scale + high_freq_pores[0] * (0.28 * lip_w))
                                .clamp(0.0, 1.0),
                            (base_l + (rgb[1] - cur_l) * c_scale + high_freq_pores[1] * (0.28 * lip_w))
                                .clamp(0.0, 1.0),
                            (base_l + (rgb[2] - cur_l) * c_scale + high_freq_pores[2] * (0.28 * lip_w))
                                .clamp(0.0, 1.0),
                        ];
                    }
                }

                row[idx] = rgb[0].clamp(0.0, 1.0);
                row[idx + 1] = rgb[1].clamp(0.0, 1.0);
                row[idx + 2] = rgb[2].clamp(0.0, 1.0);
            }
        });
}
