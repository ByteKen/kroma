//! 32-bit float parallel Local & AI Masking engine (Phase 2).
//!
//! Evaluates composable mask layers (`LinearGradient`, `RadialGradient`, `Brush`,
//! `LuminanceRange`, `ColorRange`, `AiMatte`) and applies per-mask local adjustments
//! formula-for-formula with the WebGL2 `MASK_FRAGMENT_SHADER_SRC` pass.

use crate::develop::color_math::{linear_to_srgb, rgb_to_hsl, soft_clip, srgb_to_linear};
use image::Rgb32FImage;
use kroma_domain::develop::{DevelopRecipe, NormalizedRect};
use kroma_domain::masking::{
    decode_matte_rle_b64, BackgroundGradientStyle, BackgroundReplaceMode, LocalAdjustments,
    MaskCombineOp, MaskComponent, MaskLayer,
};
use rayon::prelude::*;

struct PreparedSpatialComponent<'a> {
    comp: &'a MaskComponent,
    decoded_matte: Option<(usize, usize, Vec<u8>)>,
}

/// Applies all enabled, effective local/AI masks in `recipe.masks` in-place on `working`.
///
/// `working` has already been oriented (quarter-turns + flips) and optionally cropped to
/// `recipe.crop_transform.rect`. Each pixel `(x, y)` is mapped back to oriented full-frame
/// normalized coordinates `(u, v) ∈ [0..1]²` so masks remain pinned to the subject across crops.
pub fn apply_local_masks_f32(working: &mut Rgb32FImage, recipe: &DevelopRecipe) {
    if recipe.masks.is_empty() {
        return;
    }

    let (w, h) = working.dimensions();
    if w == 0 || h == 0 {
        return;
    }

    let rect = recipe.crop_transform.rect;
    // Full-frame oriented aspect ratio for circular brush strokes
    let full_w = (w as f32) / rect.width.max(1e-4);
    let full_h = (h as f32) / rect.height.max(1e-4);
    let aspect = (full_w / full_h.max(1.0)).clamp(0.1, 10.0);

    for layer in &recipe.masks {
        if !layer.is_effective() {
            continue;
        }
        apply_single_mask_layer_f32(working, layer, rect, aspect);
    }
}

/// Composites an opaque `SolidColor` or `Gradient` studio backdrop behind the subject matte
/// **before** local masks run, so `AI Background`, `Radial`, `Linear`, and `Brush` masks naturally
/// adjust the replaced backdrop (while global WB/Exposure stay clean on the subject).
pub fn apply_opaque_backdrop_f32(working: &mut Rgb32FImage, recipe: &DevelopRecipe) {
    let bg = &recipe.background_removal;
    if !bg.is_active()
        || !matches!(
            bg.mode,
            BackgroundReplaceMode::SolidColor | BackgroundReplaceMode::Gradient
        )
    {
        return;
    }
    let (w, h) = working.dimensions();
    if w == 0 || h == 0 {
        return;
    }
    let mw = bg.matte_width as usize;
    let mh = bg.matte_height as usize;
    let Some(decoded) = decode_matte_rle_b64(&bg.matte_rle_b64, mw * mh) else {
        return;
    };

    let w_usize = w as usize;
    let inv_w = 1.0 / (w as f32).max(1.0);
    let inv_h = 1.0 / (h as f32).max(1.0);
    let rect = recipe.crop_transform.rect;
    let feather = bg.feather;

    match bg.mode {
        BackgroundReplaceMode::SolidColor => {
            let bg_r = bg.color_rgb[0] as f32 / 255.0;
            let bg_g = bg.color_rgb[1] as f32 / 255.0;
            let bg_b = bg.color_rgb[2] as f32 / 255.0;
            let row_stride = w_usize * 3;
            working
                .as_mut()
                .par_chunks_exact_mut(row_stride)
                .enumerate()
                .for_each(|(y, row)| {
                    let v = rect.top + ((y as f32 + 0.5) * inv_h) * rect.height;
                    for x in 0..w_usize {
                        let u = rect.left + ((x as f32 + 0.5) * inv_w) * rect.width;
                        let raw_m = sample_matte_bilinear(&decoded, mw, mh, u, v);
                        let a = apply_matte_feather(raw_m, feather);
                        let inv_a = 1.0 - a;
                        let idx = x * 3;
                        row[idx] = (row[idx] * a + bg_r * inv_a).clamp(0.0, 1.0);
                        row[idx + 1] = (row[idx + 1] * a + bg_g * inv_a).clamp(0.0, 1.0);
                        row[idx + 2] = (row[idx + 2] * a + bg_b * inv_a).clamp(0.0, 1.0);
                    }
                });
        }
        BackgroundReplaceMode::Gradient => {
            let c1 = [
                bg.color_rgb[0] as f32 / 255.0,
                bg.color_rgb[1] as f32 / 255.0,
                bg.color_rgb[2] as f32 / 255.0,
            ];
            let c2 = [
                bg.color2_rgb[0] as f32 / 255.0,
                bg.color2_rgb[1] as f32 / 255.0,
                bg.color2_rgb[2] as f32 / 255.0,
            ];
            let is_linear = bg.gradient_style == BackgroundGradientStyle::Linear;
            let rad = bg.gradient_angle_deg.to_radians();
            let dir_x = rad.sin();
            let dir_y = -rad.cos();
            let span = (dir_x.abs() + dir_y.abs()).max(1e-5);
            let row_stride = w_usize * 3;
            working
                .as_mut()
                .par_chunks_exact_mut(row_stride)
                .enumerate()
                .for_each(|(y, row)| {
                    let v = rect.top + ((y as f32 + 0.5) * inv_h) * rect.height;
                    for x in 0..w_usize {
                        let u = rect.left + ((x as f32 + 0.5) * inv_w) * rect.width;
                        let raw_m = sample_matte_bilinear(&decoded, mw, mh, u, v);
                        let a = apply_matte_feather(raw_m, feather);
                        let inv_a = 1.0 - a;
                        let t = if is_linear {
                            let proj = (u - 0.5) * dir_x + (v - 0.5) * dir_y;
                            (proj / span + 0.5).clamp(0.0, 1.0)
                        } else {
                            let dx = (u - 0.5) / 0.68;
                            let dy = (v - 0.42) / 0.72;
                            let d = (dx * dx + dy * dy).sqrt().clamp(0.0, 1.0);
                            smoothstep(0.0, 1.0, d)
                        };
                        let bg_r = c1[0] + (c2[0] - c1[0]) * t;
                        let bg_g = c1[1] + (c2[1] - c1[1]) * t;
                        let bg_b = c1[2] + (c2[2] - c1[2]) * t;
                        let idx = x * 3;
                        row[idx] = (row[idx] * a + bg_r * inv_a).clamp(0.0, 1.0);
                        row[idx + 1] = (row[idx + 1] * a + bg_g * inv_a).clamp(0.0, 1.0);
                        row[idx + 2] = (row[idx + 2] * a + bg_b * inv_a).clamp(0.0, 1.0);
                    }
                });
        }
        BackgroundReplaceMode::Original | BackgroundReplaceMode::Transparent => {}
    }
}

/// Computes the per-pixel foreground alpha plane (`[0..1]`) when `BackgroundReplaceMode::Transparent`
/// is active so PNG exports write real 32-bit RGBA transparency (while JPEG composites over white).
pub fn compute_transparent_alpha_f32(
    working: &Rgb32FImage,
    recipe: &DevelopRecipe,
) -> Option<Vec<f32>> {
    let bg = &recipe.background_removal;
    if !bg.is_active() || bg.mode != BackgroundReplaceMode::Transparent {
        return None;
    }
    let (w, h) = working.dimensions();
    if w == 0 || h == 0 {
        return None;
    }
    let mw = bg.matte_width as usize;
    let mh = bg.matte_height as usize;
    let decoded = decode_matte_rle_b64(&bg.matte_rle_b64, mw * mh)?;

    let w_usize = w as usize;
    let h_usize = h as usize;
    let inv_w = 1.0 / (w as f32).max(1.0);
    let inv_h = 1.0 / (h as f32).max(1.0);
    let rect = recipe.crop_transform.rect;
    let feather = bg.feather;

    let mut alpha_plane = vec![1.0_f32; w_usize * h_usize];
    alpha_plane
        .par_chunks_exact_mut(w_usize)
        .enumerate()
        .for_each(|(y, row)| {
            let v = rect.top + ((y as f32 + 0.5) * inv_h) * rect.height;
            for (x, slot) in row.iter_mut().enumerate().take(w_usize) {
                let u = rect.left + ((x as f32 + 0.5) * inv_w) * rect.width;
                let raw_m = sample_matte_bilinear(&decoded, mw, mh, u, v);
                *slot = apply_matte_feather(raw_m, feather);
            }
        });
    Some(alpha_plane)
}

/// Rasterizes all leading spatial components (`LinearGradient`, `RadialGradient`, `Brush`, `AiMatte`)
/// of a `MaskLayer` into an `RGBA8` texture (`R=G=B=alpha, A=255`) for GPU `mask_pass.wgsl`.
/// Returns `None` if the layer has range components *before* spatial components or >4 range components
/// (in which case the caller falls back cleanly to CPU `apply_local_masks_f32`).
#[allow(clippy::type_complexity)]
pub(crate) fn compile_mask_layer_for_gpu(
    layer: &MaskLayer,
    oriented_w: u32,
    oriented_h: u32,
    max_dim: u32,
) -> Option<(bool, u32, u32, Vec<u8>, Vec<&MaskComponent>)> {
    let mut spatial_comps: Vec<&MaskComponent> = Vec::new();
    let mut range_comps: Vec<&MaskComponent> = Vec::new();
    let mut seen_range = false;

    for comp in &layer.components {
        match comp {
            MaskComponent::LuminanceRange { .. } | MaskComponent::ColorRange { .. } => {
                seen_range = true;
                range_comps.push(comp);
            }
            _ => {
                if seen_range {
                    // Interleaved range -> spatial cannot be split cleanly without extra passes; use CPU fallback
                    return None;
                }
                spatial_comps.push(comp);
            }
        }
    }

    if range_comps.len() > 4 {
        return None;
    }

    if spatial_comps.is_empty() {
        return Some((false, 1, 1, vec![255, 255, 255, 255], range_comps));
    }

    let prepared: Vec<PreparedSpatialComponent<'_>> = spatial_comps
        .into_iter()
        .map(|comp| {
            let decoded_matte = match comp {
                MaskComponent::AiMatte {
                    matte_width,
                    matte_height,
                    matte_rle_b64,
                    ..
                } => {
                    let mw = *matte_width as usize;
                    let mh = *matte_height as usize;
                    if mw > 0 && mh > 0 && !matte_rle_b64.is_empty() {
                        decode_matte_rle_b64(matte_rle_b64, mw * mh).map(|buf| (mw, mh, buf))
                    } else {
                        None
                    }
                }
                _ => None,
            };
            PreparedSpatialComponent {
                comp,
                decoded_matte,
            }
        })
        .collect();

    let src_max = oriented_w.max(oriented_h).max(1);
    let scale = if max_dim > 0 && src_max > max_dim {
        max_dim as f32 / src_max as f32
    } else {
        1.0
    };
    let rw = ((oriented_w as f32) * scale).round().max(1.0) as u32;
    let rh = ((oriented_h as f32) * scale).round().max(1.0) as u32;
    let rw_u = rw as usize;
    let rh_u = rh as usize;
    let inv_w = 1.0 / (rw as f32).max(1.0);
    let inv_h = 1.0 / (rh as f32).max(1.0);
    let aspect = (oriented_w as f32 / (oriented_h as f32).max(1.0)).clamp(0.1, 10.0);

    let mut rgba = vec![0_u8; rw_u * rh_u * 4];
    rgba.par_chunks_exact_mut(rw_u * 4)
        .enumerate()
        .for_each(|(y, row)| {
            let v = (y as f32 + 0.5) * inv_h;
            for x in 0..rw_u {
                let u = (x as f32 + 0.5) * inv_w;
                let alpha = evaluate_layer_alpha(
                    &prepared,
                    u,
                    v,
                    aspect,
                    [0.0, 0.0, 0.0],
                    false,
                    1.0,
                );
                let byte = (alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
                let idx = x * 4;
                row[idx] = byte;
                row[idx + 1] = byte;
                row[idx + 2] = byte;
                row[idx + 3] = 255;
            }
        });

    Some((true, rw, rh, rgba, range_comps))
}

fn apply_single_mask_layer_f32(
    working: &mut Rgb32FImage,
    layer: &MaskLayer,
    crop_rect: NormalizedRect,
    aspect: f32,
) {
    let (w, h) = working.dimensions();
    let w_usize = w as usize;
    let h_usize = h as usize;
    let inv_w = 1.0 / (w as f32).max(1.0);
    let inv_h = 1.0 / (h as f32).max(1.0);
    let opacity = (layer.opacity as f32 / 100.0).clamp(0.0, 1.0);

    let prepared: Vec<PreparedSpatialComponent<'_>> = layer
        .components
        .iter()
        .map(|comp| {
            let decoded_matte = match comp {
                MaskComponent::AiMatte {
                    matte_width,
                    matte_height,
                    matte_rle_b64,
                    ..
                } => {
                    let mw = *matte_width as usize;
                    let mh = *matte_height as usize;
                    if mw > 0 && mh > 0 && !matte_rle_b64.is_empty() {
                        decode_matte_rle_b64(matte_rle_b64, mw * mh).map(|buf| (mw, mh, buf))
                    } else {
                        None
                    }
                }
                _ => None,
            };
            PreparedSpatialComponent {
                comp,
                decoded_matte,
            }
        })
        .collect();

    let need_spatial = layer.adjustments.has_spatial_adjustments();
    let mut alpha_plane = if need_spatial {
        vec![0.0_f32; w_usize * h_usize]
    } else {
        Vec::new()
    };

    let adj = layer.adjustments;
    let has_tone_or_color = adj.temperature_shift != 0
        || adj.tint_shift != 0
        || adj.exposure_ev.abs() > 1e-4
        || adj.contrast != 0
        || adj.highlights != 0
        || adj.shadows != 0
        || adj.whites != 0
        || adj.blacks != 0
        || adj.dehaze != 0
        || adj.saturation != 0;

    let (wb_gains, exp_gain, linear_active) = compile_local_linear(adj);
    let row_stride = w_usize * 3;
    let raw = working.as_mut();

    if need_spatial {
        raw.par_chunks_exact_mut(row_stride)
            .zip(alpha_plane.par_chunks_exact_mut(w_usize))
            .enumerate()
            .for_each(|(y, (row, alpha_row))| {
                let v = crop_rect.top + ((y as f32 + 0.5) * inv_h) * crop_rect.height;
                for (x, alpha_slot) in alpha_row.iter_mut().enumerate().take(w_usize) {
                    let u = crop_rect.left + ((x as f32 + 0.5) * inv_w) * crop_rect.width;
                    let idx = x * 3;
                    let rgb = [row[idx], row[idx + 1], row[idx + 2]];
                    let alpha = evaluate_layer_alpha(
                        &prepared,
                        u,
                        v,
                        aspect,
                        rgb,
                        layer.inverted,
                        opacity,
                    );
                    *alpha_slot = alpha;
                    if has_tone_or_color && alpha > 1e-4 {
                        let graded =
                            apply_local_tone_color(rgb, adj, wb_gains, exp_gain, linear_active);
                        row[idx] = rgb[0] + (graded[0] - rgb[0]) * alpha;
                        row[idx + 1] = rgb[1] + (graded[1] - rgb[1]) * alpha;
                        row[idx + 2] = rgb[2] + (graded[2] - rgb[2]) * alpha;
                    }
                }
            });

        apply_local_spatial_detail_f32(working, &alpha_plane, adj);
    } else if has_tone_or_color {
        raw.par_chunks_exact_mut(row_stride)
            .enumerate()
            .for_each(|(y, row)| {
                let v = crop_rect.top + ((y as f32 + 0.5) * inv_h) * crop_rect.height;
                for x in 0..w_usize {
                    let u = crop_rect.left + ((x as f32 + 0.5) * inv_w) * crop_rect.width;
                    let idx = x * 3;
                    let rgb = [row[idx], row[idx + 1], row[idx + 2]];
                    let alpha = evaluate_layer_alpha(
                        &prepared,
                        u,
                        v,
                        aspect,
                        rgb,
                        layer.inverted,
                        opacity,
                    );
                    if alpha > 1e-4 {
                        let graded =
                            apply_local_tone_color(rgb, adj, wb_gains, exp_gain, linear_active);
                        row[idx] = rgb[0] + (graded[0] - rgb[0]) * alpha;
                        row[idx + 1] = rgb[1] + (graded[1] - rgb[1]) * alpha;
                        row[idx + 2] = rgb[2] + (graded[2] - rgb[2]) * alpha;
                    }
                }
            });
    }
}

fn evaluate_layer_alpha(
    prepared: &[PreparedSpatialComponent<'_>],
    u: f32,
    v: f32,
    aspect: f32,
    rgb: [f32; 3],
    layer_inverted: bool,
    opacity: f32,
) -> f32 {
    if prepared.is_empty() {
        return 0.0;
    }

    let mut acc = 0.0_f32;
    for (idx, item) in prepared.iter().enumerate() {
        let (op, comp_inv, mut val) = match item.comp {
            MaskComponent::LinearGradient {
                op,
                inverted,
                start_x,
                start_y,
                end_x,
                end_y,
                feather,
                ..
            } => (
                *op,
                *inverted,
                eval_linear_gradient(u, v, *start_x, *start_y, *end_x, *end_y, *feather),
            ),
            MaskComponent::RadialGradient {
                op,
                inverted,
                center_x,
                center_y,
                radius_x,
                radius_y,
                rotation_deg,
                feather,
                ..
            } => (
                *op,
                *inverted,
                eval_radial_gradient(
                    u,
                    v,
                    *center_x,
                    *center_y,
                    *radius_x,
                    *radius_y,
                    *rotation_deg,
                    *feather,
                ),
            ),
            MaskComponent::Brush {
                op,
                inverted,
                strokes,
                ..
            } => (*op, *inverted, eval_brush(u, v, aspect, strokes)),
            MaskComponent::LuminanceRange {
                op,
                inverted,
                min_luma,
                max_luma,
                feather,
                ..
            } => (
                *op,
                *inverted,
                eval_luminance_range(rgb, *min_luma, *max_luma, *feather),
            ),
            MaskComponent::ColorRange {
                op,
                inverted,
                target_hue_deg,
                target_sat,
                target_luma,
                tolerance,
                feather,
                ..
            } => (
                *op,
                *inverted,
                eval_color_range(
                    rgb,
                    *target_hue_deg,
                    *target_sat,
                    *target_luma,
                    *tolerance,
                    *feather,
                ),
            ),
            MaskComponent::AiMatte {
                op,
                inverted,
                feather,
                ..
            } => {
                let raw = match &item.decoded_matte {
                    Some((mw, mh, buf)) => sample_matte_bilinear(buf, *mw, *mh, u, v),
                    None => 0.0,
                };
                (*op, *inverted, apply_matte_feather(raw, *feather))
            }
        };

        if comp_inv {
            val = 1.0 - val;
        }

        if idx == 0 {
            acc = match op {
                MaskCombineOp::Subtract => (1.0 - val).clamp(0.0, 1.0),
                MaskCombineOp::Add | MaskCombineOp::Intersect => val.clamp(0.0, 1.0),
            };
        } else {
            acc = match op {
                MaskCombineOp::Add => (acc + val).clamp(0.0, 1.0),
                MaskCombineOp::Subtract => (acc - val).clamp(0.0, 1.0),
                MaskCombineOp::Intersect => (acc * val).clamp(0.0, 1.0),
            };
        }
    }

    if layer_inverted {
        acc = 1.0 - acc;
    }
    (acc * opacity).clamp(0.0, 1.0)
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let denom = (edge1 - edge0).max(1e-6);
    let t = ((x - edge0) / denom).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
pub fn eval_linear_gradient(
    u: f32,
    v: f32,
    start_x: f32,
    start_y: f32,
    end_x: f32,
    end_y: f32,
    feather: f32,
) -> f32 {
    let dx = end_x - start_x;
    let dy = end_y - start_y;
    let len2 = (dx * dx + dy * dy).max(1e-6);
    let t = ((u - start_x) * dx + (v - start_y) * dy) / len2;
    let half_span = 0.04 + (feather.clamp(0.0, 100.0) / 100.0) * 0.46;
    1.0 - smoothstep(0.5 - half_span, 0.5 + half_span, t)
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn eval_radial_gradient(
    u: f32,
    v: f32,
    center_x: f32,
    center_y: f32,
    radius_x: f32,
    radius_y: f32,
    rotation_deg: f32,
    feather: f32,
) -> f32 {
    let px = u - center_x;
    let py = v - center_y;
    let rad = rotation_deg.to_radians();
    let (sin_r, cos_r) = rad.sin_cos();
    let rx = px * cos_r + py * sin_r;
    let ry = -px * sin_r + py * cos_r;
    let nx = rx / radius_x.max(0.005);
    let ny = ry / radius_y.max(0.005);
    let d = (nx * nx + ny * ny).sqrt();
    let inner = (1.0 - feather.clamp(0.0, 100.0) / 100.0).clamp(0.0, 0.99);
    1.0 - smoothstep(inner, 1.0, d)
}

fn eval_brush(
    u: f32,
    v: f32,
    aspect: f32,
    strokes: &[kroma_domain::masking::BrushStroke],
) -> f32 {
    if strokes.is_empty() {
        return 0.0;
    }
    let sx = if aspect >= 1.0 { aspect } else { 1.0 };
    let sy = if aspect >= 1.0 { 1.0 } else { 1.0 / aspect };
    let pu = u * sx;
    let pv = v * sy;

    let mut acc = 0.0_f32;
    for stroke in strokes {
        if stroke.points.is_empty() {
            continue;
        }
        let radius = stroke.size.max(0.004);
        let radius2 = radius * radius;
        let mut min_d2 = f32::INFINITY;

        if stroke.points.len() == 1 {
            let ax = stroke.points[0][0] * sx;
            let ay = stroke.points[0][1] * sy;
            let dx = pu - ax;
            let dy = pv - ay;
            min_d2 = dx * dx + dy * dy;
        } else {
            for seg in stroke.points.windows(2) {
                let ax = seg[0][0] * sx;
                let ay = seg[0][1] * sy;
                let bx = seg[1][0] * sx;
                let by = seg[1][1] * sy;
                let abx = bx - ax;
                let aby = by - ay;
                let ab2 = abx * abx + aby * aby;
                let t = if ab2 > 1e-8 {
                    (((pu - ax) * abx + (pv - ay) * aby) / ab2).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let cx = ax + abx * t;
                let cy = ay + aby * t;
                let dx = pu - cx;
                let dy = pv - cy;
                let d2 = dx * dx + dy * dy;
                if d2 < min_d2 {
                    min_d2 = d2;
                }
            }
        }

        if min_d2 < radius2 {
            let dist = min_d2.sqrt();
            let inner = radius * (1.0 - stroke.feather.clamp(0.0, 1.0)).clamp(0.0, 0.98);
            let dab = (1.0 - smoothstep(inner, radius, dist)) * stroke.flow.clamp(0.05, 1.0);
            acc = (acc + dab * (1.0 - acc)).min(1.0);
        }
    }
    acc
}

#[inline]
pub fn eval_luminance_range(rgb: [f32; 3], min_luma: f32, max_luma: f32, feather: f32) -> f32 {
    let luma = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]).clamp(0.0, 1.0);
    let f = feather.clamp(0.0, 0.5).max(0.005);
    let lo = if min_luma <= 0.001 {
        1.0
    } else {
        smoothstep(min_luma - f, min_luma + f * 0.25, luma)
    };
    let hi = if max_luma >= 0.999 {
        1.0
    } else {
        1.0 - smoothstep(max_luma - f * 0.25, max_luma + f, luma)
    };
    (lo * hi).clamp(0.0, 1.0)
}

#[inline]
pub fn eval_color_range(
    rgb: [f32; 3],
    target_hue_deg: f32,
    target_sat: f32,
    target_luma: f32,
    tolerance: f32,
    feather: f32,
) -> f32 {
    let (h, s, l) = rgb_to_hsl(rgb);
    let raw_diff = (h - target_hue_deg).abs().rem_euclid(360.0);
    let hue_dist = if raw_diff > 180.0 {
        360.0 - raw_diff
    } else {
        raw_diff
    };
    let dh = hue_dist / 180.0;
    let hue_w = (s.min(target_sat) * 4.0).clamp(0.0, 1.0);
    let ds = (s - target_sat).abs();
    let dl = (l - target_luma).abs();
    let dist = ((dh * 2.2 * hue_w).powi(2) + (ds * 0.9).powi(2) + (dl * 0.75).powi(2)).sqrt();
    let tol = tolerance.clamp(0.02, 1.0);
    let f = (tol * feather.clamp(0.0, 1.0)).max(0.015);
    1.0 - smoothstep((tol - f * 0.5).max(0.0), tol + f, dist)
}

#[inline]
fn sample_matte_bilinear(buf: &[u8], mw: usize, mh: usize, u: f32, v: f32) -> f32 {
    let fx = (u.clamp(0.0, 1.0) * (mw.saturating_sub(1)) as f32).clamp(0.0, (mw - 1) as f32);
    let fy = (v.clamp(0.0, 1.0) * (mh.saturating_sub(1)) as f32).clamp(0.0, (mh - 1) as f32);
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(mw - 1);
    let y1 = (y0 + 1).min(mh - 1);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;

    let c00 = buf[y0 * mw + x0] as f32 / 255.0;
    let c10 = buf[y0 * mw + x1] as f32 / 255.0;
    let c01 = buf[y1 * mw + x0] as f32 / 255.0;
    let c11 = buf[y1 * mw + x1] as f32 / 255.0;

    let top = c00 + (c10 - c00) * tx;
    let bot = c01 + (c11 - c01) * tx;
    top + (bot - top) * ty
}

#[inline]
fn apply_matte_feather(m: f32, feather: f32) -> f32 {
    let f = feather.clamp(0.0, 100.0);
    if (f - 25.0).abs() < 0.5 {
        m
    } else if f < 25.0 {
        let k = 1.0 + ((25.0 - f) / 25.0) * 3.0;
        ((m - 0.5) * k + 0.5).clamp(0.0, 1.0)
    } else {
        let t = (f - 25.0) / 75.0;
        let soft = smoothstep(0.0, 1.0, m.clamp(0.0, 1.0).powf(1.0 - 0.35 * t));
        m + (soft - m) * t
    }
}

#[inline]
pub(crate) fn compile_local_linear(adj: LocalAdjustments) -> ([f32; 3], f32, bool) {
    let linear_active =
        adj.temperature_shift != 0 || adj.tint_shift != 0 || adj.exposure_ev.abs() > 1e-4;
    if !linear_active {
        return ([1.0, 1.0, 1.0], 1.0, false);
    }
    let tn = adj.temperature_shift as f32 / 100.0;
    let gn = adj.tint_shift as f32 / 100.0;
    let r = (1.0 + tn * 0.28 + gn * 0.06).max(0.25);
    let g = (1.0 - gn * 0.14).max(0.25);
    let b = (1.0 - tn * 0.28 + gn * 0.06).max(0.25);
    let wb = [r / g, 1.0, b / g];
    let exp_gain = 2.0_f32.powf(adj.exposure_ev.clamp(-4.0, 4.0));
    (wb, exp_gain, true)
}

#[inline]
fn apply_local_tone_color(
    mut rgb: [f32; 3],
    adj: LocalAdjustments,
    wb_gains: [f32; 3],
    exp_gain: f32,
    linear_active: bool,
) -> [f32; 3] {
    if linear_active {
        let lin_r = (srgb_to_linear(rgb[0]) * wb_gains[0] * exp_gain).max(0.0);
        let lin_g = (srgb_to_linear(rgb[1]) * wb_gains[1] * exp_gain).max(0.0);
        let lin_b = (srgb_to_linear(rgb[2]) * wb_gains[2] * exp_gain).max(0.0);
        rgb = [
            linear_to_srgb(soft_clip(lin_r)),
            linear_to_srgb(soft_clip(lin_g)),
            linear_to_srgb(soft_clip(lin_b)),
        ];
    }

    let luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    let mut delta = 0.0_f32;
    if adj.shadows != 0 {
        let sw = 1.0 - smoothstep(0.05, 0.55, luma);
        delta += (adj.shadows as f32 / 100.0) * 0.28 * sw * sw;
    }
    if adj.highlights != 0 {
        let hw = smoothstep(0.45, 0.95, luma);
        delta += (adj.highlights as f32 / 100.0) * 0.28 * hw * hw;
    }
    if adj.whites != 0 {
        delta += (adj.whites as f32 / 100.0) * 0.22 * smoothstep(0.65, 1.0, luma);
    }
    if adj.blacks != 0 {
        delta += (adj.blacks as f32 / 100.0) * 0.22 * (1.0 - smoothstep(0.0, 0.35, luma));
    }
    if delta.abs() > 1e-6 {
        let target_luma = (luma + delta).clamp(0.0, 1.0);
        if luma > 1e-4 {
            let scale = target_luma / luma;
            rgb = [
                (rgb[0] * scale).clamp(0.0, 1.0),
                (rgb[1] * scale).clamp(0.0, 1.0),
                (rgb[2] * scale).clamp(0.0, 1.0),
            ];
        } else {
            rgb = [target_luma; 3];
        }
    }

    if adj.dehaze != 0 {
        let dh = adj.dehaze as f32 / 100.0;
        let lift = -dh * 0.06;
        let sc = 1.0 + dh * 0.18;
        rgb = [
            ((rgb[0] + lift) * sc).clamp(0.0, 1.0),
            ((rgb[1] + lift) * sc).clamp(0.0, 1.0),
            ((rgb[2] + lift) * sc).clamp(0.0, 1.0),
        ];
    }

    if adj.contrast != 0 {
        let c = adj.contrast as f32 / 100.0;
        let factor = if c >= 0.0 {
            1.0 + c * 0.85
        } else {
            1.0 + c * 0.65
        };
        rgb = [
            ((rgb[0] - 0.5) * factor + 0.5).clamp(0.0, 1.0),
            ((rgb[1] - 0.5) * factor + 0.5).clamp(0.0, 1.0),
            ((rgb[2] - 0.5) * factor + 0.5).clamp(0.0, 1.0),
        ];
    }

    if adj.saturation != 0 {
        let l_cur = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
        let sat = 1.0 + adj.saturation as f32 / 100.0;
        rgb = [
            (l_cur + (rgb[0] - l_cur) * sat).clamp(0.0, 1.0),
            (l_cur + (rgb[1] - l_cur) * sat).clamp(0.0, 1.0),
            (l_cur + (rgb[2] - l_cur) * sat).clamp(0.0, 1.0),
        ];
    }

    rgb
}

fn apply_local_spatial_detail_f32(
    img: &mut Rgb32FImage,
    alpha_plane: &[f32],
    adj: LocalAdjustments,
) {
    let tex = (adj.texture as f32 / 100.0).clamp(-1.0, 1.0);
    let clar = (adj.clarity as f32 / 100.0).clamp(-1.0, 1.0);
    let sharp = (adj.sharpness as f32 / 100.0).clamp(-1.0, 1.0);
    if tex.abs() <= 1e-4 && clar.abs() <= 1e-4 && sharp.abs() <= 1e-4 {
        return;
    }

    let (w, h) = img.dimensions();
    if w < 2 || h < 2 {
        return;
    }
    let w_usize = w as usize;
    let h_usize = h as usize;
    let raw = img.as_raw();
    let mut luma_plane = vec![0.0_f32; w_usize * h_usize];
    luma_plane
        .par_chunks_exact_mut(w_usize)
        .enumerate()
        .for_each(|(y, luma_row)| {
            let base = y * w_usize * 3;
            for (x, slot) in luma_row.iter_mut().enumerate() {
                let idx = base + x * 3;
                *slot = 0.2126 * raw[idx] + 0.7152 * raw[idx + 1] + 0.0722 * raw[idx + 2];
            }
        });

    let spatial_scale = ((w.max(h) as f32) / 1024.0).clamp(0.5, 4.0);
    let r_tex = 2.5 * spatial_scale;
    let d_tex = r_tex * 0.707_106_77;
    let r_clar = 11.0 * spatial_scale;
    let in_clar = r_clar * 0.45;
    let d_clar = r_clar * 0.707_106_77;
    let r_sharp = 1.0 * spatial_scale;
    let r_soft = 1.6 * spatial_scale;

    let src_raw = raw.to_vec();
    let dst_raw = img.as_mut();
    let row_stride = w_usize * 3;

    dst_raw
        .par_chunks_exact_mut(row_stride)
        .enumerate()
        .for_each(|(y, row)| {
            let fy = y as f32;
            let alpha_base = y * w_usize;
            for x in 0..w_usize {
                let alpha = alpha_plane[alpha_base + x];
                if alpha <= 1e-4 {
                    continue;
                }
                let fx = x as f32;
                let idx = x * 3;
                let src_idx = y * row_stride + idx;
                let mut rgb = [src_raw[src_idx], src_raw[src_idx + 1], src_raw[src_idx + 2]];
                let lc = luma_plane[alpha_base + x];
                let mut total_delta = 0.0_f32;

                if tex.abs() > 1e-4 {
                    let blur_tex = 0.20 * lc
                        + 0.12
                            * (sample_luma(&luma_plane, w_usize, h_usize, fx - r_tex, fy)
                                + sample_luma(&luma_plane, w_usize, h_usize, fx + r_tex, fy)
                                + sample_luma(&luma_plane, w_usize, h_usize, fx, fy - r_tex)
                                + sample_luma(&luma_plane, w_usize, h_usize, fx, fy + r_tex))
                        + 0.08
                            * (sample_luma(&luma_plane, w_usize, h_usize, fx - d_tex, fy - d_tex)
                                + sample_luma(&luma_plane, w_usize, h_usize, fx + d_tex, fy - d_tex)
                                + sample_luma(&luma_plane, w_usize, h_usize, fx - d_tex, fy + d_tex)
                                + sample_luma(
                                    &luma_plane,
                                    w_usize,
                                    h_usize,
                                    fx + d_tex,
                                    fy + d_tex,
                                ));
                    let detail = (lc - blur_tex).clamp(-0.18, 0.18);
                    total_delta += detail * tex * 0.85;
                }

                if clar.abs() > 1e-4 {
                    let blur_clar = 0.16 * lc
                        + 0.10
                            * (sample_luma(&luma_plane, w_usize, h_usize, fx - in_clar, fy)
                                + sample_luma(&luma_plane, w_usize, h_usize, fx + in_clar, fy)
                                + sample_luma(&luma_plane, w_usize, h_usize, fx, fy - in_clar)
                                + sample_luma(&luma_plane, w_usize, h_usize, fx, fy + in_clar))
                        + 0.06
                            * (sample_luma(&luma_plane, w_usize, h_usize, fx - r_clar, fy)
                                + sample_luma(&luma_plane, w_usize, h_usize, fx + r_clar, fy)
                                + sample_luma(&luma_plane, w_usize, h_usize, fx, fy - r_clar)
                                + sample_luma(&luma_plane, w_usize, h_usize, fx, fy + r_clar))
                        + 0.05
                            * (sample_luma(&luma_plane, w_usize, h_usize, fx - d_clar, fy - d_clar)
                                + sample_luma(
                                    &luma_plane,
                                    w_usize,
                                    h_usize,
                                    fx + d_clar,
                                    fy - d_clar,
                                )
                                + sample_luma(
                                    &luma_plane,
                                    w_usize,
                                    h_usize,
                                    fx - d_clar,
                                    fy + d_clar,
                                )
                                + sample_luma(
                                    &luma_plane,
                                    w_usize,
                                    h_usize,
                                    fx + d_clar,
                                    fy + d_clar,
                                ));
                    let mid_mask = (4.0 * lc * (1.0 - lc)).clamp(1e-6, 1.0).powf(0.70);
                    let detail = (lc - blur_clar).clamp(-0.30, 0.30);
                    total_delta += detail * clar * 1.15 * mid_mask;
                }

                if sharp > 1e-4 {
                    let avg4 = 0.25
                        * (sample_luma(&luma_plane, w_usize, h_usize, fx - r_sharp, fy)
                            + sample_luma(&luma_plane, w_usize, h_usize, fx + r_sharp, fy)
                            + sample_luma(&luma_plane, w_usize, h_usize, fx, fy - r_sharp)
                            + sample_luma(&luma_plane, w_usize, h_usize, fx, fy + r_sharp));
                    let hp = lc - avg4;
                    let mag = (hp.abs() - 0.008).max(0.0);
                    total_delta += (hp.signum() * mag * sharp * 1.10).clamp(-0.25, 0.25);
                } else if sharp < -1e-4 {
                    // Negative local sharpness: edge-preserving bilateral softening (skin smoothing)
                    let soften = -sharp;
                    let sigma_r = 0.04 + soften * 0.12;
                    let inv_2sig2 = -0.5 / (sigma_r * sigma_r);
                    let mut sum_val = lc;
                    let mut sum_w = 1.0_f32;
                    let offsets: [(f32, f32, f32); 8] = [
                        (-r_soft, 0.0, 0.65),
                        (r_soft, 0.0, 0.65),
                        (0.0, -r_soft, 0.65),
                        (0.0, r_soft, 0.65),
                        (-r_soft, -r_soft, 0.45),
                        (r_soft, -r_soft, 0.45),
                        (-r_soft, r_soft, 0.45),
                        (r_soft, r_soft, 0.45),
                    ];
                    for &(ox, oy, sw) in &offsets {
                        let ln = sample_luma(&luma_plane, w_usize, h_usize, fx + ox, fy + oy);
                        let diff = ln - lc;
                        let wt = sw * (diff * diff * inv_2sig2).exp();
                        sum_val += ln * wt;
                        sum_w += wt;
                    }
                    let l_soft = sum_val / sum_w.max(1e-5);
                    total_delta += (l_soft - lc) * (soften * 0.90);
                }

                total_delta *= alpha;
                if total_delta.abs() > 1e-6 {
                    let next_luma = (lc + total_delta).clamp(0.0, 1.0);
                    let actual_delta = next_luma - lc;
                    let ratio = if lc > 1e-4 {
                        (next_luma / lc).clamp(0.0, 3.0)
                    } else {
                        1.0
                    };
                    let blend = smoothstep(0.02, 0.16, lc);
                    for channel in &mut rgb {
                        let additive = (*channel + actual_delta).clamp(0.0, 1.0);
                        let mult = if lc > 1e-4 {
                            (*channel * ratio).clamp(0.0, 1.0)
                        } else {
                            next_luma
                        };
                        *channel = additive + (mult - additive) * blend;
                    }
                    row[idx] = rgb[0];
                    row[idx + 1] = rgb[1];
                    row[idx + 2] = rgb[2];
                }
            }
        });
}

#[inline]
fn sample_luma(plane: &[f32], w: usize, h: usize, fx: f32, fy: f32) -> f32 {
    let cx = fx.clamp(0.0, (w - 1) as f32);
    let cy = fy.clamp(0.0, (h - 1) as f32);
    let x0 = cx.floor() as usize;
    let y0 = cy.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let tx = cx - x0 as f32;
    let ty = cy - y0 as f32;
    let l00 = plane[y0 * w + x0];
    let l10 = plane[y0 * w + x1];
    let l01 = plane[y1 * w + x0];
    let l11 = plane[y1 * w + x1];
    let top = l00 + (l10 - l00) * tx;
    let bot = l01 + (l11 - l01) * tx;
    top + (bot - top) * ty
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;
    use kroma_domain::masking::{encode_matte_rle_b64, AiMaskKind, BrushStroke};

    #[test]
    fn linear_gradient_mask_brightens_top_half_and_preserves_bottom() {
        let mut img = Rgb32FImage::from_pixel(32, 32, Rgb([0.35, 0.35, 0.35]));
        let mut recipe = DevelopRecipe::default();
        recipe.masks.push(MaskLayer {
            id: "m_lin".to_string(),
            name: "Sky Grad".to_string(),
            enabled: true,
            inverted: false,
            opacity: 100,
            components: vec![MaskComponent::LinearGradient {
                id: "c1".to_string(),
                op: MaskCombineOp::Add,
                inverted: false,
                start_x: 0.5,
                start_y: 0.1,
                end_x: 0.5,
                end_y: 0.9,
                feather: 50.0,
            }],
            adjustments: LocalAdjustments {
                exposure_ev: 1.5,
                ..LocalAdjustments::default()
            },
        });

        apply_local_masks_f32(&mut img, &recipe);
        let top_px = img.get_pixel(16, 1).0[0];
        let bot_px = img.get_pixel(16, 30).0[0];
        assert!(top_px > 0.55, "expected bright top pixel, got {top_px}");
        assert!(
            (bot_px - 0.35).abs() < 0.02,
            "expected untouched bottom pixel, got {bot_px}"
        );
    }

    #[test]
    fn radial_subtract_brush_and_ai_matte_compose_cleanly() {
        let mut img = Rgb32FImage::from_pixel(32, 32, Rgb([0.4, 0.4, 0.4]));
        // 8x8 AI matte: left half 255, right half 0
        let mut matte = vec![0u8; 8 * 8];
        for y in 0..8 {
            for x in 0..4 {
                matte[y * 8 + x] = 255;
            }
        }
        let encoded = encode_matte_rle_b64(&matte);

        let mut recipe = DevelopRecipe::default();
        recipe.masks.push(MaskLayer {
            id: "m_ai".to_string(),
            name: "Subject minus center".to_string(),
            enabled: true,
            inverted: false,
            opacity: 100,
            components: vec![
                MaskComponent::AiMatte {
                    id: "c_ai".to_string(),
                    op: MaskCombineOp::Add,
                    inverted: false,
                    kind: AiMaskKind::Subject,
                    feather: 25.0,
                    matte_width: 8,
                    matte_height: 8,
                    matte_rle_b64: encoded,
                },
                MaskComponent::Brush {
                    id: "c_br".to_string(),
                    op: MaskCombineOp::Subtract,
                    inverted: false,
                    strokes: vec![BrushStroke {
                        size: 0.18,
                        feather: 0.2,
                        flow: 1.0,
                        points: vec![[0.2, 0.5]],
                    }],
                },
            ],
            adjustments: LocalAdjustments {
                exposure_ev: -1.5,
                ..LocalAdjustments::default()
            },
        });

        apply_local_masks_f32(&mut img, &recipe);
        let top_left = img.get_pixel(4, 2).0[0];
        let erased_spot = img.get_pixel(6, 16).0[0];
        let right_side = img.get_pixel(28, 16).0[0];

        assert!(top_left < 0.28, "AI matte area should darken: {top_left}");
        assert!(
            (erased_spot - 0.4).abs() < 0.03,
            "subtracted brush spot should stay ~0.4: {erased_spot}"
        );
        assert!(
            (right_side - 0.4).abs() < 0.02,
            "outside matte should stay 0.4: {right_side}"
        );
    }

    #[test]
    fn luminance_and_color_range_isolate_target_pixels() {
        assert!(eval_luminance_range([0.85, 0.85, 0.85], 0.6, 1.0, 0.1) > 0.9);
        assert!(eval_luminance_range([0.1, 0.1, 0.1], 0.6, 1.0, 0.1) < 0.05);

        // Red pixel should match red target hue (0 deg) and reject blue pixel
        let red_match = eval_color_range([0.85, 0.12, 0.12], 0.0, 0.75, 0.48, 0.28, 0.3);
        let blue_match = eval_color_range([0.12, 0.20, 0.85], 0.0, 0.75, 0.48, 0.28, 0.3);
        assert!(red_match > 0.85, "expected red match > 0.85, got {red_match}");
        assert!(blue_match < 0.05, "expected blue reject < 0.05, got {blue_match}");
    }

    #[test]
    fn gradient_backdrop_renders_radial_spotlight_and_linear_angle() {
        // Pure background matte (0 everywhere) so backdrop pixels are directly visible
        let matte = vec![0u8; 8 * 8];
        let encoded = encode_matte_rle_b64(&matte);

        let mut img_radial = Rgb32FImage::from_pixel(32, 32, Rgb([0.0, 0.0, 0.0]));
        let mut recipe = DevelopRecipe::default();
        recipe.background_removal.enabled = true;
        recipe.background_removal.mode = BackgroundReplaceMode::Gradient;
        recipe.background_removal.color_rgb = [240, 240, 240];
        recipe.background_removal.color2_rgb = [20, 20, 20];
        recipe.background_removal.gradient_style = BackgroundGradientStyle::Radial;
        recipe.background_removal.matte_width = 8;
        recipe.background_removal.matte_height = 8;
        recipe.background_removal.matte_rle_b64 = encoded.clone();

        apply_opaque_backdrop_f32(&mut img_radial, &recipe);
        assert!(compute_transparent_alpha_f32(&img_radial, &recipe).is_none());
        let center_px = img_radial.get_pixel(16, 13).0[0];
        let corner_px = img_radial.get_pixel(0, 31).0[0];
        assert!(
            center_px > 0.85 && corner_px < 0.25,
            "radial spot should brighten center ({center_px}) and darken corners ({corner_px})"
        );

        let mut img_linear = Rgb32FImage::from_pixel(32, 32, Rgb([0.0, 0.0, 0.0]));
        recipe.background_removal.gradient_style = BackgroundGradientStyle::Linear;
        recipe.background_removal.gradient_angle_deg = 180.0; // top -> bottom
        apply_opaque_backdrop_f32(&mut img_linear, &recipe);
        let top_px = img_linear.get_pixel(16, 0).0[0];
        let bottom_px = img_linear.get_pixel(16, 31).0[0];
        assert!(
            top_px > 0.85 && bottom_px < 0.15,
            "180deg linear backdrop should transition top ({top_px}) -> bottom ({bottom_px})"
        );

        // Local AI Background mask applied AFTER opaque backdrop must visibly adjust the gradient!
        let bg_matte = vec![255u8; 8 * 8];
        recipe.masks.push(MaskLayer {
            id: "m_bg".to_string(),
            name: "AI Background".to_string(),
            enabled: true,
            inverted: false,
            opacity: 100,
            components: vec![MaskComponent::AiMatte {
                id: "c_bg".to_string(),
                op: MaskCombineOp::Add,
                inverted: false,
                kind: AiMaskKind::Background,
                feather: 25.0,
                matte_width: 8,
                matte_height: 8,
                matte_rle_b64: encode_matte_rle_b64(&bg_matte),
            }],
            adjustments: LocalAdjustments {
                exposure_ev: -2.0,
                ..LocalAdjustments::default()
            },
        });
        apply_local_masks_f32(&mut img_linear, &recipe);
        let darkened_top = img_linear.get_pixel(16, 0).0[0];
        assert!(
            darkened_top < 0.55,
            "AI Background mask should darken replaced gradient backdrop (was {top_px}, now {darkened_top})"
        );
    }
}
