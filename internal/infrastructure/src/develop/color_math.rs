use crate::develop::cube_lut::CubeLut3D;
use kroma_domain::develop::{
    BasicTone, CameraCalibration, ColorGradingWheel, DetailAndOptics, DevelopRecipe, HslBandShift,
    HslColorMixer, ParametricToneCurve, WhiteBalance, NEUTRAL_KELVIN,
};

/// Precomputed per-frame grading constants so the hot pixel loop performs zero redundant math.
pub struct CompiledDevelopGrade<'a> {
    linear_stage_active: bool,
    exposure_gain: f32,
    wb_gains: [f32; 3],
    cal_active: bool,
    cal_shadows_tint: f32,
    cal_matrix: [[f32; 3]; 3],
    basic: BasicTone,
    tone_luts: Option<([f32; 256], [f32; 256], [f32; 256])>,
    hsl: HslColorMixer,
    hsl_active: bool,
    color_grading_active: bool,
    cg_shd_tint: [f32; 3],
    cg_shd_lum: f32,
    cg_mid_tint: [f32; 3],
    cg_mid_lum: f32,
    cg_hi_tint: [f32; 3],
    cg_hi_lum: f32,
    cg_pivot: f32,
    cg_spread: f32,
    lut: Option<(&'a CubeLut3D, f32)>,
    optics: DetailAndOptics,
}

impl<'a> CompiledDevelopGrade<'a> {
    pub fn compile(recipe: &DevelopRecipe, lut: Option<&'a CubeLut3D>) -> Self {
        let exposure_ev = recipe.basic_tone.exposure_ev.clamp(-5.0, 5.0);
        let linear_stage_active = exposure_ev.abs() > 1e-4
            || recipe.white_balance.temperature_kelvin != NEUTRAL_KELVIN
            || recipe.white_balance.tint != 0;
        let exposure_gain = 2.0_f32.powf(exposure_ev);
        let wb_gains = compute_white_balance_gains(recipe.white_balance);
        let cal_active = !recipe.camera_calibration.is_neutral();
        let cal_matrix = compute_calibration_primary_matrix(&recipe.camera_calibration);
        let lut_binding = lut.zip(recipe.lut_3d.as_ref().map(|r| r.intensity));
        let tone_luts = if recipe.tone_curve.is_neutral() {
            None
        } else {
            Some(build_rgb_tone_curve_luts(&recipe.tone_curve))
        };
        let cg = &recipe.color_grading;
        let color_grading_active = !cg.is_neutral();
        let (cg_shd_tint, cg_shd_lum) = compile_color_wheel(cg.shadows);
        let (cg_mid_tint, cg_mid_lum) = compile_color_wheel(cg.midtones);
        let (cg_hi_tint, cg_hi_lum) = compile_color_wheel(cg.highlights);
        let balance_shift = (cg.balance as f32 / 100.0) * 0.25;
        let cg_pivot = (0.5 - balance_shift).clamp(0.18, 0.82);
        let blend = (cg.blending as f32 / 100.0).clamp(0.0, 1.0);
        let cg_spread = 0.14 + blend * 0.36;

        Self {
            linear_stage_active,
            exposure_gain,
            wb_gains,
            cal_active,
            cal_shadows_tint: recipe.camera_calibration.shadows_tint as f32 / 100.0,
            cal_matrix,
            basic: recipe.basic_tone,
            tone_luts,
            hsl: recipe.hsl_mixer,
            hsl_active: !recipe.hsl_mixer.is_neutral(),
            color_grading_active,
            cg_shd_tint,
            cg_shd_lum,
            cg_mid_tint,
            cg_mid_lum,
            cg_hi_tint,
            cg_hi_lum,
            cg_pivot,
            cg_spread,
            lut: lut_binding,
            optics: recipe.detail_optics,
        }
    }

    /// Evaluates the full Lightroom-style parametric pipeline for one sRGB pixel `(r, g, b)` in `[0, 1]`
    /// at normalized coordinates `(nx, ny)` in `[-1.0, 1.0]` relative to image center (for vignette).
    #[inline]
    pub fn grade_pixel(&self, srgb: [f32; 3], nx: f32, ny: f32) -> [f32; 3] {
        self.grade_pixel_in_space(srgb, false, nx, ny)
    }

    /// Evaluates the full parametric pipeline for either:
    /// - `input_is_linear == true`: unclipped scene-linear float RGB directly from 14-bit Camera RAW
    ///   demosaicing (retaining `> 1.0` highlight headroom for true Exposure & Highlight recovery), or
    /// - `input_is_linear == false`: sRGB float RGB in `[0, 1]` (PNG/JPEG/WebP/16-bit TIFF).
    #[inline]
    pub fn grade_pixel_in_space(
        &self,
        pixel: [f32; 3],
        input_is_linear: bool,
        nx: f32,
        ny: f32,
    ) -> [f32; 3] {
        self.grade_pixel_in_color_space(pixel, input_is_linear, false, nx, ny)
    }

    /// Evaluates the full parametric pipeline with explicit scene-referred wide-gamut (`Linear Rec.2020`)
    /// support: when `input_is_linear && input_is_wide_gamut`, White Balance, Exposure EV, and HDR
    /// Highlight Recovery operate in 32-bit Linear Rec.2020 followed by hue-preserving smooth gamut
    /// compression (`compress_rec2020_to_linear_srgb`) before display soft-clipping and sRGB encoding.
    #[inline]
    pub fn grade_pixel_in_color_space(
        &self,
        pixel: [f32; 3],
        input_is_linear: bool,
        input_is_wide_gamut: bool,
        nx: f32,
        ny: f32,
    ) -> [f32; 3] {
        // 1-2. Linear light stage: White Balance + Exposure EV + HDR Highlight Recovery + Gamut Compression + Soft Clip
        let mut rgb = if input_is_linear {
            let mut lin = if self.linear_stage_active {
                [
                    (pixel[0] * self.wb_gains[0] * self.exposure_gain).max(0.0),
                    (pixel[1] * self.wb_gains[1] * self.exposure_gain).max(0.0),
                    (pixel[2] * self.wb_gains[2] * self.exposure_gain).max(0.0),
                ]
            } else {
                [pixel[0].max(0.0), pixel[1].max(0.0), pixel[2].max(0.0)]
            };

            if self.basic.highlights < 0 || self.basic.whites < 0 {
                lin = recover_hdr_highlights(lin, self.basic.highlights, self.basic.whites);
            }

            if input_is_wide_gamut {
                lin = compress_rec2020_to_linear_srgb(lin);
            }

            [
                linear_to_srgb(soft_clip(lin[0])),
                linear_to_srgb(soft_clip(lin[1])),
                linear_to_srgb(soft_clip(lin[2])),
            ]
        } else if self.linear_stage_active {
            // Convert sRGB input to linear light for physically accurate exposure & white balance
            // (skipped when neutral so untouched sRGB highlights remain bit-exact)
            let mut lin = [
                srgb_to_linear(pixel[0]),
                srgb_to_linear(pixel[1]),
                srgb_to_linear(pixel[2]),
            ];
            lin[0] = (lin[0] * self.wb_gains[0] * self.exposure_gain).max(0.0);
            lin[1] = (lin[1] * self.wb_gains[1] * self.exposure_gain).max(0.0);
            lin[2] = (lin[2] * self.wb_gains[2] * self.exposure_gain).max(0.0);

            // Soft highlight roll-off above 0.92 linear to avoid harsh digital clipping
            [
                linear_to_srgb(soft_clip(lin[0])),
                linear_to_srgb(soft_clip(lin[1])),
                linear_to_srgb(soft_clip(lin[2])),
            ]
        } else {
            [
                pixel[0].clamp(0.0, 1.0),
                pixel[1].clamp(0.0, 1.0),
                pixel[2].clamp(0.0, 1.0),
            ]
        };

        // 2b. Camera Calibration (Shadows Tint + neutral-preserving R/G/B Primary Matrix)
        if self.cal_active {
            rgb = apply_camera_calibration(rgb, self.cal_shadows_tint, &self.cal_matrix);
        }

        // 3. Basic Tone (Highlights, Shadows, Whites, Blacks, Contrast, Dehaze)
        rgb = apply_basic_tone(rgb, &self.basic);

        // 4. Combined Point + Parametric + Per-Channel RGB Tone Curves (O(1) 256-entry 1D LUTs)
        if let Some((ref r_lut, ref g_lut, ref b_lut)) = self.tone_luts {
            rgb = [
                lookup_tone_lut(r_lut, rgb[0]),
                lookup_tone_lut(g_lut, rgb[1]),
                lookup_tone_lut(b_lut, rgb[2]),
            ];
        }

        // 5-6. Presence + HSL Color Mixer OR Perceptual B&W Color Mix
        if self.basic.black_and_white {
            rgb = apply_black_and_white_mix(rgb, &self.hsl, self.hsl_active);
        } else {
            if self.basic.vibrance != 0 || self.basic.saturation != 0 {
                rgb = apply_vibrance_and_saturation(rgb, self.basic.vibrance, self.basic.saturation);
            }
            if self.hsl_active {
                rgb = apply_hsl_mixer(rgb, &self.hsl);
            }
        }

        // 6b. 3-Way Color Grading (Shadows / Midtones / Highlights + Blending & Balance)
        // Applied after B&W conversion so split-toning monochrome portraits works just like Lightroom.
        if self.color_grading_active {
            rgb = apply_color_grading(
                rgb,
                self.cg_shd_tint,
                self.cg_shd_lum,
                self.cg_mid_tint,
                self.cg_mid_lum,
                self.cg_hi_tint,
                self.cg_hi_lum,
                self.cg_pivot,
                self.cg_spread,
            );
        }

        // 7. 3D .cube LUT (trilinear lookup + blend intensity)
        if let Some((lut, intensity)) = self.lut {
            rgb = lut.apply(rgb, intensity);
        }

        // 8. Post-crop radial vignette (Amount + Midpoint + Feather)
        if self.optics.vignette_amount != 0 {
            rgb = apply_vignette(
                rgb,
                nx,
                ny,
                self.optics.vignette_amount,
                self.optics.vignette_midpoint,
                self.optics.vignette_feather,
            );
        }

        [
            rgb[0].clamp(0.0, 1.0),
            rgb[1].clamp(0.0, 1.0),
            rgb[2].clamp(0.0, 1.0),
        ]
    }
}

#[inline]
pub fn srgb_to_linear(c: f32) -> f32 {
    let x = c.clamp(0.0, 1.0);
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
pub fn linear_to_srgb(c: f32) -> f32 {
    let x = c.clamp(0.0, 1.0);
    if x <= 0.0031308 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

#[inline]
pub(crate) fn soft_clip(x: f32) -> f32 {
    if x <= 0.92 {
        x.max(0.0)
    } else {
        let excess = x - 0.92;
        0.92 + 0.08 * (1.0 - (-excess / 0.08).exp())
    }
}

/// Exact ITU-R BT.709 (Linear sRGB) → ITU-R BT.2020 (Linear Rec.2020, D65) 3×3 CIE matrix.
/// Row sums equal `1.0000000` so neutral grays `[v, v, v]` are strictly invariant.
#[inline]
pub fn rec709_to_rec2020(rgb: [f32; 3]) -> [f32; 3] {
    [
        0.627_403_9 * rgb[0] + 0.329_283 * rgb[1] + 0.043_313_1 * rgb[2],
        0.069_097_3 * rgb[0] + 0.919_540_4 * rgb[1] + 0.011_362_3 * rgb[2],
        0.016_391_4 * rgb[0] + 0.088_013_3 * rgb[1] + 0.895_595_3 * rgb[2],
    ]
}

/// Exact ITU-R BT.2020 (Linear Rec.2020) → ITU-R BT.709 (Linear sRGB, D65) 3×3 CIE matrix.
#[inline]
pub fn rec2020_to_rec709(rgb: [f32; 3]) -> [f32; 3] {
    [
        1.660_491 * rgb[0] - 0.587_641_1 * rgb[1] - 0.072_849_9 * rgb[2],
        -0.124_550_5 * rgb[0] + 1.132_899_9 * rgb[1] - 0.008_349_4 * rgb[2],
        -0.018_150_8 * rgb[0] - 0.100_578_9 * rgb[1] + 1.118_729_7 * rgb[2],
    ]
}

/// Hue- and luminance-preserving smooth gamut compression from Linear Rec.2020 to Linear sRGB.
/// Leaves neutral grays, skin tones, and all colors inside 85% linear sRGB saturation untouched,
/// while smoothly compressing extreme / out-of-sRGB-gamut camera colors without hue shift or banding.
#[inline]
pub fn compress_rec2020_to_linear_srgb(lin2020: [f32; 3]) -> [f32; 3] {
    let srgb = rec2020_to_rec709(lin2020);
    let y = (0.2126 * srgb[0] + 0.7152 * srgb[1] + 0.0722 * srgb[2]).max(0.0);
    if y <= 1e-6 {
        return [0.0, 0.0, 0.0];
    }
    let min_c = srgb[0].min(srgb[1]).min(srgb[2]);
    let d = (y - min_c) / y;
    let threshold = 0.85_f32;
    if d <= threshold {
        return [srgb[0].max(0.0), srgb[1].max(0.0), srgb[2].max(0.0)];
    }
    let excess = d - threshold;
    let limit = 1.0 - threshold;
    let d_compressed = threshold + limit * (excess / (excess + limit));
    let scale = d_compressed / d;
    [
        (y + (srgb[0] - y) * scale).max(0.0),
        (y + (srgb[1] - y) * scale).max(0.0),
        (y + (srgb[2] - y) * scale).max(0.0),
    ]
}

/// Linear-light HDR highlight recovery applied to unclipped Camera RAW float buffers before
/// display soft-clipping, preserving exact RGB hue ratios while pulling `> 1.0` highlights cleanly
/// back into the displayable range when Highlights or Whites are negative.
#[inline]
fn recover_hdr_highlights(lin: [f32; 3], highlights: i16, whites: i16) -> [f32; 3] {
    let peak = lin[0].max(lin[1]).max(lin[2]);
    if peak <= 0.65 {
        return lin;
    }
    let hl_pull = if highlights < 0 {
        (-highlights as f32) / 100.0
    } else {
        0.0
    };
    let wh_pull = if whites < 0 {
        (-whites as f32) / 100.0
    } else {
        0.0
    };
    let pull = (hl_pull * 0.65 + wh_pull * 0.35).min(1.0);
    if pull <= 1e-5 {
        return lin;
    }
    let knee = 0.65;
    let excess = peak - knee;
    let weight = smoothstep(0.65, 1.35, peak);
    let target_peak = knee + excess / (1.0 + pull * 2.4 * weight);
    let scale = target_peak / peak.max(1e-5);
    [lin[0] * scale, lin[1] * scale, lin[2] * scale]
}

/// Computes normalized RGB linear multipliers for Kelvin temperature & green/magenta tint.
pub(crate) fn compute_white_balance_gains(wb: WhiteBalance) -> [f32; 3] {
    // Logarithmic Kelvin ratio keeps steps perceptually uniform across 2000K..50000K
    let kelvin_shift = ((wb.temperature_kelvin as f32) / (NEUTRAL_KELVIN as f32)).ln();
    let warm_strength = kelvin_shift.clamp(-1.0, 1.2);
    let r_gain = (1.0 + warm_strength * 0.42).clamp(0.45, 2.2);
    let b_gain = (1.0 - warm_strength * 0.42).clamp(0.45, 2.2);

    // Positive tint adds magenta (reduces green); negative tint adds green
    let g_gain = (1.0 - (wb.tint as f32) / 350.0).clamp(0.55, 1.65);

    // Normalize by Rec.709 luminance so WB changes don't shift apparent exposure
    let luma = 0.2126 * r_gain + 0.7152 * g_gain + 0.0722 * b_gain;
    let inv = 1.0 / luma.max(1e-4);
    [r_gain * inv, g_gain * inv, b_gain * inv]
}

#[inline]
fn rec709_luma(rgb: [f32; 3]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

fn apply_basic_tone(mut rgb: [f32; 3], tone: &BasicTone) -> [f32; 3] {
    let luma = rec709_luma(rgb);

    // Highlights & Shadows selective tonal lift/recovery
    let mut delta = 0.0_f32;
    if tone.shadows != 0 {
        let shadow_weight = (1.0 - smoothstep(0.05, 0.55, luma)).powi(2);
        delta += (tone.shadows as f32 / 100.0) * 0.28 * shadow_weight;
    }
    if tone.highlights != 0 {
        let highlight_weight = smoothstep(0.45, 0.95, luma).powi(2);
        delta += (tone.highlights as f32 / 100.0) * 0.28 * highlight_weight;
    }

    // Whites & Blacks endpoint shifts
    if tone.whites != 0 {
        let white_weight = smoothstep(0.65, 1.0, luma);
        delta += (tone.whites as f32 / 100.0) * 0.22 * white_weight;
    }
    if tone.blacks != 0 {
        let black_weight = 1.0 - smoothstep(0.0, 0.35, luma);
        delta += (tone.blacks as f32 / 100.0) * 0.22 * black_weight;
    }

    if delta.abs() > 1e-6 {
        let target_luma = (luma + delta).clamp(0.0, 1.0);
        if luma > 1e-4 {
            let ratio = target_luma / luma;
            rgb = [
                (rgb[0] * ratio).clamp(0.0, 1.0),
                (rgb[1] * ratio).clamp(0.0, 1.0),
                (rgb[2] * ratio).clamp(0.0, 1.0),
            ];
        } else {
            rgb = [target_luma, target_luma, target_luma];
        }
    }

    // Dehaze (mild shadow anchoring + local midtone expansion)
    if tone.dehaze != 0 {
        let dh = tone.dehaze as f32 / 100.0;
        let lift = -dh * 0.06;
        let scale = 1.0 + dh * 0.18;
        for ch in &mut rgb {
            *ch = ((*ch + lift) * scale).clamp(0.0, 1.0);
        }
    }

    // Contrast pivot around perceptual mid-gray (0.5)
    if tone.contrast != 0 {
        let c = tone.contrast as f32 / 100.0;
        let factor = if c >= 0.0 {
            1.0 + c * 0.85
        } else {
            1.0 + c * 0.65
        };
        for ch in &mut rgb {
            *ch = ((*ch - 0.5) * factor + 0.5).clamp(0.0, 1.0);
        }
    }

    rgb
}

pub(crate) fn build_rgb_tone_curve_luts(
    curve: &ParametricToneCurve,
) -> ([f32; 256], [f32; 256], [f32; 256]) {
    let mut r_lut = [0.0_f32; 256];
    let mut g_lut = [0.0_f32; 256];
    let mut b_lut = [0.0_f32; 256];
    for idx in 0..256 {
        let x = idx as f32 / 255.0;
        let mut master = curve.evaluate_point_curve(x);
        if curve.shadows != 0 {
            master += (curve.shadows as f32 / 100.0) * 0.16 * gaussian(x, 0.14, 0.14);
        }
        if curve.darks != 0 {
            master += (curve.darks as f32 / 100.0) * 0.16 * gaussian(x, 0.36, 0.18);
        }
        if curve.lights != 0 {
            master += (curve.lights as f32 / 100.0) * 0.16 * gaussian(x, 0.64, 0.18);
        }
        if curve.highlights != 0 {
            master += (curve.highlights as f32 / 100.0) * 0.16 * gaussian(x, 0.86, 0.14);
        }
        let m = master.clamp(0.0, 1.0);
        r_lut[idx] = ParametricToneCurve::evaluate_points_slice(&curve.red_points, m);
        g_lut[idx] = ParametricToneCurve::evaluate_points_slice(&curve.green_points, m);
        b_lut[idx] = ParametricToneCurve::evaluate_points_slice(&curve.blue_points, m);
    }
    (r_lut, g_lut, b_lut)
}

/// Builds a row-sum-1 3×3 Camera Calibration primary matrix so neutral grays/whites stay exact
/// while Red/Green/Blue Primary Hue & Saturation sliders shift color primaries cleanly.
pub fn compute_calibration_primary_matrix(cal: &CameraCalibration) -> [[f32; 3]; 3] {
    let hr = (cal.red_hue as f32 / 100.0) * 0.32;
    let hg = (cal.green_hue as f32 / 100.0) * 0.32;
    let hb = (cal.blue_hue as f32 / 100.0) * 0.32;

    let sr = (cal.red_saturation as f32 / 100.0) * 0.42;
    let sg = (cal.green_saturation as f32 / 100.0) * 0.42;
    let sb = (cal.blue_saturation as f32 / 100.0) * 0.42;

    // Columns represent where the R, G, and B camera primaries map in working RGB
    let mut m = [
        [
            1.0 + sr * 0.44,
            -hg * 0.40 - sg * 0.22,
            hb * 0.40 - sb * 0.22,
        ],
        [
            hr * 0.45 - sr * 0.22,
            1.0 + sg * 0.44,
            -hb * 0.45 - sb * 0.22,
        ],
        [
            -hr * 0.35 - sr * 0.22,
            hg * 0.40 - sg * 0.22,
            1.0 + sb * 0.44,
        ],
    ];

    // Row-normalize so M * [v, v, v] == [v, v, v] identically for all slider settings
    for row in &mut m {
        let sum = row[0] + row[1] + row[2];
        let inv = if sum.abs() > 1e-4 { 1.0 / sum } else { 1.0 };
        row[0] *= inv;
        row[1] *= inv;
        row[2] *= inv;
    }
    m
}

fn apply_camera_calibration(
    rgb: [f32; 3],
    shadows_tint: f32,
    matrix: &[[f32; 3]; 3],
) -> [f32; 3] {
    let mut out = [
        (matrix[0][0] * rgb[0] + matrix[0][1] * rgb[1] + matrix[0][2] * rgb[2]).clamp(0.0, 1.0),
        (matrix[1][0] * rgb[0] + matrix[1][1] * rgb[1] + matrix[1][2] * rgb[2]).clamp(0.0, 1.0),
        (matrix[2][0] * rgb[0] + matrix[2][1] * rgb[1] + matrix[2][2] * rgb[2]).clamp(0.0, 1.0),
    ];

    if shadows_tint.abs() > 1e-5 {
        let luma = rec709_luma(out);
        let shadow_w = (1.0 - smoothstep(0.0, 0.55, luma)).powi(2);
        let delta = shadows_tint * 0.08 * shadow_w;
        out[0] = (out[0] + delta * 0.5).clamp(0.0, 1.0);
        out[1] = (out[1] - delta).clamp(0.0, 1.0);
        out[2] = (out[2] + delta * 0.5).clamp(0.0, 1.0);
    }

    out
}

pub(crate) fn compile_color_wheel(wheel: ColorGradingWheel) -> ([f32; 3], f32) {
    let sat = (wheel.saturation as f32 / 100.0).clamp(0.0, 1.0);
    let pure = hsl_to_rgb(wheel.hue_deg, 1.0, 0.5);
    let strength = sat * 0.34;
    let tint = [
        (pure[0] - 0.5) * strength,
        (pure[1] - 0.5) * strength,
        (pure[2] - 0.5) * strength,
    ];
    let lum = (wheel.luminance as f32 / 100.0) * 0.18;
    (tint, lum)
}

#[allow(clippy::too_many_arguments)]
fn apply_color_grading(
    rgb: [f32; 3],
    shd_tint: [f32; 3],
    shd_lum: f32,
    mid_tint: [f32; 3],
    mid_lum: f32,
    hi_tint: [f32; 3],
    hi_lum: f32,
    pivot: f32,
    spread: f32,
) -> [f32; 3] {
    let luma = rec709_luma(rgb);
    let mut shd_w = (1.0 - smoothstep(pivot - spread, pivot + spread * 0.65, luma)).powf(1.5);
    let mut hi_w = smoothstep(pivot - spread * 0.65, pivot + spread, luma).powf(1.5);
    let mut mid_w = (1.0 - shd_w - hi_w).max(0.0);
    let sum = (shd_w + mid_w + hi_w).max(1.0);
    shd_w /= sum;
    mid_w /= sum;
    hi_w /= sum;

    let shd_env = 1.0 - smoothstep(0.75, 1.0, luma);
    let mid_env = smoothstep(0.0, 0.12, luma) * (1.0 - smoothstep(0.88, 1.0, luma));
    let hi_env = smoothstep(0.05, 0.30, luma);

    let lum_shift = shd_w * shd_lum + mid_w * mid_lum + hi_w * hi_lum;
    [
        (rgb[0]
            + shd_tint[0] * shd_w * shd_env
            + mid_tint[0] * mid_w * mid_env
            + hi_tint[0] * hi_w * hi_env
            + lum_shift)
            .clamp(0.0, 1.0),
        (rgb[1]
            + shd_tint[1] * shd_w * shd_env
            + mid_tint[1] * mid_w * mid_env
            + hi_tint[1] * hi_w * hi_env
            + lum_shift)
            .clamp(0.0, 1.0),
        (rgb[2]
            + shd_tint[2] * shd_w * shd_env
            + mid_tint[2] * mid_w * mid_env
            + hi_tint[2] * hi_w * hi_env
            + lum_shift)
            .clamp(0.0, 1.0),
    ]
}

fn apply_black_and_white_mix(rgb: [f32; 3], mixer: &HslColorMixer, hsl_active: bool) -> [f32; 3] {
    let base_luma = rec709_luma(rgb);
    if !hsl_active {
        return [base_luma, base_luma, base_luma];
    }
    let (h, s, _) = rgb_to_hsl(rgb);
    if s < 0.015 {
        return [base_luma, base_luma, base_luma];
    }
    const BANDS: [(f32, f32); 8] = [
        (0.0, 30.0),
        (30.0, 30.0),
        (60.0, 35.0),
        (120.0, 50.0),
        (180.0, 50.0),
        (240.0, 45.0),
        (285.0, 35.0),
        (325.0, 35.0),
    ];
    let shifts: [HslBandShift; 8] = [
        mixer.red,
        mixer.orange,
        mixer.yellow,
        mixer.green,
        mixer.aqua,
        mixer.blue,
        mixer.purple,
        mixer.magenta,
    ];
    let mut lum_delta = 0.0_f32;
    for ((center, half_width), shift) in BANDS.iter().zip(shifts.iter()) {
        if shift.luminance == 0 {
            continue;
        }
        let dist = hue_distance_deg(h, *center);
        if dist < *half_width {
            let weight = 0.5 * (1.0 + (std::f32::consts::PI * dist / *half_width).cos());
            lum_delta += (shift.luminance as f32 / 100.0) * 0.45 * weight;
        }
    }
    let bw = (base_luma + lum_delta * s).clamp(0.0, 1.0);
    [bw, bw, bw]
}

#[inline]
fn lookup_tone_lut(lut: &[f32; 256], value: f32) -> f32 {
    let scaled = value.clamp(0.0, 1.0) * 255.0;
    let lo = (scaled as usize).min(255);
    let hi = (lo + 1).min(255);
    let frac = scaled - lo as f32;
    lut[lo] + (lut[hi] - lut[lo]) * frac
}

fn apply_vibrance_and_saturation(rgb: [f32; 3], vibrance: i16, saturation: i16) -> [f32; 3] {
    let (h, mut s, l) = rgb_to_hsl(rgb);

    if vibrance != 0 {
        let vib = vibrance as f32 / 100.0;
        // Portrait skin-tone protection: attenuate positive vibrance around orange/peach hues (15..45 deg)
        let skin_proximity = gaussian(hue_distance_deg(h, 28.0), 0.0, 18.0);
        let skin_guard = if vib > 0.0 {
            1.0 - 0.6 * skin_proximity
        } else {
            1.0
        };
        // Less-saturated pixels receive a larger boost than already-vivid pixels
        let sat_guard = 1.0 - s;
        s = (s + vib * 0.55 * sat_guard * skin_guard).clamp(0.0, 1.0);
    }

    if saturation != 0 {
        let sat_scale = 1.0 + (saturation as f32 / 100.0);
        s = (s * sat_scale).clamp(0.0, 1.0);
    }

    hsl_to_rgb(h, s, l)
}

fn apply_hsl_mixer(rgb: [f32; 3], mixer: &HslColorMixer) -> [f32; 3] {
    let (h, s, l) = rgb_to_hsl(rgb);
    if s < 0.015 {
        return rgb;
    }

    const BANDS: [(f32, f32); 8] = [
        (0.0, 30.0),   // Red
        (30.0, 30.0),  // Orange
        (60.0, 35.0),  // Yellow
        (120.0, 50.0), // Green
        (180.0, 50.0), // Aqua
        (240.0, 45.0), // Blue
        (285.0, 35.0), // Purple
        (325.0, 35.0), // Magenta
    ];
    let shifts: [HslBandShift; 8] = [
        mixer.red,
        mixer.orange,
        mixer.yellow,
        mixer.green,
        mixer.aqua,
        mixer.blue,
        mixer.purple,
        mixer.magenta,
    ];

    let mut hue_delta_deg = 0.0_f32;
    let mut sat_delta = 0.0_f32;
    let mut lum_delta = 0.0_f32;

    for ((center, half_width), shift) in BANDS.iter().zip(shifts.iter()) {
        if shift.is_neutral() {
            continue;
        }
        let dist = hue_distance_deg(h, *center);
        if dist < *half_width {
            let weight = 0.5 * (1.0 + (std::f32::consts::PI * dist / *half_width).cos());
            hue_delta_deg += (shift.hue as f32 / 100.0) * 28.0 * weight;
            sat_delta += (shift.saturation as f32 / 100.0) * weight;
            lum_delta += (shift.luminance as f32 / 100.0) * 0.25 * weight;
        }
    }

    let next_h = (h + hue_delta_deg).rem_euclid(360.0);
    let next_s = (s * (1.0 + sat_delta)).clamp(0.0, 1.0);
    let next_l = (l + lum_delta * s).clamp(0.0, 1.0);
    hsl_to_rgb(next_h, next_s, next_l)
}

fn apply_vignette(
    rgb: [f32; 3],
    nx: f32,
    ny: f32,
    amount: i16,
    midpoint: u8,
    feather: u8,
) -> [f32; 3] {
    let radius = (nx * nx + ny * ny).sqrt();
    let mid = (midpoint as f32 / 100.0).clamp(0.05, 0.98) * 1.15;
    let f = (feather as f32 / 100.0).clamp(0.02, 1.0);
    let inner = mid * (1.0 - 0.65 * f);
    let outer = (mid + 0.12 + 0.72 * f).min(1.65);
    let mask = smoothstep(inner, outer, radius);
    let strength = (amount as f32 / 100.0) * 0.75 * mask;
    let factor = (1.0 + strength).clamp(0.15, 1.85);
    [
        (rgb[0] * factor).clamp(0.0, 1.0),
        (rgb[1] * factor).clamp(0.0, 1.0),
        (rgb[2] * factor).clamp(0.0, 1.0),
    ]
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn gaussian(x: f32, mean: f32, sigma: f32) -> f32 {
    let d = (x - mean) / sigma.max(1e-5);
    (-0.5 * d * d).exp()
}

#[inline]
fn hue_distance_deg(a: f32, b: f32) -> f32 {
    let diff = (a - b).abs() % 360.0;
    if diff > 180.0 {
        360.0 - diff
    } else {
        diff
    }
}

pub fn rgb_to_hsl(rgb: [f32; 3]) -> (f32, f32, f32) {
    let r = rgb[0].clamp(0.0, 1.0);
    let g = rgb[1].clamp(0.0, 1.0);
    let b = rgb[2].clamp(0.0, 1.0);

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = 0.5 * (max + min);
    let d = max - min;

    if d <= 1e-6 {
        return (0.0, 0.0, l);
    }

    let s = if l > 0.5 {
        d / (2.0 - max - min).max(1e-6)
    } else {
        d / (max + min).max(1e-6)
    };

    let mut h = if (max - r).abs() <= 1e-6 {
        (g - b) / d + if g < b { 6.0 } else { 0.0 }
    } else if (max - g).abs() <= 1e-6 {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    h *= 60.0;

    (h.rem_euclid(360.0), s.clamp(0.0, 1.0), l.clamp(0.0, 1.0))
}

pub fn hsl_to_rgb(h: f32, s: f32, l: f32) -> [f32; 3] {
    if s <= 1e-6 {
        return [l, l, l];
    }

    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let hk = h.rem_euclid(360.0) / 360.0;

    let hue_to_rgb = |mut t: f32| -> f32 {
        if t < 0.0 {
            t += 1.0;
        }
        if t > 1.0 {
            t -= 1.0;
        }
        if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        }
    };

    [
        hue_to_rgb(hk + 1.0 / 3.0),
        hue_to_rgb(hk),
        hue_to_rgb(hk - 1.0 / 3.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_recipe_preserves_pixel_value() {
        let recipe = DevelopRecipe::default();
        let grade = CompiledDevelopGrade::compile(&recipe, None);
        let input = [0.42, 0.55, 0.68];
        let output = grade.grade_pixel(input, 0.0, 0.0);

        for i in 0..3 {
            assert!((output[i] - input[i]).abs() < 1e-3);
        }
    }

    #[test]
    fn positive_exposure_brightens_and_warm_kelvin_increases_red_over_blue() {
        let mut recipe = DevelopRecipe::default();
        recipe.basic_tone.exposure_ev = 1.0;
        recipe.white_balance.temperature_kelvin = 7500;

        let grade = CompiledDevelopGrade::compile(&recipe, None);
        let out = grade.grade_pixel([0.4, 0.4, 0.4], 0.0, 0.0);

        assert!(out[0] > 0.55);
        assert!(out[0] > out[2]);
    }

    #[test]
    fn hsl_roundtrip_is_accurate() {
        let original = [0.82, 0.44, 0.29];
        let (h, s, l) = rgb_to_hsl(original);
        let back = hsl_to_rgb(h, s, l);
        for i in 0..3 {
            assert!((original[i] - back[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn point_tone_curve_applies_matte_black_lift_and_s_curve_contrast() {
        use kroma_domain::develop::ToneCurvePoint;

        let mut recipe = DevelopRecipe::default();
        recipe.tone_curve.points = vec![
            ToneCurvePoint { x: 0.0, y: 0.12 },
            ToneCurvePoint { x: 0.25, y: 0.22 },
            ToneCurvePoint { x: 0.75, y: 0.84 },
            ToneCurvePoint { x: 1.0, y: 0.96 },
        ];
        let grade = CompiledDevelopGrade::compile(&recipe, None);

        let black = grade.grade_pixel([0.0, 0.0, 0.0], 0.0, 0.0);
        let high = grade.grade_pixel([0.75, 0.75, 0.75], 0.0, 0.0);
        let white = grade.grade_pixel([1.0, 1.0, 1.0], 0.0, 0.0);

        assert!((black[0] - 0.12).abs() < 0.015, "got {}", black[0]);
        assert!((high[0] - 0.84).abs() < 0.02, "got {}", high[0]);
        assert!((white[0] - 0.96).abs() < 0.015, "got {}", white[0]);
    }

    #[test]
    fn linear_raw_pipeline_recovers_hdr_highlights_above_one() {
        let mut pulled_ev = DevelopRecipe::default();
        pulled_ev.basic_tone.exposure_ev = -1.5;
        let grade_ev = CompiledDevelopGrade::compile(&pulled_ev, None);

        // Two distinct overexposed RAW pixels above 1.0 linear (1.5 vs 2.4):
        // On an 8-bit sRGB image both would clip identically to 1.0; in linear RAW they separate cleanly!
        let lo = grade_ev.grade_pixel_in_space([1.5, 1.5, 1.5], true, 0.0, 0.0);
        let hi = grade_ev.grade_pixel_in_space([2.4, 2.4, 2.4], true, 0.0, 0.0);
        assert!(lo[0] < 0.85, "expected recovered linear highlight, got {}", lo[0]);
        assert!(hi[0] > lo[0] + 0.08, "distinct HDR highlights must stay separated: lo={} hi={}", lo[0], hi[0]);

        let mut pulled_hl = DevelopRecipe::default();
        pulled_hl.basic_tone.highlights = -100;
        let grade_hl = CompiledDevelopGrade::compile(&pulled_hl, None);
        let neutral = CompiledDevelopGrade::compile(&DevelopRecipe::default(), None);
        let before_hl = neutral.grade_pixel_in_space([1.6, 1.2, 1.0], true, 0.0, 0.0);
        let after_hl = grade_hl.grade_pixel_in_space([1.6, 1.2, 1.0], true, 0.0, 0.0);
        assert!(after_hl[0] < before_hl[0] - 0.10, "HDR highlight recovery should pull peak down");
    }

    #[test]
    fn camera_calibration_preserves_neutral_gray_while_shifting_primaries() {
        let mut recipe = DevelopRecipe::default();
        recipe.camera_calibration.blue_hue = -60;
        recipe.camera_calibration.blue_saturation = 50;
        recipe.camera_calibration.red_hue = 30;
        let grade = CompiledDevelopGrade::compile(&recipe, None);

        // Neutral gray must stay neutral (no tint cast from primary sliders)
        let gray = grade.grade_pixel([0.5, 0.5, 0.5], 0.0, 0.0);
        assert!((gray[0] - gray[1]).abs() < 1e-3);
        assert!((gray[1] - gray[2]).abs() < 1e-3);

        // Pure blue shifted toward negative hue gains teal/green energy and suppresses red
        let blue = grade.grade_pixel([0.1, 0.2, 0.85], 0.0, 0.0);
        assert!(blue[1] > 0.21, "expected teal shift on blue, got {:?}", blue);
        assert!(blue[0] < 0.05, "expected red suppression on teal-shifted blue, got {:?}", blue);
    }

    #[test]
    fn per_channel_curves_bw_mix_and_color_grading_apply_cleanly() {
        use kroma_domain::develop::ToneCurvePoint;

        // 1. Per-channel red lift leaves blue untouched
        let mut curve_recipe = DevelopRecipe::default();
        curve_recipe.tone_curve.red_points = vec![
            ToneCurvePoint { x: 0.0, y: 0.2 },
            ToneCurvePoint { x: 1.0, y: 1.0 },
        ];
        let g_curve = CompiledDevelopGrade::compile(&curve_recipe, None);
        let out_curve = g_curve.grade_pixel([0.0, 0.0, 0.0], 0.0, 0.0);
        assert!((out_curve[0] - 0.2).abs() < 0.02);
        assert!(out_curve[2] < 0.02);

        // 2. B&W Treatment + Split-Tone Color Grading (warm highlights, cool shadows)
        let mut bw_recipe = DevelopRecipe::default();
        bw_recipe.basic_tone.black_and_white = true;
        bw_recipe.color_grading.highlights.hue_deg = 40.0; // Warm golden highlights
        bw_recipe.color_grading.highlights.saturation = 70;
        bw_recipe.color_grading.shadows.hue_deg = 220.0; // Cool blue shadows
        bw_recipe.color_grading.shadows.saturation = 70;
        let g_bw = CompiledDevelopGrade::compile(&bw_recipe, None);

        let dark = g_bw.grade_pixel([0.15, 0.15, 0.15], 0.0, 0.0);
        let bright = g_bw.grade_pixel([0.82, 0.82, 0.82], 0.0, 0.0);
        assert!(dark[2] > dark[0], "shadows should tint cool blue: {:?}", dark);
        assert!(bright[0] > bright[2], "highlights should tint warm gold: {:?}", bright);
    }

    #[test]
    fn rec2020_matrices_are_inverses_and_preserve_neutrals_and_compress_out_of_gamut() {
        // 1. Neutral grays are strictly invariant through Rec.709 <-> Rec.2020 and gamut compression
        let gray = [0.42, 0.42, 0.42];
        let gray2020 = rec709_to_rec2020(gray);
        for c in gray2020 {
            assert!((c - 0.42).abs() < 1e-5);
        }
        let gray_compressed = compress_rec2020_to_linear_srgb(gray2020);
        for c in gray_compressed {
            assert!((c - 0.42).abs() < 1e-5);
        }

        // 2. Roundtrip accuracy on portrait skin tones
        let skin = [0.62, 0.38, 0.26];
        let rt = rec2020_to_rec709(rec709_to_rec2020(skin));
        for i in 0..3 {
            assert!((rt[i] - skin[i]).abs() < 1e-4, "roundtrip mismatch: {rt:?}");
        }
        // Skin tone is well inside 85% saturation threshold -> untouched by gamut compression
        let skin_out = compress_rec2020_to_linear_srgb(rec709_to_rec2020(skin));
        for i in 0..3 {
            assert!((skin_out[i] - skin[i]).abs() < 1e-4, "skin tone shifted: {skin_out:?}");
        }

        // 3. Extreme Rec.2020 primary red (outside sRGB gamut) stays >= 0, preserves luminance, and remains strictly ordered
        let deep_red = [0.9, 0.01, 0.01];
        let deeper_red = [0.9, 0.0, 0.0];
        let raw_srgb = rec2020_to_rec709(deeper_red);
        assert!(raw_srgb[1] < 0.0, "Rec.2020 pure red must be out-of-sRGB-gamut");
        let c1 = compress_rec2020_to_linear_srgb(deep_red);
        let c2 = compress_rec2020_to_linear_srgb(deeper_red);
        for ch in c2 {
            assert!(ch >= 0.0, "compressed sRGB channel must be non-negative: {c2:?}");
        }
        let y_in = 0.2126 * raw_srgb[0] + 0.7152 * raw_srgb[1] + 0.0722 * raw_srgb[2];
        let y_out = 0.2126 * c2[0] + 0.7152 * c2[1] + 0.0722 * c2[2];
        assert!((y_in - y_out).abs() < 1e-4, "luminance must be preserved: in={y_in} out={y_out}");
        let sat1 = (c1[0] - c1[1].min(c1[2])) / c1[0];
        let sat2 = (c2[0] - c2[1].min(c2[2])) / c2[0];
        assert!(sat2 > sat1, "more saturated wide-gamut red stays more vivid than less saturated red: sat1={sat1} sat2={sat2}");
    }
}
