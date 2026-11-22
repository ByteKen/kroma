// Unified 32-Bit Scene-Referred Wide-Gamut (Linear Rec.2020) Color & Tone Compute Pass.
// Shared verbatim between Kroma WebGPU Studio Preview and Rust `wgpu` Batch Export (DX12 / Metal).
// Mirrors `CompiledDevelopGrade::grade_pixel_in_space` + `apply_opaque_backdrop_f32` in Rust.

struct GradeUniforms {
  // (outW, outH, srcW, srcH)
  dims: vec4<f32>,
  // (left, top, width, height) in oriented full-frame [0..1]
  cropRect: vec4<f32>,
  // (flipH, flipV, quarterTurns, rotationRad)
  orient: vec4<f32>,
  // (inputLinear, inputWideGamut, linearActive, exposureGain)
  linearFlags: vec4<f32>,
  // (wbR, wbG, wbB, blackAndWhite)
  wbAndBw: vec4<f32>,
  // (calActive, calShadowsTint, toneCurveActive, hslActive)
  stageFlags: vec4<f32>,
  // Row-major 3x3 camera calibration matrix
  calRow0: vec4<f32>,
  calRow1: vec4<f32>,
  calRow2: vec4<f32>,
  // (shadows, highlights, whites, blacks)
  basicToneA: vec4<f32>,
  // (dehaze, contrast, vibrance, saturation)
  basicToneB: vec4<f32>,
  // 8-band HSL mixer: (hue, saturation, luminance, 0)
  hsl: array<vec4<f32>, 8>,
  // 3-way color grading wheels: (tintR, tintG, tintB, lum)
  cgShd: vec4<f32>,
  cgMid: vec4<f32>,
  cgHi: vec4<f32>,
  // (cgActive, cgPivot, cgSpread, 0)
  cgMeta: vec4<f32>,
  // (amount, midpoint, feather, 0)
  vignette: vec4<f32>,
  // (lutSize, lutIntensity, domainMinR, domainMinG)
  lutMeta: vec4<f32>,
  // (domainMinB, domainMaxR, domainMaxG, domainMaxB)
  lutDomain: vec4<f32>,
  // (r, g, b, bgMode: 0=off/transparent, 2=solid, 3=gradient)
  bgColor1: vec4<f32>,
  // (r, g, b, gradStyle: 0=radial, 1=linear)
  bgColor2: vec4<f32>,
  // (gradAngleDeg, feather, 0, 0)
  bgParams: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: GradeUniforms;
@group(0) @binding(1) var srcTex: texture_2d<f32>;
@group(0) @binding(2) var toneLutTex: texture_2d<f32>;
@group(0) @binding(3) var bgMatteTex: texture_2d<f32>;
@group(0) @binding(4) var<storage, read> lut3dData: array<vec4<f32>>;
@group(0) @binding(5) var dstTex: texture_storage_2d<rgba32float, write>;

const PI: f32 = 3.14159265359;

fn kSmoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
  let t = clamp((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

fn kGaussian(x: f32, mean: f32, sigma: f32) -> f32 {
  let d = (x - mean) / max(sigma, 1e-5);
  return exp(-0.5 * d * d);
}

fn mod360(x: f32) -> f32 {
  return x - 360.0 * floor(x / 360.0);
}

fn hueDistanceDeg(a: f32, b: f32) -> f32 {
  let diff = mod360(abs(a - b));
  return select(diff, 360.0 - diff, diff > 180.0);
}

fn srgbToLinear(c: f32) -> f32 {
  let x = clamp(c, 0.0, 1.0);
  if (x <= 0.04045) {
    return x / 12.92;
  }
  return pow((x + 0.055) / 1.055, 2.4);
}

fn linearToSrgb(c: f32) -> f32 {
  let x = clamp(c, 0.0, 1.0);
  if (x <= 0.0031308) {
    return x * 12.92;
  }
  return 1.055 * pow(max(x, 0.0), 1.0 / 2.4) - 0.055;
}

fn softClip(x: f32) -> f32 {
  if (x <= 0.92) {
    return max(x, 0.0);
  }
  let excess = x - 0.92;
  return 0.92 + 0.08 * (1.0 - exp(-excess / 0.08));
}

fn rec2020ToRec709(c: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
     1.6604910 * c.r - 0.5876411 * c.g - 0.0728499 * c.b,
    -0.1245505 * c.r + 1.1328999 * c.g - 0.0083494 * c.b,
    -0.0181508 * c.r - 0.1005789 * c.g + 1.1187297 * c.b,
  );
}

fn compressRec2020ToLinearSrgb(lin2020: vec3<f32>) -> vec3<f32> {
  let srgb = rec2020ToRec709(lin2020);
  let y = max(0.2126 * srgb.r + 0.7152 * srgb.g + 0.0722 * srgb.b, 0.0);
  if (y <= 1e-6) {
    return vec3<f32>(0.0);
  }
  let minC = min(srgb.r, min(srgb.g, srgb.b));
  let d = (y - minC) / y;
  let threshold = 0.85;
  if (d <= threshold) {
    return max(srgb, vec3<f32>(0.0));
  }
  let excess = d - threshold;
  let limit = 1.0 - threshold;
  let dCompressed = threshold + limit * (excess / (excess + limit));
  let scale = dCompressed / d;
  return max(vec3<f32>(y) + (srgb - vec3<f32>(y)) * scale, vec3<f32>(0.0));
}

fn recoverHdrHighlights(lin: vec3<f32>, highlights: f32, whites: f32) -> vec3<f32> {
  let peak = max(lin.r, max(lin.g, lin.b));
  if (peak <= 0.65) {
    return lin;
  }
  let hlPull = select(0.0, -highlights / 100.0, highlights < 0.0);
  let whPull = select(0.0, -whites / 100.0, whites < 0.0);
  let pull = min(1.0, hlPull * 0.65 + whPull * 0.35);
  if (pull <= 1e-5) {
    return lin;
  }
  let knee = 0.65;
  let excess = peak - knee;
  let weight = kSmoothstep(0.65, 1.35, peak);
  let targetPeak = knee + excess / (1.0 + pull * 2.4 * weight);
  let scale = targetPeak / max(peak, 1e-5);
  return lin * scale;
}

fn rgbToHsl(rgbIn: vec3<f32>) -> vec3<f32> {
  let rgb = clamp(rgbIn, vec3<f32>(0.0), vec3<f32>(1.0));
  let maxC = max(rgb.r, max(rgb.g, rgb.b));
  let minC = min(rgb.r, min(rgb.g, rgb.b));
  let l = 0.5 * (maxC + minC);
  let d = maxC - minC;
  if (d <= 1e-6) {
    return vec3<f32>(0.0, 0.0, l);
  }
  let s = select(
    d / max(maxC + minC, 1e-6),
    d / max(2.0 - maxC - minC, 1e-6),
    l > 0.5,
  );
  var h: f32 = 0.0;
  if (abs(maxC - rgb.r) <= 1e-6) {
    h = (rgb.g - rgb.b) / d + select(0.0, 6.0, rgb.g < rgb.b);
  } else if (abs(maxC - rgb.g) <= 1e-6) {
    h = (rgb.b - rgb.r) / d + 2.0;
  } else {
    h = (rgb.r - rgb.g) / d + 4.0;
  }
  return vec3<f32>(mod360(h * 60.0), clamp(s, 0.0, 1.0), clamp(l, 0.0, 1.0));
}

fn hueToRgbChannel(p: f32, q: f32, tIn: f32) -> f32 {
  var t = tIn;
  if (t < 0.0) { t = t + 1.0; }
  if (t > 1.0) { t = t - 1.0; }
  if (t < 1.0 / 6.0) { return p + (q - p) * 6.0 * t; }
  if (t < 0.5) { return q; }
  if (t < 2.0 / 3.0) { return p + (q - p) * (2.0 / 3.0 - t) * 6.0; }
  return p;
}

fn hslToRgb(hsl: vec3<f32>) -> vec3<f32> {
  let h = hsl.x;
  let s = hsl.y;
  let l = hsl.z;
  if (s <= 1e-6) {
    return vec3<f32>(l);
  }
  let q = select(l + s - l * s, l * (1.0 + s), l < 0.5);
  let p = 2.0 * l - q;
  let hk = mod360(h) / 360.0;
  return vec3<f32>(
    hueToRgbChannel(p, q, hk + 1.0 / 3.0),
    hueToRgbChannel(p, q, hk),
    hueToRgbChannel(p, q, hk - 1.0 / 3.0),
  );
}

fn sampleBilinear2D(tex: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
  let dims = vec2<f32>(textureDimensions(tex, 0));
  let maxCoord = max(dims - vec2<f32>(1.0), vec2<f32>(0.0));
  let f = clamp(uv * dims - vec2<f32>(0.5), vec2<f32>(0.0), maxCoord);
  let p0 = vec2<i32>(floor(f));
  let p1 = min(p0 + vec2<i32>(1, 1), vec2<i32>(maxCoord));
  let t = f - vec2<f32>(p0);
  let c00 = textureLoad(tex, p0, 0);
  let c10 = textureLoad(tex, vec2<i32>(p1.x, p0.y), 0);
  let c01 = textureLoad(tex, vec2<i32>(p0.x, p1.y), 0);
  let c11 = textureLoad(tex, p1, 0);
  return mix(mix(c00, c10, t.x), mix(c01, c11, t.x), t.y);
}

fn sampleMatteBilinear(tex: texture_2d<f32>, uv: vec2<f32>) -> f32 {
  let dims = vec2<f32>(textureDimensions(tex, 0));
  let maxCoord = max(dims - vec2<f32>(1.0), vec2<f32>(0.0));
  let f = clamp(clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * maxCoord, vec2<f32>(0.0), maxCoord);
  let p0 = vec2<i32>(floor(f));
  let p1 = min(p0 + vec2<i32>(1, 1), vec2<i32>(maxCoord));
  let t = f - vec2<f32>(p0);
  let c00 = textureLoad(tex, p0, 0).r;
  let c10 = textureLoad(tex, vec2<i32>(p1.x, p0.y), 0).r;
  let c01 = textureLoad(tex, vec2<i32>(p0.x, p1.y), 0).r;
  let c11 = textureLoad(tex, p1, 0).r;
  return mix(mix(c00, c10, t.x), mix(c01, c11, t.x), t.y);
}

fn lookupToneLutChannel(v: f32, channel: u32) -> f32 {
  let scaled = clamp(v, 0.0, 1.0) * 255.0;
  let lo = min(i32(floor(scaled)), 255);
  let hi = min(lo + 1, 255);
  let frac = scaled - f32(lo);
  let a = textureLoad(toneLutTex, vec2<i32>(lo, 0), 0);
  let b = textureLoad(toneLutTex, vec2<i32>(hi, 0), 0);
  if (channel == 0u) { return mix(a.r, b.r, frac); }
  if (channel == 1u) { return mix(a.g, b.g, frac); }
  return mix(a.b, b.b, frac);
}

fn orientSourceUv(uv: vec2<f32>) -> vec2<f32> {
  var p = uv;
  if (u.orient.x > 0.5) { p.x = 1.0 - p.x; }
  if (u.orient.y > 0.5) { p.y = 1.0 - p.y; }
  let qt = u.orient.z;
  if (qt > 0.5 && qt < 1.5) {
    return vec2<f32>(p.y, 1.0 - p.x);
  } else if (qt >= 1.5 && qt < 2.5) {
    return vec2<f32>(1.0 - p.x, 1.0 - p.y);
  } else if (qt >= 2.5) {
    return vec2<f32>(1.0 - p.y, p.x);
  }
  return p;
}

fn applyBasicTone(rgbIn: vec3<f32>) -> vec3<f32> {
  var rgb = rgbIn;
  let luma = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
  var delta: f32 = 0.0;

  let shadows = u.basicToneA.x;
  let highlights = u.basicToneA.y;
  let whites = u.basicToneA.z;
  let blacks = u.basicToneA.w;
  let dehaze = u.basicToneB.x;
  let contrast = u.basicToneB.y;

  if (shadows != 0.0) {
    let sw = 1.0 - kSmoothstep(0.05, 0.55, luma);
    delta = delta + (shadows / 100.0) * 0.28 * sw * sw;
  }
  if (highlights != 0.0) {
    let hw = kSmoothstep(0.45, 0.95, luma);
    delta = delta + (highlights / 100.0) * 0.28 * hw * hw;
  }
  if (whites != 0.0) {
    delta = delta + (whites / 100.0) * 0.22 * kSmoothstep(0.65, 1.0, luma);
  }
  if (blacks != 0.0) {
    delta = delta + (blacks / 100.0) * 0.22 * (1.0 - kSmoothstep(0.0, 0.35, luma));
  }

  if (abs(delta) > 1e-6) {
    let targetLuma = clamp(luma + delta, 0.0, 1.0);
    if (luma > 1e-4) {
      rgb = clamp(rgb * (targetLuma / luma), vec3<f32>(0.0), vec3<f32>(1.0));
    } else {
      rgb = vec3<f32>(targetLuma);
    }
  }

  if (dehaze != 0.0) {
    let dh = dehaze / 100.0;
    rgb = clamp((rgb + vec3<f32>(-dh * 0.06)) * (1.0 + dh * 0.18), vec3<f32>(0.0), vec3<f32>(1.0));
  }

  if (contrast != 0.0) {
    let c = contrast / 100.0;
    let factor = select(1.0 + c * 0.65, 1.0 + c * 0.85, c >= 0.0);
    rgb = clamp((rgb - vec3<f32>(0.5)) * factor + vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(1.0));
  }

  return rgb;
}

fn applyHslBand(acc: vec3<f32>, h: f32, center: f32, halfWidth: f32, shift: vec3<f32>) -> vec3<f32> {
  if (shift.x == 0.0 && shift.y == 0.0 && shift.z == 0.0) {
    return acc;
  }
  let dist = hueDistanceDeg(h, center);
  if (dist >= halfWidth) {
    return acc;
  }
  let weight = 0.5 * (1.0 + cos(PI * dist / halfWidth));
  return acc + vec3<f32>(
    (shift.x / 100.0) * 28.0 * weight,
    (shift.y / 100.0) * weight,
    (shift.z / 100.0) * 0.25 * weight,
  );
}

fn applyBwBand(acc: f32, h: f32, center: f32, halfWidth: f32, lumShift: f32) -> f32 {
  if (lumShift == 0.0) {
    return acc;
  }
  let dist = hueDistanceDeg(h, center);
  if (dist >= halfWidth) {
    return acc;
  }
  let weight = 0.5 * (1.0 + cos(PI * dist / halfWidth));
  return acc + (lumShift / 100.0) * 0.45 * weight;
}

fn sampleLut3D(r: u32, g: u32, b: u32, size: u32) -> vec3<f32> {
  return lut3dData[r + size * (g + size * b)].rgb;
}

fn applyLut3D(rgb: vec3<f32>) -> vec3<f32> {
  let sizeU = u32(u.lutMeta.x);
  let blend = clamp(u.lutMeta.y, 0.0, 1.0);
  if (sizeU < 2u || blend <= 1e-6) {
    return rgb;
  }
  let maxIdx = f32(sizeU - 1u);
  let dMin = vec3<f32>(u.lutMeta.z, u.lutMeta.w, u.lutDomain.x);
  let dMax = vec3<f32>(u.lutDomain.y, u.lutDomain.z, u.lutDomain.w);
  let span = max(dMax - dMin, vec3<f32>(1e-6));
  let norm = clamp((rgb - dMin) / span, vec3<f32>(0.0), vec3<f32>(1.0)) * maxIdx;

  let r0 = min(u32(floor(norm.r)), sizeU - 1u);
  let g0 = min(u32(floor(norm.g)), sizeU - 1u);
  let b0 = min(u32(floor(norm.b)), sizeU - 1u);
  let r1 = min(r0 + 1u, sizeU - 1u);
  let g1 = min(g0 + 1u, sizeU - 1u);
  let b1 = min(b0 + 1u, sizeU - 1u);

  let fr = norm.r - f32(r0);
  let fg = norm.g - f32(g0);
  let fb = norm.b - f32(b0);

  let c00 = mix(sampleLut3D(r0, g0, b0, sizeU), sampleLut3D(r1, g0, b0, sizeU), fr);
  let c10 = mix(sampleLut3D(r0, g1, b0, sizeU), sampleLut3D(r1, g1, b0, sizeU), fr);
  let c01 = mix(sampleLut3D(r0, g0, b1, sizeU), sampleLut3D(r1, g0, b1, sizeU), fr);
  let c11 = mix(sampleLut3D(r0, g1, b1, sizeU), sampleLut3D(r1, g1, b1, sizeU), fr);

  let c0 = mix(c00, c10, fg);
  let c1 = mix(c01, c11, fg);
  let mapped = mix(c0, c1, fb);
  return mix(rgb, mapped, blend);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outW = u32(u.dims.x);
  let outH = u32(u.dims.y);
  if (gid.x >= outW || gid.y >= outH) {
    return;
  }

  let uCrop = (f32(gid.x) + 0.5) / u.dims.x;
  let vCrop = (f32(gid.y) + 0.5) / u.dims.y;
  // Oriented full-frame UV in [0..1]
  let uv = vec2<f32>(
    u.cropRect.x + uCrop * u.cropRect.z,
    u.cropRect.y + vCrop * u.cropRect.w,
  );

  // Optional fine straighten rotation in oriented pixel space (used by Rust wgpu export)
  var sampleUv = uv;
  var outOfBounds = false;
  let rotRad = u.orient.w;
  if (abs(rotRad) > 1e-5) {
    let qt = u.orient.z;
    let swapAxes = (qt > 0.5 && qt < 1.5) || (qt >= 2.5);
    let oriW = select(u.dims.z, u.dims.w, swapAxes);
    let oriH = select(u.dims.w, u.dims.z, swapAxes);
    let px = uv.x * oriW - 0.5;
    let py = uv.y * oriH - 0.5;
    let cx = (oriW - 1.0) * 0.5;
    let cy = (oriH - 1.0) * 0.5;
    let dx = px - cx;
    let dy = py - cy;
    let cosA = cos(-rotRad);
    let sinA = sin(-rotRad);
    let sx = dx * cosA - dy * sinA + cx;
    let sy = dx * sinA + dy * cosA + cy;
    if (sx < 0.0 || sy < 0.0 || sx >= oriW - 1.0 || sy >= oriH - 1.0) {
      outOfBounds = true;
    } else {
      sampleUv = vec2<f32>((sx + 0.5) / oriW, (sy + 0.5) / oriH);
    }
  }

  var rgb = vec3<f32>(0.0);
  if (!outOfBounds) {
    let srcUv = orientSourceUv(sampleUv);
    rgb = sampleBilinear2D(srcTex, srcUv).rgb;
  }

  // 1-2. 32-Bit Linear Stage: White Balance + Exposure EV + HDR Recovery + Smooth Wide-Gamut Compression
  let inputLinear = u.linearFlags.x > 0.5;
  let inputWideGamut = u.linearFlags.y > 0.5;
  let linearActive = u.linearFlags.z > 0.5;
  let exposureGain = u.linearFlags.w;
  let wbGains = u.wbAndBw.xyz;

  if (inputLinear) {
    var lin = select(
      max(rgb, vec3<f32>(0.0)),
      max(rgb * wbGains * exposureGain, vec3<f32>(0.0)),
      linearActive,
    );
    if (u.basicToneA.y < 0.0 || u.basicToneA.z < 0.0) {
      lin = recoverHdrHighlights(lin, u.basicToneA.y, u.basicToneA.z);
    }
    if (inputWideGamut) {
      lin = compressRec2020ToLinearSrgb(lin);
    }
    rgb = vec3<f32>(
      linearToSrgb(softClip(lin.r)),
      linearToSrgb(softClip(lin.g)),
      linearToSrgb(softClip(lin.b)),
    );
  } else if (linearActive) {
    var lin = vec3<f32>(srgbToLinear(rgb.r), srgbToLinear(rgb.g), srgbToLinear(rgb.b));
    lin = max(lin * wbGains * exposureGain, vec3<f32>(0.0));
    rgb = vec3<f32>(
      linearToSrgb(softClip(lin.r)),
      linearToSrgb(softClip(lin.g)),
      linearToSrgb(softClip(lin.b)),
    );
  } else {
    rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  }

  // 2b. Camera Calibration (Row-normalized 3x3 Primary Matrix + Shadows Tint)
  if (u.stageFlags.x > 0.5) {
    rgb = clamp(
      vec3<f32>(
        dot(u.calRow0.xyz, rgb),
        dot(u.calRow1.xyz, rgb),
        dot(u.calRow2.xyz, rgb),
      ),
      vec3<f32>(0.0),
      vec3<f32>(1.0),
    );
    let calTint = u.stageFlags.y;
    if (abs(calTint) > 1e-5) {
      let lumaCal = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
      let sw = 1.0 - kSmoothstep(0.0, 0.55, lumaCal);
      let delta = calTint * 0.08 * sw * sw;
      rgb = clamp(vec3<f32>(rgb.r + delta * 0.5, rgb.g - delta, rgb.b + delta * 0.5), vec3<f32>(0.0), vec3<f32>(1.0));
    }
  }

  // 3. Basic Tone (Shadows, Highlights, Whites, Blacks, Dehaze, Contrast)
  rgb = applyBasicTone(rgb);

  // 4. Combined Master + Per-Channel RGB Tone Curves (256x1 LUT)
  if (u.stageFlags.z > 0.5) {
    rgb = vec3<f32>(
      lookupToneLutChannel(rgb.r, 0u),
      lookupToneLutChannel(rgb.g, 1u),
      lookupToneLutChannel(rgb.b, 2u),
    );
  }

  // 5-6. Presence + 8-Channel HSL Color Mixer OR Perceptual B&W Color Mix
  let hslActive = u.stageFlags.w > 0.5;
  if (u.wbAndBw.w > 0.5) {
    let baseLuma = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
    var bw = baseLuma;
    if (hslActive) {
      let hslVal = rgbToHsl(rgb);
      if (hslVal.y >= 0.015) {
        var lumDelta: f32 = 0.0;
        lumDelta = applyBwBand(lumDelta, hslVal.x,   0.0, 30.0, u.hsl[0].z);
        lumDelta = applyBwBand(lumDelta, hslVal.x,  30.0, 30.0, u.hsl[1].z);
        lumDelta = applyBwBand(lumDelta, hslVal.x,  60.0, 35.0, u.hsl[2].z);
        lumDelta = applyBwBand(lumDelta, hslVal.x, 120.0, 50.0, u.hsl[3].z);
        lumDelta = applyBwBand(lumDelta, hslVal.x, 180.0, 50.0, u.hsl[4].z);
        lumDelta = applyBwBand(lumDelta, hslVal.x, 240.0, 45.0, u.hsl[5].z);
        lumDelta = applyBwBand(lumDelta, hslVal.x, 285.0, 35.0, u.hsl[6].z);
        lumDelta = applyBwBand(lumDelta, hslVal.x, 325.0, 35.0, u.hsl[7].z);
        bw = clamp(baseLuma + lumDelta * hslVal.y, 0.0, 1.0);
      }
    }
    rgb = vec3<f32>(bw);
  } else {
    let vibrance = u.basicToneB.z;
    let saturation = u.basicToneB.w;
    if (vibrance != 0.0 || saturation != 0.0) {
      var hslVal = rgbToHsl(rgb);
      if (vibrance != 0.0) {
        let vib = vibrance / 100.0;
        let skinProx = kGaussian(hueDistanceDeg(hslVal.x, 28.0), 0.0, 18.0);
        let skinGuard = select(1.0, 1.0 - 0.6 * skinProx, vib > 0.0);
        hslVal.y = clamp(hslVal.y + vib * 0.55 * (1.0 - hslVal.y) * skinGuard, 0.0, 1.0);
      }
      if (saturation != 0.0) {
        hslVal.y = clamp(hslVal.y * (1.0 + saturation / 100.0), 0.0, 1.0);
      }
      rgb = hslToRgb(hslVal);
    }

    if (hslActive) {
      let hslVal = rgbToHsl(rgb);
      if (hslVal.y >= 0.015) {
        var deltas = vec3<f32>(0.0);
        deltas = applyHslBand(deltas, hslVal.x,   0.0, 30.0, u.hsl[0].xyz);
        deltas = applyHslBand(deltas, hslVal.x,  30.0, 30.0, u.hsl[1].xyz);
        deltas = applyHslBand(deltas, hslVal.x,  60.0, 35.0, u.hsl[2].xyz);
        deltas = applyHslBand(deltas, hslVal.x, 120.0, 50.0, u.hsl[3].xyz);
        deltas = applyHslBand(deltas, hslVal.x, 180.0, 50.0, u.hsl[4].xyz);
        deltas = applyHslBand(deltas, hslVal.x, 240.0, 45.0, u.hsl[5].xyz);
        deltas = applyHslBand(deltas, hslVal.x, 285.0, 35.0, u.hsl[6].xyz);
        deltas = applyHslBand(deltas, hslVal.x, 325.0, 35.0, u.hsl[7].xyz);
        let nextH = mod360(hslVal.x + deltas.x);
        let nextS = clamp(hslVal.y * (1.0 + deltas.y), 0.0, 1.0);
        let nextL = clamp(hslVal.z + deltas.z * hslVal.y, 0.0, 1.0);
        rgb = hslToRgb(vec3<f32>(nextH, nextS, nextL));
      }
    }
  }

  // 6b. 3-Way Color Grading (Shadows / Midtones / Highlights + Blending & Balance)
  if (u.cgMeta.x > 0.5) {
    let lumaCg = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
    let pivot = u.cgMeta.y;
    let spread = u.cgMeta.z;
    var shdW = pow(clamp(1.0 - kSmoothstep(pivot - spread, pivot + spread * 0.65, lumaCg), 1e-6, 1.0), 1.5);
    var hiW = pow(clamp(kSmoothstep(pivot - spread * 0.65, pivot + spread, lumaCg), 1e-6, 1.0), 1.5);
    var midW = max(0.0, 1.0 - shdW - hiW);
    let wSum = max(1.0, shdW + midW + hiW);
    shdW = shdW / wSum;
    midW = midW / wSum;
    hiW = hiW / wSum;

    let shdEnv = 1.0 - kSmoothstep(0.75, 1.0, lumaCg);
    let midEnv = kSmoothstep(0.0, 0.12, lumaCg) * (1.0 - kSmoothstep(0.88, 1.0, lumaCg));
    let hiEnv = kSmoothstep(0.05, 0.30, lumaCg);
    let lumShift = shdW * u.cgShd.w + midW * u.cgMid.w + hiW * u.cgHi.w;
    rgb = clamp(
      rgb + u.cgShd.xyz * (shdW * shdEnv) + u.cgMid.xyz * (midW * midEnv) + u.cgHi.xyz * (hiW * hiEnv) + vec3<f32>(lumShift),
      vec3<f32>(0.0),
      vec3<f32>(1.0),
    );
  }

  // 7. 3D .cube LUT (Trilinear lookup + intensity blend)
  if (u.lutMeta.x >= 2.0 && u.lutMeta.y > 1e-6) {
    rgb = applyLut3D(rgb);
  }

  // 8. Post-crop radial vignette (Amount + Midpoint + Feather)
  let vigAmount = u.vignette.x;
  if (vigAmount != 0.0) {
    let halfW = max(u.dims.x * 0.5, 1.0);
    let halfH = max(u.dims.y * 0.5, 1.0);
    let nx = (f32(gid.x) - halfW) / halfW;
    let ny = (f32(gid.y) - halfH) / halfH;
    let radius = length(vec2<f32>(nx, ny));
    let mid = clamp(u.vignette.y / 100.0, 0.05, 0.98) * 1.15;
    let f = clamp(u.vignette.z / 100.0, 0.02, 1.0);
    let inner = mid * (1.0 - 0.65 * f);
    let outer = min(1.65, mid + 0.12 + 0.72 * f);
    let mask = kSmoothstep(inner, outer, radius);
    let factor = clamp(1.0 + (vigAmount / 100.0) * 0.75 * mask, 0.15, 1.85);
    rgb = clamp(rgb * factor, vec3<f32>(0.0), vec3<f32>(1.0));
  }

  rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));

  // 9. Optional Opaque Studio Backdrop (Solid / Gradient) composited BEFORE local masks
  let bgMode = u.bgColor1.w;
  if (bgMode > 1.5) {
    let rawM = sampleMatteBilinear(bgMatteTex, uv);
    let f = clamp(u.bgParams.y, 0.0, 100.0);
    var matteA = rawM;
    if (f < 24.5) {
      let k = 1.0 + ((25.0 - f) / 25.0) * 3.0;
      matteA = clamp((rawM - 0.5) * k + 0.5, 0.0, 1.0);
    } else if (f > 25.5) {
      let t = (f - 25.0) / 75.0;
      let soft = kSmoothstep(0.0, 1.0, pow(clamp(rawM, 0.0, 1.0), 1.0 - 0.35 * t));
      matteA = mix(rawM, soft, t);
    }

    var backdrop = u.bgColor1.rgb;
    if (bgMode > 2.5) {
      var gt: f32 = 0.0;
      if (u.bgColor2.w > 0.5) {
        let rad = u.bgParams.x * 0.0174532925;
        let dir = vec2<f32>(sin(rad), -cos(rad));
        let span = max(abs(dir.x) + abs(dir.y), 1e-5);
        let proj = dot(uv - vec2<f32>(0.5, 0.5), dir);
        gt = clamp(proj / span + 0.5, 0.0, 1.0);
      } else {
        let dp = (uv - vec2<f32>(0.5, 0.42)) / vec2<f32>(0.68, 0.72);
        let d = clamp(length(dp), 0.0, 1.0);
        gt = kSmoothstep(0.0, 1.0, d);
      }
      backdrop = mix(u.bgColor1.rgb, u.bgColor2.rgb, gt);
    }
    rgb = clamp(mix(backdrop, rgb, matteA), vec3<f32>(0.0), vec3<f32>(1.0));
  }

  textureStore(dstTex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(rgb, 1.0));
}
