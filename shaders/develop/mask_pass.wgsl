// Unified 32-Bit Local & AI Masking Compute Pass.
// Shared verbatim between Kroma WebGPU Studio Preview and Rust `wgpu` Batch Export (DX12 / Metal).
// Mirrors `apply_single_mask_layer_f32` in `internal/infrastructure/src/develop/mask_engine.rs`.

struct MaskUniforms {
  // (outW, outH, spatialScale, passMode: 0=combined, 1=toneOnly, 2=spatialOnly)
  dims: vec4<f32>,
  // (left, top, width, height) in oriented full-frame [0..1]
  cropRect: vec4<f32>,
  // (hasSpatial, layerInvert, layerOpacity, overlayOnly)
  layerFlags: vec4<f32>,
  // Up to 4 range slots: (mode: 0=none/1=luma/2=color, op: 0=add/1=sub/2=int, inverted, 0)
  rangeMeta: array<vec4<f32>, 4>,
  // luma:(min,max,feather,0) | color:(hue,sat,luma,tol)
  rangeParamsA: array<vec4<f32>, 4>,
  // color: feather[0..3]
  rangeParamsB: vec4<f32>,
  // (linearActive, exposureGain, 0, 0)
  linearParams: vec4<f32>,
  // (wbR, wbG, wbB, 0)
  wbGains: vec4<f32>,
  // (highlights, shadows, whites, blacks)
  toneA: vec4<f32>,
  // (dehaze, contrast, saturation, 0)
  toneB: vec4<f32>,
  // (texture, clarity, sharpness, 0)
  detail: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: MaskUniforms;
@group(0) @binding(1) var prevTex: texture_2d<f32>;
@group(0) @binding(2) var spatialMaskTex: texture_2d<f32>;
@group(0) @binding(3) var dstTex: texture_storage_2d<rgba32float, write>;

fn kSmoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
  let t = clamp((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

fn mod360(x: f32) -> f32 {
  return x - 360.0 * floor(x / 360.0);
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

fn evalLuminanceRange(rgb: vec3<f32>, minL: f32, maxL: f32, feather: f32) -> f32 {
  let luma = clamp(0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b, 0.0, 1.0);
  let f = max(clamp(feather, 0.0, 0.5), 0.005);
  let lo = select(kSmoothstep(minL - f, minL + f * 0.25, luma), 1.0, minL <= 0.001);
  let hi = select(1.0 - kSmoothstep(maxL - f * 0.25, maxL + f, luma), 1.0, maxL >= 0.999);
  return clamp(lo * hi, 0.0, 1.0);
}

fn evalColorRange(rgb: vec3<f32>, tHue: f32, tSat: f32, tLuma: f32, tolIn: f32, featherIn: f32) -> f32 {
  let hsl = rgbToHsl(rgb);
  let rawDiff = mod360(abs(hsl.x - tHue));
  let hueDist = select(rawDiff, 360.0 - rawDiff, rawDiff > 180.0);
  let dh = hueDist / 180.0;
  let hueW = clamp(min(hsl.y, tSat) * 4.0, 0.0, 1.0);
  let ds = abs(hsl.y - tSat);
  let dl = abs(hsl.z - tLuma);
  let a = dh * 2.2 * hueW;
  let b = ds * 0.9;
  let c = dl * 0.75;
  let dist = sqrt(a * a + b * b + c * c);
  let tol = clamp(tolIn, 0.02, 1.0);
  let f = max(tol * clamp(featherIn, 0.0, 1.0), 0.015);
  return 1.0 - kSmoothstep(max(0.0, tol - f * 0.5), tol + f, dist);
}

fn sampleSpatialMask(uv: vec2<f32>) -> f32 {
  let dims = vec2<f32>(textureDimensions(spatialMaskTex, 0));
  let maxCoord = max(dims - vec2<f32>(1.0), vec2<f32>(0.0));
  let f = clamp(uv * dims - vec2<f32>(0.5), vec2<f32>(0.0), maxCoord);
  let p0 = vec2<i32>(floor(f));
  let p1 = min(p0 + vec2<i32>(1, 1), vec2<i32>(maxCoord));
  let t = f - vec2<f32>(p0);
  let c00 = textureLoad(spatialMaskTex, p0, 0).r;
  let c10 = textureLoad(spatialMaskTex, vec2<i32>(p1.x, p0.y), 0).r;
  let c01 = textureLoad(spatialMaskTex, vec2<i32>(p0.x, p1.y), 0).r;
  let c11 = textureLoad(spatialMaskTex, p1, 0).r;
  return mix(mix(c00, c10, t.x), mix(c01, c11, t.x), t.y);
}

fn sampleLumaAtPixel(fx: f32, fy: f32) -> f32 {
  let maxX = max(u.dims.x - 1.0, 0.0);
  let maxY = max(u.dims.y - 1.0, 0.0);
  let cx = clamp(fx, 0.0, maxX);
  let cy = clamp(fy, 0.0, maxY);
  let x0 = i32(floor(cx));
  let y0 = i32(floor(cy));
  let x1 = min(x0 + 1, i32(maxX));
  let y1 = min(y0 + 1, i32(maxY));
  let tx = cx - f32(x0);
  let ty = cy - f32(y0);
  let p00 = textureLoad(prevTex, vec2<i32>(x0, y0), 0).rgb;
  let p10 = textureLoad(prevTex, vec2<i32>(x1, y0), 0).rgb;
  let p01 = textureLoad(prevTex, vec2<i32>(x0, y1), 0).rgb;
  let p11 = textureLoad(prevTex, vec2<i32>(x1, y1), 0).rgb;
  let l00 = 0.2126 * p00.r + 0.7152 * p00.g + 0.0722 * p00.b;
  let l10 = 0.2126 * p10.r + 0.7152 * p10.g + 0.0722 * p10.b;
  let l01 = 0.2126 * p01.r + 0.7152 * p01.g + 0.0722 * p01.b;
  let l11 = 0.2126 * p11.r + 0.7152 * p11.g + 0.0722 * p11.b;
  return mix(mix(l00, l10, tx), mix(l01, l11, tx), ty);
}

fn getRangeFeather(idx: i32) -> f32 {
  if (idx == 0) { return u.rangeParamsB.x; }
  if (idx == 1) { return u.rangeParamsB.y; }
  if (idx == 2) { return u.rangeParamsB.z; }
  return u.rangeParamsB.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outW = u32(u.dims.x);
  let outH = u32(u.dims.y);
  if (gid.x >= outW || gid.y >= outH) {
    return;
  }

  let coord = vec2<i32>(i32(gid.x), i32(gid.y));
  let center = textureLoad(prevTex, coord, 0);
  let origRgb = center.rgb;

  let uCrop = (f32(gid.x) + 0.5) / u.dims.x;
  let vCrop = (f32(gid.y) + 0.5) / u.dims.y;
  let uv = vec2<f32>(
    u.cropRect.x + uCrop * u.cropRect.z,
    u.cropRect.y + vCrop * u.cropRect.w,
  );

  let passMode = u.dims.w;
  var alpha: f32 = 0.0;

  if (passMode > 1.5) {
    // Stage 2 (spatial-only after Stage 1 stored exact pre-tone mask alpha in center.a)
    alpha = clamp(center.a, 0.0, 1.0);
  } else {
    var acc: f32 = 0.0;
    var started = false;
    if (u.layerFlags.x > 0.5) {
      acc = sampleSpatialMask(uv);
      started = true;
    }

    for (var i: i32 = 0; i < 4; i = i + 1) {
      let rMeta = u.rangeMeta[i];
      if (rMeta.x < 0.5) {
        continue;
      }
      let pA = u.rangeParamsA[i];
      var val: f32 = 0.0;
      if (rMeta.x < 1.5) {
        val = evalLuminanceRange(origRgb, pA.x, pA.y, pA.z);
      } else {
        val = evalColorRange(origRgb, pA.x, pA.y, pA.z, pA.w, getRangeFeather(i));
      }
      if (rMeta.z > 0.5) {
        val = 1.0 - val;
      }

      if (!started) {
        acc = select(clamp(val, 0.0, 1.0), clamp(1.0 - val, 0.0, 1.0), rMeta.y > 0.5 && rMeta.y < 1.5);
        started = true;
      } else if (rMeta.y < 0.5) {
        acc = clamp(acc + val, 0.0, 1.0);
      } else if (rMeta.y < 1.5) {
        acc = clamp(acc - val, 0.0, 1.0);
      } else {
        acc = clamp(acc * val, 0.0, 1.0);
      }
    }

    if (u.layerFlags.y > 0.5) {
      acc = 1.0 - acc;
    }
    alpha = clamp(acc * u.layerFlags.z, 0.0, 1.0);
  }

  // Translucent Ruby Overlay (O) for Studio preview
  if (u.layerFlags.w > 0.5) {
    let ruby = vec3<f32>(0.957, 0.247, 0.368);
    let outRgb = mix(origRgb, ruby, alpha * 0.48);
    textureStore(dstTex, coord, vec4<f32>(outRgb, center.a));
    return;
  }

  let outAlpha = select(center.a, alpha, passMode > 0.5 && passMode < 1.5);

  if (alpha <= 1e-4) {
    textureStore(dstTex, coord, vec4<f32>(origRgb, outAlpha));
    return;
  }

  var rgb = origRgb;

  // Stage 1: Local Tone & Color adjustments (skipped when passMode == 2.0 spatialOnly)
  if (passMode < 1.5) {
    if (u.linearParams.x > 0.5) {
      var lin = vec3<f32>(srgbToLinear(rgb.r), srgbToLinear(rgb.g), srgbToLinear(rgb.b));
      lin = max(lin * u.wbGains.xyz * u.linearParams.y, vec3<f32>(0.0));
      rgb = vec3<f32>(
        linearToSrgb(softClip(lin.r)),
        linearToSrgb(softClip(lin.g)),
        linearToSrgb(softClip(lin.b)),
      );
    }

    let luma = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
    var delta: f32 = 0.0;
    if (u.toneA.y != 0.0) {
      let sw = 1.0 - kSmoothstep(0.05, 0.55, luma);
      delta = delta + (u.toneA.y / 100.0) * 0.28 * sw * sw;
    }
    if (u.toneA.x != 0.0) {
      let hw = kSmoothstep(0.45, 0.95, luma);
      delta = delta + (u.toneA.x / 100.0) * 0.28 * hw * hw;
    }
    if (u.toneA.z != 0.0) {
      delta = delta + (u.toneA.z / 100.0) * 0.22 * kSmoothstep(0.65, 1.0, luma);
    }
    if (u.toneA.w != 0.0) {
      delta = delta + (u.toneA.w / 100.0) * 0.22 * (1.0 - kSmoothstep(0.0, 0.35, luma));
    }
    if (abs(delta) > 1e-6) {
      let targetLuma = clamp(luma + delta, 0.0, 1.0);
      if (luma > 1e-4) {
        rgb = clamp(rgb * (targetLuma / luma), vec3<f32>(0.0), vec3<f32>(1.0));
      } else {
        rgb = vec3<f32>(targetLuma);
      }
    }

    if (u.toneB.x != 0.0) {
      let dh = u.toneB.x / 100.0;
      rgb = clamp((rgb + vec3<f32>(-dh * 0.06)) * (1.0 + dh * 0.18), vec3<f32>(0.0), vec3<f32>(1.0));
    }

    if (u.toneB.y != 0.0) {
      let c = u.toneB.y / 100.0;
      let factor = select(1.0 + c * 0.65, 1.0 + c * 0.85, c >= 0.0);
      rgb = clamp((rgb - vec3<f32>(0.5)) * factor + vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(1.0));
    }

    if (u.toneB.z != 0.0) {
      let lCur = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
      let sat = 1.0 + u.toneB.z / 100.0;
      rgb = clamp(vec3<f32>(lCur) + (rgb - vec3<f32>(lCur)) * sat, vec3<f32>(0.0), vec3<f32>(1.0));
    }

    rgb = mix(origRgb, rgb, alpha);
  }

  // Stage 2: Local Spatial Detail (skipped when passMode == 1.0 toneOnly)
  let texAmt = u.detail.x;
  let clarAmt = u.detail.y;
  let sharpAmt = u.detail.z;
  if (passMode < 0.5 || passMode > 1.5) {
    if (abs(texAmt) > 1e-4 || abs(clarAmt) > 1e-4 || abs(sharpAmt) > 1e-4) {
      let fx = f32(gid.x);
      let fy = f32(gid.y);
      let spatialScale = u.dims.z;
      let lc = 0.2126 * origRgb.r + 0.7152 * origRgb.g + 0.0722 * origRgb.b;
      var totalDelta: f32 = 0.0;

      if (abs(texAmt) > 1e-4) {
        let tex = texAmt / 100.0;
        let rTex = 2.5 * spatialScale;
        let dTex = rTex * 0.70710677;
        let blurTex = 0.20 * lc
          + 0.12 * (
              sampleLumaAtPixel(fx - rTex, fy) + sampleLumaAtPixel(fx + rTex, fy) +
              sampleLumaAtPixel(fx, fy - rTex) + sampleLumaAtPixel(fx, fy + rTex)
            )
          + 0.08 * (
              sampleLumaAtPixel(fx - dTex, fy - dTex) + sampleLumaAtPixel(fx + dTex, fy - dTex) +
              sampleLumaAtPixel(fx - dTex, fy + dTex) + sampleLumaAtPixel(fx + dTex, fy + dTex)
            );
        totalDelta = totalDelta + clamp(lc - blurTex, -0.18, 0.18) * tex * 0.85;
      }

      if (abs(clarAmt) > 1e-4) {
        let clar = clarAmt / 100.0;
        let rClar = 11.0 * spatialScale;
        let inClar = rClar * 0.45;
        let dClar = rClar * 0.70710677;
        let blurClar = 0.16 * lc
          + 0.10 * (
              sampleLumaAtPixel(fx - inClar, fy) + sampleLumaAtPixel(fx + inClar, fy) +
              sampleLumaAtPixel(fx, fy - inClar) + sampleLumaAtPixel(fx, fy + inClar)
            )
          + 0.06 * (
              sampleLumaAtPixel(fx - rClar, fy) + sampleLumaAtPixel(fx + rClar, fy) +
              sampleLumaAtPixel(fx, fy - rClar) + sampleLumaAtPixel(fx, fy + rClar)
            )
          + 0.05 * (
              sampleLumaAtPixel(fx - dClar, fy - dClar) + sampleLumaAtPixel(fx + dClar, fy - dClar) +
              sampleLumaAtPixel(fx - dClar, fy + dClar) + sampleLumaAtPixel(fx + dClar, fy + dClar)
            );
        let midMask = pow(clamp(4.0 * lc * (1.0 - lc), 1e-6, 1.0), 0.70);
        totalDelta = totalDelta + clamp(lc - blurClar, -0.30, 0.30) * clar * 1.15 * midMask;
      }

      if (sharpAmt > 1e-4) {
        let sharp = sharpAmt / 100.0;
        let rSharp = 1.0 * spatialScale;
        let avg4 = 0.25 * (
          sampleLumaAtPixel(fx - rSharp, fy) + sampleLumaAtPixel(fx + rSharp, fy) +
          sampleLumaAtPixel(fx, fy - rSharp) + sampleLumaAtPixel(fx, fy + rSharp)
        );
        let hp = lc - avg4;
        let mag = max(abs(hp) - 0.008, 0.0);
        totalDelta = totalDelta + clamp(select(-1.0, 1.0, hp >= 0.0) * mag * sharp * 1.10, -0.25, 0.25);
      } else if (sharpAmt < -1e-4) {
        let soften = -sharpAmt / 100.0;
        let rSoft = 1.6 * spatialScale;
        let sigmaR = 0.04 + soften * 0.12;
        let inv2Sig2 = -0.5 / (sigmaR * sigmaR);
        var sumVal = lc;
        var sumW: f32 = 1.0;

        let l1 = sampleLumaAtPixel(fx - rSoft, fy); let w1 = 0.65 * exp((l1 - lc) * (l1 - lc) * inv2Sig2); sumVal = sumVal + l1 * w1; sumW = sumW + w1;
        let l2 = sampleLumaAtPixel(fx + rSoft, fy); let w2 = 0.65 * exp((l2 - lc) * (l2 - lc) * inv2Sig2); sumVal = sumVal + l2 * w2; sumW = sumW + w2;
        let l3 = sampleLumaAtPixel(fx, fy - rSoft); let w3 = 0.65 * exp((l3 - lc) * (l3 - lc) * inv2Sig2); sumVal = sumVal + l3 * w3; sumW = sumW + w3;
        let l4 = sampleLumaAtPixel(fx, fy + rSoft); let w4 = 0.65 * exp((l4 - lc) * (l4 - lc) * inv2Sig2); sumVal = sumVal + l4 * w4; sumW = sumW + w4;
        let l5 = sampleLumaAtPixel(fx - rSoft, fy - rSoft); let w5 = 0.45 * exp((l5 - lc) * (l5 - lc) * inv2Sig2); sumVal = sumVal + l5 * w5; sumW = sumW + w5;
        let l6 = sampleLumaAtPixel(fx + rSoft, fy - rSoft); let w6 = 0.45 * exp((l6 - lc) * (l6 - lc) * inv2Sig2); sumVal = sumVal + l6 * w6; sumW = sumW + w6;
        let l7 = sampleLumaAtPixel(fx - rSoft, fy + rSoft); let w7 = 0.45 * exp((l7 - lc) * (l7 - lc) * inv2Sig2); sumVal = sumVal + l7 * w7; sumW = sumW + w7;
        let l8 = sampleLumaAtPixel(fx + rSoft, fy + rSoft); let w8 = 0.45 * exp((l8 - lc) * (l8 - lc) * inv2Sig2); sumVal = sumVal + l8 * w8; sumW = sumW + w8;

        totalDelta = totalDelta + (sumVal / max(sumW, 1e-5) - lc) * (soften * 0.90);
      }

      totalDelta = totalDelta * alpha;
      if (abs(totalDelta) > 1e-6) {
        let curL = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
        let nextL = clamp(curL + totalDelta, 0.0, 1.0);
        let actualDelta = nextL - curL;
        let ratio = select(1.0, clamp(nextL / curL, 0.0, 3.0), curL > 1e-4);
        let blend = kSmoothstep(0.02, 0.16, curL);
        let additive = clamp(rgb + vec3<f32>(actualDelta), vec3<f32>(0.0), vec3<f32>(1.0));
        let mult = select(vec3<f32>(nextL), clamp(rgb * ratio, vec3<f32>(0.0), vec3<f32>(1.0)), curL > 1e-4);
        rgb = mix(additive, mult, blend);
      }
    }
  }

  let finalAlpha = select(outAlpha, 1.0, passMode > 1.5);
  textureStore(dstTex, coord, vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), finalAlpha));
}
