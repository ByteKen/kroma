// Unified 32-Bit Spatial Detail, Noise Reduction, Film Grain & Cutout Alpha Compute Pass.
// Shared verbatim between Kroma WebGPU Studio Preview and Rust `wgpu` Batch Export (DX12 / Metal).
// Mirrors `apply_spatial_detail_f32` + `compute_transparent_alpha_f32` in Rust.

struct SpatialUniforms {
  // (outW, outH, spatialScale, 0)
  dims: vec4<f32>,
  // (left, top, width, height) in oriented full-frame [0..1]
  cropRect: vec4<f32>,
  // (nr, cnr, texture, clarity)
  detailA: vec4<f32>,
  // (sharpening, grainAmount, grainSize, 0)
  detailB: vec4<f32>,
  // (bgMode: 0=opaque, 1=write transparent cutout alpha to dst.a, bgFeather, 0, 0)
  bgParams: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: SpatialUniforms;
@group(0) @binding(1) var gradedTex: texture_2d<f32>;
@group(0) @binding(2) var bgMatteTex: texture_2d<f32>;
@group(0) @binding(3) var dstTex: texture_storage_2d<rgba32float, write>;

fn kSmoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
  let t = clamp((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

// Exact bit-identical 32-bit integer PCG hash (zero float/FMA drift across DX12, Metal, WebGPU, and CPU)
fn pcgHash(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn grainHash12(cell: vec2<f32>) -> f32 {
  let ix = u32(max(cell.x, 0.0));
  let iy = u32(max(cell.y, 0.0));
  let h = pcgHash(ix ^ pcgHash(iy * 1664525u + 1013904223u));
  return f32(h & 0x00FFFFFFu) / 16777216.0;
}

fn sampleRgbAtPixel(fx: f32, fy: f32) -> vec3<f32> {
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
  let p00 = textureLoad(gradedTex, vec2<i32>(x0, y0), 0).rgb;
  let p10 = textureLoad(gradedTex, vec2<i32>(x1, y0), 0).rgb;
  let p01 = textureLoad(gradedTex, vec2<i32>(x0, y1), 0).rgb;
  let p11 = textureLoad(gradedTex, vec2<i32>(x1, y1), 0).rgb;
  return mix(mix(p00, p10, tx), mix(p01, p11, tx), ty);
}

fn sampleLumaAtPixel(fx: f32, fy: f32) -> f32 {
  let c = sampleRgbAtPixel(fx, fy);
  return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
}

fn sampleMatteBilinear(uv: vec2<f32>) -> f32 {
  let dims = vec2<f32>(textureDimensions(bgMatteTex, 0));
  let maxCoord = max(dims - vec2<f32>(1.0), vec2<f32>(0.0));
  let f = clamp(clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * maxCoord, vec2<f32>(0.0), maxCoord);
  let p0 = vec2<i32>(floor(f));
  let p1 = min(p0 + vec2<i32>(1, 1), vec2<i32>(maxCoord));
  let t = f - vec2<f32>(p0);
  let c00 = textureLoad(bgMatteTex, p0, 0).r;
  let c10 = textureLoad(bgMatteTex, vec2<i32>(p1.x, p0.y), 0).r;
  let c01 = textureLoad(bgMatteTex, vec2<i32>(p0.x, p1.y), 0).r;
  let c11 = textureLoad(bgMatteTex, p1, 0).r;
  return mix(mix(c00, c10, t.x), mix(c01, c11, t.x), t.y);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outW = u32(u.dims.x);
  let outH = u32(u.dims.y);
  if (gid.x >= outW || gid.y >= outH) {
    return;
  }

  let coord = vec2<i32>(i32(gid.x), i32(gid.y));
  let fx = f32(gid.x);
  let fy = f32(gid.y);
  let scale = u.dims.z;
  let center = textureLoad(gradedTex, coord, 0);
  var rgb = center.rgb;
  let lc = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;

  let nrIn = u.detailA.x;
  let cnrIn = u.detailA.y;
  let texIn = u.detailA.z;
  let clarIn = u.detailA.w;
  let sharpIn = u.detailB.x;
  let grainAmtIn = u.detailB.y;
  let grainSizeIn = u.detailB.z;

  // 0. Edge-preserving YCbCr Color Noise Reduction
  if (cnrIn > 1e-4) {
    let cnr = cnrIn / 100.0;
    let rCnr = 2.2 * scale;
    let sigmaC = 0.04 + cnr * 0.14;
    let inv2Sig2 = -0.5 / (sigmaC * sigmaC);
    let cr0 = rgb.r - lc;
    let cb0 = rgb.b - lc;
    var sumCr = cr0;
    var sumCb = cb0;
    var sumW: f32 = 1.0;

    let s1 = sampleRgbAtPixel(fx - rCnr, fy); let y1 = 0.2126 * s1.r + 0.7152 * s1.g + 0.0722 * s1.b; let w1 = 0.7 * exp((y1 - lc) * (y1 - lc) * inv2Sig2); sumCr = sumCr + (s1.r - y1) * w1; sumCb = sumCb + (s1.b - y1) * w1; sumW = sumW + w1;
    let s2 = sampleRgbAtPixel(fx + rCnr, fy); let y2 = 0.2126 * s2.r + 0.7152 * s2.g + 0.0722 * s2.b; let w2 = 0.7 * exp((y2 - lc) * (y2 - lc) * inv2Sig2); sumCr = sumCr + (s2.r - y2) * w2; sumCb = sumCb + (s2.b - y2) * w2; sumW = sumW + w2;
    let s3 = sampleRgbAtPixel(fx, fy - rCnr); let y3 = 0.2126 * s3.r + 0.7152 * s3.g + 0.0722 * s3.b; let w3 = 0.7 * exp((y3 - lc) * (y3 - lc) * inv2Sig2); sumCr = sumCr + (s3.r - y3) * w3; sumCb = sumCb + (s3.b - y3) * w3; sumW = sumW + w3;
    let s4 = sampleRgbAtPixel(fx, fy + rCnr); let y4 = 0.2126 * s4.r + 0.7152 * s4.g + 0.0722 * s4.b; let w4 = 0.7 * exp((y4 - lc) * (y4 - lc) * inv2Sig2); sumCr = sumCr + (s4.r - y4) * w4; sumCb = sumCb + (s4.b - y4) * w4; sumW = sumW + w4;
    let s5 = sampleRgbAtPixel(fx - rCnr, fy - rCnr); let y5 = 0.2126 * s5.r + 0.7152 * s5.g + 0.0722 * s5.b; let w5 = 0.5 * exp((y5 - lc) * (y5 - lc) * inv2Sig2); sumCr = sumCr + (s5.r - y5) * w5; sumCb = sumCb + (s5.b - y5) * w5; sumW = sumW + w5;
    let s6 = sampleRgbAtPixel(fx + rCnr, fy - rCnr); let y6 = 0.2126 * s6.r + 0.7152 * s6.g + 0.0722 * s6.b; let w6 = 0.5 * exp((y6 - lc) * (y6 - lc) * inv2Sig2); sumCr = sumCr + (s6.r - y6) * w6; sumCb = sumCb + (s6.b - y6) * w6; sumW = sumW + w6;
    let s7 = sampleRgbAtPixel(fx - rCnr, fy + rCnr); let y7 = 0.2126 * s7.r + 0.7152 * s7.g + 0.0722 * s7.b; let w7 = 0.5 * exp((y7 - lc) * (y7 - lc) * inv2Sig2); sumCr = sumCr + (s7.r - y7) * w7; sumCb = sumCb + (s7.b - y7) * w7; sumW = sumW + w7;
    let s8 = sampleRgbAtPixel(fx + rCnr, fy + rCnr); let y8 = 0.2126 * s8.r + 0.7152 * s8.g + 0.0722 * s8.b; let w8 = 0.5 * exp((y8 - lc) * (y8 - lc) * inv2Sig2); sumCr = sumCr + (s8.r - y8) * w8; sumCb = sumCb + (s8.b - y8) * w8; sumW = sumW + w8;

    let invW = 1.0 / max(sumW, 1e-5);
    let blendC = cnr * 0.85;
    let nextCr = mix(cr0, sumCr * invW, blendC);
    let nextCb = mix(cb0, sumCb * invW, blendC);
    let nrR = clamp(lc + nextCr, 0.0, 1.0);
    let nrB = clamp(lc + nextCb, 0.0, 1.0);
    let nrG = clamp((lc - 0.2126 * nrR - 0.0722 * nrB) / 0.7152, 0.0, 1.0);
    rgb = vec3<f32>(nrR, nrG, nrB);
  }

  var totalDelta: f32 = 0.0;

  // 1. Edge-preserving 3x3 bilateral Luminance NR
  if (nrIn > 1e-4) {
    let nr = nrIn / 100.0;
    let rNr = 1.25 * scale;
    let sigmaR = 0.03 + nr * 0.11;
    let inv2Sig2 = -0.5 / (sigmaR * sigmaR);
    var sumVal = lc;
    var sumW: f32 = 1.0;

    let l1 = sampleLumaAtPixel(fx - rNr, fy); let w1 = 0.65 * exp((l1 - lc) * (l1 - lc) * inv2Sig2); sumVal = sumVal + l1 * w1; sumW = sumW + w1;
    let l2 = sampleLumaAtPixel(fx + rNr, fy); let w2 = 0.65 * exp((l2 - lc) * (l2 - lc) * inv2Sig2); sumVal = sumVal + l2 * w2; sumW = sumW + w2;
    let l3 = sampleLumaAtPixel(fx, fy - rNr); let w3 = 0.65 * exp((l3 - lc) * (l3 - lc) * inv2Sig2); sumVal = sumVal + l3 * w3; sumW = sumW + w3;
    let l4 = sampleLumaAtPixel(fx, fy + rNr); let w4 = 0.65 * exp((l4 - lc) * (l4 - lc) * inv2Sig2); sumVal = sumVal + l4 * w4; sumW = sumW + w4;
    let l5 = sampleLumaAtPixel(fx - rNr, fy - rNr); let w5 = 0.45 * exp((l5 - lc) * (l5 - lc) * inv2Sig2); sumVal = sumVal + l5 * w5; sumW = sumW + w5;
    let l6 = sampleLumaAtPixel(fx + rNr, fy - rNr); let w6 = 0.45 * exp((l6 - lc) * (l6 - lc) * inv2Sig2); sumVal = sumVal + l6 * w6; sumW = sumW + w6;
    let l7 = sampleLumaAtPixel(fx - rNr, fy + rNr); let w7 = 0.45 * exp((l7 - lc) * (l7 - lc) * inv2Sig2); sumVal = sumVal + l7 * w7; sumW = sumW + w7;
    let l8 = sampleLumaAtPixel(fx + rNr, fy + rNr); let w8 = 0.45 * exp((l8 - lc) * (l8 - lc) * inv2Sig2); sumVal = sumVal + l8 * w8; sumW = sumW + w8;

    let lNr = sumVal / max(sumW, 1e-5);
    totalDelta = totalDelta + (lNr - lc) * (nr * 0.90);
  }

  // 2. Medium-fine edge-clamped micro-contrast (Texture)
  if (abs(texIn) > 1e-4) {
    let tex = texIn / 100.0;
    let rTex = 2.5 * scale;
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

  // 3. Broad midtone-masked local contrast (Clarity)
  if (abs(clarIn) > 1e-4) {
    let clar = clarIn / 100.0;
    let rClar = 11.0 * scale;
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
    let midtoneMask = pow(clamp(4.0 * lc * (1.0 - lc), 1e-6, 1.0), 0.70);
    totalDelta = totalDelta + clamp(lc - blurClar, -0.30, 0.30) * clar * 1.15 * midtoneMask;
  }

  // 4. Thresholded luminance unsharp mask (Sharpening)
  if (sharpIn > 1e-4) {
    let sharp = sharpIn / 100.0;
    let rSharp = 1.0 * scale;
    let avg4 = 0.25 * (
      sampleLumaAtPixel(fx - rSharp, fy) + sampleLumaAtPixel(fx + rSharp, fy) +
      sampleLumaAtPixel(fx, fy - rSharp) + sampleLumaAtPixel(fx, fy + rSharp)
    );
    let hp = lc - avg4;
    let mag = max(abs(hp) - 0.008, 0.0);
    let threshHp = select(-1.0, 1.0, hp >= 0.0) * mag;
    totalDelta = totalDelta + clamp(threshHp * sharp * 1.10, -0.25, 0.25);
  }

  if (abs(totalDelta) > 1e-6) {
    let nextLuma = clamp(lc + totalDelta, 0.0, 1.0);
    let actualDelta = nextLuma - lc;
    let ratio = select(1.0, clamp(nextLuma / lc, 0.0, 3.0), lc > 1e-4);
    let blend = kSmoothstep(0.02, 0.16, lc);
    let additive = clamp(rgb + vec3<f32>(actualDelta), vec3<f32>(0.0), vec3<f32>(1.0));
    let mult = select(vec3<f32>(nextLuma), clamp(rgb * ratio, vec3<f32>(0.0), vec3<f32>(1.0)), lc > 1e-4);
    rgb = mix(additive, mult, blend);
  }

  // 5. Resolution-proportional organic Film Grain
  let uNorm = (fx + 0.5) / u.dims.x;
  let vNorm = (fy + 0.5) / u.dims.y;
  if (grainAmtIn > 1e-4) {
    let ga = grainAmtIn / 100.0;
    let gs = grainSizeIn / 100.0;
    let grid = 1024.0 / (0.8 + gs * 2.4);
    let cell = floor(vec2<f32>(uNorm, vNorm) * grid + vec2<f32>(1e-4));
    let n1 = grainHash12(cell);
    let n2 = grainHash12(cell + vec2<f32>(17.0, 41.0));
    let signedNoise = (n1 + n2) - 1.0;
    let lCur = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
    let midEnv = 0.35 + 0.65 * clamp(4.0 * lCur * (1.0 - lCur), 0.0, 1.0);
    rgb = clamp(rgb + vec3<f32>(signedNoise * ga * 0.11 * midEnv), vec3<f32>(0.0), vec3<f32>(1.0));
  }

  // 6. Optional Transparent Cutout Alpha written directly to dst.a (exact parity for PNG export & Studio checkerboard)
  var outAlpha: f32 = 1.0;
  if (u.bgParams.x > 0.5) {
    let uvFull = vec2<f32>(
      u.cropRect.x + uNorm * u.cropRect.z,
      u.cropRect.y + vNorm * u.cropRect.w,
    );
    let rawM = sampleMatteBilinear(uvFull);
    let f = clamp(u.bgParams.y, 0.0, 100.0);
    outAlpha = rawM;
    if (f < 24.5) {
      let k = 1.0 + ((25.0 - f) / 25.0) * 3.0;
      outAlpha = clamp((rawM - 0.5) * k + 0.5, 0.0, 1.0);
    } else if (f > 25.5) {
      let t = (f - 25.0) / 75.0;
      let soft = kSmoothstep(0.0, 1.0, pow(clamp(rawM, 0.0, 1.0), 1.0 - 0.35 * t));
      outAlpha = mix(rawM, soft, t);
    }
  }

  textureStore(dstTex, coord, vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), outAlpha));
}
