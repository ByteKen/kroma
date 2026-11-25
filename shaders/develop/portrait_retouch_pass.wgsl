// Unified 32-Bit Frequency-Separation & Landmark-Guided Portrait Retouch Compute Pass (v0.3.0).
// Shared verbatim between Kroma WebGPU Studio Preview and Rust `wgpu` Batch Export (DX12 / Metal).
// Mirrors `apply_portrait_retouch_f32` in `internal/infrastructure/src/develop/retouch_engine.rs`.
//
// Features:
// - 3-band edge-preserving frequency separation (evens mid-frequency blotchiness while preserving
//   natural high-frequency skin pores and keeping eyelashes, hairlines, and glasses razor-sharp)
// - Skin tone evenness (redness / uneven chroma balancing)
// - Specular shine & oil control
// - YuNet landmark-guided Under-Eye Brighten, Eye Enhance (iris clarity + natural sclera lift),
//   Teeth Whiten, and Lip Enhance across up to 4 faces per frame (batch-syncable across shoots!).

struct PortraitFaceGpu {
  // (boxLeft, boxTop, boxWidth, boxHeight) in oriented full-frame UV [0..1]
  boxRect: vec4<f32>,
  // (rightEyeU, rightEyeV, leftEyeU, leftEyeV)
  eyes: vec4<f32>,
  // (noseU, noseV, 0, 0)
  nose: vec4<f32>,
  // (rightMouthU, rightMouthV, leftMouthU, leftMouthV)
  mouth: vec4<f32>,
};

struct PortraitRetouchUniforms {
  // (outW, outH, oriW, oriH)
  dims: vec4<f32>,
  // (left, top, width, height) in oriented full-frame [0..1]
  cropRect: vec4<f32>,
  // (spatialScale, faceCount, oriW/oriH, hasFaceParseMap)
  info: vec4<f32>,
  // (skinSmoothing, skinTexturePreserve, skinToneEvenness, shineRemoval) in [0..1]
  skinParams: vec4<f32>,
  // (underEyeBrighten, eyeEnhance, teethWhiten, lipEnhance) in [0..1]
  featureParams: vec4<f32>,
  faces: array<PortraitFaceGpu, 4>,
};

@group(0) @binding(0) var<uniform> u: PortraitRetouchUniforms;
@group(0) @binding(1) var srcTex: texture_2d<f32>;
@group(0) @binding(2) var dstTex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var faceParseTex: texture_2d<f32>;

const RING8: array<vec2<f32>, 8> = array<vec2<f32>, 8>(
  vec2<f32>(1.0, 0.0),
  vec2<f32>(0.70710678, 0.70710678),
  vec2<f32>(0.0, 1.0),
  vec2<f32>(-0.70710678, 0.70710678),
  vec2<f32>(-1.0, 0.0),
  vec2<f32>(-0.70710678, -0.70710678),
  vec2<f32>(0.0, -1.0),
  vec2<f32>(0.70710678, -0.70710678),
);

fn kSmoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
  let t = clamp((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

fn luma709(c: vec3<f32>) -> f32 {
  return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
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
  let p00 = textureLoad(srcTex, vec2<i32>(x0, y0), 0).rgb;
  let p10 = textureLoad(srcTex, vec2<i32>(x1, y0), 0).rgb;
  let p01 = textureLoad(srcTex, vec2<i32>(x0, y1), 0).rgb;
  let p11 = textureLoad(srcTex, vec2<i32>(x1, y1), 0).rgb;
  return mix(mix(p00, p10, tx), mix(p01, p11, tx), ty);
}

fn sampleFaceParseBilinear(uv: vec2<f32>) -> vec4<f32> {
  let dims = textureDimensions(faceParseTex, 0);
  let mw = f32(dims.x);
  let mh = f32(dims.y);
  let maxX = max(mw - 1.0, 0.0);
  let maxY = max(mh - 1.0, 0.0);
  let fx = clamp(uv.x * mw - 0.5, 0.0, maxX);
  let fy = clamp(uv.y * mh - 0.5, 0.0, maxY);
  let x0 = i32(floor(fx));
  let y0 = i32(floor(fy));
  let x1 = min(x0 + 1, i32(maxX));
  let y1 = min(y0 + 1, i32(maxY));
  let tx = fx - f32(x0);
  let ty = fy - f32(y0);
  let p00 = textureLoad(faceParseTex, vec2<i32>(x0, y0), 0);
  let p10 = textureLoad(faceParseTex, vec2<i32>(x1, y0), 0);
  let p01 = textureLoad(faceParseTex, vec2<i32>(x0, y1), 0);
  let p11 = textureLoad(faceParseTex, vec2<i32>(x1, y1), 0);
  return mix(mix(p00, p10, tx), mix(p01, p11, tx), ty);
}

// Inclusive perceptual YCbCr skin likelihood covering fair, olive, tan, brown, and deep melanin skin.
fn ycbcrSkinProbability(rgb: vec3<f32>) -> f32 {
  let y = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
  let cb = 0.5 - 0.168736 * rgb.r - 0.331264 * rgb.g + 0.5 * rgb.b;
  let cr = 0.5 + 0.5 * rgb.r - 0.418688 * rgb.g - 0.081312 * rgb.b;

  let yGate = kSmoothstep(0.06, 0.16, y) * (1.0 - kSmoothstep(0.95, 0.995, y));
  let cbGate = kSmoothstep(0.24, 0.32, cb) * (1.0 - kSmoothstep(0.49, 0.55, cb));
  let crGate = kSmoothstep(0.505, 0.54, cr) * (1.0 - kSmoothstep(0.70, 0.76, cr));
  let warmthGate = kSmoothstep(-0.015, 0.035, rgb.r - rgb.b);
  return clamp(yGate * cbGate * crGate * warmthGate, 0.0, 1.0);
}

fn rotatedEllipseMask(
  p: vec2<f32>,
  center: vec2<f32>,
  ax: vec2<f32>,
  ay: vec2<f32>,
  rx: f32,
  ry: f32,
  innerRatio: f32,
) -> f32 {
  let d = p - center;
  let lx = dot(d, ax) / max(rx, 1e-5);
  let ly = dot(d, ay) / max(ry, 1e-5);
  let r = sqrt(lx * lx + ly * ly);
  return 1.0 - kSmoothstep(innerRatio, 1.0, r);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outW = u32(u.dims.x);
  let outH = u32(u.dims.y);
  if (gid.x >= outW || gid.y >= outH) {
    return;
  }

  let coord = vec2<i32>(i32(gid.x), i32(gid.y));
  let center = textureLoad(srcTex, coord, 0);
  var rgb = center.rgb;

  let skinSmooth = u.skinParams.x;
  let skinTexPreserve = u.skinParams.y;
  let skinEven = u.skinParams.z;
  let shineAmt = u.skinParams.w;

  let underEyeAmt = u.featureParams.x;
  let eyeAmt = u.featureParams.y;
  let teethAmt = u.featureParams.z;
  let lipAmt = u.featureParams.w;

  let anySkin = skinSmooth > 1e-4 || skinEven > 1e-4 || shineAmt > 1e-4;
  let anyFeature = underEyeAmt > 1e-4 || eyeAmt > 1e-4 || teethAmt > 1e-4 || lipAmt > 1e-4;
  if (!anySkin && !anyFeature) {
    textureStore(dstTex, coord, vec4<f32>(rgb, center.a));
    return;
  }

  let oriW = max(u.dims.z, 1.0);
  let oriH = max(u.dims.w, 1.0);
  let minSide = min(oriW, oriH);
  let scaleUv = vec2<f32>(oriW / minSide, oriH / minSide);

  let uCrop = (f32(gid.x) + 0.5) / u.dims.x;
  let vCrop = (f32(gid.y) + 0.5) / u.dims.y;
  let uv = vec2<f32>(
    u.cropRect.x + uCrop * u.cropRect.z,
    u.cropRect.y + vCrop * u.cropRect.w,
  );
  let pS = uv * scaleUv;

  let faceCount = min(u32(u.info.y), 4u);
  var faceEnv: f32 = 0.0;
  var featureProtect: f32 = 0.0;
  var underEyeMask: f32 = 0.0;
  var eyeMask: f32 = 0.0;
  var mouthMask: f32 = 0.0;

  for (var f: u32 = 0u; f < faceCount; f = f + 1u) {
    let face = u.faces[f];
    let rEye = face.eyes.xy * scaleUv;
    let lEye = face.eyes.zw * scaleUv;
    let nose = face.nose.xy * scaleUv;
    let rMouth = face.mouth.xy * scaleUv;
    let lMouth = face.mouth.zw * scaleUv;

    let eVec = lEye - rEye;
    let eyeDist = max(length(eVec), 0.02);
    let ax = select(vec2<f32>(1.0, 0.0), eVec / eyeDist, length(eVec) > 1e-4);
    var ay = vec2<f32>(-ax.y, ax.x);
    let eyeMid = 0.5 * (rEye + lEye);
    let mouthMid = 0.5 * (rMouth + lMouth);
    // Guarantee `ay` points downward from eye line toward mouth (handles flipped/inverted poses)
    if (dot(mouthMid - eyeMid, ay) < 0.0) {
      ay = -ay;
    }

    // Broad face + cheek + forehead + neck envelope
    let faceCenter = eyeMid + ay * (0.28 * eyeDist);
    let fHead = rotatedEllipseMask(pS, faceCenter, ax, ay, 1.32 * eyeDist, 1.72 * eyeDist, 0.52);
    let neckCenter = eyeMid + ay * (1.65 * eyeDist);
    let fNeck = rotatedEllipseMask(pS, neckCenter, ax, ay, 1.02 * eyeDist, 1.05 * eyeDist, 0.35);
    faceEnv = max(faceEnv, max(fHead, fNeck * 0.85));

    // Feature protection masks (keep eyes, brows, nostrils, and lips sharp during skin smoothing)
    let rEyeProt = rotatedEllipseMask(pS, rEye, ax, ay, 0.28 * eyeDist, 0.16 * eyeDist, 0.45);
    let lEyeProt = rotatedEllipseMask(pS, lEye, ax, ay, 0.28 * eyeDist, 0.16 * eyeDist, 0.45);
    let rBrowProt = rotatedEllipseMask(
      pS,
      rEye - ay * (0.24 * eyeDist),
      ax,
      ay,
      0.32 * eyeDist,
      0.12 * eyeDist,
      0.4,
    );
    let lBrowProt = rotatedEllipseMask(
      pS,
      lEye - ay * (0.24 * eyeDist),
      ax,
      ay,
      0.32 * eyeDist,
      0.12 * eyeDist,
      0.4,
    );
    let noseProt = rotatedEllipseMask(pS, nose, ax, ay, 0.15 * eyeDist, 0.11 * eyeDist, 0.4);

    let mVec = lMouth - rMouth;
    let mouthW = max(length(mVec), eyeDist * 0.55);
    let mAx = select(ax, mVec / length(mVec), length(mVec) > 1e-4);
    var mAy = vec2<f32>(-mAx.y, mAx.x);
    if (dot(mAy, ay) < 0.0) {
      mAy = -mAy;
    }
    let mProt = rotatedEllipseMask(pS, mouthMid, mAx, mAy, 0.58 * mouthW, 0.25 * mouthW, 0.45);
    featureProtect = max(
      featureProtect,
      max(max(rEyeProt, lEyeProt), max(max(rBrowProt, lBrowProt), max(noseProt, mProt))),
    );

    // Targeted Eye opening mask (iris + sclera)
    let rEyeM = rotatedEllipseMask(pS, rEye, ax, ay, 0.24 * eyeDist, 0.12 * eyeDist, 0.35);
    let lEyeM = rotatedEllipseMask(pS, lEye, ax, ay, 0.24 * eyeDist, 0.12 * eyeDist, 0.35);
    eyeMask = max(eyeMask, max(rEyeM, lEyeM));

    // Targeted Under-Eye crescent (below each eye, excluding the eye itself)
    let rUe = rotatedEllipseMask(
      pS,
      rEye + ay * (0.22 * eyeDist),
      ax,
      ay,
      0.36 * eyeDist,
      0.14 * eyeDist,
      0.25,
    ) * (1.0 - rEyeProt);
    let lUe = rotatedEllipseMask(
      pS,
      lEye + ay * (0.22 * eyeDist),
      ax,
      ay,
      0.36 * eyeDist,
      0.14 * eyeDist,
      0.25,
    ) * (1.0 - lEyeProt);
    underEyeMask = max(underEyeMask, max(rUe, lUe));

    // Targeted Mouth / Teeth / Lips mask
    let mMask = rotatedEllipseMask(pS, mouthMid, mAx, mAy, 0.50 * mouthW, 0.22 * mouthW, 0.35);
    mouthMask = max(mouthMask, mMask);
  }

  let skinProb = ycbcrSkinProbability(rgb);
  let hasFaceParse = u.info.w > 0.5;
  var skinMask = select(
    skinProb * 0.85,
    skinProb * faceEnv * (1.0 - featureProtect * 0.92),
    faceCount > 0u,
  );
  var teethMask = mouthMask;
  var lipMask = mouthMask;

  // Phase 3 Opt-In BiSeNet 19-Class Neural Face Parsing upgrade:
  // Replaces geometric ellipses with pixel-exact neural masks for skin (excluding eyeglasses,
  // bangs, hair, brows, eyes, mouth, lips), eyes, inner mouth/teeth, and upper/lower lips.
  if (hasFaceParse) {
    let fp = sampleFaceParseBilinear(uv);
    skinMask = clamp(fp.r, 0.0, 1.0);
    eyeMask = clamp(fp.g, 0.0, 1.0);
    teethMask = clamp(fp.b, 0.0, 1.0);
    lipMask = clamp(fp.a, 0.0, 1.0);
    underEyeMask = underEyeMask * clamp(fp.r * 1.15, 0.0, 1.0) * (1.0 - eyeMask);
  }

  let needSpatialSkin = (anySkin && skinMask > 0.005) || (underEyeAmt > 1e-4 && underEyeMask > 0.005);
  let needFineDetail = needSpatialSkin || (eyeAmt > 1e-4 && eyeMask > 0.005) || (lipAmt > 1e-4 && lipMask > 0.005);

  if (!needFineDetail && !(teethAmt > 1e-4 && teethMask > 0.005)) {
    textureStore(dstTex, coord, vec4<f32>(rgb, center.a));
    return;
  }

  let fx = f32(gid.x);
  let fy = f32(gid.y);
  let scale = u.info.x;

  // Fine-scale local average separates real high-frequency skin pores & eyelash texture
  var fineLow = rgb;
  if (needFineDetail) {
    let rFine = 1.15 * scale;
    let f1 = sampleRgbAtPixel(fx - rFine, fy);
    let f2 = sampleRgbAtPixel(fx + rFine, fy);
    let f3 = sampleRgbAtPixel(fx, fy - rFine);
    let f4 = sampleRgbAtPixel(fx, fy + rFine);
    fineLow = (rgb * 2.0 + f1 + f2 + f3 + f4) / 6.0;
  }
  let highFreqPores = rgb - fineLow;

  // 2-ring Edge-Preserving Bilateral Low-Frequency Surface (16 circular taps)
  var lowFreq = rgb;
  if (needSpatialSkin) {
    let r1 = 2.6 * scale;
    let r2 = 5.6 * scale;
    let sigmaColor = 0.095 + 0.085 * max(skinSmooth, max(skinEven, underEyeAmt));
    let inv2Sig2 = -0.5 / (sigmaColor * sigmaColor);

    var acc = rgb * 1.4;
    var wSum: f32 = 1.4;

    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
      let dir = RING8[k];
      let s1 = sampleRgbAtPixel(fx + dir.x * r1, fy + dir.y * r1);
      let d1 = s1 - rgb;
      let w1 = 0.85 * exp(dot(d1, d1) * inv2Sig2);
      acc = acc + s1 * w1;
      wSum = wSum + w1;

      let s2 = sampleRgbAtPixel(fx + dir.x * r2, fy + dir.y * r2);
      let d2 = s2 - rgb;
      let w2 = 0.55 * exp(dot(d2, d2) * inv2Sig2);
      acc = acc + s2 * w2;
      wSum = wSum + w2;
    }
    lowFreq = acc / max(wSum, 1e-5);
  }

  let midFreqBlotch = fineLow - lowFreq;

  // 1. True 3-Band Frequency-Separation Skin Smoothing (removes blotchiness, keeps real pores)
  if (skinSmooth > 1e-4 && skinMask > 0.005) {
    let poreKeep = mix(1.0 - skinSmooth * 0.55, 1.0, skinTexPreserve);
    let fsRgb = lowFreq + midFreqBlotch * (1.0 - skinSmooth * 0.88) + highFreqPores * poreKeep;
    let blendW = clamp(skinMask * skinSmooth * 1.05, 0.0, 1.0);
    rgb = mix(rgb, clamp(fsRgb, vec3<f32>(0.0), vec3<f32>(1.0)), blendW);
  }

  // 2. Skin Tone Evenness (softens red/magenta/yellow blotches toward local low-freq skin tone)
  if (skinEven > 1e-4 && skinMask > 0.005) {
    let curL = luma709(rgb);
    let lowL = luma709(lowFreq);
    let curChroma = rgb - vec3<f32>(curL);
    let lowChroma = lowFreq - vec3<f32>(lowL);
    let evenedChroma = mix(curChroma, lowChroma, skinMask * skinEven * 0.65);
    // Gently calm localized excess redness spikes (acne/rosacea warmth) above smooth skin
    let redSpike = max(0.0, (rgb.r - rgb.g) - (lowFreq.r - lowFreq.g) - 0.015);
    var evenRgb = vec3<f32>(curL) + evenedChroma;
    evenRgb.r = evenRgb.r - redSpike * skinMask * skinEven * 0.45;
    evenRgb.g = evenRgb.g + redSpike * skinMask * skinEven * 0.14;
    rgb = clamp(evenRgb, vec3<f32>(0.0), vec3<f32>(1.0));
  }

  // 3. Specular Shine / Oil Control (tames hot bleached forehead/nose glare)
  if (shineAmt > 1e-4 && skinMask > 0.005) {
    let curL = luma709(rgb);
    let maxC = max(rgb.r, max(rgb.g, rgb.b));
    let minC = min(rgb.r, min(rgb.g, rgb.b));
    let sat = (maxC - minC) / max(maxC, 1e-4);
    let specWeight = kSmoothstep(0.60, 0.86, curL) * (1.0 - kSmoothstep(0.14, 0.38, sat));
    let lowL = luma709(lowFreq);
    let matteTarget = clamp(
      vec3<f32>(lowL * 0.92) + (lowFreq - vec3<f32>(lowL)) * 1.18 + highFreqPores * 0.5,
      vec3<f32>(0.0),
      vec3<f32>(1.0),
    );
    rgb = mix(rgb, matteTarget, clamp(skinMask * specWeight * shineAmt * 0.78, 0.0, 0.85));
  }

  // 4. Landmark-Guided Under-Eye Dark-Circle Brighten & Softening
  if (underEyeAmt > 1e-4 && underEyeMask > 0.005) {
    let ueW = underEyeMask * clamp(skinProb * 1.15, 0.0, 1.0) * underEyeAmt;
    if (ueW > 1e-4) {
      let curL = luma709(rgb);
      let shadowLift = 0.115 * ueW * (1.0 - kSmoothstep(0.45, 0.88, curL));
      let softened = mix(rgb, lowFreq + highFreqPores * 0.75, clamp(ueW * 0.45, 0.0, 0.6));
      let lifted = softened + vec3<f32>(shadowLift * 1.06, shadowLift * 1.0, shadowLift * 0.86);
      rgb = clamp(lifted, vec3<f32>(0.0), vec3<f32>(1.0));
    }
  }

  // 5. Landmark-Guided Eye Enhance (Natural Sclera Brighten + Iris Micro-Contrast)
  if (eyeAmt > 1e-4 && eyeMask > 0.005) {
    let eW = eyeMask * eyeAmt;
    let curL = luma709(rgb);
    let maxC = max(rgb.r, max(rgb.g, rgb.b));
    let minC = min(rgb.r, min(rgb.g, rgb.b));
    let sat = (maxC - minC) / max(maxC, 1e-4);
    let scleraW = kSmoothstep(0.42, 0.72, curL) * (1.0 - kSmoothstep(0.10, 0.28, sat));

    // Clean eye whites gently without looking artificial
    let neutralWhite = mix(rgb, vec3<f32>(curL), scleraW * eW * 0.48) + vec3<f32>(0.065 * scleraW * eW * (1.0 - curL * 0.45));
    // Crisp iris / lash micro-contrast + subtle iris luminance lift
    let irisW = 1.0 - scleraW;
    let irisPop = highFreqPores * (eW * 1.15) + vec3<f32>(0.042 * irisW * eW * kSmoothstep(0.08, 0.45, curL));
    rgb = clamp(neutralWhite + irisPop, vec3<f32>(0.0), vec3<f32>(1.0));
  }

  // 6. Landmark/BiSeNet-Guided Teeth Whitening (strictly targets bright non-reddish tooth pixels)
  if (teethAmt > 1e-4 && teethMask > 0.005) {
    let curL = luma709(rgb);
    let maxC = max(rgb.r, max(rgb.g, rgb.b));
    let minC = min(rgb.r, min(rgb.g, rgb.b));
    let sat = (maxC - minC) / max(maxC, 1e-4);
    let nonLipGate = select(
      1.0 - kSmoothstep(0.06, 0.15, rgb.r - rgb.g),
      1.0 - kSmoothstep(0.10, 0.22, rgb.r - rgb.g),
      hasFaceParse,
    );
    let brightGate = kSmoothstep(0.32, 0.58, curL);
    let satGate = 1.0 - kSmoothstep(0.24, 0.46, sat);
    let teethW = teethMask * nonLipGate * brightGate * satGate * teethAmt;
    if (teethW > 1e-4) {
      let desat = mix(rgb, vec3<f32>(curL), clamp(teethW * 0.72, 0.0, 0.85));
      let lifted = desat + vec3<f32>(0.085 * teethW * (1.0 - curL * 0.55));
      rgb = clamp(lifted, vec3<f32>(0.0), vec3<f32>(1.0));
    }
  }

  // 7. Landmark/BiSeNet-Guided Lip Enhance (healthy natural color richness on rosy lip pixels only)
  if (lipAmt > 1e-4 && lipMask > 0.005) {
    let curL = luma709(rgb);
    let maxC = max(rgb.r, max(rgb.g, rgb.b));
    let minC = min(rgb.r, min(rgb.g, rgb.b));
    let sat = (maxC - minC) / max(maxC, 1e-4);
    let lipHueGate = select(kSmoothstep(0.055, 0.17, rgb.r - rgb.g), 1.0, hasFaceParse);
    let lipSatGate = select(kSmoothstep(0.10, 0.28, sat), kSmoothstep(0.04, 0.14, sat), hasFaceParse);
    let lipW = lipMask * lipHueGate * lipSatGate * lipAmt;
    if (lipW > 1e-4) {
      let boostedChroma = (rgb - vec3<f32>(curL)) * (1.0 + 0.30 * lipW);
      let lipRgb = vec3<f32>(curL * (1.0 - 0.04 * lipW)) + boostedChroma + highFreqPores * (0.28 * lipW);
      rgb = clamp(lipRgb, vec3<f32>(0.0), vec3<f32>(1.0));
    }
  }

  textureStore(dstTex, coord, vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), center.a));
}
