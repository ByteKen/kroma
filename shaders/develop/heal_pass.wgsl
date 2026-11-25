// Unified 32-Bit Non-Destructive Spot Heal & Clone Compute Pass (v0.3.0).
// Shared verbatim between Kroma WebGPU Studio Preview and Rust `wgpu` Batch Export (DX12 / Metal).
// Mirrors `apply_heal_spots_f32` in `internal/infrastructure/src/develop/retouch_engine.rs`.
//
// Uses closed-form inverse-distance harmonic boundary matching (8-tap clean ring around the
// destination perimeter vs matching donor ring) so Heal mode transfers crisp natural skin pores
// while seamlessly matching 2D lighting gradients across cheeks, foreheads, and studio backdrops.

struct HealUniforms {
  // (outW, outH, oriW, oriH)
  dims: vec4<f32>,
  // (left, top, width, height) in oriented full-frame [0..1]
  cropRect: vec4<f32>,
  // (spotCount, spatialScale, 0, 0)
  info: vec4<f32>,
};

struct HealSpotGpu {
  // (deltaU, deltaV, radiusNorm, feather)
  geom: vec4<f32>,
  // (opacity, mode: 0=heal 1=clone, segStart, segCount)
  params: vec4<f32>,
  // Tight AABB in oriented full-frame UV: (minU, minV, maxU, maxV)
  bounds: vec4<f32>,
  // (targetX, targetY, 0, 0)
  center: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: HealUniforms;
@group(0) @binding(1) var srcTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> spots: array<HealSpotGpu>;
@group(0) @binding(3) var<storage, read> segments: array<vec4<f32>>;
@group(0) @binding(4) var dstTex: texture_storage_2d<rgba32float, write>;

const RING_DIRS: array<vec2<f32>, 8> = array<vec2<f32>, 8>(
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

fn sampleBilinearUv(uv: vec2<f32>) -> vec3<f32> {
  let outW = u.dims.x;
  let outH = u.dims.y;
  let maxX = max(outW - 1.0, 0.0);
  let maxY = max(outH - 1.0, 0.0);
  let fx = clamp(((uv.x - u.cropRect.x) / max(u.cropRect.z, 1e-6)) * outW - 0.5, 0.0, maxX);
  let fy = clamp(((uv.y - u.cropRect.y) / max(u.cropRect.w, 1e-6)) * outH - 0.5, 0.0, maxY);
  let x0 = i32(floor(fx));
  let y0 = i32(floor(fy));
  let x1 = min(x0 + 1, i32(maxX));
  let y1 = min(y0 + 1, i32(maxY));
  let tx = fx - f32(x0);
  let ty = fy - f32(y0);
  let p00 = textureLoad(srcTex, vec2<i32>(x0, y0), 0).rgb;
  let p10 = textureLoad(srcTex, vec2<i32>(x1, y0), 0).rgb;
  let p01 = textureLoad(srcTex, vec2<i32>(x0, y1), 0).rgb;
  let p11 = textureLoad(srcTex, vec2<i32>(x1, y1), 0).rgb;
  return mix(mix(p00, p10, tx), mix(p01, p11, tx), ty);
}

// Computes aspect-correct distance from `uv` to segment `(a, b)` plus closest centerline UV.
fn pointToSegmentInfo(
  uv: vec2<f32>,
  a: vec2<f32>,
  b: vec2<f32>,
  scaleUv: vec2<f32>,
) -> vec3<f32> {
  let pS = uv * scaleUv;
  let aS = a * scaleUv;
  let bS = b * scaleUv;
  let ab = bS - aS;
  let lenSq = dot(ab, ab);
  var t: f32 = 0.0;
  if (lenSq > 1e-10) {
    t = clamp(dot(pS - aS, ab) / lenSq, 0.0, 1.0);
  }
  let qUv = mix(a, b, t);
  let dist = length((uv - qUv) * scaleUv);
  return vec3<f32>(dist, qUv.x, qUv.y);
}

fn minDistToStroke(
  uv: vec2<f32>,
  segStart: u32,
  segCount: u32,
  scaleUv: vec2<f32>,
) -> f32 {
  var best: f32 = 1e6;
  for (var k: u32 = 0u; k < segCount; k = k + 1u) {
    let seg = segments[segStart + k];
    let info = pointToSegmentInfo(uv, seg.xy, seg.zw, scaleUv);
    best = min(best, info.x);
  }
  return best;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outW = u32(u.dims.x);
  let outH = u32(u.dims.y);
  if (gid.x >= outW || gid.y >= outH) {
    return;
  }

  let coord = vec2<i32>(i32(gid.x), i32(gid.y));
  let centerPx = textureLoad(srcTex, coord, 0);
  var rgb = centerPx.rgb;

  let spotCount = u32(u.info.x);
  if (spotCount == 0u) {
    textureStore(dstTex, coord, vec4<f32>(rgb, centerPx.a));
    return;
  }

  let oriW = max(u.dims.z, 1.0);
  let oriH = max(u.dims.w, 1.0);
  let minSide = min(oriW, oriH);
  let scaleUv = vec2<f32>(oriW / minSide, oriH / minSide);
  let invScaleUv = vec2<f32>(minSide / oriW, minSide / oriH);

  let uCrop = (f32(gid.x) + 0.5) / u.dims.x;
  let vCrop = (f32(gid.y) + 0.5) / u.dims.y;
  let uv = vec2<f32>(
    u.cropRect.x + uCrop * u.cropRect.z,
    u.cropRect.y + vCrop * u.cropRect.w,
  );

  for (var s: u32 = 0u; s < spotCount; s = s + 1u) {
    let spot = spots[s];
    // Fast AABB reject skips 99%+ of pixels in 4 comparisons
    if (
      uv.x < spot.bounds.x ||
      uv.x > spot.bounds.z ||
      uv.y < spot.bounds.y ||
      uv.y > spot.bounds.w
    ) {
      continue;
    }

    let radius = spot.geom.z;
    let feather = spot.geom.w;
    let opacity = spot.params.x;
    let isClone = spot.params.y > 0.5;
    let segStart = u32(spot.params.z);
    let segCount = max(u32(spot.params.w), 1u);

    var bestDist: f32 = 1e6;
    var closestUv = spot.center.xy;
    for (var k: u32 = 0u; k < segCount; k = k + 1u) {
      let seg = segments[segStart + k];
      let info = pointToSegmentInfo(uv, seg.xy, seg.zw, scaleUv);
      if (info.x < bestDist) {
        bestDist = info.x;
        closestUv = vec2<f32>(info.y, info.z);
      }
    }

    if (bestDist >= radius) {
      continue;
    }

    var mask: f32 = 1.0;
    if (feather > 1e-4) {
      let innerR = radius * (1.0 - feather * 0.85);
      mask = 1.0 - kSmoothstep(innerR, radius, bestDist);
    }
    let alpha = clamp(mask * opacity, 0.0, 1.0);
    if (alpha <= 1e-5) {
      continue;
    }

    let delta = spot.geom.xy;
    let donorRgb = sampleBilinearUv(uv + delta);

    var patchRgb = donorRgb;
    if (!isClone) {
      // Closed-form discrete harmonic boundary tone/color matching from 8 clean perimeter taps
      let ringR = radius * 1.28;
      let regEps = 0.18 * radius * radius;
      var shiftSum = vec3<f32>(0.0);
      var wSum: f32 = 0.0;

      for (var d: u32 = 0u; d < 8u; d = d + 1u) {
        let ringDstUv = closestUv + RING_DIRS[d] * ringR * invScaleUv;
        var outsideWeight: f32 = 1.0;
        if (segCount > 1u) {
          let strokeDist = minDistToStroke(ringDstUv, segStart, segCount, scaleUv);
          outsideWeight = kSmoothstep(radius * 0.82, radius * 1.12, strokeDist);
        }
        if (outsideWeight <= 1e-4) {
          continue;
        }
        let ringSrcUv = ringDstUv + delta;
        let dstRingCol = sampleBilinearUv(ringDstUv);
        let srcRingCol = sampleBilinearUv(ringSrcUv);
        let pToRing = (uv - ringDstUv) * scaleUv;
        let invDistW = outsideWeight / (dot(pToRing, pToRing) + regEps);
        shiftSum = shiftSum + (dstRingCol - srcRingCol) * invDistW;
        wSum = wSum + invDistW;
      }

      if (wSum > 1e-6) {
        let shift = clamp(shiftSum / wSum, vec3<f32>(-0.35), vec3<f32>(0.35));
        patchRgb = clamp(donorRgb + shift, vec3<f32>(0.0), vec3<f32>(1.0));
      }
    }

    rgb = mix(rgb, patchRgb, alpha);
  }

  textureStore(dstTex, coord, vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), centerPx.a));
}
