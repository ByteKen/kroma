// Unified 32-Bit Non-Destructive Generative Inpaint Patch Composite Pass (Phase 3).
// Shared verbatim between Kroma WebGPU Studio Preview and Rust `wgpu` Batch Export (DX12 / Metal).
// Mirrors `apply_inpaint_patches_f32` in `internal/infrastructure/src/develop/retouch_engine.rs`.
//
// Composites a compact LaMa / Harmonic `InpaintPatch` RGBA texture (`rgb = patch`, `a = feathered mask`)
// into its tight `bounds` in oriented full-frame UV space with 8-tap perimeter lighting adaptation
// so subsequent Exposure, White Balance, or LUT edits stay seamlessly tone-matched.

struct InpaintUniforms {
  // (outW, outH, patchW, patchH)
  dims: vec4<f32>,
  // (left, top, width, height) in oriented full-frame [0..1]
  cropRect: vec4<f32>,
  // Patch bounds in oriented full-frame [0..1]: (left, top, width, height)
  bounds: vec4<f32>,
  // (opacity, 0, 0, 0)
  params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: InpaintUniforms;
@group(0) @binding(1) var srcTex: texture_2d<f32>;
@group(0) @binding(2) var patchTex: texture_2d<f32>;
@group(0) @binding(3) var dstTex: texture_storage_2d<rgba32float, write>;

const BORDER_TAPS: array<vec2<f32>, 8> = array<vec2<f32>, 8>(
  vec2<f32>(0.02, 0.02),
  vec2<f32>(0.50, 0.02),
  vec2<f32>(0.98, 0.02),
  vec2<f32>(0.98, 0.50),
  vec2<f32>(0.98, 0.98),
  vec2<f32>(0.50, 0.98),
  vec2<f32>(0.02, 0.98),
  vec2<f32>(0.02, 0.50),
);

fn sampleSrcBilinearUv(uv: vec2<f32>) -> vec3<f32> {
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

fn samplePatchBilinear(lu: f32, lv: f32) -> vec4<f32> {
  let pw = max(u.dims.z, 1.0);
  let ph = max(u.dims.w, 1.0);
  let maxX = max(pw - 1.0, 0.0);
  let maxY = max(ph - 1.0, 0.0);
  let fx = clamp(lu * pw - 0.5, 0.0, maxX);
  let fy = clamp(lv * ph - 0.5, 0.0, maxY);
  let x0 = i32(floor(fx));
  let y0 = i32(floor(fy));
  let x1 = min(x0 + 1, i32(maxX));
  let y1 = min(y0 + 1, i32(maxY));
  let tx = fx - f32(x0);
  let ty = fy - f32(y0);
  let p00 = textureLoad(patchTex, vec2<i32>(x0, y0), 0);
  let p10 = textureLoad(patchTex, vec2<i32>(x1, y0), 0);
  let p01 = textureLoad(patchTex, vec2<i32>(x0, y1), 0);
  let p11 = textureLoad(patchTex, vec2<i32>(x1, y1), 0);
  return mix(mix(p00, p10, tx), mix(p01, p11, tx), ty);
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

  let uCrop = (f32(gid.x) + 0.5) / u.dims.x;
  let vCrop = (f32(gid.y) + 0.5) / u.dims.y;
  let uv = vec2<f32>(
    u.cropRect.x + uCrop * u.cropRect.z,
    u.cropRect.y + vCrop * u.cropRect.w,
  );

  // Fast AABB reject outside patch bounds
  let lu = (uv.x - u.bounds.x) / max(u.bounds.z, 1e-6);
  let lv = (uv.y - u.bounds.y) / max(u.bounds.w, 1e-6);
  if (lu < 0.0 || lu > 1.0 || lv < 0.0 || lv > 1.0) {
    textureStore(dstTex, coord, centerPx);
    return;
  }

  let patchSample = samplePatchBilinear(lu, lv);
  let alpha = clamp(patchSample.a * u.params.x, 0.0, 1.0);
  if (alpha <= 1e-5) {
    textureStore(dstTex, coord, centerPx);
    return;
  }

  // Live 8-tap perimeter color-adaptation keeps patches seamless even after Exposure/WB/LUT edits
  var shiftSum = vec3<f32>(0.0);
  var wSum: f32 = 0.0;
  for (var k: u32 = 0u; k < 8u; k = k + 1u) {
    let tap = BORDER_TAPS[k];
    let pTap = samplePatchBilinear(tap.x, tap.y);
    let borderW = 1.0 - clamp(pTap.a * 4.0, 0.0, 1.0);
    if (borderW > 0.05) {
      let tapUv = vec2<f32>(
        u.bounds.x + tap.x * u.bounds.z,
        u.bounds.y + tap.y * u.bounds.w,
      );
      let liveRgb = sampleSrcBilinearUv(tapUv);
      let d = vec2<f32>(lu - tap.x, lv - tap.y);
      let invW = borderW / (dot(d, d) + 0.12);
      shiftSum = shiftSum + (liveRgb - pTap.rgb) * invW;
      wSum = wSum + invW;
    }
  }

  var adaptedPatch = patchSample.rgb;
  if (wSum > 1e-5) {
    let shift = clamp(shiftSum / wSum, vec3<f32>(-0.30), vec3<f32>(0.30));
    adaptedPatch = clamp(patchSample.rgb + shift, vec3<f32>(0.0), vec3<f32>(1.0));
  }

  let outRgb = mix(centerPx.rgb, adaptedPatch, alpha);
  textureStore(dstTex, coord, vec4<f32>(clamp(outRgb, vec3<f32>(0.0), vec3<f32>(1.0)), centerPx.a));
}
