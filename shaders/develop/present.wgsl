// Lightweight full-screen triangle WebGPU present shader for Studio live preview.
// Blits the final 32-bit `rgba32float` graded texture to the swapchain with object-fit UV
// mapping, optional transparent cutout checkerboard, and Lightroom Shadow/Highlight Clipping.

struct PresentUniforms {
  // (scaleX, scaleY, offsetX, offsetY)
  uvTransform: vec4<f32>,
  // (showShadowClip, showHighlightClip, showCheckerboard, 0)
  flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: PresentUniforms;
@group(0) @binding(1) var finalTex: texture_2d<f32>;

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) texCoord: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  let p = positions[idx];
  var vsOut: VsOut;
  vsOut.pos = vec4<f32>(p, 0.0, 1.0);
  // Flip Y so top-left of viewport is texCoord (0, 0) matching image coordinates
  vsOut.texCoord = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  return vsOut;
}

fn sampleBilinear(uv: vec2<f32>) -> vec4<f32> {
  let dims = vec2<f32>(textureDimensions(finalTex, 0));
  let maxCoord = max(dims - vec2<f32>(1.0), vec2<f32>(0.0));
  let f = clamp(uv * dims - vec2<f32>(0.5), vec2<f32>(0.0), maxCoord);
  let p0 = vec2<i32>(floor(f));
  let p1 = min(p0 + vec2<i32>(1, 1), vec2<i32>(maxCoord));
  let t = f - vec2<f32>(p0);
  let c00 = textureLoad(finalTex, p0, 0);
  let c10 = textureLoad(finalTex, vec2<i32>(p1.x, p0.y), 0);
  let c01 = textureLoad(finalTex, vec2<i32>(p0.x, p1.y), 0);
  let c11 = textureLoad(finalTex, p1, 0);
  return mix(mix(c00, c10, t.x), mix(c01, c11, t.x), t.y);
}

@fragment
fn fs_main(fragIn: VsOut) -> @location(0) vec4<f32> {
  let uv = fragIn.texCoord * u.uvTransform.xy + u.uvTransform.zw;
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
  }

  let sampled = sampleBilinear(uv);
  var rgb = clamp(sampled.rgb, vec3<f32>(0.0), vec3<f32>(1.0));

  if (u.flags.z > 0.5) {
    let chk = floor(fragIn.pos.xy / 12.0);
    let pat = (chk.x + chk.y) - 2.0 * floor((chk.x + chk.y) / 2.0);
    let backdrop = mix(vec3<f32>(0.18, 0.19, 0.21), vec3<f32>(0.28, 0.29, 0.32), pat);
    rgb = mix(backdrop, rgb, clamp(sampled.a, 0.0, 1.0));
  }

  if (u.flags.y > 0.5 && max(rgb.r, max(rgb.g, rgb.b)) >= 0.992) {
    rgb = vec3<f32>(0.94, 0.27, 0.27);
  } else if (u.flags.x > 0.5 && min(rgb.r, min(rgb.g, rgb.b)) <= 0.008) {
    rgb = vec3<f32>(0.23, 0.51, 0.96);
  }

  return vec4<f32>(rgb, 1.0);
}
