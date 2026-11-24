import { memo, useEffect, useId, useMemo, useRef, useState } from 'react';
import { recipeToCssFilter, type DevelopRecipe } from '../../domain/develop';
import {
  buildToneCurveLutRgba,
  compileDevelopShaderUniforms,
  compressRec2020ToLinearSrgb,
  linearToSrgb,
  softClip,
} from '../../domain/developShaderMath';
import {
  computeViewportUvTransform,
  ZERO_VIEWPORT_PAN,
  type ViewportPan,
} from '../../domain/viewportZoom';
import type { DevelopWorkingBuffer } from '../../domain/developWorkingBuffer';
import {
  compileMaskLayerForWebGl,
  decodeMatteRleB64,
  isBackgroundRemovalActive,
  isMaskLayerEffective,
  spatialMaskGeometrySignature,
  type BackgroundRemovalSettings,
  type CompiledWebGlMaskData,
  type MaskLayer,
} from '../../domain/masking';
import { isToneCurveModified, sampleToneCurveSvgTableValues } from '../../domain/toneCurve';
import {
  loadDevelopBitmap,
  peekDevelopBitmap,
  type CachedDevelopBitmap,
} from './developBitmapCache';

interface WebGlDevelopCanvasProps {
  readonly src: string;
  readonly alt: string;
  readonly recipe: DevelopRecipe;
  /** Optional 16-bit half-float working buffer (`RGBA16F`) decoded directly from Camera RAW / 16-bit TIFF/PNG. */
  readonly workingBuffer?: DevelopWorkingBuffer | null;
  readonly previewMaxDim?: number;
  readonly fit?: 'cover' | 'contain';
  readonly viewportZoom?: number;
  readonly viewportPan?: ViewportPan;
  readonly viewportPadPx?: number;
  readonly rotationDeg?: number;
  readonly rotationScale?: number;
  readonly applyCrop?: boolean;
  readonly showShadowClip?: boolean;
  readonly showHighlightClip?: boolean;
  readonly eyedropperActive?: boolean;
  readonly onSampleNeutralRgb?: (r: number, g: number, b: number) => void;
  readonly activeMaskId?: string | null;
  readonly showMaskOverlay?: boolean;
  readonly colorPickActive?: boolean;
  readonly onSampleColorRangeRgb?: (r: number, g: number, b: number) => void;
  readonly className?: string;
  readonly onNaturalSize?: (width: number, height: number) => void;
  readonly onMaxTextureDim?: (maxDim: number) => void;
}

const MAX_FBO_DIM = 4096;

function f16BitsToF32(h: number): number {
  const sign = (h & 0x8000) !== 0 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x03ff;
  if (exp === 0) return sign * Math.pow(2, -14) * (frac / 1024);
  if (exp === 31) return frac === 0 ? sign * 65504 : 0;
  return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

const VERTEX_SHADER_SRC = `
attribute vec2 a_position;
uniform float u_flipY;
varying vec2 v_texCoord;
void main() {
  float v = u_flipY > 0.5 ? (0.5 - a_position.y * 0.5) : (a_position.y * 0.5 + 0.5);
  v_texCoord = vec2(a_position.x * 0.5 + 0.5, v);
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

// Pass 1: Line-for-line GLSL equivalent of `CompiledDevelopGrade::grade_pixel_in_space`
// in `internal/infrastructure/src/develop/color_math.rs`.
const COLOR_FRAGMENT_SHADER_SRC = `
precision highp float;

uniform sampler2D u_image;
uniform sampler2D u_toneLut;
uniform vec4 u_uvTransform; // (scaleX, scaleY, offsetX, offsetY)
uniform vec4 u_cropBounds;  // (minU, minV, maxU, maxV)
uniform vec4 u_orient;      // (flipH, flipV, quarterTurns, rotationRad)
uniform vec2 u_orientedDims;

uniform float u_inputLinear;
uniform float u_inputWideGamut;
uniform float u_linearActive;
uniform float u_exposureGain;
uniform vec3 u_wbGains;

uniform float u_calActive;
uniform float u_calShadowsTint;
uniform mat3 u_calMatrix;

uniform float u_blackAndWhite;
uniform float u_shadows;
uniform float u_highlights;
uniform float u_whites;
uniform float u_blacks;
uniform float u_dehaze;
uniform float u_contrast;
uniform float u_vibrance;
uniform float u_saturation;

uniform float u_toneCurveActive;
uniform float u_hslActive;
uniform vec3 u_hsl[8];

uniform float u_cgActive;
uniform vec3 u_cgShdTint;
uniform float u_cgShdLum;
uniform vec3 u_cgMidTint;
uniform float u_cgMidLum;
uniform vec3 u_cgHiTint;
uniform float u_cgHiLum;
uniform float u_cgPivot;
uniform float u_cgSpread;

uniform float u_vignetteAmount;
uniform float u_vignetteMidpoint;
uniform float u_vignetteFeather;
uniform vec2 u_clipFlags; // (showShadowClip, showHighlightClip)

uniform sampler2D u_bgMatte;
uniform float u_bgMode;      // 0=off/transparent, 2=solid color, 3=gradient
uniform vec3 u_bgColor;
uniform vec3 u_bgColor2;
uniform float u_bgGradStyle; // 0=radial spot, 1=linear angle
uniform float u_bgGradAngle; // CSS degrees (0..360)
uniform float u_bgFeather;

varying vec2 v_texCoord;

float srgbToLinear(float c) {
  float x = clamp(c, 0.0, 1.0);
  return x <= 0.04045 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4);
}

float linearToSrgb(float c) {
  float x = clamp(c, 0.0, 1.0);
  return x <= 0.0031308 ? x * 12.92 : 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

float softClip(float x) {
  if (x <= 0.92) {
    return max(x, 0.0);
  }
  float excess = x - 0.92;
  return 0.92 + 0.08 * (1.0 - exp(-excess / 0.08));
}

vec3 rec2020ToRec709(vec3 c) {
  return vec3(
     1.6604910 * c.r - 0.5876411 * c.g - 0.0728499 * c.b,
    -0.1245505 * c.r + 1.1328999 * c.g - 0.0083494 * c.b,
    -0.0181508 * c.r - 0.1005789 * c.g + 1.1187297 * c.b
  );
}

vec3 compressRec2020ToLinearSrgb(vec3 lin2020) {
  vec3 srgb = rec2020ToRec709(lin2020);
  float y = max(0.2126 * srgb.r + 0.7152 * srgb.g + 0.0722 * srgb.b, 0.0);
  if (y <= 1e-6) {
    return vec3(0.0);
  }
  float minC = min(srgb.r, min(srgb.g, srgb.b));
  float d = (y - minC) / y;
  float threshold = 0.85;
  if (d <= threshold) {
    return max(srgb, vec3(0.0));
  }
  float excess = d - threshold;
  float limit = 1.0 - threshold;
  float dCompressed = threshold + limit * (excess / (excess + limit));
  float scale = dCompressed / d;
  return max(vec3(y) + (srgb - vec3(y)) * scale, vec3(0.0));
}

float kSmoothstep(float edge0, float edge1, float x) {
  float t = clamp((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

float grainHash12(vec2 pIn) {
  vec3 p3 = fract(vec3(pIn.x, pIn.y, pIn.x) * 0.1031);
  p3 += dot(p3, vec3(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
  return fract((p3.x + p3.y) * p3.z);
}

vec3 recoverHdrHighlights(vec3 lin, float highlights, float whites) {
  float peak = max(lin.r, max(lin.g, lin.b));
  if (peak <= 0.65) {
    return lin;
  }
  float hlPull = highlights < 0.0 ? -highlights / 100.0 : 0.0;
  float whPull = whites < 0.0 ? -whites / 100.0 : 0.0;
  float pull = min(1.0, hlPull * 0.65 + whPull * 0.35);
  if (pull <= 1e-5) {
    return lin;
  }
  float knee = 0.65;
  float excess = peak - knee;
  float weight = kSmoothstep(0.65, 1.35, peak);
  float targetPeak = knee + excess / (1.0 + pull * 2.4 * weight);
  float scale = targetPeak / max(peak, 1e-5);
  return lin * scale;
}

float kGaussian(float x, float mean, float sigma) {
  float d = (x - mean) / max(sigma, 1e-5);
  return exp(-0.5 * d * d);
}

float hueDistanceDeg(float a, float b) {
  float diff = mod(abs(a - b), 360.0);
  return diff > 180.0 ? 360.0 - diff : diff;
}

vec3 rgbToHsl(vec3 rgb) {
  float r = clamp(rgb.r, 0.0, 1.0);
  float g = clamp(rgb.g, 0.0, 1.0);
  float b = clamp(rgb.b, 0.0, 1.0);
  float maxC = max(r, max(g, b));
  float minC = min(r, min(g, b));
  float l = 0.5 * (maxC + minC);
  float d = maxC - minC;
  if (d <= 1e-6) {
    return vec3(0.0, 0.0, l);
  }
  float s = l > 0.5 ? d / max(2.0 - maxC - minC, 1e-6) : d / max(maxC + minC, 1e-6);
  float h = 0.0;
  if (abs(maxC - r) <= 1e-6) {
    h = (g - b) / d + (g < b ? 6.0 : 0.0);
  } else if (abs(maxC - g) <= 1e-6) {
    h = (b - r) / d + 2.0;
  } else {
    h = (r - g) / d + 4.0;
  }
  h = mod(h * 60.0 + 360.0, 360.0);
  return vec3(h, clamp(s, 0.0, 1.0), clamp(l, 0.0, 1.0));
}

float hueToRgbChannel(float p, float q, float tIn) {
  float t = tIn;
  if (t < 0.0) t += 1.0;
  if (t > 1.0) t -= 1.0;
  if (t < 1.0 / 6.0) return p + (q - p) * 6.0 * t;
  if (t < 0.5) return q;
  if (t < 2.0 / 3.0) return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
  return p;
}

vec3 hslToRgb(vec3 hsl) {
  float h = hsl.x;
  float s = hsl.y;
  float l = hsl.z;
  if (s <= 1e-6) {
    return vec3(l);
  }
  float q = l < 0.5 ? l * (1.0 + s) : l + s - l * s;
  float p = 2.0 * l - q;
  float hk = mod(h + 360.0, 360.0) / 360.0;
  return vec3(
    hueToRgbChannel(p, q, hk + 1.0 / 3.0),
    hueToRgbChannel(p, q, hk),
    hueToRgbChannel(p, q, hk - 1.0 / 3.0)
  );
}

vec3 applyBasicTone(vec3 rgb) {
  float luma = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
  float delta = 0.0;

  if (u_shadows != 0.0) {
    float sw = 1.0 - kSmoothstep(0.05, 0.55, luma);
    delta += (u_shadows / 100.0) * 0.28 * sw * sw;
  }
  if (u_highlights != 0.0) {
    float hw = kSmoothstep(0.45, 0.95, luma);
    delta += (u_highlights / 100.0) * 0.28 * hw * hw;
  }
  if (u_whites != 0.0) {
    delta += (u_whites / 100.0) * 0.22 * kSmoothstep(0.65, 1.0, luma);
  }
  if (u_blacks != 0.0) {
    delta += (u_blacks / 100.0) * 0.22 * (1.0 - kSmoothstep(0.0, 0.35, luma));
  }

  if (abs(delta) > 1e-6) {
    float targetLuma = clamp(luma + delta, 0.0, 1.0);
    if (luma > 1e-4) {
      rgb = clamp(rgb * (targetLuma / luma), 0.0, 1.0);
    } else {
      rgb = vec3(targetLuma);
    }
  }

  if (u_dehaze != 0.0) {
    float dh = u_dehaze / 100.0;
    float lift = -dh * 0.06;
    float sc = 1.0 + dh * 0.18;
    rgb = clamp((rgb + vec3(lift)) * sc, 0.0, 1.0);
  }

  if (u_contrast != 0.0) {
    float c = u_contrast / 100.0;
    float factor = c >= 0.0 ? 1.0 + c * 0.85 : 1.0 + c * 0.65;
    rgb = clamp((rgb - vec3(0.5)) * factor + vec3(0.5), 0.0, 1.0);
  }

  return rgb;
}

vec3 applyHslBand(vec3 acc, float h, float center, float halfWidth, vec3 shift) {
  if (shift.x == 0.0 && shift.y == 0.0 && shift.z == 0.0) {
    return acc;
  }
  float dist = hueDistanceDeg(h, center);
  if (dist >= halfWidth) {
    return acc;
  }
  float weight = 0.5 * (1.0 + cos(3.14159265 * dist / halfWidth));
  return acc + vec3(
    (shift.x / 100.0) * 28.0 * weight,
    (shift.y / 100.0) * weight,
    (shift.z / 100.0) * 0.25 * weight
  );
}

float applyBwBand(float acc, float h, float center, float halfWidth, float lumShift) {
  if (lumShift == 0.0) return acc;
  float dist = hueDistanceDeg(h, center);
  if (dist >= halfWidth) return acc;
  float weight = 0.5 * (1.0 + cos(3.14159265 * dist / halfWidth));
  return acc + (lumShift / 100.0) * 0.45 * weight;
}

vec2 orientSourceUv(vec2 uv) {
  vec2 p = uv;
  if (u_orient.x > 0.5) p.x = 1.0 - p.x;
  if (u_orient.y > 0.5) p.y = 1.0 - p.y;
  float qt = u_orient.z;
  if (qt > 0.5 && qt < 1.5) {
    return vec2(p.y, 1.0 - p.x);
  } else if (qt >= 1.5 && qt < 2.5) {
    return vec2(1.0 - p.x, 1.0 - p.y);
  } else if (qt >= 2.5) {
    return vec2(1.0 - p.y, p.x);
  }
  return p;
}

void main() {
  vec2 uv = v_texCoord * u_uvTransform.xy + u_uvTransform.zw;
  if (
    uv.x < u_cropBounds.x || uv.x > u_cropBounds.z ||
    uv.y < u_cropBounds.y || uv.y > u_cropBounds.w
  ) {
    gl_FragColor = vec4(0.0, 0.0, 0.0, 0.0);
    return;
  }

  vec2 sampleUv = uv;
  bool outOfBounds = false;
  float rotRad = u_orient.w;
  if (abs(rotRad) > 1e-5) {
    float oriW = max(u_orientedDims.x, 1.0);
    float oriH = max(u_orientedDims.y, 1.0);
    float px = uv.x * oriW - 0.5;
    float py = uv.y * oriH - 0.5;
    float cx = (oriW - 1.0) * 0.5;
    float cy = (oriH - 1.0) * 0.5;
    float dx = px - cx;
    float dy = py - cy;
    float cosA = cos(-rotRad);
    float sinA = sin(-rotRad);
    float sx = dx * cosA - dy * sinA + cx;
    float sy = dx * sinA + dy * cosA + cy;
    if (sx < 0.0 || sy < 0.0 || sx >= oriW - 1.0 || sy >= oriH - 1.0) {
      outOfBounds = true;
    } else {
      sampleUv = vec2((sx + 0.5) / oriW, (sy + 0.5) / oriH);
    }
  }

  vec4 texel = outOfBounds ? vec4(0.0, 0.0, 0.0, 1.0) : texture2D(u_image, orientSourceUv(sampleUv));
  vec3 rgb = texel.rgb;

  // 1-2. Linear White Balance + Exposure EV + HDR Highlight Recovery + Smooth Wide-Gamut Compression
  if (u_inputLinear > 0.5) {
    vec3 lin = u_linearActive > 0.5
      ? max(rgb * u_wbGains * u_exposureGain, vec3(0.0))
      : max(rgb, vec3(0.0));
    if (u_highlights < 0.0 || u_whites < 0.0) {
      lin = recoverHdrHighlights(lin, u_highlights, u_whites);
    }
    if (u_inputWideGamut > 0.5) {
      lin = compressRec2020ToLinearSrgb(lin);
    }
    rgb = vec3(
      linearToSrgb(softClip(lin.r)),
      linearToSrgb(softClip(lin.g)),
      linearToSrgb(softClip(lin.b))
    );
  } else if (u_linearActive > 0.5) {
    vec3 lin = vec3(srgbToLinear(rgb.r), srgbToLinear(rgb.g), srgbToLinear(rgb.b));
    lin = max(lin * u_wbGains * u_exposureGain, vec3(0.0));
    rgb = vec3(
      linearToSrgb(softClip(lin.r)),
      linearToSrgb(softClip(lin.g)),
      linearToSrgb(softClip(lin.b))
    );
  }

  // 2b. Camera Calibration (Shadows Tint + neutral-preserving R/G/B primary matrix)
  if (u_calActive > 0.5) {
    rgb = clamp(u_calMatrix * rgb, 0.0, 1.0);
    if (abs(u_calShadowsTint) > 1e-5) {
      float lumaCal = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
      float sw = 1.0 - kSmoothstep(0.0, 0.55, lumaCal);
      float delta = u_calShadowsTint * 0.08 * sw * sw;
      rgb = clamp(vec3(rgb.r + delta * 0.5, rgb.g - delta, rgb.b + delta * 0.5), 0.0, 1.0);
    }
  }

  // 3. Basic Tone (Shadows, Highlights, Whites, Blacks, Dehaze, Contrast)
  rgb = applyBasicTone(rgb);

  // 4. Combined Master + Per-Channel RGB Tone Curves (256x1 RGBA LUT)
  if (u_toneCurveActive > 0.5) {
    float rLut = texture2D(u_toneLut, vec2((clamp(rgb.r, 0.0, 1.0) * 255.0 + 0.5) / 256.0, 0.5)).r;
    float gLut = texture2D(u_toneLut, vec2((clamp(rgb.g, 0.0, 1.0) * 255.0 + 0.5) / 256.0, 0.5)).g;
    float bLut = texture2D(u_toneLut, vec2((clamp(rgb.b, 0.0, 1.0) * 255.0 + 0.5) / 256.0, 0.5)).b;
    rgb = vec3(rLut, gLut, bLut);
  }

  // 5-6. Presence + 8-Channel HSL Color Mixer OR Perceptual B&W Color Mix
  if (u_blackAndWhite > 0.5) {
    float baseLuma = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
    float bw = baseLuma;
    if (u_hslActive > 0.5) {
      vec3 hsl = rgbToHsl(rgb);
      if (hsl.y >= 0.015) {
        float lumDelta = 0.0;
        lumDelta = applyBwBand(lumDelta, hsl.x, 0.0,   30.0, u_hsl[0].z);
        lumDelta = applyBwBand(lumDelta, hsl.x, 30.0,  30.0, u_hsl[1].z);
        lumDelta = applyBwBand(lumDelta, hsl.x, 60.0,  35.0, u_hsl[2].z);
        lumDelta = applyBwBand(lumDelta, hsl.x, 120.0, 50.0, u_hsl[3].z);
        lumDelta = applyBwBand(lumDelta, hsl.x, 180.0, 50.0, u_hsl[4].z);
        lumDelta = applyBwBand(lumDelta, hsl.x, 240.0, 45.0, u_hsl[5].z);
        lumDelta = applyBwBand(lumDelta, hsl.x, 285.0, 35.0, u_hsl[6].z);
        lumDelta = applyBwBand(lumDelta, hsl.x, 325.0, 35.0, u_hsl[7].z);
        bw = clamp(baseLuma + lumDelta * hsl.y, 0.0, 1.0);
      }
    }
    rgb = vec3(bw);
  } else {
    if (u_vibrance != 0.0 || u_saturation != 0.0) {
      vec3 hsl = rgbToHsl(rgb);
      if (u_vibrance != 0.0) {
        float vib = u_vibrance / 100.0;
        float skinProx = kGaussian(hueDistanceDeg(hsl.x, 28.0), 0.0, 18.0);
        float skinGuard = vib > 0.0 ? 1.0 - 0.6 * skinProx : 1.0;
        hsl.y = clamp(hsl.y + vib * 0.55 * (1.0 - hsl.y) * skinGuard, 0.0, 1.0);
      }
      if (u_saturation != 0.0) {
        hsl.y = clamp(hsl.y * (1.0 + u_saturation / 100.0), 0.0, 1.0);
      }
      rgb = hslToRgb(hsl);
    }

    if (u_hslActive > 0.5) {
      vec3 hsl = rgbToHsl(rgb);
      if (hsl.y >= 0.015) {
        vec3 deltas = vec3(0.0);
        deltas = applyHslBand(deltas, hsl.x, 0.0,   30.0, u_hsl[0]);
        deltas = applyHslBand(deltas, hsl.x, 30.0,  30.0, u_hsl[1]);
        deltas = applyHslBand(deltas, hsl.x, 60.0,  35.0, u_hsl[2]);
        deltas = applyHslBand(deltas, hsl.x, 120.0, 50.0, u_hsl[3]);
        deltas = applyHslBand(deltas, hsl.x, 180.0, 50.0, u_hsl[4]);
        deltas = applyHslBand(deltas, hsl.x, 240.0, 45.0, u_hsl[5]);
        deltas = applyHslBand(deltas, hsl.x, 285.0, 35.0, u_hsl[6]);
        deltas = applyHslBand(deltas, hsl.x, 325.0, 35.0, u_hsl[7]);
        float nextH = mod(hsl.x + deltas.x + 360.0, 360.0);
        float nextS = clamp(hsl.y * (1.0 + deltas.y), 0.0, 1.0);
        float nextL = clamp(hsl.z + deltas.z * hsl.y, 0.0, 1.0);
        rgb = hslToRgb(vec3(nextH, nextS, nextL));
      }
    }
  }

  // 6b. 3-Way Color Grading (Shadows / Midtones / Highlights + Blending & Balance)
  if (u_cgActive > 0.5) {
    float lumaCg = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
    float shdW = pow(clamp(1.0 - kSmoothstep(u_cgPivot - u_cgSpread, u_cgPivot + u_cgSpread * 0.65, lumaCg), 1e-6, 1.0), 1.5);
    float hiW = pow(clamp(kSmoothstep(u_cgPivot - u_cgSpread * 0.65, u_cgPivot + u_cgSpread, lumaCg), 1e-6, 1.0), 1.5);
    float midW = max(0.0, 1.0 - shdW - hiW);
    float wSum = max(1.0, shdW + midW + hiW);
    shdW /= wSum;
    midW /= wSum;
    hiW /= wSum;

    float shdEnv = 1.0 - kSmoothstep(0.75, 1.0, lumaCg);
    float midEnv = kSmoothstep(0.0, 0.12, lumaCg) * (1.0 - kSmoothstep(0.88, 1.0, lumaCg));
    float hiEnv = kSmoothstep(0.05, 0.30, lumaCg);
    float lumShift = shdW * u_cgShdLum + midW * u_cgMidLum + hiW * u_cgHiLum;
    rgb = clamp(
      rgb + u_cgShdTint * (shdW * shdEnv) + u_cgMidTint * (midW * midEnv) + u_cgHiTint * (hiW * hiEnv) + vec3(lumShift),
      0.0,
      1.0
    );
  }

  // 7. Post-crop radial vignette (Amount + Midpoint + Feather)
  if (u_vignetteAmount != 0.0) {
    vec2 centered = v_texCoord * 2.0 - vec2(1.0);
    float radius = length(centered);
    float mid = clamp(u_vignetteMidpoint / 100.0, 0.05, 0.98) * 1.15;
    float f = clamp(u_vignetteFeather / 100.0, 0.02, 1.0);
    float inner = mid * (1.0 - 0.65 * f);
    float outer = min(1.65, mid + 0.12 + 0.72 * f);
    float mask = kSmoothstep(inner, outer, radius);
    float factor = clamp(1.0 + (u_vignetteAmount / 100.0) * 0.75 * mask, 0.15, 1.85);
    rgb = clamp(rgb * factor, 0.0, 1.0);
  }

  rgb = clamp(rgb, 0.0, 1.0);

  // 8. Optional Opaque Studio Backdrop (Solid / Gradient) composited BEFORE local masks so
  // AI Background / Radial / Linear / Brush masks naturally adjust the replaced backdrop.
  if (u_bgMode > 1.5) {
    float rawM = texture2D(u_bgMatte, uv).r;
    float f = clamp(u_bgFeather, 0.0, 100.0);
    float matteA = rawM;
    if (f < 24.5) {
      float k = 1.0 + ((25.0 - f) / 25.0) * 3.0;
      matteA = clamp((rawM - 0.5) * k + 0.5, 0.0, 1.0);
    } else if (f > 25.5) {
      float t = (f - 25.0) / 75.0;
      float soft = kSmoothstep(0.0, 1.0, pow(clamp(rawM, 0.0, 1.0), 1.0 - 0.35 * t));
      matteA = mix(rawM, soft, t);
    }

    vec3 backdrop = u_bgColor;
    if (u_bgMode > 2.5) {
      float gt = 0.0;
      if (u_bgGradStyle > 0.5) {
        float rad = u_bgGradAngle * 0.0174532925;
        vec2 dir = vec2(sin(rad), -cos(rad));
        float span = max(abs(dir.x) + abs(dir.y), 1e-5);
        float proj = dot(uv - vec2(0.5, 0.5), dir);
        gt = clamp(proj / span + 0.5, 0.0, 1.0);
      } else {
        vec2 dp = (uv - vec2(0.5, 0.42)) / vec2(0.68, 0.72);
        float d = clamp(length(dp), 0.0, 1.0);
        gt = kSmoothstep(0.0, 1.0, d);
      }
      backdrop = mix(u_bgColor, u_bgColor2, gt);
    }
    rgb = mix(backdrop, rgb, matteA);
  }

  // Optional Lightroom Highlight/Shadow Clipping Overlay (only active when single-pass to screen)
  if (u_clipFlags.y > 0.5 && max(rgb.r, max(rgb.g, rgb.b)) >= 0.992) {
    rgb = vec3(0.94, 0.27, 0.27);
  } else if (u_clipFlags.x > 0.5 && min(rgb.r, min(rgb.g, rgb.b)) <= 0.008) {
    rgb = vec3(0.23, 0.51, 0.96);
  }

  gl_FragColor = vec4(rgb, texel.a);
}
`;

// Pass 2: Resolution-proportional spatial detail + Color NR + Film Grain shader matching
// `apply_spatial_detail_f32` in `internal/infrastructure/src/develop/pixel_engine.rs`.
const SPATIAL_FRAGMENT_SHADER_SRC = `
precision highp float;

uniform sampler2D u_graded;
uniform sampler2D u_bgMatte;
uniform vec4 u_uvTransform;
uniform vec4 u_cropBounds;
uniform vec2 u_texelSize;
uniform float u_spatialScale;

uniform float u_nr;
uniform float u_cnr;
uniform float u_texture;
uniform float u_clarity;
uniform float u_sharpening;
uniform float u_grainAmount;
uniform float u_grainSize;
uniform vec2 u_clipFlags;
uniform float u_bgMode;     // 0=off, 1=transparent checkerboard cutout
uniform float u_bgFeather;

varying vec2 v_texCoord;

float sampleLuma(vec2 uv) {
  vec3 c = texture2D(u_graded, clamp(uv, vec2(0.0), vec2(1.0))).rgb;
  return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
}

float kSmoothstep(float edge0, float edge1, float x) {
  float t = clamp((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

float grainHash12(vec2 pIn) {
  vec3 p3 = fract(vec3(pIn.x, pIn.y, pIn.x) * 0.1031);
  p3 += dot(p3, vec3(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
  return fract((p3.x + p3.y) * p3.z);
}

void main() {
  vec2 uv = v_texCoord * u_uvTransform.xy + u_uvTransform.zw;
  if (
    uv.x < u_cropBounds.x || uv.x > u_cropBounds.z ||
    uv.y < u_cropBounds.y || uv.y > u_cropBounds.w
  ) {
    gl_FragColor = vec4(0.0, 0.0, 0.0, 0.0);
    return;
  }

  vec4 center = texture2D(u_graded, uv);
  vec3 rgb = center.rgb;
  float lc = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;

  // 0. Edge-preserving YCbCr Color Noise Reduction
  if (u_cnr > 1e-4) {
    float cnr = u_cnr / 100.0;
    vec2 stepC = u_texelSize * (2.2 * u_spatialScale);
    float sigmaC = 0.04 + cnr * 0.14;
    float inv2Sig2 = -0.5 / (sigmaC * sigmaC);
    float cr0 = rgb.r - lc;
    float cb0 = rgb.b - lc;
    float sumCr = cr0;
    float sumCb = cb0;
    float sumW = 1.0;

    vec3 s1 = texture2D(u_graded, clamp(uv + vec2(-stepC.x, 0.0), vec2(0.0), vec2(1.0))).rgb;
    float y1 = 0.2126 * s1.r + 0.7152 * s1.g + 0.0722 * s1.b;
    float w1 = 0.7 * exp((y1 - lc) * (y1 - lc) * inv2Sig2);
    sumCr += (s1.r - y1) * w1; sumCb += (s1.b - y1) * w1; sumW += w1;

    vec3 s2 = texture2D(u_graded, clamp(uv + vec2(stepC.x, 0.0), vec2(0.0), vec2(1.0))).rgb;
    float y2 = 0.2126 * s2.r + 0.7152 * s2.g + 0.0722 * s2.b;
    float w2 = 0.7 * exp((y2 - lc) * (y2 - lc) * inv2Sig2);
    sumCr += (s2.r - y2) * w2; sumCb += (s2.b - y2) * w2; sumW += w2;

    vec3 s3 = texture2D(u_graded, clamp(uv + vec2(0.0, -stepC.y), vec2(0.0), vec2(1.0))).rgb;
    float y3 = 0.2126 * s3.r + 0.7152 * s3.g + 0.0722 * s3.b;
    float w3 = 0.7 * exp((y3 - lc) * (y3 - lc) * inv2Sig2);
    sumCr += (s3.r - y3) * w3; sumCb += (s3.b - y3) * w3; sumW += w3;

    vec3 s4 = texture2D(u_graded, clamp(uv + vec2(0.0, stepC.y), vec2(0.0), vec2(1.0))).rgb;
    float y4 = 0.2126 * s4.r + 0.7152 * s4.g + 0.0722 * s4.b;
    float w4 = 0.7 * exp((y4 - lc) * (y4 - lc) * inv2Sig2);
    sumCr += (s4.r - y4) * w4; sumCb += (s4.b - y4) * w4; sumW += w4;

    vec3 s5 = texture2D(u_graded, clamp(uv + vec2(-stepC.x, -stepC.y), vec2(0.0), vec2(1.0))).rgb;
    float y5 = 0.2126 * s5.r + 0.7152 * s5.g + 0.0722 * s5.b;
    float w5 = 0.5 * exp((y5 - lc) * (y5 - lc) * inv2Sig2);
    sumCr += (s5.r - y5) * w5; sumCb += (s5.b - y5) * w5; sumW += w5;

    vec3 s6 = texture2D(u_graded, clamp(uv + vec2(stepC.x, -stepC.y), vec2(0.0), vec2(1.0))).rgb;
    float y6 = 0.2126 * s6.r + 0.7152 * s6.g + 0.0722 * s6.b;
    float w6 = 0.5 * exp((y6 - lc) * (y6 - lc) * inv2Sig2);
    sumCr += (s6.r - y6) * w6; sumCb += (s6.b - y6) * w6; sumW += w6;

    vec3 s7 = texture2D(u_graded, clamp(uv + vec2(-stepC.x, stepC.y), vec2(0.0), vec2(1.0))).rgb;
    float y7 = 0.2126 * s7.r + 0.7152 * s7.g + 0.0722 * s7.b;
    float w7 = 0.5 * exp((y7 - lc) * (y7 - lc) * inv2Sig2);
    sumCr += (s7.r - y7) * w7; sumCb += (s7.b - y7) * w7; sumW += w7;

    vec3 s8 = texture2D(u_graded, clamp(uv + vec2(stepC.x, stepC.y), vec2(0.0), vec2(1.0))).rgb;
    float y8 = 0.2126 * s8.r + 0.7152 * s8.g + 0.0722 * s8.b;
    float w8 = 0.5 * exp((y8 - lc) * (y8 - lc) * inv2Sig2);
    sumCr += (s8.r - y8) * w8; sumCb += (s8.b - y8) * w8; sumW += w8;

    float invW = 1.0 / max(sumW, 1e-5);
    float blendC = cnr * 0.85;
    float nextCr = mix(cr0, sumCr * invW, blendC);
    float nextCb = mix(cb0, sumCb * invW, blendC);
    float nrR = clamp(lc + nextCr, 0.0, 1.0);
    float nrB = clamp(lc + nextCb, 0.0, 1.0);
    float nrG = clamp((lc - 0.2126 * nrR - 0.0722 * nrB) / 0.7152, 0.0, 1.0);
    rgb = vec3(nrR, nrG, nrB);
  }

  float totalDelta = 0.0;

  // 1. Edge-preserving 3x3 bilateral Luminance NR
  if (u_nr > 1e-4) {
    float nr = u_nr / 100.0;
    vec2 stepNr = u_texelSize * (1.25 * u_spatialScale);
    float sigmaR = 0.03 + nr * 0.11;
    float inv2Sig2 = -0.5 / (sigmaR * sigmaR);
    float sumVal = lc;
    float sumW = 1.0;

    float l1 = sampleLuma(uv + vec2(-stepNr.x, 0.0));
    float w1 = 0.65 * exp((l1 - lc) * (l1 - lc) * inv2Sig2);
    sumVal += l1 * w1; sumW += w1;

    float l2 = sampleLuma(uv + vec2(stepNr.x, 0.0));
    float w2 = 0.65 * exp((l2 - lc) * (l2 - lc) * inv2Sig2);
    sumVal += l2 * w2; sumW += w2;

    float l3 = sampleLuma(uv + vec2(0.0, -stepNr.y));
    float w3 = 0.65 * exp((l3 - lc) * (l3 - lc) * inv2Sig2);
    sumVal += l3 * w3; sumW += w3;

    float l4 = sampleLuma(uv + vec2(0.0, stepNr.y));
    float w4 = 0.65 * exp((l4 - lc) * (l4 - lc) * inv2Sig2);
    sumVal += l4 * w4; sumW += w4;

    float l5 = sampleLuma(uv + vec2(-stepNr.x, -stepNr.y));
    float w5 = 0.45 * exp((l5 - lc) * (l5 - lc) * inv2Sig2);
    sumVal += l5 * w5; sumW += w5;

    float l6 = sampleLuma(uv + vec2(stepNr.x, -stepNr.y));
    float w6 = 0.45 * exp((l6 - lc) * (l6 - lc) * inv2Sig2);
    sumVal += l6 * w6; sumW += w6;

    float l7 = sampleLuma(uv + vec2(-stepNr.x, stepNr.y));
    float w7 = 0.45 * exp((l7 - lc) * (l7 - lc) * inv2Sig2);
    sumVal += l7 * w7; sumW += w7;

    float l8 = sampleLuma(uv + vec2(stepNr.x, stepNr.y));
    float w8 = 0.45 * exp((l8 - lc) * (l8 - lc) * inv2Sig2);
    sumVal += l8 * w8; sumW += w8;

    float lNr = sumVal / max(sumW, 1e-5);
    totalDelta += (lNr - lc) * (nr * 0.90);
  }

  // 2. Medium-fine edge-clamped micro-contrast (Texture)
  if (abs(u_texture) > 1e-4) {
    float tex = u_texture / 100.0;
    vec2 rTex = u_texelSize * (2.5 * u_spatialScale);
    vec2 dTex = rTex * 0.70710677;
    float blurTex = 0.20 * lc
      + 0.12 * (
          sampleLuma(uv + vec2(-rTex.x, 0.0)) +
          sampleLuma(uv + vec2( rTex.x, 0.0)) +
          sampleLuma(uv + vec2(0.0, -rTex.y)) +
          sampleLuma(uv + vec2(0.0,  rTex.y))
        )
      + 0.08 * (
          sampleLuma(uv + vec2(-dTex.x, -dTex.y)) +
          sampleLuma(uv + vec2( dTex.x, -dTex.y)) +
          sampleLuma(uv + vec2(-dTex.x,  dTex.y)) +
          sampleLuma(uv + vec2( dTex.x,  dTex.y))
        );
    float detailTex = clamp(lc - blurTex, -0.18, 0.18);
    totalDelta += detailTex * tex * 0.85;
  }

  // 3. Broad midtone-masked local contrast (Clarity)
  if (abs(u_clarity) > 1e-4) {
    float clar = u_clarity / 100.0;
    vec2 rClar = u_texelSize * (11.0 * u_spatialScale);
    vec2 inClar = rClar * 0.45;
    vec2 dClar = rClar * 0.70710677;
    float blurClar = 0.16 * lc
      + 0.10 * (
          sampleLuma(uv + vec2(-inClar.x, 0.0)) +
          sampleLuma(uv + vec2( inClar.x, 0.0)) +
          sampleLuma(uv + vec2(0.0, -inClar.y)) +
          sampleLuma(uv + vec2(0.0,  inClar.y))
        )
      + 0.06 * (
          sampleLuma(uv + vec2(-rClar.x, 0.0)) +
          sampleLuma(uv + vec2( rClar.x, 0.0)) +
          sampleLuma(uv + vec2(0.0, -rClar.y)) +
          sampleLuma(uv + vec2(0.0,  rClar.y))
        )
      + 0.05 * (
          sampleLuma(uv + vec2(-dClar.x, -dClar.y)) +
          sampleLuma(uv + vec2( dClar.x, -dClar.y)) +
          sampleLuma(uv + vec2(-dClar.x,  dClar.y)) +
          sampleLuma(uv + vec2( dClar.x,  dClar.y))
        );
    float midtoneMask = pow(clamp(4.0 * lc * (1.0 - lc), 1e-6, 1.0), 0.70);
    float detailClar = clamp(lc - blurClar, -0.30, 0.30);
    totalDelta += detailClar * clar * 1.15 * midtoneMask;
  }

  // 4. Thresholded luminance unsharp mask (Sharpening)
  if (u_sharpening > 1e-4) {
    float sharp = u_sharpening / 100.0;
    vec2 rSharp = u_texelSize * (1.0 * u_spatialScale);
    float avg4 = 0.25 * (
      sampleLuma(uv + vec2(-rSharp.x, 0.0)) +
      sampleLuma(uv + vec2( rSharp.x, 0.0)) +
      sampleLuma(uv + vec2(0.0, -rSharp.y)) +
      sampleLuma(uv + vec2(0.0,  rSharp.y))
    );
    float hp = lc - avg4;
    float mag = max(abs(hp) - 0.008, 0.0);
    float threshHp = (hp >= 0.0 ? 1.0 : -1.0) * mag;
    totalDelta += clamp(threshHp * sharp * 1.10, -0.25, 0.25);
  }

  if (abs(totalDelta) > 1e-6) {
    float nextLuma = clamp(lc + totalDelta, 0.0, 1.0);
    float actualDelta = nextLuma - lc;
    float ratio = lc > 1e-4 ? clamp(nextLuma / lc, 0.0, 3.0) : 1.0;
    float blend = kSmoothstep(0.02, 0.16, lc);
    vec3 additive = clamp(rgb + vec3(actualDelta), 0.0, 1.0);
    vec3 mult = lc > 1e-4 ? clamp(rgb * ratio, 0.0, 1.0) : vec3(nextLuma);
    rgb = mix(additive, mult, blend);
  }

  // 5. Resolution-proportional organic Film Grain
  if (u_grainAmount > 1e-4) {
    float ga = u_grainAmount / 100.0;
    float gs = u_grainSize / 100.0;
    float grid = 1024.0 / (0.8 + gs * 2.4);
    vec2 cell = floor(uv * grid + vec2(1e-4));
    float n1 = grainHash12(cell);
    float n2 = grainHash12(cell + vec2(17.3, 41.7));
    float signedNoise = (n1 + n2) - 1.0;
    float lCur = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
    float midEnv = 0.35 + 0.65 * clamp(4.0 * lCur * (1.0 - lCur), 0.0, 1.0);
    rgb = clamp(rgb + vec3(signedNoise * ga * 0.11 * midEnv), 0.0, 1.0);
  }

  rgb = clamp(rgb, 0.0, 1.0);

  // 6. Optional Transparent Cutout Checkerboard Overlay (opaque Solid/Gradient ran before masks in Pass 1)
  if (u_bgMode > 0.5 && u_bgMode < 1.5) {
    float rawM = texture2D(u_bgMatte, uv).r;
    float f = clamp(u_bgFeather, 0.0, 100.0);
    float matteA = rawM;
    if (f < 24.5) {
      float k = 1.0 + ((25.0 - f) / 25.0) * 3.0;
      matteA = clamp((rawM - 0.5) * k + 0.5, 0.0, 1.0);
    } else if (f > 25.5) {
      float t = (f - 25.0) / 75.0;
      float soft = kSmoothstep(0.0, 1.0, pow(clamp(rawM, 0.0, 1.0), 1.0 - 0.35 * t));
      matteA = mix(rawM, soft, t);
    }

    vec2 chk = floor(gl_FragCoord.xy / 12.0);
    float pat = mod(chk.x + chk.y, 2.0);
    vec3 backdrop = mix(vec3(0.18, 0.19, 0.21), vec3(0.28, 0.29, 0.32), pat);
    rgb = mix(backdrop, rgb, matteA);
  }

  if (u_clipFlags.y > 0.5 && max(rgb.r, max(rgb.g, rgb.b)) >= 0.992) {
    rgb = vec3(0.94, 0.27, 0.27);
  } else if (u_clipFlags.x > 0.5 && min(rgb.r, min(rgb.g, rgb.b)) <= 0.008) {
    rgb = vec3(0.23, 0.51, 0.96);
  }

  gl_FragColor = vec4(rgb, center.a);
}
`;

// Local & AI Mask Pass: Exact GLSL mirror of `internal/infrastructure/src/develop/mask_engine.rs`.
// Combines pre-rasterized spatial alpha (`Linear`, `Radial`, `Brush`, `AiMatte`) with up to 4
// full-resolution per-pixel Range components (`LuminanceRange`, `ColorRange`), applies local
// tone/color/spatial adjustments, or renders the translucent Ruby Overlay (`u_overlayOnly = 1.0`).
const MASK_FRAGMENT_SHADER_SRC = `
precision highp float;

uniform sampler2D u_prevImage;
uniform sampler2D u_spatialMask;
uniform vec4 u_uvTransform;
uniform vec4 u_cropBounds;
uniform vec2 u_texelSize;
uniform float u_spatialScale;

uniform float u_hasSpatial;
uniform float u_layerInvert;
uniform float u_layerOpacity;
uniform float u_overlayOnly;

uniform vec4 u_rangeMeta[4];    // (mode: 0=none/1=luma/2=color, op: 0=add/1=sub/2=int, inv, 0)
uniform vec4 u_rangeParamsA[4]; // luma:(min,max,feather,0) | color:(hue,sat,luma,tol)
uniform float u_rangeParamsB[4];// color: feather

uniform float u_linearActive;
uniform vec3 u_wbGains;
uniform float u_exposureGain;
uniform float u_highlights;
uniform float u_shadows;
uniform float u_whites;
uniform float u_blacks;
uniform float u_dehaze;
uniform float u_contrast;
uniform float u_saturation;
uniform float u_texture;
uniform float u_clarity;
uniform float u_sharpness;

varying vec2 v_texCoord;

float srgbToLinear(float c) {
  float x = clamp(c, 0.0, 1.0);
  return x <= 0.04045 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4);
}

float linearToSrgb(float c) {
  float x = clamp(c, 0.0, 1.0);
  return x <= 0.0031308 ? x * 12.92 : 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

float softClip(float x) {
  if (x <= 0.92) return max(x, 0.0);
  float excess = x - 0.92;
  return 0.92 + 0.08 * (1.0 - exp(-excess / 0.08));
}

float kSmoothstep(float edge0, float edge1, float x) {
  float t = clamp((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

float grainHash12(vec2 pIn) {
  vec3 p3 = fract(vec3(pIn.x, pIn.y, pIn.x) * 0.1031);
  p3 += dot(p3, vec3(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
  return fract((p3.x + p3.y) * p3.z);
}

vec3 rgbToHsl(vec3 rgb) {
  float r = clamp(rgb.r, 0.0, 1.0);
  float g = clamp(rgb.g, 0.0, 1.0);
  float b = clamp(rgb.b, 0.0, 1.0);
  float maxC = max(r, max(g, b));
  float minC = min(r, min(g, b));
  float l = 0.5 * (maxC + minC);
  float d = maxC - minC;
  if (d <= 1e-6) return vec3(0.0, 0.0, l);
  float s = l > 0.5 ? d / max(2.0 - maxC - minC, 1e-6) : d / max(maxC + minC, 1e-6);
  float h = 0.0;
  if (abs(maxC - r) <= 1e-6) {
    h = (g - b) / d + (g < b ? 6.0 : 0.0);
  } else if (abs(maxC - g) <= 1e-6) {
    h = (b - r) / d + 2.0;
  } else {
    h = (r - g) / d + 4.0;
  }
  h = mod(h * 60.0 + 360.0, 360.0);
  return vec3(h, clamp(s, 0.0, 1.0), clamp(l, 0.0, 1.0));
}

float evalLuminanceRange(vec3 rgb, float minL, float maxL, float feather) {
  float luma = clamp(0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b, 0.0, 1.0);
  float f = max(clamp(feather, 0.0, 0.5), 0.005);
  float lo = minL <= 0.001 ? 1.0 : kSmoothstep(minL - f, minL + f * 0.25, luma);
  float hi = maxL >= 0.999 ? 1.0 : (1.0 - kSmoothstep(maxL - f * 0.25, maxL + f, luma));
  return clamp(lo * hi, 0.0, 1.0);
}

float evalColorRange(vec3 rgb, float tHue, float tSat, float tLuma, float tolIn, float featherIn) {
  vec3 hsl = rgbToHsl(rgb);
  float rawDiff = mod(abs(hsl.x - tHue), 360.0);
  float hueDist = rawDiff > 180.0 ? 360.0 - rawDiff : rawDiff;
  float dh = hueDist / 180.0;
  float hueW = clamp(min(hsl.y, tSat) * 4.0, 0.0, 1.0);
  float ds = abs(hsl.y - tSat);
  float dl = abs(hsl.z - tLuma);
  float dist = sqrt(pow(dh * 2.2 * hueW, 2.0) + pow(ds * 0.9, 2.0) + pow(dl * 0.75, 2.0));
  float tol = clamp(tolIn, 0.02, 1.0);
  float f = max(tol * clamp(featherIn, 0.0, 1.0), 0.015);
  return 1.0 - kSmoothstep(max(0.0, tol - f * 0.5), tol + f, dist);
}

float sampleLuma(vec2 coord) {
  vec3 c = texture2D(u_prevImage, clamp(coord, vec2(0.0), vec2(1.0))).rgb;
  return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
}

void main() {
  vec2 uv = v_texCoord * u_uvTransform.xy + u_uvTransform.zw;
  if (
    uv.x < u_cropBounds.x || uv.x > u_cropBounds.z ||
    uv.y < u_cropBounds.y || uv.y > u_cropBounds.w
  ) {
    gl_FragColor = vec4(0.0, 0.0, 0.0, 0.0);
    return;
  }

  vec4 center = texture2D(u_prevImage, uv);
  vec3 origRgb = center.rgb;

  float acc = 0.0;
  float started = 0.0;
  if (u_hasSpatial > 0.5) {
    acc = texture2D(u_spatialMask, uv).r;
    started = 1.0;
  }

  for (int i = 0; i < 4; i++) {
    vec4 meta = u_rangeMeta[i];
    if (meta.x < 0.5) continue;
    vec4 pA = u_rangeParamsA[i];
    float val = meta.x < 1.5
      ? evalLuminanceRange(origRgb, pA.x, pA.y, pA.z)
      : evalColorRange(origRgb, pA.x, pA.y, pA.z, pA.w, u_rangeParamsB[i]);
    if (meta.z > 0.5) val = 1.0 - val;

    if (started < 0.5) {
      acc = (meta.y > 0.5 && meta.y < 1.5) ? clamp(1.0 - val, 0.0, 1.0) : clamp(val, 0.0, 1.0);
      started = 1.0;
    } else if (meta.y < 0.5) {
      acc = clamp(acc + val, 0.0, 1.0);
    } else if (meta.y < 1.5) {
      acc = clamp(acc - val, 0.0, 1.0);
    } else {
      acc = clamp(acc * val, 0.0, 1.0);
    }
  }

  if (u_layerInvert > 0.5) {
    acc = 1.0 - acc;
  }
  float alpha = clamp(acc * u_layerOpacity, 0.0, 1.0);

  if (u_overlayOnly > 0.5) {
    gl_FragColor = vec4(0.957, 0.247, 0.368, alpha * 0.48);
    return;
  }

  if (alpha <= 1e-4) {
    gl_FragColor = center;
    return;
  }

  vec3 rgb = origRgb;
  if (u_linearActive > 0.5) {
    vec3 lin = vec3(srgbToLinear(rgb.r), srgbToLinear(rgb.g), srgbToLinear(rgb.b));
    lin = max(lin * u_wbGains * u_exposureGain, vec3(0.0));
    rgb = vec3(
      linearToSrgb(softClip(lin.r)),
      linearToSrgb(softClip(lin.g)),
      linearToSrgb(softClip(lin.b))
    );
  }

  float luma = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
  float delta = 0.0;
  if (u_shadows != 0.0) {
    float sw = 1.0 - kSmoothstep(0.05, 0.55, luma);
    delta += (u_shadows / 100.0) * 0.28 * sw * sw;
  }
  if (u_highlights != 0.0) {
    float hw = kSmoothstep(0.45, 0.95, luma);
    delta += (u_highlights / 100.0) * 0.28 * hw * hw;
  }
  if (u_whites != 0.0) {
    delta += (u_whites / 100.0) * 0.22 * kSmoothstep(0.65, 1.0, luma);
  }
  if (u_blacks != 0.0) {
    delta += (u_blacks / 100.0) * 0.22 * (1.0 - kSmoothstep(0.0, 0.35, luma));
  }
  if (abs(delta) > 1e-6) {
    float targetLuma = clamp(luma + delta, 0.0, 1.0);
    if (luma > 1e-4) {
      rgb = clamp(rgb * (targetLuma / luma), 0.0, 1.0);
    } else {
      rgb = vec3(targetLuma);
    }
  }

  if (u_dehaze != 0.0) {
    float dh = u_dehaze / 100.0;
    rgb = clamp((rgb + vec3(-dh * 0.06)) * (1.0 + dh * 0.18), 0.0, 1.0);
  }

  if (u_contrast != 0.0) {
    float c = u_contrast / 100.0;
    float factor = c >= 0.0 ? 1.0 + c * 0.85 : 1.0 + c * 0.65;
    rgb = clamp((rgb - vec3(0.5)) * factor + vec3(0.5), 0.0, 1.0);
  }

  if (u_saturation != 0.0) {
    float lCur = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
    float sat = 1.0 + u_saturation / 100.0;
    rgb = clamp(vec3(lCur) + (rgb - vec3(lCur)) * sat, 0.0, 1.0);
  }

  rgb = mix(origRgb, rgb, alpha);

  // Local Spatial Detail (Texture, Clarity, Sharpness / negative skin softening)
  if (abs(u_texture) > 1e-4 || abs(u_clarity) > 1e-4 || abs(u_sharpness) > 1e-4) {
    float lc = 0.2126 * origRgb.r + 0.7152 * origRgb.g + 0.0722 * origRgb.b;
    float totalDelta = 0.0;

    if (abs(u_texture) > 1e-4) {
      float tex = u_texture / 100.0;
      vec2 rTex = u_texelSize * (2.5 * u_spatialScale);
      vec2 dTex = rTex * 0.70710677;
      float blurTex = 0.20 * lc
        + 0.12 * (
            sampleLuma(uv + vec2(-rTex.x, 0.0)) + sampleLuma(uv + vec2(rTex.x, 0.0)) +
            sampleLuma(uv + vec2(0.0, -rTex.y)) + sampleLuma(uv + vec2(0.0, rTex.y))
          )
        + 0.08 * (
            sampleLuma(uv + vec2(-dTex.x, -dTex.y)) + sampleLuma(uv + vec2(dTex.x, -dTex.y)) +
            sampleLuma(uv + vec2(-dTex.x, dTex.y)) + sampleLuma(uv + vec2(dTex.x, dTex.y))
          );
      totalDelta += clamp(lc - blurTex, -0.18, 0.18) * tex * 0.85;
    }

    if (abs(u_clarity) > 1e-4) {
      float clar = u_clarity / 100.0;
      vec2 rClar = u_texelSize * (11.0 * u_spatialScale);
      vec2 inClar = rClar * 0.45;
      vec2 dClar = rClar * 0.70710677;
      float blurClar = 0.16 * lc
        + 0.10 * (
            sampleLuma(uv + vec2(-inClar.x, 0.0)) + sampleLuma(uv + vec2(inClar.x, 0.0)) +
            sampleLuma(uv + vec2(0.0, -inClar.y)) + sampleLuma(uv + vec2(0.0, inClar.y))
          )
        + 0.06 * (
            sampleLuma(uv + vec2(-rClar.x, 0.0)) + sampleLuma(uv + vec2(rClar.x, 0.0)) +
            sampleLuma(uv + vec2(0.0, -rClar.y)) + sampleLuma(uv + vec2(0.0, rClar.y))
          )
        + 0.05 * (
            sampleLuma(uv + vec2(-dClar.x, -dClar.y)) + sampleLuma(uv + vec2(dClar.x, -dClar.y)) +
            sampleLuma(uv + vec2(-dClar.x, dClar.y)) + sampleLuma(uv + vec2(dClar.x, dClar.y))
          );
      float midMask = pow(clamp(4.0 * lc * (1.0 - lc), 1e-6, 1.0), 0.70);
      totalDelta += clamp(lc - blurClar, -0.30, 0.30) * clar * 1.15 * midMask;
    }

    if (u_sharpness > 1e-4) {
      float sharp = u_sharpness / 100.0;
      vec2 rSharp = u_texelSize * (1.0 * u_spatialScale);
      float avg4 = 0.25 * (
        sampleLuma(uv + vec2(-rSharp.x, 0.0)) + sampleLuma(uv + vec2(rSharp.x, 0.0)) +
        sampleLuma(uv + vec2(0.0, -rSharp.y)) + sampleLuma(uv + vec2(0.0, rSharp.y))
      );
      float hp = lc - avg4;
      float mag = max(abs(hp) - 0.008, 0.0);
      totalDelta += clamp((hp >= 0.0 ? 1.0 : -1.0) * mag * sharp * 1.10, -0.25, 0.25);
    } else if (u_sharpness < -1e-4) {
      float soften = -u_sharpness / 100.0;
      vec2 stepS = u_texelSize * (1.6 * u_spatialScale);
      float sigmaR = 0.04 + soften * 0.12;
      float inv2Sig2 = -0.5 / (sigmaR * sigmaR);
      float sumVal = lc;
      float sumW = 1.0;

      float l1 = sampleLuma(uv + vec2(-stepS.x, 0.0)); float w1 = 0.65 * exp((l1 - lc) * (l1 - lc) * inv2Sig2); sumVal += l1 * w1; sumW += w1;
      float l2 = sampleLuma(uv + vec2( stepS.x, 0.0)); float w2 = 0.65 * exp((l2 - lc) * (l2 - lc) * inv2Sig2); sumVal += l2 * w2; sumW += w2;
      float l3 = sampleLuma(uv + vec2(0.0, -stepS.y)); float w3 = 0.65 * exp((l3 - lc) * (l3 - lc) * inv2Sig2); sumVal += l3 * w3; sumW += w3;
      float l4 = sampleLuma(uv + vec2(0.0,  stepS.y)); float w4 = 0.65 * exp((l4 - lc) * (l4 - lc) * inv2Sig2); sumVal += l4 * w4; sumW += w4;
      float l5 = sampleLuma(uv + vec2(-stepS.x, -stepS.y)); float w5 = 0.45 * exp((l5 - lc) * (l5 - lc) * inv2Sig2); sumVal += l5 * w5; sumW += w5;
      float l6 = sampleLuma(uv + vec2( stepS.x, -stepS.y)); float w6 = 0.45 * exp((l6 - lc) * (l6 - lc) * inv2Sig2); sumVal += l6 * w6; sumW += w6;
      float l7 = sampleLuma(uv + vec2(-stepS.x,  stepS.y)); float w7 = 0.45 * exp((l7 - lc) * (l7 - lc) * inv2Sig2); sumVal += l7 * w7; sumW += w7;
      float l8 = sampleLuma(uv + vec2( stepS.x,  stepS.y)); float w8 = 0.45 * exp((l8 - lc) * (l8 - lc) * inv2Sig2); sumVal += l8 * w8; sumW += w8;

      totalDelta += (sumVal / max(sumW, 1e-5) - lc) * (soften * 0.90);
    }

    totalDelta *= alpha;
    if (abs(totalDelta) > 1e-6) {
      float curL = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
      float nextL = clamp(curL + totalDelta, 0.0, 1.0);
      float actualDelta = nextL - curL;
      float ratio = curL > 1e-4 ? clamp(nextL / curL, 0.0, 3.0) : 1.0;
      float blend = kSmoothstep(0.02, 0.16, curL);
      vec3 additive = clamp(rgb + vec3(actualDelta), 0.0, 1.0);
      vec3 mult = curL > 1e-4 ? clamp(rgb * ratio, 0.0, 1.0) : vec3(nextL);
      rgb = mix(additive, mult, blend);
    }
  }

  gl_FragColor = vec4(clamp(rgb, 0.0, 1.0), center.a);
}
`;

type GlContext = WebGLRenderingContext | WebGL2RenderingContext;

function isWebGl2Context(gl: GlContext): gl is WebGL2RenderingContext {
  return typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext;
}

interface CachedSpatialMaskEntry {
  readonly sig: string;
  readonly tex: WebGLTexture;
  readonly compiled: CompiledWebGlMaskData;
}

interface GlPipeline {
  readonly gl: GlContext;
  readonly isWebGl2: boolean;
  readonly hasHalfFloatFbo: boolean;
  readonly colorProgram: WebGLProgram;
  readonly maskProgram: WebGLProgram;
  readonly spatialProgram: WebGLProgram;
  readonly buffer: WebGLBuffer;
  readonly imageTex: WebGLTexture;
  readonly toneLutTex: WebGLTexture;
  readonly fbo: WebGLFramebuffer;
  readonly fboTex: WebGLTexture;
  readonly fboB: WebGLFramebuffer;
  readonly fboTexB: WebGLTexture;
  readonly spatialMaskCache: Map<string, CachedSpatialMaskEntry>;
  readonly bgMatteTex: WebGLTexture;
  bgMatteSig: string;
  fboWidth: number;
  fboHeight: number;
  readonly colorPosLoc: number;
  readonly maskPosLoc: number;
  readonly spatialPosLoc: number;
  readonly colorLoc: {
    readonly flipY: WebGLUniformLocation | null;
    readonly image: WebGLUniformLocation | null;
    readonly toneLut: WebGLUniformLocation | null;
    readonly uvTransform: WebGLUniformLocation | null;
    readonly cropBounds: WebGLUniformLocation | null;
    readonly orient: WebGLUniformLocation | null;
    readonly orientedDims: WebGLUniformLocation | null;
    readonly inputLinear: WebGLUniformLocation | null;
    readonly inputWideGamut: WebGLUniformLocation | null;
    readonly linearActive: WebGLUniformLocation | null;
    readonly exposureGain: WebGLUniformLocation | null;
    readonly wbGains: WebGLUniformLocation | null;
    readonly calActive: WebGLUniformLocation | null;
    readonly calShadowsTint: WebGLUniformLocation | null;
    readonly calMatrix: WebGLUniformLocation | null;
    readonly blackAndWhite: WebGLUniformLocation | null;
    readonly shadows: WebGLUniformLocation | null;
    readonly highlights: WebGLUniformLocation | null;
    readonly whites: WebGLUniformLocation | null;
    readonly blacks: WebGLUniformLocation | null;
    readonly dehaze: WebGLUniformLocation | null;
    readonly contrast: WebGLUniformLocation | null;
    readonly vibrance: WebGLUniformLocation | null;
    readonly saturation: WebGLUniformLocation | null;
    readonly toneCurveActive: WebGLUniformLocation | null;
    readonly hslActive: WebGLUniformLocation | null;
    readonly hsl: WebGLUniformLocation | null;
    readonly cgActive: WebGLUniformLocation | null;
    readonly cgShdTint: WebGLUniformLocation | null;
    readonly cgShdLum: WebGLUniformLocation | null;
    readonly cgMidTint: WebGLUniformLocation | null;
    readonly cgMidLum: WebGLUniformLocation | null;
    readonly cgHiTint: WebGLUniformLocation | null;
    readonly cgHiLum: WebGLUniformLocation | null;
    readonly cgPivot: WebGLUniformLocation | null;
    readonly cgSpread: WebGLUniformLocation | null;
    readonly vignetteAmount: WebGLUniformLocation | null;
    readonly vignetteMidpoint: WebGLUniformLocation | null;
    readonly vignetteFeather: WebGLUniformLocation | null;
    readonly clipFlags: WebGLUniformLocation | null;
    readonly bgMatte: WebGLUniformLocation | null;
    readonly bgMode: WebGLUniformLocation | null;
    readonly bgColor: WebGLUniformLocation | null;
    readonly bgColor2: WebGLUniformLocation | null;
    readonly bgGradStyle: WebGLUniformLocation | null;
    readonly bgGradAngle: WebGLUniformLocation | null;
    readonly bgFeather: WebGLUniformLocation | null;
  };
  readonly spatialLoc: {
    readonly flipY: WebGLUniformLocation | null;
    readonly graded: WebGLUniformLocation | null;
    readonly uvTransform: WebGLUniformLocation | null;
    readonly cropBounds: WebGLUniformLocation | null;
    readonly texelSize: WebGLUniformLocation | null;
    readonly spatialScale: WebGLUniformLocation | null;
    readonly nr: WebGLUniformLocation | null;
    readonly cnr: WebGLUniformLocation | null;
    readonly texture: WebGLUniformLocation | null;
    readonly clarity: WebGLUniformLocation | null;
    readonly sharpening: WebGLUniformLocation | null;
    readonly grainAmount: WebGLUniformLocation | null;
    readonly grainSize: WebGLUniformLocation | null;
    readonly clipFlags: WebGLUniformLocation | null;
    readonly bgMatte: WebGLUniformLocation | null;
    readonly bgMode: WebGLUniformLocation | null;
    readonly bgFeather: WebGLUniformLocation | null;
  };
  readonly maskLoc: {
    readonly flipY: WebGLUniformLocation | null;
    readonly prevImage: WebGLUniformLocation | null;
    readonly spatialMask: WebGLUniformLocation | null;
    readonly uvTransform: WebGLUniformLocation | null;
    readonly cropBounds: WebGLUniformLocation | null;
    readonly texelSize: WebGLUniformLocation | null;
    readonly spatialScale: WebGLUniformLocation | null;
    readonly hasSpatial: WebGLUniformLocation | null;
    readonly layerInvert: WebGLUniformLocation | null;
    readonly layerOpacity: WebGLUniformLocation | null;
    readonly overlayOnly: WebGLUniformLocation | null;
    readonly rangeMeta: WebGLUniformLocation | null;
    readonly rangeParamsA: WebGLUniformLocation | null;
    readonly rangeParamsB: WebGLUniformLocation | null;
    readonly linearActive: WebGLUniformLocation | null;
    readonly wbGains: WebGLUniformLocation | null;
    readonly exposureGain: WebGLUniformLocation | null;
    readonly highlights: WebGLUniformLocation | null;
    readonly shadows: WebGLUniformLocation | null;
    readonly whites: WebGLUniformLocation | null;
    readonly blacks: WebGLUniformLocation | null;
    readonly dehaze: WebGLUniformLocation | null;
    readonly contrast: WebGLUniformLocation | null;
    readonly saturation: WebGLUniformLocation | null;
    readonly texture: WebGLUniformLocation | null;
    readonly clarity: WebGLUniformLocation | null;
    readonly sharpness: WebGLUniformLocation | null;
  };
}

function compileShader(gl: GlContext, type: number, source: string): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function linkProgram(
  gl: GlContext,
  vertexSrc: string,
  fragmentSrc: string,
): WebGLProgram | null {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vertexSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSrc);
  if (!vs || !fs) {
    if (vs) gl.deleteShader(vs);
    if (fs) gl.deleteShader(fs);
    return null;
  }
  const program = gl.createProgram();
  if (!program) {
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return null;
  }
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    gl.deleteProgram(program);
    return null;
  }
  return program;
}

function initGlPipeline(canvas: HTMLCanvasElement): GlPipeline | null {
  const contextOptions: WebGLContextAttributes = {
    alpha: true,
    antialias: false,
    depth: false,
    stencil: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: true,
  };
  const gl2 = canvas.getContext('webgl2', contextOptions);
  const gl: GlContext | null = gl2 ?? canvas.getContext('webgl', contextOptions);
  if (!gl) return null;

  const isWebGl2 = isWebGl2Context(gl);
  const hasHalfFloatFbo = isWebGl2
    ? Boolean(
      gl.getExtension('EXT_color_buffer_float') ||
      gl.getExtension('EXT_color_buffer_half_float'),
    )
    : false;

  const colorProgram = linkProgram(gl, VERTEX_SHADER_SRC, COLOR_FRAGMENT_SHADER_SRC);
  const maskProgram = linkProgram(gl, VERTEX_SHADER_SRC, MASK_FRAGMENT_SHADER_SRC);
  const spatialProgram = linkProgram(gl, VERTEX_SHADER_SRC, SPATIAL_FRAGMENT_SHADER_SRC);
  if (!colorProgram || !maskProgram || !spatialProgram) {
    if (colorProgram) gl.deleteProgram(colorProgram);
    if (maskProgram) gl.deleteProgram(maskProgram);
    if (spatialProgram) gl.deleteProgram(spatialProgram);
    return null;
  }

  const buffer = gl.createBuffer();
  const imageTex = gl.createTexture();
  const toneLutTex = gl.createTexture();
  const fbo = gl.createFramebuffer();
  const fboTex = gl.createTexture();
  const fboB = gl.createFramebuffer();
  const fboTexB = gl.createTexture();
  const bgMatteTex = gl.createTexture();
  if (!buffer || !imageTex || !toneLutTex || !fbo || !fboTex || !fboB || !fboTexB || !bgMatteTex) {
    if (buffer) gl.deleteBuffer(buffer);
    if (imageTex) gl.deleteTexture(imageTex);
    if (toneLutTex) gl.deleteTexture(toneLutTex);
    if (fbo) gl.deleteFramebuffer(fbo);
    if (fboTex) gl.deleteTexture(fboTex);
    if (fboB) gl.deleteFramebuffer(fboB);
    if (fboTexB) gl.deleteTexture(fboTexB);
    if (bgMatteTex) gl.deleteTexture(bgMatteTex);
    gl.deleteProgram(colorProgram);
    gl.deleteProgram(maskProgram);
    gl.deleteProgram(spatialProgram);
    return null;
  }

  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
    gl.STATIC_DRAW,
  );

  for (const tex of [imageTex, toneLutTex, fboTex, fboTexB, bgMatteTex]) {
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  }
  gl.bindTexture(gl.TEXTURE_2D, bgMatteTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([255, 255, 255, 255]));

  return {
    gl,
    isWebGl2,
    hasHalfFloatFbo,
    colorProgram,
    maskProgram,
    spatialProgram,
    buffer,
    imageTex,
    toneLutTex,
    fbo,
    fboTex,
    fboB,
    fboTexB,
    spatialMaskCache: new Map(),
    bgMatteTex,
    bgMatteSig: '',
    fboWidth: 0,
    fboHeight: 0,
    colorPosLoc: gl.getAttribLocation(colorProgram, 'a_position'),
    maskPosLoc: gl.getAttribLocation(maskProgram, 'a_position'),
    spatialPosLoc: gl.getAttribLocation(spatialProgram, 'a_position'),
    colorLoc: {
      flipY: gl.getUniformLocation(colorProgram, 'u_flipY'),
      image: gl.getUniformLocation(colorProgram, 'u_image'),
      toneLut: gl.getUniformLocation(colorProgram, 'u_toneLut'),
      uvTransform: gl.getUniformLocation(colorProgram, 'u_uvTransform'),
      cropBounds: gl.getUniformLocation(colorProgram, 'u_cropBounds'),
      orient: gl.getUniformLocation(colorProgram, 'u_orient'),
      orientedDims: gl.getUniformLocation(colorProgram, 'u_orientedDims'),
      inputLinear: gl.getUniformLocation(colorProgram, 'u_inputLinear'),
      inputWideGamut: gl.getUniformLocation(colorProgram, 'u_inputWideGamut'),
      linearActive: gl.getUniformLocation(colorProgram, 'u_linearActive'),
      exposureGain: gl.getUniformLocation(colorProgram, 'u_exposureGain'),
      wbGains: gl.getUniformLocation(colorProgram, 'u_wbGains'),
      calActive: gl.getUniformLocation(colorProgram, 'u_calActive'),
      calShadowsTint: gl.getUniformLocation(colorProgram, 'u_calShadowsTint'),
      calMatrix: gl.getUniformLocation(colorProgram, 'u_calMatrix'),
      blackAndWhite: gl.getUniformLocation(colorProgram, 'u_blackAndWhite'),
      shadows: gl.getUniformLocation(colorProgram, 'u_shadows'),
      highlights: gl.getUniformLocation(colorProgram, 'u_highlights'),
      whites: gl.getUniformLocation(colorProgram, 'u_whites'),
      blacks: gl.getUniformLocation(colorProgram, 'u_blacks'),
      dehaze: gl.getUniformLocation(colorProgram, 'u_dehaze'),
      contrast: gl.getUniformLocation(colorProgram, 'u_contrast'),
      vibrance: gl.getUniformLocation(colorProgram, 'u_vibrance'),
      saturation: gl.getUniformLocation(colorProgram, 'u_saturation'),
      toneCurveActive: gl.getUniformLocation(colorProgram, 'u_toneCurveActive'),
      hslActive: gl.getUniformLocation(colorProgram, 'u_hslActive'),
      hsl:
        gl.getUniformLocation(colorProgram, 'u_hsl[0]') ??
        gl.getUniformLocation(colorProgram, 'u_hsl'),
      cgActive: gl.getUniformLocation(colorProgram, 'u_cgActive'),
      cgShdTint: gl.getUniformLocation(colorProgram, 'u_cgShdTint'),
      cgShdLum: gl.getUniformLocation(colorProgram, 'u_cgShdLum'),
      cgMidTint: gl.getUniformLocation(colorProgram, 'u_cgMidTint'),
      cgMidLum: gl.getUniformLocation(colorProgram, 'u_cgMidLum'),
      cgHiTint: gl.getUniformLocation(colorProgram, 'u_cgHiTint'),
      cgHiLum: gl.getUniformLocation(colorProgram, 'u_cgHiLum'),
      cgPivot: gl.getUniformLocation(colorProgram, 'u_cgPivot'),
      cgSpread: gl.getUniformLocation(colorProgram, 'u_cgSpread'),
      vignetteAmount: gl.getUniformLocation(colorProgram, 'u_vignetteAmount'),
      vignetteMidpoint: gl.getUniformLocation(colorProgram, 'u_vignetteMidpoint'),
      vignetteFeather: gl.getUniformLocation(colorProgram, 'u_vignetteFeather'),
      clipFlags: gl.getUniformLocation(colorProgram, 'u_clipFlags'),
      bgMatte: gl.getUniformLocation(colorProgram, 'u_bgMatte'),
      bgMode: gl.getUniformLocation(colorProgram, 'u_bgMode'),
      bgColor: gl.getUniformLocation(colorProgram, 'u_bgColor'),
      bgColor2: gl.getUniformLocation(colorProgram, 'u_bgColor2'),
      bgGradStyle: gl.getUniformLocation(colorProgram, 'u_bgGradStyle'),
      bgGradAngle: gl.getUniformLocation(colorProgram, 'u_bgGradAngle'),
      bgFeather: gl.getUniformLocation(colorProgram, 'u_bgFeather'),
    },
    spatialLoc: {
      flipY: gl.getUniformLocation(spatialProgram, 'u_flipY'),
      graded: gl.getUniformLocation(spatialProgram, 'u_graded'),
      uvTransform: gl.getUniformLocation(spatialProgram, 'u_uvTransform'),
      cropBounds: gl.getUniformLocation(spatialProgram, 'u_cropBounds'),
      texelSize: gl.getUniformLocation(spatialProgram, 'u_texelSize'),
      spatialScale: gl.getUniformLocation(spatialProgram, 'u_spatialScale'),
      nr: gl.getUniformLocation(spatialProgram, 'u_nr'),
      cnr: gl.getUniformLocation(spatialProgram, 'u_cnr'),
      texture: gl.getUniformLocation(spatialProgram, 'u_texture'),
      clarity: gl.getUniformLocation(spatialProgram, 'u_clarity'),
      sharpening: gl.getUniformLocation(spatialProgram, 'u_sharpening'),
      grainAmount: gl.getUniformLocation(spatialProgram, 'u_grainAmount'),
      grainSize: gl.getUniformLocation(spatialProgram, 'u_grainSize'),
      clipFlags: gl.getUniformLocation(spatialProgram, 'u_clipFlags'),
      bgMatte: gl.getUniformLocation(spatialProgram, 'u_bgMatte'),
      bgMode: gl.getUniformLocation(spatialProgram, 'u_bgMode'),
      bgFeather: gl.getUniformLocation(spatialProgram, 'u_bgFeather'),
    },
    maskLoc: {
      flipY: gl.getUniformLocation(maskProgram, 'u_flipY'),
      prevImage: gl.getUniformLocation(maskProgram, 'u_prevImage'),
      spatialMask: gl.getUniformLocation(maskProgram, 'u_spatialMask'),
      uvTransform: gl.getUniformLocation(maskProgram, 'u_uvTransform'),
      cropBounds: gl.getUniformLocation(maskProgram, 'u_cropBounds'),
      texelSize: gl.getUniformLocation(maskProgram, 'u_texelSize'),
      spatialScale: gl.getUniformLocation(maskProgram, 'u_spatialScale'),
      hasSpatial: gl.getUniformLocation(maskProgram, 'u_hasSpatial'),
      layerInvert: gl.getUniformLocation(maskProgram, 'u_layerInvert'),
      layerOpacity: gl.getUniformLocation(maskProgram, 'u_layerOpacity'),
      overlayOnly: gl.getUniformLocation(maskProgram, 'u_overlayOnly'),
      rangeMeta:
        gl.getUniformLocation(maskProgram, 'u_rangeMeta[0]') ??
        gl.getUniformLocation(maskProgram, 'u_rangeMeta'),
      rangeParamsA:
        gl.getUniformLocation(maskProgram, 'u_rangeParamsA[0]') ??
        gl.getUniformLocation(maskProgram, 'u_rangeParamsA'),
      rangeParamsB:
        gl.getUniformLocation(maskProgram, 'u_rangeParamsB[0]') ??
        gl.getUniformLocation(maskProgram, 'u_rangeParamsB'),
      linearActive: gl.getUniformLocation(maskProgram, 'u_linearActive'),
      wbGains: gl.getUniformLocation(maskProgram, 'u_wbGains'),
      exposureGain: gl.getUniformLocation(maskProgram, 'u_exposureGain'),
      highlights: gl.getUniformLocation(maskProgram, 'u_highlights'),
      shadows: gl.getUniformLocation(maskProgram, 'u_shadows'),
      whites: gl.getUniformLocation(maskProgram, 'u_whites'),
      blacks: gl.getUniformLocation(maskProgram, 'u_blacks'),
      dehaze: gl.getUniformLocation(maskProgram, 'u_dehaze'),
      contrast: gl.getUniformLocation(maskProgram, 'u_contrast'),
      saturation: gl.getUniformLocation(maskProgram, 'u_saturation'),
      texture: gl.getUniformLocation(maskProgram, 'u_texture'),
      clarity: gl.getUniformLocation(maskProgram, 'u_clarity'),
      sharpness: gl.getUniformLocation(maskProgram, 'u_sharpness'),
    },
  };
}

function ensureFboSize(
  pipeline: GlPipeline,
  imgW: number,
  imgH: number,
  needSecondFbo = false,
): { w: number; h: number; complete: boolean } {
  const maxDim = Math.max(imgW, imgH, 1);
  const scale = maxDim > MAX_FBO_DIM ? MAX_FBO_DIM / maxDim : 1;
  const targetW = Math.max(1, Math.round(imgW * scale));
  const targetH = Math.max(1, Math.round(imgH * scale));

  const { gl, fbo, fboTex, fboB, fboTexB } = pipeline;
  const setupSingleFbo = (
    targetFbo: WebGLFramebuffer,
    targetTex: WebGLTexture,
    useHalfFloat: boolean,
  ): boolean => {
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, targetTex);
    if (useHalfFloat && isWebGl2Context(gl)) {
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA16F,
        targetW,
        targetH,
        0,
        gl.RGBA,
        gl.HALF_FLOAT,
        null,
      );
    } else {
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        targetW,
        targetH,
        0,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        null,
      );
    }
    gl.bindTexture(gl.TEXTURE_2D, null);

    gl.bindFramebuffer(gl.FRAMEBUFFER, targetFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, targetTex, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return status === gl.FRAMEBUFFER_COMPLETE;
  };

  if (pipeline.fboWidth !== targetW || pipeline.fboHeight !== targetH) {
    const okA =
      (pipeline.hasHalfFloatFbo && setupSingleFbo(fbo, fboTex, true)) ||
      setupSingleFbo(fbo, fboTex, false);
    const okB =
      !needSecondFbo ||
      (pipeline.hasHalfFloatFbo && setupSingleFbo(fboB, fboTexB, true)) ||
      setupSingleFbo(fboB, fboTexB, false);

    if (!okA || !okB) {
      pipeline.fboWidth = 0;
      pipeline.fboHeight = 0;
      return { w: 0, h: 0, complete: false };
    }
    pipeline.fboWidth = targetW;
    pipeline.fboHeight = targetH;
  } else if (needSecondFbo) {
    // Ensure second FBO matches size if masks were added after Pass 2 already allocated fboA
    const okB =
      (pipeline.hasHalfFloatFbo && setupSingleFbo(fboB, fboTexB, true)) ||
      setupSingleFbo(fboB, fboTexB, false);
    if (!okB) {
      return { w: 0, h: 0, complete: false };
    }
  }

  return { w: pipeline.fboWidth, h: pipeline.fboHeight, complete: true };
}

function ensureBgMatteTexture(
  pipeline: GlPipeline,
  bg: BackgroundRemovalSettings | undefined,
): boolean {
  if (!bg || !isBackgroundRemovalActive(bg)) return false;
  const sig = `${bg.matteWidth}x${bg.matteHeight}:${bg.matteRleB64}`;
  if (pipeline.bgMatteSig === sig) return true;
  const decoded = decodeMatteRleB64(bg.matteRleB64, bg.matteWidth * bg.matteHeight);
  if (!decoded) return false;

  const rgba = new Uint8Array(decoded.length * 4);
  for (let i = 0; i < decoded.length; i++) {
    const v = decoded[i]!;
    const o = i * 4;
    rgba[o] = v;
    rgba[o + 1] = v;
    rgba[o + 2] = v;
    rgba[o + 3] = 255;
  }
  const { gl, bgMatteTex } = pipeline;
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, bgMatteTex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    bg.matteWidth,
    bg.matteHeight,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    rgba,
  );
  gl.bindTexture(gl.TEXTURE_2D, null);
  pipeline.bgMatteSig = sig;
  return true;
}

function getOrUploadSpatialMask(
  pipeline: GlPipeline,
  layer: MaskLayer,
  orientedW: number,
  orientedH: number,
): CachedSpatialMaskEntry | null {
  const { gl, spatialMaskCache } = pipeline;
  const sig = spatialMaskGeometrySignature(layer, orientedW, orientedH);
  const cached = spatialMaskCache.get(layer.id);
  if (cached && cached.sig === sig) {
    // Still recompile lightweight range uniform slots in case range sliders moved
    const rangeOnly = compileMaskLayerForWebGl(
      { ...layer, components: layer.components.filter((c) => c.type === 'luminance_range' || c.type === 'color_range') },
      orientedW,
      orientedH,
      16,
    );
    return {
      sig,
      tex: cached.tex,
      compiled: { ...cached.compiled, ranges: rangeOnly.ranges },
    };
  }

  const hasAiMatte = layer.components.some((c) => c.type === 'ai_matte');
  const compiled = compileMaskLayerForWebGl(layer, orientedW, orientedH, hasAiMatte ? 512 : 320);
  const tex = cached?.tex ?? gl.createTexture();
  if (!tex) return null;

  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    compiled.spatialWidth,
    compiled.spatialHeight,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    compiled.spatialRgba,
  );
  gl.bindTexture(gl.TEXTURE_2D, null);

  const entry: CachedSpatialMaskEntry = { sig, tex, compiled };
  spatialMaskCache.set(layer.id, entry);
  return entry;
}

function orientSourceUvJs(
  u: number,
  v: number,
  flipH: boolean,
  flipV: boolean,
  quarterTurns: number,
): [number, number] {
  const px = flipH ? 1 - u : u;
  const py = flipV ? 1 - v : v;
  const qt = ((quarterTurns % 4) + 4) % 4;
  if (qt === 1) return [py, 1 - px];
  if (qt === 2) return [1 - px, 1 - py];
  if (qt === 3) return [1 - py, px];
  return [px, py];
}

/**
 * Hardware-accelerated 2-pass WebGL2 Develop viewport running the complete Lightroom-style Rust
 * `color_math.rs` + `pixel_engine.rs` pipeline (16-bit Linear RAW HDR recovery, Camera Calibration,
 * Per-Channel RGB Curves, B&W Mix, 8-Band HSL, 3-Way Color Grading, Vignette Midpoint/Feather,
 * Luminance + Color NR, Texture, Clarity, Sharpening, Film Grain, Crop/Flip, and Clipping Warnings)
 * at 60–120fps.
 */
export const WebGlDevelopCanvas = memo(function WebGlDevelopCanvas({
  src,
  alt,
  recipe,
  workingBuffer = null,
  previewMaxDim = 2560,
  fit = 'cover',
  viewportZoom = 1,
  viewportPan = ZERO_VIEWPORT_PAN,
  viewportPadPx = 0,
  rotationDeg = 0,
  rotationScale = 1.06,
  applyCrop = true,
  showShadowClip = false,
  showHighlightClip = false,
  eyedropperActive = false,
  onSampleNeutralRgb,
  activeMaskId = null,
  showMaskOverlay = false,
  colorPickActive = false,
  onSampleColorRangeRgb,
  className = '',
  onNaturalSize,
  onMaxTextureDim,
}: WebGlDevelopCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const pipelineRef = useRef<GlPipeline | null>(null);
  const imgSizeRef = useRef<{ w: number; h: number }>({ w: 0, h: 0 });
  const sourceBitmapRef = useRef<CachedDevelopBitmap | null>(null);
  const loadedSrcRef = useRef<string>('');
  const inputLinearRef = useRef(false);
  const inputWideGamutRef = useRef(false);
  const textureReadyRef = useRef(false);
  const workingBufferRef = useRef<DevelopWorkingBuffer | null>(workingBuffer);
  const recipeRef = useRef(recipe);
  const fitRef = useRef(fit);
  const viewportZoomRef = useRef(viewportZoom);
  const viewportPanRef = useRef(viewportPan);
  const viewportPadPxRef = useRef(viewportPadPx);
  const applyCropRef = useRef(applyCrop);
  const clipFlagsRef = useRef({ shadow: showShadowClip, highlight: showHighlightClip });
  const maskOverlayRef = useRef({ activeMaskId, showMaskOverlay });
  const onNaturalSizeRef = useRef(onNaturalSize);
  const onMaxTextureDimRef = useRef(onMaxTextureDim);
  workingBufferRef.current = workingBuffer;
  recipeRef.current = recipe;
  fitRef.current = fit;
  viewportZoomRef.current = viewportZoom;
  viewportPanRef.current = viewportPan;
  viewportPadPxRef.current = viewportPadPx;
  applyCropRef.current = applyCrop;
  clipFlagsRef.current = { shadow: showShadowClip, highlight: showHighlightClip };
  maskOverlayRef.current = { activeMaskId, showMaskOverlay };
  onNaturalSizeRef.current = onNaturalSize;
  onMaxTextureDimRef.current = onMaxTextureDim;

  const [fallbackMode, setFallbackMode] = useState(false);
  const [textureVersion, setTextureVersion] = useState(0);
  const renderFrameRef = useRef<() => void>(() => {});
  const pendingRafRef = useRef<number | null>(null);
  const scheduleRender = () => {
    if (pendingRafRef.current !== null) return;
    pendingRafRef.current = window.requestAnimationFrame(() => {
      pendingRafRef.current = null;
      renderFrameRef.current();
    });
  };

  const rawFilterId = useId();
  const curveFilterId = `kroma-fb-curve-${rawFilterId.replace(/[^a-zA-Z0-9_-]/g, '')}`;

  const renderFrame = () => {
    const pipeline = pipelineRef.current;
    const canvas = canvasRef.current;
    if (!pipeline || !canvas || !textureReadyRef.current) return;
    // Skip GPU grading & swapchain resize while the Studio tab is hidden (`display: none`)
    if (canvas.clientWidth === 0 || canvas.clientHeight === 0) return;

    const {
      gl,
      colorProgram,
      maskProgram,
      spatialProgram,
      buffer,
      imageTex,
      toneLutTex,
      fbo,
      fboTex,
      fboB,
      fboTexB,
      spatialMaskCache,
      colorPosLoc,
      maskPosLoc,
      spatialPosLoc,
      colorLoc,
      maskLoc,
      spatialLoc,
    } = pipeline;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cssW = canvas.clientWidth;
    const cssH = canvas.clientHeight;
    const pixelW = Math.max(1, Math.round(cssW * dpr));
    const pixelH = Math.max(1, Math.round(cssH * dpr));

    if (canvas.width !== pixelW || canvas.height !== pixelH) {
      canvas.width = pixelW;
      canvas.height = pixelH;
    }

    const curRecipe = recipeRef.current;
    const crop = curRecipe.cropTransform;
    const qt = ((crop.quarterTurns ?? 0) % 4 + 4) % 4;
    const swapAxes = qt % 2 === 1;
    const orientedW = swapAxes ? imgSizeRef.current.h : imgSizeRef.current.w;
    const orientedH = swapAxes ? imgSizeRef.current.w : imgSizeRef.current.h;

    const u = compileDevelopShaderUniforms(curRecipe);
    const uv = computeViewportUvTransform(
      orientedW,
      orientedH,
      cssW,
      cssH,
      fitRef.current,
      applyCropRef.current ? crop.rect : undefined,
      viewportPadPxRef.current,
      viewportZoomRef.current,
      viewportPanRef.current,
    );

    const allMasks = curRecipe.masks ?? [];
    const effectiveMasks = allMasks.filter(isMaskLayerEffective);
    const overlayMask = maskOverlayRef.current.showMaskOverlay
      ? allMasks.find(
        (m) =>
          m.id === maskOverlayRef.current.activeMaskId &&
          m.enabled &&
          m.components.length > 0,
      ) ?? null
      : null;

    // Prune deleted mask GPU textures so VRAM stays tight
    if (spatialMaskCache.size > 0) {
      const liveIds = new Set(allMasks.map((m) => m.id));
      for (const [id, entry] of spatialMaskCache.entries()) {
        if (!liveIds.has(id)) {
          gl.deleteTexture(entry.tex);
          spatialMaskCache.delete(id);
        }
      }
    }

    const bgActive = ensureBgMatteTexture(pipeline, curRecipe.backgroundRemoval);
    const bgCfg = curRecipe.backgroundRemoval;
    const bgTransparent = Boolean(bgActive && bgCfg?.mode === 'transparent');
    const colorBgModeVal =
      bgActive && bgCfg
        ? bgCfg.mode === 'solid_color'
          ? 2.0
          : bgCfg.mode === 'gradient'
            ? 3.0
            : 0.0
        : 0.0;

    const needMultiPass =
      u.spatialStageActive || effectiveMasks.length > 0 || overlayMask !== null || bgTransparent;
    const fboState = needMultiPass
      ? ensureFboSize(pipeline, orientedW, orientedH, effectiveMasks.length > 0)
      : { w: 0, h: 0, complete: false };
    const runMultiPass = needMultiPass && fboState.complete;

    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.useProgram(colorProgram);
    gl.enableVertexAttribArray(colorPosLoc);
    gl.vertexAttribPointer(colorPosLoc, 2, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, imageTex);
    gl.uniform1i(colorLoc.image, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, toneLutTex);
    if (u.toneCurveActive) {
      const lutBytes = buildToneCurveLutRgba(curRecipe.toneCurve);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, lutBytes);
    }
    gl.uniform1i(colorLoc.toneLut, 1);

    const rotRad = Math.abs(crop.rotationDeg) > 1e-3 ? (crop.rotationDeg * Math.PI) / 180 : 0;
    gl.uniform4f(
      colorLoc.orient,
      crop.flipHorizontal ? 1.0 : 0.0,
      crop.flipVertical ? 1.0 : 0.0,
      qt,
      rotRad,
    );
    gl.uniform2f(colorLoc.orientedDims, orientedW, orientedH);
    gl.uniform1f(colorLoc.inputLinear, inputLinearRef.current ? 1.0 : 0.0);
    gl.uniform1f(colorLoc.inputWideGamut, inputWideGamutRef.current ? 1.0 : 0.0);
    gl.uniform1f(colorLoc.linearActive, u.linearStageActive ? 1.0 : 0.0);
    gl.uniform1f(colorLoc.exposureGain, u.exposureGain);
    gl.uniform3f(colorLoc.wbGains, u.wbGains[0], u.wbGains[1], u.wbGains[2]);
    gl.uniform1f(colorLoc.calActive, u.calActive ? 1.0 : 0.0);
    gl.uniform1f(colorLoc.calShadowsTint, u.calShadowsTint);
    gl.uniformMatrix3fv(colorLoc.calMatrix, false, new Float32Array(u.calMatrixColMajor));
    gl.uniform1f(colorLoc.blackAndWhite, u.blackAndWhite ? 1.0 : 0.0);
    gl.uniform1f(colorLoc.shadows, u.shadows);
    gl.uniform1f(colorLoc.highlights, u.highlights);
    gl.uniform1f(colorLoc.whites, u.whites);
    gl.uniform1f(colorLoc.blacks, u.blacks);
    gl.uniform1f(colorLoc.dehaze, u.dehaze);
    gl.uniform1f(colorLoc.contrast, u.contrast);
    gl.uniform1f(colorLoc.vibrance, u.vibrance);
    gl.uniform1f(colorLoc.saturation, u.saturation);
    gl.uniform1f(colorLoc.toneCurveActive, u.toneCurveActive ? 1.0 : 0.0);
    gl.uniform1f(colorLoc.hslActive, u.hslActive ? 1.0 : 0.0);
    gl.uniform3fv(colorLoc.hsl, new Float32Array(u.hslShifts));
    gl.uniform1f(colorLoc.cgActive, u.colorGradingActive ? 1.0 : 0.0);
    gl.uniform3f(colorLoc.cgShdTint, u.cgShdTint[0], u.cgShdTint[1], u.cgShdTint[2]);
    gl.uniform1f(colorLoc.cgShdLum, u.cgShdLum);
    gl.uniform3f(colorLoc.cgMidTint, u.cgMidTint[0], u.cgMidTint[1], u.cgMidTint[2]);
    gl.uniform1f(colorLoc.cgMidLum, u.cgMidLum);
    gl.uniform3f(colorLoc.cgHiTint, u.cgHiTint[0], u.cgHiTint[1], u.cgHiTint[2]);
    gl.uniform1f(colorLoc.cgHiLum, u.cgHiLum);
    gl.uniform1f(colorLoc.cgPivot, u.cgPivot);
    gl.uniform1f(colorLoc.cgSpread, u.cgSpread);
    gl.uniform1f(colorLoc.vignetteAmount, u.vignetteAmount);
    gl.uniform1f(colorLoc.vignetteMidpoint, u.vignetteMidpoint);
    gl.uniform1f(colorLoc.vignetteFeather, u.vignetteFeather);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, pipeline.bgMatteTex);
    gl.uniform1i(colorLoc.bgMatte, 2);
    gl.uniform1f(colorLoc.bgMode, colorBgModeVal);
    gl.uniform3f(
      colorLoc.bgColor,
      (bgCfg?.colorRgb[0] ?? 255) / 255,
      (bgCfg?.colorRgb[1] ?? 255) / 255,
      (bgCfg?.colorRgb[2] ?? 255) / 255,
    );
    gl.uniform3f(
      colorLoc.bgColor2,
      (bgCfg?.color2Rgb?.[0] ?? 20) / 255,
      (bgCfg?.color2Rgb?.[1] ?? 22) / 255,
      (bgCfg?.color2Rgb?.[2] ?? 27) / 255,
    );
    gl.uniform1f(colorLoc.bgGradStyle, bgCfg?.gradientStyle === 'linear' ? 1.0 : 0.0);
    gl.uniform1f(colorLoc.bgGradAngle, bgCfg?.gradientAngleDeg ?? 180);
    gl.uniform1f(colorLoc.bgFeather, bgCfg?.feather ?? 25);

    const clipX = clipFlagsRef.current.shadow ? 1.0 : 0.0;
    const clipY = clipFlagsRef.current.highlight ? 1.0 : 0.0;

    if (!runMultiPass) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, pixelW, pixelH);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniform1f(colorLoc.flipY, 1.0);
      gl.uniform4f(colorLoc.uvTransform, uv.scaleX, uv.scaleY, uv.offsetX, uv.offsetY);
      gl.uniform4f(
        colorLoc.cropBounds,
        uv.cropBounds[0],
        uv.cropBounds[1],
        uv.cropBounds[2],
        uv.cropBounds[3],
      );
      gl.uniform2f(colorLoc.clipFlags, clipX, clipY);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      return;
    }

    // Multi-pass path:
    // Pass 1: Global color grade into fboA
    const { w: fboW, h: fboH } = fboState;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.viewport(0, 0, fboW, fboH);
    gl.uniform1f(colorLoc.flipY, 0.0);
    gl.uniform4f(colorLoc.uvTransform, 1.0, 1.0, 0.0, 0.0);
    gl.uniform4f(colorLoc.cropBounds, 0.0, 0.0, 1.0, 1.0);
    gl.uniform2f(colorLoc.clipFlags, 0.0, 0.0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    // Helper to bind maskProgram uniforms for one mask layer (adjustment pass or Ruby Overlay pass)
    const bindMaskUniforms = (
      layer: MaskLayer,
      srcTex: WebGLTexture,
      cached: CachedSpatialMaskEntry,
      overlayOnly: boolean,
      flipY: number,
      uvX: number,
      uvY: number,
      uvOX: number,
      uvOY: number,
      cropBounds: readonly [number, number, number, number] = [0, 0, 1, 1],
    ) => {
      gl.useProgram(maskProgram);
      gl.enableVertexAttribArray(maskPosLoc);
      gl.vertexAttribPointer(maskPosLoc, 2, gl.FLOAT, false, 0, 0);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, srcTex);
      gl.uniform1i(maskLoc.prevImage, 0);

      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, cached.tex);
      gl.uniform1i(maskLoc.spatialMask, 1);

      gl.uniform1f(maskLoc.flipY, flipY);
      gl.uniform4f(maskLoc.uvTransform, uvX, uvY, uvOX, uvOY);
      gl.uniform4f(
        maskLoc.cropBounds,
        cropBounds[0],
        cropBounds[1],
        cropBounds[2],
        cropBounds[3],
      );
      gl.uniform2f(maskLoc.texelSize, 1.0 / fboW, 1.0 / fboH);
      gl.uniform1f(maskLoc.spatialScale, Math.max(0.5, Math.min(4.0, Math.max(fboW, fboH) / 1024.0)));
      gl.uniform1f(maskLoc.hasSpatial, cached.compiled.hasSpatial ? 1.0 : 0.0);
      gl.uniform1f(maskLoc.layerInvert, layer.inverted ? 1.0 : 0.0);
      gl.uniform1f(maskLoc.layerOpacity, Math.max(0, Math.min(1, layer.opacity / 100)));
      gl.uniform1f(maskLoc.overlayOnly, overlayOnly ? 1.0 : 0.0);

      const metaArr = new Float32Array(16);
      const pAArr = new Float32Array(16);
      const pBArr = new Float32Array(4);
      for (let i = 0; i < 4; i++) {
        const slot = cached.compiled.ranges[i];
        if (!slot) continue;
        metaArr[i * 4] = slot.mode;
        metaArr[i * 4 + 1] = slot.op;
        metaArr[i * 4 + 2] = slot.inverted;
        pAArr[i * 4] = slot.p0;
        pAArr[i * 4 + 1] = slot.p1;
        pAArr[i * 4 + 2] = slot.p2;
        pAArr[i * 4 + 3] = slot.p3;
        pBArr[i] = slot.p4;
      }
      gl.uniform4fv(maskLoc.rangeMeta, metaArr);
      gl.uniform4fv(maskLoc.rangeParamsA, pAArr);
      gl.uniform1fv(maskLoc.rangeParamsB, pBArr);

      const adj = layer.adjustments;
      const linActive =
        adj.temperatureShift !== 0 || adj.tintShift !== 0 || Math.abs(adj.exposureEv) > 1e-4;
      const tn = adj.temperatureShift / 100;
      const gn = adj.tintShift / 100;
      const rG = Math.max(0.25, 1 + tn * 0.28 + gn * 0.06);
      const gG = Math.max(0.25, 1 - gn * 0.14);
      const bG = Math.max(0.25, 1 - tn * 0.28 + gn * 0.06);
      gl.uniform1f(maskLoc.linearActive, linActive ? 1.0 : 0.0);
      gl.uniform3f(maskLoc.wbGains, rG / gG, 1.0, bG / gG);
      gl.uniform1f(maskLoc.exposureGain, Math.pow(2, Math.max(-4, Math.min(4, adj.exposureEv))));
      gl.uniform1f(maskLoc.highlights, adj.highlights);
      gl.uniform1f(maskLoc.shadows, adj.shadows);
      gl.uniform1f(maskLoc.whites, adj.whites);
      gl.uniform1f(maskLoc.blacks, adj.blacks);
      gl.uniform1f(maskLoc.dehaze, adj.dehaze);
      gl.uniform1f(maskLoc.contrast, adj.contrast);
      gl.uniform1f(maskLoc.saturation, adj.saturation);
      gl.uniform1f(maskLoc.texture, adj.texture);
      gl.uniform1f(maskLoc.clarity, adj.clarity);
      gl.uniform1f(maskLoc.sharpness, adj.sharpness);
    };

    // Pass 1b: Ping-pong Local & AI Mask passes (0 overhead when masks array is empty)
    let curSrcTex = fboTex;
    let curDstFbo = fboB;
    let curDstTex = fboTexB;
    let altFbo = fbo;

    for (const layer of effectiveMasks) {
      const cached = getOrUploadSpatialMask(pipeline, layer, orientedW, orientedH);
      if (!cached) continue;
      gl.bindFramebuffer(gl.FRAMEBUFFER, curDstFbo);
      gl.viewport(0, 0, fboW, fboH);
      bindMaskUniforms(layer, curSrcTex, cached, false, 0.0, 1.0, 1.0, 0.0, 0.0);
      gl.drawArrays(gl.TRIANGLES, 0, 6);

      // Swap ping-pong buffers
      const prevSrcTex = curSrcTex;
      const prevDstFbo = curDstFbo;
      curSrcTex = curDstTex;
      curDstTex = prevSrcTex;
      curDstFbo = altFbo;
      altFbo = prevDstFbo;
    }

    // Pass 2: Global spatial detail (Color NR, Luma NR, Texture, Clarity, Sharpening, Grain) + clip overlay
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, pixelW, pixelH);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(spatialProgram);
    gl.enableVertexAttribArray(spatialPosLoc);
    gl.vertexAttribPointer(spatialPosLoc, 2, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, pipeline.bgMatteTex);
    gl.uniform1i(spatialLoc.bgMatte, 1);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, curSrcTex);
    gl.uniform1i(spatialLoc.graded, 0);
    gl.uniform1f(spatialLoc.flipY, 1.0);
    gl.uniform4f(spatialLoc.uvTransform, uv.scaleX, uv.scaleY, uv.offsetX, uv.offsetY);
    gl.uniform4f(
      spatialLoc.cropBounds,
      uv.cropBounds[0],
      uv.cropBounds[1],
      uv.cropBounds[2],
      uv.cropBounds[3],
    );
    gl.uniform2f(spatialLoc.texelSize, 1.0 / fboW, 1.0 / fboH);
    gl.uniform1f(spatialLoc.spatialScale, Math.max(1.0, Math.max(fboW, fboH) / 1024.0));
    gl.uniform1f(spatialLoc.nr, u.luminanceNoiseReduction);
    gl.uniform1f(spatialLoc.cnr, u.colorNoiseReduction);
    gl.uniform1f(spatialLoc.texture, u.texture);
    gl.uniform1f(spatialLoc.clarity, u.clarity);
    gl.uniform1f(spatialLoc.sharpening, u.sharpeningAmount);
    gl.uniform1f(spatialLoc.grainAmount, u.grainAmount);
    gl.uniform1f(spatialLoc.grainSize, u.grainSize);
    gl.uniform2f(spatialLoc.clipFlags, clipX, clipY);
    gl.uniform1f(spatialLoc.bgMode, bgTransparent ? 1.0 : 0.0);
    gl.uniform1f(spatialLoc.bgFeather, bgCfg?.feather ?? 25);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    // Optional Ruby Mask Overlay Pass (O) blended cleanly on top without affecting image math
    if (overlayMask) {
      const cachedOverlay = getOrUploadSpatialMask(pipeline, overlayMask, orientedW, orientedH);
      if (cachedOverlay) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        bindMaskUniforms(
          overlayMask,
          curSrcTex,
          cachedOverlay,
          true,
          1.0,
          uv.scaleX,
          uv.scaleY,
          uv.offsetX,
          uv.offsetY,
          uv.cropBounds,
        );
        gl.drawArrays(gl.TRIANGLES, 0, 6);
        gl.disable(gl.BLEND);
      }
    }
  };
  renderFrameRef.current = renderFrame;

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    // Color Range Eyedropper: sample 5x5 average directly from the live graded WebGL buffer
    if (colorPickActive && onSampleColorRangeRgb && pipelineRef.current) {
      const { gl } = pipelineRef.current;
      const dprX = canvas.width / rect.width;
      const dprY = canvas.height / rect.height;
      const px = Math.round((event.clientX - rect.left) * dprX);
      const py = Math.round(canvas.height - 1 - (event.clientY - rect.top) * dprY);
      const x0 = Math.max(0, Math.min(canvas.width - 5, px - 2));
      const y0 = Math.max(0, Math.min(canvas.height - 5, py - 2));
      const pixels = new Uint8Array(5 * 5 * 4);
      gl.readPixels(x0, y0, 5, 5, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
      let sr = 0;
      let sg = 0;
      let sb = 0;
      let cnt = 0;
      for (let i = 0; i < pixels.length; i += 4) {
        if (pixels[i + 3]! > 0) {
          sr += pixels[i]! / 255;
          sg += pixels[i + 1]! / 255;
          sb += pixels[i + 2]! / 255;
          cnt += 1;
        }
      }
      if (cnt > 0) {
        onSampleColorRangeRgb(sr / cnt, sg / cnt, sb / cnt);
      }
      return;
    }

    if (!eyedropperActive || !onSampleNeutralRgb) return;

    const nx = (event.clientX - rect.left) / rect.width;
    const ny = (event.clientY - rect.top) / rect.height;
    const crop = recipeRef.current.cropTransform;
    const qt = ((crop.quarterTurns ?? 0) % 4 + 4) % 4;
    const swapAxes = qt % 2 === 1;
    const orientedW = swapAxes ? imgSizeRef.current.h : imgSizeRef.current.w;
    const orientedH = swapAxes ? imgSizeRef.current.w : imgSizeRef.current.h;
    const uv = computeViewportUvTransform(
      orientedW,
      orientedH,
      rect.width,
      rect.height,
      fitRef.current,
      applyCropRef.current ? crop.rect : undefined,
      viewportPadPxRef.current,
      viewportZoomRef.current,
      viewportPanRef.current,
    );
    const uOriented = nx * uv.scaleX + uv.offsetX;
    const vOriented = ny * uv.scaleY + uv.offsetY;
    const [minU, minV, maxU, maxV] = uv.cropBounds;
    if (uOriented < minU || uOriented > maxU || vOriented < minV || vOriented > maxV) return;

    const [srcU, srcV] = orientSourceUvJs(
      uOriented,
      vOriented,
      Boolean(crop.flipHorizontal),
      Boolean(crop.flipVertical),
      qt,
    );

    const wb = workingBufferRef.current;
    if (wb) {
      const cx = Math.min(wb.width - 1, Math.max(0, Math.round(srcU * (wb.width - 1))));
      const cy = Math.min(wb.height - 1, Math.max(0, Math.round(srcV * (wb.height - 1))));
      let sr = 0;
      let sg = 0;
      let sb = 0;
      let count = 0;
      for (let dy = -2; dy <= 2; dy++) {
        const y = Math.min(wb.height - 1, Math.max(0, cy + dy));
        for (let dx = -2; dx <= 2; dx++) {
          const x = Math.min(wb.width - 1, Math.max(0, cx + dx));
          const idx = (y * wb.width + x) * 4;
          let r = 0;
          let g = 0;
          let b = 0;
          if (wb.float32Rgba) {
            r = wb.float32Rgba[idx] ?? 0;
            g = wb.float32Rgba[idx + 1] ?? 0;
            b = wb.float32Rgba[idx + 2] ?? 0;
          } else if (wb.halfFloatRgba) {
            r = f16BitsToF32(wb.halfFloatRgba[idx] ?? 0);
            g = f16BitsToF32(wb.halfFloatRgba[idx + 1] ?? 0);
            b = f16BitsToF32(wb.halfFloatRgba[idx + 2] ?? 0);
          }
          if (wb.isLinear) {
            const lin709 = wb.isWideGamut
              ? compressRec2020ToLinearSrgb([r, g, b])
              : ([r, g, b] as [number, number, number]);
            sr += linearToSrgb(softClip(lin709[0]));
            sg += linearToSrgb(softClip(lin709[1]));
            sb += linearToSrgb(softClip(lin709[2]));
          } else {
            sr += r;
            sg += g;
            sb += b;
          }
          count += 1;
        }
      }
      if (count > 0) {
        onSampleNeutralRgb(sr / count, sg / count, sb / count);
      }
      return;
    }

    const cachedSource = sourceBitmapRef.current;
    if (cachedSource && cachedSource.width > 0 && cachedSource.height > 0) {
      try {
        const sampleCanvas = document.createElement('canvas');
        sampleCanvas.width = 5;
        sampleCanvas.height = 5;
        const ctx2d = sampleCanvas.getContext('2d');
        if (!ctx2d) return;
        const sx = Math.min(
          cachedSource.width - 5,
          Math.max(0, Math.round(srcU * cachedSource.width) - 2),
        );
        const sy = Math.min(
          cachedSource.height - 5,
          Math.max(0, Math.round(srcV * cachedSource.height) - 2),
        );
        ctx2d.drawImage(cachedSource.bitmap, sx, sy, 5, 5, 0, 0, 5, 5);
        const data = ctx2d.getImageData(0, 0, 5, 5).data;
        let sr = 0;
        let sg = 0;
        let sb = 0;
        const pxCount = data.length / 4;
        for (let i = 0; i < data.length; i += 4) {
          sr += data[i] / 255;
          sg += data[i + 1] / 255;
          sb += data[i + 2] / 255;
        }
        if (pxCount > 0) {
          onSampleNeutralRgb(sr / pxCount, sg / pxCount, sb / pxCount);
        }
      } catch {
        // Ignore cross-origin canvas tainting cleanly in Demo mode
      }
    }
  };

  // Initialize WebGL2/WebGL context once per mount
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const pipeline = initGlPipeline(canvas);
    if (!pipeline) {
      setFallbackMode(true);
      return;
    }
    pipelineRef.current = pipeline;

    const { gl, toneLutTex } = pipeline;
    const identity = new Uint8Array(256 * 4);
    for (let i = 0; i < 256; i++) {
      identity[i * 4] = i;
      identity[i * 4 + 1] = i;
      identity[i * 4 + 2] = i;
      identity[i * 4 + 3] = 255;
    }
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, toneLutTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, identity);

    const handleContextLost = (event: Event) => {
      event.preventDefault();
      setFallbackMode(true);
    };
    canvas.addEventListener('webglcontextlost', handleContextLost);

    const observer = new ResizeObserver(() => scheduleRender());
    observer.observe(canvas);
    const reportedMaxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    if (reportedMaxTex > 0) {
      onMaxTextureDimRef.current?.(reportedMaxTex);
    }

    return () => {
      if (pendingRafRef.current !== null) {
        window.cancelAnimationFrame(pendingRafRef.current);
        pendingRafRef.current = null;
      }
      canvas.removeEventListener('webglcontextlost', handleContextLost);
      observer.disconnect();
      pipelineRef.current = null;
      textureReadyRef.current = false;
      gl.deleteFramebuffer(pipeline.fbo);
      gl.deleteTexture(pipeline.fboTex);
      gl.deleteFramebuffer(pipeline.fboB);
      gl.deleteTexture(pipeline.fboTexB);
      gl.deleteTexture(pipeline.bgMatteTex);
      for (const entry of pipeline.spatialMaskCache.values()) {
        gl.deleteTexture(entry.tex);
      }
      pipeline.spatialMaskCache.clear();
      gl.deleteTexture(pipeline.imageTex);
      gl.deleteTexture(pipeline.toneLutTex);
      gl.deleteBuffer(pipeline.buffer);
      gl.deleteProgram(pipeline.colorProgram);
      gl.deleteProgram(pipeline.maskProgram);
      gl.deleteProgram(pipeline.spatialProgram);
    };
  }, []);

  // Upload 16-bit half-float (`RGBA16F`) working buffer immediately when available on WebGL2
  useEffect(() => {
    if (fallbackMode || !workingBuffer) return;
    const pipeline = pipelineRef.current;
    if (!pipeline || !isWebGl2Context(pipeline.gl)) return;

    try {
      const { gl, imageTex } = pipeline;
      const maxTexSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
      if (workingBuffer.width > maxTexSize || workingBuffer.height > maxTexSize) {
        return;
      }
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, imageTex);
      if (workingBuffer.halfFloatRgba) {
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 2);
        gl.texImage2D(
          gl.TEXTURE_2D,
          0,
          gl.RGBA16F,
          workingBuffer.width,
          workingBuffer.height,
          0,
          gl.RGBA,
          gl.HALF_FLOAT,
          workingBuffer.halfFloatRgba,
        );
      } else if (workingBuffer.float32Rgba) {
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
        gl.texImage2D(
          gl.TEXTURE_2D,
          0,
          gl.RGBA32F,
          workingBuffer.width,
          workingBuffer.height,
          0,
          gl.RGBA,
          gl.FLOAT,
          workingBuffer.float32Rgba,
        );
      } else {
        return;
      }
      imgSizeRef.current = { w: workingBuffer.width, h: workingBuffer.height };
      inputLinearRef.current = workingBuffer.isLinear;
      inputWideGamutRef.current = workingBuffer.isWideGamut;
      textureReadyRef.current = true;
      if (workingBuffer.width > 0 && workingBuffer.height > 0) {
        onNaturalSizeRef.current?.(workingBuffer.width, workingBuffer.height);
      }
      scheduleRender();
      setTextureVersion((v) => v + 1);
    } catch {
      // Keep the already-displayed 8-bit preview cleanly if driver rejects half-float upload
    }
  }, [workingBuffer, fallbackMode]);

  // Upload 8-bit preview/source photo texture immediately when `src` changes using the shared
  // downscaled ImageBitmap LRU cache (synchronous hit for pre-decoded neighbors, zero flash on 1:1 upgrade).
  const hasActiveHalfFloat = workingBuffer !== null;
  const clampedPreviewMaxDim = Math.min(8192, Math.max(640, previewMaxDim));
  useEffect(() => {
    if (fallbackMode) return;
    const pipeline = pipelineRef.current;
    if (!pipeline || !src) return;
    if (hasActiveHalfFloat && pipeline.isWebGl2) return;

    let cancelled = false;

    const uploadCachedBitmap = (entry: CachedDevelopBitmap) => {
      if (cancelled || !pipelineRef.current) return;
      if (workingBufferRef.current && pipelineRef.current.isWebGl2) return;
      try {
        sourceBitmapRef.current = entry;
        loadedSrcRef.current = src;
        const { gl, imageTex } = pipelineRef.current;
        const maxTexSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
        if (entry.width > maxTexSize || entry.height > maxTexSize) {
          setFallbackMode(true);
          return;
        }
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, imageTex);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, entry.bitmap);
        imgSizeRef.current = { w: entry.width, h: entry.height };
        inputLinearRef.current = false;
        inputWideGamutRef.current = false;
        textureReadyRef.current = true;
        if (entry.naturalWidth > 0 && entry.naturalHeight > 0) {
          onNaturalSizeRef.current?.(entry.naturalWidth, entry.naturalHeight);
        }
        scheduleRender();
        setTextureVersion((v) => v + 1);
      } catch {
        setFallbackMode(true);
      }
    };

    const syncHit = peekDevelopBitmap(src, clampedPreviewMaxDim);
    if (syncHit) {
      uploadCachedBitmap(syncHit);
      return () => {
        cancelled = true;
      };
    }

    if (loadedSrcRef.current !== src) {
      textureReadyRef.current = false;
    }
    void loadDevelopBitmap(src, clampedPreviewMaxDim).then((entry) => {
      if (cancelled) return;
      if (!entry) {
        setFallbackMode(true);
        return;
      }
      uploadCachedBitmap(entry);
    });

    return () => {
      cancelled = true;
    };
  }, [src, hasActiveHalfFloat, fallbackMode, clampedPreviewMaxDim]);

  // Coalesce redraws on any slider/recipe/mask/clipping/crop/texture/viewport change to at most one GPU pass per display frame
  useEffect(() => {
    if (!fallbackMode) {
      scheduleRender();
    }
  }, [
    recipe,
    fit,
    viewportZoom,
    viewportPan.x,
    viewportPan.y,
    viewportPadPx,
    applyCrop,
    showShadowClip,
    showHighlightClip,
    activeMaskId,
    showMaskOverlay,
    textureVersion,
    fallbackMode,
  ]);

  const hasToneCurve = fallbackMode && isToneCurveModified(recipe.toneCurve);
  const curveTableValues = useMemo(
    () => (hasToneCurve ? sampleToneCurveSvgTableValues(recipe.toneCurve, 33) : null),
    [hasToneCurve, recipe.toneCurve],
  );

  const transformStyle =
    rotationDeg !== 0
      ? rotationScale !== 1
        ? `rotate(${rotationDeg}deg) scale(${rotationScale})`
        : `rotate(${rotationDeg}deg)`
      : undefined;

  if (fallbackMode) {
    const baseFilter = recipeToCssFilter(recipe);
    const gradedFilter = curveTableValues ? `${baseFilter} url(#${curveFilterId})` : baseFilter;
    const vignette = recipe.detailOptics.vignetteAmount;
    const vignetteStrength = Math.min(0.8, Math.abs(vignette) / 115);
    const vignetteOverlay =
      vignette < 0
        ? `radial-gradient(circle, transparent 42%, rgba(0,0,0,${vignetteStrength.toFixed(2)}) 100%)`
        : vignette > 0
          ? `radial-gradient(circle, transparent 46%, rgba(255,255,255,${(vignetteStrength * 0.65).toFixed(2)}) 100%)`
          : undefined;

    return (
      <>
        {curveTableValues && (
          <svg width="0" height="0" className="absolute pointer-events-none" aria-hidden="true">
            <defs>
              <filter id={curveFilterId} colorInterpolationFilters="sRGB">
                <feComponentTransfer>
                  <feFuncR type="table" tableValues={curveTableValues} />
                  <feFuncG type="table" tableValues={curveTableValues} />
                  <feFuncB type="table" tableValues={curveTableValues} />
                </feComponentTransfer>
              </filter>
            </defs>
          </svg>
        )}
        <img
          src={src}
          alt={alt}
          draggable={false}
          decoding="async"
          onLoad={(e) => {
            const el = e.currentTarget;
            if (el.naturalWidth > 0 && el.naturalHeight > 0) {
              onNaturalSizeRef.current?.(el.naturalWidth, el.naturalHeight);
            }
          }}
          style={{ filter: gradedFilter, transform: transformStyle }}
          className={`w-full h-full ${fit === 'contain' ? 'object-contain' : 'object-cover'} ${className}`}
        />
        {vignetteOverlay && (
          <div
            className="absolute inset-0 pointer-events-none"
            style={{ background: vignetteOverlay }}
          />
        )}
      </>
    );
  }

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label={alt}
      onClick={handleCanvasClick}
      className={`block w-full h-full ${eyedropperActive || colorPickActive ? 'cursor-crosshair' : ''} ${className}`}
    />
  );
});
