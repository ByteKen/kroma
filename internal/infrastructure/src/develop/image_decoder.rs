use crate::develop::color_math::rec709_to_rec2020;
use image::imageops::FilterType;
use image::{DynamicImage, ImageDecoder, ImageReader, Rgb32FImage, RgbImage};
use kroma_app::error::AppError;
use kroma_domain::identity::parse_exif_datetime_epoch_sec;
use rawler::decoders::RawDecodeParams;
use rawler::imgop::develop::{Intermediate, ProcessingStep, RawDevelop};
use rawler::rawsource::RawSource;
use rawler::Orientation;
use rayon::prelude::*;
use std::fs::File;
use std::io::Read;
use std::path::Path;

const STANDARD_IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp", "tif", "tiff"];
const WEB_NATIVE_IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp"];
const EXIF_BEARING_STD_EXTENSIONS: &[&str] = &["jpg", "jpeg", "tif", "tiff"];
const TIFF_EXTENSIONS: &[&str] = &["tif", "tiff"];

pub const KR16_MAGIC: &[u8; 4] = b"KR16";
pub const KR32_MAGIC: &[u8; 4] = b"KR32";
pub const KR16_FLAG_LINEAR: u8 = 0x01;
pub const KR16_FLAG_RAW_DEMOSAIC: u8 = 0x02;
pub const KR_FLAG_WIDE_GAMUT_REC2020: u8 = 0x04;

/// High-bit-depth decoded pixel buffer shared by the Studio Develop WebGPU/WebGL2 viewport and
/// the full-resolution Export renderer (`wgpu` GPU + Rayon CPU) so previews and exports match 1:1.
/// - Camera RAWs decode into unclipped 32-bit scene-linear **Linear Rec.2020** `f32` RGB
///   (`is_linear = true`, `is_wide_gamut = true`, retaining >1.0 highlight headroom and wide gamut).
/// - Standard images (16-bit TIFF/PNG, 8-bit PNG/JPEG/WebP) decode into sRGB `f32` RGB.
#[derive(Debug, Clone)]
pub struct DecodedDevelopBuffer {
    pub image: Rgb32FImage,
    pub is_linear: bool,
    pub is_wide_gamut: bool,
    pub bit_depth: u8,
    pub is_raw_demosaic: bool,
}

/// Returns `true` if the path is a TIFF image (often 16-bit and not natively renderable in WebView2).
pub fn is_tiff_photo_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| {
            let lower = ext.to_ascii_lowercase();
            TIFF_EXTENSIONS.contains(&lower.as_str())
        })
}

/// Returns `true` if the path is a PNG file whose IHDR header declares 16-bit channels.
/// Reads only the first 26 bytes (<0.1ms) without decoding any pixels.
pub fn is_16bit_png_path(path: &Path) -> bool {
    let is_png = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("png"));
    if !is_png {
        return false;
    }
    let Ok(mut file) = File::open(path) else {
        return false;
    };
    let mut header = [0_u8; 26];
    if file.read_exact(&mut header).is_err() {
        return false;
    }
    // PNG signature (8 bytes) + IHDR length (4 bytes) + "IHDR" (4 bytes) + width (4) + height (4) + bit_depth (byte 24)
    header[0..8] == [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]
        && &header[12..16] == b"IHDR"
        && header[24] == 16
}

/// Returns `true` if WebView2 / Chromium can render this file directly without a cached JPEG proxy.
pub fn is_web_native_image_path(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    let lower = ext.to_ascii_lowercase();
    WEB_NATIVE_IMAGE_EXTENSIONS
        .iter()
        .any(|supported| *supported == lower)
}

/// Returns `true` if the path has a camera RAW file extension supported by `rawler`
/// (e.g., `.ARW`, `.CR2`, `.CR3`, `.NEF`, `.DNG`, `.RAF`, `.ORF`, `.RW2`, `.PEF`, `.SRW`).
pub fn is_raw_photo_path(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    let upper = ext.to_ascii_uppercase();
    rawler::decoders::supported_extensions()
        .iter()
        .any(|supported| *supported == upper)
}

/// Returns `true` if the path has a standard raster image extension (`JPG`, `PNG`, `WEBP`, `TIFF`).
pub fn is_standard_photo_path(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    let lower = ext.to_ascii_lowercase();
    STANDARD_IMAGE_EXTENSIONS
        .iter()
        .any(|supported| *supported == lower)
}

/// Returns `true` for both camera RAW formats and standard raster formats.
pub fn is_supported_photo_path(path: &Path) -> bool {
    is_raw_photo_path(path) || is_standard_photo_path(path)
}

/// Returns all 29 camera RAW extensions supported by the backend (uppercase).
pub fn supported_raw_extensions() -> &'static [&'static str] {
    rawler::decoders::supported_extensions()
}

/// Extracts the exact camera shutter capture timestamp (`DateTimeOriginal` / `CreateDate`) from
/// EXIF metadata in either camera RAW files (`.ARW`, `.CR3`, `.NEF`, etc.) or standard JPEGs/TIFFs.
/// Returns `None` if no embedded EXIF capture date exists (never fabricates from file mtime).
pub fn extract_exif_capture_timestamp_epoch_sec(source_path: &str) -> Option<i64> {
    let path = Path::new(source_path);
    if is_raw_photo_path(path) {
        let raw_source = RawSource::new(path).ok()?;
        let decoder = rawler::get_decoder(&raw_source).ok()?;
        let meta = decoder
            .raw_metadata(&raw_source, &RawDecodeParams::default())
            .ok()?;
        let date_str = meta
            .exif
            .date_time_original
            .as_deref()
            .or(meta.exif.create_date.as_deref())?;
        return parse_exif_datetime_epoch_sec(date_str);
    }

    // Fast skip for PNG / WebP which do not carry camera shutter EXIF in standard shoot folders
    let has_std_exif = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| {
            let lower = ext.to_ascii_lowercase();
            EXIF_BEARING_STD_EXTENSIONS.contains(&lower.as_str())
        });
    if !has_std_exif {
        return None;
    }

    let reader = ImageReader::open(path).ok()?.with_guessed_format().ok()?;
    let mut decoder = reader.into_decoder().ok()?;
    let exif_bytes = decoder.exif_metadata().ok().flatten()?;
    extract_exif_datetime_from_tiff_bytes(&exif_bytes)
}

/// Lightweight parser locating EXIF `DateTimeOriginal` (`0x9003`), `DateTimeDigitized` (`0x9004`),
/// or standard `YYYY:MM:DD HH:MM:SS` ASCII timestamps inside raw TIFF/EXIF header bytes.
fn extract_exif_datetime_from_tiff_bytes(bytes: &[u8]) -> Option<i64> {
    if bytes.len() < 19 {
        return None;
    }
    // Scan ASCII EXIF header payload for standard 19-byte "YYYY:MM:DD HH:MM:SS" camera timestamps.
    // Exif IFD typically stores DateTimeOriginal right after ModifyDate; if multiple timestamps exist,
    // prefer the earliest valid shutter timestamp (since export tools update ModifyDate to a later date!).
    let mut earliest: Option<i64> = None;
    for window in bytes.windows(19) {
        if window[4] == b':'
            && window[7] == b':'
            && window[10] == b' '
            && window[13] == b':'
            && window[16] == b':'
            && let Ok(text) = std::str::from_utf8(window)
            && let Some(ts) = parse_exif_datetime_epoch_sec(text)
        {
            earliest = Some(earliest.map_or(ts, |prev| prev.min(ts)));
        }
    }
    earliest
}

/// Full-quality photo decode for Develop & Export:
/// - Standard files (`JPEG`, `PNG`, `TIFF`, `WebP`) decode with automatic EXIF orientation uprighting.
/// - Camera RAW files (`.ARW`, `.CR3`, `.CR2`, `.NEF`, `.DNG`, `.RAF`, etc.) run full
///   16-bit Bayer demosaicing + camera color matrix calibration + EXIF orientation uprighting.
pub fn decode_full_dynamic(source_path: &str) -> Result<DynamicImage, AppError> {
    let path = Path::new(source_path);
    if !is_raw_photo_path(path) {
        return decode_standard_upright(path, source_path);
    }

    decode_raw_full_dynamic(path)
}

/// Decodes any supported photo into a 32-bit float `DecodedDevelopBuffer` (optionally downscaled
/// to `max_dim` when `max_dim > 0`) without clipping Camera RAW linear highlights above `1.0`.
/// - Camera RAW: `rawler` Bayer PPG demosaic + As-Shot WB + D65 camera matrix -> **scene-linear f32**.
/// - Standard (16-bit TIFF/PNG, 8-bit PNG/JPEG/WebP): upright sRGB `f32` preserving full 16-bit precision.
pub fn decode_develop_buffer(
    source_path: &str,
    max_dim: u32,
) -> Result<DecodedDevelopBuffer, AppError> {
    let path = Path::new(source_path);
    if is_raw_photo_path(path) {
        return decode_raw_linear_buffer(path, max_dim);
    }

    let dyn_img = decode_standard_upright_max_dim(path, source_path, max_dim)?;
    let bit_depth = dynamic_image_bit_depth(&dyn_img);
    Ok(DecodedDevelopBuffer {
        image: dyn_img.to_rgb32f(),
        is_linear: false,
        is_wide_gamut: false,
        bit_depth,
        is_raw_demosaic: false,
    })
}

/// Fast-path loader for the Studio Develop WebGL2 `RGBA16F` working buffer:
/// - Camera RAW (all 29 formats) & TIFF: always returns `Some(DecodedDevelopBuffer)`.
/// - PNG: inspects only the 26-byte IHDR header; returns `Some(...)` for 16-bit PNGs and `Ok(None)`
///   immediately (<0.1ms) for 8-bit PNGs so WebView2 can keep zero-latency native rendering.
/// - 8-bit JPEG/WebP: returns `Ok(None)` immediately.
pub fn decode_high_bit_depth_working_buffer(
    source_path: &str,
    max_dim: u32,
) -> Result<Option<DecodedDevelopBuffer>, AppError> {
    let path = Path::new(source_path);
    if is_raw_photo_path(path) || is_tiff_photo_path(path) || is_16bit_png_path(path) {
        return Ok(Some(decode_develop_buffer(source_path, max_dim)?));
    }
    Ok(None)
}

/// Fast parallel Rayon box/area-averaging downscaler for 32-bit float RGB images that NEVER clamps
/// values > 1.0 (preserving scene-linear Camera RAW highlight headroom and suppressing moire).
pub fn downscale_rgb32f_unclamped(src: &Rgb32FImage, max_dim: u32) -> Rgb32FImage {
    let (sw, sh) = src.dimensions();
    if max_dim == 0 || sw.max(sh) <= max_dim || sw == 0 || sh == 0 {
        return src.clone();
    }
    let scale = (max_dim as f32) / (sw.max(sh) as f32);
    let dw = ((sw as f32) * scale).round().max(1.0) as u32;
    let dh = ((sh as f32) * scale).round().max(1.0) as u32;

    let sw_u = sw as usize;
    let sh_u = sh as usize;
    let dw_u = dw as usize;
    let dh_u = dh as usize;
    let raw = src.as_raw();
    let rx = (sw as f32) / (dw as f32);
    let ry = (sh as f32) / (dh as f32);

    let mut out = vec![0.0_f32; dw_u * dh_u * 3];
    out.par_chunks_exact_mut(dw_u * 3)
        .enumerate()
        .for_each(|(dy, row)| {
            let sy0 = ((dy as f32) * ry).floor() as usize;
            let sy1 = (((dy + 1) as f32) * ry).ceil() as usize;
            let sy0 = sy0.min(sh_u - 1);
            let sy1 = sy1.clamp(sy0 + 1, sh_u);

            for dx in 0..dw_u {
                let sx0 = ((dx as f32) * rx).floor() as usize;
                let sx1 = (((dx + 1) as f32) * rx).ceil() as usize;
                let sx0 = sx0.min(sw_u - 1);
                let sx1 = sx1.clamp(sx0 + 1, sw_u);

                let mut sr = 0.0_f32;
                let mut sg = 0.0_f32;
                let mut sb = 0.0_f32;
                let mut count = 0_u32;
                for sy in sy0..sy1 {
                    let row_off = sy * sw_u * 3;
                    for sx in sx0..sx1 {
                        let idx = row_off + sx * 3;
                        sr += raw[idx];
                        sg += raw[idx + 1];
                        sb += raw[idx + 2];
                        count += 1;
                    }
                }
                let inv = 1.0 / (count.max(1) as f32);
                let dst_idx = dx * 3;
                row[dst_idx] = sr * inv;
                row[dst_idx + 1] = sg * inv;
                row[dst_idx + 2] = sb * inv;
            }
        });

    Rgb32FImage::from_raw(dw, dh, out).expect("downscaled rgb32f dimensions match buffer")
}

/// Converts a 32-bit float to IEEE 754 binary16 (half-float) bit representation (`u16`).
/// Preserves HDR scene-linear highlights up to 65504.0 and subnormal precision near black.
#[inline]
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x007F_FFFF;

    if exp == 0xFF {
        if frac != 0 {
            return 0;
        }
        return sign | 0x7BFF;
    }

    let unbiased = exp - 127;
    if unbiased > 15 {
        return sign | 0x7BFF;
    }
    if unbiased < -24 {
        return sign;
    }
    if unbiased < -14 {
        let mant = 0x0080_0000 | frac;
        let shift = (-14 - unbiased) as u32 + 13;
        let half_frac = (mant + (1_u32 << (shift - 1))) >> shift;
        return sign | (half_frac as u16);
    }

    let biased = (unbiased + 15) as u32;
    let rounded_frac = frac + 0x0000_1000;
    if rounded_frac >= 0x0080_0000 {
        let next_exp = biased + 1;
        if next_exp >= 31 {
            return sign | 0x7BFF;
        }
        return sign | ((next_exp as u16) << 10);
    }
    sign | ((biased as u16) << 10) | ((rounded_frac >> 13) as u16)
}

/// Encodes a `DecodedDevelopBuffer` into Kroma's zero-copy binary `KR16` wire format:
/// - 16-byte aligned header (`"KR16"`, LE `width: u32`, LE `height: u32`, `flags: u8`, `bit_depth: u8`, `reserved: u16`)
/// - Followed by `width * height * 4` little-endian IEEE 754 `f16` RGBA values (alpha = `1.0f16` = `0x3C00`),
///   ready for zero-copy `new Uint16Array(buf, 16)` upload into WebGL2 `gl.RGBA16F` / `gl.HALF_FLOAT`.
pub fn encode_kr16_rgba16f(buffer: &DecodedDevelopBuffer) -> Vec<u8> {
    let (w, h) = buffer.image.dimensions();
    let pixel_count = (w as usize) * (h as usize);
    let mut bytes = vec![0_u8; 16 + pixel_count * 8];

    let mut flags: u8 = 0;
    if buffer.is_linear {
        flags |= KR16_FLAG_LINEAR;
    }
    if buffer.is_raw_demosaic {
        flags |= KR16_FLAG_RAW_DEMOSAIC;
    }
    if buffer.is_wide_gamut {
        flags |= KR_FLAG_WIDE_GAMUT_REC2020;
    }

    bytes[0..4].copy_from_slice(KR16_MAGIC);
    bytes[4..8].copy_from_slice(&w.to_le_bytes());
    bytes[8..12].copy_from_slice(&h.to_le_bytes());
    bytes[12] = flags;
    bytes[13] = buffer.bit_depth;
    bytes[14] = 0;
    bytes[15] = 0;

    let src = buffer.image.as_raw();
    const ALPHA_ONE_F16: [u8; 2] = 0x3C00_u16.to_le_bytes();
    bytes[16..]
        .par_chunks_exact_mut(8)
        .enumerate()
        .for_each(|(i, dst_px)| {
            let s = i * 3;
            let r = f32_to_f16_bits(src[s].max(0.0)).to_le_bytes();
            let g = f32_to_f16_bits(src[s + 1].max(0.0)).to_le_bytes();
            let b = f32_to_f16_bits(src[s + 2].max(0.0)).to_le_bytes();
            dst_px[0] = r[0];
            dst_px[1] = r[1];
            dst_px[2] = g[0];
            dst_px[3] = g[1];
            dst_px[4] = b[0];
            dst_px[5] = b[1];
            dst_px[6] = ALPHA_ONE_F16[0];
            dst_px[7] = ALPHA_ONE_F16[1];
        });

    bytes
}

/// Encodes a `DecodedDevelopBuffer` into Kroma's zero-copy 32-bit float binary `KR32` wire format:
/// - 16-byte aligned header (`"KR32"`, LE `width: u32`, LE `height: u32`, `flags: u8`, `bit_depth: u8`, `reserved: u16`)
/// - Followed by `width * height * 4` little-endian IEEE 754 `f32` RGBA values (`alpha = 1.0_f32`),
///   ready for zero-copy `new Float32Array(buf, 16)` upload into WebGPU `rgba32float` textures.
pub fn encode_kr32_rgba32f(buffer: &DecodedDevelopBuffer) -> Vec<u8> {
    let (w, h) = buffer.image.dimensions();
    let pixel_count = (w as usize) * (h as usize);
    let mut bytes = vec![0_u8; 16 + pixel_count * 16];

    let mut flags: u8 = 0;
    if buffer.is_linear {
        flags |= KR16_FLAG_LINEAR;
    }
    if buffer.is_raw_demosaic {
        flags |= KR16_FLAG_RAW_DEMOSAIC;
    }
    if buffer.is_wide_gamut {
        flags |= KR_FLAG_WIDE_GAMUT_REC2020;
    }

    bytes[0..4].copy_from_slice(KR32_MAGIC);
    bytes[4..8].copy_from_slice(&w.to_le_bytes());
    bytes[8..12].copy_from_slice(&h.to_le_bytes());
    bytes[12] = flags;
    bytes[13] = buffer.bit_depth.max(32);
    bytes[14] = 0;
    bytes[15] = 0;

    let src = buffer.image.as_raw();
    const ALPHA_ONE_F32: [u8; 4] = 1.0_f32.to_le_bytes();
    bytes[16..]
        .par_chunks_exact_mut(16)
        .enumerate()
        .for_each(|(i, dst_px)| {
            let s = i * 3;
            let r = src[s].max(0.0).to_le_bytes();
            let g = src[s + 1].max(0.0).to_le_bytes();
            let b = src[s + 2].max(0.0).to_le_bytes();
            dst_px[0..4].copy_from_slice(&r);
            dst_px[4..8].copy_from_slice(&g);
            dst_px[8..12].copy_from_slice(&b);
            dst_px[12..16].copy_from_slice(&ALPHA_ONE_F32);
        });

    bytes
}

fn dynamic_image_bit_depth(img: &DynamicImage) -> u8 {
    match img {
        DynamicImage::ImageLuma16(_)
        | DynamicImage::ImageLumaA16(_)
        | DynamicImage::ImageRgb16(_)
        | DynamicImage::ImageRgba16(_)
        | DynamicImage::ImageRgb32F(_)
        | DynamicImage::ImageRgba32F(_) => 16,
        _ => 8,
    }
}

/// Converts `rawler`'s calibrated linear output into wide-gamut **Linear Rec.2020** `Rgb32FImage`.
/// Signed out-of-sRGB-gamut camera values (negative Rec.709 channels) are converted to Rec.2020
/// *before* clamping `< 0.0`, preserving 75.8% of the CIE visible gamut and >1.0 HDR headroom.
fn intermediate_to_rgb32f_rec2020(intermediate: Intermediate) -> Option<Rgb32FImage> {
    match intermediate {
        Intermediate::Monochrome(pixels) => {
            let w = pixels.width as u32;
            let h = pixels.height as u32;
            let mut buf = Vec::with_capacity(pixels.data.len() * 3);
            for v in pixels.data {
                let c = if v.is_finite() { v.max(0.0) } else { 0.0 };
                buf.push(c);
                buf.push(c);
                buf.push(c);
            }
            Rgb32FImage::from_raw(w, h, buf)
        }
        Intermediate::ThreeColor(pixels) => {
            let w = pixels.width as u32;
            let h = pixels.height as u32;
            let mut buf = Vec::with_capacity(pixels.data.len() * 3);
            for px in pixels.data {
                let r709 = if px[0].is_finite() { px[0] } else { 0.0 };
                let g709 = if px[1].is_finite() { px[1] } else { 0.0 };
                let b709 = if px[2].is_finite() { px[2] } else { 0.0 };
                let rec2020 = rec709_to_rec2020([r709, g709, b709]);
                buf.push(rec2020[0].max(0.0));
                buf.push(rec2020[1].max(0.0));
                buf.push(rec2020[2].max(0.0));
            }
            Rgb32FImage::from_raw(w, h, buf)
        }
        Intermediate::FourColor(pixels) => {
            let w = pixels.width as u32;
            let h = pixels.height as u32;
            let mut buf = Vec::with_capacity(pixels.data.len() * 3);
            for px in pixels.data {
                let r709 = if px[0].is_finite() { px[0] } else { 0.0 };
                let g709 = if px[1].is_finite() { px[1] } else { 0.0 };
                let b709 = if px[2].is_finite() { px[2] } else { 0.0 };
                let rec2020 = rec709_to_rec2020([r709, g709, b709]);
                buf.push(rec2020[0].max(0.0));
                buf.push(rec2020[1].max(0.0));
                buf.push(rec2020[2].max(0.0));
            }
            Rgb32FImage::from_raw(w, h, buf)
        }
    }
}

fn decode_raw_linear_buffer(path: &Path, max_dim: u32) -> Result<DecodedDevelopBuffer, AppError> {
    let raw_source = RawSource::new(path).map_err(|err| {
        AppError::ImageProcessingFailure(format!(
            "failed to open RAW source '{}': {err}",
            path.display()
        ))
    })?;
    let decoder = rawler::get_decoder(&raw_source).map_err(|err| {
        AppError::ImageProcessingFailure(format!(
            "unsupported RAW format '{}': {err}",
            path.display()
        ))
    })?;
    let params = RawDecodeParams::default();

    // Scene-linear RAW pipeline (omit ProcessingStep::SRgb so full linear dynamic range,
    // wide-gamut Rec.2020 chromaticities, and >1.0 highlight headroom stay intact in 32-bit float).
    let linear_developer = RawDevelop {
        steps: vec![
            ProcessingStep::Rescale,
            ProcessingStep::Demosaic,
            ProcessingStep::CropActiveArea,
            ProcessingStep::WhiteBalance,
            ProcessingStep::Calibrate,
            ProcessingStep::CropDefault,
        ],
    };

    match decoder.raw_image(&raw_source, &params, false) {
        Ok(raw_image) => {
            let orientation = raw_image.orientation;
            match linear_developer.develop_intermediate(&raw_image) {
                Ok(intermediate) => {
                    if let Some(linear_full) = intermediate_to_rgb32f_rec2020(intermediate) {
                        // Downscale BEFORE rotating so portrait orientation operates on compact buffers
                        let scaled = if max_dim > 0 {
                            downscale_rgb32f_unclamped(&linear_full, max_dim)
                        } else {
                            linear_full
                        };
                        let upright =
                            apply_exif_orientation(DynamicImage::ImageRgb32F(scaled), orientation)
                                .into_rgb32f();
                        return Ok(DecodedDevelopBuffer {
                            image: upright,
                            is_linear: true,
                            is_wide_gamut: true,
                            bit_depth: 32,
                            is_raw_demosaic: true,
                        });
                    }
                }
                Err(dev_err) => {
                    if let Some(fallback) =
                        try_embedded_preview(decoder.as_ref(), &raw_source, &params)
                    {
                        let shrunk = resize_if_larger(fallback, max_dim);
                        let upright = apply_exif_orientation(shrunk, orientation);
                        return Ok(DecodedDevelopBuffer {
                            image: upright.to_rgb32f(),
                            is_linear: false,
                            is_wide_gamut: false,
                            bit_depth: 8,
                            is_raw_demosaic: false,
                        });
                    }
                    return Err(AppError::ImageProcessingFailure(format!(
                        "RAW demosaic failed for '{}': {dev_err}",
                        path.display()
                    )));
                }
            }
        }
        Err(raw_err) => {
            let orientation = read_raw_orientation(decoder.as_ref(), &raw_source, &params);
            if let Some(fallback) = try_embedded_preview(decoder.as_ref(), &raw_source, &params) {
                let shrunk = resize_if_larger(fallback, max_dim);
                let upright = apply_exif_orientation(shrunk, orientation);
                return Ok(DecodedDevelopBuffer {
                    image: upright.to_rgb32f(),
                    is_linear: false,
                    is_wide_gamut: false,
                    bit_depth: 8,
                    is_raw_demosaic: false,
                });
            }
            return Err(AppError::ImageProcessingFailure(format!(
                "RAW decode failed for '{}': {raw_err}",
                path.display()
            )));
        }
    }

    Err(AppError::ImageProcessingFailure(format!(
        "RAW pipeline produced empty buffer for '{}'",
        path.display()
    )))
}

/// Fast preview decode for high-speed AI scanning (`YuNet` + `SFace` + `Slate OCR`) & UI thumbnails:
/// extracts the camera's embedded JPEG preview straight from RAW containers and applies EXIF
/// orientation so portrait shots are upright for accurate YuNet face detection.
pub fn decode_fast_preview_rgb(source_path: &str) -> Result<RgbImage, AppError> {
    decode_fast_preview_max_dim(source_path, 0)
}

/// Fast preview decode that downscales to `max_dim` (when `max_dim > 0`) *before* applying
/// EXIF rotation, dramatically reducing pixel work on large RAW previews and 24–60MP JPEGs.
pub fn decode_fast_preview_max_dim(source_path: &str, max_dim: u32) -> Result<RgbImage, AppError> {
    let path = Path::new(source_path);
    if !is_raw_photo_path(path) {
        let img = decode_standard_upright_max_dim(path, source_path, max_dim)?;
        return Ok(img.to_rgb8());
    }

    let raw_source = RawSource::new(path).map_err(|err| {
        AppError::ImageProcessingFailure(format!(
            "failed to open RAW source '{}': {err}",
            path.display()
        ))
    })?;
    let decoder = rawler::get_decoder(&raw_source).map_err(|err| {
        AppError::ImageProcessingFailure(format!(
            "no RAW decoder available for '{}': {err}",
            path.display()
        ))
    })?;
    let params = RawDecodeParams::default();
    let orientation = read_raw_orientation(decoder.as_ref(), &raw_source, &params);

    if let Ok(Some(preview)) = decoder.preview_image(&raw_source, &params) {
        let shrunk = resize_if_larger(preview, max_dim);
        return Ok(apply_exif_orientation(shrunk, orientation).to_rgb8());
    }
    if let Ok(Some(full)) = decoder.full_image(&raw_source, &params) {
        let shrunk = resize_if_larger(full, max_dim);
        return Ok(apply_exif_orientation(shrunk, orientation).to_rgb8());
    }

    // Fallback: demosaic if this RAW file contains no embedded JPEG preview
    let full_img = decode_raw_full_dynamic(path)?;
    Ok(resize_if_larger(full_img, max_dim).to_rgb8())
}

fn resize_if_larger(img: DynamicImage, max_dim: u32) -> DynamicImage {
    if max_dim == 0 {
        return img;
    }
    let (w, h) = (img.width(), img.height());
    if w.max(h) <= max_dim {
        return img;
    }
    let scale = (max_dim as f32) / (w.max(h) as f32);
    let nw = ((w as f32) * scale).round().max(1.0) as u32;
    let nh = ((h as f32) * scale).round().max(1.0) as u32;
    img.resize_exact(nw, nh, FilterType::Triangle)
}

fn decode_standard_upright(path: &Path, source_path: &str) -> Result<DynamicImage, AppError> {
    decode_standard_upright_max_dim(path, source_path, 0)
}

fn decode_standard_upright_max_dim(
    path: &Path,
    source_path: &str,
    max_dim: u32,
) -> Result<DynamicImage, AppError> {
    let reader = ImageReader::open(path)
        .map_err(|err| {
            AppError::ImageProcessingFailure(format!("failed to open '{source_path}': {err}"))
        })?
        .with_guessed_format()
        .map_err(|err| {
            AppError::ImageProcessingFailure(format!(
                "failed to detect image format for '{source_path}': {err}"
            ))
        })?;

    let mut decoder = reader.into_decoder().map_err(|err| {
        AppError::ImageProcessingFailure(format!("failed to decode '{source_path}': {err}"))
    })?;
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let img = DynamicImage::from_decoder(decoder).map_err(|err| {
        AppError::ImageProcessingFailure(format!("failed to read pixels of '{source_path}': {err}"))
    })?;
    let mut shrunk = resize_if_larger(img, max_dim);
    shrunk.apply_orientation(orientation);
    Ok(shrunk)
}

fn read_raw_orientation(
    decoder: &dyn rawler::decoders::Decoder,
    raw_source: &RawSource,
    params: &RawDecodeParams,
) -> Orientation {
    decoder
        .raw_metadata(raw_source, params)
        .ok()
        .and_then(|meta| meta.exif.orientation)
        .map(Orientation::from_u16)
        .unwrap_or(Orientation::Normal)
}

fn decode_raw_full_dynamic(path: &Path) -> Result<DynamicImage, AppError> {
    let raw_source = RawSource::new(path).map_err(|err| {
        AppError::ImageProcessingFailure(format!(
            "failed to open RAW source '{}': {err}",
            path.display()
        ))
    })?;
    let decoder = rawler::get_decoder(&raw_source).map_err(|err| {
        AppError::ImageProcessingFailure(format!(
            "unsupported RAW format '{}': {err}",
            path.display()
        ))
    })?;
    let params = RawDecodeParams::default();

    match decoder.raw_image(&raw_source, &params, false) {
        Ok(raw_image) => {
            let orientation = raw_image.orientation;
            let developer = RawDevelop::default();
            match developer.develop_intermediate(&raw_image) {
                Ok(intermediate) => {
                    if let Some(img) = intermediate.to_dynamic_image() {
                        return Ok(apply_exif_orientation(img, orientation));
                    }
                }
                Err(dev_err) => {
                    if let Some(fallback) =
                        try_embedded_preview(decoder.as_ref(), &raw_source, &params)
                    {
                        return Ok(apply_exif_orientation(fallback, orientation));
                    }
                    return Err(AppError::ImageProcessingFailure(format!(
                        "RAW demosaic failed for '{}': {dev_err}",
                        path.display()
                    )));
                }
            }
        }
        Err(raw_err) => {
            let orientation = read_raw_orientation(decoder.as_ref(), &raw_source, &params);
            if let Some(fallback) = try_embedded_preview(decoder.as_ref(), &raw_source, &params) {
                return Ok(apply_exif_orientation(fallback, orientation));
            }
            return Err(AppError::ImageProcessingFailure(format!(
                "RAW decode failed for '{}': {raw_err}",
                path.display()
            )));
        }
    }

    Err(AppError::ImageProcessingFailure(format!(
        "RAW pipeline produced empty buffer for '{}'",
        path.display()
    )))
}

fn try_embedded_preview(
    decoder: &dyn rawler::decoders::Decoder,
    raw_source: &RawSource,
    params: &RawDecodeParams,
) -> Option<DynamicImage> {
    if let Ok(Some(full)) = decoder.full_image(raw_source, params) {
        return Some(full);
    }
    if let Ok(Some(preview)) = decoder.preview_image(raw_source, params) {
        return Some(preview);
    }
    None
}

/// Applies camera EXIF orientation so vertical portrait RAW shots render upright.
fn apply_exif_orientation(img: DynamicImage, orientation: Orientation) -> DynamicImage {
    match orientation {
        Orientation::Normal | Orientation::Unknown => img,
        Orientation::HorizontalFlip => img.fliph(),
        Orientation::VerticalFlip => img.flipv(),
        Orientation::Rotate180 => img.rotate180(),
        Orientation::Rotate90 => img.rotate90(),
        Orientation::Rotate270 => img.rotate270(),
        Orientation::Transpose => img.fliph().rotate270(),
        Orientation::Transverse => img.fliph().rotate90(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_major_camera_raw_and_standard_formats() {
        for raw_file in [
            "DSC_0401.ARW",
            "IMG_1022.CR3",
            "IMG_1023.cr2",
            "DSC_9900.NEF",
            "L1002912.DNG",
            "DSCF4410.RAF",
            "P1010012.RW2",
            "P4120001.ORF",
        ] {
            let p = Path::new(raw_file);
            assert!(is_raw_photo_path(p), "expected RAW: {raw_file}");
            assert!(is_supported_photo_path(p));
        }

        for std_file in ["photo.jpg", "portrait.PNG", "scan.tiff", "web.webp"] {
            let p = Path::new(std_file);
            assert!(!is_raw_photo_path(p));
            assert!(is_standard_photo_path(p));
            assert!(is_supported_photo_path(p));
        }

        assert!(!is_supported_photo_path(Path::new("notes.txt")));
        assert!(!is_supported_photo_path(Path::new("sidecar.kroma.json")));
    }

    #[test]
    fn exif_orientation_rotate90_stands_landscape_portrait_upright() {
        // Landscape sensor 4x2 rotated 90 CW becomes upright portrait 2x4
        let img = DynamicImage::new_rgb8(4, 2);
        let upright = apply_exif_orientation(img, Orientation::Rotate90);
        assert_eq!(upright.width(), 2);
        assert_eq!(upright.height(), 4);
    }

    #[test]
    fn f32_to_f16_bits_encodes_hdr_and_standard_values_accurately() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(0.5), 0x3800);
        assert_eq!(f32_to_f16_bits(1.0), 0x3C00);
        assert_eq!(f32_to_f16_bits(1.5), 0x3E00);
        assert_eq!(f32_to_f16_bits(2.0), 0x4000);
        assert_eq!(f32_to_f16_bits(-1.0), 0xBC00);
        assert_eq!(f32_to_f16_bits(f32::NAN), 0x0000);
        assert_eq!(f32_to_f16_bits(100_000.0), 0x7BFF);
    }

    #[test]
    fn downscale_and_kr16_encoder_preserve_hdr_linear_highlights_above_one() {
        let mut img = Rgb32FImage::new(4, 4);
        for px in img.pixels_mut() {
            px.0 = [2.0, 1.5, 0.5];
        }
        let down = downscale_rgb32f_unclamped(&img, 2);
        assert_eq!(down.dimensions(), (2, 2));
        let sample = down.get_pixel(0, 0).0;
        assert!((sample[0] - 2.0).abs() < 1e-4, "HDR >1.0 highlight must not clamp");
        assert!((sample[1] - 1.5).abs() < 1e-4);
        assert!((sample[2] - 0.5).abs() < 1e-4);

        let buf = DecodedDevelopBuffer {
            image: down,
            is_linear: true,
            is_wide_gamut: true,
            bit_depth: 32,
            is_raw_demosaic: true,
        };
        let encoded = encode_kr16_rgba16f(&buf);
        assert_eq!(&encoded[0..4], b"KR16");
        assert_eq!(u32::from_le_bytes(encoded[4..8].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(encoded[8..12].try_into().unwrap()), 2);
        assert_eq!(
            encoded[12],
            KR16_FLAG_LINEAR | KR16_FLAG_RAW_DEMOSAIC | KR_FLAG_WIDE_GAMUT_REC2020
        );
        assert_eq!(encoded[13], 32);
        assert_eq!(encoded.len(), 16 + 2 * 2 * 8);
        // First pixel RGBA16F: R=2.0 (0x4000), G=1.5 (0x3E00), B=0.5 (0x3800), A=1.0 (0x3C00)
        assert_eq!(u16::from_le_bytes(encoded[16..18].try_into().unwrap()), 0x4000);
        assert_eq!(u16::from_le_bytes(encoded[18..20].try_into().unwrap()), 0x3E00);
        assert_eq!(u16::from_le_bytes(encoded[20..22].try_into().unwrap()), 0x3800);
        assert_eq!(u16::from_le_bytes(encoded[22..24].try_into().unwrap()), 0x3C00);

        let encoded32 = encode_kr32_rgba32f(&buf);
        assert_eq!(&encoded32[0..4], b"KR32");
        assert_eq!(u32::from_le_bytes(encoded32[4..8].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(encoded32[8..12].try_into().unwrap()), 2);
        assert_eq!(
            encoded32[12],
            KR16_FLAG_LINEAR | KR16_FLAG_RAW_DEMOSAIC | KR_FLAG_WIDE_GAMUT_REC2020
        );
        assert_eq!(encoded32[13], 32);
        assert_eq!(encoded32.len(), 16 + 2 * 2 * 16);
        assert_eq!(f32::from_le_bytes(encoded32[16..20].try_into().unwrap()), 2.0);
        assert_eq!(f32::from_le_bytes(encoded32[20..24].try_into().unwrap()), 1.5);
        assert_eq!(f32::from_le_bytes(encoded32[24..28].try_into().unwrap()), 0.5);
        assert_eq!(f32::from_le_bytes(encoded32[28..32].try_into().unwrap()), 1.0);
    }
}
