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

/// Reads a photo's upright native pixel dimensions `(width, height)` from container/EXIF headers
/// without decoding or demosaicing full pixel payloads (~0.5ms, ~0 MB RAM).
pub fn read_native_photo_dimensions(source_path: &str) -> Result<(u32, u32), AppError> {
    let path = Path::new(source_path);
    if !is_raw_photo_path(path) {
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
        let (w, h) = decoder.dimensions();
        if w == 0 || h == 0 {
            return Err(AppError::ImageProcessingFailure(format!(
                "zero image dimensions for '{source_path}'"
            )));
        }
        let orientation = decoder
            .orientation()
            .unwrap_or(image::metadata::Orientation::NoTransforms);
        return Ok(oriented_standard_dims(w, h, orientation));
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

    // `dummy = true` parses RAW header/crop geometry without decoding or demosaicing Bayer pixels.
    if let Ok(raw_image) = decoder.raw_image(&raw_source, &params, true) {
        let orientation = resolve_raw_orientation(
            raw_image.orientation,
            decoder.as_ref(),
            &raw_source,
            &params,
        );
        let (w, h) = raw_image
            .crop_area
            .or(raw_image.active_area)
            .map(|rect| (rect.d.w as u32, rect.d.h as u32))
            .filter(|&(w, h)| w > 0 && h > 0)
            .unwrap_or((raw_image.width as u32, raw_image.height as u32));
        if w > 0 && h > 0 {
            return Ok(oriented_raw_dims(w, h, orientation));
        }
    }

    let orientation = read_raw_orientation(decoder.as_ref(), &raw_source, &params);
    if let Some(preview) = try_embedded_preview(decoder.as_ref(), &raw_source, &params) {
        let (w, h) = (preview.width(), preview.height());
        if w > 0 && h > 0 {
            return Ok(oriented_raw_dims(w, h, orientation));
        }
    }

    Err(AppError::ImageProcessingFailure(format!(
        "could not read native RAW dimensions for '{}'",
        path.display()
    )))
}

#[inline]
fn oriented_standard_dims(
    w: u32,
    h: u32,
    orientation: image::metadata::Orientation,
) -> (u32, u32) {
    match orientation {
        image::metadata::Orientation::Rotate90
        | image::metadata::Orientation::Rotate270
        | image::metadata::Orientation::Rotate90FlipH
        | image::metadata::Orientation::Rotate270FlipH => (h, w),
        _ => (w, h),
    }
}

#[inline]
fn oriented_raw_dims(w: u32, h: u32, orientation: Orientation) -> (u32, u32) {
    match orientation {
        Orientation::Rotate90
        | Orientation::Rotate270
        | Orientation::Transpose
        | Orientation::Transverse => (h, w),
        _ => (w, h),
    }
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

/// Converts an IEEE 754 binary16 (half-float) bit pattern (`u16`) back to `f32`.
#[inline]
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = if (h & 0x8000) != 0 { -1.0_f32 } else { 1.0_f32 };
    let exp = ((h >> 10) & 0x1F) as i32;
    let frac = (h & 0x03FF) as f32;
    if exp == 0 {
        sign * (2.0_f32).powi(-14) * (frac / 1024.0)
    } else if exp == 31 {
        if frac == 0.0 {
            sign * 65504.0
        } else {
            0.0
        }
    } else {
        sign * (2.0_f32).powi(exp - 15) * (1.0 + frac / 1024.0)
    }
}

/// Fast parallel Rayon box-downscaler directly from a cached `KR16` binary buffer into a compact
/// `DecodedDevelopBuffer` (e.g. `384px` live histogram proxy or `640px` 32-bit thumbnail proxy)
/// in ~2–4 ms without re-decoding the original camera RAW file.
pub fn decode_kr16_downscaled_rgb32f(kr16_bytes: &[u8], max_dim: u32) -> Option<DecodedDevelopBuffer> {
    if kr16_bytes.len() < 16 || &kr16_bytes[0..4] != KR16_MAGIC {
        return None;
    }
    let sw = u32::from_le_bytes(kr16_bytes[4..8].try_into().ok()?);
    let sh = u32::from_le_bytes(kr16_bytes[8..12].try_into().ok()?);
    if sw == 0 || sh == 0 {
        return None;
    }
    let flags = kr16_bytes[12];
    let bit_depth = kr16_bytes[13];
    let expected_len = 16usize.checked_add((sw as usize).checked_mul(sh as usize)?.checked_mul(8)?)?;
    if kr16_bytes.len() < expected_len {
        return None;
    }

    let (dw, dh) = if max_dim == 0 || sw.max(sh) <= max_dim {
        (sw, sh)
    } else {
        let scale = (max_dim as f32) / (sw.max(sh) as f32);
        (
            ((sw as f32) * scale).round().max(1.0) as u32,
            ((sh as f32) * scale).round().max(1.0) as u32,
        )
    };

    let sw_u = sw as usize;
    let sh_u = sh as usize;
    let dw_u = dw as usize;
    let dh_u = dh as usize;
    let payload = &kr16_bytes[16..expected_len];
    let rx = (sw as f32) / (dw as f32);
    let ry = (sh as f32) / (dh as f32);

    let mut out = vec![0.0_f32; dw_u * dh_u * 3];
    out.par_chunks_exact_mut(dw_u * 3)
        .enumerate()
        .for_each(|(dy, row)| {
            let sy0 = (((dy as f32) * ry).floor() as usize).min(sh_u - 1);
            let sy1 = ((((dy + 1) as f32) * ry).ceil() as usize).clamp(sy0 + 1, sh_u);
            for dx in 0..dw_u {
                let sx0 = (((dx as f32) * rx).floor() as usize).min(sw_u - 1);
                let sx1 = ((((dx + 1) as f32) * rx).ceil() as usize).clamp(sx0 + 1, sw_u);
                let mut sr = 0.0_f32;
                let mut sg = 0.0_f32;
                let mut sb = 0.0_f32;
                let mut count = 0_u32;
                for sy in sy0..sy1 {
                    let row_off = sy * sw_u * 8;
                    for sx in sx0..sx1 {
                        let o = row_off + sx * 8;
                        let r = u16::from_le_bytes([payload[o], payload[o + 1]]);
                        let g = u16::from_le_bytes([payload[o + 2], payload[o + 3]]);
                        let b = u16::from_le_bytes([payload[o + 4], payload[o + 5]]);
                        sr += f16_bits_to_f32(r);
                        sg += f16_bits_to_f32(g);
                        sb += f16_bits_to_f32(b);
                        count += 1;
                    }
                }
                let inv = 1.0 / (count.max(1) as f32);
                let dst = dx * 3;
                row[dst] = sr * inv;
                row[dst + 1] = sg * inv;
                row[dst + 2] = sb * inv;
            }
        });

    let image = Rgb32FImage::from_raw(dw, dh, out)?;
    Some(DecodedDevelopBuffer {
        image,
        is_linear: (flags & KR16_FLAG_LINEAR) != 0,
        is_wide_gamut: (flags & KR_FLAG_WIDE_GAMUT_REC2020) != 0,
        bit_depth,
        is_raw_demosaic: (flags & KR16_FLAG_RAW_DEMOSAIC) != 0,
    })
}

/// Decodes either a `KR16` (RGBA16F) or `KR32` (RGBA32F) cached Studio working buffer into a
/// `DecodedDevelopBuffer` (optionally box-downscaled to `max_dim`), preserving scene-linear
/// Rec.2020 RAW flags so features like AI Remove grade identically to Studio's 32-bit canvas
/// without re-demosaicing the original RAW file from disk.
pub fn decode_working_wire_downscaled_rgb32f(
    wire_bytes: &[u8],
    max_dim: u32,
) -> Option<DecodedDevelopBuffer> {
    if wire_bytes.len() < 16 {
        return None;
    }
    if &wire_bytes[0..4] == KR16_MAGIC {
        return decode_kr16_downscaled_rgb32f(wire_bytes, max_dim);
    }
    if &wire_bytes[0..4] != KR32_MAGIC {
        return None;
    }

    let sw = u32::from_le_bytes(wire_bytes[4..8].try_into().ok()?);
    let sh = u32::from_le_bytes(wire_bytes[8..12].try_into().ok()?);
    if sw == 0 || sh == 0 {
        return None;
    }
    let flags = wire_bytes[12];
    let bit_depth = wire_bytes[13];
    let expected_len = 16usize.checked_add((sw as usize).checked_mul(sh as usize)?.checked_mul(16)?)?;
    if wire_bytes.len() < expected_len {
        return None;
    }

    let (dw, dh) = if max_dim == 0 || sw.max(sh) <= max_dim {
        (sw, sh)
    } else {
        let scale = (max_dim as f32) / (sw.max(sh) as f32);
        (
            ((sw as f32) * scale).round().max(1.0) as u32,
            ((sh as f32) * scale).round().max(1.0) as u32,
        )
    };

    let sw_u = sw as usize;
    let sh_u = sh as usize;
    let dw_u = dw as usize;
    let dh_u = dh as usize;
    let payload = &wire_bytes[16..expected_len];
    let rx = (sw as f32) / (dw as f32);
    let ry = (sh as f32) / (dh as f32);

    let mut out = vec![0.0_f32; dw_u * dh_u * 3];
    if sw_u == dw_u && sh_u == dh_u {
        out.par_chunks_exact_mut(dw_u * 3)
            .enumerate()
            .for_each(|(y, row)| {
                let row_off = y * sw_u * 16;
                for x in 0..dw_u {
                    let o = row_off + x * 16;
                    let dst = x * 3;
                    row[dst] = f32::from_le_bytes([
                        payload[o],
                        payload[o + 1],
                        payload[o + 2],
                        payload[o + 3],
                    ]);
                    row[dst + 1] = f32::from_le_bytes([
                        payload[o + 4],
                        payload[o + 5],
                        payload[o + 6],
                        payload[o + 7],
                    ]);
                    row[dst + 2] = f32::from_le_bytes([
                        payload[o + 8],
                        payload[o + 9],
                        payload[o + 10],
                        payload[o + 11],
                    ]);
                }
            });
    } else {
        out.par_chunks_exact_mut(dw_u * 3)
            .enumerate()
            .for_each(|(dy, row)| {
                let sy0 = (((dy as f32) * ry).floor() as usize).min(sh_u - 1);
                let sy1 = ((((dy + 1) as f32) * ry).ceil() as usize).clamp(sy0 + 1, sh_u);
                for dx in 0..dw_u {
                    let sx0 = (((dx as f32) * rx).floor() as usize).min(sw_u - 1);
                    let sx1 = ((((dx + 1) as f32) * rx).ceil() as usize).clamp(sx0 + 1, sw_u);
                    let mut sr = 0.0_f32;
                    let mut sg = 0.0_f32;
                    let mut sb = 0.0_f32;
                    let mut count = 0_u32;
                    for sy in sy0..sy1 {
                        let row_off = sy * sw_u * 16;
                        for sx in sx0..sx1 {
                            let o = row_off + sx * 16;
                            sr += f32::from_le_bytes([
                                payload[o],
                                payload[o + 1],
                                payload[o + 2],
                                payload[o + 3],
                            ]);
                            sg += f32::from_le_bytes([
                                payload[o + 4],
                                payload[o + 5],
                                payload[o + 6],
                                payload[o + 7],
                            ]);
                            sb += f32::from_le_bytes([
                                payload[o + 8],
                                payload[o + 9],
                                payload[o + 10],
                                payload[o + 11],
                            ]);
                            count += 1;
                        }
                    }
                    let inv = 1.0 / (count.max(1) as f32);
                    let dst = dx * 3;
                    row[dst] = sr * inv;
                    row[dst + 1] = sg * inv;
                    row[dst + 2] = sb * inv;
                }
            });
    }

    let image = Rgb32FImage::from_raw(dw, dh, out)?;
    Some(DecodedDevelopBuffer {
        image,
        is_linear: (flags & KR16_FLAG_LINEAR) != 0,
        is_wide_gamut: (flags & KR_FLAG_WIDE_GAMUT_REC2020) != 0,
        bit_depth,
        is_raw_demosaic: (flags & KR16_FLAG_RAW_DEMOSAIC) != 0,
    })
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

/// Fused parallel downscaler + wide-gamut **Linear Rec.2020** converter for `rawler`'s
/// calibrated linear `Intermediate` buffer.
/// Because box-averaging and the 3×3 Rec.709→Rec.2020 matrix are both linear operators,
/// downscaling *before* the matrix multiply avoids allocating a 288–720 MB full-res RGB buffer
/// and cuts per-pixel Rec.2020 matrix work by ~10× while running in parallel across all CPU cores.
fn intermediate_to_rgb32f_rec2020_max_dim(
    intermediate: Intermediate,
    max_dim: u32,
) -> Option<Rgb32FImage> {
    match intermediate {
        Intermediate::Monochrome(pixels) => {
            let sw = pixels.width as u32;
            let sh = pixels.height as u32;
            if sw == 0 || sh == 0 {
                return None;
            }
            let (dw, dh) = target_scaled_dims(sw, sh, max_dim);
            let sw_u = sw as usize;
            let sh_u = sh as usize;
            let dw_u = dw as usize;
            let dh_u = dh as usize;
            let src = &pixels.data;
            let mut out = vec![0.0_f32; dw_u * dh_u * 3];

            if dw == sw && dh == sh {
                out.par_chunks_exact_mut(dw_u * 3)
                    .enumerate()
                    .for_each(|(y, row)| {
                        let src_row = &src[y * sw_u..(y + 1) * sw_u];
                        for (x, &v) in src_row.iter().enumerate() {
                            let c = if v.is_finite() { v.max(0.0) } else { 0.0 };
                            let dst = x * 3;
                            row[dst] = c;
                            row[dst + 1] = c;
                            row[dst + 2] = c;
                        }
                    });
            } else {
                let rx = (sw as f32) / (dw as f32);
                let ry = (sh as f32) / (dh as f32);
                out.par_chunks_exact_mut(dw_u * 3)
                    .enumerate()
                    .for_each(|(dy, row)| {
                        let sy0 = (((dy as f32) * ry).floor() as usize).min(sh_u - 1);
                        let sy1 = ((((dy + 1) as f32) * ry).ceil() as usize).clamp(sy0 + 1, sh_u);
                        for dx in 0..dw_u {
                            let sx0 = (((dx as f32) * rx).floor() as usize).min(sw_u - 1);
                            let sx1 = ((((dx + 1) as f32) * rx).ceil() as usize).clamp(sx0 + 1, sw_u);
                            let mut sum = 0.0_f32;
                            let mut count = 0_u32;
                            for sy in sy0..sy1 {
                                let row_off = sy * sw_u;
                                for sx in sx0..sx1 {
                                    let v = src[row_off + sx];
                                    if v.is_finite() {
                                        sum += v;
                                    }
                                    count += 1;
                                }
                            }
                            let c = (sum / (count.max(1) as f32)).max(0.0);
                            let dst = dx * 3;
                            row[dst] = c;
                            row[dst + 1] = c;
                            row[dst + 2] = c;
                        }
                    });
            }
            Rgb32FImage::from_raw(dw, dh, out)
        }
        Intermediate::ThreeColor(pixels) => {
            downscale_and_convert_slice_to_rec2020(
                &pixels.data,
                pixels.width as u32,
                pixels.height as u32,
                max_dim,
                |px| [px[0], px[1], px[2]],
            )
        }
        Intermediate::FourColor(pixels) => {
            downscale_and_convert_slice_to_rec2020(
                &pixels.data,
                pixels.width as u32,
                pixels.height as u32,
                max_dim,
                |px| [px[0], px[1], px[2]],
            )
        }
    }
}

#[inline]
fn target_scaled_dims(sw: u32, sh: u32, max_dim: u32) -> (u32, u32) {
    if max_dim == 0 || sw.max(sh) <= max_dim {
        return (sw, sh);
    }
    let scale = (max_dim as f32) / (sw.max(sh) as f32);
    (
        ((sw as f32) * scale).round().max(1.0) as u32,
        ((sh as f32) * scale).round().max(1.0) as u32,
    )
}

fn downscale_and_convert_slice_to_rec2020<T: Sync, F: Fn(&T) -> [f32; 3] + Sync>(
    src: &[T],
    sw: u32,
    sh: u32,
    max_dim: u32,
    extract_rgb: F,
) -> Option<Rgb32FImage> {
    if sw == 0 || sh == 0 || src.len() < (sw as usize) * (sh as usize) {
        return None;
    }
    let (dw, dh) = target_scaled_dims(sw, sh, max_dim);
    let sw_u = sw as usize;
    let sh_u = sh as usize;
    let dw_u = dw as usize;
    let dh_u = dh as usize;
    let mut out = vec![0.0_f32; dw_u * dh_u * 3];

    if dw == sw && dh == sh {
        out.par_chunks_exact_mut(dw_u * 3)
            .enumerate()
            .for_each(|(y, row)| {
                let src_row = &src[y * sw_u..(y + 1) * sw_u];
                for (x, px) in src_row.iter().enumerate() {
                    let [r, g, b] = extract_rgb(px);
                    let r709 = if r.is_finite() { r } else { 0.0 };
                    let g709 = if g.is_finite() { g } else { 0.0 };
                    let b709 = if b.is_finite() { b } else { 0.0 };
                    let rec2020 = rec709_to_rec2020([r709, g709, b709]);
                    let dst = x * 3;
                    row[dst] = rec2020[0].max(0.0);
                    row[dst + 1] = rec2020[1].max(0.0);
                    row[dst + 2] = rec2020[2].max(0.0);
                }
            });
    } else {
        let rx = (sw as f32) / (dw as f32);
        let ry = (sh as f32) / (dh as f32);
        out.par_chunks_exact_mut(dw_u * 3)
            .enumerate()
            .for_each(|(dy, row)| {
                let sy0 = (((dy as f32) * ry).floor() as usize).min(sh_u - 1);
                let sy1 = ((((dy + 1) as f32) * ry).ceil() as usize).clamp(sy0 + 1, sh_u);
                for dx in 0..dw_u {
                    let sx0 = (((dx as f32) * rx).floor() as usize).min(sw_u - 1);
                    let sx1 = ((((dx + 1) as f32) * rx).ceil() as usize).clamp(sx0 + 1, sw_u);
                    let mut sr = 0.0_f32;
                    let mut sg = 0.0_f32;
                    let mut sb = 0.0_f32;
                    let mut count = 0_u32;
                    for sy in sy0..sy1 {
                        let row_off = sy * sw_u;
                        for sx in sx0..sx1 {
                            let [r, g, b] = extract_rgb(&src[row_off + sx]);
                            if r.is_finite() {
                                sr += r;
                            }
                            if g.is_finite() {
                                sg += g;
                            }
                            if b.is_finite() {
                                sb += b;
                            }
                            count += 1;
                        }
                    }
                    let inv = 1.0 / (count.max(1) as f32);
                    let rec2020 = rec709_to_rec2020([sr * inv, sg * inv, sb * inv]);
                    let dst = dx * 3;
                    row[dst] = rec2020[0].max(0.0);
                    row[dst + 1] = rec2020[1].max(0.0);
                    row[dst + 2] = rec2020[2].max(0.0);
                }
            });
    }

    Rgb32FImage::from_raw(dw, dh, out)
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
            let orientation = resolve_raw_orientation(
                raw_image.orientation,
                decoder.as_ref(),
                &raw_source,
                &params,
            );
            match linear_developer.develop_intermediate(&raw_image) {
                Ok(intermediate) => {
                    if let Some(scaled) =
                        intermediate_to_rgb32f_rec2020_max_dim(intermediate, max_dim)
                    {
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
    match img {
        DynamicImage::ImageRgb8(rgb) => {
            DynamicImage::ImageRgb8(downscale_rgb8_area_parallel(&rgb, max_dim))
        }
        DynamicImage::ImageRgba8(rgba) => {
            DynamicImage::ImageRgba8(downscale_rgba8_area_parallel(&rgba, max_dim))
        }
        other => {
            let (nw, nh) = target_scaled_dims(w, h, max_dim);
            other.resize_exact(nw, nh, FilterType::Triangle)
        }
    }
}

/// Fast multi-threaded integer area-averaging downscaler for 8-bit RGB camera JPEGs and embedded
/// RAW previews. Eliminates aliasing/moiré on 6:1 camera-to-1024px reductions and scales across
/// all CPU cores via Rayon without intermediate horizontal/vertical pass allocations.
pub fn downscale_rgb8_area_parallel(src: &RgbImage, max_dim: u32) -> RgbImage {
    let (sw, sh) = src.dimensions();
    if max_dim == 0 || sw.max(sh) <= max_dim || sw == 0 || sh == 0 {
        return src.clone();
    }
    let (dw, dh) = target_scaled_dims(sw, sh, max_dim);
    let sw_u = sw as usize;
    let sh_u = sh as usize;
    let dw_u = dw as usize;
    let dh_u = dh as usize;
    let raw = src.as_raw();
    let x_bounds: Vec<(usize, usize)> = (0..dw_u)
        .map(|dx| {
            let x0 = (dx * sw_u) / dw_u;
            let x1 = (((dx + 1) * sw_u).div_ceil(dw_u)).clamp(x0 + 1, sw_u);
            (x0, x1)
        })
        .collect();

    let mut out = vec![0_u8; dw_u * dh_u * 3];
    out.par_chunks_exact_mut(dw_u * 3)
        .enumerate()
        .for_each(|(dy, row)| {
            let y0 = (dy * sh_u) / dh_u;
            let y1 = (((dy + 1) * sh_u).div_ceil(dh_u)).clamp(y0 + 1, sh_u);
            let y_count = (y1 - y0) as u32;
            for (dx, &(x0, x1)) in x_bounds.iter().enumerate() {
                let mut sr = 0_u32;
                let mut sg = 0_u32;
                let mut sb = 0_u32;
                for sy in y0..y1 {
                    let row_off = sy * sw_u * 3;
                    for sx in x0..x1 {
                        let idx = row_off + sx * 3;
                        sr += raw[idx] as u32;
                        sg += raw[idx + 1] as u32;
                        sb += raw[idx + 2] as u32;
                    }
                }
                let count = y_count * (x1 - x0) as u32;
                let half = count >> 1;
                let dst = dx * 3;
                row[dst] = ((sr + half) / count) as u8;
                row[dst + 1] = ((sg + half) / count) as u8;
                row[dst + 2] = ((sb + half) / count) as u8;
            }
        });

    RgbImage::from_raw(dw, dh, out).expect("downscaled rgb8 dimensions match buffer")
}

fn downscale_rgba8_area_parallel(src: &image::RgbaImage, max_dim: u32) -> image::RgbaImage {
    let (sw, sh) = src.dimensions();
    if max_dim == 0 || sw.max(sh) <= max_dim || sw == 0 || sh == 0 {
        return src.clone();
    }
    let (dw, dh) = target_scaled_dims(sw, sh, max_dim);
    let sw_u = sw as usize;
    let sh_u = sh as usize;
    let dw_u = dw as usize;
    let dh_u = dh as usize;
    let raw = src.as_raw();
    let x_bounds: Vec<(usize, usize)> = (0..dw_u)
        .map(|dx| {
            let x0 = (dx * sw_u) / dw_u;
            let x1 = (((dx + 1) * sw_u).div_ceil(dw_u)).clamp(x0 + 1, sw_u);
            (x0, x1)
        })
        .collect();

    let mut out = vec![0_u8; dw_u * dh_u * 4];
    out.par_chunks_exact_mut(dw_u * 4)
        .enumerate()
        .for_each(|(dy, row)| {
            let y0 = (dy * sh_u) / dh_u;
            let y1 = (((dy + 1) * sh_u).div_ceil(dh_u)).clamp(y0 + 1, sh_u);
            let y_count = (y1 - y0) as u32;
            for (dx, &(x0, x1)) in x_bounds.iter().enumerate() {
                let mut sr = 0_u32;
                let mut sg = 0_u32;
                let mut sb = 0_u32;
                let mut sa = 0_u32;
                for sy in y0..y1 {
                    let row_off = sy * sw_u * 4;
                    for sx in x0..x1 {
                        let idx = row_off + sx * 4;
                        sr += raw[idx] as u32;
                        sg += raw[idx + 1] as u32;
                        sb += raw[idx + 2] as u32;
                        sa += raw[idx + 3] as u32;
                    }
                }
                let count = y_count * (x1 - x0) as u32;
                let half = count >> 1;
                let dst = dx * 4;
                row[dst] = ((sr + half) / count) as u8;
                row[dst + 1] = ((sg + half) / count) as u8;
                row[dst + 2] = ((sb + half) / count) as u8;
                row[dst + 3] = ((sa + half) / count) as u8;
            }
        });

    image::RgbaImage::from_raw(dw, dh, out).expect("downscaled rgba8 dimensions match buffer")
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

/// Resolves RAW camera orientation reliably: `rawler` currently leaves `RawImage::orientation`
/// as `Orientation::Normal` (`// TODO fixme` upstream), so whenever it reports `Normal`/`Unknown`
/// we query the RAW container's EXIF IFD via `raw_metadata`.
fn resolve_raw_orientation(
    raw_orientation: Orientation,
    decoder: &dyn rawler::decoders::Decoder,
    raw_source: &RawSource,
    params: &RawDecodeParams,
) -> Orientation {
    prefer_explicit_orientation(raw_orientation, || {
        read_raw_orientation(decoder, raw_source, params)
    })
}

fn prefer_explicit_orientation<F>(raw_orientation: Orientation, read_exif: F) -> Orientation
where
    F: FnOnce() -> Orientation,
{
    match raw_orientation {
        Orientation::Normal | Orientation::Unknown => read_exif(),
        explicit => explicit,
    }
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
            let orientation = resolve_raw_orientation(
                raw_image.orientation,
                decoder.as_ref(),
                &raw_source,
                &params,
            );
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

        // 32-bit scene-linear HDR float buffer must also rotate cleanly without clamping >1.0 highlights
        let mut hdr = Rgb32FImage::new(4, 2);
        hdr.put_pixel(0, 0, image::Rgb([2.5, 1.25, 0.5]));
        let upright_hdr = apply_exif_orientation(DynamicImage::ImageRgb32F(hdr), Orientation::Rotate90)
            .into_rgb32f();
        assert_eq!(upright_hdr.dimensions(), (2, 4));
        assert!((upright_hdr.get_pixel(1, 0).0[0] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn prefer_explicit_orientation_falls_back_to_exif_when_raw_image_reports_normal() {
        assert_eq!(
            prefer_explicit_orientation(Orientation::Normal, || Orientation::Rotate90),
            Orientation::Rotate90
        );
        assert_eq!(
            prefer_explicit_orientation(Orientation::Unknown, || Orientation::Rotate270),
            Orientation::Rotate270
        );
        assert_eq!(
            prefer_explicit_orientation(Orientation::Rotate180, || Orientation::Rotate90),
            Orientation::Rotate180
        );
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

        // Verify direct KR16 downscaled round-trip preserves flags, HDR >1.0 highlights, and dimensions
        let decoded_small =
            decode_kr16_downscaled_rgb32f(&encoded, 1).expect("kr16 downscale decodes cleanly");
        assert_eq!(decoded_small.image.dimensions(), (1, 1));
        assert!(decoded_small.is_linear);
        assert!(decoded_small.is_wide_gamut);
        assert!(decoded_small.is_raw_demosaic);
        let px = decoded_small.image.get_pixel(0, 0).0;
        assert!((px[0] - 2.0).abs() < 1e-3);
        assert!((px[1] - 1.5).abs() < 1e-3);
        assert!((px[2] - 0.5).abs() < 1e-3);
    }

    #[test]
    fn fused_rec2020_downscale_preserves_hdr_and_matches_full_resolution_conversion() {
        let src = vec![[2.0_f32, 1.0_f32, 0.5_f32]; 16];
        let full = downscale_and_convert_slice_to_rec2020(&src, 4, 4, 0, |p| *p).unwrap();
        let half = downscale_and_convert_slice_to_rec2020(&src, 4, 4, 2, |p| *p).unwrap();
        assert_eq!(full.dimensions(), (4, 4));
        assert_eq!(half.dimensions(), (2, 2));
        let p_full = full.get_pixel(0, 0).0;
        let p_half = half.get_pixel(0, 0).0;
        assert!(p_half[0] > 1.0, "HDR linear highlight >1.0 must remain unclamped");
        for c in 0..3 {
            assert!((p_full[c] - p_half[c]).abs() < 1e-5);
        }
    }

    #[test]
    fn oriented_dims_swaps_for_quarter_turn_orientations_and_reads_png_header() {
        assert_eq!(
            oriented_standard_dims(6000, 4000, image::metadata::Orientation::NoTransforms),
            (6000, 4000)
        );
        assert_eq!(
            oriented_standard_dims(6000, 4000, image::metadata::Orientation::Rotate90),
            (4000, 6000)
        );
        assert_eq!(
            oriented_standard_dims(6000, 4000, image::metadata::Orientation::Rotate270FlipH),
            (4000, 6000)
        );
        assert_eq!(
            oriented_raw_dims(6000, 4000, Orientation::Rotate270),
            (4000, 6000)
        );
        assert_eq!(
            oriented_raw_dims(6000, 4000, Orientation::Rotate180),
            (6000, 4000)
        );

        let tmp = std::env::temp_dir().join(format!(
            "kroma_dims_test_{}.png",
            std::process::id()
        ));
        let img = RgbImage::new(320, 180);
        img.save(&tmp).expect("save temp png");
        let dims = read_native_photo_dimensions(tmp.to_str().unwrap()).expect("read dims");
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(dims, (320, 180));
    }

    #[test]
    fn downscale_rgb8_area_parallel_averages_blocks_and_preserves_uniform_color() {
        // Uniform 8x4 image downscaled to max_dim=4 -> exact 4x2 with identical RGB
        let mut uniform = RgbImage::new(8, 4);
        for px in uniform.pixels_mut() {
            px.0 = [210, 130, 45];
        }
        let down_uniform = downscale_rgb8_area_parallel(&uniform, 4);
        assert_eq!(down_uniform.dimensions(), (4, 2));
        for px in down_uniform.pixels() {
            assert_eq!(px.0, [210, 130, 45]);
        }

        // 4x4 checkerboard of 0 and 200 downscaled 2x -> 2x2 with exact average 100
        let mut checker = RgbImage::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                let v = if (x + y) % 2 == 0 { 0 } else { 200 };
                checker.put_pixel(x, y, image::Rgb([v, v, v]));
            }
        }
        let down_checker = downscale_rgb8_area_parallel(&checker, 2);
        assert_eq!(down_checker.dimensions(), (2, 2));
        for px in down_checker.pixels() {
            assert_eq!(px.0, [100, 100, 100]);
        }
    }

    #[test]
    fn working_wire_decoder_roundtrips_kr32_and_kr16_preserving_linear_hdr_flags() {
        let mut hdr = Rgb32FImage::new(8, 4);
        for px in hdr.pixels_mut() {
            px.0 = [1.85, 0.42, 0.15];
        }
        let buf = DecodedDevelopBuffer {
            image: hdr,
            is_linear: true,
            is_wide_gamut: true,
            bit_depth: 32,
            is_raw_demosaic: true,
        };

        let kr32 = encode_kr32_rgba32f(&buf);
        let decoded_full =
            decode_working_wire_downscaled_rgb32f(&kr32, 1600).expect("kr32 1:1 decode");
        assert_eq!(decoded_full.image.dimensions(), (8, 4));
        assert!(decoded_full.is_linear);
        assert!(decoded_full.is_wide_gamut);
        assert!(decoded_full.is_raw_demosaic);
        let p = decoded_full.image.get_pixel(3, 2).0;
        assert!((p[0] - 1.85).abs() < 1e-5);
        assert!((p[1] - 0.42).abs() < 1e-5);
        assert!((p[2] - 0.15).abs() < 1e-5);

        let decoded_down =
            decode_working_wire_downscaled_rgb32f(&kr32, 4).expect("kr32 downscaled decode");
        assert_eq!(decoded_down.image.dimensions(), (4, 2));
        assert!((decoded_down.image.get_pixel(1, 1).0[0] - 1.85).abs() < 1e-4);

        let kr16 = encode_kr16_rgba16f(&buf);
        let decoded_kr16 =
            decode_working_wire_downscaled_rgb32f(&kr16, 1600).expect("kr16 wire decode");
        assert_eq!(decoded_kr16.image.dimensions(), (8, 4));
        assert!(decoded_kr16.is_linear);
        assert!(decoded_kr16.is_wide_gamut);
        assert!((decoded_kr16.image.get_pixel(0, 0).0[0] - 1.85).abs() < 0.01);
    }
}
