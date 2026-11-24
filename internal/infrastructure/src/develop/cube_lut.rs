use kroma_app::error::AppError;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::UNIX_EPOCH;

const MAX_CACHED_CUBE_LUTS: usize = 16;

struct CachedCubeLutEntry {
    file_size: u64,
    mtime_ns: u128,
    lut: Arc<CubeLut3D>,
}

static CUBE_LUT_CACHE: OnceLock<Mutex<HashMap<String, CachedCubeLutEntry>>> = OnceLock::new();

/// Parsed standard Adobe `.cube` 3D Lookup Table with trilinear RGB interpolation.
#[derive(Debug, Clone)]
pub struct CubeLut3D {
    size: usize,
    domain_min: [f32; 3],
    domain_max: [f32; 3],
    /// Flattened RGB triplets indexed as `r + size * (g + size * b)` per Adobe `.cube` specification.
    table: Vec<[f32; 3]>,
}

impl CubeLut3D {
    /// Returns a fast deterministic content fingerprint for GPU buffer cache invalidation.
    pub fn fingerprint(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.size.hash(&mut h);
        for v in self.domain_min {
            v.to_bits().hash(&mut h);
        }
        for v in self.domain_max {
            v.to_bits().hash(&mut h);
        }
        self.table.len().hash(&mut h);
        // Sample evenly across the table so identical sizes with different grades never collide
        let step = (self.table.len() / 64).max(1);
        for rgb in self.table.iter().step_by(step) {
            rgb[0].to_bits().hash(&mut h);
            rgb[1].to_bits().hash(&mut h);
            rgb[2].to_bits().hash(&mut h);
        }
        if let Some(last) = self.table.last() {
            last[0].to_bits().hash(&mut h);
            last[1].to_bits().hash(&mut h);
            last[2].to_bits().hash(&mut h);
        }
        h.finish()
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, AppError> {
        let content = fs::read_to_string(path.as_ref()).map_err(|err| {
            AppError::ImageProcessingFailure(format!(
                "failed to read .cube LUT '{}': {err}",
                path.as_ref().display()
            ))
        })?;
        Self::parse(&content)
    }

    /// Cached loader for batch exports and repeated previews: returns an `Arc<CubeLut3D>` from a
    /// bounded process-wide cache keyed by path + `(file_size, mtime)` so exporting 500 photos with
    /// the same `.cube` LUT parses it once instead of 500 times.
    pub fn load_cached(path_or_id: &str) -> Result<Arc<Self>, AppError> {
        let p = Path::new(path_or_id);
        let (file_size, mtime_ns) = match fs::metadata(p) {
            Ok(meta) if meta.is_file() => {
                let mtime = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_nanos())
                    .unwrap_or(0);
                (meta.len(), mtime)
            }
            _ => (0, 0),
        };

        let cache = CUBE_LUT_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Ok(guard) = cache.lock()
            && let Some(entry) = guard.get(path_or_id)
            && entry.file_size == file_size
            && entry.mtime_ns == mtime_ns
        {
            return Ok(Arc::clone(&entry.lut));
        }

        let parsed = Arc::new(Self::load_or_builtin(path_or_id)?);
        if let Ok(mut guard) = cache.lock() {
            if guard.len() >= MAX_CACHED_CUBE_LUTS && !guard.contains_key(path_or_id) {
                guard.clear();
            }
            guard.insert(
                path_or_id.to_string(),
                CachedCubeLutEntry {
                    file_size,
                    mtime_ns,
                    lut: Arc::clone(&parsed),
                },
            );
        }
        Ok(parsed)
    }

    /// Loads a `.cube` 3D LUT from disk, or synthesizes one of Kroma's 4 built-in Studio `.cube`
    /// profiles (`Kodak_Portra_400.cube`, `Fuji_Pro_400H.cube`, `Kroma_Studio_Neutral.cube`,
    /// `Ilford_HP5_Mono.cube`) when referenced by built-in virtual path.
    pub fn load_or_builtin(path_or_id: &str) -> Result<Self, AppError> {
        let p = Path::new(path_or_id);
        if p.is_file() {
            return Self::from_file(p);
        }
        if let Some(builtin) = Self::builtin_by_name(path_or_id) {
            return Ok(builtin);
        }
        Self::from_file(p)
    }

    pub fn builtin_by_name(name_or_path: &str) -> Option<Self> {
        let lower = name_or_path.trim().to_ascii_lowercase();
        let kind = if lower.contains("kodak_portra_400") {
            1
        } else if lower.contains("fuji_pro_400h") {
            2
        } else if lower.contains("kroma_studio_neutral") {
            3
        } else if lower.contains("ilford_hp5_mono") {
            4
        } else {
            return None;
        };

        let size = 17usize;
        let max_idx = (size - 1) as f32;
        let mut table = Vec::with_capacity(size * size * size);

        for b_idx in 0..size {
            let b = b_idx as f32 / max_idx;
            for g_idx in 0..size {
                let g = g_idx as f32 / max_idx;
                for r_idx in 0..size {
                    let r = r_idx as f32 / max_idx;
                    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                    let mapped = match kind {
                        // 1: Kodak Portra 400 — warm golden skin tones, gentle toe lift, creamy highlights
                        1 => {
                            let lift = |x: f32| {
                                let s = x * x * (3.0 - 2.0 * x);
                                (x * 0.35 + s * 0.65) * 0.96 + 0.025
                            };
                            let mr = (lift(r) * 1.035 + 0.012 * (1.0 - luma)).clamp(0.0, 1.0);
                            let mg = (lift(g) * 1.005 + 0.004 * luma).clamp(0.0, 1.0);
                            let mb = (lift(b) * 0.945 + 0.015 * (1.0 - luma)).clamp(0.0, 1.0);
                            [mr, mg, mb]
                        }
                        // 2: Fuji Pro 400H — airy pastel cyan-green shadows, clean neutral-warm highlights
                        2 => {
                            let lift = |x: f32| {
                                let s = x * x * (3.0 - 2.0 * x);
                                (x * 0.42 + s * 0.58) * 0.97 + 0.02
                            };
                            let shd = (1.0 - luma).powi(2);
                            let mr = (lift(r) * 0.985 + 0.008 * luma).clamp(0.0, 1.0);
                            let mg = (lift(g) * 1.018 + 0.012 * shd).clamp(0.0, 1.0);
                            let mb = (lift(b) * 1.015 + 0.018 * shd).clamp(0.0, 1.0);
                            [mr, mg, mb]
                        }
                        // 3: Kroma Studio Neutral — subtle studio contrast S-curve with natural saturation
                        3 => {
                            let s_curve = |x: f32| {
                                let s = x * x * (3.0 - 2.0 * x);
                                x * 0.45 + s * 0.55
                            };
                            let sr = s_curve(r);
                            let sg = s_curve(g);
                            let sb = s_curve(b);
                            let sl = 0.2126 * sr + 0.7152 * sg + 0.0722 * sb;
                            [
                                (sl + (sr - sl) * 1.06).clamp(0.0, 1.0),
                                (sl + (sg - sl) * 1.06).clamp(0.0, 1.0),
                                (sl + (sb - sl) * 1.06).clamp(0.0, 1.0),
                            ]
                        }
                        // 4: Ilford HP5 Mono — punchy silver-halide panchromatic B&W film curve
                        _ => {
                            let pan = (0.26 * r + 0.64 * g + 0.10 * b).clamp(0.0, 1.0);
                            let s = pan * pan * (3.0 - 2.0 * pan);
                            let film = (pan * 0.28 + s * 0.72) * 0.95 + 0.025;
                            [film, film, film]
                        }
                    };
                    table.push(mapped);
                }
            }
        }

        Some(Self {
            size,
            domain_min: [0.0, 0.0, 0.0],
            domain_max: [1.0, 1.0, 1.0],
            table,
        })
    }

    pub fn parse(content: &str) -> Result<Self, AppError> {
        let mut size: Option<usize> = None;
        let mut domain_min = [0.0_f32; 3];
        let mut domain_max = [1.0_f32; 3];
        let mut table: Vec<[f32; 3]> = Vec::new();

        for raw_line in content.lines() {
            let line = raw_line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }

            if let Some(rest) = line.strip_prefix("TITLE") {
                let _ = rest;
                continue;
            }

            if let Some(rest) = line.strip_prefix("LUT_3D_SIZE") {
                let parsed = rest.trim().parse::<usize>().map_err(|err| {
                    AppError::ImageProcessingFailure(format!("invalid LUT_3D_SIZE: {err}"))
                })?;
                if !(2..=129).contains(&parsed) {
                    return Err(AppError::ImageProcessingFailure(format!(
                        "unsupported LUT_3D_SIZE {parsed} (expected 2..=129)"
                    )));
                }
                size = Some(parsed);
                table.reserve(parsed * parsed * parsed);
                continue;
            }

            if let Some(rest) = line.strip_prefix("DOMAIN_MIN") {
                domain_min = parse_rgb_triplet(rest, "DOMAIN_MIN")?;
                continue;
            }

            if let Some(rest) = line.strip_prefix("DOMAIN_MAX") {
                domain_max = parse_rgb_triplet(rest, "DOMAIN_MAX")?;
                continue;
            }

            if line.starts_with("LUT_1D_SIZE") {
                return Err(AppError::ImageProcessingFailure(
                    "1D .cube LUTs are not supported; provide a 3D .cube LUT".to_string(),
                ));
            }

            table.push(parse_rgb_triplet(line, "LUT entry")?);
        }

        let grid_size = size.ok_or_else(|| {
            AppError::ImageProcessingFailure("missing LUT_3D_SIZE header in .cube file".to_string())
        })?;
        let expected_len = grid_size * grid_size * grid_size;
        if table.len() != expected_len {
            return Err(AppError::ImageProcessingFailure(format!(
                "expected {expected_len} RGB triplets for {grid_size}^3 LUT, found {}",
                table.len()
            )));
        }

        Ok(Self {
            size: grid_size,
            domain_min,
            domain_max,
            table,
        })
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn domain_min(&self) -> [f32; 3] {
        self.domain_min
    }

    pub fn domain_max(&self) -> [f32; 3] {
        self.domain_max
    }

    pub fn table(&self) -> &[[f32; 3]] {
        &self.table
    }

    /// Samples the 3D LUT via trilinear interpolation and blends with the source pixel by `intensity`.
    pub fn apply(&self, rgb: [f32; 3], intensity: f32) -> [f32; 3] {
        let blend = intensity.clamp(0.0, 1.0);
        if blend <= f32::EPSILON {
            return rgb;
        }

        let max_idx = (self.size - 1) as f32;
        let norm_r = normalize_channel(rgb[0], self.domain_min[0], self.domain_max[0]) * max_idx;
        let norm_g = normalize_channel(rgb[1], self.domain_min[1], self.domain_max[1]) * max_idx;
        let norm_b = normalize_channel(rgb[2], self.domain_min[2], self.domain_max[2]) * max_idx;

        let r0 = (norm_r.floor() as usize).min(self.size - 1);
        let g0 = (norm_g.floor() as usize).min(self.size - 1);
        let b0 = (norm_b.floor() as usize).min(self.size - 1);

        let r1 = (r0 + 1).min(self.size - 1);
        let g1 = (g0 + 1).min(self.size - 1);
        let b1 = (b0 + 1).min(self.size - 1);

        let fr = norm_r - (r0 as f32);
        let fg = norm_g - (g0 as f32);
        let fb = norm_b - (b0 as f32);

        let c000 = self.sample(r0, g0, b0);
        let c100 = self.sample(r1, g0, b0);
        let c010 = self.sample(r0, g1, b0);
        let c110 = self.sample(r1, g1, b0);
        let c001 = self.sample(r0, g0, b1);
        let c101 = self.sample(r1, g0, b1);
        let c011 = self.sample(r0, g1, b1);
        let c111 = self.sample(r1, g1, b1);

        let c00 = lerp_rgb(c000, c100, fr);
        let c10 = lerp_rgb(c010, c110, fr);
        let c01 = lerp_rgb(c001, c101, fr);
        let c11 = lerp_rgb(c011, c111, fr);

        let c0 = lerp_rgb(c00, c10, fg);
        let c1 = lerp_rgb(c01, c11, fg);

        let mapped = lerp_rgb(c0, c1, fb);
        lerp_rgb(rgb, mapped, blend)
    }

    #[inline]
    fn sample(&self, r: usize, g: usize, b: usize) -> [f32; 3] {
        self.table[r + self.size * (g + self.size * b)]
    }
}

fn normalize_channel(val: f32, min: f32, max: f32) -> f32 {
    let span = (max - min).max(1e-6);
    ((val - min) / span).clamp(0.0, 1.0)
}

#[inline]
fn lerp_rgb(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn parse_rgb_triplet(text: &str, context: &str) -> Result<[f32; 3], AppError> {
    let mut parts = text.split_whitespace();
    let r = parts
        .next()
        .ok_or_else(|| AppError::ImageProcessingFailure(format!("missing R value in {context}")))?
        .parse::<f32>()
        .map_err(|err| AppError::ImageProcessingFailure(format!("invalid R in {context}: {err}")))?;
    let g = parts
        .next()
        .ok_or_else(|| AppError::ImageProcessingFailure(format!("missing G value in {context}")))?
        .parse::<f32>()
        .map_err(|err| AppError::ImageProcessingFailure(format!("invalid G in {context}: {err}")))?;
    let b = parts
        .next()
        .ok_or_else(|| AppError::ImageProcessingFailure(format!("missing B value in {context}")))?
        .parse::<f32>()
        .map_err(|err| AppError::ImageProcessingFailure(format!("invalid B in {context}: {err}")))?;
    Ok([r, g, b])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_identity_cube_lut_and_interpolates_midtones() {
        let identity_2x2x2 = "\
# Minimal identity 2x2x2 LUT
TITLE \"Identity\"
LUT_3D_SIZE 2
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 0.0
0.0 0.0 1.0
1.0 0.0 1.0
0.0 1.0 1.0
1.0 1.0 1.0
";
        let lut = CubeLut3D::parse(identity_2x2x2).expect("valid LUT");
        assert_eq!(lut.size(), 2);

        let out = lut.apply([0.25, 0.5, 0.75], 1.0);
        assert!((out[0] - 0.25).abs() < 1e-5);
        assert!((out[1] - 0.50).abs() < 1e-5);
        assert!((out[2] - 0.75).abs() < 1e-5);
    }

    #[test]
    fn load_cached_returns_shared_arc_and_distinct_fingerprints() {
        let portra_a = CubeLut3D::load_cached("Kodak_Portra_400.cube").expect("builtin portra");
        let portra_b = CubeLut3D::load_cached("Kodak_Portra_400.cube").expect("cached portra");
        let fuji = CubeLut3D::load_cached("Fuji_Pro_400H.cube").expect("builtin fuji");

        assert!(Arc::ptr_eq(&portra_a, &portra_b));
        assert_eq!(portra_a.fingerprint(), portra_b.fingerprint());
        assert_ne!(portra_a.fingerprint(), fuji.fingerprint());
    }
}
