use kroma_app::error::AppError;
use std::fs;
use std::path::Path;

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
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, AppError> {
        let content = fs::read_to_string(path.as_ref()).map_err(|err| {
            AppError::ImageProcessingFailure(format!(
                "failed to read .cube LUT '{}': {err}",
                path.as_ref().display()
            ))
        })?;
        Self::parse(&content)
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
}
