use image::{Rgb, RgbImage};
use kroma_domain::identity::{Point2D, YuNetLandmarks};

pub const SFACE_INPUT_SIZE: u32 = 112;

/// Canonical 5-point ArcFace / OpenCV SFace reference landmarks for a 112x112 upright face crop.
/// Order matches YuNet `kps` output: right eye, left eye, nose tip, right mouth, left mouth.
pub const SFACE_REFERENCE_POINTS: [Point2D; 5] = [
    Point2D { x: 38.2946, y: 51.6963 },
    Point2D { x: 73.5318, y: 51.5014 },
    Point2D { x: 56.0252, y: 71.7366 },
    Point2D { x: 41.5493, y: 92.3655 },
    Point2D { x: 70.7299, y: 92.2041 },
];

/// Closed-form 2D non-reflective similarity transform:
///   dst_x = a * src_x - b * src_y + tx
///   dst_y = b * src_x + a * src_y + ty
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimilarityTransform2D {
    pub a: f32,
    pub b: f32,
    pub tx: f32,
    pub ty: f32,
}

impl SimilarityTransform2D {
    /// Estimates the least-squares 2D similarity transform mapping `src` -> `dst` (Umeyama 2D).
    pub fn estimate(src: &[Point2D; 5], dst: &[Point2D; 5]) -> Option<Self> {
        let mut src_mean_x = 0.0_f32;
        let mut src_mean_y = 0.0_f32;
        let mut dst_mean_x = 0.0_f32;
        let mut dst_mean_y = 0.0_f32;

        for i in 0..5 {
            src_mean_x += src[i].x;
            src_mean_y += src[i].y;
            dst_mean_x += dst[i].x;
            dst_mean_y += dst[i].y;
        }
        src_mean_x *= 0.2;
        src_mean_y *= 0.2;
        dst_mean_x *= 0.2;
        dst_mean_y *= 0.2;

        let mut denom = 0.0_f32;
        let mut num_a = 0.0_f32;
        let mut num_b = 0.0_f32;

        for i in 0..5 {
            let sx = src[i].x - src_mean_x;
            let sy = src[i].y - src_mean_y;
            let dx = dst[i].x - dst_mean_x;
            let dy = dst[i].y - dst_mean_y;

            denom += sx * sx + sy * sy;
            num_a += sx * dx + sy * dy;
            num_b += sx * dy - sy * dx;
        }

        if denom <= 1e-6 {
            return None;
        }

        let a = num_a / denom;
        let b = num_b / denom;
        let tx = dst_mean_x - (a * src_mean_x - b * src_mean_y);
        let ty = dst_mean_y - (b * src_mean_x + a * src_mean_y);

        Some(Self { a, b, tx, ty })
    }

    #[inline]
    pub fn apply(&self, p: Point2D) -> Point2D {
        Point2D {
            x: self.a * p.x - self.b * p.y + self.tx,
            y: self.b * p.x + self.a * p.y + self.ty,
        }
    }

    /// Maps a destination pixel `(u, v)` back to source image coordinates `(src_x, src_y)`.
    #[inline]
    pub fn inverse_map(&self, u: f32, v: f32) -> Option<(f32, f32)> {
        let det = self.a * self.a + self.b * self.b;
        if det <= 1e-8 {
            return None;
        }
        let ux = u - self.tx;
        let vy = v - self.ty;
        let sx = (self.a * ux + self.b * vy) / det;
        let sy = (-self.b * ux + self.a * vy) / det;
        Some((sx, sy))
    }
}

/// Aligns and crops a detected face into a canonical `112x112` RGB patch for SFace inference,
/// using normalized `[0..1]` YuNet landmarks scaled to the source image dimensions.
pub fn align_face_112(source: &RgbImage, normalized_landmarks: &YuNetLandmarks) -> Option<RgbImage> {
    let (w, h) = source.dimensions();
    if w < 2 || h < 2 {
        return None;
    }

    let fw = w as f32;
    let fh = h as f32;
    let src_pts = [
        Point2D {
            x: normalized_landmarks.right_eye.x * fw,
            y: normalized_landmarks.right_eye.y * fh,
        },
        Point2D {
            x: normalized_landmarks.left_eye.x * fw,
            y: normalized_landmarks.left_eye.y * fh,
        },
        Point2D {
            x: normalized_landmarks.nose_tip.x * fw,
            y: normalized_landmarks.nose_tip.y * fh,
        },
        Point2D {
            x: normalized_landmarks.right_mouth_corner.x * fw,
            y: normalized_landmarks.right_mouth_corner.y * fh,
        },
        Point2D {
            x: normalized_landmarks.left_mouth_corner.x * fw,
            y: normalized_landmarks.left_mouth_corner.y * fh,
        },
    ];

    let transform = SimilarityTransform2D::estimate(&src_pts, &SFACE_REFERENCE_POINTS)?;
    let mut aligned = RgbImage::new(SFACE_INPUT_SIZE, SFACE_INPUT_SIZE);
    let max_x = fw - 1.0;
    let max_y = fh - 1.0;

    for v in 0..SFACE_INPUT_SIZE {
        for u in 0..SFACE_INPUT_SIZE {
            let Some((sx, sy)) = transform.inverse_map(u as f32, v as f32) else {
                continue;
            };
            if sx < 0.0 || sy < 0.0 || sx >= max_x || sy >= max_y {
                continue;
            }
            let x0 = sx.floor() as u32;
            let y0 = sy.floor() as u32;
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;

            let p00 = source.get_pixel(x0, y0).0;
            let p10 = source.get_pixel(x0 + 1, y0).0;
            let p01 = source.get_pixel(x0, y0 + 1).0;
            let p11 = source.get_pixel(x0 + 1, y0 + 1).0;

            let mut rgb = [0_u8; 3];
            for c in 0..3 {
                let top = p00[c] as f32 * (1.0 - fx) + p10[c] as f32 * fx;
                let bot = p01[c] as f32 * (1.0 - fx) + p11[c] as f32 * fx;
                rgb[c] = (top * (1.0 - fy) + bot * fy).round().clamp(0.0, 255.0) as u8;
            }
            aligned.put_pixel(u, v, Rgb(rgb));
        }
    }

    Some(aligned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similarity_transform_recovers_exact_rotated_scaled_translated_landmarks() {
        // Synthesize source landmarks rotated by +18 deg, scaled by 2.5x, and shifted by (+120, +80)
        let angle = 18.0_f32.to_radians();
        let scale = 2.5_f32;
        let cos_a = angle.cos() * scale;
        let sin_a = angle.sin() * scale;

        let mut synthetic_src = [Point2D { x: 0.0, y: 0.0 }; 5];
        for (i, pt) in SFACE_REFERENCE_POINTS.iter().enumerate() {
            synthetic_src[i] = Point2D {
                x: cos_a * pt.x - sin_a * pt.y + 120.0,
                y: sin_a * pt.x + cos_a * pt.y + 80.0,
            };
        }

        let estimated =
            SimilarityTransform2D::estimate(&synthetic_src, &SFACE_REFERENCE_POINTS).unwrap();
        for i in 0..5 {
            let mapped = estimated.apply(synthetic_src[i]);
            assert!((mapped.x - SFACE_REFERENCE_POINTS[i].x).abs() < 1e-3);
            assert!((mapped.y - SFACE_REFERENCE_POINTS[i].y).abs() < 1e-3);

            let (inv_x, inv_y) = estimated
                .inverse_map(SFACE_REFERENCE_POINTS[i].x, SFACE_REFERENCE_POINTS[i].y)
                .unwrap();
            assert!((inv_x - synthetic_src[i].x).abs() < 1e-3);
            assert!((inv_y - synthetic_src[i].y).abs() < 1e-3);
        }
    }
}
