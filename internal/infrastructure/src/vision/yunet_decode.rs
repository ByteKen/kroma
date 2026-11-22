use kroma_app::ports::vision::RawFaceDetection;
use kroma_domain::identity::{BoundingBox, Point2D, YuNetLandmarks};

pub const YUNET_STRIDES: [usize; 3] = [8, 16, 32];

/// Raw feature-pyramid slices for one YuNet stride (`8`, `16`, or `32`).
pub struct YuNetStrideOutputs<'a> {
    pub stride: usize,
    pub cls: &'a [f32],
    pub obj: &'a [f32],
    pub bbox: &'a [f32],
    pub kps: &'a [f32],
}

#[derive(Debug, Clone, Copy)]
pub struct YuNetDecodeSpec {
    pub pad_w: usize,
    pub pad_h: usize,
    pub resized_w: usize,
    pub resized_h: usize,
    pub score_threshold: f32,
    pub nms_iou_threshold: f32,
    pub top_k: usize,
}

/// Decodes OpenCV Zoo `face_detection_yunet_2023mar.onnx` 12-tensor feature-pyramid outputs
/// into normalized `[0..1]` `RawFaceDetection`s and applies Non-Maximum Suppression (NMS).
pub fn decode_yunet_outputs(
    pyramid: &[YuNetStrideOutputs<'_>],
    spec: YuNetDecodeSpec,
) -> Vec<RawFaceDetection> {
    let mut candidates = Vec::new();
    let inv_w = 1.0 / (spec.resized_w.max(1) as f32);
    let inv_h = 1.0 / (spec.resized_h.max(1) as f32);

    for level in pyramid {
        let s = level.stride;
        let cols = spec.pad_w / s;
        let rows = spec.pad_h / s;
        let stride_f = s as f32;

        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                if idx >= level.cls.len() || idx >= level.obj.len() {
                    continue;
                }

                let cls_score = level.cls[idx].clamp(0.0, 1.0);
                let obj_score = level.obj[idx].clamp(0.0, 1.0);
                let score = (cls_score * obj_score).sqrt();
                if score < spec.score_threshold {
                    continue;
                }

                let b_off = idx * 4;
                let k_off = idx * 10;
                if b_off + 3 >= level.bbox.len() || k_off + 9 >= level.kps.len() {
                    continue;
                }

                let cx = (c as f32 + level.bbox[b_off]) * stride_f;
                let cy = (r as f32 + level.bbox[b_off + 1]) * stride_f;
                let w = level.bbox[b_off + 2].exp() * stride_f;
                let h = level.bbox[b_off + 3].exp() * stride_f;

                let x1 = (cx - w * 0.5).clamp(0.0, spec.resized_w as f32);
                let y1 = (cy - h * 0.5).clamp(0.0, spec.resized_h as f32);
                let bw = w.min((spec.resized_w as f32) - x1).max(0.0);
                let bh = h.min((spec.resized_h as f32) - y1).max(0.0);

                let pt = |k: usize| Point2D {
                    x: (((c as f32 + level.kps[k_off + 2 * k]) * stride_f) * inv_w).clamp(0.0, 1.0),
                    y: (((r as f32 + level.kps[k_off + 2 * k + 1]) * stride_f) * inv_h)
                        .clamp(0.0, 1.0),
                };

                candidates.push(RawFaceDetection {
                    bounding_box: BoundingBox {
                        x: x1 * inv_w,
                        y: y1 * inv_h,
                        width: bw * inv_w,
                        height: bh * inv_h,
                    },
                    landmarks: YuNetLandmarks {
                        right_eye: pt(0),
                        left_eye: pt(1),
                        nose_tip: pt(2),
                        right_mouth_corner: pt(3),
                        left_mouth_corner: pt(4),
                    },
                    detection_confidence: score,
                });
            }
        }
    }

    apply_nms(candidates, spec.nms_iou_threshold, spec.top_k)
}

pub fn apply_nms(
    mut detections: Vec<RawFaceDetection>,
    iou_threshold: f32,
    top_k: usize,
) -> Vec<RawFaceDetection> {
    detections.sort_by(|a, b| b.detection_confidence.total_cmp(&a.detection_confidence));
    if detections.len() > top_k {
        detections.truncate(top_k);
    }

    let mut kept: Vec<RawFaceDetection> = Vec::with_capacity(detections.len());
    for det in detections {
        let suppressed = kept
            .iter()
            .any(|existing| compute_iou(existing.bounding_box, det.bounding_box) > iou_threshold);
        if !suppressed {
            kept.push(det);
        }
    }
    kept
}

fn compute_iou(a: BoundingBox, b: BoundingBox) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.width).min(b.x + b.width);
    let y2 = (a.y + a.height).min(b.y + b.height);

    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let inter_area = inter_w * inter_h;
    let union_area = a.area() + b.area() - inter_area;
    if union_area <= 1e-8 {
        0.0
    } else {
        inter_area / union_area
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_yunet_stride_anchor_and_suppresses_duplicate_box_via_nms() {
        // 32x32 padded grid with stride 16 -> 2x2 = 4 cells
        let cls = [0.95, 0.80, 0.01, 0.01];
        let obj = [0.95, 0.80, 0.01, 0.01];
        // Anchor (0,0) and Anchor (1,0) both predict nearly the exact same face box centered at (16, 16)
        let bbox = [
            1.0, 1.0, 0.0, 0.0, // cx=16, cy=16, w=16, h=16
            0.02, 1.0, 0.0, 0.0, // cx=16.32, cy=16, w=16, h=16 (heavy overlap -> NMS drops)
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0,
        ];
        let kps = [0.5_f32; 40];

        let pyramid = [YuNetStrideOutputs {
            stride: 16,
            cls: &cls,
            obj: &obj,
            bbox: &bbox,
            kps: &kps,
        }];

        let results = decode_yunet_outputs(
            &pyramid,
            YuNetDecodeSpec {
                pad_w: 32,
                pad_h: 32,
                resized_w: 32,
                resized_h: 32,
                score_threshold: 0.6,
                nms_iou_threshold: 0.3,
                top_k: 100,
            },
        );
        assert_eq!(results.len(), 1);
        assert!((results[0].detection_confidence - 0.95).abs() < 1e-4);
        assert!((results[0].bounding_box.x - 0.25).abs() < 1e-4);
        assert!((results[0].bounding_box.width - 0.5).abs() < 1e-4);
    }
}
