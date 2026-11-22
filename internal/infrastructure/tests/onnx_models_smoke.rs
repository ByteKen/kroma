use image::{Rgb, RgbImage};
use kroma_domain::identity::{Point2D, YuNetLandmarks, ARCFACE_EMBEDDING_DIM, SFACE_EMBEDDING_DIM};
use kroma_domain::masking::{decode_matte_rle_b64, AiMaskKind};
use kroma_infra::vision::ppocr_v5::{PpOcrV5Config, PpOcrV5Engine};
use kroma_infra::vision::sface_recognizer::{FaceEmbeddingModelProfile, SFaceOnnxRecognizer};
use kroma_infra::vision::subject_segmenter::{generate_ai_mask_for_rgb, OnnxSubjectSegmenter};
use kroma_infra::vision::yunet_detector::{YuNetConfig, YuNetOnnxDetector};
use std::path::PathBuf;

/// Live integration smoke test verifying the real downloaded ONNX models:
/// - `models/face_detection_yunet_2026may.onnx` (dynamic input ORT export)
/// - `models/w600k_r50.onnx` (InsightFace ArcFace ResNet-50 512-D embedding)
/// - `models/face_recognition_sface_2021dec.onnx` (OpenCV Zoo SFace 128-D fallback)
#[test]
fn real_yunet_2026_and_face_recognition_models_load_and_infer_cleanly() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let models_dir = workspace_root.join("models");
    let yunet_path = models_dir.join("face_detection_yunet_2026may.onnx");
    let sface_path = models_dir.join("face_recognition_sface_2021dec.onnx");
    let arcface_path = models_dir.join("w600k_r50.onnx");

    if !yunet_path.exists() {
        eprintln!("skipping live ONNX smoke test (YuNet not found at {yunet_path:?})");
        return;
    }

    // 1. Test YuNet 2026may on a non-square portrait image (480x640)
    let detector =
        YuNetOnnxDetector::from_model_file(&yunet_path, YuNetConfig::default()).expect("load YuNet");
    let portrait_img = RgbImage::from_pixel(480, 640, Rgb([128, 128, 128]));
    let detections = detector
        .detect_faces_in_rgb(&portrait_img)
        .expect("YuNet forward pass succeeds on dynamic portrait shape");
    assert!(
        detections.is_empty(),
        "flat gray image should produce 0 false face detections"
    );

    let landmarks = YuNetLandmarks {
        right_eye: Point2D { x: 0.35, y: 0.42 },
        left_eye: Point2D { x: 0.65, y: 0.42 },
        nose_tip: Point2D { x: 0.50, y: 0.56 },
        right_mouth_corner: Point2D { x: 0.40, y: 0.72 },
        left_mouth_corner: Point2D { x: 0.60, y: 0.72 },
    };

    // 2. Test ArcFace ResNet-50 512-D (w600k_r50.onnx) single + batched inference when installed
    if arcface_path.exists() {
        let arcface = SFaceOnnxRecognizer::from_model_file(&arcface_path, 2).expect("load ArcFace");
        assert_eq!(arcface.profile(), FaceEmbeddingModelProfile::ArcFaceRgbNorm512);
        assert_eq!(arcface.embedding_dim(), ARCFACE_EMBEDDING_DIM);
        let emb = arcface
            .extract_embedding_from_rgb(&portrait_img, &landmarks)
            .expect("ArcFace forward pass succeeds");
        assert_eq!(emb.dim(), ARCFACE_EMBEDDING_DIM);
        assert!((emb.cosine_similarity(&emb) - 1.0).abs() < 1e-4);

        let batch_crops = vec![RgbImage::from_pixel(112, 112, Rgb([128, 128, 128])); 4];
        let started = std::time::Instant::now();
        let batch_embs = arcface
            .extract_embeddings_batch_from_aligned(&batch_crops)
            .expect("ArcFace chunked inference succeeds");
        eprintln!(
            "ArcFace R50 ({}) 4 faces in {}ms ({:.1}ms/face)",
            arcface.execution_provider_label(),
            started.elapsed().as_millis(),
            started.elapsed().as_secs_f64() * 1000.0 / 4.0
        );
        assert_eq!(batch_embs.len(), 4);
        assert_eq!(batch_embs[0].dim(), ARCFACE_EMBEDDING_DIM);
    }

    // 3. Test SFace 2021dec 128-D fallback when installed
    if sface_path.exists() {
        let sface = SFaceOnnxRecognizer::from_model_file(&sface_path, 2).expect("load SFace");
        assert_eq!(sface.profile(), FaceEmbeddingModelProfile::SFaceRgb255_128);
        assert_eq!(sface.embedding_dim(), SFACE_EMBEDDING_DIM);
        let emb = sface
            .extract_embedding_from_rgb(&portrait_img, &landmarks)
            .expect("SFace forward pass succeeds");
        assert_eq!(emb.dim(), SFACE_EMBEDDING_DIM);
        assert!((emb.cosine_similarity(&emb) - 1.0).abs() < 1e-4);
    }
}

/// Live end-to-end PP-OCRv5 smoke test verifying that the downloaded `PP-OCRv5_server_det` +
/// `en_PP-OCRv5_mobile_rec` + `en_ppocrv5_dict.txt` accurately detect and read whiteboard text!
#[test]
fn real_ppocr_v5_models_detect_and_read_synthetic_whiteboard_slate() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let ocr_dir = workspace_root.join("models").join("ocr");
    let mobile_det = ocr_dir.join("ppocrv5_mobile_det.onnx");
    let det_path = if mobile_det.exists() {
        mobile_det
    } else {
        ocr_dir.join("ppocrv5_det.onnx")
    };
    let rec_path = ocr_dir.join("en_ppocrv5_rec.onnx");
    let dict_path = ocr_dir.join("en_ppocrv5_dict.txt");

    if !det_path.exists() || !rec_path.exists() || !dict_path.exists() {
        eprintln!("skipping live PP-OCRv5 smoke test (models not found at {ocr_dir:?})");
        return;
    }

    let engine = PpOcrV5Engine::from_model_files(
        &det_path,
        &rec_path,
        &dict_path,
        PpOcrV5Config::default(),
    )
    .expect("load PP-OCRv5 ONNX engine");

    // Draw a realistic studio portrait frame with a bright white placard holding "ELENA CRUZ"
    let mut img = RgbImage::from_pixel(640, 800, Rgb([52, 54, 58]));
    for y in 420..650 {
        for x in 90..550 {
            img.put_pixel(x, y, Rgb([244, 245, 247]));
        }
    }
    draw_block_text(&mut img, 125, 465, "ELENA CRUZ", 7, Rgb([22, 24, 28]));
    draw_block_text(&mut img, 225, 555, "042", 7, Rgb([22, 24, 28]));

    let started = std::time::Instant::now();
    let (lines, parsed) = engine
        .read_slate_from_rgb(&img)
        .expect("PP-OCRv5 forward pass succeeds");
    let first_ms = started.elapsed().as_millis();
    let second_started = std::time::Instant::now();
    let _ = engine
        .read_slate_from_rgb(&img)
        .expect("PP-OCRv5 second pass succeeds");
    let warm_ms = second_started.elapsed().as_millis();
    eprintln!(
        "PP-OCRv5 ({}, rec_h={}) read {} line(s) (warmup {}ms, steady {}ms): {:?} -> parsed: {:?}",
        engine.execution_provider_label(),
        engine.rec_height(),
        lines.len(),
        first_ms,
        warm_ms,
        lines.iter().map(|l| (&l.text, l.confidence)).collect::<Vec<_>>(),
        parsed
    );

    let parsed = parsed.expect("PP-OCRv5 should parse whiteboard name");
    let upper = parsed.subject_name.to_ascii_uppercase();
    assert!(
        upper.contains("ELENA") && upper.contains("CRUZ"),
        "expected 'ELENA CRUZ' in '{}'",
        parsed.subject_name
    );
    assert_eq!(parsed.subject_code, None);
    assert!(parsed.confidence > 0.70);
}

#[test]
fn real_rmbg_1_4_subject_segmenter_loads_and_masks_cleanly() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let rmbg_path = workspace_root.join("models").join("rmbg-1.4.onnx");
    if !rmbg_path.exists() {
        eprintln!("skipping live RMBG-1.4 smoke test (not found at {rmbg_path:?})");
        return;
    }

    let segmenter = OnnxSubjectSegmenter::from_model_file(&rmbg_path, 4).expect("load RMBG-1.4");
    let mut img = RgbImage::from_pixel(320, 420, Rgb([48, 50, 54]));
    for y in 70..165 {
        for x in 115..205 {
            img.put_pixel(x, y, Rgb([215, 175, 148]));
        }
    }
    for y in 165..410 {
        for x in 80..240 {
            img.put_pixel(x, y, Rgb([242, 238, 224]));
        }
    }

    let result =
        generate_ai_mask_for_rgb(&img, AiMaskKind::Subject, Some(&segmenter), None)
            .expect("RMBG-1.4 inference succeeds");
    eprintln!(
        "RMBG-1.4 ({}) generated {}x{} matte ({} b64 bytes) in {}ms",
        result.engine_label,
        result.matte_width,
        result.matte_height,
        result.matte_rle_b64.len(),
        result.elapsed_ms
    );
    let mw = result.matte_width as usize;
    let mh = result.matte_height as usize;
    let matte = decode_matte_rle_b64(&result.matte_rle_b64, mw * mh).expect("decode matte");
    let center = matte[(mh / 2) * mw + (mw / 2)];
    let corner = matte[4 * mw + 4];
    assert!(center > 180, "expected subject center > 180, got {center}");
    assert!(corner < 40, "expected backdrop corner < 40, got {corner}");
}

fn draw_block_text(
    img: &mut RgbImage,
    start_x: u32,
    start_y: u32,
    text: &str,
    scale: u32,
    color: Rgb<u8>,
) {
    let mut cursor_x = start_x;
    for ch in text.chars() {
        if let Some(glyph) = bitmap_5x7(ch) {
            for (row, &bits) in glyph.iter().enumerate() {
                for col in 0..5u32 {
                    if (bits >> (4 - col)) & 1 == 1 {
                        for sy in 0..scale {
                            for sx in 0..scale {
                                let px = cursor_x + col * scale + sx;
                                let py = start_y + (row as u32) * scale + sy;
                                if px < img.width() && py < img.height() {
                                    img.put_pixel(px, py, color);
                                }
                            }
                        }
                    }
                }
            }
        }
        cursor_x += 6 * scale;
    }
}

fn bitmap_5x7(ch: char) -> Option<[u8; 7]> {
    Some(match ch {
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        ' ' => [0; 7],
        _ => return None,
    })
}
