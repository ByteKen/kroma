use image::{DynamicImage, Rgb, Rgb32FImage, RgbImage};
use kroma_domain::develop::{DevelopRecipe, NormalizedRect};
use kroma_domain::retouch::{
    FaceParseMap, HealSpot, HealSpotMode, PortraitFaceGuide, PortraitRetouch,
};
use kroma_infra::develop::image_decoder::{decode_develop_buffer, DecodedDevelopBuffer};
use kroma_infra::develop::pixel_engine::RayonImageDevelopEngine;
use kroma_infra::develop::wgpu_engine::shared_wgpu_develop_engine;
use kroma_infra::vision::lama_inpainter::{generate_inpaint_patch, InpaintStrokeInput};
use kroma_infra::vision::yunet_detector::{YuNetConfig, YuNetOnnxDetector};
use rayon::prelude::*;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn make_portrait_buffer(w: u32, h: u32, is_linear: bool) -> DecodedDevelopBuffer {
    let mut img = Rgb32FImage::new(w, h);
    for (x, y, px) in img.enumerate_pixels_mut() {
        let u = x as f32 / (w.saturating_sub(1).max(1) as f32);
        let v = y as f32 / (h.saturating_sub(1).max(1) as f32);
        let pore = ((x as f32 * 0.8).sin() * (y as f32 * 0.8).cos()) * 0.03;
        *px = Rgb([
            (0.72 + 0.12 * u + pore).clamp(0.0, 1.0),
            (0.56 + 0.10 * v + pore).clamp(0.0, 1.0),
            (0.46 + 0.08 * (1.0 - u) + pore).clamp(0.0, 1.0),
        ]);
    }
    DecodedDevelopBuffer {
        image: img,
        is_linear,
        is_wide_gamut: is_linear,
        bit_depth: if is_linear { 32 } else { 8 },
        is_raw_demosaic: is_linear,
    }
}

fn sample_face_guide() -> PortraitFaceGuide {
    PortraitFaceGuide::from_normalized(
        0.25,
        0.18,
        0.48,
        0.56,
        [0.41, 0.39],
        [0.59, 0.39],
        [0.50, 0.51],
        [0.43, 0.63],
        [0.57, 0.63],
    )
}

#[test]
fn retouch_32bit_thumbnails_render_cleanly_with_lazy_wgpu_and_concurrency() {
    let buf = make_portrait_buffer(640, 427, true);
    let guide = sample_face_guide();
    let odd_fp = FaceParseMap {
        width: 319,
        height: 479,
        rgba: vec![200u8; 319 * 479 * 4],
    };

    let hi_rgb = RgbImage::from_fn(640, 427, |x, y| {
        Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
    });
    let patch = generate_inpaint_patch(
        &hi_rgb,
        &[InpaintStrokeInput {
            radius: 0.05,
            points: vec![[0.35, 0.40], [0.43, 0.44]],
        }],
        None,
    )
    .expect("inpaint patch")
    .patch;

    // 1. Exercise lazy per-pass GPU pipeline initialization across every retouch mode when a
    //    hardware GPU adapter is present.
    if let Some(gpu) = shared_wgpu_develop_engine() {
        let mut r_tone = DevelopRecipe::default();
        r_tone.basic_tone.exposure_ev = 0.65;
        let (out_tone, _) = gpu
            .process_buffer_with_alpha_and_retouch_f32(&buf, &r_tone, None, &[], None)
            .expect("basic tone");
        assert_eq!(out_tone.dimensions(), (640, 427));

        let mut r_heal = DevelopRecipe::default();
        r_heal.retouch.heal_spots.push(HealSpot {
            id: "h1".into(),
            mode: HealSpotMode::Heal,
            target_x: 0.45,
            target_y: 0.46,
            source_x: 0.58,
            source_y: 0.46,
            radius: 0.035,
            feather: 0.55,
            opacity: 0.95,
            points: vec![[0.45, 0.46]],
        });
        gpu.process_buffer_with_alpha_and_retouch_f32(&buf, &r_heal, None, &[], None)
            .expect("heal pass");

        let mut r_port = DevelopRecipe::default();
        r_port.retouch.portrait = PortraitRetouch {
            enabled: true,
            skin_smoothing: 60,
            skin_texture_preserve: 70,
            skin_tone_evenness: 40,
            shine_removal: 30,
            under_eye_brighten: 35,
            eye_enhance: 30,
            teeth_whiten: 45,
            lip_enhance: 25,
        };
        gpu.process_buffer_with_alpha_and_retouch_f32(&buf, &r_port, None, &[guide], Some(&odd_fp))
            .expect("portrait pass with odd BiSeNet parse map");

        let mut r_inp = DevelopRecipe::default();
        r_inp.retouch.inpaint_patches.push(patch.clone());
        r_inp.crop_transform.rect = NormalizedRect::new(0.05, 0.05, 0.9, 0.9).unwrap();
        gpu.process_buffer_with_alpha_and_retouch_f32(&buf, &r_inp, None, &[], None)
            .expect("inpaint pass");
    }

    // 2. Concurrent multi-thread + Rayon `par_iter` 32-bit JPEG thumbnail generation alongside
    //    live DirectML YuNet inference (mirrors `render_32bit_thumbnails` during Retouch editing).
    let tmp_dir = std::env::temp_dir().join(format!(
        "kroma_32bit_thumb_verify_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let src_jpg = tmp_dir.join("sample_portrait.jpg");
    DynamicImage::ImageRgb8(RgbImage::from_fn(1200, 801, |x, y| {
        Rgb([
            (160 + (x % 60)) as u8,
            (120 + (y % 50)) as u8,
            (100 + ((x + y) % 40)) as u8,
        ])
    }))
    .save(&src_jpg)
    .expect("save sample jpg");

    let yunet_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .join("models")
        .join("face_detection_yunet_2026may.onnx");
    let yunet_thread = if yunet_path.exists() {
        let det = YuNetOnnxDetector::from_model_file(&yunet_path, YuNetConfig::default())
            .expect("load YuNet");
        Some(std::thread::spawn(move || {
            let img = RgbImage::from_pixel(640, 427, Rgb([170, 140, 120]));
            for _ in 0..3 {
                det.detect_faces_in_rgb(&img)
                    .expect("concurrent YuNet inference");
            }
        }))
    } else {
        None
    };

    let mut handles = Vec::new();
    for w_idx in 0..4usize {
        let src_str = src_jpg.to_string_lossy().to_string();
        let out_dir = tmp_dir.clone();
        let patch_clone = patch.clone();
        let fp_clone = odd_fp.clone();
        handles.push(std::thread::spawn(move || {
            let engine = RayonImageDevelopEngine::new();
            let jobs: Vec<(usize, DevelopRecipe)> = (0..6)
                .map(|j| {
                    let mut r = DevelopRecipe::default();
                    r.retouch.portrait.enabled = true;
                    r.retouch.portrait.skin_smoothing = ((w_idx * 15 + j * 10) % 90 + 10) as u8;
                    r.retouch.inpaint_patches.push(patch_clone.clone());
                    (j, r)
                })
                .collect();
            jobs.par_iter().for_each(|(j, recipe)| {
                let decoded = decode_develop_buffer(&src_str, 640).expect("decode");
                let dest = out_dir.join(format!("thumb_w{w_idx}_j{j}.jpg"));
                engine
                    .render_buffer_to_jpeg_with_retouch(
                        &decoded,
                        recipe,
                        &[guide],
                        Some(&fp_clone),
                        &dest,
                        86,
                    )
                    .expect("render 32-bit thumb");
                assert!(dest.is_file());
            });
        }));
    }
    for h in handles {
        h.join().expect("worker panicked");
    }
    if let Some(yt) = yunet_thread {
        yt.join().expect("YuNet thread panicked");
    }
    let _ = std::fs::remove_dir_all(&tmp_dir);
}
