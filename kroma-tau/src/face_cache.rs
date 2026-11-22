use image::RgbImage;
use kroma_app::use_cases::cluster_shoot::{
    select_prominent_faces, ClusterScanFailure, PhotoScanTarget, PreExtractedPhotoScan,
};
use kroma_domain::identity::{BoundingBox, DetectedFace, FaceEmbedding, Point2D, YuNetLandmarks};
use kroma_infra::develop::image_decoder::decode_fast_preview_max_dim;
use kroma_infra::vision::face_align::align_face_112;
use kroma_infra::vision::sface_recognizer::SFaceOnnxRecognizer;
use kroma_infra::vision::yunet_detector::{YuNetConfig, YuNetOnnxDetector};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::UNIX_EPOCH;

const FACE_SCAN_MAX_DIM: u32 = 1024;
const ARCFACE_BATCH_SIZE: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedFaceDto {
    bbox: [f32; 4],
    landmarks: [f32; 10],
    confidence: f32,
    embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedPhotoEntryDto {
    file_size: u64,
    mtime_sec: i64,
    faces: Vec<CachedFaceDto>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct DiskFaceCacheDto {
    version: u32,
    model_key: String,
    entries: HashMap<String, CachedPhotoEntryDto>,
}

pub struct ParallelFaceScanOutcome {
    pub scans: Vec<PreExtractedPhotoScan>,
    pub failures: Vec<ClusterScanFailure>,
    pub cache_hits: usize,
    pub provider_label: &'static str,
}

/// Loads cached face detections & embeddings from disk for unchanged files, then runs a
/// 2-stage parallel pipeline on any uncached photos:
/// 1. Rayon worker pool decodes (`<=1024px`), runs YuNet 2026, and crops `112x112` aligned faces in parallel.
/// 2. Batched ArcFace/SFace (`B=16`, DirectML GPU with CPU fallback) extracts embeddings in bulk.
#[allow(clippy::too_many_arguments)]
pub fn load_or_extract_face_scans<F>(
    folder_path: &str,
    scan_targets: &[PhotoScanTarget],
    source_to_preview: &HashMap<String, String>,
    yunet_path: &Path,
    recognizer_path: &Path,
    preview_cache_dir: &Path,
    group_face_min_area_ratio: f32,
    on_progress: F,
) -> Result<ParallelFaceScanOutcome, String>
where
    F: Fn(usize, usize, String) + Sync,
{
    let total = scan_targets.len();
    let model_key = model_fingerprint_key(recognizer_path);
    let cache_file = disk_cache_path(preview_cache_dir, folder_path, &model_key);
    let mut disk_cache = load_disk_cache(&cache_file, &model_key);

    let mut scans: Vec<PreExtractedPhotoScan> = Vec::with_capacity(total);
    let mut misses: Vec<(PhotoScanTarget, u64, i64)> = Vec::new();
    let mut cache_hits = 0usize;

    for target in scan_targets {
        let (file_size, mtime_sec) = file_fingerprint(&target.source_path);
        if let Some(entry) = disk_cache.entries.get(&target.source_path)
            && entry.file_size == file_size
            && entry.mtime_sec == mtime_sec
            && let Some(faces) = hydrate_cached_faces(&target.photo_id, &entry.faces)
        {
            scans.push(PreExtractedPhotoScan {
                target: target.clone(),
                faces,
                slate_reading: None,
            });
            cache_hits += 1;
            continue;
        }
        misses.push((target.clone(), file_size, mtime_sec));
    }

    if misses.is_empty() {
        on_progress(
            total,
            total,
            format!("Loaded {cache_hits} cached face embeddings instantaneously"),
        );
        return Ok(ParallelFaceScanOutcome {
            scans,
            failures: Vec::new(),
            cache_hits,
            provider_label: "Cache",
        });
    }

    let miss_count = misses.len();
    on_progress(
        cache_hits,
        total,
        format!("Detecting faces across {miss_count} photo(s) in parallel..."),
    );

    // Stage 1: Parallel decode + YuNet detection + 112x112 face alignment across CPU cores
    let detected_counter = AtomicUsize::new(cache_hits);
    let yunet_path_buf = yunet_path.to_path_buf();

    struct PendingAlignedPhoto {
        target: PhotoScanTarget,
        file_size: u64,
        mtime_sec: i64,
        aligned_faces: Vec<(BoundingBox, YuNetLandmarks, f32, RgbImage)>,
    }

    enum Stage1Result {
        Ready(PendingAlignedPhoto),
        Failed(ClusterScanFailure),
    }

    let stage1_results: Vec<Stage1Result> = misses
        .into_par_iter()
        .map_init(
            || {
                YuNetOnnxDetector::from_model_file(
                    &yunet_path_buf,
                    YuNetConfig {
                        intra_threads: 1,
                        ..YuNetConfig::default()
                    },
                )
            },
            |detector_res, (target, file_size, mtime_sec)| {
                let done = detected_counter.fetch_add(1, Ordering::Relaxed) + 1;
                if done == 1 || done == total || done.is_multiple_of(8) {
                    on_progress(
                        done,
                        total,
                        format!("Detecting faces (YuNet parallel): {done} / {total}"),
                    );
                }

                let detector = match detector_res.as_ref() {
                    Ok(d) => d,
                    Err(err) => {
                        return Stage1Result::Failed(ClusterScanFailure {
                            photo_id: target.photo_id,
                            source_path: target.source_path,
                            reason: err.to_string(),
                        });
                    }
                };

                let fast_path = source_to_preview
                    .get(&target.source_path)
                    .map(String::as_str)
                    .unwrap_or(&target.source_path);
                let rgb = match decode_fast_preview_max_dim(fast_path, FACE_SCAN_MAX_DIM) {
                    Ok(img) => img,
                    Err(err) => {
                        return Stage1Result::Failed(ClusterScanFailure {
                            photo_id: target.photo_id,
                            source_path: target.source_path,
                            reason: err.to_string(),
                        });
                    }
                };

                let detections = match detector.detect_faces_in_rgb(&rgb) {
                    Ok(d) => d,
                    Err(err) => {
                        return Stage1Result::Failed(ClusterScanFailure {
                            photo_id: target.photo_id,
                            source_path: target.source_path,
                            reason: err.to_string(),
                        });
                    }
                };

                if detections.is_empty() {
                    return Stage1Result::Ready(PendingAlignedPhoto {
                        target,
                        file_size,
                        mtime_sec,
                        aligned_faces: Vec::new(),
                    });
                }

                let prominent = select_prominent_faces(detections, group_face_min_area_ratio);
                let mut aligned_faces = Vec::with_capacity(prominent.len());
                for det in prominent {
                    if let Some(crop) = align_face_112(&rgb, &det.landmarks) {
                        aligned_faces.push((
                            det.bounding_box,
                            det.landmarks,
                            det.detection_confidence,
                            crop,
                        ));
                    }
                }

                Stage1Result::Ready(PendingAlignedPhoto {
                    target,
                    file_size,
                    mtime_sec,
                    aligned_faces,
                })
            },
        )
        .collect();

    let mut failures = Vec::new();
    let mut photos_with_crops: Vec<PendingAlignedPhoto> = Vec::new();

    for res in stage1_results {
        match res {
            Stage1Result::Failed(f) => failures.push(f),
            Stage1Result::Ready(pending) => {
                if pending.aligned_faces.is_empty() {
                    disk_cache.entries.insert(
                        pending.target.source_path.clone(),
                        CachedPhotoEntryDto {
                            file_size: pending.file_size,
                            mtime_sec: pending.mtime_sec,
                            faces: Vec::new(),
                        },
                    );
                    scans.push(PreExtractedPhotoScan {
                        target: pending.target,
                        faces: Vec::new(),
                        slate_reading: None,
                    });
                } else {
                    photos_with_crops.push(pending);
                }
            }
        }
    }

    // Stage 2: Batched ArcFace / SFace inference (DirectML GPU with automatic CPU fallback)
    let cpu_threads = std::thread::available_parallelism()
        .map(|n| n.get().clamp(2, 8))
        .unwrap_or(4);
    let recognizer =
        SFaceOnnxRecognizer::from_model_file(recognizer_path, cpu_threads).map_err(|e| e.to_string())?;
    let provider_label = recognizer.execution_provider_label();
    let model_label = recognizer.model_label();

    // Flatten all face crops across photos so we can saturate GPU/SIMD in full B=16 batches
    let total_crops: usize = photos_with_crops.iter().map(|p| p.aligned_faces.len()).sum();
    let mut flat_crops: Vec<RgbImage> = Vec::with_capacity(total_crops);
    for p in &photos_with_crops {
        for (_, _, _, crop) in &p.aligned_faces {
            flat_crops.push(crop.clone());
        }
    }

    let mut flat_embeddings: Vec<FaceEmbedding> = Vec::with_capacity(total_crops);
    let mut processed_crops = 0usize;
    for chunk in flat_crops.chunks(ARCFACE_BATCH_SIZE) {
        let batch_out = recognizer
            .extract_embeddings_batch_from_aligned(chunk)
            .map_err(|e| e.to_string())?;
        flat_embeddings.extend(batch_out);
        processed_crops += chunk.len();
        on_progress(
            processed_crops,
            total_crops.max(1),
            format!(
                "Extracting {model_label} ({provider_label}): {processed_crops} / {total_crops}"
            ),
        );
    }

    let mut emb_iter = flat_embeddings.into_iter();
    for pending in photos_with_crops {
        let mut detected_faces = Vec::with_capacity(pending.aligned_faces.len());
        let mut cached_faces = Vec::with_capacity(pending.aligned_faces.len());

        for (bbox, landmarks, confidence, _) in pending.aligned_faces {
            if let Some(embedding) = emb_iter.next() {
                cached_faces.push(CachedFaceDto {
                    bbox: [bbox.x, bbox.y, bbox.width, bbox.height],
                    landmarks: [
                        landmarks.right_eye.x,
                        landmarks.right_eye.y,
                        landmarks.left_eye.x,
                        landmarks.left_eye.y,
                        landmarks.nose_tip.x,
                        landmarks.nose_tip.y,
                        landmarks.right_mouth_corner.x,
                        landmarks.right_mouth_corner.y,
                        landmarks.left_mouth_corner.x,
                        landmarks.left_mouth_corner.y,
                    ],
                    confidence,
                    embedding: embedding.as_slice().to_vec(),
                });
                detected_faces.push(DetectedFace {
                    photo_id: pending.target.photo_id.clone(),
                    bounding_box: bbox,
                    landmarks,
                    detection_confidence: confidence,
                    embedding,
                });
            }
        }

        disk_cache.entries.insert(
            pending.target.source_path.clone(),
            CachedPhotoEntryDto {
                file_size: pending.file_size,
                mtime_sec: pending.mtime_sec,
                faces: cached_faces,
            },
        );
        scans.push(PreExtractedPhotoScan {
            target: pending.target,
            faces: detected_faces,
            slate_reading: None,
        });
    }

    let _ = save_disk_cache(&cache_file, &disk_cache);

    Ok(ParallelFaceScanOutcome {
        scans,
        failures,
        cache_hits,
        provider_label,
    })
}

fn hydrate_cached_faces(
    photo_id: &kroma_domain::photo::PhotoId,
    cached: &[CachedFaceDto],
) -> Option<Vec<DetectedFace>> {
    let mut out = Vec::with_capacity(cached.len());
    for c in cached {
        let embedding = FaceEmbedding::from_vec(c.embedding.clone()).ok()?;
        out.push(DetectedFace {
            photo_id: photo_id.clone(),
            bounding_box: BoundingBox {
                x: c.bbox[0],
                y: c.bbox[1],
                width: c.bbox[2],
                height: c.bbox[3],
            },
            landmarks: YuNetLandmarks {
                right_eye: Point2D {
                    x: c.landmarks[0],
                    y: c.landmarks[1],
                },
                left_eye: Point2D {
                    x: c.landmarks[2],
                    y: c.landmarks[3],
                },
                nose_tip: Point2D {
                    x: c.landmarks[4],
                    y: c.landmarks[5],
                },
                right_mouth_corner: Point2D {
                    x: c.landmarks[6],
                    y: c.landmarks[7],
                },
                left_mouth_corner: Point2D {
                    x: c.landmarks[8],
                    y: c.landmarks[9],
                },
            },
            detection_confidence: c.confidence,
            embedding,
        });
    }
    Some(out)
}

fn file_fingerprint(source_path: &str) -> (u64, i64) {
    let Ok(meta) = fs::metadata(source_path) else {
        return (0, 0);
    };
    let size = meta.len();
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    (size, mtime)
}

fn model_fingerprint_key(model_path: &Path) -> String {
    let name = model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model");
    let size = fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);
    format!("{name}:{size}")
}

fn disk_cache_path(cache_dir: &Path, folder_path: &str, model_key: &str) -> PathBuf {
    let mut h = DefaultHasher::new();
    folder_path.hash(&mut h);
    model_key.hash(&mut h);
    cache_dir.join(format!("face_cache_v1_{:016x}.json", h.finish()))
}

fn load_disk_cache(path: &Path, model_key: &str) -> DiskFaceCacheDto {
    if let Ok(bytes) = fs::read(path)
        && let Ok(parsed) = serde_json::from_slice::<DiskFaceCacheDto>(&bytes)
        && parsed.version == 1
        && parsed.model_key == model_key
    {
        return parsed;
    }
    DiskFaceCacheDto {
        version: 1,
        model_key: model_key.to_string(),
        entries: HashMap::new(),
    }
}

fn save_disk_cache(path: &Path, cache: &DiskFaceCacheDto) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let bytes = serde_json::to_vec(cache).map_err(|e| e.to_string())?;
    fs::write(path, bytes).map_err(|e| e.to_string())
}
