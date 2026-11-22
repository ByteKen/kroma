use crate::error::AppError;
use crate::ports::vision::{
    FaceDetectorPort, FaceRecognizerPort, RawFaceDetection, SlateOcrReading, SlateReaderPort,
};
use kroma_domain::identity::{
    parse_filename_capture_hints, AmbiguousMatchCandidate, CaptureIdentityHints, ClusterId,
    DetectedFace, GroupPhotoEntry, NoFacePhotoEntry, ShootOrganizationPlan, SimilarityThresholds,
    SlateMetadata, SubjectCluster,
};
use kroma_domain::photo::PhotoId;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhotoScanTarget {
    pub photo_id: PhotoId,
    pub source_path: String,
    /// Ordering timestamp (EXIF DateTimeOriginal -> filename timestamp -> file mtime fallback).
    pub capture_timestamp_epoch_sec: i64,
    /// True shutter capture time (EXIF DateTimeOriginal or parsed filename timestamp ONLY;
    /// `None` if only filesystem mtime is known so bulk-export mtimes never trigger false boosts).
    pub exact_capture_timestamp_epoch_sec: Option<i64>,
}

/// Pre-extracted face detections & embeddings for a single photo, enabling instant (`<0.1s`)
/// re-clustering from cached embeddings whenever the photographer adjusts Match strictness or Time Assist.
#[derive(Debug, Clone, PartialEq)]
pub struct PreExtractedPhotoScan {
    pub target: PhotoScanTarget,
    pub faces: Vec<DetectedFace>,
    pub slate_reading: Option<SlateOcrReading>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClusterShootInput {
    pub photos: Vec<PhotoScanTarget>,
    pub thresholds: SimilarityThresholds,
    /// Minimum ratio between a secondary face's area and the largest face's area
    /// to treat the frame as a multi-person group shot (filters tiny background bystanders!).
    pub group_face_min_area_ratio: f32,
    /// Enables conservative same-stem (`DSC03440.ARW` + `DSC03440.jpg`) and adjacent
    /// capture-timestamp / camera-sequence assist gated by a strict facial similarity floor.
    pub use_temporal_sequence_assist: bool,
}

impl ClusterShootInput {
    pub fn with_defaults(photos: Vec<PhotoScanTarget>) -> Self {
        Self {
            photos,
            thresholds: SimilarityThresholds::default(),
            group_face_min_area_ratio: 0.35,
            use_temporal_sequence_assist: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClusterScanFailure {
    pub photo_id: PhotoId,
    pub source_path: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClusterShootOutput {
    pub plan: ShootOrganizationPlan,
    pub scanned_photo_count: usize,
    pub failures: Vec<ClusterScanFailure>,
}

struct ScannedSingleFace {
    target: PhotoScanTarget,
    face: DetectedFace,
    slate_reading: Option<SlateOcrReading>,
    hints: CaptureIdentityHints,
    effective_capture_ts: Option<i64>,
}

struct ScannedGroupPhoto {
    target: PhotoScanTarget,
    faces: Vec<DetectedFace>,
}

struct WorkingGroup {
    members: Vec<usize>,
    slate_member_idx: Option<usize>,
    active: bool,
    version: u32,
    neighbors: HashSet<usize>,
}

#[derive(Debug, Clone, Copy)]
struct MergeCandidate {
    avg_eff: f32,
    a: usize,
    b: usize,
    ver_a: u32,
    ver_b: u32,
}

impl PartialEq for MergeCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.a == other.a
            && self.b == other.b
            && self.ver_a == other.ver_a
            && self.ver_b == other.ver_b
            && self.avg_eff.to_bits() == other.avg_eff.to_bits()
    }
}

impl Eq for MergeCandidate {}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.avg_eff
            .total_cmp(&other.avg_eff)
            // Deterministic tie-breaker on lower cluster indices
            .then_with(|| other.a.cmp(&self.a))
            .then_with(|| other.b.cmp(&self.b))
    }
}

/// Face-First Identity Clustering Use Case (`YuNet` + `ArcFace / SFace`, with **OPTIONAL** `SlateReaderPort`).
///
/// Uses **Drift-Free Average + Complete-Linkage Agglomerative Clustering** accelerated by a
/// mathematically exact sparse candidate max-heap (eliminating `O(N³)` scans on 2,000+ photo shoots),
/// plus optional conservative RAW+Edited stem pairing and shutter-time / camera-sequence assist.
pub struct ClusterShootFacesUseCase<'a, D: FaceDetectorPort, R: FaceRecognizerPort> {
    detector: &'a D,
    recognizer: &'a R,
    slate_reader: Option<&'a dyn SlateReaderPort>,
}

impl<'a, D: FaceDetectorPort, R: FaceRecognizerPort> ClusterShootFacesUseCase<'a, D, R> {
    pub fn new(
        detector: &'a D,
        recognizer: &'a R,
        slate_reader: Option<&'a dyn SlateReaderPort>,
    ) -> Self {
        Self {
            detector,
            recognizer,
            slate_reader,
        }
    }

    pub fn execute(&self, mut input: ClusterShootInput) -> Result<ClusterShootOutput, AppError> {
        if input.photos.is_empty() {
            return Err(AppError::InvalidInput(
                "at least one photo is required for face clustering".to_string(),
            ));
        }

        // Chronological order ensures deterministic Person_001..N numbering
        input.photos.sort_by_key(|p| {
            (
                p.exact_capture_timestamp_epoch_sec
                    .unwrap_or(p.capture_timestamp_epoch_sec),
                p.photo_id.as_str().to_string(),
            )
        });

        let scanned_photo_count = input.photos.len();
        let mut failures = Vec::new();
        let mut extracted_scans: Vec<PreExtractedPhotoScan> =
            Vec::with_capacity(scanned_photo_count);

        // Pass 1: Detect faces (YuNet), extract L2-normalized embeddings (ArcFace/SFace), and optionally check Slates
        for target in input.photos {
            let detections = match self.detector.detect_faces(&target.source_path) {
                Ok(d) => d,
                Err(err) => {
                    failures.push(ClusterScanFailure {
                        photo_id: target.photo_id,
                        source_path: target.source_path,
                        reason: err.to_string(),
                    });
                    continue;
                }
            };

            if detections.is_empty() {
                extracted_scans.push(PreExtractedPhotoScan {
                    target,
                    faces: Vec::new(),
                    slate_reading: None,
                });
                continue;
            }

            let prominent = select_prominent_faces(detections, input.group_face_min_area_ratio);
            if prominent.len() == 1 {
                let det = prominent.into_iter().next().expect("single prominent face");
                match self.recognizer.extract_embedding(&target.source_path, &det) {
                    Ok(embedding) => {
                        let slate_reading = self
                            .slate_reader
                            .and_then(|reader| reader.read_slate(&target.source_path).ok().flatten());
                        extracted_scans.push(PreExtractedPhotoScan {
                            faces: vec![DetectedFace {
                                photo_id: target.photo_id.clone(),
                                bounding_box: det.bounding_box,
                                landmarks: det.landmarks,
                                detection_confidence: det.detection_confidence,
                                embedding,
                            }],
                            target,
                            slate_reading,
                        });
                    }
                    Err(err) => failures.push(ClusterScanFailure {
                        photo_id: target.photo_id,
                        source_path: target.source_path,
                        reason: err.to_string(),
                    }),
                }
            } else {
                let mut extracted = Vec::with_capacity(prominent.len());
                for det in prominent {
                    if let Ok(embedding) = self.recognizer.extract_embedding(&target.source_path, &det)
                    {
                        extracted.push(DetectedFace {
                            photo_id: target.photo_id.clone(),
                            bounding_box: det.bounding_box,
                            landmarks: det.landmarks,
                            detection_confidence: det.detection_confidence,
                            embedding,
                        });
                    }
                }
                if extracted.is_empty() {
                    failures.push(ClusterScanFailure {
                        photo_id: target.photo_id,
                        source_path: target.source_path,
                        reason: "failed to extract face embeddings for group photo".to_string(),
                    });
                } else {
                    extracted_scans.push(PreExtractedPhotoScan {
                        target,
                        faces: extracted,
                        slate_reading: None,
                    });
                }
            }
        }

        cluster_preextracted_scans(
            extracted_scans,
            scanned_photo_count,
            failures,
            input.thresholds,
            input.use_temporal_sequence_assist,
        )
    }
}

/// Pure, detail-free clustering entry point operating directly on cached/pre-extracted face scans.
/// Enables `<0.1s` interactive re-clustering on 2,000+ photos when adjusting Match strictness or Time Assist.
pub fn cluster_preextracted_scans(
    mut scans: Vec<PreExtractedPhotoScan>,
    scanned_photo_count: usize,
    failures: Vec<ClusterScanFailure>,
    thresholds: SimilarityThresholds,
    use_temporal_sequence_assist: bool,
) -> Result<ClusterShootOutput, AppError> {
    if scans.is_empty() && failures.is_empty() {
        return Err(AppError::InvalidInput(
            "at least one photo is required for face clustering".to_string(),
        ));
    }

    scans.sort_by_key(|s| {
        (
            s.target
                .exact_capture_timestamp_epoch_sec
                .unwrap_or(s.target.capture_timestamp_epoch_sec),
            s.target.photo_id.as_str().to_string(),
        )
    });

    let mut plan = ShootOrganizationPlan::default();
    let mut single_faces: Vec<ScannedSingleFace> = Vec::new();
    let mut group_candidates: Vec<ScannedGroupPhoto> = Vec::new();

    for scan in scans {
        if scan.faces.is_empty() {
            plan.no_face_photos.push(NoFacePhotoEntry {
                photo_id: scan.target.photo_id,
                source_path: scan.target.source_path,
            });
        } else if scan.faces.len() == 1 {
            let face = scan.faces.into_iter().next().expect("single face");
            let hints = parse_filename_capture_hints(&scan.target.source_path);
            let effective_capture_ts = scan
                .target
                .exact_capture_timestamp_epoch_sec
                .or(hints.filename_timestamp_epoch_sec);
            single_faces.push(ScannedSingleFace {
                target: scan.target,
                face,
                slate_reading: scan.slate_reading,
                hints,
                effective_capture_ts,
            });
        } else {
            group_candidates.push(ScannedGroupPhoto {
                target: scan.target,
                faces: scan.faces,
            });
        }
    }

    // Pass 2: Drift-free sparse-heap agglomerative clustering
    let cluster_member_indices = build_verified_and_ambiguous_clusters(
        &single_faces,
        thresholds,
        use_temporal_sequence_assist,
        &mut plan,
    )?;

    // Pass 3: Cross-reference faces inside multi-person group photos using honest pairwise similarity
    let min_group_pair_floor = (thresholds.verified_match_min - 0.12).max(0.25);
    for group in group_candidates {
        let detected_face_count = group.faces.len();
        let mut matched_cluster_ids = Vec::new();
        for face in &group.faces {
            if let Some((cluster_idx, avg_sim, min_sim)) =
                find_best_cluster_pairwise(&single_faces, &cluster_member_indices, face)
                && avg_sim >= thresholds.verified_match_min
                && min_sim >= min_group_pair_floor
            {
                let cid = plan.clusters[cluster_idx].id.clone();
                if !matched_cluster_ids.contains(&cid) {
                    matched_cluster_ids.push(cid);
                }
            }
        }
        plan.group_photos.push(GroupPhotoEntry {
            photo_id: group.target.photo_id,
            source_path: group.target.source_path,
            matched_cluster_ids,
            detected_face_count,
        });
    }

    Ok(ClusterShootOutput {
        plan,
        scanned_photo_count,
        failures,
    })
}

/// Computes exact pairwise cosine similarities (`raw_sim`) and optional conservative temporal/stem
/// affinity (`eff_sim`), then runs order-independent hierarchical agglomerative clustering via a
/// mathematically exact sparse candidate max-heap (only touching compatible neighbor clusters on
/// each merge — avoiding `O(N³)` all-pairs scans on 2,000+ photo shoots!).
fn build_verified_and_ambiguous_clusters(
    singles: &[ScannedSingleFace],
    thresholds: SimilarityThresholds,
    use_temporal_assist: bool,
    plan: &mut ShootOrganizationPlan,
) -> Result<Vec<Vec<usize>>, AppError> {
    let n = singles.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let verified_min = thresholds.verified_match_min;
    // Complete-linkage floor prevents A~B + B~C bridging when A and C look nothing alike
    let min_raw_pair_floor = (verified_min - 0.12).max(0.25);

    // Pack embeddings contiguously for SIMD auto-vectorized dot products
    let dim = singles[0].face.embedding.dim();
    let uniform_dim = singles.iter().all(|s| s.face.embedding.dim() == dim);
    let flat_embs: Vec<f32> = if uniform_dim {
        let mut buf = Vec::with_capacity(n * dim);
        for s in singles {
            buf.extend_from_slice(s.face.embedding.as_slice());
        }
        buf
    } else {
        Vec::new()
    };

    let mut raw_sim = vec![0.0_f32; n * n];
    let mut eff_sim = vec![0.0_f32; n * n];

    let mut groups: Vec<WorkingGroup> = (0..n)
        .map(|i| WorkingGroup {
            members: vec![i],
            slate_member_idx: singles[i].slate_reading.as_ref().map(|_| i),
            active: true,
            version: 0,
            neighbors: HashSet::new(),
        })
        .collect();

    let mut heap: BinaryHeap<MergeCandidate> = BinaryHeap::new();

    for i in 0..n {
        raw_sim[i * n + i] = 1.0;
        eff_sim[i * n + i] = 1.0;
        let emb_i = if uniform_dim {
            &flat_embs[i * dim..(i + 1) * dim]
        } else {
            singles[i].face.embedding.as_slice()
        };
        let i_has_slate = groups[i].slate_member_idx.is_some();

        for j in (i + 1)..n {
            let raw = if uniform_dim {
                dot_slice(emb_i, &flat_embs[j * dim..(j + 1) * dim])
            } else {
                singles[i]
                    .face
                    .embedding
                    .cosine_similarity(&singles[j].face.embedding)
            };
            let bonus = if use_temporal_assist {
                compute_temporal_and_stem_bonus(&singles[i], &singles[j], raw, verified_min)
            } else {
                0.0
            };
            let effective = (raw + bonus).clamp(-1.0, 0.99);
            raw_sim[i * n + j] = raw;
            raw_sim[j * n + i] = raw;
            eff_sim[i * n + j] = effective;
            eff_sim[j * n + i] = effective;

            if effective >= verified_min
                && raw >= min_raw_pair_floor
                && !(i_has_slate && groups[j].slate_member_idx.is_some())
            {
                groups[i].neighbors.insert(j);
                groups[j].neighbors.insert(i);
                heap.push(MergeCandidate {
                    avg_eff: effective,
                    a: i,
                    b: j,
                    ver_a: 0,
                    ver_b: 0,
                });
            }
        }
    }

    // Sparse-heap exact agglomerative merge: because avg_eff(A∪B, C) <= max(avg_eff(A,C), avg_eff(B,C)),
    // any future valid merge partner C for (A∪B) is guaranteed to lie inside neighbors(A) ∪ neighbors(B)!
    while let Some(cand) = heap.pop() {
        if !groups[cand.a].active
            || !groups[cand.b].active
            || groups[cand.a].version != cand.ver_a
            || groups[cand.b].version != cand.ver_b
        {
            continue;
        }
        if groups[cand.a].slate_member_idx.is_some() && groups[cand.b].slate_member_idx.is_some() {
            continue;
        }

        let keep = cand.a.min(cand.b);
        let drop_id = cand.a.max(cand.b);

        let dropped_members = std::mem::take(&mut groups[drop_id].members);
        let dropped_slate = groups[drop_id].slate_member_idx.take();
        let dropped_neighbors = std::mem::take(&mut groups[drop_id].neighbors);
        groups[drop_id].active = false;
        groups[drop_id].version += 1;

        if groups[keep].slate_member_idx.is_none() {
            groups[keep].slate_member_idx = dropped_slate;
        }
        groups[keep].members.extend(dropped_members);
        groups[keep].members.sort_unstable();
        groups[keep].version += 1;
        groups[keep].neighbors.remove(&drop_id);

        // Union candidate neighbors from keep and drop_id, updating back-references
        let mut candidate_neighbors: Vec<usize> = groups[keep]
            .neighbors
            .union(&dropped_neighbors)
            .copied()
            .filter(|&nbr| nbr != keep && nbr != drop_id && groups[nbr].active)
            .collect();
        candidate_neighbors.sort_unstable();

        let keep_version = groups[keep].version;
        let keep_has_slate = groups[keep].slate_member_idx.is_some();
        let mut next_neighbors = HashSet::new();

        for nbr in candidate_neighbors {
            groups[nbr].neighbors.remove(&drop_id);
            if keep_has_slate && groups[nbr].slate_member_idx.is_some() {
                groups[nbr].neighbors.remove(&keep);
                continue;
            }
            let (avg_eff, _) =
                group_linkage_stats(&groups[keep].members, &groups[nbr].members, &eff_sim, n);
            if avg_eff < verified_min {
                groups[nbr].neighbors.remove(&keep);
                continue;
            }
            let (_, min_raw) =
                group_linkage_stats(&groups[keep].members, &groups[nbr].members, &raw_sim, n);
            if min_raw >= min_raw_pair_floor {
                next_neighbors.insert(nbr);
                groups[nbr].neighbors.insert(keep);
                let (a, b, ver_a, ver_b) = if keep < nbr {
                    (keep, nbr, keep_version, groups[nbr].version)
                } else {
                    (nbr, keep, groups[nbr].version, keep_version)
                };
                heap.push(MergeCandidate {
                    avg_eff,
                    a,
                    b,
                    ver_a,
                    ver_b,
                });
            } else {
                groups[nbr].neighbors.remove(&keep);
            }
        }

        groups[keep].neighbors = next_neighbors;
    }

    let mut active_groups: Vec<WorkingGroup> = groups.into_iter().filter(|g| g.active).collect();

    // Order clusters deterministically: slate-seeded subjects first (chronologically), then
    // face-only subjects by earliest capture order.
    active_groups.sort_by_key(|g| {
        let is_non_slate = u8::from(g.slate_member_idx.is_none());
        let earliest = g.members.first().copied().unwrap_or(usize::MAX);
        (is_non_slate, earliest)
    });

    // Separate truly borderline singleton non-slate faces into the Ambiguous Review Queue
    let review_min = thresholds.ambiguous_review_min;
    let review_pair_floor = (review_min - 0.08).max(0.20);
    let has_review_band = review_min < verified_min;

    let mut finalized_groups: Vec<WorkingGroup> = Vec::with_capacity(active_groups.len());
    let mut deferred_singletons: Vec<WorkingGroup> = Vec::new();

    for group in active_groups {
        if has_review_band && group.slate_member_idx.is_none() && group.members.len() == 1 {
            deferred_singletons.push(group);
        } else {
            finalized_groups.push(group);
        }
    }

    let mut pending_ambiguous: Vec<(usize, usize, f32)> = Vec::new();
    for singleton in deferred_singletons {
        let idx = singleton.members[0];
        let best_target = finalized_groups
            .iter()
            .enumerate()
            .map(|(g_idx, grp)| {
                let (avg_eff, _) = group_linkage_stats(&[idx], &grp.members, &eff_sim, n);
                let (avg_raw, min_raw) = group_linkage_stats(&[idx], &grp.members, &raw_sim, n);
                (g_idx, avg_eff, avg_raw, min_raw)
            })
            .filter(|&(_, avg_eff, _, min_raw)| {
                avg_eff >= review_min && min_raw >= review_pair_floor
            })
            .max_by(|a, b| a.1.total_cmp(&b.1));

        if let Some((target_group_idx, _, avg_raw, _)) = best_target {
            pending_ambiguous.push((idx, target_group_idx, avg_raw));
        } else {
            finalized_groups.push(singleton);
        }
    }

    // Materialize domain SubjectClusters with medoid representative & honest raw facial scores
    let mut member_index_lists: Vec<Vec<usize>> = Vec::with_capacity(finalized_groups.len());
    let mut cluster_ids_by_group: Vec<ClusterId> = Vec::with_capacity(finalized_groups.len());
    let mut next_cluster_seq = 1usize;

    for group in finalized_groups {
        let rep_idx = group
            .slate_member_idx
            .unwrap_or_else(|| select_medoid_member(&group.members, &raw_sim, n));

        let generated_code = Some(format!("#{next_cluster_seq:03}"));
        let (cluster_id, display_name, subject_code, slate_meta) =
            if let Some(slate_idx) = group.slate_member_idx
                && let Some(slate) = &singles[slate_idx].slate_reading
            {
                let cid = ClusterId::new(format!("subject-{next_cluster_seq:03}"))?;
                next_cluster_seq += 1;
                let metadata = SlateMetadata {
                    slate_photo_id: singles[slate_idx].target.photo_id.clone(),
                    ocr_raw_text: slate.subject_name.clone(),
                    ocr_confidence: slate.ocr_confidence,
                    calibrated_white_balance_kelvin: None,
                };
                (
                    cid,
                    slate.subject_name.clone(),
                    generated_code,
                    Some(metadata),
                )
            } else {
                let cid = ClusterId::new(format!("person-{next_cluster_seq:03}"))?;
                let name = format!("Person {next_cluster_seq:03}");
                next_cluster_seq += 1;
                (cid, name, generated_code, None)
            };

        let mut cluster = SubjectCluster::new_from_face(
            cluster_id.clone(),
            display_name,
            subject_code,
            singles[rep_idx].face.clone(),
            singles[rep_idx].target.source_path.clone(),
            slate_meta,
        )?;

        for &member_idx in &group.members {
            if member_idx == rep_idx {
                continue;
            }
            let honest_score = honest_member_similarity(member_idx, &group.members, &raw_sim, n);
            cluster.add_verified_match(
                singles[member_idx].face.clone(),
                singles[member_idx].target.source_path.clone(),
                honest_score,
                false,
            );
        }

        cluster_ids_by_group.push(cluster_id);
        member_index_lists.push(group.members);
        plan.clusters.push(cluster);
    }

    for (photo_idx, target_group_idx, score) in pending_ambiguous {
        let scanned = &singles[photo_idx];
        plan.ambiguous_review_queue.push(AmbiguousMatchCandidate {
            photo_id: scanned.target.photo_id.clone(),
            source_path: scanned.target.source_path.clone(),
            candidate_cluster_id: cluster_ids_by_group[target_group_idx].clone(),
            cosine_similarity: score,
            face_bounding_box: scanned.face.bounding_box,
            landmarks: scanned.face.landmarks,
        });
    }

    Ok(member_index_lists)
}

#[inline]
fn dot_slice(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x * y)
        .sum::<f32>()
        .clamp(-1.0, 1.0)
}

/// Computes a conservative, facial-floor-gated affinity boost from:
/// 1. Identical canonical camera stem (`RAW/DSC03440.ARW` <-> `EDITED/DSC03440.jpg`): `+0.20`
///    (requires `raw_sim >= 0.30` so unrelated frames from different days never collide).
/// 2. Tight shutter-time (`<= 15s` / `<= 45s`) or adjacent camera counter (`<= 2` / `<= 4` frames):
///    `+0.08` / `+0.05`, ONLY when `raw_sim >= max(verified_min - 0.10, 0.34)` so consecutive
///    strangers standing next in line (`0.10–0.32`) receive zero bonus.
fn compute_temporal_and_stem_bonus(
    a: &ScannedSingleFace,
    b: &ScannedSingleFace,
    raw_sim: f32,
    verified_min: f32,
) -> f32 {
    if !a.hints.canonical_stem.is_empty()
        && a.hints.canonical_stem == b.hints.canonical_stem
        && raw_sim >= 0.30
    {
        return 0.20;
    }

    let facial_safety_floor = (verified_min - 0.10).max(0.34);
    if raw_sim < facial_safety_floor {
        return 0.0;
    }

    let time_gap_sec = match (a.effective_capture_ts, b.effective_capture_ts) {
        (Some(ta), Some(tb)) => Some(ta.abs_diff(tb)),
        _ => None,
    };

    let seq_gap = match (
        &a.hints.camera_prefix,
        a.hints.camera_sequence,
        &b.hints.camera_prefix,
        b.hints.camera_sequence,
    ) {
        (pa, Some(sa), pb, Some(sb)) if pa == pb => Some(sa.abs_diff(sb)),
        _ => None,
    };

    if time_gap_sec.is_some_and(|dt| dt <= 15) || seq_gap.is_some_and(|ds| ds <= 2) {
        0.08
    } else if time_gap_sec.is_some_and(|dt| dt <= 45) || seq_gap.is_some_and(|ds| ds <= 4) {
        0.05
    } else {
        0.0
    }
}

fn group_linkage_stats(a: &[usize], b: &[usize], sim: &[f32], stride: usize) -> (f32, f32) {
    let mut sum = 0.0_f32;
    let mut min_s = f32::INFINITY;
    let mut count = 0usize;

    for &i in a {
        let row = i * stride;
        for &j in b {
            let s = sim[row + j];
            sum += s;
            if s < min_s {
                min_s = s;
            }
            count += 1;
        }
    }

    if count == 0 {
        (0.0, 0.0)
    } else {
        (sum / (count as f32), min_s)
    }
}

/// Selects the cluster medoid: the member with the highest honest average pairwise similarity
/// to all other members of the cluster (breaking ties by earliest chronological index).
fn select_medoid_member(members: &[usize], sim: &[f32], stride: usize) -> usize {
    if members.len() <= 1 {
        return members[0];
    }
    let mut best_idx = members[0];
    let mut best_avg = f32::NEG_INFINITY;

    for &candidate in members {
        let avg = honest_member_similarity(candidate, members, sim, stride);
        if avg > best_avg + 1e-6 {
            best_avg = avg;
            best_idx = candidate;
        }
    }
    best_idx
}

fn honest_member_similarity(member: usize, members: &[usize], sim: &[f32], stride: usize) -> f32 {
    if members.len() <= 1 {
        return 1.0;
    }
    let row = member * stride;
    let mut sum = 0.0_f32;
    let mut count = 0usize;
    for &other in members {
        if other != member {
            sum += sim[row + other];
            count += 1;
        }
    }
    if count == 0 {
        1.0
    } else {
        (sum / (count as f32)).clamp(-1.0, 1.0)
    }
}

fn find_best_cluster_pairwise(
    singles: &[ScannedSingleFace],
    cluster_members: &[Vec<usize>],
    face: &DetectedFace,
) -> Option<(usize, f32, f32)> {
    cluster_members
        .iter()
        .enumerate()
        .filter(|(_, members)| !members.is_empty())
        .map(|(cluster_idx, members)| {
            let mut sum = 0.0_f32;
            let mut min_s = f32::INFINITY;
            for &m in members {
                let s = singles[m].face.embedding.cosine_similarity(&face.embedding);
                sum += s;
                if s < min_s {
                    min_s = s;
                }
            }
            (cluster_idx, sum / (members.len() as f32), min_s)
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
}

pub fn select_prominent_faces(
    mut detections: Vec<RawFaceDetection>,
    min_area_ratio: f32,
) -> Vec<RawFaceDetection> {
    detections.sort_by(|a, b| {
        b.bounding_box
            .area()
            .total_cmp(&a.bounding_box.area())
            .then_with(|| b.detection_confidence.total_cmp(&a.detection_confidence))
    });

    let Some(largest_area) = detections.first().map(|d| d.bounding_box.area()) else {
        return Vec::new();
    };
    if largest_area <= f32::EPSILON {
        return detections.into_iter().take(1).collect();
    }

    let ratio_floor = min_area_ratio.clamp(0.05, 1.0);
    detections
        .into_iter()
        .filter(|d| (d.bounding_box.area() / largest_area) >= ratio_floor)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kroma_domain::identity::{BoundingBox, FaceEmbedding, Point2D, YuNetLandmarks};
    use std::collections::HashMap;

    struct StubDetectorRecognizer {
        embeddings: HashMap<String, FaceEmbedding>,
    }

    impl FaceDetectorPort for StubDetectorRecognizer {
        fn detect_faces(&self, source_path: &str) -> Result<Vec<RawFaceDetection>, AppError> {
            if !self.embeddings.contains_key(source_path) {
                return Ok(Vec::new());
            }
            let p = Point2D { x: 0.5, y: 0.5 };
            Ok(vec![RawFaceDetection {
                bounding_box: BoundingBox {
                    x: 0.2,
                    y: 0.2,
                    width: 0.4,
                    height: 0.4,
                },
                landmarks: YuNetLandmarks {
                    right_eye: p,
                    left_eye: p,
                    nose_tip: p,
                    right_mouth_corner: p,
                    left_mouth_corner: p,
                },
                detection_confidence: 0.99,
            }])
        }
    }

    impl FaceRecognizerPort for StubDetectorRecognizer {
        fn extract_embedding(
            &self,
            source_path: &str,
            _detection: &RawFaceDetection,
        ) -> Result<FaceEmbedding, AppError> {
            self.embeddings
                .get(source_path)
                .cloned()
                .ok_or_else(|| AppError::NotFound(source_path.to_string()))
        }
    }

    #[test]
    fn prevents_running_centroid_snowball_drift_across_homogeneous_batch() {
        let mut embeddings = HashMap::new();
        let mut targets = Vec::new();

        let shared = 0.55_f32;
        let unique = (1.0 - shared * shared).sqrt();

        for person_idx in 0..12usize {
            let mut raw = vec![0.0_f32; 64];
            raw[0] = shared;
            raw[person_idx + 1] = unique;
            let emb = FaceEmbedding::from_vec(raw).unwrap();
            let path = format!("student_{person_idx:02}_a.arw");
            embeddings.insert(path.clone(), emb);
            targets.push(PhotoScanTarget {
                photo_id: PhotoId::new(format!("S{person_idx:02}A")).unwrap(),
                source_path: path,
                capture_timestamp_epoch_sec: (person_idx as i64) * 120,
                exact_capture_timestamp_epoch_sec: Some((person_idx as i64) * 120),
            });
        }

        let mut s0_second = vec![0.0_f32; 64];
        s0_second[0] = shared;
        s0_second[1] = unique * 0.92;
        s0_second[40] = unique * 0.39;
        embeddings.insert(
            "student_00_b.arw".to_string(),
            FaceEmbedding::from_vec(s0_second).unwrap(),
        );
        targets.push(PhotoScanTarget {
            photo_id: PhotoId::new("S00B").unwrap(),
            source_path: "student_00_b.arw".to_string(),
            capture_timestamp_epoch_sec: 5,
            exact_capture_timestamp_epoch_sec: Some(5),
        });

        let stub = StubDetectorRecognizer { embeddings };
        let uc = ClusterShootFacesUseCase::new(&stub, &stub, None);
        let out = uc
            .execute(ClusterShootInput::with_defaults(targets))
            .unwrap();

        assert_eq!(out.plan.clusters.len(), 12);
        assert!(out.plan.ambiguous_review_queue.is_empty());
        let s0_cluster = out
            .plan
            .clusters
            .iter()
            .find(|c| c.entries.iter().any(|e| e.photo_id.as_str() == "S00A"))
            .unwrap();
        assert_eq!(s0_cluster.entries.len(), 2);
        assert!(s0_cluster.entries.iter().any(|e| e.photo_id.as_str() == "S00B"));
    }

    #[test]
    fn prevents_single_linkage_chaining_between_dissimilar_faces() {
        let mut embeddings = HashMap::new();
        let mut a = vec![0.0_f32; 16];
        a[0] = 1.0;
        let mut c = vec![0.0_f32; 16];
        c[1] = 1.0;
        let mut b = vec![0.0_f32; 16];
        b[0] = 1.0;
        b[1] = 1.0;

        embeddings.insert("a.arw".to_string(), FaceEmbedding::from_vec(a).unwrap());
        embeddings.insert("b.arw".to_string(), FaceEmbedding::from_vec(b).unwrap());
        embeddings.insert("c.arw".to_string(), FaceEmbedding::from_vec(c).unwrap());

        let targets = vec![
            PhotoScanTarget {
                photo_id: PhotoId::new("A").unwrap(),
                source_path: "a.arw".to_string(),
                capture_timestamp_epoch_sec: 1,
                exact_capture_timestamp_epoch_sec: None,
            },
            PhotoScanTarget {
                photo_id: PhotoId::new("B").unwrap(),
                source_path: "b.arw".to_string(),
                capture_timestamp_epoch_sec: 200,
                exact_capture_timestamp_epoch_sec: None,
            },
            PhotoScanTarget {
                photo_id: PhotoId::new("C").unwrap(),
                source_path: "c.arw".to_string(),
                capture_timestamp_epoch_sec: 400,
                exact_capture_timestamp_epoch_sec: None,
            },
        ];

        let stub = StubDetectorRecognizer { embeddings };
        let uc = ClusterShootFacesUseCase::new(&stub, &stub, None);
        let out = uc
            .execute(ClusterShootInput::with_defaults(targets))
            .unwrap();

        assert!(
            out.plan
                .clusters
                .iter()
                .all(|cluster| !(cluster.entries.iter().any(|e| e.photo_id.as_str() == "A")
                    && cluster.entries.iter().any(|e| e.photo_id.as_str() == "C"))),
            "A and C (cosine 0.0) must never be transitively bridged into one subject"
        );
    }

    #[test]
    fn temporal_sequence_and_same_stem_assist_matches_borderline_student_without_merging_stranger() {
        let mut embeddings = HashMap::new();

        let mut student_a_slate = vec![0.0_f32; 16];
        student_a_slate[0] = 1.0;
        let mut student_a_smile = vec![0.0_f32; 16];
        student_a_smile[0] = 0.40;
        student_a_smile[1] = (1.0_f32 - 0.16).sqrt();
        let mut student_a_edited = vec![0.0_f32; 16];
        student_a_edited[0] = 0.85;
        student_a_edited[2] = (1.0_f32 - 0.85 * 0.85).sqrt();
        let mut student_b = vec![0.0_f32; 16];
        student_b[0] = 0.25;
        student_b[3] = (1.0_f32 - 0.0625).sqrt();

        embeddings.insert(
            "RAW/DSC03512.ARW".to_string(),
            FaceEmbedding::from_vec(student_a_slate).unwrap(),
        );
        embeddings.insert(
            "RAW/DSC03513.ARW".to_string(),
            FaceEmbedding::from_vec(student_a_smile).unwrap(),
        );
        embeddings.insert(
            "EDITED/DSC03512_edited.jpg".to_string(),
            FaceEmbedding::from_vec(student_a_edited).unwrap(),
        );
        embeddings.insert(
            "RAW/DSC03514.ARW".to_string(),
            FaceEmbedding::from_vec(student_b).unwrap(),
        );

        let targets = vec![
            PhotoScanTarget {
                photo_id: PhotoId::new("RAW_DSC03512").unwrap(),
                source_path: "RAW/DSC03512.ARW".to_string(),
                capture_timestamp_epoch_sec: 1_700_001_000,
                exact_capture_timestamp_epoch_sec: Some(1_700_001_000),
            },
            PhotoScanTarget {
                photo_id: PhotoId::new("RAW_DSC03513").unwrap(),
                source_path: "RAW/DSC03513.ARW".to_string(),
                capture_timestamp_epoch_sec: 1_700_001_004,
                exact_capture_timestamp_epoch_sec: Some(1_700_001_004),
            },
            PhotoScanTarget {
                photo_id: PhotoId::new("EDITED_DSC03512").unwrap(),
                source_path: "EDITED/DSC03512_edited.jpg".to_string(),
                capture_timestamp_epoch_sec: 1_700_001_000,
                exact_capture_timestamp_epoch_sec: Some(1_700_001_000),
            },
            PhotoScanTarget {
                photo_id: PhotoId::new("RAW_DSC03514").unwrap(),
                source_path: "RAW/DSC03514.ARW".to_string(),
                capture_timestamp_epoch_sec: 1_700_001_012,
                exact_capture_timestamp_epoch_sec: Some(1_700_001_012),
            },
        ];

        let stub = StubDetectorRecognizer { embeddings };
        let uc = ClusterShootFacesUseCase::new(&stub, &stub, None);
        let out = uc
            .execute(ClusterShootInput::with_defaults(targets))
            .unwrap();

        assert_eq!(out.plan.clusters.len(), 2);
        let cluster_a = out
            .plan
            .clusters
            .iter()
            .find(|c| c.entries.iter().any(|e| e.photo_id.as_str() == "RAW_DSC03512"))
            .unwrap();
        assert_eq!(cluster_a.entries.len(), 3);
        assert!(cluster_a.entries.iter().any(|e| e.photo_id.as_str() == "RAW_DSC03513"));
        assert!(cluster_a.entries.iter().any(|e| e.photo_id.as_str() == "EDITED_DSC03512"));
        assert!(!cluster_a.entries.iter().any(|e| e.photo_id.as_str() == "RAW_DSC03514"));
    }

    #[test]
    fn sparse_heap_clusters_large_shoot_from_preextracted_scans_accurately() {
        // 100 distinct subjects x 4 frames each = 400 photos (512-D embeddings)
        let p = Point2D { x: 0.5, y: 0.5 };
        let lm = YuNetLandmarks {
            right_eye: p,
            left_eye: p,
            nose_tip: p,
            right_mouth_corner: p,
            left_mouth_corner: p,
        };
        let bbox = BoundingBox {
            x: 0.2,
            y: 0.2,
            width: 0.3,
            height: 0.3,
        };

        let mut scans = Vec::with_capacity(400);
        for subj in 0..100usize {
            for shot in 0..4usize {
                let mut raw = vec![0.0_f32; 512];
                raw[subj] = 0.95;
                raw[100 + ((subj * 7 + shot) % 400)] = 0.15;
                let pid = PhotoId::new(format!("P{subj:03}_{shot}")).unwrap();
                scans.push(PreExtractedPhotoScan {
                    target: PhotoScanTarget {
                        photo_id: pid.clone(),
                        source_path: format!("DSC{:05}.ARW", subj * 10 + shot),
                        capture_timestamp_epoch_sec: (subj * 100 + shot * 3) as i64,
                        exact_capture_timestamp_epoch_sec: Some((subj * 100 + shot * 3) as i64),
                    },
                    faces: vec![DetectedFace {
                        photo_id: pid,
                        bounding_box: bbox,
                        landmarks: lm,
                        detection_confidence: 0.98,
                        embedding: FaceEmbedding::from_vec(raw).unwrap(),
                    }],
                    slate_reading: None,
                });
            }
        }

        let out = cluster_preextracted_scans(
            scans,
            400,
            Vec::new(),
            SimilarityThresholds::default(),
            true,
        )
        .unwrap();
        assert_eq!(out.plan.clusters.len(), 100);
        assert!(out.plan.clusters.iter().all(|c| c.entries.len() == 4));
    }
}
