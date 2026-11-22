use crate::error::DomainError;

/// Rule-based export path template using subject name & optional slate ID tokens:
/// e.g. `/[Slate_Name]_[ID]/IMG_001.jpg` ->
///   - With slate ID `#104`: `/Sophia_Martinez_104/IMG_001.jpg`
///   - Slate-free (`None`):  `/Person_001/IMG_001.jpg` (automatically trims dangling `_`)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputNamingTemplate {
    pattern: String,
    exclude_slate_photo_from_client_folder: bool,
}

impl OutputNamingTemplate {
    pub fn new(
        pattern: impl Into<String>,
        exclude_slate_photo_from_client_folder: bool,
    ) -> Result<Self, DomainError> {
        let pattern = pattern.into();
        if pattern.trim().is_empty() {
            return Err(DomainError::EmptyOutputTemplate);
        }
        Ok(Self {
            pattern,
            exclude_slate_photo_from_client_folder,
        })
    }

    pub fn default_portrait_template() -> Self {
        Self {
            pattern: "[Slate_Name]/[Original_Name].jpg".to_string(),
            exclude_slate_photo_from_client_folder: true,
        }
    }

    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    pub fn exclude_slate_photo_from_client_folder(&self) -> bool {
        self.exclude_slate_photo_from_client_folder
    }

    /// Formats a clean relative output path, gracefully collapsing optional `[ID]` when absent.
    pub fn format_relative_path(
        &self,
        subject_name: &str,
        subject_code: Option<&str>,
        sequence_number: usize,
    ) -> String {
        self.format_relative_path_in_folder(&[], subject_name, subject_code, sequence_number)
    }

    /// Formats a relative output path with optional virtual folder segments (e.g. `["Engineering", "BSARC"]`).
    pub fn format_relative_path_in_folder(
        &self,
        folder_segments: &[String],
        subject_name: &str,
        subject_code: Option<&str>,
        sequence_number: usize,
    ) -> String {
        self.format_relative_path_for_file(
            folder_segments,
            subject_name,
            subject_code,
            sequence_number,
            None,
        )
    }

    /// Formats a relative output path with optional virtual folder segments and original file stem.
    /// - Supports `[Original_Name]` / `[Original]` to preserve original camera filenames (`DSC04123.jpg`).
    /// - If the pattern includes `[Folder]` or `[Department]`, substitutes the sanitized folder path
    ///   (and collapses it cleanly when unassigned).
    /// - Otherwise, if `folder_segments` is non-empty, prepends `<FolderPath>/` automatically so
    ///   virtual folders export directly without manual template tweaking.
    pub fn format_relative_path_for_file(
        &self,
        folder_segments: &[String],
        subject_name: &str,
        subject_code: Option<&str>,
        sequence_number: usize,
        original_stem: Option<&str>,
    ) -> String {
        let sanitized_name = sanitize_path_component(subject_name);
        let safe_name = if sanitized_name.is_empty() {
            "Unnamed_Subject".to_string()
        } else {
            sanitized_name
        };
        let sanitized_code = subject_code
            .map(sanitize_path_component)
            .filter(|s| !s.is_empty());
        let seq_padded = format!("{sequence_number:03}");
        let safe_original = original_stem
            .map(sanitize_path_component)
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| format!("IMG_{seq_padded}"));
        let folder_path: String = folder_segments
            .iter()
            .map(|seg| sanitize_path_component(seg))
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("/");

        let has_leading_slash = self.pattern.starts_with('/');
        let has_folder_token =
            self.pattern.contains("[Folder]") || self.pattern.contains("[Department]");

        // Replace sequence placeholders BEFORE substituting subject_name / subject_code / original_name
        // so a name like "Person_001" or filename "DSC_0001" is never altered by sequence numbers!
        let with_sequence = self
            .pattern
            .replace("[Seq]", &seq_padded)
            .replace("001", &seq_padded);

        let with_subject = match sanitized_code {
            Some(code) => with_sequence
                .replace("[Original_Name]", &safe_original)
                .replace("[Original]", &safe_original)
                .replace("[Slate_Name]", &safe_name)
                .replace("[Subject_Name]", &safe_name)
                .replace("[ID]", &code),
            None => with_sequence
                .replace("_[ID]", "")
                .replace("-[ID]", "")
                .replace("[ID]_", "")
                .replace("[ID]-", "")
                .replace("[ID]", "")
                .replace("[Original_Name]", &safe_original)
                .replace("[Original]", &safe_original)
                .replace("[Slate_Name]", &safe_name)
                .replace("[Subject_Name]", &safe_name),
        };

        let combined = if has_folder_token {
            if folder_path.is_empty() {
                with_subject
                    .replace("/[Folder]/", "/")
                    .replace("[Folder]/", "")
                    .replace("/[Folder]", "")
                    .replace("[Folder]", "")
                    .replace("/[Department]/", "/")
                    .replace("[Department]/", "")
                    .replace("/[Department]", "")
                    .replace("[Department]", "")
            } else {
                with_subject
                    .replace("[Folder]", &folder_path)
                    .replace("[Department]", &folder_path)
            }
        } else if !folder_path.is_empty() {
            let trimmed = with_subject.trim_start_matches('/');
            if has_leading_slash {
                format!("/{folder_path}/{trimmed}")
            } else {
                format!("{folder_path}/{trimmed}")
            }
        } else {
            with_subject
        };

        collapse_double_slashes(&combined)
    }
}

fn collapse_double_slashes(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut prev_slash = false;
    for ch in raw.chars() {
        if ch == '/' {
            if !prev_slash {
                out.push(ch);
            }
            prev_slash = true;
        } else {
            prev_slash = false;
            out.push(ch);
        }
    }
    out
}

pub fn sanitize_path_component(raw: &str) -> String {
    raw.trim()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::develop::{
        DevelopHistory, DevelopRecipe, DevelopSyncMask, RelativeDevelopDelta, WhiteBalance,
    };
    use crate::identity::{
        FaceEmbedding, MatchConfidence, SimilarityThresholds, SFACE_EMBEDDING_DIM,
    };

    #[test]
    fn calibrated_white_balance_warms_bluish_slate_sample() {
        let wb = WhiteBalance::calibrated_from_neutral_rgb(0.75, 0.85, 0.95);
        assert!(wb.temperature_kelvin > 5500);
    }

    #[test]
    fn develop_recipe_sync_preserves_unmasked_crop() {
        let mut source = DevelopRecipe::default();
        source.basic_tone.exposure_ev = 0.75;
        source.white_balance.temperature_kelvin = 6200;
        source.crop_transform.rotation_deg = 4.5;

        let mut target = DevelopRecipe::default();
        target.crop_transform.rotation_deg = -1.2;

        target.sync_from(&source, DevelopSyncMask::color_and_tone_only());

        assert_eq!(target.basic_tone.exposure_ev, 0.75);
        assert_eq!(target.white_balance.temperature_kelvin, 6200);
        assert_eq!(target.crop_transform.rotation_deg, -1.2);
    }

    #[test]
    fn develop_history_supports_undo_redo_and_branch_truncation() {
        let mut history = DevelopHistory::new(DevelopRecipe::default());
        let mut step1 = DevelopRecipe::default();
        step1.basic_tone.exposure_ev = 0.5;
        history.push("Exposure +0.5 EV", step1);

        let mut step2 = history.current().clone();
        step2.basic_tone.contrast = 20;
        history.push("Contrast +20", step2);

        assert!(history.undo());
        assert_eq!(history.current().basic_tone.contrast, 0);
        assert_eq!(history.current().basic_tone.exposure_ev, 0.5);

        let mut branched = history.current().clone();
        branched.basic_tone.vibrance = 15;
        history.push("Vibrance +15", branched);

        assert!(!history.redo());
        assert_eq!(history.step_count(), 3);
        assert_eq!(history.current().basic_tone.vibrance, 15);
    }

    #[test]
    fn relative_develop_delta_clamps_to_safe_bounds() {
        let mut recipe = DevelopRecipe::default();
        recipe.basic_tone.exposure_ev = 4.8;
        recipe.apply_relative_delta(RelativeDevelopDelta {
            exposure_delta_ev: 0.5,
            temperature_delta_kelvin: -400,
            ..Default::default()
        });

        assert_eq!(recipe.basic_tone.exposure_ev, 5.0);
        assert_eq!(recipe.white_balance.temperature_kelvin, 5100);
    }

    #[test]
    fn face_embedding_computes_exact_cosine_similarity_and_centroid() {
        let mut raw_a = [0.0_f32; SFACE_EMBEDDING_DIM];
        raw_a[0] = 3.0;
        raw_a[1] = 4.0;
        let emb_a = FaceEmbedding::from_raw(raw_a).expect("valid embedding");
        let emb_same = FaceEmbedding::from_raw(raw_a).expect("valid embedding");
        assert!((emb_a.cosine_similarity(&emb_same) - 1.0).abs() < 1e-5);

        let merged = emb_a.merge_into_centroid(1, &emb_same);
        assert!((merged.cosine_similarity(&emb_a) - 1.0).abs() < 1e-5);

        let thresholds = SimilarityThresholds::default();
        assert_eq!(thresholds.classify(0.991), MatchConfidence::Verified);
        assert_eq!(thresholds.classify(0.38), MatchConfidence::AmbiguousNeedsReview);
        assert_eq!(thresholds.classify(0.15), MatchConfidence::Unmatched);
    }

    #[test]
    fn output_naming_template_handles_both_slate_id_and_slate_free_subjects() {
        let tpl = OutputNamingTemplate::new("/[Slate_Name]_[ID]/IMG_001.jpg", true)
            .expect("valid template");
        assert_eq!(
            tpl.format_relative_path("Sophia Martinez", Some("#104"), 12),
            "/Sophia_Martinez_104/IMG_012.jpg"
        );
        assert_eq!(
            tpl.format_relative_path("Person 001", None, 3),
            "/Person_001/IMG_003.jpg"
        );
        assert_eq!(
            tpl.format_relative_path_in_folder(
                &["College of Engineering".to_string(), "BSARC".to_string()],
                "Villafranca, Rejelmie P.",
                None,
                1,
            ),
            "/College_of_Engineering/BSARC/Villafranca__Rejelmie_P/IMG_001.jpg"
        );

        let explicit = OutputNamingTemplate::new("/[Folder]/[Slate_Name]/IMG_[Seq].jpg", true)
            .expect("valid explicit folder template");
        assert_eq!(
            explicit.format_relative_path_in_folder(
                &["BSARC".to_string()],
                "Elena Cruz",
                None,
                2
            ),
            "/BSARC/Elena_Cruz/IMG_002.jpg"
        );
        assert_eq!(
            explicit.format_relative_path_in_folder(&[], "Elena Cruz", None, 2),
            "/Elena_Cruz/IMG_002.jpg"
        );

        let keep_orig = OutputNamingTemplate::new("/[Slate_Name]/[Original_Name].jpg", true)
            .expect("valid original-name template");
        assert_eq!(
            keep_orig.format_relative_path_for_file(
                &["Session A".to_string()],
                "Elena Cruz",
                None,
                1,
                Some("DSC06967"),
            ),
            "/Session_A/Elena_Cruz/DSC06967.jpg"
        );
    }
}
