use crate::develop::NormalizedRect;

/// A single recognized text line on a whiteboard / placard with its normalized box and score.
#[derive(Debug, Clone, PartialEq)]
pub struct OcrTextLine {
    pub text: String,
    pub confidence: f32,
    pub region: NormalizedRect,
}

/// Structured whiteboard parse produced from raw PP-OCRv5 text lines.
#[derive(Debug, Clone, PartialEq)]
pub struct SlateCandidateParse {
    pub subject_name: String,
    pub subject_code: Option<String>,
    pub confidence: f32,
    pub whiteboard_region: Option<NormalizedRect>,
}

const NOISE_ONLY_WORDS: &[&str] = &[
    "name",
    "full name",
    "student",
    "student name",
    "subject",
    "id",
    "id no",
    "no",
    "number",
    "date",
    "section",
    "grade",
    "class",
    "course",
    "year",
    "school",
    "studio",
    "kroma",
];

const LABEL_PREFIXES: &[&str] = &[
    "full name:",
    "full name :",
    "student name:",
    "student name :",
    "name:",
    "name :",
    "student:",
    "student :",
    "subject:",
    "subject :",
    "name-",
    "name -",
];

const CODE_PREFIXES: &[&str] = &["id:", "id :", "id#", "id #", "id no:", "id no.", "no:", "no.", "#"];

const METADATA_ONLY_WORDS: &[&str] = &[
    // Academic strands & common degree/course codes
    "stem",
    "abm",
    "humss",
    "gas",
    "tvl",
    "ict",
    "beed",
    "bsed",
    "bfa",
    "bpa",
    "llb",
    "dmd",
    "dvm",
    "abcomm",
    "bacomm",
    // Graduation & studio shoot attire/prop/package labels
    "toga",
    "filipiniana",
    "barong",
    "sablay",
    "hardhat",
    "hard hat",
    "creative",
    "formal",
    "casual",
    "corporate",
    "uniform",
    "gown",
    "cap",
    "grad",
    "graduation",
    "portrait",
    "package",
    "solo",
    "group",
    "faculty",
    "staff",
    "alumni",
];

struct NameLineCandidate {
    text: String,
    confidence: f32,
    region: NormalizedRect,
    is_initial: bool,
}

/// Extracts a clean subject name + optional numeric/alphanumeric code from raw OCR text lines
/// read off a handheld whiteboard or name card.
pub fn parse_whiteboard_text_lines(lines: &[OcrTextLine]) -> Option<SlateCandidateParse> {
    let valid: Vec<&OcrTextLine> = lines
        .iter()
        .filter(|l| l.confidence >= 0.35 && !l.text.trim().is_empty())
        .collect();
    if valid.is_empty() {
        return None;
    }

    // Merge word-level boxes that sit on the same visual horizontal line (e.g. "JUAN" + "DELA" + "CRUZ")
    let merged_lines = merge_same_baseline_lines(&valid);

    let mut name_parts: Vec<NameLineCandidate> = Vec::new();
    let mut subject_code: Option<String> = None;
    let mut best_code_conf = 0.0_f32;
    let mut union_rect: Option<NormalizedRect> = None;

    for line in &merged_lines {
        let trimmed = collapse_whitespace(line.text.trim());
        if trimmed.is_empty() {
            continue;
        }
        let lower = trimmed.to_ascii_lowercase();

        // Skip pure form header labels ("NAME", "DATE", "ID NO", etc.)
        if NOISE_ONLY_WORDS
            .iter()
            .any(|w| lower == *w || lower.trim_end_matches([':', '.', '-', '#']).trim() == *w)
        {
            continue;
        }

        // Skip obvious dates like "11/21/2026" or "2026-11-21"
        if looks_like_date(&trimmed) {
            continue;
        }

        // Check for standalone or prefixed subject codes (#042, ID: 104, 042, 2026-0142)
        if let Some(code) = extract_standalone_code(&trimmed) {
            if line.confidence >= best_code_conf {
                subject_code = Some(code);
                best_code_conf = line.confidence;
            }
            union_rect = merge_rect(union_rect, line.region);
            continue;
        }

        // Strip leading form label ("Name: Maria Santos" -> "Maria Santos")
        let mut cleaned = strip_known_label_prefix(&trimmed);

        // Extract inline trailing/leading code if paired with a name ("#042 Maria Santos")
        if let Some((inline_code, rest)) = split_inline_code(&cleaned) {
            if subject_code.is_none() {
                subject_code = Some(inline_code);
                best_code_conf = line.confidence;
            }
            cleaned = rest;
        }

        let cleaned = strip_edge_punctuation(&cleaned);
        let alpha_count = cleaned.chars().filter(|c| c.is_alphabetic()).count();
        let is_middle_initial = looks_like_middle_initial(&cleaned);

        // Allow a single-letter middle initial (e.g. "P.") ONLY after a real name line,
        // while still rejecting standalone single-character glitches like "X".
        if alpha_count < 2 && !(is_middle_initial && !name_parts.is_empty()) {
            continue;
        }
        if looks_like_section_or_grade(&cleaned) || looks_like_metadata_tag(&cleaned) {
            continue;
        }

        name_parts.push(NameLineCandidate {
            text: cleaned,
            confidence: line.confidence,
            region: line.region,
            is_initial: is_middle_initial,
        });
        union_rect = merge_rect(union_rect, line.region);
    }

    if name_parts.is_empty() {
        // Fallback: if only a clean code was written on the whiteboard (e.g. "#042"), keep it
        // as a readable fallback name while letting Kroma generate sequential slate numbers.
        let code = subject_code?;
        return Some(SlateCandidateParse {
            subject_name: format!("Subject {code}"),
            subject_code: None,
            confidence: best_code_conf,
            whiteboard_region: union_rect,
        });
    }

    // Keep the top name block without pulling in lower course/prop blocks separated by whitespace
    let mut kept: Vec<&NameLineCandidate> = vec![&name_parts[0]];
    for cand in name_parts.iter().skip(1) {
        if kept.len() >= 3 {
            break;
        }
        let prev = *kept.last().unwrap();
        let prev_bot = prev.region.top + prev.region.height;
        let ref_h = prev.region.height.max(cand.region.height).max(0.02);
        let vertical_gap = cand.region.top - prev_bot;

        let first = kept[0];
        let first_looks_complete =
            first.text.contains(',') || first.text.split_whitespace().count() >= 2;
        let is_suffix = looks_like_name_suffix(&cand.text);

        if kept.len() == 1 {
            if first_looks_complete {
                // Only attach a closely spaced middle initial ("P.") or suffix ("Jr.", "III")
                if (cand.is_initial || is_suffix) && vertical_gap <= ref_h * 1.8 {
                    kept.push(cand);
                } else {
                    break;
                }
            } else if vertical_gap <= ref_h * 2.5 {
                // Normal 2-line First + Last name ("ELENA" / "CRUZ")
                kept.push(cand);
            } else {
                break;
            }
        } else if (cand.is_initial || is_suffix) && vertical_gap <= ref_h * 1.8 {
            // Optional 3rd line only for a closely spaced middle initial or suffix
            kept.push(cand);
        } else {
            break;
        }
    }

    let joined_raw = kept
        .iter()
        .map(|c| c.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let normalized_name = normalize_person_name(&joined_raw);
    if normalized_name.is_empty() {
        return None;
    }

    let avg_conf = kept.iter().map(|c| c.confidence).sum::<f32>() / kept.len() as f32;
    // Slate numbers are auto-generated sequentially per subject (`#001`, `#002`, ...) rather
    // than guessed from OCR, so OCR focuses solely on reading a clean subject name.
    Some(SlateCandidateParse {
        subject_name: normalized_name,
        subject_code: None,
        confidence: avg_conf.clamp(0.0, 1.0),
        whiteboard_region: union_rect,
    })
}

/// Normalizes handwritten ALL-CAPS or mixed-case names into clean folder-safe Title Case
/// while preserving Roman numerals (`II`, `III`, `IV`), initials (`J.R.`), and Latin accents (`Ñ`).
pub fn normalize_person_name(raw: &str) -> String {
    let collapsed = collapse_whitespace(raw);
    let tokens: Vec<String> = collapsed
        .split_whitespace()
        .filter_map(|token| {
            let clean = token.trim_matches(|c: char| {
                !c.is_alphanumeric() && c != '.' && c != '-' && c != '\'' && c != ','
            });
            if clean.is_empty() {
                return None;
            }
            Some(format_name_token(clean))
        })
        .collect();
    tokens.join(" ")
}

fn format_name_token(token: &str) -> String {
    let upper = token.to_uppercase();
    let stripped_punct = upper.trim_end_matches([',', '.']);
    if matches!(stripped_punct, "II" | "III" | "IV" | "VI" | "VII" | "VIII") {
        return upper;
    }
    if matches!(stripped_punct, "JR" | "SR") {
        let mut s = title_case_word(stripped_punct);
        if token.ends_with('.') {
            s.push('.');
        }
        if token.ends_with(',') {
            s.push(',');
        }
        return s;
    }
    // Preserve short dotted initials like "J.R." or "M."
    if token.contains('.') && token.chars().filter(|c| c.is_alphabetic()).count() <= 3 {
        return token.to_uppercase();
    }
    // Handle hyphenated surnames like "DELA-CRUZ" -> "Dela-Cruz"
    if token.contains('-') {
        return token
            .split('-')
            .map(title_case_word)
            .collect::<Vec<_>>()
            .join("-");
    }
    title_case_word(token)
}

fn title_case_word(word: &str) -> String {
    let mut chars = word.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };
    let mut out = String::with_capacity(word.len());
    for c in first.to_uppercase() {
        out.push(c);
    }
    for c in chars {
        for lc in c.to_lowercase() {
            out.push(lc);
        }
    }
    out
}

fn collapse_whitespace(input: &str) -> String {
    input.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn strip_known_label_prefix(input: &str) -> String {
    let lower = input.to_ascii_lowercase();
    for prefix in LABEL_PREFIXES {
        if lower.starts_with(prefix) {
            return input[prefix.len()..].trim().to_string();
        }
    }
    input.to_string()
}

fn strip_edge_punctuation(input: &str) -> String {
    input
        .trim_matches(|c: char| {
            !c.is_alphanumeric() && c != '.' && c != '\'' && c != '-'
        })
        .trim()
        .to_string()
}

fn extract_standalone_code(trimmed: &str) -> Option<String> {
    let lower = trimmed.to_ascii_lowercase();
    let mut body = trimmed;
    let mut has_explicit_prefix = false;
    for prefix in CODE_PREFIXES {
        if lower.starts_with(prefix) {
            body = trimmed[prefix.len()..].trim();
            has_explicit_prefix = true;
            break;
        }
    }
    if body.is_empty() || body.contains(' ') {
        return None;
    }
    let digit_count = body.chars().filter(|c| c.is_ascii_digit()).count();
    let valid_code_chars = body
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if !valid_code_chars || digit_count == 0 {
        return None;
    }
    // Pure digits (2..=8 chars) or prefixed alphanumeric code ("#042", "ID: A-104")
    if has_explicit_prefix || body.chars().all(|c| c.is_ascii_digit() || c == '-') && (2..=10).contains(&body.len()) {
        return Some(format!("#{}", body.trim_start_matches('#')));
    }
    None
}

fn split_inline_code(cleaned: &str) -> Option<(String, String)> {
    let tokens: Vec<&str> = cleaned.split_whitespace().collect();
    if tokens.len() < 2 {
        return None;
    }
    if let Some(code) = extract_standalone_code(tokens[0]) {
        return Some((code, tokens[1..].join(" ")));
    }
    if let Some(code) = extract_standalone_code(tokens[tokens.len() - 1]) {
        return Some((code, tokens[..tokens.len() - 1].join(" ")));
    }
    None
}

fn looks_like_date(trimmed: &str) -> bool {
    let sep_count = trimmed.chars().filter(|c| *c == '/' || *c == '-').count();
    let digit_count = trimmed.chars().filter(|c| c.is_ascii_digit()).count();
    let alpha_count = trimmed.chars().filter(|c| c.is_alphabetic()).count();
    sep_count >= 2 && digit_count >= 4 && alpha_count == 0
}

fn looks_like_section_or_grade(trimmed: &str) -> bool {
    let lower = trimmed.to_ascii_lowercase();
    lower.starts_with("grade ")
        || lower.starts_with("section ")
        || lower.starts_with("sec ")
        || lower.starts_with("class ")
        || lower.starts_with("batch ")
        || lower.starts_with("room ")
        || lower.starts_with("date ")
        || lower.starts_with("date:")
}

fn looks_like_middle_initial(cleaned: &str) -> bool {
    let stripped = cleaned.trim_end_matches('.');
    stripped.chars().count() == 1 && stripped.chars().all(|c| c.is_alphabetic())
}

fn looks_like_name_suffix(cleaned: &str) -> bool {
    let upper = cleaned.trim_end_matches(['.', ',']).to_ascii_uppercase();
    matches!(
        upper.as_str(),
        "JR" | "SR" | "II" | "III" | "IV" | "V" | "VI" | "VII" | "VIII"
    )
}

fn looks_like_metadata_tag(cleaned: &str) -> bool {
    let lower = cleaned.trim_end_matches(['.', ':', '-']).to_ascii_lowercase();
    if METADATA_ONLY_WORDS.contains(&lower.as_str()) {
        return true;
    }
    // Single-token Bachelor of Science degree acronyms like BSARC, BSIT, BSCE, BSBA, BSHM, BSN
    let compact = cleaned.trim_end_matches('.');
    !compact.contains(' ')
        && (3..=8).contains(&compact.len())
        && compact.chars().all(|c| c.is_ascii_alphabetic())
        && compact.to_ascii_uppercase().starts_with("BS")
}

fn merge_same_baseline_lines(lines: &[&OcrTextLine]) -> Vec<OcrTextLine> {
    let mut sorted: Vec<&OcrTextLine> = lines.to_vec();
    sorted.sort_by(|a, b| a.region.top.total_cmp(&b.region.top));

    let mut rows: Vec<Vec<&OcrTextLine>> = Vec::new();
    for line in sorted {
        let cy = line.region.top + line.region.height * 0.5;
        let mut placed = false;
        for row in &mut rows {
            let row_top = row.iter().map(|r| r.region.top).fold(f32::INFINITY, f32::min);
            let row_bot = row
                .iter()
                .map(|r| r.region.top + r.region.height)
                .fold(f32::NEG_INFINITY, f32::max);
            let row_h = (row_bot - row_top).max(0.01);
            let row_cy = (row_top + row_bot) * 0.5;
            if (cy - row_cy).abs() <= row_h.max(line.region.height) * 0.6 {
                row.push(line);
                placed = true;
                break;
            }
        }
        if !placed {
            rows.push(vec![line]);
        }
    }

    rows.into_iter()
        .map(|mut row| {
            row.sort_by(|a, b| a.region.left.total_cmp(&b.region.left));
            let text = row
                .iter()
                .map(|r| r.text.trim())
                .filter(|t| !t.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            let conf = row.iter().map(|r| r.confidence).sum::<f32>() / row.len() as f32;
            let mut region = row[0].region;
            for item in &row[1..] {
                if let Some(m) = merge_rect(Some(region), item.region) {
                    region = m;
                }
            }
            OcrTextLine {
                text,
                confidence: conf,
                region,
            }
        })
        .collect()
}

fn merge_rect(acc: Option<NormalizedRect>, next: NormalizedRect) -> Option<NormalizedRect> {
    match acc {
        None => Some(next),
        Some(a) => {
            let x1 = a.left.min(next.left);
            let y1 = a.top.min(next.top);
            let x2 = (a.left + a.width).max(next.left + next.width);
            let y2 = (a.top + a.height).max(next.top + next.height);
            Some(NormalizedRect {
                left: x1.clamp(0.0, 1.0),
                top: y1.clamp(0.0, 1.0),
                width: (x2 - x1).clamp(0.0, 1.0),
                height: (y2 - y1).clamp(0.0, 1.0),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str, confidence: f32, y: f32) -> OcrTextLine {
        OcrTextLine {
            text: text.to_string(),
            confidence,
            region: NormalizedRect {
                left: 0.2,
                top: y,
                width: 0.6,
                height: 0.08,
            },
        }
    }

    #[test]
    fn parses_handwritten_whiteboard_name_and_ignores_form_and_code_noise() {
        let lines = vec![
            line("STUDENT NAME:", 0.96, 0.50),
            line("DELA CRUZ, JUAN", 0.92, 0.58),
            line("GRADE 12 - STEM", 0.90, 0.66),
            line("#042", 0.95, 0.74),
            line("11/21/2026", 0.94, 0.82),
        ];
        let parsed = parse_whiteboard_text_lines(&lines).expect("parsed");
        assert_eq!(parsed.subject_name, "Dela Cruz, Juan");
        assert_eq!(parsed.subject_code, None);
        assert!(parsed.confidence > 0.9);
        assert!(parsed.whiteboard_region.is_some());
    }

    #[test]
    fn combines_two_line_handwritten_first_and_last_name_preserving_accents_and_suffixes() {
        let lines = vec![
            line("Name: JOSÉ PEÑA", 0.91, 0.45),
            line("III", 0.89, 0.55),
        ];
        let parsed = parse_whiteboard_text_lines(&lines).expect("parsed");
        assert_eq!(parsed.subject_name, "José Peña III");
    }

    #[test]
    fn strips_inline_code_from_name_and_falls_back_to_code_only_whiteboard() {
        let inline = parse_whiteboard_text_lines(&[line("#108 MARIA SANTOS", 0.88, 0.50)])
            .expect("inline");
        assert_eq!(inline.subject_name, "Maria Santos");
        assert_eq!(inline.subject_code, None);

        let code_only = parse_whiteboard_text_lines(&[line("ID: 007", 0.93, 0.55)])
            .expect("code only");
        assert_eq!(code_only.subject_name, "Subject #007");
        assert_eq!(code_only.subject_code, None);
    }

    #[test]
    fn merges_three_word_name_split_into_separate_boxes_on_same_baseline() {
        let box_at = |text: &str, left: f32, top: f32| OcrTextLine {
            text: text.to_string(),
            confidence: 0.95,
            region: NormalizedRect {
                left,
                top,
                width: 0.16,
                height: 0.06,
            },
        };
        let parsed = parse_whiteboard_text_lines(&[
            box_at("JUAN", 0.15, 0.50),
            box_at("DELA", 0.35, 0.51),
            box_at("CRUZ", 0.55, 0.50),
            box_at("#104", 0.35, 0.62),
        ])
        .expect("merged 3-word name");
        assert_eq!(parsed.subject_name, "Juan Dela Cruz");
        assert_eq!(parsed.subject_code, None);
    }

    #[test]
    fn preserves_middle_initial_and_skips_course_and_attire_tags_on_grad_whiteboard() {
        let lines = vec![
            line("VILLAFRANCA, REJELMIE", 0.97, 0.42),
            line("P.", 0.98, 0.53),
            line("BSARC", 0.99, 0.70),
            line("HARDHAT", 0.99, 0.75),
        ];
        let parsed = parse_whiteboard_text_lines(&lines).expect("parsed");
        assert_eq!(parsed.subject_name, "Villafranca, Rejelmie P.");
        assert_eq!(parsed.subject_code, None);
    }

    #[test]
    fn rejects_low_confidence_or_single_character_glitches() {
        assert!(parse_whiteboard_text_lines(&[line("X", 0.99, 0.5)]).is_none());
        assert!(parse_whiteboard_text_lines(&[line("MARIA", 0.20, 0.5)]).is_none());
    }
}
