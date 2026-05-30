//! Labels-map loader and the curation pipeline from raw ImageNet scores to
//! the curated, deduplicated list returned in the synthetic-data response.
//!
//! Pipeline (per spec):
//!   1. Take 1000 raw softmax scores.
//!   2. Drop scores below `SCORE_FLOOR`.
//!   3. Look up each surviving class in the labels-map.
//!   4. Drop entries whose `label` is null.
//!   5. Drop entries whose `category == "person"`.
//!   6. De-duplicate by display label, keeping the max raw score.
//!   7. Sort descending by score; tiebreak alphabetically by name.
//!   8. Apply count cap (set `truncated` if anything dropped).

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Minimum raw softmax score that survives step 2 of the curation pipeline.
pub const SCORE_FLOOR: f32 = 0.05;

/// Category name (in labels-map.json) for ImageNet classes that overlap with
/// face recognition output. These are dropped in step 5.
const PERSON_CATEGORY: &str = "person";

/// Loaded labels-map: maps an ImageNet class index (0..=999) to its curation
/// outcome.
pub struct LabelsMap {
    entries: HashMap<u32, Entry>,
}

struct Entry {
    /// Display label (`None` means "drop this class").
    label: Option<String>,
    /// True iff the source `category` was `"person"`.
    is_person: bool,
}

/// One curated label as returned in the response.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Label {
    pub name: String,
    pub score: f32,
}

#[derive(Debug, thiserror::Error)]
pub enum LabelsMapError {
    #[error("read {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("parse {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("non-integer class key {key:?} in {path}: {source}")]
    BadKey {
        path: String,
        key: String,
        #[source]
        source: std::num::ParseIntError,
    },
}

#[derive(Deserialize)]
struct RawEntry {
    // `raw` is the original ImageNet class name; we don't surface it.
    #[allow(dead_code)]
    raw: String,
    label: Option<String>,
    category: String,
}

impl LabelsMap {
    /// Parses `labels-map.json` from disk.
    ///
    /// Schema: a JSON object whose keys are stringified ImageNet class
    /// indices (`"0"`..`"999"`) and whose values carry `raw`, `label`, and
    /// `category` fields. Missing or unparseable file is a fatal startup
    /// error for the caller to handle.
    pub fn load(path: &Path) -> Result<Self, LabelsMapError> {
        let path_display = path.display().to_string();
        let text = fs::read_to_string(path).map_err(|e| LabelsMapError::Read {
            path: path_display.clone(),
            source: e,
        })?;
        let raw: HashMap<String, RawEntry> =
            serde_json::from_str(&text).map_err(|e| LabelsMapError::Parse {
                path: path_display.clone(),
                source: e,
            })?;
        let mut entries = HashMap::with_capacity(raw.len());
        for (key, value) in raw {
            let idx: u32 = key.parse().map_err(|source| LabelsMapError::BadKey {
                path: path_display.clone(),
                key: key.clone(),
                source,
            })?;
            entries.insert(
                idx,
                Entry {
                    label: value.label,
                    is_person: value.category == PERSON_CATEGORY,
                },
            );
        }
        Ok(Self { entries })
    }

    /// Number of class entries (1000 for a well-formed ImageNet map).
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Result of [`curate`]: the curated label list and whether the count cap
/// trimmed any entries.
#[derive(Debug, PartialEq)]
pub struct CurationResult {
    pub labels: Vec<Label>,
    pub truncated: bool,
}

/// Runs the curation pipeline. `raw_scores` is `(class_index, softmax_score)`
/// pairs for the classifier's 1000 outputs.
pub fn curate(raw_scores: &[(u32, f32)], map: &LabelsMap, cap: usize) -> CurationResult {
    let mut buckets: HashMap<&str, f32> = HashMap::new();
    for &(idx, score) in raw_scores {
        if score < SCORE_FLOOR {
            continue;
        }
        let Some(entry) = map.entries.get(&idx) else {
            continue;
        };
        if entry.is_person {
            continue;
        }
        let Some(label) = entry.label.as_deref() else {
            continue;
        };
        buckets
            .entry(label)
            .and_modify(|s| {
                if score > *s {
                    *s = score;
                }
            })
            .or_insert(score);
    }
    let mut labels: Vec<Label> = buckets
        .into_iter()
        .map(|(name, score)| Label {
            name: name.to_owned(),
            score,
        })
        .collect();
    labels.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.name.cmp(&b.name))
    });
    let truncated = labels.len() > cap;
    labels.truncate(cap);
    CurationResult { labels, truncated }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(entries: &[(u32, Option<&str>, &str)]) -> LabelsMap {
        let entries = entries
            .iter()
            .map(|(k, label, category)| {
                (
                    *k,
                    Entry {
                        label: label.map(str::to_owned),
                        is_person: *category == PERSON_CATEGORY,
                    },
                )
            })
            .collect();
        LabelsMap { entries }
    }

    #[test]
    fn drops_scores_below_floor() {
        let m = map(&[(1, Some("dog"), "animal"), (2, Some("cat"), "animal")]);
        let r = curate(&[(1, 0.04), (2, 0.5)], &m, 20);
        assert_eq!(
            r.labels,
            vec![Label {
                name: "cat".into(),
                score: 0.5
            }]
        );
        assert!(!r.truncated);
    }

    #[test]
    fn drops_null_labels() {
        let m = map(&[(1, None, "object"), (2, Some("car"), "vehicle")]);
        let r = curate(&[(1, 0.9), (2, 0.4)], &m, 20);
        assert_eq!(
            r.labels,
            vec![Label {
                name: "car".into(),
                score: 0.4
            }]
        );
    }

    #[test]
    fn drops_person_category() {
        let m = map(&[
            (1, Some("baseball player"), "person"),
            (2, Some("baseball"), "sport"),
        ]);
        let r = curate(&[(1, 0.95), (2, 0.6)], &m, 20);
        assert_eq!(
            r.labels,
            vec![Label {
                name: "baseball".into(),
                score: 0.6
            }]
        );
    }

    #[test]
    fn dedupes_by_label_taking_max_score() {
        let m = map(&[
            (1, Some("bird"), "animal"),
            (2, Some("bird"), "animal"),
            (3, Some("bird"), "animal"),
        ]);
        let r = curate(&[(1, 0.3), (2, 0.7), (3, 0.5)], &m, 20);
        assert_eq!(
            r.labels,
            vec![Label {
                name: "bird".into(),
                score: 0.7
            }]
        );
    }

    #[test]
    fn sorts_desc_with_alphabetical_tiebreak() {
        let m = map(&[
            (1, Some("zebra"), "animal"),
            (2, Some("apple"), "food"),
            (3, Some("monkey"), "animal"),
        ]);
        let r = curate(&[(1, 0.5), (2, 0.5), (3, 0.9)], &m, 20);
        assert_eq!(
            r.labels,
            vec![
                Label {
                    name: "monkey".into(),
                    score: 0.9
                },
                Label {
                    name: "apple".into(),
                    score: 0.5
                },
                Label {
                    name: "zebra".into(),
                    score: 0.5
                },
            ]
        );
    }

    #[test]
    fn cap_truncates_and_sets_flag() {
        let entries: Vec<_> = (0u32..30)
            .map(|i| (i, Some(format!("label{i:02}")), "object".to_string()))
            .collect();
        let m = map(&entries
            .iter()
            .map(|(k, l, c)| (*k, l.as_deref(), c.as_str()))
            .collect::<Vec<_>>());
        let raw: Vec<(u32, f32)> = (0u32..30).map(|i| (i, 0.5 + (i as f32) * 0.01)).collect();
        let r = curate(&raw, &m, 20);
        assert_eq!(r.labels.len(), 20);
        assert!(r.truncated);
        // Highest score should be the last index, which had the highest input.
        assert_eq!(r.labels[0].name, "label29");
    }

    #[test]
    fn unknown_class_indices_are_silently_dropped() {
        let m = map(&[(1, Some("dog"), "animal")]);
        let r = curate(&[(1, 0.6), (999, 0.9)], &m, 20);
        assert_eq!(
            r.labels,
            vec![Label {
                name: "dog".into(),
                score: 0.6
            }]
        );
    }

    /// Smoke test against the real downloaded labels-map.json if present.
    /// Skips quietly when run in an environment that hasn't fetched models.
    #[test]
    fn loads_real_labels_map_when_present() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join("labels-map.json");
        if !path.exists() {
            eprintln!("skip: {} not present", path.display());
            return;
        }
        let m = LabelsMap::load(&path).expect("load");
        assert_eq!(m.len(), 1000);
        let person_count = m.entries.values().filter(|e| e.is_person).count();
        assert_eq!(person_count, 3, "expected 3 person-category entries");
    }
}
