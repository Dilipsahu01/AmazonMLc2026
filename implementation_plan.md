# Final Implementation Plan — Amazon ML Challenge 2026

## Status: Locked (Pending Dataset)

This plan incorporates feedback from two independent expert reviews. All architectural debates have been resolved. **We do not start coding the sophisticated pieces until we see the actual dataset** — the dataset's size, match density, singleton ratio, and noise patterns will determine K values, blocking strategy, and model complexity.

---

## Core Principle: Experiment-Driven Architecture

We do **not** assume any component is better than another. We build a modular pipeline, then fill this table empirically:

| Experiment | Blocking Recall | Avg Candidates/S1 | F₀.₅ | Runtime |
|---|---:|---:|---:|---:|
| TF-IDF only | — | — | — | — |
| TF-IDF + Pretrained Embedding | — | — | — | — |
| TF-IDF + Fine-tuned Embedding | — | — | — | — |
| LightGBM (base features) | — | — | — | — |
| LightGBM + calibration | — | — | — | — |
| + Cross-Encoder rerank | — | — | — | — |
| + Soft Graph Consistency | — | — | — | — |
| + Data Augmentation (France) | — | — | — | — |

> [!IMPORTANT]
> Every architectural decision after the baseline is justified by this table, not by assumption.

---

## What We Agreed On (All Reviews Aligned)

### ✅ 1. Country-Aware Pooling BEFORE Blocking

```
S2 + S3 records
      │
      ▼
Group by country
      │
      ├── US pool
      ├── India pool
      └── France pool (open-set, never hardcoded away)
            │
            ▼
      For each S1 entity:
        search ONLY within same-country pool
```

**Why:** 3× speed boost, cleaner TF-IDF vocabulary, zero cross-country false positives. Country must remain an open-set string filter — France is unseen in training but present in test.

---

### ✅ 2. Singleton-Aware Decision Logic (Not a Separate Model)

Start with simple threshold logic:

```
best_candidate_probability
           │
     < singleton_threshold?
        ↙          ↘
     EMPTY        MATCH(ES)
```

This is **not** a separate ML model. It's a calibrated threshold on the matcher's output. The singleton threshold is tuned independently from the match threshold on validation data, because the F₀.₅ payoff structure is asymmetric:
- Correctly predicting "no match" → 1.0 per-entity score
- Hallucinating a match for a true singleton → 0.0 per-entity score

A dedicated singleton classifier is only built if experiments show the simple threshold is insufficient.

---

### ✅ 3. LightGBM First, No Stacking Until Proven Necessary

```
Features → LightGBM → Calibration → Threshold
```

Stacking (LR + GBM + NN + meta-learner) is an **experiment**, not a core component. We add a second model only if validation F₀.₅ plateaus and we have evidence that a different model captures signal LightGBM misses.

---

### ✅ 4. LLM Arbiter → Last Resort

We don't touch this until:
- The conventional pipeline (blocking + features + LightGBM) is fully working
- We've exhausted feature engineering improvements
- We've tried calibration + cross-encoder
- There's still a clear "ambiguous band" of pairs where precision is low

---

### ✅ 5. Benchmark Pretrained vs. Fine-Tuned Embeddings

Don't assume fine-tuning helps. Benchmark on leave-one-country-out validation:

| Embedding Strategy | India-held-out F₀.₅ | US-held-out F₀.₅ | Blocking Recall |
|---|---|---|---|
| No embedding (TF-IDF only) | — | — | — |
| Pretrained multilingual (frozen) | — | — | — |
| Fine-tuned on US+India pairs | — | — | — |

If fine-tuning hurts cross-country generalization, we use pretrained-only.

---

### ✅ 6. Three First-Class Metrics

Every experiment reports all three:

| Metric | What It Measures | Why It Matters |
|---|---|---|
| **Blocking Recall** | % of true matches present in candidate set | Ceiling on final recall — lost matches are unrecoverable |
| **Avg Candidates / S1** | Mean size of candidate set per S1 entity | Amazon explicitly ranks smaller sets higher in final evaluation |
| **F₀.₅** | Precision-weighted harmonic mean | The leaderboard score |

---

### ✅ 7. Soft Graph Consistency Only

No hard "one S2 → one S1" constraint. The problem permits many-to-many matching.

Conflict resolution rule:

```
If S2-X has high scores to both S1-A and S1-B:
    score_gap = score(S1-A) - score(S1-B)
    
    If score_gap > margin_threshold:
        Keep only best match (confident disambiguation)
    Else:
        Keep all matches (ambiguous — don't risk recall)
```

---

### ✅ 8. Modular Design (Swappable Components)

Every component is a module with a clean interface. We can swap:
- Blocker: TF-IDF ↔ Embedding ↔ Union
- Feature set: add/remove feature groups
- Matcher: LightGBM ↔ XGBoost ↔ Cross-Encoder
- Threshold: independent tuning per component

---

## Resolved Disagreements

### Blocking vs. Feature Engineering

Both reviews converged on this framing:

> **Blocking determines what is *possible* (recall ceiling).**
> **Features + Matcher determine what you *actually select* (precision).**

Neither dominates. They optimize different ceilings. Our build order reflects this — we build blocking first (because it gates everything downstream), then features + matcher (because that's where precision comes from).

```
BLOCKING                         FEATURES + MATCHER
    │                                    │
    └── "Can the true match              └── "Can the model distinguish
         enter the model?"                    true from false candidates?"
              ↓                                       ↓
         Recall ceiling                          Precision
```

---

### Cross-Encoder Status

**Classification: 🟡 Measured Experiment (not core, not automatic High Value)**

We build it. We measure it. We keep it only if:

```
F₀.₅(LightGBM + Cross-Encoder) > F₀.₅(LightGBM alone) + meaningful margin
```

AND inference time remains acceptable. If the gain is tiny while runtime doubles, we drop it.

---

### The "0.4 Swing" Claim

**Removed.** The correct statement is:

> Singleton handling can materially affect macro F₀.₅, especially if the dataset contains many true singletons. The actual impact depends on the singleton ratio in the test set, which we will measure on validation data.

---

## Final Locked Architecture

```
                    ┌───────────────────────┐
                    │      S1 / S2 / S3     │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  NORMALIZATION        │
                    │  + ADDRESS PARSING    │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ COUNTRY-AWARE POOLING │
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
             ┌─────────────┐         ┌─────────────┐
             │ TF-IDF/BM25 │         │ Embedding   │
             │ Blocking    │         │ + FAISS     │
             └──────┬──────┘         └──────┬──────┘
                    │                       │
                    └───────────┬───────────┘
                                ▼
                    ┌───────────────────────┐
                    │ UNION + DEDUP         │
                    │ → candidate_pairs.tsv │
                    └───────────┬───────────┘
                                │
                         ~small candidate set
                                │
                                ▼
                    ┌───────────────────────┐
                    │ FEATURE ENGINEERING   │
                    │ Name (6 features)     │
                    │ Address comp (8)      │
                    │ Phonetic (2)          │
                    │ Cross-field (2)       │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │       LightGBM        │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ CALIBRATION           │
                    │ + F₀.₅ threshold      │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ SINGLETON-AWARE       │
                    │ DECISION              │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ SOFT GLOBAL           │
                    │ CONSISTENCY           │
                    │ (if experiments show  │
                    │  it helps)            │
                    └───────────┬───────────┘
                                │
                                ▼
                     matching_results.tsv
```

### Experimental Branches (Data-Driven Promotion)

```
EXPERIMENTS (build, measure, keep only if proven)
     │
     ├── Fine-tuned embeddings (vs pretrained baseline)
     ├── Cross-Encoder reranker (on top-K after LightGBM)
     ├── Model stacking (LR + GBM meta-learner)
     ├── Data augmentation for France robustness
     └── LLM arbiter for ambiguous band (last resort)
```

---

## Build Order

### Phase 1: Foundation (Hours 0–6) 🔴 Essential
| Module | Purpose |
|---|---|
| `config.py` | Paths, hyperparameters, thresholds |
| `data_loader.py` | Load TSVs, handle missing fields (null addresses) |
| `transliteration.py` | Transliterate Indic scripts (Devanagari, Kannada, etc.) to Latin |
| `text_normalizer.py` | Unicode NFKD, lowercase, punctuation |
| `name_normalizer.py` | Abbreviation expansion (Corp→Corporation, &→and) |
| `country_normalizer.py` | Open-set country standardization |
| `address_parser.py` | Regex fallback (libpostal likely too slow for 10M records) |

**Checkpoint:** Print normalized records. Verify they look clean. Measure singleton ratio in ground truth.

### Phase 2: Blocking Baseline (Hours 6–12) 🔴 Essential
| Module | Purpose |
|---|---|
| `country_partitioner.py` | Split S2+S3 into country pools |
| `tfidf_blocker.py` | TF-IDF top-K within same country |
| `blocking_metrics.py` | Measure blocking recall + avg candidates |
| `validation_splits.py` | Random split + leave-one-country-out |

**Checkpoint:** `candidate_pairs.tsv` exists. Blocking recall > 95%. Avg candidates < 50.

### Phase 3: First Matcher (Hours 12–20) 🔴 Essential
| Module | Purpose |
|---|---|
| `name_features.py` | 6 name similarity scores |
| `address_features.py` | 8 address component scores |
| `phonetic_features.py` | 2 phonetic similarity scores |
| `cross_features.py` | 2 cross-field features |
| `lgbm_matcher.py` | LightGBM binary classifier |
| `singleton_detector.py` | Threshold-based singleton logic |
| `f05_scorer.py` | Per-entity F₀.₅ + macro average |

**Checkpoint:** Local F₀.₅ on validation. **→ FIRST SUBMISSION**

### Phase 4: Embedding Boost (Hours 20–28) 🟠 High Value
| Module | Purpose |
|---|---|
| `embedding_blocker.py` | Pretrained multilingual sentence-transformer + FAISS |
| `union_blocker.py` | Merge TF-IDF + embedding candidates, dedup |

**Checkpoint:** Compare 3 metrics (blocking recall, avg candidates, F₀.₅) before and after embedding. Keep only if it improves.

### Phase 5: Precision Experiments (Hours 28–40) 🟡 Experiments
| Module | Purpose |
|---|---|
| `calibrator.py` | Platt scaling / isotonic regression |
| `cross_encoder.py` | DeBERTa reranker on top-K |
| `graph_consistency.py` | Soft conflict resolution |
| Data augmentation | Synthetic perturbations for France robustness |

**Checkpoint:** Fill the experiment tracking table. Keep only what improves F₀.₅. **→ MULTIPLE SUBMISSIONS**

### Phase 6: Ship (Hours 40–48) 🔴 Essential
| Task | Purpose |
|---|---|
| Final threshold sweep | Optimize match + singleton thresholds on full validation |
| Best submission | Upload the configuration with highest F₀.₅ |
| Documentation | Fill `Documentation_template.md` with methodology |
| Zip packaging | Structure as `<team_name>_submission.zip` per spec |

---

## Project Structure

```
amazonmlc/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── pipeline.py               # End-to-end orchestration
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── transliteration.py
│   │   ├── text_normalizer.py
│   │   ├── name_normalizer.py
│   │   ├── address_parser.py
│   │   └── country_normalizer.py
│   │
│   ├── blocking/
│   │   ├── __init__.py
│   │   ├── country_partitioner.py
│   │   ├── tfidf_blocker.py
│   │   ├── embedding_blocker.py   # Phase 4
│   │   └── union_blocker.py       # Phase 4
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── name_features.py
│   │   ├── address_features.py
│   │   ├── phonetic_features.py
│   │   └── cross_features.py
│   │
│   ├── matching/
│   │   ├── __init__.py
│   │   ├── lgbm_matcher.py
│   │   ├── calibrator.py          # Phase 5
│   │   ├── cross_encoder.py       # Phase 5 experiment
│   │   └── singleton_detector.py
│   │
│   ├── postprocessing/
│   │   ├── __init__.py
│   │   └── graph_consistency.py   # Phase 5 experiment
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── f05_scorer.py
│       ├── blocking_metrics.py
│       └── validation_splits.py
│
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
│
├── experiments/                   # Experiment tracking logs
│   └── tracking_table.csv
│
├── utils/
│   └── validate_submission.py
│
├── requirements.txt
├── README.md
└── Documentation_template.md
```

---

## What We're Waiting For

> [!IMPORTANT]
> **The dataset.** Before writing any code, we need to see:
> 1. **Dataset size** — How many records in S1, S2, S3? This determines blocking K values and compute budget.
> 2. **Singleton ratio** — What % of S1 entities have zero matches? This determines how important singleton detection is.
> 3. **Match density** — How many matches does the average S1 entity have? 1? 5? 20?
> 4. **Noise severity** — How messy are the names/addresses actually? This determines whether simple features suffice or we need heavy preprocessing.
> 5. **Country distribution** — What's the split between US and India in training?
>
> These numbers directly inform every hyperparameter and design decision in the pipeline.
