# EPL Prediction System - Full Technical Documentation

**Last Updated:** March 4, 2026  
**Code Version:** current `feature_engineer.py` + `ml_model.py` in this workspace

---

## 1) Executive Summary

This project predicts EPL match outcomes (`H`, `D`, `A`) using engineered historical team-form features and a Random Forest classifier.

Current production training flow includes:

- Leakage-safe rolling features (only past matches used for each match feature row)
- Time-based train/test split (old matches -> train, newer matches -> test)
- Draw-focused tuning (class weighting + post-prediction draw rule)

### Latest verified results (run on March 4, 2026)

- Total modeled rows: `12,271`
- Train/Test rows: `9,816 / 2,455`
- Baseline (majority-class) accuracy: `44.11%`
- Final model accuracy: `47.82%`
- Weighted precision / recall / F1: `0.4848 / 0.4782 / 0.4804`
- Draw recall improved by tuning: `0.1331 -> 0.2890`

---

## 2) Project Components

| File | Role | Notes |
|---|---|---|
| `feature_engineer.py` | Data loading + feature engineering | Includes legacy and new leakage-safe feature pipelines |
| `ml_model.py` | Model training, tuning, evaluation | Main training entrypoint |
| `app.py` | Streamlit UI | Current Tab-1 match prediction is formula-based, not RF inference |
| `scraper.py` | Player data ingestion from FPL API | Used by Player Stats/Transfer tabs |
| `data/matches.csv` | Legacy match source | Column names normalized in loader |
| `data/epl_matches.csv` | Main/modern match source | Includes many bookmaker + match stats columns |
| `confusion_matrix.png` | Latest confusion matrix plot | Overwritten each training run |
| `rf_model_no_player.pkl`, `scaler_no_player.pkl`, `label_encoder_no_player.pkl` | Legacy serialized artifacts | From old pipeline, not aligned with current `ml_model.py` |

---

## 3) Data Pipeline

## 3.1 Input files

The loader reads and combines:

- `data/matches.csv`
- `data/epl_matches.csv`

## 3.2 Column normalization

To merge old/new schemas, loader renames:

- `FTH Goals -> FTHG`
- `FTA Goals -> FTAG`
- `FT Result -> FTR`
- `H Shots -> HS`
- `A Shots -> AS`
- `H SOT -> HST`
- `A SOT -> AST`

## 3.3 Cleaning and typing

Steps:

1. Drop rows missing `Season`, `HomeTeam`, `AwayTeam`
2. Parse `Date` with `dayfirst=True`
3. Convert numeric match columns (`FTHG`, `FTAG`, `HS`, `AS`, `HST`, `AST`)
4. Sort chronologically

## 3.4 Dataset size after preprocessing

From latest run context:

- Raw combined rows: `12,393`
- Model-usable rows after feature filtering and history constraints: `12,271`
- Date range: `1993-08-23` to `2026-02-02`

Class counts in modeled dataset:

- Home wins (`H`): `5,615`
- Draws (`D`): `3,142`
- Away wins (`A`): `3,514`

---

## 4) Feature Engineering

Two pipelines exist:

## 4.1 Legacy UI feature function (for Streamlit)

`create_team_features()` computes static team-level averages.  
This supports `app.py` tabs and quick heuristics.

## 4.2 ML training feature function (current)

`create_match_level_features(window=5, min_history_matches=3)` is the main model pipeline.

### Core logic

1. Keep only finished matches with valid `FTR in {H, D, A}`
2. Build per-team match rows (home and away perspective)
3. Compute rolling means on shifted history (`shift(1)`) for:
   - Goals for/against
   - Points
   - Shots for/against
   - Shots on target for/against
4. Merge home-side and away-side form features into each match row
5. Require both teams to have at least 3 previous matches
6. Create delta features:
   - `Diff_Form_*`
7. Create closeness features for draw behavior:
   - `AbsDiff_Form_*`
8. Encode target:
   - `H -> 0`, `D -> 1`, `A -> 2`

### Why this matters

- Prevents future-data leakage into past matches
- Makes training/test evaluation more realistic
- Adds closeness signals that help draw detection

---

## 5) Model Training Pipeline

Entrypoint: `train_enhanced_model(test_size=0.2)` in `ml_model.py`

## 5.1 Feature selection

Model uses columns that start with:

- `H_`
- `A_`
- `Diff_`
- `AbsDiff_`

## 5.2 Split strategy

Chronological split (not random):

- First 80% rows -> train
- Last 20% rows -> test

Latest split:

- Train rows: `9,816`
- Test rows: `2,455`
- Boundary dates: train `<= 2019-02-23`, test `>= 2019-02-23`

## 5.3 Missing value handling

- Fit means on training inputs only
- Apply same means to validation/test

## 5.4 Draw-focused optimization

Training includes two layers:

1. **Class weighting**
   - Base inverse-frequency class weights
   - Additional draw boost (`draw_boost=1.6`)

2. **Post-prediction draw rule**
   - Tune thresholds on a time-aware validation subset
   - Rule uses:
     - Draw probability threshold
     - Home-vs-away probability gap threshold
   - Objective balances draw F1, weighted F1, accuracy, and draw-rate control

Latest tuned values:

- `draw_prob_threshold=0.28`
- `side_gap_threshold=0.12`

---

## 6) Current Model Configuration

Final Random Forest parameters:

| Parameter | Value |
|---|---|
| `n_estimators` | `700` |
| `max_depth` | `16` |
| `min_samples_split` | `8` |
| `min_samples_leaf` | `2` |
| `class_weight` | `{0: 0.72198, 1: 2.03625, 2: 1.20604}` |
| `random_state` | `42` |
| `n_jobs` | `-1` |

---

## 7) Evaluation Results (Latest Run)

## 7.1 Overall metrics

| Metric | Value |
|---|---|
| Baseline accuracy (majority class) | `44.11%` |
| Model accuracy | `47.82%` |
| Weighted precision | `0.4848` |
| Weighted recall | `0.4782` |
| Weighted F1 | `0.4804` |

## 7.2 Draw-target improvement check

| Metric | Raw model | After draw tuning |
|---|---:|---:|
| Draw recall | `0.1331` | `0.2890` |

## 7.3 Per-class report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Home Win (H) | 0.59 | 0.60 | 0.60 | 1083 |
| Draw (D) | 0.26 | 0.29 | 0.27 | 571 |
| Away Win (A) | 0.51 | 0.44 | 0.47 | 801 |

## 7.4 Confusion matrix

```
                Predicted
                H     D     A
Actual  H      654   256   173
        D      237   165   169
        A      223   223   355
```

## 7.5 TP/TN/FP/FN (one-vs-rest)

| Class | TP | TN | FP | FN |
|---|---:|---:|---:|---:|
| Home Win (H) | 654 | 912 | 460 | 429 |
| Draw (D) | 165 | 1405 | 479 | 406 |
| Away Win (A) | 355 | 1312 | 342 | 446 |

---

## 8) Experiment History Summary

| Model variant | Accuracy | Weighted F1 | Draw recall | Notes |
|---|---:|---:|---:|---|
| Legacy static-feature RF (old doc baseline) | 47.68% | 0.4359 | 0.11 | Random split + weaker feature logic |
| Leakage-safe RF (no draw tuning) | 51.24% | 0.4709 | 0.07 | Better overall accuracy, weak draw detection |
| Current draw-optimized RF | 47.82% | 0.4804 | 0.289 | Balanced for draw detection use-case |

Interpretation:

- If objective is overall accuracy only, previous non-draw-tuned setup can score higher.
- If objective is draw usability, current setup gives materially better draw recall.

---

## 9) Streamlit App Behavior

`app.py` has 5 tabs:

1. Match Prediction
2. Historical Analysis
3. Player Stats
4. Team Analysis
5. Transfer Scout

Important:

- Tab 1 currently uses heuristic formulas from `create_team_features()`
- It does **not** call the trained Random Forest from `ml_model.py`

So training metrics and app prediction behavior are currently separate paths.

---

## 10) Legacy Saved Artifacts

These files exist but are from old feature pipelines:

- `rf_model_no_player.pkl`
- `scaler_no_player.pkl`
- `label_encoder_no_player.pkl`

They are not automatically produced by current `ml_model.py`.

If required, add explicit model persistence in training script.

---

## 11) How To Run

## 11.1 Train and evaluate

```bash
python ml_model.py
```

Outputs:

- Console metrics
- `confusion_matrix.png`

## 11.2 Run UI

```bash
streamlit run app.py
```

Default URL:

`http://localhost:8501`

---

## 12) Current Risks and Gaps

1. Draw precision is still modest (`0.26`) even though recall improved
2. No probability calibration report (Brier score / reliability curve)
3. App inference path does not use trained RF model
4. No automatic model/version artifact tracking

---

## 13) Recommended Next Steps

1. Add `save_model()` + metadata snapshot (feature list, class weights, thresholds)
2. Expose two modes in training:
   - `accuracy_priority`
   - `draw_priority`
3. Wire Streamlit Tab 1 to actual RF inference + probabilities
4. Add backtesting by season (`train: past seasons`, `test: next season`)
5. Add calibration and confidence diagnostics

