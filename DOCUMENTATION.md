# EPL Match Prediction System - Project Documentation

**Last Updated:** March 4, 2026

This is the short project guide.  
For full technical detail, read `ML_MODEL_DOCUMENTATION.md`.

---

## 1) What this project does

- Predicts EPL outcomes: Home Win (`H`), Draw (`D`), Away Win (`A`)
- Provides Streamlit analytics UI (matches, teams, players, transfer scouting)
- Uses historical match data and feature engineering

---

## 2) Main files

| File | Purpose |
|---|---|
| `feature_engineer.py` | Data loading + feature engineering |
| `ml_model.py` | Training + draw-aware tuning + evaluation |
| `app.py` | Streamlit user interface |
| `scraper.py` | Player stats from FPL API |
| `data/matches.csv` | Legacy match data |
| `data/epl_matches.csv` | Current match data |
| `ML_MODEL_DOCUMENTATION.md` | Full deep documentation |
| `CONFUSION_MATRIX_EXPLAINED.md` | Detailed confusion matrix interpretation |

---

## 3) Latest model result snapshot

From the latest `python ml_model.py` run:

| Metric | Value |
|---|---:|
| Baseline Accuracy | 44.11% |
| Model Accuracy | 47.82% |
| Weighted Precision | 0.4848 |
| Weighted Recall | 0.4782 |
| Weighted F1 | 0.4804 |
| Draw Recall (raw -> tuned) | 0.1331 -> 0.2890 |

---

## 4) How to run

## Train model

```bash
python ml_model.py
```

## Run Streamlit app

```bash
streamlit run app.py
```

Open in browser:

`http://localhost:8501`

---

## 5) Important note about app predictions

`app.py` Tab 1 currently uses heuristic team-strength formulas and does not directly call the Random Forest model in `ml_model.py`.

If you want app predictions to match model metrics, connect Tab 1 to RF inference.

