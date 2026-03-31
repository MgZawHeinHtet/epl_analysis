# Confusion Matrix Explained (Latest Run)

**Model Run Date:** March 4, 2026  
**Source:** `python ml_model.py`

---

## 1) Latest Confusion Matrix

```
                Predicted
                H     D     A
Actual  H      654   256   173
        D      237   165   169
        A      223   223   355
```

Where:

- `H` = Home Win
- `D` = Draw
- `A` = Away Win

---

## 2) TP, TN, FP, FN Meaning (One-vs-Rest)

For each class:

- `TP`: predicted this class and actually this class
- `FP`: predicted this class but actually another class
- `FN`: actually this class but predicted another class
- `TN`: all other correct non-class predictions

---

## 3) Class-by-Class Breakdown

## Home Win (H)

- TP = 654
- FP = 460
- FN = 429
- TN = 912

Interpretation:

- Home recall is moderate (`654 / 1083 = 0.60`)
- Model still misses 429 real home wins

## Draw (D)

- TP = 165
- FP = 479
- FN = 406
- TN = 1405

Interpretation:

- Draw recall improved to `0.289` compared to raw model `0.133`
- Draw precision remains low due many false positives

## Away Win (A)

- TP = 355
- FP = 342
- FN = 446
- TN = 1312

Interpretation:

- Away recall is `0.44`
- Away wins are still under-detected in many close matches

---

## 4) How values are computed

Given confusion matrix `cm` and class index `i`:

```python
tp = cm[i, i]
fp = cm[:, i].sum() - tp
fn = cm[i, :].sum() - tp
tn = cm.sum() - (tp + fp + fn)
```

---

## 5) Practical Reading Guide

- If diagonal values (`654`, `165`, `355`) grow, model improves.
- For draw quality:
  - reduce `FP` in draw column for better precision
  - reduce `FN` in draw row for better recall

Current draw-oriented tuning favors higher recall, so precision tradeoff is expected.

