import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engineer import load_and_prepare_data, create_match_level_features


def build_class_weights(y, draw_boost=1.6):
    counts = y.value_counts().to_dict()
    total = len(y)
    class_weights = {}
    for label in [0, 1, 2]:
        count = counts.get(label, 1)
        class_weights[label] = total / (3.0 * count)
    class_weights[1] = class_weights[1] * draw_boost
    return class_weights


def apply_draw_rule(probabilities, draw_prob_threshold, side_gap_threshold):
    """
    Force draw prediction only when:
    - draw probability is high enough, and
    - home/away win probabilities are close to each other.
    """
    base_pred = np.argmax(probabilities, axis=1)
    draw_prob = probabilities[:, 1]
    side_gap = np.abs(probabilities[:, 0] - probabilities[:, 2])
    draw_mask = (draw_prob >= draw_prob_threshold) & (side_gap <= side_gap_threshold)
    tuned_pred = base_pred.copy()
    tuned_pred[draw_mask] = 1
    return tuned_pred


def tune_draw_rule(model, x_val, y_val):
    val_prob = model.predict_proba(x_val)
    raw_pred = np.argmax(val_prob, axis=1)
    raw_acc = accuracy_score(y_val, raw_pred)
    raw_weighted_f1 = f1_score(y_val, raw_pred, average="weighted", zero_division=0)
    actual_draw_rate = float((y_val == 1).mean())

    min_acc = max(0.0, raw_acc - 0.03)
    min_weighted_f1 = max(0.0, raw_weighted_f1 - 0.02)
    best = None

    # Grid is intentionally small for speed but wide enough to find useful settings.
    for draw_threshold in np.arange(0.24, 0.56, 0.02):
        for side_gap in np.arange(0.04, 0.23, 0.02):
            pred = apply_draw_rule(val_prob, draw_threshold, side_gap)
            draw_f1 = f1_score((y_val == 1).astype(int), (pred == 1).astype(int), zero_division=0)
            weighted_f1 = f1_score(y_val, pred, average="weighted", zero_division=0)
            accuracy = accuracy_score(y_val, pred)
            if accuracy < min_acc or weighted_f1 < min_weighted_f1:
                continue

            pred_draw_rate = float((pred == 1).mean())
            draw_rate_penalty = abs(pred_draw_rate - actual_draw_rate)
            score = (0.45 * draw_f1) + (0.35 * weighted_f1) + (0.20 * accuracy) - (0.35 * draw_rate_penalty)

            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "draw_prob_threshold": float(draw_threshold),
                    "side_gap_threshold": float(side_gap),
                    "draw_f1": float(draw_f1),
                    "weighted_f1": float(weighted_f1),
                    "accuracy": float(accuracy),
                    "pred_draw_rate": pred_draw_rate,
                    "actual_draw_rate": actual_draw_rate,
                    "raw_accuracy": float(raw_acc),
                    "raw_weighted_f1": float(raw_weighted_f1),
                }
    if best is None:
        # Fallback: no tuned rule met constraints, disable draw forcing.
        best = {
            "score": 0.0,
            "draw_prob_threshold": 1.01,
            "side_gap_threshold": 0.0,
            "draw_f1": f1_score((y_val == 1).astype(int), (raw_pred == 1).astype(int), zero_division=0),
            "weighted_f1": float(raw_weighted_f1),
            "accuracy": float(raw_acc),
            "pred_draw_rate": float((raw_pred == 1).mean()),
            "actual_draw_rate": actual_draw_rate,
            "raw_accuracy": float(raw_acc),
            "raw_weighted_f1": float(raw_weighted_f1),
        }
    return best


def train_enhanced_model(test_size=0.2):
    raw_df = load_and_prepare_data()
    if raw_df.empty:
        print("Data not found!")
        return None

    df = create_match_level_features(raw_df, window=5, min_history_matches=3)
    if df.empty:
        print("Not enough usable match history after feature engineering.")
        return None

    df = df.sort_values("Date").reset_index(drop=True)

    feature_cols = [col for col in df.columns if col.startswith(("H_", "A_", "Diff_", "AbsDiff_"))]
    if not feature_cols:
        print("No model features were generated.")
        return None

    split_idx = int(len(df) * (1 - test_size))
    if split_idx <= 0 or split_idx >= len(df):
        print("Invalid split. Need more rows.")
        return None

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Time-aware validation split inside training for draw-rule tuning.
    val_size = max(300, int(len(train_df) * 0.15))
    if len(train_df) <= val_size + 500:
        val_size = max(100, int(len(train_df) * 0.1))

    fit_df = train_df.iloc[:-val_size].copy()
    val_df = train_df.iloc[-val_size:].copy()

    x_fit = fit_df[feature_cols]
    y_fit = fit_df["Result_Encoded"].astype(int)
    x_val = val_df[feature_cols]
    y_val = val_df["Result_Encoded"].astype(int)
    x_test = test_df[feature_cols]
    y_test = test_df["Result_Encoded"].astype(int)

    fit_means = x_fit.mean(numeric_only=True)
    x_fit = x_fit.fillna(fit_means)
    x_val = x_val.fillna(fit_means)

    class_weights = build_class_weights(y_fit, draw_boost=1.6)
    model = RandomForestClassifier(
        n_estimators=700,
        max_depth=16,
        min_samples_split=8,
        min_samples_leaf=2,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_fit, y_fit)

    draw_rule = tune_draw_rule(model, x_val, y_val)

    # Re-train on full training data after tuning thresholds.
    x_train = train_df[feature_cols]
    y_train = train_df["Result_Encoded"].astype(int)
    train_means = x_train.mean(numeric_only=True)
    x_train = x_train.fillna(train_means)
    x_test = x_test.fillna(train_means)

    final_class_weights = build_class_weights(y_train, draw_boost=1.6)
    model = RandomForestClassifier(
        n_estimators=700,
        max_depth=16,
        min_samples_split=8,
        min_samples_leaf=2,
        class_weight=final_class_weights,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    majority_class = y_train.mode().iloc[0]
    baseline_acc = (y_test == majority_class).mean()

    # Compare raw model prediction vs draw-tuned decision rule.
    raw_pred = model.predict(x_test)
    raw_draw_recall = recall_score((y_test == 1).astype(int), (raw_pred == 1).astype(int), zero_division=0)
    test_prob = model.predict_proba(x_test)
    y_pred = apply_draw_rule(
        test_prob,
        draw_prob_threshold=draw_rule["draw_prob_threshold"],
        side_gap_threshold=draw_rule["side_gap_threshold"],
    )
    tuned_draw_recall = recall_score((y_test == 1).astype(int), (y_pred == 1).astype(int), zero_division=0)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n[OK] Training Complete.")
    print(f"[INFO] Rows used: {len(df)} (train={len(train_df)}, test={len(test_df)})")
    print(f"[INFO] Date split: train <= {train_df['Date'].max().date()}, test >= {test_df['Date'].min().date()}")
    print(f"[BASELINE] Majority-class accuracy: {baseline_acc:.2%}")
    print(f"[CLASS_WEIGHT] {final_class_weights}")
    print(
        "[DRAW_TUNING] "
        f"draw_prob_threshold={draw_rule['draw_prob_threshold']:.2f}, "
        f"side_gap_threshold={draw_rule['side_gap_threshold']:.2f}, "
        f"val_draw_f1={draw_rule['draw_f1']:.4f}, "
        f"val_draw_rate={draw_rule['pred_draw_rate']:.3f} (actual={draw_rule['actual_draw_rate']:.3f})"
    )
    print(f"[DRAW_COMPARE] recall raw={raw_draw_recall:.4f}, tuned={tuned_draw_recall:.4f}")

    print("[METRICS] Model Performance Metrics:")
    print(f"   Accuracy:  {acc:.2%}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print("\n[REPORT] Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2],
            target_names=["Home Win (H)", "Draw (D)", "Away Win (A)"],
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    print("\n[MATRIX] Confusion Matrix:")
    print(cm)

    print("\n[DETAIL] TP, TN, FP, FN Analysis:")
    print("=" * 60)
    class_names = ["Home Win (H)", "Draw (D)", "Away Win (A)"]
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        print(f"\n{class_name}:")
        print(f"   TP (True Positives):  {tp}")
        print(f"   TN (True Negatives):  {tn}")
        print(f"   FP (False Positives): {fp}")
        print(f"   FN (False Negatives): {fn}")
        print(f"   Total Actual: {cm[i, :].sum()}")
        print(f"   Total Predicted: {cm[:, i].sum()}")
    print("=" * 60)

    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Home", "Draw", "Away"],
            yticklabels=["Home", "Draw", "Away"],
            annot_kws={"size": 14},
        )
        plt.title("Confusion Matrix - Match Outcome Prediction", fontsize=14, pad=20)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
        print("\n[OK] Confusion matrix saved as 'confusion_matrix.png'")
        plt.close()
    except Exception as e:
        print(f"\n[WARNING] Could not save confusion matrix image: {e}")

    return model


if __name__ == "__main__":
    train_enhanced_model()
