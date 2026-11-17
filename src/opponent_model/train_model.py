"""
Train the XGBoost opponent model on the synthetic CSV and save it.

Run from the project root:
    python -m src.opponent_model.train_model
Saves:
    models/opponent_model.json
    models/feature_columns.json
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                             roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from src.opponent_model.data_generator import FEATURES, LABEL
from src.utils.config import GENERATOR as G

MODEL_PATH = "models/opponent_model.json"
COLS_PATH = "models/feature_columns.json"


def split_by_negotiation(df, test_size=0.2, seed=0):
    """Keep every row of a negotiation on the same side of the split."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df["negotiation_id"]))
    return df.iloc[train_idx], df.iloc[test_idx]


def build_model(seed):
    return XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        monotone_constraints=(0, 0, 0, 0, 1, 1, 1, -1, -1),
        eval_metric="logloss", random_state=seed,
    )


def report(model, X_test, y_test):
    p = model.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)
    baseline = max(y_test.mean(), 1 - y_test.mean())
    print(f"test rows: {len(y_test)}, accepted share: {y_test.mean():.1%}")
    print(f"accuracy:            {accuracy_score(y_test, pred):.3f}"
          f"   (always-majority baseline: {baseline:.3f})")
    print(f"AUC:                 {roc_auc_score(y_test, p):.3f}   (0.5 = guessing)")
    print(f"log loss:            {log_loss(y_test, p):.3f}")
    print(f"Brier score:         {brier_score_loss(y_test, p):.3f}   (lower is better)")


def main():
    df = pd.read_csv(G["output_path"])
    train, test = split_by_negotiation(df, seed=G["seed"])

    # Only observable features. Never the hidden_* debug columns.
    X_train, y_train = train[FEATURES], train[LABEL]
    X_test, y_test = test[FEATURES], test[LABEL]

    model = build_model(G["seed"])
    model.fit(X_train, y_train)

    report(model, X_test, y_test)

    print("\nfeature importance:")
    imp = pd.Series(model.feature_importances_, index=FEATURES)
    print(imp.sort_values(ascending=False).round(3).to_string())

    os.makedirs("models", exist_ok=True)
    model.save_model(MODEL_PATH)
    with open(COLS_PATH, "w") as f:
        json.dump(FEATURES, f)
    print(f"\nsaved {MODEL_PATH} and {COLS_PATH}")

    # Reload check: saved model must match the in-memory one.
    reloaded = XGBClassifier()
    reloaded.load_model(MODEL_PATH)
    row = X_test.iloc[[0]]
    a = model.predict_proba(row)[0, 1]
    b = reloaded.predict_proba(row)[0, 1]
    print(f"reload check: {a:.6f} vs {b:.6f} -> match: {np.isclose(a, b)}")


if __name__ == "__main__":
    main()