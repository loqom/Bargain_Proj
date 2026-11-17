"""
P(supplier accepts) from the trained XGBoost opponent model.

predict_accept(offer, state) works for ONE offer (scalars) or MANY offers
(numpy arrays in offer values), so the optimizer can score a whole batch
of candidates in one model call.

offer : {"price", "delivery", "warranty", "payment"}
state : {"concession_rate_observed", "initial_ask_gap", "rounds_completed",
         "market_context", "last_ask"}   (all observable mid-negotiation)

Sanity check:  python -m src.opponent_model.predict
"""
import json

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

MODEL_PATH = "models/opponent_model.json"
COLS_PATH = "models/feature_columns.json"

_cache = {}


def _load():
    if not _cache:
        model = XGBClassifier()
        model.load_model(MODEL_PATH)
        with open(COLS_PATH) as f:
            _cache["cols"] = json.load(f)
        _cache["model"] = model
    return _cache["model"], _cache["cols"]


def predict_accept(offer, state):
    model, cols = _load()
    price = np.atleast_1d(np.asarray(offer["price"], dtype=float))
    scalar = np.ndim(offer["price"]) == 0

    row = {
        "concession_rate_observed": state["concession_rate_observed"],
        "initial_ask_gap": state["initial_ask_gap"],
        "rounds_completed": state["rounds_completed"],
        "market_context": state["market_context"],
        "offer_vs_last_ask": price / state["last_ask"],
        "offer_price": price,
        "offer_delivery": np.broadcast_to(
            np.asarray(offer["delivery"], dtype=float), price.shape),
        "offer_warranty": np.broadcast_to(
            np.asarray(offer["warranty"], dtype=float), price.shape),
        "offer_payment": np.broadcast_to(
            np.asarray(offer["payment"], dtype=float), price.shape),
    }
    X = pd.DataFrame(row)[cols]          # saved training column order
    p = model.predict_proba(X)[:, 1]
    return float(p[0]) if scalar else p


if __name__ == "__main__":
    # Acme example: supplier opened at $1,200 and has come down to $1,140
    state = {"concession_rate_observed": 0.025,
             "initial_ask_gap": (1200 - 850) / 850,
             "rounds_completed": 2,
             "market_context": 0.5,
             "last_ask": 1140.0}
    terms = {"delivery": 45, "warranty": 2, "payment": 30}

    prices = np.arange(840, 961, 10, dtype=float)
    p = predict_accept({"price": prices, **terms}, state)
    print("price  P(accept)")
    for pr, pp in zip(prices, p):
        print(f"{pr:5.0f}  {pp:.3f}")

    drops = int(np.sum(np.diff(p) < -1e-9))
    print(f"\nnon-monotonic steps along price: {drops} (want 0)")

    base = {"price": 900.0, "delivery": 45, "warranty": 2, "payment": 30}
    print("\nterm effects at $900 (supplier-side expectation in brackets):")
    for label, change, expect in [
        ("faster delivery (30d)", {"delivery": 30}, "lower"),
        ("longer warranty (3y)", {"warranty": 3}, "lower"),
        ("longer payment (Net 45)", {"payment": 45}, "lower"),
    ]:
        alt = predict_accept({**base, **change}, state)
        print(f"  {label:26s} {predict_accept(base, state):.3f} -> {alt:.3f}"
              f"  [{expect}]")

    low = predict_accept(base, {**state, "concession_rate_observed": 0.005})
    high = predict_accept(base, {**state, "concession_rate_observed": 0.045})
    print(f"\nconcession 0.5% vs 4.5% at $900: {low:.3f} vs {high:.3f}"
          "  [flexible supplier should be higher]")