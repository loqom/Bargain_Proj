import numpy as np
import pandas as pd

from src.opponent_model.data_generator import (
    G, FEATURES, sample_supplier, ask_path, true_p_accept)

PRICE_MIN, PRICE_MAX = 850, 920
GRID = np.linspace(PRICE_MIN, PRICE_MAX, 701)
# ADAPT A: non-price terms held fixed (match the units in G["offer_ranges"])
TERMS = {"delivery": 35, "warranty": 2, "payment": 45}

def buyer_utility(price):
    # ADAPT B: replace with the real utility from nash_solver.py
    return (PRICE_MAX - price) / (PRICE_MAX - PRICE_MIN)

def load_model():
    # ADAPT C: load your trained XGBoost exactly as predict.py does
    raise NotImplementedError
MODEL = None   # MODEL = load_model()

def make_case(rng):
    """One supplier + the observable state at a random round."""
    s = sample_supplier(rng)
    r = int(rng.integers(1, G["max_rounds"] + 1))
    asks = ask_path(s, r, rng)
    return dict(
        s=s, last_ask=asks[r],
        concession_rate_observed=(asks[0] - asks[r]) / asks[0] / r,
        initial_ask_gap=(asks[0] - G["target_price"]) / G["target_price"],
        rounds_completed=r, market_context=s["market_context"])

def true_p(case):
    return np.array([true_p_accept({"price": p, **TERMS}, case["s"]) for p in GRID])

def model_p(case):
    df = pd.DataFrame({
        "concession_rate_observed": case["concession_rate_observed"],
        "initial_ask_gap": case["initial_ask_gap"],
        "rounds_completed": case["rounds_completed"],
        "market_context": case["market_context"],
        "offer_vs_last_ask": GRID / case["last_ask"],
        "offer_price": GRID,
        "offer_delivery": TERMS["delivery"],
        "offer_warranty": TERMS["warranty"],
        "offer_payment": TERMS["payment"]})[FEATURES]
    return MODEL.predict_proba(df)[:, 1]

def sigmoid_p(case):
    return 1.0 / (1.0 + np.exp(-(GRID - 880) / 8))   # your old mock; match its real form

def run(n=200, seed=0):
    rng = np.random.default_rng(seed)
    util = np.array([buyer_utility(p) for p in GRID])
    rows = []
    for _ in range(n):
        case = make_case(rng)
        tp = true_p(case)
        out = {}
        for name, belief in [("oracle", tp), ("xgb", model_p(case)), ("sigmoid", sigmoid_p(case))]:
            i = (util * belief).argmax()
            out[name] = dict(price=GRID[i], believed=belief[i], true_score=util[i] * tp[i], true_p=tp[i])
        rows.append(dict(
            oracle=out["oracle"]["true_score"],
            xgb=out["xgb"]["true_score"],
            sigmoid=out["sigmoid"]["true_score"],
            xgb_regret=out["oracle"]["true_score"] - out["xgb"]["true_score"],
            sigmoid_regret=out["oracle"]["true_score"] - out["sigmoid"]["true_score"],
            xgb_price_err=out["xgb"]["price"] - out["oracle"]["price"],
            xgb_overconf=out["xgb"]["believed"] - out["xgb"]["true_p"]))
    df = pd.DataFrame(rows)
    print(df.mean().round(4).to_string())
    df.to_csv("data/oracle_benchmark.csv", index=False)


def load_model():
    import xgboost as xgb
    m = xgb.XGBClassifier()
    m.load_model("models/opponent_model.json") 
    return m

if __name__ == "__main__":
    MODEL = load_model()
    run()
