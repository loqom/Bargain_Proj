"""
Synthetic negotiation data generator.

Hidden traits (reservation, flexibility, cost rates, archetype) drive behaviour
but are written to the CSV as debug columns only. They must NEVER be used as
training features (leakage).

Run from the project root:
    python -m src.opponent_model.data_generator
"""


import os
import numpy as np
import pandas as pd

from scipy.stats import norm 
from src.utils.config import GENERATOR as G

FEATURES = ["concession_rate_observed", "initial_ask_gap", "rounds_completed",
            "market_context", "offer_vs_last_ask",
            "offer_price", "offer_delivery", "offer_warranty", "offer_payment"]
LABEL = "accepted"
DEBUG_COLS = ["negotiation_id", "hidden_archetype", "hidden_reservation",
              "hidden_flexibility"]


def _u(rng, bounds):
    return rng.uniform(bounds[0], bounds[1])


def sample_supplier(rng):
    """Draw one supplier: archetype -> hidden traits + observable market_context."""
    names = list(G["archetypes"])
    weights = [G["archetypes"][n]["weight"] for n in names]
    arch = rng.choice(names, p=weights)
    a = G["archetypes"][arch]

    market_context = rng.uniform(0, 1)
    reservation = _u(rng, a["reservation"]) + market_context * G["market_push"]
    return {
        "archetype": arch,
        "reservation": reservation,
        "concession": _u(rng, a["concession"]),
        "flexibility": _u(rng, a["flexibility"]),
        "opening_markup": _u(rng, G["opening_markup"]),
        "market_context": market_context,
        "cost": {k: _u(rng, v) for k, v in G["cost_rates"].items()},
    }


def ask_path(supplier, n_rounds, rng):
    """Supplier's asking price: ask_0 ... ask_n, sliding toward the reservation."""
    asks = [supplier["reservation"] * supplier["opening_markup"]]
    for _ in range(n_rounds):
        noise = rng.normal(0, G["ask_noise_sd"])
        nxt = asks[-1] * (1 - supplier["concession"] + noise)
        asks.append(max(supplier["reservation"], nxt))
    return asks


def random_offer(rng):
    return {k: _u(rng, v) for k, v in G["offer_ranges"].items()}


def supplier_effective_value(offer, supplier):
    """Price minus what the buyer's terms cost THIS supplier (hidden rates)."""
    base, c = G["baseline"], supplier["cost"]
    return (offer["price"]
            - c["delivery"] * max(0, base["delivery"] - offer["delivery"])
            - c["warranty"] * max(0, offer["warranty"] - base["warranty"])
            - c["payment"] * max(0, offer["payment"] - base["payment"]))


def accepts(offer, supplier, rng):
    """Noisy threshold with a flexibility allowance (not a sigmoid)."""
    margin = supplier_effective_value(offer, supplier) - supplier["reservation"]
    noisy = margin + rng.normal(0, G["noise_sd"])
    return int(noisy >= -(supplier["flexibility"] * G["flex_allowance"]))


def true_p_accept(offer, supplier):
    """Exact P(accept) behind accepts(): Phi((margin + flex*allowance) / noise_sd)."""
    margin = supplier_effective_value(offer, supplier) - supplier["reservation"]
    z = (margin + supplier["flexibility"] * G["flex_allowance"]) / G["noise_sd"]
    return float(norm.cdf(z))


def simulate_negotiation(neg_id, rng):
    """One supplier, 1..max_rounds probes. Returns a list of row dicts."""
    s = sample_supplier(rng)
    n_rounds = int(rng.integers(1, G["max_rounds"] + 1))
    asks = ask_path(s, n_rounds, rng)
    rows = []
    for r in range(1, n_rounds + 1):
        offer = random_offer(rng)
        rows.append({
            "concession_rate_observed": (asks[0] - asks[r]) / asks[0] / r,
            "initial_ask_gap": (asks[0] - G["target_price"]) / G["target_price"],
            "rounds_completed": r,
            "market_context": s["market_context"],
            "offer_vs_last_ask": offer["price"] / asks[r],
            "offer_price": offer["price"],
            "offer_delivery": offer["delivery"],
            "offer_warranty": offer["warranty"],
            "offer_payment": offer["payment"],
            LABEL: accepts(offer, s, rng),
            "negotiation_id": neg_id,
            "hidden_archetype": s["archetype"],
            "hidden_reservation": s["reservation"],
            "hidden_flexibility": s["flexibility"],
        })
    return rows

def build_dataset(seed=None):
    rng = np.random.default_rng(G["seed"] if seed is None else seed)
    rows = []
    for i in range(G["n_negotiations"]):
        rows.extend(simulate_negotiation(i, rng))
    return pd.DataFrame(rows)[FEATURES + [LABEL] + DEBUG_COLS]


def run_checks(df):
    print(f"rows: {len(df)}, negotiations: {df['negotiation_id'].nunique()}")
    print(f"NaNs: {int(df.isna().sum().sum())}")
    print(f"overall acceptance rate: {df[LABEL].mean():.1%}  (want 25-45%)")
    print("\nacceptance by archetype (want flexible > moderate > tough):")
    print(df.groupby("hidden_archetype")[LABEL].mean().round(3))
    print("\nacceptance by offer-price bin (want rising):")
    bins = pd.cut(df["offer_price"], [0, 860, 900, 940, 10_000])
    print(df.groupby(bins, observed=True)[LABEL].mean().round(3))


def main():
    df = build_dataset()
    os.makedirs(os.path.dirname(G["output_path"]), exist_ok=True)
    df.to_csv(G["output_path"], index=False)
    print(f"saved {G['output_path']}\n")
    run_checks(df)
    # reproducibility check
    same = df.equals(build_dataset())
    print(f"\nsame seed -> identical data: {same}")


if __name__ == "__main__":
    main()