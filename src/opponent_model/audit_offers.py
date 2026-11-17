"""
Ground-truth audit: how often is each offer REALLY accepted by simulated
suppliers that look like the Acme state (not just what a model predicts)?

Run from the project root:
    python -m src.opponent_model.audit_offers
"""
import numpy as np

from src.opponent_model.data_generator import sample_supplier, ask_path, accepts
from src.opponent_model.predict import predict_accept

STATE = {"concession_rate_observed": 0.025,
         "initial_ask_gap": 350 / 850,
         "rounds_completed": 2,
         "market_context": 0.5,
         "last_ask": 1140.0}

# Offers and buyer utilities taken from the solver runs
OFFERS = {
    "sigmoid optimizer": ({"price": 889.67, "delivery": 60.0,
                           "warranty": 1.0, "payment": 45.0}, 0.310),
    "XGBoost optimizer": ({"price": 900.98, "delivery": 59.71,
                           "warranty": 1.51, "payment": 38.00}, 0.2298),
}

N_SIM = 150_000
rng = np.random.default_rng(123)       # different seed from the training data

matched = []
for _ in range(N_SIM):
    s = sample_supplier(rng)
    asks = ask_path(s, 2, rng)
    conc = (asks[0] - asks[2]) / asks[0] / 2
    # keep suppliers whose OBSERVABLE state resembles the Acme state
    if abs(conc - STATE["concession_rate_observed"]) < 0.005 \
            and abs(asks[2] - STATE["last_ask"]) < 40 \
            and abs(s["market_context"] - STATE["market_context"]) < 0.25:
        matched.append(s)

print(f"matching simulated suppliers: {len(matched)}\n")
print(f"{'offer':20s} {'model P':>8s} {'true P':>8s} {'utility':>8s} "
      f"{'true score':>11s}")
for name, (offer, util) in OFFERS.items():
    model_p = predict_accept(offer, STATE)
    true_p = np.mean([accepts(offer, s, rng) for s in matched])
    print(f"{name:20s} {model_p:8.3f} {true_p:8.3f} {util:8.3f} "
          f"{util * true_p:11.3f}")