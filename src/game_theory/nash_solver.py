import math
import warnings
import numpy as np
from src.opponent_model.predict import predict_accept
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
from scipy.optimize import differential_evolution, NonlinearConstraint
TERMS = ["price", "delivery", "warranty", "payment"]
def normalize(value, worst, ideal):
    """
    Map a term to a 0-1 score: 1.0 at `ideal`, 0.0 at `worst`, clamped.
    Works in both directions because the sign cancels out:
      lower-is-better (price): worst > ideal
      higher-is-better (warranty): worst < ideal
    """

    if(worst==ideal) :
        raise ValueError("worst and ideal must differ")
    score=(value-worst)/(ideal-worst)
    return max(0.0, min(1.0, score))

def buyer_utility(offer, params):
    """
    offer:  {"price": 900, "delivery": 35, "warranty": 2, "payment": 45}
    params: {"ranges":  {term: (worst, ideal), ...},
             "weights": {term: weight, ...}}
    Returns the weighted sum of normalized scores, in [0, 1].
    """
    total = 0.0
    for term, weight in params["weights"].items():
        worst, ideal = params["ranges"][term]
        total += weight * normalize(offer[term], worst, ideal)
    return total

def mock_acceptance_prob(price, supplier_reservation, scale):
    """
    Mock P(supplier accepts) at a given price. Sigmoid, increasing in price:
    higher price = better for the supplier = more likely to accept.
    0.5 exactly at `supplier_reservation`. Replaced by XGBoost later.
    """
    z = (price - supplier_reservation) / scale
    return 1.0 / (1.0 + math.exp(-z))


def supplier_value(offer, supplier_params):
    """
    Effective value of an offer to the supplier: price minus what the
    buyer's terms cost them. Terms at or easier than the supplier's
    baseline cost nothing (clipped at 0, so easy terms are not a bonus).
    """
    base = supplier_params["baseline"]
    rate = supplier_params["cost_rates"]
    delivery_cost = rate["delivery"] * max(0, base["delivery"] - offer["delivery"])
    warranty_cost = rate["warranty"] * max(0, offer["warranty"] - base["warranty"])
    payment_cost = rate["payment"] * max(0, offer["payment"] - base["payment"])
    return offer["price"] - delivery_cost - warranty_cost - payment_cost


def scan_prices(fixed_terms, params, supplier_params, supplier_reservation,
                scale, low, high, step):
    """
    Sweep price from low to high, holding non-price terms fixed.
    Returns (price, utility, supplier_value, p_accept, score) tuples.
    Acceptance now depends on the supplier's effective value, not raw price.
    """
    rows = []
    price = low
    while price <= high:
        offer = {**fixed_terms, "price": price}
        u = buyer_utility(offer, params)
        sv = supplier_value(offer, supplier_params)
        p = mock_acceptance_prob(sv, supplier_reservation, scale)
        rows.append((price, u, sv, p, u * p))
        price += step
    return rows

def score_offer(offer, params, supplier_params, supplier_reservation, scale):
    """Score = buyer utility x P(supplier accepts). Returns (score, u, p)."""
    u = buyer_utility(offer, params)
    sv = supplier_value(offer, supplier_params)
    p = mock_acceptance_prob(sv, supplier_reservation, scale)
    return u * p, u, p


def find_best_offer(params, supplier_params, supplier_reservation, scale,
                    bounds, seed=0):
    """
    Maximize score over all four terms at once.
    bounds: {term: (low, high)}. Returns the best offer and its numbers.
    """
    def objective(x):
        offer = dict(zip(TERMS, x))
        score, _, _ = score_offer(offer, params, supplier_params,
                                  supplier_reservation, scale)
        return -score          # SciPy minimizes, so negate

    result = differential_evolution(objective, [bounds[t] for t in TERMS],
                                    seed=seed, tol=1e-8)
    best = dict(zip(TERMS, result.x))
    score, u, p = score_offer(best, params, supplier_params,
                              supplier_reservation, scale)
    return {"offer": best, "score": score, "utility": u, "p_accept": p}

def supplier_utility(offer, supplier_params, reservation_value, best_value):
    """
    Supplier's 0-1 utility. 0 at their reservation value (walk-away),
    1 at best_value (e.g. max price with baseline terms). Reuses normalize().
    """
    sv = supplier_value(offer, supplier_params)
    return normalize(sv, reservation_value, best_value)


def find_nash_offer(params, supplier_params, reservation_value, best_value,
                    bounds, seed=0):
    """
    Nash bargaining solution: maximize buyer_utility x supplier_utility.
    Disagreement point is 0 for both sides (no deal = worst outcome).
    """
    def objective(x):
        offer = dict(zip(TERMS, x))
        ub = buyer_utility(offer, params)
        us = supplier_utility(offer, supplier_params, reservation_value, best_value)
        return -(ub * us)

    result = differential_evolution(objective, [bounds[t] for t in TERMS],
                                    seed=seed, tol=1e-8)
    offer = dict(zip(TERMS, result.x))
    ub = buyer_utility(offer, params)
    us = supplier_utility(offer, supplier_params, reservation_value, best_value)
    return {"offer": offer, "buyer_u": ub, "supplier_u": us, "product": ub * us}


def pareto_frontier(params, supplier_params, reservation_value, best_value,
                    bounds, targets, seed=0):
    """
    For each target supplier utility t: maximize buyer utility subject to
    supplier utility >= t. Returns a list of frontier points.
    """
    points = []
    for t in targets:
        def us_of(x):
            return supplier_utility(dict(zip(TERMS, x)), supplier_params,
                                    reservation_value, best_value)

        def objective(x):
            return -buyer_utility(dict(zip(TERMS, x)), params)

        constraint = NonlinearConstraint(us_of, t, float("inf"))
        result = differential_evolution(objective, [bounds[k] for k in TERMS],
                                        constraints=(constraint,), seed=seed,
                                        tol=1e-8)
        if not result.success:
            continue                    # target not reachable, skip it
        offer = dict(zip(TERMS, result.x))
        points.append({"target": t, "offer": offer,
                       "buyer_u": buyer_utility(offer, params),
                       "supplier_u": us_of(result.x)})
    return points

def buyer_utility_batch(offer, params):
    """
    Same maths as buyer_utility, but each offer[term] may be a numpy array
    (one entry per candidate). Needed by the vectorized optimizer.
    """
    total = 0.0
    for term, weight in params["weights"].items():
        worst, ideal = params["ranges"][term]
        score = (np.asarray(offer[term], dtype=float) - worst) / (ideal - worst)
        total = total + weight * np.clip(score, 0.0, 1.0)
    return total


def score_offer_ml(offer, params, opponent_state):
    """Score = buyer utility x P(accept) from the trained model. Returns (score, u, p)."""
    u = float(buyer_utility_batch(offer, params))
    p = float(predict_accept(offer, opponent_state))
    return u * p, u, p


def find_best_offer_ml(params, opponent_state, bounds, seed=0):
    """
    Maximize utility x P(accept) over all four terms, with P(accept) coming
    from the XGBoost opponent model.

    vectorized=True hands the objective ALL candidates of a generation at once
    (x has shape (4, n_candidates)), so the model runs once per generation
    instead of once per candidate.

    Keep `bounds` inside the ranges the model was trained on (trees cannot
    extrapolate): price 850-960, delivery 30-60, warranty 1-3, payment 30-45.
    """
    def objective(x):
        offer = {t: x[i] for i, t in enumerate(TERMS)}
        u = buyer_utility_batch(offer, params)
        p = predict_accept(offer, opponent_state)
        return -(u * p)

    result = differential_evolution(objective, [bounds[t] for t in TERMS],
                                    seed=seed, tol=1e-8,
                                    vectorized=True, updating="deferred")
    best = {t: float(v) for t, v in zip(TERMS, result.x)}
    score, u, p = score_offer_ml(best, params, opponent_state)
    return {"offer": best, "score": score, "utility": u, "p_accept": p}