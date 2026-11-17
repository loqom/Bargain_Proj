from src.utils import config
from src.game_theory.nash_solver import scan_prices, find_best_offer

PARAMS = {
    "ranges": {
        "price":    (config.MAX_PRICE, config.TARGET_PRICE),
        "delivery": (config.WORST_DELIVERY_DAYS, config.IDEAL_DELIVERY_DAYS),
        "warranty": (config.MIN_WARRANTY_YEARS, config.IDEAL_WARRANTY_YEARS),
        "payment":  (config.WORST_PAYMENT_DAYS, config.IDEAL_PAYMENT_DAYS),
    },
    "weights": config.WEIGHTS,
}

SUPPLIER_PARAMS = {
    "baseline": {"delivery": config.SUPPLIER_BASE_DELIVERY_DAYS,
                 "warranty": config.SUPPLIER_BASE_WARRANTY_YEARS,
                 "payment": config.SUPPLIER_BASE_PAYMENT_DAYS},
    "cost_rates": {"delivery": config.SUPPLIER_COST_PER_DAY_FASTER,
                   "warranty": config.SUPPLIER_COST_PER_EXTRA_WARRANTY_YEAR,
                   "payment": config.SUPPLIER_COST_PER_EXTRA_PAYMENT_DAY},
}

BOUNDS = {
    "price":    (config.MIN_OFFER_PRICE, config.MAX_PRICE),
    "delivery": (config.IDEAL_DELIVERY_DAYS, config.WORST_DELIVERY_DAYS),
    "warranty": (config.MIN_WARRANTY_YEARS, config.IDEAL_WARRANTY_YEARS),
    "payment":  (config.WORST_PAYMENT_DAYS, config.IDEAL_PAYMENT_DAYS),
}

IDEAL_TERMS = {"delivery": 30, "warranty": 3, "payment": 45}
RELAXED_TERMS = {"delivery": 45, "warranty": 2, "payment": 30}


def test_optimizer_beats_grid():
    best = find_best_offer(PARAMS, SUPPLIER_PARAMS, 880, 15, BOUNDS, seed=42)
    for terms in (IDEAL_TERMS, RELAXED_TERMS):
        rows = scan_prices(terms, PARAMS, SUPPLIER_PARAMS, 880, 15, 780, 920, 5)
        assert best["score"] >= max(r[4] for r in rows) - 1e-6