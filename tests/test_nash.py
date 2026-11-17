from src.utils import config
from src.game_theory.nash_solver import (supplier_utility, find_nash_offer,
                                         find_best_offer, buyer_utility,
                                         pareto_frontier)

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
RES, BEST = config.SUPPLIER_RESERVATION_VALUE, config.MAX_PRICE
BASE = {"delivery": 60, "warranty": 1, "payment": 30}


def su(price):
    return supplier_utility({**BASE, "price": price}, SUPPLIER_PARAMS, RES, BEST)


def test_supplier_utility_at_reservation():
    assert abs(su(880)) < 1e-9

def test_supplier_utility_at_best():
    assert abs(su(920) - 1.0) < 1e-9

def test_supplier_utility_midpoint():
    assert abs(su(900) - 0.5) < 1e-9

def test_nash_product_beats_expected_utility_offer():
    nash = find_nash_offer(PARAMS, SUPPLIER_PARAMS, RES, BEST, BOUNDS, seed=42)
    opt = find_best_offer(PARAMS, SUPPLIER_PARAMS, 880, 15, BOUNDS, seed=42)["offer"]
    opt_product = (buyer_utility(opt, PARAMS)
                   * supplier_utility(opt, SUPPLIER_PARAMS, RES, BEST))
    assert nash["product"] >= opt_product - 1e-6

def test_frontier_buyer_utility_falls_as_supplier_target_rises():
    pts = pareto_frontier(PARAMS, SUPPLIER_PARAMS, RES, BEST, BOUNDS,
                          [0.1, 0.3, 0.5, 0.7], seed=42)
    ubs = [p["buyer_u"] for p in pts]
    assert all(a >= b - 0.01 for a, b in zip(ubs, ubs[1:]))