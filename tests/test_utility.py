from src.utils import config
from src.game_theory.nash_solver import normalize, buyer_utility

PARAMS = {
    "ranges": {
        "price":    (config.MAX_PRICE, config.TARGET_PRICE),
        "delivery": (config.WORST_DELIVERY_DAYS, config.IDEAL_DELIVERY_DAYS),
        "warranty": (config.MIN_WARRANTY_YEARS, config.IDEAL_WARRANTY_YEARS),
        "payment":  (config.WORST_PAYMENT_DAYS, config.IDEAL_PAYMENT_DAYS),
    },
    "weights": config.WEIGHTS,
}

PERFECT = {"price": 850, "delivery": 30, "warranty": 3, "payment": 45}
WORST   = {"price": 920, "delivery": 60, "warranty": 1, "payment": 30}
COUNTER = {"price": 900, "delivery": 35, "warranty": 2, "payment": 45}


def approx(a, b, tol=1e-3):
    return abs(a - b) < tol


def test_normalize_lower_is_better():
    assert approx(normalize(900, 920, 850), 0.286)


def test_normalize_higher_is_better():
    assert approx(normalize(2, 1, 3), 0.5)


def test_normalize_clamps():
    assert normalize(800, 920, 850) == 1.0
    assert normalize(1200, 920, 850) == 0.0


def test_perfect_offer():
    assert approx(buyer_utility(PERFECT, PARAMS), 1.0)


def test_worst_offer():
    assert approx(buyer_utility(WORST, PARAMS), 0.0)


def test_counter_offer():
    assert approx(buyer_utility(COUNTER, PARAMS), 0.463)