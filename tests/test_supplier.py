from src.utils import config
from src.game_theory.nash_solver import supplier_value

SUPPLIER_PARAMS = {
    "baseline": {"delivery": config.SUPPLIER_BASE_DELIVERY_DAYS,
                 "warranty": config.SUPPLIER_BASE_WARRANTY_YEARS,
                 "payment": config.SUPPLIER_BASE_PAYMENT_DAYS},
    "cost_rates": {"delivery": config.SUPPLIER_COST_PER_DAY_FASTER,
                   "warranty": config.SUPPLIER_COST_PER_EXTRA_WARRANTY_YEAR,
                   "payment": config.SUPPLIER_COST_PER_EXTRA_PAYMENT_DAY},
}

def test_baseline_terms_cost_nothing():
    offer = {"price": 900, "delivery": 60, "warranty": 1, "payment": 30}
    assert supplier_value(offer, SUPPLIER_PARAMS) == 900

def test_ideal_terms_reduce_value():
    offer = {"price": 900, "delivery": 30, "warranty": 3, "payment": 45}
    assert abs(supplier_value(offer, SUPPLIER_PARAMS) - 847.5) < 1e-9

def test_easier_than_baseline_is_not_a_bonus():
    offer = {"price": 900, "delivery": 90, "warranty": 1, "payment": 30}
    assert supplier_value(offer, SUPPLIER_PARAMS) == 900