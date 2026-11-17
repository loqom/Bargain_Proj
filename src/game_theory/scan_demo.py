from src.utils import config
from src.game_theory.nash_solver import find_best_offer_ml
from src.game_theory.nash_solver import (scan_prices, find_best_offer,find_nash_offer, pareto_frontier)

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

IDEAL_TERMS = {"delivery": 30, "warranty": 3, "payment": 45}
RELAXED_TERMS = {"delivery": 45, "warranty": 2, "payment": 30}


def run(label, terms):
    rows = scan_prices(terms, PARAMS, SUPPLIER_PARAMS,
                       supplier_reservation=880, scale=15,
                       low=780, high=920, step=5)
    print(f"\n=== {label}: {terms} ===")
    print(f"{'price':>6} {'utility':>8} {'sup_val':>8} {'P(acc)':>8} {'score':>8}")
    for price, u, sv, p, s in rows:
        print(f"{price:>6} {u:>8.3f} {sv:>8.1f} {p:>8.3f} {s:>8.3f}")
    best = max(rows, key=lambda r: r[4])
    print(f"Best price: ${best[0]} (score {best[4]:.3f})")


run("IDEAL terms", IDEAL_TERMS)
run("RELAXED terms", RELAXED_TERMS)

BOUNDS = {
    "price":    (config.MIN_OFFER_PRICE, config.MAX_PRICE),
    "delivery": (config.IDEAL_DELIVERY_DAYS, config.WORST_DELIVERY_DAYS),
    "warranty": (config.MIN_WARRANTY_YEARS, config.IDEAL_WARRANTY_YEARS),
    "payment":  (config.WORST_PAYMENT_DAYS, config.IDEAL_PAYMENT_DAYS),
}

result = find_best_offer(PARAMS, SUPPLIER_PARAMS, 880, 15, BOUNDS,
                         seed=config.RANDOM_SEED)
print("\n=== OPTIMIZER ===")
for k, v in result["offer"].items():
    print(f"{k:>9}: {v:.2f}")
print(f"utility {result['utility']:.3f} | P(accept) {result['p_accept']:.3f} "
      f"| score {result['score']:.3f}")

RES = config.SUPPLIER_RESERVATION_VALUE
BEST = config.MAX_PRICE      # supplier's best case: max price, baseline terms

print("\n=== PARETO FRONTIER ===")
print(f"{'target Us':>9} {'price':>7} {'deliv':>6} {'warr':>5} {'pay':>5} "
      f"{'Ub':>6} {'Us':>6}")
for pt in pareto_frontier(PARAMS, SUPPLIER_PARAMS, RES, BEST, BOUNDS,
                          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                          seed=config.RANDOM_SEED):
    o = pt["offer"]
    print(f"{pt['target']:>9.1f} {o['price']:>7.1f} {o['delivery']:>6.1f} "
          f"{o['warranty']:>5.2f} {o['payment']:>5.1f} "
          f"{pt['buyer_u']:>6.3f} {pt['supplier_u']:>6.3f}")

nash = find_nash_offer(PARAMS, SUPPLIER_PARAMS, RES, BEST, BOUNDS,
                       seed=config.RANDOM_SEED)
print("\n=== NASH BARGAINING OFFER ===")
for k, v in nash["offer"].items():
    print(f"{k:>9}: {v:.2f}")
print(f"buyer U {nash['buyer_u']:.3f} | supplier U {nash['supplier_u']:.3f} "
      f"| product {nash['product']:.4f}")

# Compare with the earlier expected-utility optimizer
opt_us = (result["offer"]["price"] - 4.5 - RES) / (BEST - RES)
print(f"\nExpected-utility offer: buyer U {result['utility']:.3f}, "
      f"supplier U ~{opt_us:.3f}")


state = {"concession_rate_observed": 0.025, "initial_ask_gap": 350/850,
         "rounds_completed": 2, "market_context": 0.5, "last_ask": 1140.0}
bounds = {"price": (850, 920), "delivery": (30, 60),
          "warranty": (1, 3), "payment": (30, 45)}

for seed in range(5):
    print(seed, find_best_offer_ml(PARAMS, state, bounds, seed=seed))