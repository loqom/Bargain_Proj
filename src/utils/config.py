"""
Central configuration for the DealForge Acme laptop scenario.
Constants only, no logic. Every other module imports from here.
"""

# ---------- Scenario ----------
QUANTITY = 100                # laptops
RANDOM_SEED = 42              # reproducible synthetic data

# ---------- Price (per unit, USD), buyer's perspective ----------
TARGET_PRICE = 850            # ideal price: utility is maximal here
MAX_PRICE = 920               # reservation price: buyer never pays above this
MIN_OFFER_PRICE = 780         # lowest opening offer worth making; below this
                              # the supplier rejects outright (lowball floor)
SUPPLIER_OPENING_ASK = 1200   # supplier's first message

# ---------- Non-price terms: buyer's ideal ----------
IDEAL_DELIVERY_DAYS = 30
WORST_DELIVERY_DAYS = 60      # utility = 0 at or beyond this
IDEAL_WARRANTY_YEARS = 3
MIN_WARRANTY_YEARS = 1        # utility = 0 at or below this
IDEAL_PAYMENT_DAYS = 45       # Net 45
WORST_PAYMENT_DAYS = 30       # Net 30

# ---------- Utility weights (must sum to 1) ----------
WEIGHTS = {
    "price": 0.60,
    "delivery": 0.20,
    "warranty": 0.15,
    "payment": 0.05,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Utility weights must sum to 1"

assert MIN_OFFER_PRICE < TARGET_PRICE < MAX_PRICE < SUPPLIER_OPENING_ASK


# ---------- Supplier cost model (mock) ----------
SUPPLIER_BASE_DELIVERY_DAYS = 60           # supplier's normal lead time
SUPPLIER_BASE_WARRANTY_YEARS = 1           # supplier's standard warranty
SUPPLIER_BASE_PAYMENT_DAYS = 30            # supplier's standard Net 30
SUPPLIER_COST_PER_DAY_FASTER = 0.8         # $/unit per day faster than baseline
SUPPLIER_COST_PER_EXTRA_WARRANTY_YEAR = 12 # $/unit per year beyond baseline
SUPPLIER_COST_PER_EXTRA_PAYMENT_DAY = 0.3  # $/unit per day of longer payment wait
SUPPLIER_RESERVATION_VALUE = 880   # supplier's walk-away: effective value below this = utility 0

GENERATOR = {
    "seed": 42,
    "n_negotiations": 10000,
    "max_rounds": 5,
    "output_path": "data/negotiations.csv",
 
    # Hidden supplier archetypes: (low, high) ranges sampled uniformly
    "archetypes": {
        "tough":    {"weight": 0.30, "reservation": (900, 940),
                     "concession": (0.010, 0.020), "flexibility": (0.0, 0.3)},
        "moderate": {"weight": 0.40, "reservation": (870, 910),
                     "concession": (0.020, 0.035), "flexibility": (0.3, 0.6)},
        "flexible": {"weight": 0.30, "reservation": (840, 890),
                     "concession": (0.035, 0.050), "flexibility": (0.6, 1.0)},
    },
    "opening_markup": (1.2, 1.5),     # first ask = reservation * markup
    "market_push": 20,                # tight market raises reservation by up to $20
    "ask_noise_sd": 0.005,            # noise on each round's concession
 
    # Hidden per-supplier cost rates (same meaning as supplier_params in nash_solver)
    "cost_rates": {"delivery": (0.3, 1.5), "warranty": (6, 18), "payment": (0.2, 1.0)},
    "baseline": {"delivery": 60, "warranty": 1, "payment": 30},
 
    # Random buyer offers (NOT from the solver, so the model sees wide coverage)
    "offer_ranges": {"price": (840, 930), "delivery": (30, 60),
                     "warranty": (1, 3), "payment": (30, 45)},
 
    # Acceptance rule: margin + N(0, noise_sd) >= -(flexibility * flex_allowance)
    "noise_sd": 8,
    "flex_allowance": 15,
 
    "target_price": 850,              # used for initial_ask_gap
}

