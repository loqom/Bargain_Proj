# DealForge: Game Theory Formulation (Week 1)

**Status:** Week 1 complete. All results below are reproducible with `python -m src.game_theory.scan_demo`, and 15 tests pass.
**Scope:** one buyer, one supplier, one round, four issues. The supplier side is a *mock* model with assumed parameters, to be replaced by the learned opponent model in Weeks 2 to 4.

---

## 1. Problem setup

Acme Corp (buyer, B) negotiates with a supplier (S) over 100 laptops. An offer is a vector of four issues:

$$x = (p,\ d,\ w,\ n)$$

where $p$ is the unit price in USD, $d$ the delivery time in days, $w$ the warranty in years, and $n$ the payment term in days (Net $n$). Each side has a different idea of a good deal, so the task is to choose an $x$ that is good for the buyer and still likely to be accepted.

## 2. Buyer utility

Each issue is mapped to a score in $[0,1]$, where 1 is the buyer's ideal and 0 is the worst value it will accept:

$$s(v;\ \text{worst},\ \text{ideal}) = \mathrm{clip}_{[0,1]}\!\left(\frac{v - \text{worst}}{\text{ideal} - \text{worst}}\right)$$

The same formula covers "lower is better" issues (price, delivery) and "higher is better" issues (warranty, payment), because the sign cancels when worst and ideal are swapped. Values beyond the ideal are clipped, so every issue lives on the same scale and the weights mean what they say.

| Issue | Worst | Ideal | Weight |
|---|---|---|---|
| Price | $920 | $850 | 0.60 |
| Delivery | 60 days | 30 days | 0.20 |
| Warranty | 1 year | 3 years | 0.15 |
| Payment | Net 30 | Net 45 | 0.05 |

Buyer utility is the weighted sum:

$$U_B(x) = 0.60\,s_p + 0.20\,s_d + 0.15\,s_w + 0.05\,s_n$$

**Worked example.** The counter-offer $x = (900, 35, 2, 45)$ gives $s_p = 20/70 = 0.286$, $s_d = 25/30 = 0.833$, $s_w = 0.5$, $s_n = 1.0$, so $U_B = 0.171 + 0.167 + 0.075 + 0.05 = 0.463$.

## 3. Supplier model

The supplier dislikes fast delivery, long warranties and long payment waits, because each costs it money. Its **effective value** of an offer is the price minus those costs:

$$v_S(x) = p - 0.8\max(0,\,60-d) - 12\max(0,\,w-1) - 0.3\max(0,\,n-30)$$

The baseline terms (60 days, 1 year, Net 30) cost nothing, and terms easier than baseline give no bonus. The rates (\$0.80 per day faster, \$12 per extra warranty year, \$0.30 per extra payment day) are **assumptions**, not estimates from data.

Supplier utility uses the same scoring function, with a reservation value of \$880 (below this the supplier gets 0) and a best case of \$920 (maximum price, baseline terms):

$$U_S(x) = s\big(v_S(x);\ 880,\ 920\big)$$

**Example.** For the counter-offer $(900, 35, 2, 45)$ the costs are $20 + 12 + 4.5 = 36.5$, so $v_S = 863.50$. That is below the supplier's reservation of \$880, so $U_S = 0$ and the acceptance probability is only about 0.25 even though the headline price is \$900.

## 4. Acceptance model and objective

The probability that the supplier accepts is modelled as a sigmoid of its effective value, equal to 0.5 at the reservation:

$$A(x) = \frac{1}{1 + \exp\!\big(-(v_S(x) - 880)/15\big)}$$

This is a **placeholder**. In Week 2 it is replaced by an XGBoost model trained on synthetic negotiations, with the same inputs and the same output (a probability), so nothing downstream changes.

The buyer's decision rule balances how good the offer is against how likely it is to be accepted:

$$\text{Score}(x) = U_B(x) \cdot A(x)$$

A great offer that is always rejected scores near zero, and a sure deal that is poor for the buyer also scores low.

## 5. Solution concepts

Because `normalize` clips values, the objectives are piecewise linear with flat regions, so gradient-based solvers can stall. All three searches use SciPy's `differential_evolution` (a global, gradient-free method) over bounds on all four issues, with a fixed seed for reproducibility.

**(a) Expected-utility optimum.** Maximize $\text{Score}(x)$ over $x$. This is the buyer's single-sided best response to the supplier model.

**(b) Pareto frontier.** An offer is Pareto-optimal if neither side can gain without the other losing. A weighted sum $\lambda U_B + (1-\lambda) U_S$ is the usual way to trace the frontier, but it fails here: both utilities are linear in price over the relevant range, so the sum is minimized at one end for any $\lambda$ and only the two endpoints are returned. Instead we use the **epsilon-constraint method**: for each target $t$, solve

$$\max_x\ U_B(x)\quad \text{subject to}\quad U_S(x) \ge t$$

and record the result for $t = 0.1, 0.2, \dots, 0.8$.

**(c) Nash bargaining solution.** Choose the offer that maximizes the product of both sides' gains over their disagreement points:

$$x^{\text{Nash}} = \arg\max_x\ \big(U_B(x) - d_B\big)\big(U_S(x) - d_S\big)$$

We set $d_B = d_S = 0$, meaning "no deal" is the worst outcome for both. This is a modelling choice. A realistic disagreement point is each side's best alternative (for the buyer, the next-best supplier), which we do not model yet.

## 6. Results

**Offers found** (continuous values from the optimizer; a real system would round them):

| Method | Price | Delivery | Warranty | Payment | $U_B$ | $U_S$ |
|---|---|---|---|---|---|---|
| Expected-utility optimum | \$889.67 | 60 days | 1 year | Net 45 | 0.310 | 0.129 |
| Nash bargaining | \$905.17 | 60 days | 1 year | Net 45 | 0.177 | 0.517 |

The expected-utility optimum has acceptance probability 0.585 and Score 0.181, which beats both hand-picked grid scans (ideal terms: 0.121 at \$920; relaxed terms of 45 days, 2 years, Net 30: 0.157 at \$905). Asking for every ideal term backfires: the supplier's effective value stays so low that acceptance never exceeds about 30%.

**Pareto frontier** (every point uses 60 days, 1 year, Net 45):

| Target $U_S$ | 0.1 | 0.3 | 0.5 | 0.8 |
|---|---|---|---|---|
| Price | \$888.50 | \$896.50 | \$904.50 | \$916.50 |
| $U_B$ | 0.320 | 0.251 | 0.183 | 0.080 |

Along the frontier $U_B$ falls as $U_S$ rises. With only price varying, $U_B = 0.05 + 0.6(920-p)/70$ and $U_S = (p - 884.5)/40$. Maximizing their product gives $p^* = \tfrac12\!\left(\tfrac{7.9357}{0.008571} + 884.5\right) \approx 905.17$, which matches the solver and confirms the Nash result analytically.

## 7. Limitations

1. **The supplier is assumed, not learned.** The reservation (\$880), the sigmoid steepness (15) and all cost rates are made-up values. Every result above is a property of this model.
2. **Only payment terms are worth conceding.** With these cost rates, warranty and delivery cost the supplier more (in the buyer's price-equivalent units) than they are worth to the buyer, so the optimizer drops them and the frontier collapses to a line in price. Lowering the warranty cost to roughly \$6 per year should change that. A sensitivity analysis over the cost rates is planned.
3. **This is not a Nash equilibrium of a full game.** We compute a Nash *bargaining* solution plus a best response to a fixed opponent model. There is no strategic interaction over multiple rounds yet.
4. **The disagreement point is zero for both sides** (section 5c), and the supplier's reservation value is used both in the acceptance curve and in $U_S$, which ties the two together.
5. **Utility is additive across issues**, so interactions (for example, a longer warranty mattering more when delivery is slow) are not captured.
6. **The Week 1 success criterion of reaching 75% of ideal utility is not met** (the best offer has $U_B = 0.31$). Against this supplier model a 75%-utility offer would almost never be accepted, so the target should be revisited once the opponent model is learned from data.
7. **Offers are continuous.** Rounding to realistic values (a whole-dollar price, whole days) will change the numbers slightly.

## 8. Next steps

| Week | Task |
|---|---|
| 2 | Build the synthetic negotiation generator with varied opponent profiles (concession rate, aggressiveness, flexibility) |
| 2 to 4 | Train the XGBoost acceptance model and swap it in for the sigmoid in `nash_solver.py` |
| 4 | Validate the model against external data, since a model tested on its own generator will look perfect without proving anything |
| Later | Sensitivity analysis over cost rates, a realistic disagreement point, and multi-round dynamics |