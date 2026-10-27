# Simulator validation (L3 vs the real Wallet series)

| Invariant | Real | Synthetic | Rule | Status |
|---|---|---|---|---|
| median amount | 12,000 | 12,000 | ratio 0.3-3.0 | OK (1.00) |
| share < 10k | 0.343 | 0.430 | gap <= 0.15 | OK |
| weekend multiplier | 0.77x | 0.83x | gap <= 0.40 | OK |
| KS log-amount (descriptive) | - | - | not gated | 0.095 |

KS is not gated on purpose: pushing it down would mean fitting the generator to the
evaluation set, which is leakage. Only the three invariants above are enforced.
