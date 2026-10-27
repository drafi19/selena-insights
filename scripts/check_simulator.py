"""Compare the synthetic corpus (L3) against the real Wallet series (L1).

The simulator is NOT fitted to the evaluation set. Only coarse invariants are gated; KS is
reported as a description. Run manually:

    uv run python -m scripts.check_simulator > docs/SIMULATOR.md
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

from selena_ml.config import SEED
from selena_ml.data.load import load_transactions
from selena_ml.data.synthetic import simulate_transactions

MIN_MEDIAN_RATIO = 0.3
MAX_MEDIAN_RATIO = 3.0
MAX_SMALL_SHARE_GAP = 0.15
MAX_WEEKEND_GAP = 0.40


def main() -> None:
    real = load_transactions("data/processed/spending.parquet")
    synth = simulate_transactions(n=5000, seed=SEED)

    real_med = float(real["amount"].median())
    synth_med = float(synth["amount"].median())
    ratio = synth_med / real_med

    real_small = float((real["amount"] < 10_000).mean())
    synth_small = float((synth["amount"] < 10_000).mean())

    real_wk = float(real.loc[real["date"].dt.dayofweek >= 5, "amount"].median())
    real_wd = float(real.loc[real["date"].dt.dayofweek < 5, "amount"].median())
    synth_wk = float(synth.loc[synth["date"].dt.dayofweek >= 5, "amount"].median())
    synth_wd = float(synth.loc[synth["date"].dt.dayofweek < 5, "amount"].median())

    ks = ks_2samp(np.log1p(real["amount"].to_numpy()), np.log1p(synth["amount"].to_numpy()))

    ok_ratio = MIN_MEDIAN_RATIO <= ratio <= MAX_MEDIAN_RATIO
    ok_small = abs(real_small - synth_small) <= MAX_SMALL_SHARE_GAP
    real_mult = real_wk / real_wd
    synth_mult = synth_wk / synth_wd
    ok_weekend = abs(synth_mult - real_mult) <= MAX_WEEKEND_GAP

    print("# Simulator validation (L3 vs the real Wallet series)\n")
    print("| Invariant | Real | Synthetic | Rule | Status |")
    print("|---|---|---|---|---|")
    print(
        f"| median amount | {real_med:,.0f} | {synth_med:,.0f} | ratio 0.3-3.0 | "
        f"{'OK' if ok_ratio else 'FAIL'} ({ratio:.2f}) |"
    )
    print(
        f"| share < 10k | {real_small:.3f} | {synth_small:.3f} | gap <= 0.15 | "
        f"{'OK' if ok_small else 'FAIL'} |"
    )
    print(
        f"| weekend multiplier | {real_mult:.2f}x | {synth_mult:.2f}x | "
        f"gap <= 0.40 | {'OK' if ok_weekend else 'FAIL'} |"
    )
    print(f"| KS log-amount (descriptive) | - | - | not gated | {ks.statistic:.3f} |")
    print()
    print("KS is not gated on purpose: pushing it down would mean fitting the generator to the")
    print("evaluation set, which is leakage. Only the three invariants above are enforced.")


if __name__ == "__main__":
    main()
