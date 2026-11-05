# ADR 0003 - score the amount channel, and evaluate features with their history

Status: accepted (v0.4.0, November 2026). Supersedes nothing; amends ADR 0002 where it concerns
which z-score features are scored.

## Context

v0.3.0 shipped a z-score whose score was `max |z|` over all seven features. Manual review of its 30
highest flags found that 25 of them were driven by `days_since_prev`, and because that feature is
clipped at 60 days many rows collapsed onto identical scores: 24 of the top 30 were ties. A long gap
is a holiday, not an anomaly, so the flag list was describing my calendar rather than my spending.

While measuring a fix, a second defect surfaced. `scripts/eval_protocol.py` rebuilt features
separately on train, validation and test. The rolling features need five prior rows
(`min_periods=5`), so the first rows of every slice carried `amount_z_30 = 0` and any injected spike
inside a 38-row test slice became invisible. The evaluation was understating every amount-based
feature by construction.

## Decision

1. **Score only the amount channel.** `SCORED_FEATURES = ["amount_z_30", "amount_z_category_30"]`.
   `FEATURES` stays at seven for the autoencoder; the gap is still computed and still shown in the
   manual review sheet as `gap_days`, but it no longer enters the score.
2. **Evaluate with history.** Features are built once over the full series and sliced by split
   position, and every injection replicate is scored on (real history + injected rows). This is not
   leakage: features are causal, and the history precedes the test window. The cold path is kept in
   the report as appendix A0 so the size of the artefact stays visible.
3. **Keep the operating threshold at 3.0** and report what it actually does instead of re-tuning it
   per detector. With warm features it flags 3 of the 38 test rows per replicate and reaches pooled
   recall 0.650 at precision 1.000.
4. **Publish the ablation, including the schemes that were not shipped** (`amount3`, `no_gap`) in
   `reports/score_ablation.md`.

## Consequences

- Pooled PR-AUC of the shipped score: 0.296 cold/7-feature (v0.3.0) -> 0.436 cold/amount-only ->
  **0.820 warm/amount-only**. ROC-AUC: 0.791 -> 0.762 cold -> 0.914 warm.
- Top-30 ties on the clean window drop from 24 to 2.
- The demo page is a single cold batch by nature: it has no history to give the rolling features, so
  it shows few flags. The daily scorer (`scripts/daily_score.py --context-days 120`) does keep
  context, and that is the path the warm numbers describe.
- `amount3` scores slightly higher (0.834 warm) but re-introduces raw magnitude, the signal the
  manual review already found weak for this user; it stays an ablation row rather than the default.
- The autoencoder is untouched and still loses (0.200 warm). ADR 0002 stands: the served detector is
  chosen by evidence.
