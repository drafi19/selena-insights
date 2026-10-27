# ADR 0002 - which detector gets served

## Context
Three detectors were trained and scored with the same protocol (20 injection replicates,
pooled PR-AUC over ~760 labels) on 251 real spending rows:

| Version | Autoencoder | z-score | IsolationForest |
|---|---|---|---|
| v0.1.0 (6 features) | 0.170 | 0.296 | 0.320 |
| v0.2.0 (7 features) | 0.154 | 0.296 | 0.318 |

Adding the category-relative feature did not help the autoencoder. On the synthetic corpus
(10,000 rows, ~7,000 for training) the autoencoder reaches PR-AUC 0.197 and beats the baselines,
which says the gap is sample size, not architecture.

## Decision
1. **Serve the detector that won on evidence, not the one that sounds impressive.** The serving
   layer exposes a `detector` field; the default is the best pooled PR-AUC *and* the best recall
   at the calibrated threshold, which at n=251 is z-score.
2. Keep the autoencoder in the repo, trained and evaluated by the same protocol. It is the
   experiment, not the product, until it wins.
3. Revisit at n >= 400 spending rows (roughly +2 months of logging): retrain both, re-run
   `scripts.ablation`, and let the table decide again.

## Consequences
- README and the API report the detector actually used, so nobody can accuse the project of
  shipping a "deep learning" label over a z-score.
- The negative ablation is published. That is the point: it shows the evaluation decides, not taste.
