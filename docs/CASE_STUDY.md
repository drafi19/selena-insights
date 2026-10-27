# From a capstone that looked alive to a detector I can defend

**Ringkasan (ID).** Proyek capstone Bangkit saya punya dashboard yang menampilkan "anomali" untuk
separuh transaksi - karena normalisasinya tidak pernah benar-benar membagi, dan threshold-nya
persentil 50. Saya membangun ulang komponen ML-nya di atas data rekening saya sendiri, mengukur
tiga detektor dengan protokol yang sama, dan hasilnya: autoencoder kalah dari z-score. Tulisan ini
menjelaskan kenapa saya tetap menganggap proyek ini berhasil.

## Context

Selena was a seven-person Bangkit capstone: an Android app for tracking personal finances, an
Express backend on Cloud Run, and a "machine learning" component that flagged unusual spending.
I was one of three people on the ML side.

The ML deliverable was a Keras autoencoder, 6 -> 32 -> 16 -> 8 -> 16 -> 32 -> 6, exported to
TensorFlow.js so the Node backend could run it in-process. It had 1,774 parameters, four input
features, and no evaluation of any kind - no test set, no metric, no baseline.

## What was actually broken

When I reopened the repo two years later, three things stood out. All of them are still visible in
the original code, which is why I can quote them.

**1. The data was never normalised.** `preprocessData.js` did this:

```js
.div((maxValues.sub(minValues) === 0 || 1));
```

`maxValues.sub(minValues)` returns a `tf.Tensor`. Comparing a tensor to `0` with `===` is always
`false`, so `(false || 1)` is `1`, and the division always runs with 1. The data was only shifted,
never scaled - while the decoder's output layer was `sigmoid`, which can only emit values in
[0, 1]. Input and output lived on different scales, so the reconstruction error mostly measured
that mismatch.

**2. The threshold was a fixed share of every batch.** The handler called
`detectAnomalies(..., percentileThreshold = 50)`. A percentile-50 threshold flags *half the rows
in the batch by construction*. The "anomalies" were an artifact of batch size, not a signal.

**3. The model was retrained on every request.** Each `/dashboard` call trained 100 epochs, saved
the model, uploaded two files to Cloud Storage, then deleted the temp folder - and the model
object was process-global, shared by every user.

A demo that never throws an error, and never says anything true, is the worst kind of working.

## The dataset was two datasets

The capstone CSV had 1,412 rows. Splitting by date produced 806 rows before 2023 and 606 after,
with a median amount of **594,620 versus 68,484** - an 8.7x gap. The first block came from a public
Kaggle dataset in USD converted at a hardcoded rate; the second came from three teammates' bank
statement exports.

That is not a dataset, it is two populations wearing one coat. A time-based split would have
landed exactly on the seam, so a model could look good by learning "which era is this" instead of
"is this transaction unusual".

I could not use that data anyway: it contains three other people's bank statements, and the team
is no longer reachable to ask. So the capstone data stays out of this repository entirely, and the
finding above goes in the write-up instead.

## Rebuilding on my own data

I rebuilt the ML component against my own BudgetBakers Wallet account, pulled read-only over their
REST API. Two things improved immediately: consent stopped being a question, and the records had
fields the capstone never had - category, counterparty, account, labels, and an explicit transfer
flag.

Then the filters, which turned out to be the most consequential code in the project:

| Filter | Rows dropped |
|---|---|
| Transfers between my own accounts | 82 |
| Lending ledger (`Lending, renting`, `Loan, interests`, label `Titip`) | 105 |
| Income records | 126 |
| Zero-amount placeholders | 3 |

From 567 records, **251 are actual spending**. Without the first two filters, roughly 40% of what
the model sees would not be spending at all - the same category of error as the original
"percentile 50" bug, just less visible.

## Evaluation without labels

Nobody labels their own transactions as anomalous, so I injected anomalies: spikes (8-20x the
amount), near-duplicates, round-number bursts, and micro-amount drips, with a fixed seed, mutated
in place so every row keeps its label.

The first version of this evaluation was wrong in a way worth admitting. With a 38-row test set,
injecting at 2% gives **four positives**. One PR-AUC computed from four positives is noise. The
protocol now injects at 10% and runs **20 replicates**, then reports the pooled number over ~760
labels alongside the spread - because quoting one run from 38 rows is exactly the kind of number
that collapses in an interview.

The other correction: every detector gets its own threshold, calibrated on a clean holdout at a 1%
false-positive rate. In an early version I reused the autoencoder's reconstruction threshold for
IsolationForest, whose scores live on a completely different scale. That baseline could never have
won, and a reviewer would have spotted it.

## What the numbers said

Pooled over 20 replicates, chance level 0.105:

| Detector | PR-AUC | ROC-AUC | Recall @ FPR 1% |
|---|---|---|---|
| Autoencoder, 6 features | 0.170 | 0.598 | 0.025 |
| Autoencoder, 7 features (+ category-relative) | 0.154 | 0.570 | 0.025 |
| z-score | 0.296 | 0.791 | 0.138 |
| IsolationForest | 0.318 | 0.762 | 0.037 |

The autoencoder - the thing the capstone called the ML component - lost to a z-score.

I added a category-relative feature (how unusual is this amount *for this category*) expecting it
to close the gap. It did not: 0.170 -> 0.154, a negative result I published rather than quietly
deleted (`reports/ablation.md`). The feature is active on 56.6% of rows, so it was not a dead
column; there simply are not enough transactions per category yet.

## Why the autoencoder lost

Two experiments answered that.

**Experiment 1: more data, same architecture.** I trained a separate model on a 10,000-row
synthetic corpus (7,000 for training) and ran the same protocol. The autoencoder reached PR-AUC
0.215 and beat both baselines (z-score 0.173). So the architecture is not the problem.

**Experiment 2: same data, different seeds.** Three seeds, single runs: PR-AUC 0.196 +- 0.092. The
spread is enormous relative to the effect. At 175 training rows, the autoencoder's score depends
more on the initialisation than on anything else.

The conclusion is boring and correct: 175 training rows cannot support 1,839 parameters, and the
anomalies I inject are single-feature magnitude outliers - the exact shape a z-score is built to
catch. A model of *multivariate* normality needs both more data and a harder anomaly definition
before it can demonstrate an advantage.

So the service ships the detector that won. `POST /score` returns a `detector` field, and it says
`zscore`. Shipping a deep model that loses to a baseline, while calling the project "AI", would
have been the actual failure.

## What I would do with more data

1. Retrain at >= 400 spending rows (~2 more months of logging) and re-run the ablation. The
   synthetic control says the autoencoder needs roughly that order of magnitude to compete.
2. Make the anomaly definition harder: "ordinary amount, impossible time", "amount consistent
   with the category but inconsistent with the day", duplicates across accounts. Right now the
   benchmark mostly measures magnitude.
3. Add the 30 hand-reviewed flags as a human-labelled precision number. It is 30 rows, but it is
   real labels instead of injected ones.

## Limitations

- One person, 92 days, 251 rows. Nothing here generalises to other users, and I do not claim it.
- Labels are synthetic; the metrics measure sensitivity to patterns I defined.
- Manual logging: unrecorded transactions do not exist in the series.
- The 38-row test set makes every per-replicate number noisy, which is why the protocol pools.

## What is not my work

The Android app and the Express backend were built by the Cloud Computing and Mobile Development
members of the team; the GCP deployment was theirs. I did not write them and none of their code is
in this repository. This repository is my own re-implementation of the ML component, written after
the programme, with a different data source, a different stack, and an evaluation protocol the
original never had. The original team repo is linked in the README for context.
