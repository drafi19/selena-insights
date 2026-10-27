# ADR 0001 - Python 3.11 + PyTorch + ONNX for training and serving

## Context
The original project: a Keras autoencoder (6-32-16-8-16-32-6) exported to tfjs and served by
`@tensorflow/tfjs-node` inside an Express app. In `getDashboardData.js` every `/dashboard` request
retrained the model for 100 epochs, saved it, uploaded the artefact to a cloud bucket, and then
flagged anomalies with `percentile = 50` - which labels half of all transactions as anomalies.
The model object was process-global and shared by every user.

## Decision
Train in Python 3.11 with PyTorch; serve with ONNX Runtime.

## Reasons
1. TensorFlow 2.18 ships no wheel for Python 3.14, the default interpreter on this machine.
2. Retraining per request cannot be defended: seconds of latency per call, per-user artefacts to
   store, and a shared mutable model (race condition between users).
3. ONNX Runtime removes torch from the serving image (~800 MB -> ~200 MB) and cuts cold start.
4. Evaluation lives in Python (scikit-learn); the original had no metric at all.

## Consequences
- The original tfjs weights are not migrated. The model is retrained from scratch, keeping the
  architecture identical (1,774 parameters) so comparisons stay fair.
- The original notebook is kept in `legacy/` as evidence, not executed.
- No GCP services are used anywhere in this repo.
