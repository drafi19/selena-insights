# Bug 0001 - the old backend never actually normalised the data

## Location
`legacy/cc-helpers/preprocessData.js` (original capstone backend):

```js
const normalizedData = dataTensor
  .sub(minValues)
  .div((maxValues.sub(minValues) === 0 || 1));
```

## Why it is wrong
`maxValues.sub(minValues)` returns a `tf.Tensor`. Comparing a tensor with `=== 0` is always
`false` in JavaScript, so the expression `(false || 1)` evaluates to `1`. The division therefore
always runs with `1`, and the data is only shifted (`x - min`) - never scaled into [0, 1].

That matters because the decoder's output layer is `sigmoid`, which can only produce values in
[0, 1]. Inputs and outputs live on different scales, so the reconstruction error is dominated by
the scale difference rather than by the shape of the transaction pattern.

## Knock-on effect
`getDashboardData.js` called `detectAnomalies(..., percentileThreshold = 50)`. A percentile-50
threshold flags **half of the rows in the batch** by construction: the "anomaly list" was a fixed
share of whatever came in, not a signal. Combined with the normalisation bug, the dashboard was
guaranteed to show anomalies that meant nothing.

## Fix in this repository
- `selena_ml/scaling.py` computes the span with `np.where(span == 0, 1.0, span)` and does not
  divide by a boolean.
- The threshold is calibrated **once** on a clean holdout at a 1% false-positive rate and
  persisted in the artefact (`threshold.json`), never recomputed per batch.
- Evidence: `tests/test_scaling.py::test_zero_variance_column_is_safe` and
  `tests/test_scaling.py::test_fit_transform_range`.

## Lesson
A silent type bug (`Tensor === 0`) plus a batch-relative threshold makes a demo look alive while
carrying zero information. Both are now covered by tests, and neither can be reintroduced without
turning CI red.
