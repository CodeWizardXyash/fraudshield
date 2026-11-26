# Interview guide

Read the code until you can explain each answer without this page. Then change something
yourself so the project is truly yours.

## 60-second pitch
"I worked on a fraud detection system that scores a transaction at authorisation time. The
same feature code runs in training and serving, the splits are chronological, probabilities
are calibrated, and the threshold minimises a cost function. It has an API, a model registry
with a promotion gate, and drift monitoring. The data is synthetic, so I treat the metrics as
a check on the pipeline rather than real-world performance."

## Questions to expect
1. **Why PR-AUC?** Fraud is rare. ROC-AUC (0.998 here) looks good even when precision is poor.
2. **How do you prevent leakage?** Compute before update, time splits, and the tests
   `test_no_future_leakage` and `test_label_does_not_affect_features`.
3. **What is training/serving skew?** Different logic offline and online. One `FeatureStore`
   class and a parity test address it.
4. **Why calibrate?** The threshold and downstream cost decisions treat scores as probabilities.
5. **How was the threshold chosen?** Lowest cost over all score cut-offs on calibration data.
6. **Why not SMOTE?** Resampling distorts probabilities and, over time series, can mix periods.
   It would need to be compared fairly, inside the training split only.
7. **Why is the score so high?** Synthetic data. Say this before the interviewer does.
8. **Weakest area?** Account takeover recall, 74.7%.
9. **How would you scale it?** Shared state in Redis, atomic per-customer updates, partition by
   customer ID, then several workers.
10. **What does PSI miss?** It detects input shift, not loss of accuracy. That needs labels.
11. **What if labels arrive late?** Leave a gap between training and evaluation, and use only
    labels that were known at training time.
12. **Why no random split?** It lets the model learn from the future and from the same fraud
    episode on both sides.

## Make it yours (do at least two)
- Add a feature (for example, distinct countries in 24h) with a test.
- Add a LightGBM or XGBoost candidate and compare.
- Move feature state to Redis.
- Evaluate on a public fraud dataset and write up what broke.

## Resume bullets (use after you have run it and can explain it)
- Built a real-time fraud detection system (Python, scikit-learn, Starlette) with shared
  training/serving features, chronological evaluation and cost-based thresholding.
- Reached 0.95 PR-AUC vs. 0.47 for a rules baseline on 254K **synthetic** transactions.
- Added a model registry with promotion gate, PSI drift monitoring, 36 automated tests and
  a p95 latency of 10 ms for single requests on 2 vCPUs.
