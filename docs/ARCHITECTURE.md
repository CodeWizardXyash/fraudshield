# Architecture

## Training flow
1. Simulate (or load) time-ordered transactions and validate the batch.
2. Replay them through `FeatureStore`: compute features, then update state.
3. Split by time: train 60%, selection 10%, calibration 10%, test 20%.
4. Fit four candidates on train and pick the best PR-AUC on selection.
5. Fit sigmoid calibration, then choose the cost threshold, on the calibration split.
6. Report once on test, with a day-level bootstrap interval and a rules baseline.
7. Save model, threshold, drift reference and feature state; apply the promotion gate.

## Serving flow
Validate (Pydantic) -> reject future timestamps -> lock -> check duplicate ID ->
compute features -> score -> update state -> respond.

## Trade-offs
| Choice | Why | Cost |
|---|---|---|
| In-process feature state | Simple, exact parity with training | Single worker, lost on restart |
| One lock around scoring | Correct ordering of state updates | Throughput does not scale |
| File registry | No extra infrastructure | No access control or audit log |
| Starlette + Pydantic | Small and explicit | No automatic OpenAPI page |
| One thread for inference | Better concurrent latency on small batches | Large batches would be slower |
| Synthetic data | Shareable and reproducible | No evidence about real traffic |

## Promotion gate
The candidate needs a minimum selection PR-AUC. If it was trained on the same data as the
champion, it must also not fall more than `max_regression` below it. With new data that
comparison is skipped; the better fix is to score both models on one fixed recent window.

## Failure handling
- No champion, checksum mismatch or library mismatch: the service refuses to start.
- Invalid input: 422. Out-of-order event or conflicting retry: 409.
- Unexpected scoring error: 500 with details only in server logs. A real payment flow also
  needs a fallback rule in the caller for when the service is unavailable.
