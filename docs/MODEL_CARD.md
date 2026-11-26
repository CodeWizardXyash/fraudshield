# Model card: FraudShield v1.0.0

**Model:** histogram gradient boosting (15 leaves, 200 iterations) with sigmoid calibration.

**Intended use:** demonstration of a fraud-scoring system that routes risky transactions to
review. **Not for real decisions** about people or payments.

**Data:** 254,355 synthetic transactions, 2,000 customers, 180 days, 1.17% fraud, seed 42.
Fraud patterns: card testing, stolen card, account takeover. Legitimate noise: travel,
new phones, shopping sprees, unusually large purchases.

**Features:** 17 behavioural and transaction features. Country is used only as "new for this
customer". No name, age, gender or other personal attributes exist in the data.

**Test results:** see the README and `reports/metrics.json`.

**Threshold:** 0.042, minimising `5 * alerts + value of missed fraud` on the calibration split.

**Caveats**
- Scores are high partly because the simulator's patterns are simpler than real fraud.
- Merchant category dominates feature importance, because simulated fraud concentrates in a
  few categories. Real data would need checking for the same shortcut.
- The cost model ignores customer friction from false alerts and chargeback fees.
- Calibration and threshold share one split.
- The selection score both picks the model and gates it, so it is slightly optimistic.
- A real deployment needs fairness analysis across customer groups and human review.
