# Meta-labeling v1 — BTC H1 trend entries

Registered and pushed before any meta-labeling analysis on real data.

## Research question and fixed primary rule

Does an out-of-sample meta-model improve the pre-registered BTC H1 rule's risk-adjusted return or drawdown, net of costs, without materially reducing CAGR?

The primary rule is fixed to H1 BTC trend from `preregistration/h1-time-series-trend.md`: daily BTC-USD; equal vote across 20, 60, 120, and 250-calendar-day sign-of-return signals; long when the mean vote is positive and otherwise cash; trailing 30-day realized-volatility scaling to the already selected 40% target; 1× exposure cap; decision at the daily close and execution at the next daily open. The rule is not retuned. The published prompt #6 report found that H2 reduced BTC performance and H3 lost money, so neither is eligible for a second meta-model.

An entry event begins when H1 changes from cash to long. A meta decision and multiplier remain fixed for that entire H1 episode; daily H1 volatility targeting continues underneath it. The episode ends when H1 returns to cash. The baseline takes every episode with multiplier 1.

## Eligible observations and timing

- Use only the frozen `research-snapshot-v1` public-data release through 2026-09-30.
- Every feature is computed at the H1 signal bar's close. Entry remains the next observed BTC daily open.
- Outer evaluation uses the existing expanding 730-day minimum-training / 12-month test schedule, but with a 20-day purge before each test fold and a 20-day embargo after it. Training labels must have reached their vertical barrier before the purge boundary.
- Model selection and calibration use only each outer training fold. Inner folds are expanding and time ordered, with the same 20-day purge and embargo. No test-fold value may affect preprocessing, imputation, percentile ranks, hyperparameters, calibration, or thresholds.
- The headline cost model remains 25 bps per side plus the existing volume-scaled slippage model. Exposure is capped at 1×. Cash earns 0%.

## Labels and weights

For each H1 entry event:

1. Compute trailing daily volatility as the standard deviation of the prior 30 close-to-close returns, using data through the signal close. This is unannualized for barrier construction.
2. Set symmetric profit and stop barriers at entry open × (1 ± 2 × trailing daily volatility). Barrier width is floored at 1% and capped at 25% to prevent degenerate or implausibly distant barriers.
3. Observe the following 20 BTC daily closes, beginning with the execution day. Label `1` if the profit barrier is reached before the stop barrier. Label `0` if the stop is reached first or neither barrier is reached before the twentieth close. If both would be crossed between observed closes, the close path determines the first observed crossing; no intraday ordering is inferred.
4. Store the realized net event return from the entry open to the first observed barrier close, or to the twentieth close at the vertical barrier, less the registered round-trip costs.

Overlapping labels receive average uniqueness weights: for every event, average `1 / concurrent_events` across its active label interval, then normalize weights to mean 1 within the outer training fold. The same weights are used for model fitting and classification scoring. Strategy returns themselves are never reweighted.

## Point-in-time feature groups

All continuous values are winsorized to the training fold's 1st/99th percentiles, median-imputed from the training fold, and standardized from the training fold. Missingness indicators are retained where stated.

### Trend state

- H1 vote strength from the four fixed lookbacks.
- Fraction of the four votes agreeing with the positive H1 decision.
- BTC close divided by its trailing 200-day moving average minus 1.
- Trend acceleration: trailing 20-day return minus the preceding 20-day return.

### Volatility

- Trailing 10-, 30-, and 90-day realized volatility from daily close returns.
- Volatility of volatility: 30-day standard deviation of the trailing 30-day volatility series.
- Percentile of current trailing 30-day volatility versus prior observations available inside the training history. For validation/test rows, the reference distribution ends at the row's decision time and contains no future observations.

### Positioning

- Backward-aligned Deribit BTC perpetual daily funding level.
- Trailing 90-observation funding z-score.
- Funding trend: trailing 7-day mean minus trailing 30-day mean.
- Explicit funding-missing indicator. Missing funding values are imputed only after this indicator is created.

### Market breadth

- Share of the point-in-time top-20 universe above each asset's trailing 50-day moving average.
- BTC dollar-volume share of the same point-in-time top-20 universe.

### Calendar

- Day-of-week one-hot indicators. No month, holiday, seasonality, or other calendar feature is tested.

### Optional on-chain group

Skipped. No free, keyless, U.S.-accessible source with a sufficiently documented historical publication lag is locked for this study. Adding an on-chain source later requires a new preregistration and prospective analysis.

## Models and locked variants

### Rule-only baseline

Take every H1 episode at its original daily volatility-targeted exposure.

### L2 logistic regression

- Standardized features and uniqueness sample weights.
- Inner-fold grid: `C ∈ {0.01, 0.1, 1, 10}`.
- Choose `C` by minimum weighted inner-fold log loss; ties choose the smaller `C`.
- Apply sigmoid/Platt calibration fitted only to concatenated inner out-of-fold predictions. If an inner fold has one class, it is omitted; if calibration is not estimable, use the uncalibrated probability and report that fact.

### Gradient boosting

- `HistGradientBoostingClassifier` with log-loss objective and uniqueness sample weights.
- Inner-fold grid: learning rate `{0.03, 0.05}`, max leaf nodes `{3, 7}`, max depth `{2, 3}`, minimum leaf samples `{20, 40}`, L2 regularization `{1, 10}`, maximum 150 iterations, early stopping disabled for deterministic time-ordered fitting.
- Select by minimum weighted inner-fold log loss; ties use fewer leaves, shallower depth, more minimum leaf samples, greater L2, then lower learning rate.
- Apply the same training-only sigmoid calibration rule.

### Locked bet sizing and threshold variants

- **Sized headline variant:** episode multiplier = `clip(2p − 1, 0, 1)`. Daily position = multiplier × the original H1 volatility-target exposure.
- **Threshold diagnostic:** take the full H1 episode only when `p > t`; choose `t` from `{0.50, 0.55, 0.60, 0.65}` on inner out-of-fold predictions by highest net Sharpe, requiring at least 10 accepted training events. Ties choose the higher threshold. If none qualifies, use 0.50.
- The logistic sized variant is the sole pre-registered forward-paper candidate. After the results commit is merged, freeze its fitted preprocessing, calibration, and coefficients in a versioned artifact and begin logging it at the first successful scheduled paper workflow. It never trades live.

## Evaluation

- Compare on identical outer out-of-sample dates: BTC buy-and-hold, rule alone, logistic sized, logistic threshold, boosting sized, and boosting threshold.
- Strategy metrics: CAGR, annualized volatility, Sharpe, Sortino, Calmar, maximum drawdown, underwater days, worst month, worst week, turnover, exposure, and hit rate.
- Classification metrics: weighted AUC, Brier score, and log loss. Compare each model with a training-fold base-rate probability on the identical events. State explicitly whether each model beats the base rate on ranking and probability quality.
- Use 1,000 seeded paired weekly-block bootstrap samples for meta-model minus rule differences in CAGR, Sharpe, and maximum drawdown. Classification differences use event-week block bootstrap samples. Fewer than three blocks is reported as not estimable.
- Recompute Deflated Sharpe using all seven prompt #6 strategy variants plus the four locked sized/threshold meta variants (11 total trials). Recompute CSCV PBO over the same combined variant set on their common dates.
- Feature-group ablation refits the full outer pipeline after dropping one group at a time. Report changes versus the corresponding full model for OOS AUC, Brier, strategy CAGR, Sharpe, and maximum drawdown, with event-week or return-week block CIs as appropriate.
- Run 200 seeded permutations. Shuffle labels and realized event returns together within calendar-quarter blocks inside each outer training fold, refit the complete model-selection/calibration pipeline, and preserve untouched outer test labels and market returns. Compare the real OOS Sharpe improvement over the rule with the null distribution. Seed: 20261003.
- Guard tests must reject a feature equal to a future return or shifted future price, verify all fitted preprocessing cutoffs precede each test fold, and verify next-open execution.

## Success criteria

Meta-labeling is supported only if at least one **sized** model satisfies every condition:

1. Its paired 95% weekly-block CI for Sharpe minus the rule has a lower bound above 0.
2. Its CAGR is no more than 2 percentage points below the rule and its maximum drawdown is no worse at the point estimate; or its paired maximum-drawdown improvement CI has a lower bound above 0 while its CAGR CI lower bound is above −2 percentage points.
3. Its OOS Brier score and log loss are both lower than the fold-specific base-rate baseline at the point estimate, and at least one paired event-week CI excludes 0 in the favorable direction.
4. Its real Sharpe improvement exceeds the 95th percentile of the 200-permutation null.
5. Its Deflated Sharpe probability is higher than the rule's under the combined 11-trial correction, and combined-set PBO does not rise by more than 5 percentage points from the published prompt #6 value.

Threshold variants, ablations, feature importance, and regime slices are diagnostic and cannot replace the sized primary comparison. Failure is reported plainly; no parameter is changed after results are viewed.
