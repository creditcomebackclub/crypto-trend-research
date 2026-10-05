# H3 — Cross-sectional momentum

Registered before any real-data strategy run.

- **Question:** Does fixed cross-sectional momentum beat an equal-weight point-in-time universe after costs?
- **Universe:** Monthly point-in-time top 20 by trailing 30-day source-exchange dollar volume, excluding stable, wrapped, and pegged assets and requiring 90 prior days of history.
- **Signal:** At each monthly rebalance, average the ranks of trailing 30- and 90-day returns after skipping the most recent 7 days. Hold the top quintile equally weighted until the next rebalance.
- **Variants:** Unfiltered and a BTC-trend-filtered version that holds cash when the H1 BTC vote is non-positive.
- **Execution:** Signal at the rebalance bar close, trade at the next bar open, with turnover-based costs and volume-scaled slippage.
- **Primary metric:** Out-of-sample paired Sharpe difference versus equal-weight top 20.
- **Success criterion:** Positive paired Sharpe-difference CI, positive net CAGR difference, and lower max drawdown.
- **Forward start:** the first successful scheduled paper-trade workflow after the initial results release.

