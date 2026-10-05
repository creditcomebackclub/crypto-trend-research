# H1 — Time-series trend with volatility targeting

Registered before any real-data strategy run.

- **Question:** Can a fixed trend ensemble on liquid crypto assets approach BTC buy-and-hold CAGR with materially smaller drawdowns out of sample?
- **Eligible data:** Coinbase/Kraken spot bars available at the fixed analysis end date. Signals use only information available at each bar close; execution is at the next bar open.
- **Universe:** BTC and ETH first, then the point-in-time monthly top 10 by trailing 30-day source-exchange dollar volume, with at least 90 prior calendar days of history. Stable, wrapped, and pegged assets are excluded.
- **Signal:** Equal vote across fixed 20, 60, 120, and 250-day sign-of-return signals. The secondary 4-hour study uses the pre-specified equivalents in `config.yaml`.
- **Position:** Long when the mean vote is positive, otherwise cash. Exposure is scaled by trailing realized volatility to the target selected on the training fold only and capped at 1×.
- **Variants:** BTC, ETH, and equal-weight eligible top-10 assets; volatility targets 20%, 30%, and 40%. A long/short perpetual variant is secondary and includes funding.
- **Primary metric:** Out-of-sample CAGR difference from BTC buy-and-hold paired with max-drawdown reduction.
- **Success criterion:** CAGR no more than 2 percentage points below BTC, max drawdown at least 25% smaller, and the 95% block-bootstrap CI for the Sharpe difference has a lower bound above zero.
- **Forward start:** the first successful scheduled paper-trade workflow after the initial results release.

