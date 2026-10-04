# H2 — Funding-rate crowding filter

Registered before any real-data strategy run.

- **Question:** Does reducing trend exposure during extremely positive perpetual funding improve downside risk without materially reducing return?
- **Eligible data:** Public Deribit funding observations aligned backward to each decision bar. No future funding observation may enter a signal.
- **Universe:** Assets with both an H1 signal and sufficiently deep public funding history, expected initially to be BTC and ETH. SOL is included only if the source exposes qualifying history.
- **Signal:** H1 exposure multiplied by 0.5 when the trailing 90-observation funding z-score exceeds +2.0. All parameters are fixed in advance.
- **Primary metric:** Out-of-sample max-drawdown difference versus the matching H1 strategy, with CAGR and Sharpe as secondary metrics.
- **Success criterion:** At least 10% relative drawdown reduction, no more than 2 percentage points of CAGR sacrificed, and positive paired Sharpe difference with a 95% weekly-block CI.
- **Diagnostic:** Forward return and drawdown distributions after positive and negative funding extremes are reported whether or not the strategy succeeds.
- **Forward start:** the first successful scheduled paper-trade workflow after the initial results release.

