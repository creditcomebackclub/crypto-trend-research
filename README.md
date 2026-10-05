# Crypto Trend Research

> **Research question:** Can a systematic strategy on liquid crypto assets match Bitcoin's buy-and-hold return with materially smaller drawdowns, net of realistic costs, out of sample?

## Headline results

The table below is the locked out-of-sample result from the frozen public-data snapshot through 2026-09-30. Every row includes next-open execution, 25 bps per side, volume-scaled slippage, and weekly-block confidence intervals.

| Strategy | OOS CAGR (95% CI) | Max drawdown (95% CI) | Sharpe (95% CI) |
|---|---:|---:|---:|
| BTC buy-and-hold | 44.0% [-9.8%, 127.7%] | -76.7% [-90.9%, -44.2%] | 0.91 [0.17, 1.66] |
| BTC trend, volatility-targeted | 28.0% [2.6%, 58.8%] | -33.6% [-57.3%, -22.1%] | 1.00 [0.23, 1.73] |
| Top-20 momentum | -27.9% [-63.1%, 37.9%] | -98.7% [-99.9%, -81.2%] | 0.15 [-0.47, 0.81] |

**Interpretation:** the BTC trend ensemble cut drawdown substantially but did not match BTC's return, and its paired Sharpe-difference CI crossed zero, so it failed the full pre-registered success criterion.

### Key findings

- BTC trend returned 28.0% annualized versus 44.0% for buy-and-hold, while reducing max drawdown from -76.7% to -33.6%.
- The funding-crowding filter hurt rather than helped: BTC trend CAGR fell from 28.0% to 21.6% and drawdown worsened slightly.
- Cross-sectional top-20 momentum lost 27.9% annualized after costs and suffered a -98.7% drawdown.
- After adding the four registered meta variants, the corrected Deflated Sharpe probability was 84.3% across eleven variants and CSCV PBO rose from 7.1% to 32.9%. Those diagnostics do not rescue a failed primary criterion.
- The delta-neutral funding-carry diagnostic lost 5.0% annualized after realized funding and costs. The 4-hour study was not estimable because Kraken exposes only 720 recent bars, short of the locked training window.
- Meta-labeling did not improve the BTC trend rule out of sample: logistic sizing reduced Sharpe from 1.005 to 0.669, ranked events worse than the fold-specific base-rate forecast (AUC 0.475 versus 0.545), and no feature group produced a robust improvement. Its smaller -18.5% drawdown mainly came from cutting average exposure to 12.9%.

## What this repository tests

- **H1:** a fixed multi-lookback time-series trend ensemble with past-only volatility targeting and at most 1× exposure.
- **H2:** whether reducing H1 exposure during extremely positive perpetual funding improves downside risk.
- **H3:** point-in-time cross-sectional momentum inside monthly top-20 universes reconstructed from trailing source-exchange volume.
- **H4:** a diagnostic spot/perpetual funding-carry simulation that includes fees, sign flips, and margin stress.

Every headline result executes a close-time signal at the next bar open, includes turnover costs and volume-scaled slippage, and is measured only in purged expanding-window test blocks. The report includes weekly-block confidence intervals, paired benchmark comparisons, Deflated Sharpe, CSCV Probability of Backtest Overfitting, regime breakdowns, and explicit failures.

## Reproduce

Requires Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python scripts/fetch_snapshot.py
python scripts/run_all.py
```

The committed synthetic fixture exercises the complete pipeline in CI without downloading market data:

```bash
python scripts/run_all.py --synthetic
```

The resulting report is [reports/REPORT.md](reports/REPORT.md). Synthetic output is labeled and is not evidence for a strategy.

## Build a new frozen snapshot

```bash
python scripts/fetch_spot.py --source coinbase --frequency 1d
python scripts/fetch_spot.py --source coinbase --frequency 4h
python scripts/fetch_spot.py --source kraken --frequency 1d
python scripts/fetch_spot.py --source kraken --frequency 4h
python scripts/fetch_funding.py
python scripts/build_snapshot.py
```

Raw and processed data stay under ignored `data/` paths. The 57 MB release archive's SHA-256 (`9550a10693103434fad92e87ce3814730dd66b5f156f94073e5ea877d722c7cc`) and all member hashes live in `data/SNAPSHOT.json`. Fetchers merge cached bars, skip completed ranges, and can resume after interruption. Endpoint limits and history constraints are documented in [docs/data-sources.md](docs/data-sources.md).

## Research safeguards

- All four hypotheses were committed before any real-data strategy result.
- The analysis end date and random seed are fixed in `config.yaml`.
- Monthly membership uses only volume and listing history available at that date.
- A deliberately future-leaked feature must fail the guard test.
- Current product catalogs may omit dead assets. That survivorship bias is reported and is likely optimistic.
- No exchange keys, wallets, order placement, or headline leverage are present.

## Forward paper test

The keyless scheduled workflow computes target weights from fresh public data and appends them to `forward/paper_log.csv`. It records the previous target's one-day paper P&L, commits only that log, and never submits an order. The forward start date becomes the first successful scheduled run and will be written into the pre-registrations when known.

## Project layout

- `ctr/`: reusable data, signals, portfolio, validation, statistics, and reporting logic
- `scripts/`: resumable fetchers, snapshot tools, report runner, and paper-target runner
- `preregistration/`: hypotheses fixed before empirical results
- `notebooks/`: narrative audit views; no core logic
- `tests/fixtures/`: deterministic synthetic CI data
- `reports/`: generated report and figures

This repository is research, not trading advice.
