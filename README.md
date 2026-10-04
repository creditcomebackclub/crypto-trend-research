# Crypto Trend Research

> **Research question:** Can a systematic strategy on liquid crypto assets match Bitcoin's buy-and-hold return with materially smaller drawdowns, net of realistic costs, out of sample?

## Headline results

The empirical snapshot has not been published yet. The table remains deliberately blank until the pre-registered pipeline runs on the frozen public-data release; synthetic CI output is never presented as market evidence.

| Strategy | OOS CAGR (95% CI) | Max drawdown (95% CI) | Sharpe (95% CI) |
|---|---:|---:|---:|
| BTC buy-and-hold | Pending snapshot | Pending snapshot | Pending snapshot |
| BTC trend, volatility-targeted | Pending snapshot | Pending snapshot | Pending snapshot |
| Top-20 momentum | Pending snapshot | Pending snapshot | Pending snapshot |

**Current interpretation:** no strategy claim is supported until the frozen snapshot is released and the pre-registered analysis is run. “No” is an acceptable final answer.

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

Raw and processed data stay under ignored `data/` paths. The release archive's SHA-256 and its member hashes live in `data/SNAPSHOT.json`. Fetchers merge cached bars, skip completed ranges, and can resume after interruption. Endpoint limits and history constraints are documented in [docs/data-sources.md](docs/data-sources.md).

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

