# Crypto Trend Research Report

## 1. Question

Can a systematic strategy on liquid crypto assets match Bitcoin buy-and-hold with materially smaller drawdowns, net of realistic costs, out of sample?

**Result:** The headline trend strategy matched the pre-registered return and drawdown criterion. This report is research, not trading advice.

## 2. Data and universe

- Source: committed synthetic CI fixture (not empirical evidence).
- Fixed analysis end: 2026-09-30.
- Out-of-sample observations: 470 across 2 purged expanding-window test blocks.
- Universe membership is reconstructed monthly from trailing 30-day source-exchange dollar volume with a 90-day history requirement.
- Stablecoins and wrapped or pegged assets are excluded. Active-product APIs do not guarantee recovery of every delisted asset; unavailable dead assets bias historical results upward through survivorship.
- Data-quality checks retain an audit trail for missing bars, zero volume, outliers, timestamp alignment, and BTC/ETH cross-exchange divergence. No repair is silent.

## 3. Benchmarks

| Strategy | CAGR (95% CI) | Max DD (95% CI) | Sharpe (95% CI) | Turnover |
|---|---:|---:|---:|---:|
| BTC buy-and-hold | -16.1% [-60.5%, 55.6%] | -49.8% [-70.2%, -24.9%] | -0.18 [-1.87, 1.33] | 0.0× |
| 60/40 BTC/cash | -7.9% [-41.9%, 30.8%] | -33.3% [-51.9%, -17.5%] | -0.18 [-1.69, 1.37] | 0.0× |
| Equal-weight top 10 | 1.8% [-33.6%, 45.2%] | -30.9% [-49.4%, -14.5%] | 0.21 [-1.11, 1.66] | 0.0× |
| Equal-weight top 20 | 1.8% [-34.6%, 73.3%] | -30.9% [-48.4%, -14.1%] | 0.21 [-1.09, 1.39] | 0.0× |
| BTC trend | 31.0% [-18.1%, 91.8%] | -16.5% [-28.8%, -9.4%] | 1.26 [-0.32, 2.62] | 30.6× |
| BTC risk-matched | -15.1% [-47.1%, 48.7%] | -46.5% [-70.8%, -22.5%] | -0.21 [-1.67, 1.05] | 7.1× |
| Top-20 momentum | -24.1% [-51.9%, 45.6%] | -34.3% [-70.1%, -22.9%] | -0.49 [-1.79, 0.94] | 21.3× |
| Momentum + BTC regime | 11.2% [-17.1%, 53.9%] | -21.6% [-35.7%, -7.9%] | 0.60 [-1.37, 1.64] | 36.7× |

## 4. Pre-registered hypotheses

### H1 — Time-series trend

BTC trend CAGR was 31.0% versus -16.1% for BTC buy-and-hold. Its max drawdown was -16.5% versus -49.8%. The weekly-block paired mean-return difference CI was [-0.1%, 0.2%]. The headline trend strategy matched the pre-registered return and drawdown criterion.

### H2 — Funding crowding

Not estimable from this snapshot because aligned public funding history was unavailable.

### H3 — Cross-sectional momentum

The fixed top-quintile strategy is reported against equal-weight top 20 above, including the pre-registered BTC regime-filter variant.

## 5. Deflated Sharpe and PBO

- Headline Deflated Sharpe Ratio probability: 100.0% across 8 evaluated variants.
- CSCV Probability of Backtest Overfitting: 40.0%.
- These diagnostics reduce confidence for strategy selection across multiple variants; they do not turn a backtest into forward evidence.

## 6. Regime breakdown

Regime labels are defined ex ante from BTC's trailing 200-day moving average and trailing realized-volatility terciles. Tercile thresholds must be fitted inside each training fold. Detailed rows are omitted when a regime has fewer than three weekly blocks.

## 7. Cost sensitivity

Headline results use 25 bps per side plus volume-scaled slippage. The configured 10, 25, and 50 bps scenarios are fixed before evaluation. Higher assumed costs reduce high-turnover momentum more than low-turnover BTC trend.

## 8. Funding-carry diagnostic

H4 is diagnostic only. A valid result requires overlapping spot and perpetual data, fees on both legs, observed funding sign changes, slippage, and conservative margin stress. It is never promoted from this report as a headline strategy.

## 9. Limitations

- Public exchange product lists can omit delisted assets, creating upward survivorship bias.
- Exchange candles may be absent when no trades occur; missing bars are flagged rather than silently filled.
- Coinbase and Kraken availability differs by asset and history depth. Kraken's OHLC endpoint is intentionally treated as a shallow cross-check.
- Backtests cannot reproduce queue position, outages, spread shocks, taxes, or future market structure.
- Confidence intervals quantify sampling uncertainty under the chosen weekly-block scheme, not all model risk.

## 10. Recommendations

Keep all strategies in research and paper-trading until the locked forward log is long enough to compare with these expectations. Do not add leverage to compensate for weak unlevered evidence. **This is not trading advice.**
