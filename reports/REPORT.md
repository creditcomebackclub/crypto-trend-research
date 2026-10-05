# Crypto Trend Research Report



## 1. Question

Can a systematic strategy on liquid crypto assets match Bitcoin buy-and-hold with materially smaller drawdowns, net of realistic costs, out of sample?

**Result:** The headline trend strategy did not satisfy all pre-registered return, drawdown, and paired-Sharpe criteria. This report is research, not trading advice.

## 2. Data and universe

- Source: frozen public-data snapshot.
- Fixed analysis end: 2026-09-30.
- Out-of-sample observations: 2,465 across 7 purged expanding-window test blocks.
- Universe membership is reconstructed monthly from trailing 30-day source-exchange dollar volume with a 90-day history requirement.
- Stablecoins and wrapped or pegged assets are excluded. Active-product APIs do not guarantee recovery of every delisted asset; unavailable dead assets bias historical results upward through survivorship.
- The observed top-20 test-period membership recorded 270 entries and 250 exits; the detailed membership audit is generated from the same point-in-time matrix.
- Data-quality checks retain an audit trail for missing bars, zero volume, outliers, timestamp alignment, and BTC/ETH cross-exchange divergence. No repair is silent.

**Audit summary:** Across 472 selected base-asset histories, the audit found 1,424 missing daily intervals, 0 zero-volume bars, and 764 robust return-outlier flags. The catalog exposed 326 offline/delisted products and cached 195; it still cannot prove completeness for removed listings. Cross-exchange close divergence — BTC-USD: median 0.01%, p99 0.08% (718 overlaps); ETH-USD: median 0.01%, p99 0.09% (718 overlaps).

## 3. Benchmarks

| Strategy | CAGR (95% CI) | Max DD (95% CI) | Sharpe (95% CI) | Turnover |
|---|---:|---:|---:|---:|
| BTC buy-and-hold | 44.0% [-9.8%, 127.7%] | -76.7% [-90.9%, -44.2%] | 0.91 [0.17, 1.66] | 0.0× |
| 60/40 BTC/cash | 30.2% [-0.2%, 69.8%] | -55.8% [-68.7%, -28.7%] | 0.91 [0.18, 1.65] | 0.0× |
| Equal-weight top 10 | 7.6% [-42.6%, 108.5%] | -90.6% [-99.2%, -66.4%] | 0.51 [-0.21, 1.33] | 25.2× |
| Equal-weight top 20 | 7.7% [-42.6%, 102.3%] | -92.2% [-99.3%, -65.5%] | 0.51 [-0.20, 1.28] | 25.6× |
| BTC trend | 28.0% [2.6%, 58.8%] | -33.6% [-57.3%, -22.1%] | 1.00 [0.23, 1.73] | 120.0× |
| ETH trend | 29.9% [2.4%, 61.2%] | -35.3% [-61.7%, -22.9%] | 1.02 [0.23, 1.71] | 96.5× |
| Top-10 trend | 10.6% [-34.7%, 86.3%] | -89.7% [-98.2%, -55.2%] | 0.48 [-0.33, 1.30] | 501.4× |
| BTC risk-matched | 30.2% [-5.8%, 79.9%] | -64.8% [-80.2%, -34.6%] | 0.84 [0.08, 1.63] | 33.9× |
| Top-20 momentum | -27.9% [-63.1%, 37.9%] | -98.7% [-99.9%, -81.2%] | 0.15 [-0.47, 0.81] | 125.1× |
| Momentum + BTC regime | 7.5% [-27.5%, 71.6%] | -79.1% [-96.2%, -53.8%] | 0.43 [-0.20, 1.16] | 192.3× |
| BTC trend + funding filter | 21.6% [1.1%, 49.5%] | -34.6% [-57.9%, -21.8%] | 0.87 [0.17, 1.61] | 174.2× |
| ETH trend + funding filter | 25.2% [2.8%, 55.4%] | -36.3% [-58.4%, -22.6%] | 0.92 [0.24, 1.68] | 122.7× |
| BTC trend + logistic sized | 7.9% [-1.5%, 18.8%] | -18.5% [-32.2%, -9.6%] | 0.67 [-0.05, 1.39] | 32.8× |
| BTC trend + logistic threshold | 22.4% [-0.1%, 46.9%] | -31.2% [-53.5%, -18.9%] | 0.93 [0.12, 1.62] | 92.6× |
| BTC trend + boosting sized | 10.0% [0.2%, 21.6%] | -19.6% [-31.6%, -10.3%] | 0.76 [0.08, 1.45] | 34.0× |
| BTC trend + boosting threshold | 30.5% [6.0%, 60.5%] | -33.6% [-53.7%, -18.8%] | 1.11 [0.35, 1.83] | 93.1× |

## 4. Pre-registered hypotheses

### H1 — Time-series trend

BTC trend CAGR was 28.0% versus 44.0% for BTC buy-and-hold. Its max drawdown was -33.6% versus -76.7%. Paired weekly-block difference CIs (strategy minus BTC) were CAGR [-80.9%, 26.3%], Sharpe [-0.54, 0.74], and max drawdown [7.8%, 53.3%]. The headline trend strategy did not satisfy all pre-registered return, drawdown, and paired-Sharpe criteria.

The secondary 4-hour study uses the pre-registered fixed equivalent lookbacks, past-only volatility, next-open execution, and a 1× cap:

| Asset | Strategy | OOS CAGR | Max DD | Sharpe |
|---|---|---:|---:|---:|
| — | not estimable: insufficient history for a purged fold | — | — | — |

The separately labeled perpetual variant can be long or short, remains capped at 1× per leg, executes at the next open, and includes observed funding and trading costs:

| Asset | Net CAGR | Max DD | Sharpe | Funding contribution | 100% short-margin breaches |
|---|---:|---:|---:|---:|---:|
| BTC-USD | 7.0% | -56.1% | 0.37 | -34.3% | 0 |
| ETH-USD | 4.9% | -74.2% | 0.33 | -26.6% | 0 |

### H2 — Funding crowding

BTC and ETH funding-filter runs are included in the benchmark table where coverage permits.

| Funding state | n | Mean 7d return | Median 7d return | Mean forward drawdown | 10th-pct drawdown |
|---|---:|---:|---:|---:|---:|
| positive extreme | 174 | 1.5% | 1.2% | -2.8% | -8.3% |
| normal | 2472 | 0.9% | 0.4% | -3.6% | -10.2% |
| negative extreme | 29 | 3.7% | 0.3% | -3.6% | -14.6% |

The table is descriptive; its funding state is formed from trailing observations and its returns begin after the decision bar.

**Filter effect:** BTC: 132 executed days changed; CAGR 28.0% to 21.6%, max drawdown -33.6% to -34.6%. ETH: 85 executed days changed; CAGR 29.9% to 25.2%, max drawdown -35.3% to -36.3%.

### H3 — Cross-sectional momentum

The fixed top-quintile strategy is reported against equal-weight top 20 above, including the pre-registered BTC regime-filter variant. Paired weekly-block difference CIs (momentum minus equal-weight top 20) were CAGR [-74.1%, -6.0%], Sharpe [-0.64, -0.00], and max drawdown [-22.2%, -0.1%].

## Meta-labeling

The fixed BTC H1 rule generated 73 labeled entry events; 36 (49.3%) hit the +2σ close barrier first. Every probability below is outer-fold out of sample, and every strategy return is next-open and net of the locked costs. **Result: No sized meta-model passed every pre-registered success gate.**

| Strategy | CAGR | Vol | Sharpe | Sortino | Calmar | Max DD | Underwater days | Worst month | Turnover | Paired Sharpe CI vs rule |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BTC trend | 28.0% | 28.6% | 1.005 | 1.159 | 0.833 | -33.6% | 753 | -13.7% | 120.0× | — |
| BTC trend + logistic sized | 7.9% | 12.5% | 0.669 | 0.600 | 0.427 | -18.5% | 931 | -12.2% | 32.8× | [-0.897, 0.225] |
| BTC trend + logistic threshold | 22.4% | 25.1% | 0.932 | 0.921 | 0.719 | -31.2% | 931 | -13.7% | 92.6× | [-0.415, 0.271] |
| BTC trend + boosting sized | 10.0% | 13.8% | 0.760 | 0.712 | 0.512 | -19.6% | 616 | -12.6% | 34.0× | [-0.807, 0.297] |
| BTC trend + boosting threshold | 30.5% | 27.3% | 1.110 | 1.194 | 0.906 | -33.6% | 753 | -13.7% | 93.1× | [-0.173, 0.367] |

### Probability quality

| Model | n | AUC vs base | Brier vs base | Log loss vs base | Paired difference CIs (model − base) |
|---|---:|---:|---:|---:|---|
| logistic | 64 | 0.475 vs 0.545 | 0.307 vs 0.251 | 0.925 vs 0.695 | AUC [-0.281, 0.119]; Brier [0.005, 0.114]; log loss [0.039, 0.452] |
| boosting | 64 | 0.496 vs 0.545 | 0.299 vs 0.251 | 1.029 vs 0.695 | AUC [-0.279, 0.160]; Brier [-0.005, 0.110]; log loss [0.037, 0.685] |

Logistic regression does not beat the base-rate baseline on probability quality (log loss and Brier) and does not beat it on ranking (AUC), under the locked CI rule.

Gradient boosting does not beat the base-rate baseline on probability quality (log loss and Brier) and does not beat it on ranking (AUC), under the locked CI rule.

### Feature-group ablation

Negative ΔAUC and positive ΔBrier mean the dropped group helped the full model. Strategy CIs are paired weekly-block differences for the ablated sized strategy minus its full-model counterpart.

| Model | Dropped group | ΔAUC (95% CI) | ΔBrier (95% CI) | ΔCAGR | ΔSharpe (95% CI) | ΔMax DD |
|---|---|---:|---:|---:|---:|---:|
| logistic | trend state | 0.049 [-0.000, 0.103] | -0.003 [-0.014, 0.007] | 0.9% | 0.036 [-0.030, 0.107] | -0.4% |
| logistic | volatility | -0.015 [-0.089, 0.057] | 0.006 [-0.007, 0.019] | -0.0% | -0.024 [-0.128, 0.079] | -0.7% |
| logistic | positioning | 0.022 [-0.048, 0.090] | -0.007 [-0.022, 0.008] | -0.1% | 0.029 [-0.043, 0.108] | 0.8% |
| logistic | market breadth | -0.033 [-0.110, 0.045] | -0.011 [-0.030, 0.007] | -1.1% | -0.022 [-0.127, 0.077] | 0.5% |
| logistic | calendar | 0.022 [-0.055, 0.092] | -0.002 [-0.016, 0.012] | 1.7% | 0.085 [-0.059, 0.235] | -0.6% |
| boosting | trend state | 0.014 [0.000, 0.047] | -0.002 [-0.005, 0.000] | 0.1% | 0.006 [-0.004, 0.027] | 0.0% |
| boosting | volatility | -0.010 [-0.037, 0.000] | 0.006 [0.000, 0.015] | -0.0% | -0.001 [-0.004, 0.001] | 0.0% |
| boosting | positioning | -0.009 [-0.031, 0.000] | 0.005 [0.000, 0.012] | -0.0% | -0.001 [-0.004, 0.001] | 0.0% |
| boosting | market breadth | 0.000 not estimable (zero-width bootstrap) | 0.000 [-0.001, 0.000] | -0.0% | -0.000 [-0.002, 0.000] | 0.0% |
| boosting | calendar | 0.000 not estimable (zero-width bootstrap) | 0.000 not estimable (zero-width bootstrap) | 0.0% | 0.000 not estimable (zero-width bootstrap) | 0.0% |

### Permutation null

The null refits model selection, preprocessing, calibration, and sizing after quarter-block label/return shuffles. Production uses 200 seeded permutations.

| Model | Real ΔSharpe | Null 95th percentile | Empirical p |
|---|---:|---:|---:|
| logistic | -0.336 | -0.139 | 0.373 |
| boosting | -0.245 | -0.039 | 0.647 |

### Multiple-testing correction

- Deflated Sharpe probability with 11 total strategy trials: rule 84.3%; logistic sized 54.7%; boosting sized 64.3%.
- CSCV PBO: prior seven-variant set 7.1%; combined eleven-variant set 32.9%.
- Optional on-chain features were skipped because no qualifying free, keyless source with a locked historical publication lag was available.
- The logistic sized model remains a paper-only forward candidate regardless of this backtest verdict; nothing trades automatically.


## 5. Deflated Sharpe and PBO

- Headline Deflated Sharpe Ratio probability: 84.3% across 11 registered strategy variants from prompts #6 and #7 (benchmarks excluded).
- CSCV Probability of Backtest Overfitting: 32.9%.
- These diagnostics reduce confidence for strategy selection across multiple variants; they do not turn a backtest into forward evidence.

## 6. Regime breakdown

Regime labels use BTC's trailing 200-day moving average with a fixed ±2% sideways band. Realized-volatility terciles are fitted on each training fold and applied to its test block.

| Regime | Strategy CAGR | Strategy max DD | BTC CAGR | BTC max DD |
|---|---:|---:|---:|---:|
| bull | 37.0% | -26.6% | 100.7% | -32.6% |
| bear | -3.7% | -24.6% | -30.4% | -93.4% |
| sideways | -2.8% | -21.9% | 3.4% | -34.9% |
| low-vol | 11.8% | -34.9% | 11.0% | -56.6% |
| mid-vol | 10.0% | -34.4% | 24.8% | -61.4% |
| high-vol | 4.7% | -16.8% | 4.5% | -46.8% |

## 7. Cost sensitivity

Headline results use 25 bps per side plus volume-scaled slippage. These fixed scenarios rerun the identical signal:

| Cost per side | CAGR | Max DD | Sharpe | Turnover |
|---:|---:|---:|---:|---:|
| 10 bps | 31.5% | -32.4% | 1.10 | 120.0× |
| 25 bps | 28.0% | -33.6% | 1.00 | 120.0× |
| 50 bps | 22.4% | -35.7% | 0.85 | 120.0× |

Cash earns 0% in every headline row. As a separately labeled sensitivity, assuming 4.0% annual cash/stablecoin yield changes the 60/40 reference CAGR from 30.2% to 32.3%. This is an assumption, not an observed or risk-free return.

## 8. Funding-carry diagnostic

BTC diagnostic net CAGR -5.0%, max drawdown -42.3%, Sharpe 0.11, with 0 conservative 100% one-day short-margin breaches. This 1×-per-leg delta-neutral diagnostic includes observed funding, both-leg entry/exit fees, and daily hedge rebalancing costs. H4 is diagnostic only and is never promoted from this report as a headline strategy.

## Figures

- [Out-of-sample equity curves](figures/equity-curves.png)
- [Drawdown curves](figures/drawdowns.png)
- [Rolling one-year Sharpe](figures/rolling-sharpe.png)
- [Exposure](figures/exposure.png)
- [Funding z-score versus forward returns](figures/funding-z-forward-return.png)

## 9. Limitations

- The original preregistration and results commits were created only 20 seconds apart. Git history proves their order, but it does not prove that those hypotheses were locked before the results were viewed. Meta-labeling v1 was instead pushed to GitHub separately before its real-data analysis.
- Public exchange product lists can omit delisted assets, creating upward survivorship bias.
- Exchange candles may be absent when no trades occur; missing bars are flagged rather than silently filled.
- Coinbase and Kraken availability differs by asset and history depth. Kraken's OHLC endpoint is intentionally treated as a shallow cross-check.
- Backtests cannot reproduce queue position, outages, spread shocks, taxes, or future market structure.
- Confidence intervals quantify sampling uncertainty under the chosen weekly-block scheme, not all model risk.

### Complete metric appendix

| Strategy | Volatility | Sortino | Calmar | Underwater days | Worst month | Worst week | Hit rate | Avg hold | Exposure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BTC buy-and-hold | 60.5% | 1.23 | 0.57 | 846 | -37.1% | -33.5% | 51.0% | 2464.0 | 100.0% |
| 60/40 BTC/cash | 36.3% | 1.23 | 0.54 | 841 | -23.6% | -19.0% | 51.0% | 2464.0 | 60.0% |
| Equal-weight top 10 | 82.1% | 0.68 | 0.08 | 1967 | -51.0% | -45.2% | 53.2% | 2414.0 | 100.0% |
| Equal-weight top 20 | 82.7% | 0.68 | 0.08 | 1967 | -42.6% | -47.3% | 52.6% | 2393.0 | 100.0% |
| BTC trend | 28.6% | 1.16 | 0.83 | 753 | -13.7% | -12.4% | 26.2% | 19.7 | 39.8% |
| ETH trend | 30.0% | 1.18 | 0.85 | 932 | -12.4% | -13.6% | 26.4% | 20.2 | 30.3% |
| Top-10 trend | 63.4% | 0.59 | 0.12 | 1967 | -30.0% | -44.8% | 41.0% | 41.4 | 69.5% |
| BTC risk-matched | 42.1% | 1.14 | 0.47 | 840 | -22.8% | -23.4% | 50.9% | 2464.0 | 76.4% |
| Top-20 momentum | 97.1% | 0.21 | -0.28 | 1970 | -63.3% | -50.4% | 50.5% | 2458.0 | 100.0% |
| Momentum + BTC regime | 63.7% | 0.46 | 0.09 | 915 | -37.2% | -29.1% | 26.7% | 19.7 | 51.3% |
| BTC trend + funding filter | 26.6% | 0.98 | 0.62 | 790 | -13.0% | -12.4% | 25.8% | 19.7 | 37.7% |
| ETH trend + funding filter | 28.9% | 1.04 | 0.69 | 932 | -12.4% | -13.6% | 26.3% | 20.2 | 29.3% |
| BTC trend + logistic sized | 12.5% | 0.60 | 0.43 | 931 | -12.2% | -8.4% | 21.9% | 20.6 | 12.9% |
| BTC trend + logistic threshold | 25.1% | 0.92 | 0.72 | 931 | -13.7% | -12.4% | 19.4% | 19.0 | 28.0% |
| BTC trend + boosting sized | 13.8% | 0.71 | 0.51 | 616 | -12.6% | -8.7% | 22.7% | 21.7 | 14.4% |
| BTC trend + boosting threshold | 27.3% | 1.19 | 0.91 | 753 | -13.7% | -12.4% | 22.7% | 21.7 | 32.8% |

### Exact point-in-time top-20 membership changes

| Month | Entered top 20 | Left top 20 |
|---|---|---|
| 2020-01 | ALGO-USD, BCH-USD, BTC-USD, DASH-USD, EOS-USD, ETC-USD, ETH-USD, GNT-USDC, LINK-USD, LTC-USD, REP-USD, XLM-USD, XRP-USD, XTZ-USD, ZRX-USD | — |
| 2020-02 | — | — |
| 2020-03 | — | — |
| 2020-04 | OXT-USD | — |
| 2020-05 | ATOM-USD | — |
| 2020-06 | KNC-USD | — |
| 2020-07 | — | — |
| 2020-08 | — | — |
| 2020-09 | OMG-USD | — |
| 2020-10 | COMP-USD, MKR-USD | GNT-USDC |
| 2020-11 | — | — |
| 2020-12 | BAND-USD, NMR-USD | MKR-USD, REP-USD |
| 2021-01 | LRC-USD, UMA-USD, UNI-USD, YFI-USD | DASH-USD, ETC-USD, KNC-USD, NMR-USD |
| 2021-02 | CGLD-USD, ETC-USD, MKR-USD, REN-USD | BAND-USD, OXT-USD, UMA-USD, XRP-USD |
| 2021-03 | DASH-USD, OXT-USD, UMA-USD | CGLD-USD, LRC-USD, MKR-USD |
| 2021-04 | AAVE-USD, BNT-USD, FIL-USD, GRT-USD, KNC-USD, NU-USD | COMP-USD, DASH-USD, EOS-USD, ETC-USD, UMA-USD, YFI-USD |
| 2021-05 | COMP-USD, EOS-USD, ETC-USD, MKR-USD, YFI-USD | BNT-USD, KNC-USD, NU-USD, OXT-USD, REN-USD |
| 2021-06 | SNX-USD | ZRX-USD |
| 2021-07 | ADA-USD, ANKR-USD, CGLD-USD, MATIC-USD, NU-USD | COMP-USD, MKR-USD, OMG-USD, SNX-USD, XTZ-USD |
| 2021-08 | COMP-USD, MANA-USD, SNX-USD, SUSHI-USD | ANKR-USD, CGLD-USD, EOS-USD, NU-USD |
| 2021-09 | DOGE-USD, EOS-USD, ICP-USD, XTZ-USD | COMP-USD, MANA-USD, SNX-USD, YFI-USD |
| 2021-10 | CGLD-USD, DOT-USD, OMG-USD, QNT-USD, SOL-USD | AAVE-USD, BCH-USD, EOS-USD, ETC-USD, GRT-USD |
| 2021-11 | GRT-USD, MANA-USD, NU-USD | CGLD-USD, QNT-USD, SUSHI-USD |
| 2021-12 | AMP-USD, ANKR-USD, ENJ-USD, IOTX-USD, LRC-USD, REQ-USD | FIL-USD, GRT-USD, ICP-USD, NU-USD, UNI-USD, XTZ-USD |
| 2022-01 | AVAX-USD, FIL-USD, ICP-USD, SHIB-USD, WLUNA-USD | AMP-USD, ANKR-USD, ENJ-USD, IOTX-USD, OMG-USD |
| 2022-02 | CRO-USD, CRV-USD | FIL-USD, REQ-USD |
| 2022-03 | GALA-USD | CRV-USD |
| 2022-04 | JASMY-USD | CRO-USD |
| 2022-05 | AAVE-USD | LRC-USD |
| 2022-06 | LRC-USD | AAVE-USD |
| 2022-07 | AAVE-USD, APE-USD, XTZ-USD | JASMY-USD, LRC-USD, WLUNA-USD |
| 2022-08 | ETC-USD, UNI-USD, VGX-USD | MANA-USD, XLM-USD, XTZ-USD |
| 2022-09 | FIL-USD, JASMY-USD, XLM-USD | AAVE-USD, ICP-USD, VGX-USD |
| 2022-10 | EOS-USD, VGX-USD | GALA-USD, JASMY-USD |
| 2022-11 | AAVE-USD, GALA-USD, MKR-USD, QNT-USD | APE-USD, EOS-USD, ETC-USD, VGX-USD |
| 2022-12 | APE-USD, CHZ-USD, MASK-USD | AAVE-USD, FIL-USD, MKR-USD |
| 2023-01 | FIL-USD, ICP-USD, YFII-USD | CHZ-USD, MASK-USD, QNT-USD |
| 2023-02 | AAVE-USD, FET-USD, MANA-USD | ICP-USD, UNI-USD, YFII-USD |
| 2023-03 | ACH-USD, GRT-USD, OP-USD, RNDR-USD | AAVE-USD, APE-USD, MANA-USD, XLM-USD |
| 2023-04 | MKR-USD, STX-USD, XLM-USD | ALGO-USD, GRT-USD, RNDR-USD |
| 2023-05 | ICP-USD, RNDR-USD | FIL-USD, MKR-USD |
| 2023-06 | BLUR-USD, GRT-USD, JASMY-USD, LDO-USD | ACH-USD, GALA-USD, ICP-USD, STX-USD |
| 2023-07 | ARB-USD, BCH-USD, STX-USD | BLUR-USD, JASMY-USD, LDO-USD |
| 2023-08 | AAVE-USD, ALGO-USD, COMP-USD, MKR-USD, SUSHI-USD | ATOM-USD, DOT-USD, FET-USD, RNDR-USD, STX-USD |
| 2023-09 | CRV-USD, HBAR-USD, UNI-USD, XRP-USD | AAVE-USD, ALGO-USD, COMP-USD, GRT-USD |
| 2023-10 | AAVE-USD, FET-USD, SHPING-USD, TRB-USD | AVAX-USD, CRV-USD, SUSHI-USD, UNI-USD |
| 2023-11 | AVAX-USD, DOT-USD, INJ-USD, RNDR-USD | ARB-USD, HBAR-USD, MKR-USD, SHPING-USD |
| 2023-12 | ATOM-USD, GRT-USD, LDO-USD, YFI-USD | BCH-USD, DOT-USD, OP-USD, TRB-USD |
| 2024-01 | ARB-USD, DOT-USD, ICP-USD, OP-USD, SEI-USD, STX-USD | AAVE-USD, ATOM-USD, GRT-USD, LDO-USD, XLM-USD, YFI-USD |
| 2024-02 | LDO-USD, SUI-USD, TIA-USD, TRB-USD | DOT-USD, FET-USD, MATIC-USD, SHIB-USD |
| 2024-03 | FET-USD, FIL-USD, JASMY-USD, MATIC-USD, SHIB-USD | ARB-USD, LDO-USD, LTC-USD, TIA-USD, TRB-USD |
| 2024-04 | BCH-USD, BONK-USD, LTC-USD, NEAR-USD | ICP-USD, OP-USD, STX-USD, SUI-USD |
| 2024-05 | HBAR-USD, JTO-USD, ONDO-USD, OP-USD | ADA-USD, FIL-USD, MATIC-USD, SEI-USD |
| 2024-06 | STX-USD, SUI-USD, TRB-USD | BCH-USD, HBAR-USD, INJ-USD |
| 2024-07 | INJ-USD, LDO-USD, UNI-USD | JTO-USD, SUI-USD, TRB-USD |
| 2024-08 | ADA-USD, AERO-USD, BCH-USD, TIA-USD, XLM-USD | LDO-USD, NEAR-USD, OP-USD, STX-USD, UNI-USD |
| 2024-09 | AAVE-USD, RARE-USD, SUI-USD | AERO-USD, BCH-USD, TIA-USD |
| 2024-10 | NEAR-USD, SEI-USD, TIA-USD | ADA-USD, RARE-USD, XLM-USD |
| 2024-11 | AERO-USD, APE-USD, APT-USD | AAVE-USD, JASMY-USD, NEAR-USD |
| 2024-12 | ADA-USD, DOT-USD, HBAR-USD, UNI-USD, XLM-USD | AERO-USD, APE-USD, APT-USD, INJ-USD, RNDR-USD |
| 2025-01 | AAVE-USD, ALGO-USD, JASMY-USD | FET-USD, SEI-USD, TIA-USD |
| 2025-02 | SWFTC-USD, XCN-USD | DOT-USD, UNI-USD |
| 2025-03 | ACH-USD, PEPE-USD | ALGO-USD, AVAX-USD |
| 2025-04 | AUCTION-USD, AVAX-USD, CRV-USD | ACH-USD, JASMY-USD, SWFTC-USD |
| 2025-05 | AERGO-USD, JASMY-USD, TRUMP-USD | AUCTION-USD, AVAX-USD, SHIB-USD |
| 2025-06 | MOODENG-USD, TOSHI-USD, UNI-USD, WIF-USD | AERGO-USD, CRV-USD, JASMY-USD, XCN-USD |
| 2025-07 | AERO-USD, SEI-USD, SYRUP-USD, TAO-USD | BONK-USD, LTC-USD, TOSHI-USD, TRUMP-USD |
| 2025-08 | BONK-USD, CRV-USD, LTC-USD, PENGU-USD, ZORA-USD | AAVE-USD, AERO-USD, SYRUP-USD, TAO-USD, UNI-USD |
| 2025-09 | AAVE-USD, AERO-USD, AVAX-USD, CRO-USD | CRV-USD, MOODENG-USD, SEI-USD, WIF-USD |
| 2025-10 | FARTCOIN-USD, IP-USD | AAVE-USD, ZORA-USD |
| 2025-11 | PUMP-USD, TAO-USD, ZEC-USD, ZORA-USD | AERO-USD, CRO-USD, IP-USD, ONDO-USD |
| 2025-12 | AERO-USD, DASH-USD, FIL-USD, ICP-USD, NEAR-USD, UNI-USD | BONK-USD, PENGU-USD, PEPE-USD, PUMP-USD, XLM-USD, ZORA-USD |
| 2026-01 | AAVE-USD, BCH-USD, PENGU-USD, XLM-USD | AERO-USD, DASH-USD, FIL-USD, NEAR-USD |
| 2026-02 | BONK-USD, DASH-USD, PEPE-USD, PUMP-USD | AAVE-USD, PENGU-USD, TAO-USD, UNI-USD |
| 2026-03 | BNKR-USD, MON-USD, ONDO-USD, TAO-USD | BONK-USD, DASH-USD, FARTCOIN-USD, ICP-USD |
| 2026-04 | FARTCOIN-USD, FET-USD, NEAR-USD, VVV-USD | BCH-USD, BNKR-USD, MON-USD, PUMP-USD |
| 2026-05 | AAVE-USD, MON-USD, PENGU-USD | FET-USD, NEAR-USD, ONDO-USD |
| 2026-06 | HYPE-USD, ICP-USD, INJ-USD, NEAR-USD, ONDO-USD | AAVE-USD, AVAX-USD, FARTCOIN-USD, MON-USD, PEPE-USD |
| 2026-07 | ALLO-USD, JTO-USD, WLD-USD | HBAR-USD, INJ-USD, PENGU-USD |
| 2026-08 | AAVE-USD, AERO-USD, HBAR-USD, UNI-USD | ALLO-USD, ICP-USD, JTO-USD, WLD-USD |
| 2026-09 | AVAX-USD, ENA-USD, PUMP-USD, TRUMP-USD | AAVE-USD, AERO-USD, HBAR-USD, VVV-USD |

## 10. Recommendations

Keep all strategies in research and paper-trading until the locked forward log is long enough to compare with these expectations. Do not add leverage to compensate for weak unlevered evidence. **This is not trading advice.**
