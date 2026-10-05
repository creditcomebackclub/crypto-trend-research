from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import StrategyRun, run_trend, run_weights
from .config import ROOT, load_config
from .metrics import calculate_metrics, drawdown, equity_curve
from .portfolio import PortfolioResult, equal_weight, simulate_portfolio
from .signals import (
    apply_funding_crowding_filter,
    cross_sectional_momentum_weights,
    funding_zscore,
    trailing_realized_volatility,
    trend_signal,
    trend_votes,
    volatility_targeted_weights,
)
from .stats import deflated_sharpe_ratio, paired_weekly_block_bootstrap_metric_cis, probability_of_backtest_overfitting, weekly_block_bootstrap_metric_cis
from .universe import membership_changes, point_in_time_universe
from .validation import expanding_walk_forward


def _fmt_pct(value: float) -> str:
    return "n/e" if not np.isfinite(value) else f"{value * 100:.1f}%"


def _fmt_num(value: float) -> str:
    return "n/e" if not np.isfinite(value) else f"{value:.2f}"


def _ci_text(ci: tuple[float, float] | None, percent: bool = False) -> str:
    if ci is None:
        return "not estimable"
    return f"[{_fmt_pct(ci[0])}, {_fmt_pct(ci[1])}]" if percent else f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def _metric_cis(run: StrategyRun, seed: int, samples: int) -> dict[str, tuple[float, float] | None]:
    return weekly_block_bootstrap_metric_cis(run.portfolio.returns, samples=samples, seed=seed)


def _slice_run(run: StrategyRun, dates: pd.DatetimeIndex) -> StrategyRun:
    p = run.portfolio
    mask = p.returns.index.intersection(dates)
    sliced = type(p)(
        p.returns.loc[mask], p.gross_returns.loc[mask], p.weights.loc[mask], p.turnover.loc[mask], p.costs.loc[mask], p.exposure.loc[mask]
    )
    return StrategyRun(run.name, sliced, calculate_metrics(sliced.returns, sliced.turnover, sliced.exposure))


def _make_figures(runs: dict[str, StrategyRun], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    selected = {k: v for k, v in runs.items() if k in {"BTC buy-and-hold", "BTC trend", "Top-20 momentum"}}
    if not selected:
        selected = dict(list(runs.items())[:3])
    plt.figure(figsize=(10, 5))
    for name, run in selected.items():
        equity_curve(run.portfolio.returns).plot(label=name)
    plt.yscale("log"); plt.title("Out-of-sample equity curves"); plt.ylabel("Growth of $1 (log scale)"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "equity-curves.png", dpi=150); plt.close()
    plt.figure(figsize=(10, 5))
    for name, run in selected.items():
        drawdown(run.portfolio.returns).plot(label=name)
    plt.title("Drawdowns"); plt.ylabel("Drawdown"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "drawdowns.png", dpi=150); plt.close()
    plt.figure(figsize=(10, 5))
    for name, run in selected.items():
        r = run.portfolio.returns
        rolling = r.rolling(365).mean() / r.rolling(365).std() * np.sqrt(365)
        rolling.plot(label=name)
    plt.title("Rolling one-year Sharpe"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "rolling-sharpe.png", dpi=150); plt.close()
    plt.figure(figsize=(10, 4))
    for name, run in selected.items():
        run.portfolio.exposure.plot(label=name)
    plt.title("Portfolio exposure"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "exposure.png", dpi=150); plt.close()


def _forward_funding_rows(z: pd.Series, close: pd.Series, threshold: float) -> tuple[list[str], str]:
    forward_return = close.shift(-7) / close - 1
    paths = pd.concat([close.shift(-day) / close - 1 for day in range(1, 8)], axis=1)
    forward_drawdown = paths.min(axis=1)
    buckets = {
        "positive extreme": z > threshold,
        "normal": z.abs() <= threshold,
        "negative extreme": z < -threshold,
    }
    rows = ["| Funding state | n | Mean 7d return | Median 7d return | Mean forward drawdown | 10th-pct drawdown |", "|---|---:|---:|---:|---:|---:|"]
    for label, mask in buckets.items():
        sample = pd.DataFrame({"return": forward_return[mask], "drawdown": forward_drawdown[mask]}).dropna()
        if len(sample) < 20:
            rows.append(f"| {label} | {len(sample)} | not estimable | not estimable | not estimable | not estimable |")
        else:
            rows.append(
                f"| {label} | {len(sample)} | {_fmt_pct(sample['return'].mean())} | {_fmt_pct(sample['return'].median())} | "
                f"{_fmt_pct(sample['drawdown'].mean())} | {_fmt_pct(sample['drawdown'].quantile(0.10))} |"
            )
    statement = "The table is descriptive; its funding state is formed from trailing observations and its returns begin after the decision bar."
    return rows, statement


def _secondary_4h_rows(
    secondary: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None,
    cfg: dict,
) -> list[str]:
    rows = ["| Asset | Strategy | OOS CAGR | Max DD | Sharpe |", "|---|---|---:|---:|---:|"]
    if secondary is None:
        return rows + ["| — | not estimable: no frozen 4-hour panel | — | — | — |"]
    opens, closes, volumes = secondary
    folds = expanding_walk_forward(
        closes.index,
        cfg["validation"]["minimum_training_days"],
        cfg["validation"]["test_block_months"],
        cfg["validation"]["purge_days"],
    )
    dates = pd.DatetimeIndex(sorted(set().union(*(set(fold.test) for fold in folds)))) if folds else closes.index[:0]
    if not len(dates):
        return rows + ["| — | not estimable: insufficient history for a purged fold | — | — | — |"]
    for asset in ("BTC-USD", "ETH-USD"):
        if asset not in closes:
            continue
        eligible = pd.DataFrame(False, index=closes.index, columns=closes.columns)
        eligible[asset] = True
        run = _slice_run(
            run_trend(
                f"{asset} 4h trend",
                opens,
                closes,
                volumes,
                cfg["strategy"]["four_hour_lookbacks"],
                cfg["strategy"]["headline_volatility_target"],
                cfg["costs"]["headline_bps_per_side"],
                eligible,
                periods_per_year=6 * 365,
            ),
            dates,
        )
        rows.append(f"| {asset} | fixed 4h ensemble | {_fmt_pct(run.metrics.cagr)} | {_fmt_pct(run.metrics.max_drawdown)} | {_fmt_num(run.metrics.sharpe)} |")
    return rows


def _data_audit_text(source_label: str) -> str:
    path = ROOT / "data" / "processed" / "data_quality.json"
    if "synthetic" in source_label.lower() or not path.exists():
        return "Synthetic fixture: production data-quality totals are not applicable."
    audit = json.loads(path.read_text(encoding="utf-8"))
    assets = audit.get("assets", {})
    missing = sum(int(item.get("missing_intervals", 0)) for item in assets.values())
    zero = sum(int(item.get("zero_volume", 0)) for item in assets.values())
    outliers = sum(int(item.get("return_outliers", 0)) for item in assets.values())
    catalog = audit.get("catalog", {})
    divergence = audit.get("cross_exchange", {})
    cross = "; ".join(
        f"{asset}: median {item.get('median_absolute_pct', float('nan')):.2f}%, p99 {item.get('p99_absolute_pct', float('nan')):.2f}% ({item.get('observations', 0)} overlaps)"
        for asset, item in divergence.items()
    ) or "not estimable"
    return (
        f"Across {len(assets)} selected base-asset histories, the audit found {missing:,} missing daily intervals, "
        f"{zero:,} zero-volume bars, and {outliers:,} robust return-outlier flags. The catalog exposed "
        f"{catalog.get('offline_or_delisted_visible', 0):,} offline/delisted products and cached "
        f"{catalog.get('cached_offline_or_delisted', 0):,}; it still cannot prove completeness for removed listings. "
        f"Cross-exchange close divergence — {cross}."
    )


def _perp_long_short_rows(
    perp_open: pd.DataFrame | None,
    perp_close: pd.DataFrame | None,
    perp_high: pd.DataFrame | None,
    perp_volume: pd.DataFrame | None,
    funding: pd.DataFrame | None,
    oos_dates: pd.DatetimeIndex,
    cfg: dict,
) -> list[str]:
    rows = ["| Asset | Net CAGR | Max DD | Sharpe | Funding contribution | 100% short-margin breaches |", "|---|---:|---:|---:|---:|---:|"]
    if any(item is None for item in (perp_open, perp_close, perp_high, perp_volume, funding)):
        return rows + ["| — | not estimable | — | — | — | — |"]
    for asset in ("BTC-USD", "ETH-USD"):
        if any(asset not in item.columns for item in (perp_open, perp_close, perp_high, perp_volume, funding)):
            continue
        index = perp_open.index.intersection(perp_close.index).intersection(funding.index)
        opens = perp_open.loc[index, [asset]]
        closes = perp_close.loc[index, [asset]]
        volumes = perp_volume.reindex(index)[[asset]]
        vote = trend_votes(closes, cfg["strategy"]["daily_lookbacks"])
        direction = pd.DataFrame(np.where(vote > 0, 1.0, -1.0), index=index, columns=[asset])
        scale = (
            cfg["strategy"]["headline_volatility_target"]
            / trailing_realized_volatility(closes).replace(0, np.nan)
        ).clip(upper=1.0).fillna(0.0)
        decisions = direction * scale
        portfolio = simulate_portfolio(
            opens,
            decisions,
            closes * volumes,
            bps_per_side=cfg["costs"]["headline_bps_per_side"],
        )
        paid = (portfolio.weights[asset] * funding.reindex(index)[asset]).reindex(portfolio.returns.index)
        adjusted = portfolio.returns - paid
        adjusted_portfolio = PortfolioResult(
            adjusted,
            portfolio.gross_returns - paid,
            portfolio.weights,
            portfolio.turnover,
            portfolio.costs,
            portfolio.exposure,
        )
        run = _slice_run(StrategyRun(f"{asset} perp long/short", adjusted_portfolio, calculate_metrics(adjusted)), oos_dates)
        short = portfolio.weights[asset] < 0
        adverse = perp_high.reindex(index)[asset] / perp_open.reindex(index)[asset] - 1
        breaches = int((short & (adverse >= 1.0)).reindex(oos_dates, fill_value=False).sum())
        funding_contribution = float((-paid.reindex(oos_dates)).dropna().sum())
        rows.append(
            f"| {asset} | {_fmt_pct(run.metrics.cagr)} | {_fmt_pct(run.metrics.max_drawdown)} | {_fmt_num(run.metrics.sharpe)} | "
            f"{_fmt_pct(funding_contribution)} | {breaches} |"
        )
    return rows if len(rows) > 2 else rows + ["| — | not estimable | — | — | — | — |"]


def _regime_rows(
    strategy: StrategyRun,
    btc: StrategyRun,
    close: pd.Series,
    folds,
) -> list[str]:
    direction = pd.Series(index=close.index, dtype="object")
    volatility_bucket = pd.Series(index=close.index, dtype="object")
    ma200 = close.rolling(200, min_periods=200).mean()
    realized = close.pct_change(fill_method=None).rolling(30, min_periods=30).std() * np.sqrt(365)
    for fold in folds:
        train_vol = realized.loc[fold.train].dropna()
        if len(train_vol) < 30:
            continue
        low, high = train_vol.quantile([1 / 3, 2 / 3])
        ratio = close.loc[fold.test] / ma200.loc[fold.test] - 1
        direction.loc[fold.test] = np.where(ratio > 0.02, "bull", np.where(ratio < -0.02, "bear", "sideways"))
        vol = realized.loc[fold.test]
        volatility_bucket.loc[fold.test] = np.where(vol < low, "low", np.where(vol > high, "high", "mid"))
    rows = ["| Regime | Strategy CAGR | Strategy max DD | BTC CAGR | BTC max DD |", "|---|---:|---:|---:|---:|"]
    for label in ["bull", "bear", "sideways"]:
        dates = direction.index[direction == label].intersection(strategy.portfolio.returns.index)
        if len(dates) < 21:
            rows.append(f"| {label} | not estimable | not estimable | not estimable | not estimable |")
            continue
        sm = calculate_metrics(strategy.portfolio.returns.loc[dates])
        bm = calculate_metrics(btc.portfolio.returns.reindex(dates).dropna())
        rows.append(f"| {label} | {_fmt_pct(sm.cagr)} | {_fmt_pct(sm.max_drawdown)} | {_fmt_pct(bm.cagr)} | {_fmt_pct(bm.max_drawdown)} |")
    for label in ["low", "mid", "high"]:
        dates = volatility_bucket.index[volatility_bucket == label].intersection(strategy.portfolio.returns.index)
        if len(dates) < 21:
            rows.append(f"| {label}-vol | not estimable | not estimable | not estimable | not estimable |")
            continue
        sm = calculate_metrics(strategy.portfolio.returns.loc[dates])
        bm = calculate_metrics(btc.portfolio.returns.reindex(dates).dropna())
        rows.append(f"| {label}-vol | {_fmt_pct(sm.cagr)} | {_fmt_pct(sm.max_drawdown)} | {_fmt_pct(bm.cagr)} | {_fmt_pct(bm.max_drawdown)} |")
    return rows


def run_research(
    open_prices: pd.DataFrame,
    close_prices: pd.DataFrame,
    volume: pd.DataFrame,
    funding: pd.DataFrame | None = None,
    perp_open: pd.DataFrame | None = None,
    perp_close: pd.DataFrame | None = None,
    perp_high: pd.DataFrame | None = None,
    perp_volume: pd.DataFrame | None = None,
    secondary_4h: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None,
    source_label: str = "frozen snapshot",
    config: dict | None = None,
) -> tuple[str, dict[str, StrategyRun]]:
    cfg = config or load_config()
    common = open_prices.index.intersection(close_prices.index).intersection(volume.index)
    open_prices = open_prices.loc[common].sort_index()
    close_prices = close_prices.loc[common].sort_index()
    volume = volume.loc[common].sort_index()
    excluded = set(cfg["universe"]["excluded_symbols"])
    top10 = point_in_time_universe(close_prices, volume, 10, excluded, cfg["data"]["min_history_days"])
    top20 = point_in_time_universe(close_prices, volume, 20, excluded, cfg["data"]["min_history_days"])
    folds = expanding_walk_forward(
        common,
        cfg["validation"]["minimum_training_days"],
        cfg["validation"]["test_block_months"],
        cfg["validation"]["purge_days"],
    )
    oos_dates = pd.DatetimeIndex(sorted(set().union(*(set(fold.test) for fold in folds)))) if folds else common[0:0]
    if len(oos_dates) == 0:
        raise ValueError("not enough history for an out-of-sample fold")
    bps = cfg["costs"]["headline_bps_per_side"]
    lookbacks = cfg["strategy"]["daily_lookbacks"]
    target = cfg["strategy"]["headline_volatility_target"]
    btc_col = next((c for c in close_prices if c.upper().split("-")[0] in {"BTC", "XBT"}), None)
    if btc_col is None:
        raise ValueError("BTC column is required")
    btc_eligible = pd.DataFrame(False, index=common, columns=close_prices.columns); btc_eligible[btc_col] = True
    eth_col = next((c for c in close_prices if c.upper().split("-")[0] == "ETH"), None)
    runs: dict[str, StrategyRun] = {}
    bh_weight = btc_eligible.astype(float)
    runs["BTC buy-and-hold"] = _slice_run(run_weights("BTC buy-and-hold", open_prices, close_prices, volume, bh_weight, bps), oos_dates)
    cash40 = bh_weight * 0.60
    runs["60/40 BTC/cash"] = _slice_run(run_weights("60/40 BTC/cash", open_prices, close_prices, volume, cash40, bps), oos_dates)
    runs["Equal-weight top 10"] = _slice_run(run_weights("Equal-weight top 10", open_prices, close_prices, volume, equal_weight(top10), bps), oos_dates)
    runs["Equal-weight top 20"] = _slice_run(run_weights("Equal-weight top 20", open_prices, close_prices, volume, equal_weight(top20), bps), oos_dates)
    runs["BTC trend"] = _slice_run(run_trend("BTC trend", open_prices, close_prices, volume, lookbacks, target, bps, btc_eligible), oos_dates)
    if eth_col is not None:
        eth_eligible = pd.DataFrame(False, index=common, columns=close_prices.columns); eth_eligible[eth_col] = True
        runs["ETH trend"] = _slice_run(run_trend("ETH trend", open_prices, close_prices, volume, lookbacks, target, bps, eth_eligible), oos_dates)
    runs["Top-10 trend"] = _slice_run(run_trend("Top-10 trend", open_prices, close_prices, volume, lookbacks, target, bps, top10), oos_dates)
    btc_vol = trailing_realized_volatility(close_prices[[btc_col]])
    risk_match = (target / btc_vol).clip(upper=1.0).fillna(0.0).reindex(columns=close_prices.columns, fill_value=0.0)
    runs["BTC risk-matched"] = _slice_run(run_weights("BTC risk-matched", open_prices, close_prices, volume, risk_match, bps), oos_dates)
    momentum = cross_sectional_momentum_weights(close_prices, top20)
    runs["Top-20 momentum"] = _slice_run(run_weights("Top-20 momentum", open_prices, close_prices, volume, momentum, bps), oos_dates)
    btc_regime = trend_signal(close_prices[[btc_col]], lookbacks)[btc_col]
    filtered_momentum = momentum.mul(btc_regime, axis=0)
    runs["Momentum + BTC regime"] = _slice_run(run_weights("Momentum + BTC regime", open_prices, close_prices, volume, filtered_momentum, bps), oos_dates)
    funding_diagnostic = "Aligned public funding history was unavailable."
    funding_rows = ["Not estimable."]
    if funding is not None and btc_col in funding.columns:
        threshold = cfg["strategy"]["funding_extreme_z"]
        for asset_col, label in ((btc_col, "BTC"), (eth_col, "ETH")):
            if asset_col is None or asset_col not in funding.columns:
                continue
            base_signal = trend_signal(close_prices[[asset_col]], lookbacks)
            base_vol = trailing_realized_volatility(close_prices[[asset_col]])
            base = volatility_targeted_weights(base_signal, base_vol, target)
            aligned_funding = funding[[asset_col]].reindex(common).ffill()
            aligned_z = funding_zscore(aligned_funding, cfg["strategy"]["funding_z_lookback"])
            filt = apply_funding_crowding_filter(base, aligned_z, threshold)
            full = filt.reindex(columns=close_prices.columns, fill_value=0.0)
            name = f"{label} trend + funding filter"
            runs[name] = _slice_run(run_weights(name, open_prices, close_prices, volume, full, bps), oos_dates)
        z = funding_zscore(funding[[btc_col]].reindex(common).ffill(), cfg["strategy"]["funding_z_lookback"])[btc_col]
        forward_7d = close_prices[btc_col].shift(-7) / close_prices[btc_col] - 1
        funding_rows, funding_diagnostic = _forward_funding_rows(z, close_prices[btc_col], threshold)
        figure_data = pd.DataFrame({"funding_z": z, "forward_7d": forward_7d}).dropna()
        if len(figure_data):
            plt.figure(figsize=(8, 5)); plt.scatter(figure_data["funding_z"], figure_data["forward_7d"], s=8, alpha=0.35)
            plt.axvline(cfg["strategy"]["funding_extreme_z"], color="red", linestyle="--")
            plt.xlabel("Trailing funding z-score"); plt.ylabel("Forward 7-day BTC return"); plt.title("Funding crowding and forward returns"); plt.tight_layout()
            figure_path = ROOT / "reports" / "figures"; figure_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(figure_path / "funding-z-forward-return.png", dpi=150); plt.close()
    seed = cfg["analysis"]["seed"]
    samples = cfg["validation"]["bootstrap_samples"]
    cis = {name: _metric_cis(run, seed + i * 10, samples) for i, (name, run) in enumerate(runs.items())}
    filter_effects = []
    for label in ("BTC", "ETH"):
        base_name, filtered_name = f"{label} trend", f"{label} trend + funding filter"
        if base_name in runs and filtered_name in runs:
            base_weights = runs[base_name].portfolio.weights
            filtered_weights = runs[filtered_name].portfolio.weights.reindex_like(base_weights)
            changed = int(((base_weights - filtered_weights).abs().sum(axis=1) > 1e-12).sum())
            filter_effects.append(
                f"{label}: {changed} executed days changed; CAGR {_fmt_pct(runs[base_name].metrics.cagr)} to "
                f"{_fmt_pct(runs[filtered_name].metrics.cagr)}, max drawdown {_fmt_pct(runs[base_name].metrics.max_drawdown)} to "
                f"{_fmt_pct(runs[filtered_name].metrics.max_drawdown)}."
            )
    funding_filter_effect = " ".join(filter_effects) or "Not estimable."
    strategy_names = [name for name in runs if name not in {"BTC buy-and-hold", "60/40 BTC/cash", "Equal-weight top 10", "Equal-weight top 20", "BTC risk-matched"}]
    variants = pd.concat({name: runs[name].portfolio.returns for name in strategy_names}, axis=1).dropna()
    pbo = probability_of_backtest_overfitting(variants)
    headline = "BTC trend"
    dsr = deflated_sharpe_ratio(runs[headline].portfolio.returns, len(strategy_names))
    btc = runs["BTC buy-and-hold"]
    strategy = runs[headline]
    paired = paired_weekly_block_bootstrap_metric_cis(strategy.portfolio.returns, btc.portfolio.returns, samples=samples, seed=seed)
    momentum_paired = paired_weekly_block_bootstrap_metric_cis(
        runs["Top-20 momentum"].portfolio.returns,
        runs["Equal-weight top 20"].portfolio.returns,
        samples=samples,
        seed=seed + 500,
    )
    matched = strategy.metrics.cagr >= btc.metrics.cagr - cfg["success"]["max_cagr_shortfall_vs_btc"]
    dd_better = abs(strategy.metrics.max_drawdown) <= abs(btc.metrics.max_drawdown) * (1 - cfg["success"]["minimum_drawdown_reduction"])
    sharpe_supported = paired["sharpe"] is not None and paired["sharpe"][0] > cfg["success"]["paired_sharpe_difference_ci_lower"]
    interpretation = (
        "The headline trend strategy satisfied all three pre-registered return, drawdown, and paired-Sharpe criteria."
        if matched and dd_better and sharpe_supported
        else "The headline trend strategy did not satisfy all pre-registered return, drawdown, and paired-Sharpe criteria."
    )
    cost_rows = ["| Cost per side | CAGR | Max DD | Sharpe | Turnover |", "|---:|---:|---:|---:|---:|"]
    for scenario_bps in cfg["costs"]["sensitivity_bps_per_side"]:
        scenario = _slice_run(
            run_trend(f"BTC trend {scenario_bps} bps", open_prices, close_prices, volume, lookbacks, target, scenario_bps, btc_eligible),
            oos_dates,
        )
        cost_rows.append(
            f"| {scenario_bps} bps | {_fmt_pct(scenario.metrics.cagr)} | {_fmt_pct(scenario.metrics.max_drawdown)} | {_fmt_num(scenario.metrics.sharpe)} | {scenario.metrics.turnover:.1f}× |"
        )
    regime_rows = _regime_rows(strategy, btc, close_prices[btc_col], folds)
    changes = membership_changes(top20.loc[oos_dates])
    entries = int(changes["entered"].map(len).sum()) if len(changes) else 0
    exits = int(changes["left"].map(len).sum()) if len(changes) else 0
    membership_rows = ["| Month | Entered top 20 | Left top 20 |", "|---|---|---|"]
    for row in changes.itertuples(index=False):
        membership_rows.append(f"| {row.month} | {', '.join(row.entered) or '—'} | {', '.join(row.left) or '—'} |")
    full_rows = ["| Strategy | Volatility | Sortino | Calmar | Underwater days | Worst month | Worst week | Hit rate | Avg hold | Exposure |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, run in runs.items():
        m = run.metrics
        full_rows.append(f"| {name} | {_fmt_pct(m.annualized_volatility)} | {_fmt_num(m.sortino)} | {_fmt_num(m.calmar)} | {m.longest_underwater_days} | {_fmt_pct(m.worst_month)} | {_fmt_pct(m.worst_week)} | {_fmt_pct(m.hit_rate)} | {m.average_holding_days:.1f} | {_fmt_pct(m.exposure)} |")
    cash_run = runs["60/40 BTC/cash"]
    cash_assumption = cfg["costs"]["cash_yield_sensitivity"]
    cash_return = cash_run.portfolio.returns + (1 - cash_run.portfolio.exposure) * ((1 + cash_assumption) ** (1 / 365.25) - 1)
    cash_sensitivity = calculate_metrics(cash_return, cash_run.portfolio.turnover, cash_run.portfolio.exposure)
    carry_text = "Not estimable: overlapping Deribit perpetual prices and funding were unavailable."
    if funding is not None and perp_close is not None and btc_col in funding.columns and btc_col in perp_close.columns:
        spot_ret = close_prices[btc_col].pct_change(fill_method=None)
        perp_ret = perp_close[btc_col].reindex(common).pct_change(fill_method=None)
        funding_ret = funding[btc_col].reindex(common)
        carry_inputs = pd.concat({"spot": spot_ret, "perp": perp_ret, "funding": funding_ret}, axis=1).dropna()
        hedge_turnover = (carry_inputs["spot"] - carry_inputs["perp"]).abs()
        carry = (
            carry_inputs["spot"] - carry_inputs["perp"] + carry_inputs["funding"]
            - hedge_turnover * 2 * bps / 10_000
        ).reindex(oos_dates).dropna()
        if len(carry):
            carry.iloc[0] -= 2 * bps / 10_000
            carry.iloc[-1] -= 2 * bps / 10_000
            liquidations = 0
            if perp_high is not None and btc_col in perp_high.columns:
                high_jump = perp_high[btc_col].reindex(common) / perp_close[btc_col].reindex(common).shift(1) - 1
                liquidations = int((high_jump.reindex(oos_dates) >= 1.0).sum())
            cm = calculate_metrics(carry)
            carry_text = (
                f"BTC diagnostic net CAGR {_fmt_pct(cm.cagr)}, max drawdown {_fmt_pct(cm.max_drawdown)}, "
                f"Sharpe {_fmt_num(cm.sharpe)}, with {liquidations} conservative 100% one-day short-margin breaches. "
                "This 1×-per-leg delta-neutral diagnostic includes observed funding, both-leg entry/exit fees, and daily hedge rebalancing costs."
            )
    secondary_rows = _secondary_4h_rows(secondary_4h, cfg)
    perp_long_short_rows = _perp_long_short_rows(perp_open, perp_close, perp_high, perp_volume, funding, oos_dates, cfg)
    audit_text = _data_audit_text(source_label)
    synthetic_banner = "> **Synthetic fixture only — the numbers below test software behavior and are not market evidence.**\n\n" if "synthetic" in source_label.lower() else ""
    table = ["| Strategy | CAGR (95% CI) | Max DD (95% CI) | Sharpe (95% CI) | Turnover |", "|---|---:|---:|---:|---:|"]
    for name, run in runs.items():
        table.append(
            f"| {name} | {_fmt_pct(run.metrics.cagr)} {_ci_text(cis[name]['cagr'], True)} | {_fmt_pct(run.metrics.max_drawdown)} {_ci_text(cis[name]['max_drawdown'], True)} | {_fmt_num(run.metrics.sharpe)} {_ci_text(cis[name]['sharpe'])} | {run.metrics.turnover:.1f}× |"
        )
    report = f"""# Crypto Trend Research Report

{synthetic_banner}

## 1. Question

Can a systematic strategy on liquid crypto assets match Bitcoin buy-and-hold with materially smaller drawdowns, net of realistic costs, out of sample?

**Result:** {interpretation} This report is research, not trading advice.

## 2. Data and universe

- Source: {source_label}.
- Fixed analysis end: {cfg['analysis']['end']}.
- Out-of-sample observations: {len(oos_dates):,} across {len(folds)} purged expanding-window test blocks.
- Universe membership is reconstructed monthly from trailing 30-day source-exchange dollar volume with a 90-day history requirement.
- Stablecoins and wrapped or pegged assets are excluded. Active-product APIs do not guarantee recovery of every delisted asset; unavailable dead assets bias historical results upward through survivorship.
- The observed top-20 test-period membership recorded {entries} entries and {exits} exits; the detailed membership audit is generated from the same point-in-time matrix.
- Data-quality checks retain an audit trail for missing bars, zero volume, outliers, timestamp alignment, and BTC/ETH cross-exchange divergence. No repair is silent.

**Audit summary:** {audit_text}

## 3. Benchmarks

{chr(10).join(table)}

## 4. Pre-registered hypotheses

### H1 — Time-series trend

BTC trend CAGR was {_fmt_pct(strategy.metrics.cagr)} versus {_fmt_pct(btc.metrics.cagr)} for BTC buy-and-hold. Its max drawdown was {_fmt_pct(strategy.metrics.max_drawdown)} versus {_fmt_pct(btc.metrics.max_drawdown)}. Paired weekly-block difference CIs (strategy minus BTC) were CAGR {_ci_text(paired['cagr'], True)}, Sharpe {_ci_text(paired['sharpe'])}, and max drawdown {_ci_text(paired['max_drawdown'], True)}. {interpretation}

The secondary 4-hour study uses the pre-registered fixed equivalent lookbacks, past-only volatility, next-open execution, and a 1× cap:

{chr(10).join(secondary_rows)}

The separately labeled perpetual variant can be long or short, remains capped at 1× per leg, executes at the next open, and includes observed funding and trading costs:

{chr(10).join(perp_long_short_rows)}

### H2 — Funding crowding

{'BTC and ETH funding-filter runs are included in the benchmark table where coverage permits.' if 'BTC trend + funding filter' in runs else 'Not estimable from this snapshot because aligned public funding history was unavailable.'}

{chr(10).join(funding_rows)}

{funding_diagnostic}

**Filter effect:** {funding_filter_effect}

### H3 — Cross-sectional momentum

The fixed top-quintile strategy is reported against equal-weight top 20 above, including the pre-registered BTC regime-filter variant. Paired weekly-block difference CIs (momentum minus equal-weight top 20) were CAGR {_ci_text(momentum_paired['cagr'], True)}, Sharpe {_ci_text(momentum_paired['sharpe'])}, and max drawdown {_ci_text(momentum_paired['max_drawdown'], True)}.

## 5. Deflated Sharpe and PBO

- Headline Deflated Sharpe Ratio probability: {_fmt_pct(dsr)} across {len(strategy_names)} strategy variants (benchmarks excluded).
- CSCV Probability of Backtest Overfitting: {_fmt_pct(pbo)}.
- These diagnostics reduce confidence for strategy selection across multiple variants; they do not turn a backtest into forward evidence.

## 6. Regime breakdown

Regime labels use BTC's trailing 200-day moving average with a fixed ±2% sideways band. Realized-volatility terciles are fitted on each training fold and applied to its test block.

{chr(10).join(regime_rows)}

## 7. Cost sensitivity

Headline results use {bps} bps per side plus volume-scaled slippage. These fixed scenarios rerun the identical signal:

{chr(10).join(cost_rows)}

Cash earns 0% in every headline row. As a separately labeled sensitivity, assuming {cash_assumption * 100:.1f}% annual cash/stablecoin yield changes the 60/40 reference CAGR from {_fmt_pct(cash_run.metrics.cagr)} to {_fmt_pct(cash_sensitivity.cagr)}. This is an assumption, not an observed or risk-free return.

## 8. Funding-carry diagnostic

{carry_text} H4 is diagnostic only and is never promoted from this report as a headline strategy.

## Figures

- [Out-of-sample equity curves](figures/equity-curves.png)
- [Drawdown curves](figures/drawdowns.png)
- [Rolling one-year Sharpe](figures/rolling-sharpe.png)
- [Exposure](figures/exposure.png)
- [Funding z-score versus forward returns](figures/funding-z-forward-return.png)

## 9. Limitations

- Public exchange product lists can omit delisted assets, creating upward survivorship bias.
- Exchange candles may be absent when no trades occur; missing bars are flagged rather than silently filled.
- Coinbase and Kraken availability differs by asset and history depth. Kraken's OHLC endpoint is intentionally treated as a shallow cross-check.
- Backtests cannot reproduce queue position, outages, spread shocks, taxes, or future market structure.
- Confidence intervals quantify sampling uncertainty under the chosen weekly-block scheme, not all model risk.

### Complete metric appendix

{chr(10).join(full_rows)}

### Exact point-in-time top-20 membership changes

{chr(10).join(membership_rows)}

## 10. Recommendations

Keep all strategies in research and paper-trading until the locked forward log is long enough to compare with these expectations. Do not add leverage to compensate for weak unlevered evidence. **This is not trading advice.**
"""
    _make_figures(runs, ROOT / "reports" / "figures")
    return report, runs


def write_results(report: str, runs: dict[str, StrategyRun], destination: Path | None = None) -> None:
    target = destination or ROOT / "reports" / "REPORT.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report, encoding="utf-8")
    summary = {name: run.metrics.to_dict() for name, run in runs.items()}
    (target.parent / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8")
