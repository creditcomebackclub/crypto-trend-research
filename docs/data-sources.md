# Public data sources

All collectors are keyless, read-only, and designed for reproduction from the United States. The fixed analysis end date in `config.yaml` prevents results from changing as APIs add new bars.

## Coinbase Exchange spot

- Products: `GET https://api.exchange.coinbase.com/products`
- Candles: `GET https://api.exchange.coinbase.com/products/{product_id}/candles`
- The official candle endpoint permits 60, 300, 900, 3,600, 21,600, and 86,400-second granularity and at most 300 candles per request. It warns that intervals without ticks are absent and history may be incomplete.
- Public REST allowance: 10 requests/second/IP, burst 15. The collector deliberately sleeps 120 ms between paginated calls and retries transient failures.
- History depth: probed per product and recorded in the snapshot audit. There is no promise that a current-products response exposes every delisted market.
- Official docs: <https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles> and <https://docs.cdp.coinbase.com/exchange/introduction/rate-limits-overview>.

## Kraken spot cross-check

- Pairs: `GET https://api.kraken.com/0/public/AssetPairs`
- OHLC: `GET https://api.kraken.com/0/public/OHLC`
- The official endpoint returns at most the 720 most recent entries; older observations cannot be recovered through `since`. The last row is an uncommitted current interval and must be excluded from historical comparisons.
- History depth: about 720 bars at the requested interval. Kraken is therefore a recent cross-exchange quality check, not the primary long-history backtest source.
- Official docs: <https://docs.kraken.com/api-reference/market-data/get-ohlc-data> and <https://docs.kraken.com/exchange/guides/rest/ratelimits>.

## Deribit perpetual funding

- Funding: `GET https://www.deribit.com/api/v2/public/get_funding_rate_history`
- Perpetual candles: `GET https://www.deribit.com/api/v2/public/get_tradingview_chart_data`
- Parameters: `instrument_name`, millisecond `start_timestamp`, millisecond `end_timestamp`, and `count` up to 1,000 per page.
- Funding history is returned hourly. The research aggregates `interest_1h` into realized daily funding; `interest_8h` is a rolling quote and is not summed across hourly rows.
- The shared client paces all public calls to one request every 120 ms and retries with exponential backoff. Deribit documents credit-based limits by endpoint and account state rather than one fixed anonymous-request rate, so the collector stays deliberately below the other documented venue limits.
- The collector attempts BTC, ETH, and SOL perpetual instruments and records unavailable instruments rather than substituting another venue. Availability and first/last timestamps are part of the snapshot audit.
- Official docs: <https://docs.deribit.com/#public-get_funding_rate_history>, <https://docs.deribit.com/#public-get_tradingview_chart_data>, and <https://docs.deribit.com/#rate-limits>.

## Survivorship implication

The APIs above do not guarantee that every delisted market remains discoverable and downloadable. Missing dead assets make historical top-N universes look healthier than the true opportunity set, so the likely direction of bias is upward. The report lists every observed entry and exit and states this limitation next to the headline results.
