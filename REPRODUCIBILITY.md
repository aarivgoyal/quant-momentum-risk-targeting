# Reproducibility

Every table and figure in the paper is reproducible offline from the cached data and a fixed random seed. No live data download is required.

## Environment

- **Python:** 3.10–3.12 (developed on 3.12)
- **Packages:** see `requirements.txt`
  - Required: `numpy>=1.23`, `pandas>=2.2`, `matplotlib>=3.6`
  - `pandas>=2.2` matters: the script uses the `'ME'` month-end resample alias.
  - Optional: `yfinance` (only to download fresh data on a first run), `scipy` (not needed by the all-in-one script; it is fully self-contained).

## Data

- **Source:** Yahoo Finance, via the `yfinance` library (dividend/split-adjusted daily closes).
- **Universe:** VOO (U.S. equities), VXUS (international equities), BND (aggregate bonds), SHY (short-term Treasuries / cash-like sleeve). BIL (1–3 month T-bill ETF) supplies the risk-free rate.
- **Sample span:** 2012-01-03 to 2026-06-15. The backtest is active from 2013-02-01 (the first month after a full 252-day momentum lookback is available).

### The two cached data files (required for exact reproduction)

| File | What it is | Used for |
|---|---|---|
| `prices_cache.csv` | Daily adjusted prices for VOO, VXUS, BND, SHY (3,633 rows) | All returns, signals, weights, and benchmarks |
| `rf_cache.csv` | Daily return of BIL as the risk-free rate (3,632 rows) | Sharpe / Sortino excess-return calculations |

The loaders (`load_prices`, `load_rf_daily`) read these caches first. As long as both files are in the working directory, the script never contacts the internet and reproduces the paper's exact numbers. If the caches are absent and `yfinance` is installed, the script will download and re-create them.

## Fixed random seed

All randomness lives in the block-bootstrap significance tests and uses a fixed **seed = 42** (set in CELL 1 as `RNG = np.random.default_rng(42)`, and passed as `seed=42` into the bootstrap functions). Re-running therefore yields identical confidence intervals and p-values every time.

## How to run

1. Put `quant_momentum_ALL_IN_ONE.py`, `prices_cache.csv`, and `rf_cache.csv` in the same folder (or upload the two CSVs into the Colab files panel — or just use `colab_run.ipynb`, which clones this repo so the caches come along automatically).
2. Run:

```bash
pip install -r requirements.txt
python quant_momentum_ALL_IN_ONE.py
```

This prints every table (baseline, in/out-of-sample, walk-forward, crisis windows, up/down conditioning, sensitivity, bootstrap significance, extension, and the full v2-vs-v3 crash-guard analysis) and saves all seven figures.

## Regenerating figures and tables

- **Tables:** printed to stdout by the run above.
- **Figures:** saved automatically into `figures/` as `fig1_equity.png`, `fig2_drawdown.png`, `fig3_asset_timeline.png`, `fig4_sensitivity.png`, `fig5_v3_equity.png`, `fig6_covid_zoom.png`, `fig7_crisis_bars.png` — the same paths the README's image links point to, so everything resolves with no manual file moves.
- **Interactive page:** `index.html` is self-contained — it embeds the precomputed results as JSON and recomputes the charts/tables in the browser, so it needs no build step. Its numbers match the script output (e.g., baseline 8.30% CAGR / 0.58 Sharpe / −28.8% max drawdown).

## Known limitations

- Historical backtest only; the regime mix of the sample shapes every conclusion.
- Compact four-ETF universe (aids interpretability, limits diversification).
- Turnover-based transaction costs are charged, but taxes, bid-ask spreads, slippage, and market impact are not separately modeled.
- The crash guard was designed *after* diagnosing the COVID failure, so it is an engineered extension, not evidence of future crash avoidance.
- The monthly signal latency is both a design choice and the direct cause of the fast-crash weakness.

*This project is for educational research only and is not investment advice.*
