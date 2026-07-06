# ============================================================================
#  ALL-IN-ONE SCRIPT  —  paste this whole thing into ONE Google Colab cell.
#  It contains the full v2 engine AND the v3 crash-guard fix, and runs the
#  entire study (prints all tables + shows all 7 figures) when executed.
#
#  DATA / CACHE: the first run downloads prices from Yahoo and caches them;
#  later runs reuse the cache. To reproduce the paper's EXACT numbers offline,
#  put prices_cache.csv and rf_cache.csv in the working directory (or upload
#  them into the Colab files panel) before running. The loaders read the cache
#  first, so with those two files present the script needs no internet at all.
#
#  REPRODUCIBILITY: all randomness (the block bootstrap) uses a FIXED SEED = 42,
#  set in CELL 1. The tunable strategy parameters are the ALL-CAPS knobs at the
#  top of CELL 1 (lookback, target vol, transaction cost, etc.).
#
#  DISCLAIMER: This project is for educational research only and is not
#  investment advice.
# ============================================================================

"""
================================================================================
 QUANTITATIVE MULTI-ASSET MOMENTUM STRATEGY WITH RISK TARGETING  —  VERSION 2
 (heavily commented edition — read top to bottom like a guided tour)
 Aariv Goyal
================================================================================

WHAT THIS PROGRAM DOES, IN ONE PARAGRAPH
----------------------------------------
It simulates a trading rule on historical ETF prices. Once a month the rule
looks at recent price trends ("momentum") to decide whether to be invested in
stocks and, if so, which stock fund; it then decides HOW MUCH to invest based on
how volatile that fund has recently been ("volatility targeting"). The program
runs that rule day by day, compares the result to two simple benchmarks, and
then stress-tests it five different ways to find out *when* the rule helps and
whether any apparent edge is real or just luck.

THE FOUR THINGS THIS VERSION FIXES vs. THE FIRST DRAFT
------------------------------------------------------
(1) LOOKAHEAD: a decision made using a day's closing prices is now applied
    starting the NEXT day, so we never "trade on" information we couldn't have
    had yet.
(2) TRANSACTION COSTS: we now charge a cost whenever ANY target weight changes
    (including the monthly re-sizing), not only when we switch which fund we hold.
(3) RISK-FREE RATE: the Sharpe ratio now subtracts a REAL, changing interest
    rate (a T-bill series), instead of pretending interest rates were always 0%.
(4) REPRODUCIBILITY: data is downloaded once and saved to a CSV, so re-running
    the program always gives the exact same numbers.

THE FIVE NEW ANALYSES
---------------------
(A) In-sample vs out-of-sample   (B) Walk-forward overfitting test
(C) Regime / crisis-window analysis   (D) Parameter sensitivity grid + heatmap
(E) Block-bootstrap significance test   (F) Extension model (basket + skip-month)

HOW TO RUN
----------
In Google Colab, paste the cells in order, or run the whole file. The first run
downloads and caches the data; later runs read the cache (no internet needed).
================================================================================
"""

# %% CELL 1 — IMPORTS AND SETTINGS ------------------------------------------
import os                       # to check whether a cached data file exists
import warnings                 # to silence noisy library warnings
import numpy as np              # math on arrays (means, std devs, etc.)
import pandas as pd             # tables of time-series data (the workhorse)
import matplotlib.pyplot as plt # charts
from itertools import product   # to build every combination in the param grid

warnings.filterwarnings("ignore")

# We import yfinance "lazily" (inside a try/except) so the analysis code can be
# tested even on a computer with no internet / no yfinance installed.
try:
    import yfinance as yf
except Exception:
    # auto-install on a fresh Colab so a single paste "just works"
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yfinance"], capture_output=True)
    try:
        import yfinance as yf
    except Exception:
        yf = None

# A fixed random-number generator seed. This makes the bootstrap (Cell 11)
# produce the SAME "random" resamples every run -> reproducible p-values.
RNG = np.random.default_rng(42)

# ---- STRATEGY SETTINGS: change these to experiment ----
TICKERS          = ["VOO", "VXUS", "BND", "SHY"]  # the 4 funds we can hold
RISK_ON_ASSETS   = ["VOO", "VXUS"]   # growth funds we choose between when "risk-on"
RISK_OFF_ASSET   = "BND"             # defensive fund we flee to when "risk-off"
CASH_TICKER      = "SHY"             # cash-like fund that holds leftover weight
ABS_FILTER_ASSET = "VOO"             # the fund whose trend decides risk-on/off

START_DATE       = "2012-01-01"
END_DATE         = None              # None = up to today

LOOKBACK_MOM_DAYS = 252   # how many days of history define "momentum" (~1 year)
SKIP_DAYS         = 0     # 0 = plain momentum; 21 = skip last month ("12-1")
VOL_LOOKBACK_DAYS = 20    # how many days define "recent volatility" (~1 month)
TARGET_ANNUAL_VOL = 0.10  # we aim for the position to have ~10% annual volatility
MAX_WEIGHT        = 1.0   # never invest more than 100% (no borrowing/leverage)
MIN_WEIGHT        = 0.0   # never go below 0% (no short-selling)
TRANSACTION_COST_RATE = 0.001   # 0.1% cost charged on how much we trade (turnover)
INITIAL_CAPITAL   = 10_000

RF_PROXY_TICKER   = "BIL"  # 1-3 month T-bill ETF; its daily return ≈ risk-free rate
TRADING_DAYS      = 252     # trading days per year, used to "annualize" numbers

FIG_DIR           = "figures"  # all charts save into this folder so the README's
os.makedirs(FIG_DIR, exist_ok=True)  # image links (figures/figN_*.png) always resolve


# %% CELL 2 — LOADING DATA (with a reusable on-disk cache) -------------------
def load_prices(tickers, start=START_DATE, end=END_DATE,
                cache_csv="prices_cache.csv", use_cache=True):
    """Get daily adjusted prices for our funds.

    First run: download from Yahoo and SAVE to a CSV file.
    Later runs: just READ that CSV. This guarantees identical results every time
    (the reproducibility fix) and means the analysis can run with no internet.
    """
    # If we already have a cache with all the tickers we need, use it.
    if use_cache and os.path.exists(cache_csv):
        px = pd.read_csv(cache_csv, index_col=0, parse_dates=True)
        if set(tickers).issubset(px.columns):
            return px[tickers].dropna()
    if yf is None:
        raise RuntimeError("yfinance not available and no cache found.")
    # auto_adjust=True -> the 'Close' column is already adjusted for dividends/splits,
    # which is what we want for a fair total-return backtest.
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        px = raw["Close"].copy()
    else:                                  # edge case: a single ticker
        px = raw[["Close"]].copy()
        px.columns = tickers
    px = px[tickers].dropna()              # keep only days where ALL funds have data
    px.to_csv(cache_csv)                   # pin the data for reproducibility
    return px


def load_rf_daily(index, start=START_DATE, end=END_DATE,
                  cache_csv="rf_cache.csv", use_cache=True):
    """Build a daily risk-free interest-rate series from a T-bill ETF (BIL).

    The Sharpe ratio measures return EARNED ABOVE the risk-free rate, so we need
    a real interest-rate series. BIL barely moves in price but its daily return
    tracks short-term rates, so we use its daily return as the risk-free rate.
    If BIL is unavailable, we fall back to 0 (and you can swap in a constant).
    """
    try:
        if use_cache and os.path.exists(cache_csv):
            bil = pd.read_csv(cache_csv, index_col=0, parse_dates=True).iloc[:, 0]
        elif yf is not None:
            raw = yf.download(RF_PROXY_TICKER, start=start, end=end,
                              auto_adjust=True, progress=False)
            bil = raw["Close"] if "Close" in raw else raw.iloc[:, 0]
            bil = bil.pct_change().dropna()
            bil.to_frame("rf").to_csv(cache_csv)
        else:
            raise RuntimeError("no rf source")
        # Line up the rf series with our trading days; fill any gaps with 0.
        return bil.reindex(index).fillna(0.0)
    except Exception:
        return pd.Series(0.0, index=index)  # safe fallback


# %% CELL 3 — SIGNALS (momentum and volatility) -----------------------------
def compute_momentum(prices, lookback, skip=0):
    """Momentum = how much the price has risen over the lookback window.

    Plain version (skip=0): today's price / price `lookback` days ago - 1.
    Skip version (skip=21): ignore the most recent ~month first. Academics do
    this because the very last month often REVERSES, which can add noise.
    """
    base = prices.shift(skip) if skip > 0 else prices
    return base.pct_change(lookback)


def compute_vol(rets, window):
    """Recent volatility = standard deviation of daily returns over `window`
    days, scaled up to an annual number by multiplying by sqrt(252)."""
    return rets.rolling(window).std() * np.sqrt(TRADING_DAYS)


def month_end_dates(prices):
    """The list of month-end trading days. We only make decisions monthly, so
    these are our 'rebalance' dates. ('ME' = month-end in modern pandas.)"""
    me = prices.resample("ME").last().index
    return me.intersection(prices.index)   # keep only real trading days


def trailing_basket_vol(rets, assets, dt, window):
    """For the EXTENSION model: the annualized volatility of an equally-weighted
    mix of several funds over the last `window` days ending at date dt. Used to
    size a basket position the same way we size a single one."""
    sub = rets.loc[:dt, assets].tail(window)
    if len(sub) < 2:
        return np.nan
    ew = sub.mean(axis=1)                   # equal-weight combo's daily return
    return ew.std() * np.sqrt(TRADING_DAYS)


# %% CELL 4 — TURN SIGNALS INTO TARGET WEIGHTS ------------------------------
def build_weights(prices, rets, cfg):
    """The heart of the strategy. For each month-end, decide:
        (1) Are we risk-on or risk-off?  (absolute momentum of VOO)
        (2) If risk-on, which fund?       (relative momentum: pick the strongest)
        (3) How much of it?               (volatility targeting)
    Returns a table: one row per rebalance date, one column per fund, holding the
    target weight (0 to 1). Whatever is left over goes to cash (SHY).
    `cfg` is a dictionary of parameters so we can reuse this for every experiment.
    """
    L         = cfg["lookback"]
    skip      = cfg.get("skip", 0)
    K         = cfg["vol_window"]
    tgt       = cfg["target_vol"]
    mode      = cfg.get("mode", "single")     # "single" fund or "basket"
    risk_on   = cfg.get("risk_on", RISK_ON_ASSETS)
    risk_off  = cfg.get("risk_off", RISK_OFF_ASSET)
    cash      = cfg.get("cash", CASH_TICKER)
    abs_asset = cfg.get("abs_asset", ABS_FILTER_ASSET)
    max_w     = cfg.get("max_w", MAX_WEIGHT)
    min_w     = cfg.get("min_w", MIN_WEIGHT)

    mom_d = compute_momentum(prices, L, skip)   # momentum every day
    vol_d = compute_vol(rets, K)                # volatility every day
    me = month_end_dates(prices)

    # Sample the daily signals only on month-end dates, and drop early dates
    # where we don't yet have a full year of history to compute momentum.
    mom_m = mom_d.loc[me]
    vol_m = vol_d.loc[me]
    valid = mom_m.dropna().index.intersection(vol_m.dropna().index)

    def clip_w(w):                              # keep weight within [min, max]
        return float(np.clip(w, min_w, max_w))

    rows = {}
    for dt in valid:
        w = {t: 0.0 for t in TICKERS}           # start every fund at 0%
        risk_on_ok = mom_m.loc[dt, abs_asset] > 0   # is VOO's 1-year trend positive?

        if risk_on_ok:
            if mode == "single":
                # Pick whichever growth fund has the strongest momentum...
                pick = mom_m.loc[dt, risk_on].idxmax()
                # ...and size it so its expected volatility ≈ our 10% target.
                wr = clip_w(tgt / vol_m.loc[dt, pick]) if vol_m.loc[dt, pick] > 0 else max_w
                w[pick] = wr
            else:  # "basket": hold ALL growth funds with positive momentum, equally
                pos = [a for a in risk_on if mom_m.loc[dt, a] > 0]
                if not pos:                     # if none qualify, act defensive
                    pos = [risk_off]
                bvol = trailing_basket_vol(rets, pos, dt, K)
                wr = clip_w(tgt / bvol) if (bvol and bvol > 0) else max_w
                for a in pos:
                    w[a] += wr / len(pos)        # split the sized weight evenly
        else:
            # Risk-off: VOO's trend is negative, so flee to the defensive fund,
            # also volatility-targeted.
            v = vol_m.loc[dt, risk_off]
            wr = clip_w(tgt / v) if v > 0 else max_w
            w[risk_off] = wr

        # Whatever weight we didn't put into a risky/defensive fund sits in cash.
        risky_sum = sum(w[t] for t in TICKERS if t != cash)
        w[cash] = max(0.0, 1.0 - risky_sum)
        rows[dt] = w
    return pd.DataFrame.from_dict(rows, orient="index")[TICKERS]


# %% CELL 5 — THE BACKTEST (simulate the portfolio day by day) ---------------
def backtest(rets, target_weights, cost_rate=TRANSACTION_COST_RATE,
             initial=INITIAL_CAPITAL):
    """Run the strategy forward in time and track the portfolio's value.

    KEY FIX (#1, no lookahead): we take the monthly target weights, hold them
    between rebalances (ffill), then SHIFT them forward one day (.shift(1)). That
    means today's holdings were decided using yesterday's-or-earlier info — never
    today's, which we couldn't have known when deciding.

    KEY FIX (#2, honest costs): each day we measure 'turnover' = how much the
    weights changed since yesterday, and charge a cost proportional to it. This
    captures the cost of the monthly re-sizing, not just switching funds.
    """
    # Spread monthly decisions across every day, then delay by one day.
    W = target_weights.reindex(rets.index, method="ffill").shift(1)
    W = W.dropna(how="all")
    common = W.index.intersection(rets.index)
    W = W.loc[common]
    R = rets.loc[common, TICKERS]

    gross = (W * R).sum(axis=1)                 # daily return before costs

    turnover = W.diff().abs().sum(axis=1).fillna(0.0)  # how much we traded
    cost = turnover * cost_rate                  # the trading cost
    net = gross - cost                           # daily return after costs

    equity = (1 + net).cumprod() * initial       # compound it into a value curve
    equity.name = "Strategy"
    net.name = "Strategy_Return"
    return equity, net, W


def benchmark_returns(rets, kind="VOO"):
    """Daily returns for our two comparison portfolios:
       'VOO'  = 100% stocks (buy and hold).
       '6040' = 60% stocks / 40% bonds, rebalanced monthly (a classic baseline).
    """
    if kind == "VOO":
        return rets["VOO"].copy()
    if kind == "6040":
        return 0.6 * rets["VOO"] + 0.4 * rets["BND"]
    raise ValueError(kind)


# %% CELL 6 — PERFORMANCE METRICS -------------------------------------------
def cagr(equity):
    """Compound Annual Growth Rate: the steady yearly return that would turn the
    starting value into the ending value over the elapsed time."""
    yrs = (equity.index[-1] - equity.index[0]).days / 365.25
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan

def ann_vol(r):
    """Annualized volatility = yearly standard deviation of returns (risk)."""
    return r.std() * np.sqrt(TRADING_DAYS)

def sharpe(r, rf_daily):
    """Sharpe ratio = average return ABOVE the risk-free rate, per unit of risk.
    Higher is better. We subtract the real daily risk-free rate here (fix #3)."""
    ex = r - rf_daily.reindex(r.index).fillna(0.0)
    s = ex.std()
    return (ex.mean() / s) * np.sqrt(TRADING_DAYS) if s > 0 else np.nan

def sortino(r, rf_daily):
    """Like Sharpe, but only counts DOWNSIDE volatility as risk (ignores upside
    swings, which investors don't mind)."""
    ex = r - rf_daily.reindex(r.index).fillna(0.0)
    downside = ex[ex < 0].std()
    return (ex.mean() / downside) * np.sqrt(TRADING_DAYS) if downside and downside > 0 else np.nan

def max_drawdown(equity):
    """The worst peak-to-trough drop the portfolio ever suffered (a key measure
    of how painful the strategy was to hold). Returns the number AND the full
    drawdown curve for plotting."""
    peak = equity.cummax()
    dd = equity / peak - 1
    return dd.min(), dd

def equity_from_returns(r, initial=INITIAL_CAPITAL):
    """Turn a daily-return series into a value curve starting at `initial`."""
    return (1 + r.fillna(0)).cumprod() * initial

def summarize(name, r, rf_daily, initial=INITIAL_CAPITAL):
    """Bundle all the metrics above into one labeled row."""
    eq = equity_from_returns(r, initial)
    mdd, _ = max_drawdown(eq)
    return {"Strategy": name, "CAGR": cagr(eq), "Ann.Vol": ann_vol(r),
            "Sharpe": sharpe(r, rf_daily), "Sortino": sortino(r, rf_daily),
            "MaxDD": mdd, "Calmar": (cagr(eq) / abs(mdd)) if mdd and mdd < 0 else np.nan,
            "Final($)": eq.iloc[-1]}

def summary_table(strat_r, rets, rf_daily):
    """A 3-row table comparing the strategy to both benchmarks."""
    voo = benchmark_returns(rets, "VOO").reindex(strat_r.index)
    s6040 = benchmark_returns(rets, "6040").reindex(strat_r.index)
    return pd.DataFrame([
        summarize("Risk-Targeted Dual Momentum", strat_r, rf_daily),
        summarize("Buy & Hold VOO", voo, rf_daily),
        summarize("60/40 (monthly)", s6040, rf_daily)])


# %% CELL 7 — (A) IN-SAMPLE vs OUT-OF-SAMPLE --------------------------------
def in_out_sample(strat_r, rets, rf_daily, split="2020-01-01"):
    """Split the history in two and report metrics separately. If the strategy
    behaves similarly before and after the split, it isn't a one-era fluke."""
    out = {}
    for label, mask in [("In-Sample", strat_r.index < split),
                        ("Out-of-Sample", strat_r.index >= split)]:
        sr = strat_r[mask]
        if len(sr) < 30:
            continue
        out[label] = summary_table(sr, rets, rf_daily)
    return out


# %% CELL 8 — (B) WALK-FORWARD: the overfitting test -------------------------
def run_config_returns(prices, rets, cfg, cost_rate=TRANSACTION_COST_RATE):
    """Helper: build weights for one parameter set and return its daily returns."""
    W = build_weights(prices, rets, cfg)
    if W.empty:
        return pd.Series(dtype=float)
    _, net, _ = backtest(rets, W, cost_rate)
    return net

def walk_forward(prices, rets, rf_daily, grid,
                 train_years=4, test_years=1, cost_rate=TRANSACTION_COST_RATE):
    """Honest test of whether 'tuning' the parameters actually helps.

    Roll through time. At each step: look back `train_years`, pick the parameter
    set that WOULD have had the best Sharpe over that past window, then apply ONLY
    that choice to the NEXT `test_years` (which the choice has not 'seen'). Glue
    those untouched test pieces together. If this stitched track is worse than a
    fixed sensible setting, it means optimizing the parameters was overfitting.
    """
    start, end = rets.index.min(), rets.index.max()
    pieces, chosen_log = [], []
    t0 = start + pd.DateOffset(years=train_years)
    while t0 < end:
        train_lo = t0 - pd.DateOffset(years=train_years)
        test_hi = min(t0 + pd.DateOffset(years=test_years), end)
        best, best_sharpe = None, -np.inf
        for cfg in grid:                         # try every parameter set
            r = run_config_returns(prices, rets, cfg, cost_rate)
            r_tr = r[(r.index >= train_lo) & (r.index < t0)]   # past window only
            if len(r_tr) < 60:
                continue
            sh = sharpe(r_tr, rf_daily)
            if np.isfinite(sh) and sh > best_sharpe:
                best, best_sharpe = cfg, sh       # remember the best past performer
        if best is not None:
            r = run_config_returns(prices, rets, best, cost_rate)
            r_te = r[(r.index >= t0) & (r.index < test_hi)]    # apply to the future
            pieces.append(r_te)
            chosen_log.append({"test_start": t0, **best, "train_sharpe": best_sharpe})
        t0 = t0 + pd.DateOffset(years=test_years)
    wf = pd.concat(pieces).sort_index() if pieces else pd.Series(dtype=float)
    wf.name = "WalkForward"
    return wf, pd.DataFrame(chosen_log)


# %% CELL 9 — (C) REGIME / CONDITIONAL ANALYSIS -----------------------------
# Three well-known stock-market stress periods to examine.
CRISIS_WINDOWS = {
    "2018 Q4 Selloff": ("2018-10-01", "2018-12-24"),
    "COVID Crash":     ("2020-02-19", "2020-03-23"),
    "2022 Bear":       ("2022-01-03", "2022-10-12"),
}

def window_perf(strat_r, rets, windows=CRISIS_WINDOWS):
    """For each crisis window, compute the total RETURN and the worst DRAWDOWN of
    the strategy and both benchmarks. This is where a defensive strategy should
    prove its worth (or not).

    NOTE - 'Ret' and 'MaxDD' are two DIFFERENT numbers and must not be conflated
    when writing up results:
      * 'Ret'   = total compounded return from the window's start to its end
                  (how the period ended).
      * 'MaxDD' = the deepest peak-to-trough loss *within* the window (the worst
                  point along the way). MaxDD is always at least as negative as Ret.
    Example (COVID window, v3 crash guard): return = -1.9%, max drawdown = -4.0%.
    """
    voo = benchmark_returns(rets, "VOO")
    s6040 = benchmark_returns(rets, "6040")
    rows = []
    for name, (lo, hi) in windows.items():
        sl = slice(lo, hi)
        def tot(r):                              # total compounded return in window
            rr = r.loc[sl].fillna(0)
            return (1 + rr).prod() - 1 if len(rr) else np.nan
        def mdd(r):                              # worst drawdown in window
            rr = r.loc[sl].fillna(0)
            return max_drawdown(equity_from_returns(rr))[0] if len(rr) else np.nan
        rows.append({"Window": name,
                     "Strat Ret": tot(strat_r), "Strat MaxDD": mdd(strat_r),
                     "VOO Ret": tot(voo), "VOO MaxDD": mdd(voo),
                     "60/40 Ret": tot(s6040), "60/40 MaxDD": mdd(s6040)})
    return pd.DataFrame(rows)

def up_down_conditional(strat_r, rets):
    """Compare average monthly performance in months when stocks rose vs. fell.
    Ideal pattern: give up a little in up-months, lose much less in down-months."""
    voo = benchmark_returns(rets, "VOO").reindex(strat_r.index).fillna(0)
    m_strat = (1 + strat_r).resample("ME").prod() - 1   # strategy monthly returns
    m_voo = (1 + voo).resample("ME").prod() - 1          # VOO monthly returns
    up = m_voo > 0
    return pd.DataFrame({
        "VOO Up Months":   [m_strat[up].mean(),  m_voo[up].mean(),  up.sum()],
        "VOO Down Months": [m_strat[~up].mean(), m_voo[~up].mean(), (~up).sum()],
    }, index=["Strategy avg", "VOO avg", "n months"])


# %% CELL 10 — (D) PARAMETER SENSITIVITY ------------------------------------
def sensitivity(prices, rets, rf_daily,
                lookbacks=(63, 126, 252, 504), vol_windows=(20, 60, 90),
                target_vols=(0.05, 0.10, 0.15), cost_rate=TRANSACTION_COST_RATE):
    """Re-run the strategy for EVERY combination of the three main parameters and
    record CAGR/Sharpe/MaxDD. Tells us whether good results depend on one lucky
    setting (bad — overfit) or hold across many settings (good — robust)."""
    recs = []
    for L, K, tv in product(lookbacks, vol_windows, target_vols):
        r = run_config_returns(prices, rets,
                               {"lookback": L, "vol_window": K, "target_vol": tv}, cost_rate)
        if r.empty:
            continue
        eq = equity_from_returns(r)
        recs.append({"lookback": L, "vol_window": K, "target_vol": tv,
                     "CAGR": cagr(eq), "Sharpe": sharpe(r, rf_daily),
                     "MaxDD": max_drawdown(eq)[0]})
    return pd.DataFrame(recs)

def sensitivity_heatmap(sens_df, value="Sharpe", vol_window=20, fname=None):
    """Draw a colored grid of, say, Sharpe across lookback (rows) and target
    volatility (columns). Smooth color gradients = robust; lone bright cell = fragile."""
    sub = sens_df[sens_df["vol_window"] == vol_window]
    piv = sub.pivot(index="lookback", columns="target_vol", values=value)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(piv.columns)), [f"{c:.0%}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), piv.index)
    ax.set_xlabel("Target annual volatility"); ax.set_ylabel("Momentum lookback (days)")
    ax.set_title(f"{value} sensitivity (vol window = {vol_window}d)")
    for i in range(piv.shape[0]):                 # write the number in each cell
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax); fig.tight_layout()
    if fname: fig.savefig(fname, dpi=150)
    return fig


# %% CELL 11 — (E) STATISTICAL SIGNIFICANCE (block bootstrap) ----------------
def block_bootstrap_diff(strat_r, bench_r, rf_daily, metric="Sharpe",
                         n_boot=2000, block=21, seed=42):
    """Is the strategy-vs-benchmark difference REAL, or could it be luck?

    We can't re-run history, so we 'resample' it: repeatedly stitch together
    random CHUNKS (blocks of 21 days, to keep day-to-day patterns intact) of the
    paired return series, recompute the metric difference each time, and build a
    distribution of possible outcomes. If that distribution's 95% range includes
    zero, the difference is NOT statistically distinguishable from luck.
    Returns the real difference, the 95% confidence interval, and a p-value.
    """
    rng = np.random.default_rng(seed)
    a = strat_r.fillna(0).values
    b = bench_r.reindex(strat_r.index).fillna(0).values
    rf = rf_daily.reindex(strat_r.index).fillna(0).values
    n = len(a)
    n_blocks = int(np.ceil(n / block))

    def _cagr_r(r):   # CAGR computed from a plain return array (no dates needed)
        return (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1 if len(r) else np.nan

    def metric_diff(idx):                         # metric(strategy) - metric(benchmark)
        sa, sb, sr = pd.Series(a[idx]), pd.Series(b[idx]), pd.Series(rf[idx])
        if metric == "Sharpe":
            return sharpe(sa, sr) - sharpe(sb, sr)
        if metric == "CAGR":
            return _cagr_r(sa) - _cagr_r(sb)
        if metric == "MaxDD":
            return max_drawdown(equity_from_returns(sa))[0] - max_drawdown(equity_from_returns(sb))[0]
        raise ValueError(metric)

    point = metric_diff(np.arange(n))             # the actual, observed difference
    diffs = np.empty(n_boot)
    for i in range(n_boot):                       # build the resampled distribution
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) % n for s in starts])[:n]
        diffs[i] = metric_diff(idx)
    lo, hi = np.percentile(diffs, [2.5, 97.5])    # 95% confidence interval
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())  # two-sided p-value
    return {"metric": metric, "point_diff": point, "ci_low": lo, "ci_high": hi,
            "p_value": min(p, 1.0)}


# %% CELL 12 — (F) EXTENSION MODEL ------------------------------------------
def extension_config():
    """The 'improved' version: hold a diversified BASKET of growth funds and use
    the academic skip-month (12-1) momentum. We test whether these textbook
    refinements actually help (in this study, they did not — an honest result)."""
    return {"lookback": LOOKBACK_MOM_DAYS, "skip": 21, "vol_window": VOL_LOOKBACK_DAYS,
            "target_vol": TARGET_ANNUAL_VOL, "mode": "basket"}


# %% CELL 13 — DRIVER: run everything ---------------------------------------
def run_all(prices, rets, rf_daily, make_plots=True):
    """Run the baseline plus all five analyses and print the results."""
    base_cfg = {"lookback": LOOKBACK_MOM_DAYS, "skip": SKIP_DAYS,
                "vol_window": VOL_LOOKBACK_DAYS, "target_vol": TARGET_ANNUAL_VOL,
                "mode": "single"}
    W = build_weights(prices, rets, base_cfg)
    equity, strat_r, _ = backtest(rets, W)

    print("=== BASELINE (corrected costs, real rf, no lookahead) ===")
    print(summary_table(strat_r, rets, rf_daily).to_string(index=False))

    print("\n=== IN / OUT OF SAMPLE ===")
    for k, v in in_out_sample(strat_r, rets, rf_daily).items():
        print(f"\n[{k}]"); print(v.to_string(index=False))

    print("\n=== REGIME / CRISIS WINDOWS ==="); print(window_perf(strat_r, rets).to_string(index=False))
    print("\n=== UP vs DOWN MARKET ==="); print(up_down_conditional(strat_r, rets).to_string())

    print("\n=== SENSITIVITY (top by Sharpe) ===")
    sens = sensitivity(prices, rets, rf_daily)
    print(sens.sort_values("Sharpe", ascending=False).head(8).to_string(index=False))

    print("\n=== BOOTSTRAP SIGNIFICANCE vs 60/40 ===")
    s6040 = benchmark_returns(rets, "6040")
    for m in ["Sharpe", "CAGR", "MaxDD"]:
        print(block_bootstrap_diff(strat_r, s6040, rf_daily, metric=m, n_boot=2000))

    print("\n=== EXTENSION (basket + skip-month) ===")
    _, ext_r, _ = backtest(rets, build_weights(prices, rets, extension_config()))
    print(summary_table(ext_r, rets, rf_daily).to_string(index=False))

    if make_plots:   # save the two figures the paper references most
        bv = equity_from_returns(benchmark_returns(rets, "VOO").reindex(strat_r.index))
        b6 = equity_from_returns(benchmark_returns(rets, "6040").reindex(strat_r.index))
        plt.figure(figsize=(11, 6))
        plt.plot(equity.index, equity, label="Strategy")
        plt.plot(bv.index, bv, label="Buy&Hold VOO"); plt.plot(b6.index, b6, label="60/40")
        plt.yscale("log"); plt.legend(); plt.title("Equity Curve Comparison")
        plt.savefig(FIG_DIR + "/fig1_equity.png", dpi=150); plt.close()
        sensitivity_heatmap(sens, "Sharpe", 20, fname=FIG_DIR + "/fig4_sensitivity.png"); plt.close()
    return strat_r, sens


# This block runs only when you execute the file directly (not on import).


# ============================================================================
# VERSION 3 ADDITIONS: fast crash-guard overlay + conditional significance
# ============================================================================
# ---- V3 SETTINGS ----
FAST_DAYS   = 20          # the "faster than monthly" trend window (~1 month)
FAST_GAUGE  = "VOO"       # we watch U.S. equities for the crash signal


# %% --- The fast crash-guard overlay ---------------------------------------
def fast_gate(prices, fast_days=FAST_DAYS, gauge=FAST_GAUGE):
    """Daily on/off switch for equity risk.
    gate = 1 (stay invested) if the gauge's fast `fast_days`-day trend is
    positive; gate = 0 (step out) if it has turned negative. Computed from past
    prices only; it is applied with a one-day delay in the backtest so it never
    'sees' the day it trades on."""
    mom = prices[gauge].pct_change(fast_days)
    return (mom > 0).astype(float)


def backtest_v3(prices, rets, target_weights, fast_days=FAST_DAYS,
                cost_rate=TRANSACTION_COST_RATE, gauge=FAST_GAUGE):
    """Same daily simulation as v2, but each day the equity weights (VOO/VXUS)
    are multiplied by the fast gate; any weight the guard removes is parked in
    cash (SHY). Defensive bonds and cash are never gated."""
    # Monthly target weights -> daily, held between rebalances, delayed one day
    W = target_weights.reindex(rets.index, method="ffill").shift(1).dropna(how="all")

    # Daily crash-guard gate, also delayed one day (no lookahead)
    g = fast_gate(prices, fast_days, gauge).reindex(W.index).shift(1).fillna(1.0)

    W = W.copy()
    equity_cols = [c for c in RISK_ON_ASSETS if c in W.columns]   # VOO, VXUS
    removed = (W[equity_cols].mul(1 - g, axis=0)).sum(axis=1)        # weight cut
    W[equity_cols] = W[equity_cols].mul(g, axis=0)                   # apply gate
    W[CASH_TICKER] = W[CASH_TICKER] + removed                    # park in cash

    common = W.index.intersection(rets.index)
    W, R = W.loc[common], rets.loc[common, TICKERS]
    gross = (W * R).sum(axis=1)
    turnover = W.diff().abs().sum(axis=1).fillna(0.0)                # extra trading
    net = gross - turnover * cost_rate
    equity = (1 + net).cumprod() * INITIAL_CAPITAL
    equity.name, net.name = "Strategy_v3", "Strategy_v3_Return"
    return equity, net, W


# %% --- Conditional (stress-regime) significance test ----------------------
def down_month_significance(strat_r, rets, bench_kind="VOO",
                            n_boot=5000, seed=42):
    """Test whether, in months when U.S. equities FELL, the strategy lost
    statistically less than a benchmark.

    Steps: turn daily returns into monthly returns; keep only the months where
    VOO was negative; bootstrap (resample those months with replacement) the
    mean monthly difference (strategy - benchmark); report the average
    difference, a 95% confidence interval, and a two-sided p-value for 'no
    difference'. A positive, significant number means real downside protection.
    """
    rng = np.random.default_rng(seed)
    voo = benchmark_returns(rets, "VOO").reindex(strat_r.index)
    bench = benchmark_returns(rets, bench_kind).reindex(strat_r.index)

    m_strat = (1 + strat_r).resample("ME").prod() - 1
    m_voo   = (1 + voo).resample("ME").prod() - 1
    m_bench = (1 + bench).resample("ME").prod() - 1

    down = m_voo < 0                              # the stress months
    diff = (m_strat[down] - m_bench[down]).dropna().values
    n = len(diff)
    point = diff.mean()
    boot = np.array([rng.choice(diff, size=n, replace=True).mean()
                     for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"benchmark": bench_kind, "n_down_months": n,
            "mean_monthly_diff": point, "ci_low": lo, "ci_high": hi,
            "p_value": min(p, 1.0)}


# %% --- Driver: run v2 baseline, v3, and the conditional test --------------
def run_v3(prices, rets, rf):
    base_cfg = {"lookback": 252, "skip": 0, "vol_window": 20,
                "target_vol": 0.10, "mode": "single"}
    Wm = build_weights(prices, rets, base_cfg)

    # v2 baseline (no guard) and v3 (with guard)
    eq2, r2, _ = backtest(rets, Wm)
    eq3, r3, _ = backtest_v3(prices, rets, Wm)

    print("=== v2 baseline vs v3 (full sample) ===")
    s2 = summary_table(r2, rets, rf); s2.loc[0, "Strategy"] = "v2 Baseline (no guard)"
    s3 = summarize("v3 (+ fast crash guard)", r3, rf)
    tbl = pd.concat([s2, pd.DataFrame([s3])], ignore_index=True)
    print(tbl.to_string(index=False))

    print("\n=== CRISIS WINDOWS: v2 vs v3 ===")
    w2 = window_perf(r2, rets)[["Window", "Strat Ret", "Strat MaxDD"]].rename(
        columns={"Strat Ret": "v2 Ret", "Strat MaxDD": "v2 MaxDD"})
    w3 = window_perf(r3, rets)[["Strat Ret", "Strat MaxDD"]].rename(
        columns={"Strat Ret": "v3 Ret", "Strat MaxDD": "v3 MaxDD"})
    print(pd.concat([w2, w3], axis=1).to_string(index=False))

    print("\n=== CONDITIONAL (down-month) SIGNIFICANCE ===")
    for r, tag in [(r2, "v2"), (r3, "v3")]:
        for bk in ["VOO", "6040"]:
            res = down_month_significance(r, rets, bk)
            print(tag, res)

    print("\n=== FAST_DAYS robustness (v3 Sharpe) ===")
    for fd in [10, 20, 40, 60]:
        _, rr, _ = backtest_v3(prices, rets, Wm, fast_days=fd)
        print(f"  fast_days={fd:>3}:  Sharpe={sharpe(rr, rf):.3f}  "
              f"CAGR={cagr(equity_from_returns(rr)):.4f}  "
              f"MaxDD={max_drawdown(equity_from_returns(rr))[0]:.4f}")

    return eq2, r2, eq3, r3


# ============================================================================
#  FIGURES + MAIN DRIVER (added for the all-in-one script)
# ============================================================================
def make_all_figures(prices, rets, rf):
    """Generate, save, and display all seven figures."""
    base_cfg = {"lookback":252,"skip":0,"vol_window":20,"target_vol":0.10,"mode":"single"}
    W = build_weights(prices, rets, base_cfg)
    equity, strat_r, Wdaily = backtest(rets, W)
    eq3, r3, _ = backtest_v3(prices, rets, W)
    voo = benchmark_returns(rets,"VOO").reindex(strat_r.index)
    s60 = benchmark_returns(rets,"6040").reindex(strat_r.index)
    evoo = equity_from_returns(voo); e60 = equity_from_returns(s60)

    plt.figure(figsize=(11,6))
    plt.plot(equity.index, equity, label="Strategy (v2)")
    plt.plot(evoo.index, evoo, label="Buy & Hold VOO")
    plt.plot(e60.index, e60, label="60/40")
    plt.yscale("log"); plt.legend(); plt.grid(alpha=.3)
    plt.title("Figure 1 - Equity Curve Comparison (log)"); plt.ylabel("Value ($, log)")
    plt.tight_layout(); plt.savefig(FIG_DIR + "/fig1_equity.png", dpi=150); plt.show()

    _, dd_s = max_drawdown(equity); _, dd_v = max_drawdown(evoo); _, dd_6 = max_drawdown(e60)
    plt.figure(figsize=(11,5))
    plt.plot(dd_s.index, dd_s*100, label="Strategy"); plt.plot(dd_v.index, dd_v*100, label="VOO"); plt.plot(dd_6.index, dd_6*100, label="60/40")
    plt.title("Figure 2 - Drawdowns"); plt.ylabel("Drawdown (%)"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(FIG_DIR + "/fig2_drawdown.png", dpi=150); plt.show()

    code = {a:i for i,a in enumerate(TICKERS)}
    held = Wdaily.idxmax(axis=1).map(code)
    plt.figure(figsize=(11,3.2)); plt.plot(held.index, held.values, lw=.8)
    plt.yticks(list(code.values()), list(code.keys())); plt.title("Figure 3 - Primary Asset Held")
    plt.tight_layout(); plt.savefig(FIG_DIR + "/fig3_asset_timeline.png", dpi=150); plt.show()

    sens = sensitivity(prices, rets, rf)
    sensitivity_heatmap(sens, "Sharpe", 20, fname=FIG_DIR + "/fig4_sensitivity.png"); plt.show()

    plt.figure(figsize=(11,6))
    plt.plot(equity.index, equity, label="v2 (no guard)")
    plt.plot(eq3.index, eq3, label="v3 (+ crash guard)")
    plt.plot(evoo.index, evoo, label="VOO", alpha=.8); plt.plot(e60.index, e60, label="60/40", alpha=.8)
    plt.yscale("log"); plt.legend(); plt.grid(alpha=.3)
    plt.title("Figure 5 - v2 vs v3 vs Benchmarks (log)"); plt.ylabel("Value ($, log)")
    plt.tight_layout(); plt.savefig(FIG_DIR + "/fig5_v3_equity.png", dpi=150); plt.show()

    sl = slice("2020-01-01","2020-06-30")
    def grow(r): rr=r.loc[sl].fillna(0); return (1+rr).cumprod()
    plt.figure(figsize=(10,5.5))
    plt.plot(grow(strat_r), label="v2 (no guard)")
    plt.plot(grow(r3), label="v3 (+ crash guard)", lw=2)
    plt.plot(grow(voo), label="VOO", alpha=.85)
    plt.axhline(1.0, color="gray", lw=.7, ls="--"); plt.legend(); plt.grid(alpha=.3)
    plt.ylabel("Growth of $1"); plt.title("Figure 6 - COVID Crash Zoom (Jan-Jun 2020)")
    plt.tight_layout(); plt.savefig(FIG_DIR + "/fig6_covid_zoom.png", dpi=150); plt.show()

    w2 = window_perf(strat_r, rets).set_index("Window")["Strat MaxDD"]*100
    w3 = window_perf(r3, rets).set_index("Window")["Strat MaxDD"]*100
    xw = np.arange(len(w2)); wbar=0.35
    plt.figure(figsize=(9,5))
    plt.bar(xw-wbar/2, w2.values, wbar, label="v2 (no guard)")
    plt.bar(xw+wbar/2, w3.values, wbar, label="v3 (+ crash guard)")
    plt.xticks(xw, w2.index); plt.ylabel("Max drawdown in window (%)")
    plt.title("Figure 7 - Crisis-Window Drawdowns: v2 vs v3"); plt.legend(); plt.grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(FIG_DIR + "/fig7_crisis_bars.png", dpi=150); plt.show()


def main():
    px = load_prices(TICKERS)
    rets = px.pct_change().dropna()
    rf = load_rf_daily(rets.index)

    print("#"*70 + "\n# VERSION 2 ANALYSIS\n" + "#"*70)
    run_all(px, rets, rf, make_plots=False)

    print("\n" + "#"*70 + "\n# WALK-FORWARD OVERFITTING TEST  (this step takes ~1-2 minutes)\n" + "#"*70)
    grid = [{"lookback":L,"vol_window":K,"target_vol":tv}
            for L in (63,126,252,504) for K in (20,60) for tv in (0.05,0.10,0.15)]
    wf, log = walk_forward(px, rets, rf, grid, train_years=4, test_years=1)
    if not wf.empty:
        wfe = equity_from_returns(wf)
        print(f"Walk-forward  CAGR={cagr(wfe):.4f}  Sharpe={sharpe(wf,rf):.3f}  MaxDD={max_drawdown(wfe)[0]:.4f}")

    print("\n" + "#"*70 + "\n# VERSION 3 ANALYSIS (the engineered fix)\n" + "#"*70)
    run_v3(px, rets, rf)

    print("\nGenerating all figures...")
    make_all_figures(px, rets, rf)
    print("\nDONE. Figures saved as fig1_equity.png ... fig7_crisis_bars.png")


if __name__ == "__main__":
    main()
