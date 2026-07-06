# Project Summary — the 60-second version

**What I built.** A Python framework that tests a rules-based investing strategy against 13 years of market data (2013–2026). Once a month, the rules look at recent price trends to decide whether to hold stocks and which stock fund to hold, then size the position based on how turbulent the market has recently been. The framework avoids the mistakes that make most amateur backtests untrustworthy: it never lets the strategy "peek" at future prices, it charges realistic trading costs, and it measures performance against real interest rates.

**What I asked.** Not "does it beat the market?" — a defensive strategy is *expected* to trail stocks during a historic bull run, so that question answers itself. The real question: **under what conditions does risk management add value, and can that value be distinguished from luck?**

**What I found.**

1. Over the full period, the strategy reduced risk versus holding stocks — but a simple 60/40 stock/bond portfolio beat it on every risk-adjusted measure, across every one of 36 parameter settings tested. I report that plainly.
2. Its value is **conditional on the speed of a downturn**: in the slow 2022 bear market it beat both benchmarks decisively, but in the one-month COVID crash it took its worst loss of the sample, because a monthly signal can't react in days.
3. I engineered a fix — a daily "crash guard" — that cut the COVID loss from −28.5% to −1.9%, with statistically significant downside protection. But the same speed that protects in crashes bleeds money in calm markets. **Crash protection and whipsaw are two ends of one dial** — the project's deepest finding.
4. Trying to *optimize* the parameters each year made results worse, not better (Sharpe 0.33 vs 0.58) — an empirical demonstration of the "backtest overfitting" trap the academic literature warns about.

**Why it's trustworthy.** Every number regenerates from cached data and a fixed random seed — anyone can re-run the single script and reproduce the paper exactly. The results include out-of-sample validation, a walk-forward test, regime-conditional analysis, and block-bootstrap significance tests (2,000 resamples).

**Why it matters to me.** I started investing at eleven and lost money chasing hype. This project is the opposite habit: turn an instinct into explicit rules, stress-test them, find exactly where they fail, fix the failure, and report the truth even when it doesn't flatter the strategy. That's the discipline I want to bring to quantitative finance.

**Where to look next:** the full paper (`paper/Quant_Momentum_Strategy.pdf`), the interactive demo (`index.html` / GitHub Pages), and `REPRODUCIBILITY.md` to run it yourself.
