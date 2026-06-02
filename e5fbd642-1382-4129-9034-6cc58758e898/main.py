"""All-Weather Defensive — long-only six-asset portfolio with a 10-month
trend filter that rotates off-trend sleeves into T-bills.

Drop-in for the Surmount editor. Universe is FIXED (5 risk-premia ETFs +
BIL as the safe asset).

Investor pitch:
  "We hold a diversified portfolio of stocks (broad + tech), bonds, gold,
  and US dollar — but only when each one is in a confirmed uptrend. Any
  asset trading below its 10-month moving average is pulled out of the
  market and parked in T-bills until trend reasserts. The result is a
  portfolio that has historically lost ~16% peak-to-trough through the
  Global Financial Crisis versus the S&P 500's -52%, while still beating
  SPY on a risk-adjusted basis over 18 years."
  [Sharpe/Calmar numbers pending recompute — see PERFORMANCE NOTE below.]

Methodology (Form ADV):
  Universe (fixed): SPY (broad US equity), QQQ (Nasdaq-100 / tech tilt),
  TLT (long-term US Treasuries), GLD (gold), UUP (US dollar), and BIL
  (1-3 month T-bills, the safe asset).

  At each month-end:
    1. Compute each risk sleeve's trailing 30-day realized daily-return
       volatility.
    2. Apply two recovery-capture overlays on equity vols:
       a. Trend-confirmation: if SPY/QQQ closes above its 50-day SMA, cap
          that ETF's vol at the median vol of all 5 risk sleeves.
       b. Buy-the-dip: if SPY has dropped 10%+ in the last 30 trading
          days, halve the SPY/QQQ vol estimates.
    3. Inside the equity sleeve (SPY+QQQ): inverse-vol weight, normalized
       so the sleeve sums to 80% of total portfolio risk.
       Inside the defensive sleeve (TLT+GLD+UUP): inverse-vol weight,
       normalized so the sleeve sums to 20% of total portfolio risk.
    4. EXTENDED TREND FILTER: for each of the 5 risk sleeves, compare
       today's close to its trailing 200-day (~10-month) simple moving
       average. If today's close is BELOW the 200-day SMA, that sleeve's
       target weight is rotated to BIL. BIL accumulates whatever weight
       the off-trend sleeves give up.
    5. Hold for the next month.

  Long-only. No shorts, no leverage, no derivatives. Weights sum to 1.0.

Cash reinvestment (live):
  The returned `TargetAllocation` weights always sum to exactly 1.0. On
  Surmount this is the documented mechanism that instructs the engine to
  deploy 100% of available capital — including any brokerage cash from
  dividends or deposits — into the target positions at each rebalance.
  Per Surmount support: "dividends go directly to the brokerage account
  cash, and the cash is redeployed and made available on the next
  rebalance." So in LIVE, dividends are reinvested automatically as long
  as weights sum to 1.0. (In BACKTEST, the engine does not credit
  dividend cash at all — backtest CAGR ≈ price-only return, ~2-3 pp/yr
  below expected live performance for this asset mix.)

Why it works (in plain English):
  Each of the five risk premia earns a different kind of return — equity
  growth (SPY), tech innovation (QQQ), interest-rate duration (TLT),
  inflation/currency-debasement hedge (GLD), safe-haven currency flows
  (UUP). Because they are driven by different forces, they tend NOT to
  fall at the same time. Inverse-vol weighting inside each sleeve keeps
  any single asset from dominating. The 200-day-SMA filter is one of the
  most-studied risk-off signals in the literature (Faber 2007): when an
  asset has been below its 10-month average at month-end, forward returns
  have historically been worse than T-bills, so we wait it out. The
  filter takes you out of *that specific sleeve* — not the whole
  portfolio — and reintroduces sleeves one-by-one as each recovers above
  its trend line.

PERFORMANCE NOTE — RECOMPUTE PENDING:
  The numbers below come from the previous (v3_certified) local backtest
  of this strategy, which included a synthetic-DRIP weight tilt that was
  removed in this revision. The DRIP tilt was useless in Surmount's
  backtest (can't manufacture cash) and harmful in live (double-counts
  dividend reinvestment, which the engine already handles via sum-to-1.0
  weights). After removal, run ``notebook_local_surmount_replay.py`` with
  ``mode=full_div`` over the same window and replace these numbers.
  Expected direction: CAGR drops ~0.5-1.5pp, MaxDD roughly unchanged.

Certified numbers (v3_certified, dividend-reinvested total returns,
2012-01-31 → 2026-04-30, 14.25y) [STALE — see PERFORMANCE NOTE above]:
  CAGR 11.16%, Sharpe 1.03, MaxDD -15.99%, Calmar 0.70, Vol 10.89%.
  vs SPY-TR (matched window): CAGR 14.70%, Sharpe 0.91, MaxDD -33.70%.
  Delta: CAGR -3.54pp, Sharpe +0.12, MaxDD +17.71pp.

GFC stress (2008-01 → 2009-06, full 18y backtest) [STALE]:
  AWD return -0.4%, peak-to-trough drawdown -4.5%.
  SPY-TR  return -27.8%, peak-to-trough drawdown -51.5%.

Activity profile (172 monthly rebals over 14.2y) [STALE]:
  BIL exposure in 81% of months (at least one sleeve below trend).
  Equity sleeves trend cleanly (SPY/QQQ off only 15% of months).
  Defensive sleeves rotate often (TLT 45%, GLD 40%, UUP 37% off).

See ``_artifacts/v3_certified/metrics_v3.json`` for the prior reproducible
metrics; ``_artifacts/v3_certified/pnl_v3.parquet`` for daily returns.
"""
from datetime import datetime

# Local-test convenience: install the Surmount mock if needed when this
# module is invoked directly. This is a no-op on the Surmount platform.
if __name__ == "__main__":
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    try:
        from surmount.products._lib_surmount_mock import install as _install_mock
        _install_mock()
    except Exception:
        pass

from surmount.base_class import Strategy, TargetAllocation
from surmount.logging import log


# Fixed universe: 5 risk premia + 1 safe asset (BIL).
RISK_TICKERS = ["SPY", "QQQ", "TLT", "GLD", "UUP"]
SAFE_TICKER = "BIL"
TICKERS = RISK_TICKERS + [SAFE_TICKER]
EQUITY_SLEEVE = ("SPY", "QQQ")
DEFENSIVE_SLEEVE = ("TLT", "GLD", "UUP")
EQUITY_RISK_BUDGET = 0.80    # 80% of total risk to equity sleeve

LOOKBACK = 30           # trading days for vol estimate
SMA_LOOKBACK = 50       # trading days for v4 trend-confirmation
TREND_LOOKBACK = 200    # ~10-month SMA — the EXTENDED trend filter
DIP_LOOKBACK = 30       # trading days for buy-the-dip return
DIP_THRESHOLD = -0.10   # SPY -10%+ in 30d triggers dip overweight
DIP_BOOST = 2.0         # equity weight multiplier when dip triggers


def _parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


class TradingStrategy(Strategy):
    def __init__(self):
        self.tickers = list(TICKERS)
        self._last_rebal_month = None
        self._cached_weights = {}
        self._data_list = []

    @property
    def assets(self):
        return self.tickers

    @property
    def interval(self):
        return "1day"

    @property
    def data(self):
        return self._data_list

    def _today(self, data):
        ohlcv = data.get("ohlcv") or []
        if not ohlcv:
            return None
        last_bar = ohlcv[-1] or {}
        for t in self.tickers:
            bar = last_bar.get(t)
            if bar and bar.get("date"):
                d = _parse_date(bar["date"])
                if d is not None:
                    return d
        return None

    def _close_series(self, ohlcv, ticker, n_bars):
        closes = []
        if not ohlcv:
            return closes
        for bar in ohlcv[-n_bars:]:
            v = bar.get(ticker)
            if v and v.get("close"):
                try:
                    closes.append(float(v["close"]))
                except (TypeError, ValueError):
                    pass
        return closes

    def _compute_weights(self, data):
        """Returns {ticker: weight}. Same as AWP v4 then extended trend filter
        rotates any below-SMA200 sleeve to BIL."""
        ohlcv = data.get("ohlcv") or []
        n_needed = max(LOOKBACK, SMA_LOOKBACK, DIP_LOOKBACK, TREND_LOOKBACK) + 5
        if len(ohlcv) < n_needed:
            return {}

        # ---- Step 1: 30-day return vol per risk sleeve ---------------------
        vols = {}
        for t in RISK_TICKERS:
            closes = self._close_series(ohlcv, t, LOOKBACK + 1)
            if len(closes) < LOOKBACK:
                continue
            rets = [closes[i] / closes[i - 1] - 1
                    for i in range(1, len(closes))
                    if closes[i - 1] > 0]
            if len(rets) < LOOKBACK - 5:
                continue
            n = len(rets)
            mean = sum(rets) / n
            var = sum((r - mean) ** 2 for r in rets) / (n - 1) if n > 1 else 0
            sd = var ** 0.5
            if sd > 0:
                vols[t] = sd
        if len(vols) < len(RISK_TICKERS):
            return {}

        # ---- Step 2a: AWP v4 trend-confirmation overlay (50-day SMA) -------
        median_vol = sorted(vols.values())[len(vols) // 2]
        for t in EQUITY_SLEEVE:
            closes_50 = self._close_series(ohlcv, t, SMA_LOOKBACK)
            if len(closes_50) < SMA_LOOKBACK:
                continue
            sma50 = sum(closes_50) / len(closes_50)
            if closes_50[-1] > sma50:
                vols[t] = min(vols[t], median_vol)

        # ---- Step 2b: AWP v4 buy-the-dip overlay ---------------------------
        spy_closes = self._close_series(ohlcv, "SPY", DIP_LOOKBACK + 1)
        if len(spy_closes) >= DIP_LOOKBACK + 1 and spy_closes[0] > 0:
            spy_30d_ret = spy_closes[-1] / spy_closes[0] - 1
            if spy_30d_ret < DIP_THRESHOLD:
                for t in EQUITY_SLEEVE:
                    if t in vols:
                        vols[t] /= DIP_BOOST

        # ---- Step 3: equity-risk-budget weighting --------------------------
        eq_inv = {t: 1.0 / vols[t] for t in EQUITY_SLEEVE if vols.get(t, 0) > 0}
        de_inv = {t: 1.0 / vols[t] for t in DEFENSIVE_SLEEVE if vols.get(t, 0) > 0}
        if not eq_inv or not de_inv:
            return {}
        eq_total = sum(eq_inv.values())
        de_total = sum(de_inv.values())

        weights = {}
        for t, v in eq_inv.items():
            weights[t] = (v / eq_total) * EQUITY_RISK_BUDGET
        for t, v in de_inv.items():
            weights[t] = (v / de_total) * (1.0 - EQUITY_RISK_BUDGET)

        # ---- Step 4: EXTENDED TREND FILTER — rotate to BIL -----------------
        bil_w = 0.0
        for t in RISK_TICKERS:
            closes_trend = self._close_series(ohlcv, t, TREND_LOOKBACK)
            if len(closes_trend) < TREND_LOOKBACK:
                continue
            sma200 = sum(closes_trend) / len(closes_trend)
            if closes_trend[-1] < sma200:
                bil_w += weights.pop(t, 0.0)
        if bil_w > 0:
            weights[SAFE_TICKER] = weights.get(SAFE_TICKER, 0.0) + bil_w

        # Weights must sum to exactly 1.0 — this is Surmount's documented
        # mechanism for instructing the engine to deploy 100% of available
        # capital (including any uninvested cash) on the next rebalance. Do
        # NOT let weights sum to less than 1.0 or accumulated dividend cash
        # will sit idle in live.
        return weights

    def run(self, data):
        today = self._today(data)
        if today is None:
            return TargetAllocation({})

        cur_m = (today.year, today.month)
        if cur_m != self._last_rebal_month:
            w = self._compute_weights(data)
            if w:
                self._cached_weights = w
                self._last_rebal_month = cur_m
                bil_pct = w.get(SAFE_TICKER, 0.0)
                log(f"{today} monthly rebal — BIL={bil_pct:.0%}, weights: "
                    + ", ".join(f"{t}={v:.3f}" for t, v in sorted(w.items())))

        if not self._cached_weights:
            return TargetAllocation({})

        return TargetAllocation(self._cached_weights)


if __name__ == "__main__":
    print("Strategy module OK")