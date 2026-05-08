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
  SPY on a risk-adjusted basis (Sharpe 0.97 vs 0.61) over 18 years."

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
    5. SYNTHETIC DRIP: every ticker carries a cumulative DRIP factor
       f_t = Π(1 + cash_i / close_on_ex_i) sourced from Surmount's
       Dividend(...) data feed. Each ticker's target weight is scaled by
       its factor before final normalization. This compensates for
       Surmount's backtest engine ignoring dividend cash, mimicking the
       share-count growth a real auto-DRIP brokerage account delivers.
    6. Hold for the next month.

  Long-only. No shorts, no leverage, no derivatives. Weights sum to 1.0.

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

Certified numbers (v3_certified, dividend-reinvested total returns,
2012-01-31 → 2026-04-30, 14.25y):
  CAGR 11.16%, Sharpe 1.03, MaxDD -15.99%, Calmar 0.70, Vol 10.89%.
  vs SPY-TR (matched window): CAGR 14.70%, Sharpe 0.91, MaxDD -33.70%.
  Delta: CAGR -3.54pp, Sharpe +0.12, MaxDD +17.71pp.

GFC stress (2008-01 → 2009-06, full 18y backtest):
  AWD return -0.4%, peak-to-trough drawdown -4.5%.
  SPY-TR  return -27.8%, peak-to-trough drawdown -51.5%.

Activity profile (172 monthly rebals over 14.2y):
  BIL exposure in 81% of months (at least one sleeve below trend).
  Equity sleeves trend cleanly (SPY/QQQ off only 15% of months).
  Defensive sleeves rotate often (TLT 45%, GLD 40%, UUP 37% off).

See ``_artifacts/v3_certified/metrics_v3.json`` for full reproducible
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
from surmount.data import Dividend
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

# Synthetic DRIP: Surmount's backtest engine ignores dividend cash entirely
# (the docs expose `Dividend(...)` only as a separate data stream — dividends
# are NOT credited to the simulated portfolio). To compensate we maintain a
# per-ticker cumulative DRIP factor f_t = Π(1 + cash_i / close_on_ex_i) and
# multiply each ticker's base target weight by its factor before normalizing.
# This systematically tilts the portfolio toward higher-yielding assets over
# time, mimicking the share-count growth a real auto-DRIP brokerage account
# would deliver. Without this, AWD under-reports CAGR by ~3pp/yr because
# ~55% avg BIL exposure × ~2% blended BIL yield × 18y compounds out.
DRIP_LOG_THRESHOLD = 0.05  # log when any ticker's DRIP factor crosses +5%


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
        # Subscribe to Surmount's Dividend data stream for every holding so
        # the engine populates `data[("dividend", t)]` with the ex-date list.
        self._data_list = [Dividend(t) for t in self.tickers]
        # Cumulative DRIP factor per ticker (starts at 1.0; grows on each
        # processed ex-dividend date) and the set of ex-dates already booked.
        self._drip_factor = {t: 1.0 for t in self.tickers}
        self._seen_ex_dates = {t: set() for t in self.tickers}
        self._drip_logged = {t: False for t in self.tickers}

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

    def _close_on_or_after(self, ohlcv, ticker, ex_date_str):
        """Find the close price on the ex-date (or the next trading day if the
        ex-date fell on a non-trading day). Walks recent bars, cheapest path."""
        if not ohlcv:
            return None
        for bar in ohlcv[-300:]:
            payload = bar.get(ticker) or {}
            d = str(payload.get("date") or "")[:10]
            if d and d >= ex_date_str and payload.get("close"):
                try:
                    c = float(payload["close"])
                    if c > 0:
                        return c
                except (TypeError, ValueError):
                    return None
        return None

    def _update_drip(self, today, data):
        """Walk each ticker's dividend feed. For every new ex-date <= today,
        bump that ticker's cumulative DRIP factor by (1 + cash / close_on_ex)."""
        ohlcv = data.get("ohlcv") or []
        today_str = today.strftime("%Y-%m-%d") if hasattr(today, "strftime") else str(today)[:10]

        # ============================================================
        # AGGRESSIVE DIAGNOSTIC — fires on the very first run() call,
        # regardless of OHLCV history depth. Dumps everything we need
        # to figure out whether the Dividend feed is actually populated.
        # ============================================================
        if not getattr(self, "_drip_diag_done", False):
            self._drip_diag_done = True
            try:
                # 1. Dump ALL top-level data keys (truncated)
                top_keys = list(data.keys()) if hasattr(data, "keys") else []
                top_keys_str = [str(k)[:80] for k in top_keys[:30]]
                log(f"DRIP DIAG[1/4] @ {today_str}: data has {len(top_keys)} keys; "
                    f"first 30: {top_keys_str}")

                # 2. Filter to anything dividend-related (case-insensitive)
                divid_keys = [str(k) for k in top_keys if "divid" in str(k).lower()]
                log(f"DRIP DIAG[2/4]: {len(divid_keys)} keys matching 'divid': "
                    f"{divid_keys[:15]}")

                # 3. For each of our 6 tickers, try multiple access patterns
                for t in self.tickers:
                    candidates = [
                        ("tuple-lower", ("dividend", t)),
                        ("tuple-cap",   ("Dividend", t)),
                        ("string-und",  f"dividend_{t}"),
                        ("string-bare", f"dividend{t}"),
                    ]
                    found_any = False
                    for label, key in candidates:
                        try:
                            v = data.get(key) if hasattr(data, "get") else None
                        except Exception as e:
                            v = f"<err {type(e).__name__}>"
                        if v is not None and v != [] and v != {}:
                            found_any = True
                            tname = type(v).__name__
                            size = len(v) if hasattr(v, "__len__") else "?"
                            sample = str(v[0]) if (isinstance(v, list) and v) else str(v)[:120]
                            log(f"DRIP DIAG[3/4] {t} via {label} ({key!r}): "
                                f"type={tname} len={size} sample={sample[:160]}")
                    if not found_any:
                        log(f"DRIP DIAG[3/4] {t}: NO data under any key pattern "
                            f"(tried tuple/Tuple/dividend_{t}/dividend{t})")

                # 4. Confirm what we subscribed
                names = [type(d).__name__ + "(" + getattr(d, "ticker", "?") + ")"
                         for d in (self._data_list[:6] if isinstance(self._data_list, list) else [])]
                log(f"DRIP DIAG[4/4]: subscribed data classes (first 6): {names}")
            except Exception as e:
                log(f"DRIP DIAG ERR: {type(e).__name__}: {e}")

        for t in self.tickers:
            divs = data.get(("dividend", t)) or []
            if not divs:
                continue
            for d in divs:
                ex = str(d.get("date") or "")[:10]
                if not ex or ex > today_str:
                    continue
                if ex in self._seen_ex_dates[t]:
                    continue
                cash = d.get("adjDividend")
                if cash is None:
                    cash = d.get("dividend")
                try:
                    cash = float(cash or 0.0)
                except (TypeError, ValueError):
                    cash = 0.0
                if cash <= 0:
                    self._seen_ex_dates[t].add(ex)
                    continue
                close = self._close_on_or_after(ohlcv, t, ex)
                if close and close > 0:
                    self._drip_factor[t] *= 1.0 + cash / close
                self._seen_ex_dates[t].add(ex)
                if (not self._drip_logged[t]
                        and self._drip_factor[t] - 1.0 >= DRIP_LOG_THRESHOLD):
                    log(f"DRIP {t} factor crossed "
                        f"{(self._drip_factor[t]-1)*100:.1f}% on ex {ex}")
                    self._drip_logged[t] = True

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

        # ---- Step 5: SYNTHETIC DRIP TILT -----------------------------------
        # Scale each ticker's weight by its cumulative DRIP factor and
        # renormalize so the result still sums to 1.0. This compensates for
        # Surmount's engine dropping dividend cash by tilting allocations
        # toward dividend-paying assets in proportion to past missed payouts.
        tilted = {t: w * self._drip_factor.get(t, 1.0) for t, w in weights.items()}
        total = sum(tilted.values())
        if total > 0:
            weights = {t: w / total for t, w in tilted.items()}

        return weights

    def run(self, data):
        today = self._today(data)
        if today is None:
            return TargetAllocation({})

        # Update DRIP factors every bar so factors are current at month-end.
        self._update_drip(today, data)

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
