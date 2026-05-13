"""SURMOUNT DIAGNOSTIC — one-shot evidence collector.

Universe: SPY (50%) + BIL (50%), daily rebal back to that target.
Recommended backtest window: 2024-01-01 → 2024-12-31 (one full calendar year).

This portfolio is chosen because:
  - BIL has a ~5% dividend yield and nearly flat price — any NAV growth
    beyond price drift IS the dividend credit
  - SPY has a low ~1.3% dividend yield and large price moves — confirms
    the price feed is split-adj (raw close, not div-adj)
  - 50/50 forces the engine to trade DAILY (drift correction), exposing
    any rebalance friction
  - Mathematical ground truth for 2024 is computable from any independent
    price source

-------------------------------------------------------------------
2024 calendar year, daily-rebal continuous 50/50:

  BIL: close $91.27 → $91.55  → price-only +0.31% | divs ~$4.78 / $91.40 = +5.23% | TR ~+5.55%
  SPY: close $474.59 → $586.08 → price-only +23.49% | divs ~$6.95 / $530 = +1.31% | TR ~+25.10%

  Composite (daily-rebal 50/50, continuous compounding):
    Price-only:     ~ +11.40%   (no dividends credited at all)
    Cash-credit:    ~ +13.85%   (divs credited as cash, sits at 0% to next-day rebal)
    Full DRIP:      ~ +15.32%   (divs credited and immediately reinvested via daily rebal)

DECISION TREE (set BEFORE looking at result)
--------------------------------------------
  Surmount Total Return ≈ +11.4%       → Engine drops divs entirely (no credit)
  Surmount Total Return ≈ +13.8%       → Engine credits but cash sits idle
  Surmount Total Return ≈ +15.3%       → Engine fully DRIPs
  Anywhere else                        → Something we haven't modeled (slippage,
                                          discrete shares, price feed mismatch);
                                          requires further investigation

LOGS WE EMIT
------------
  1. INIT dump: data dict shape, dividend feed counts and first/last records
  2. EVERY MONTH: closes for SPY+BIL, price-only return-since-start,
     cumulative dividends visible in feed, current holdings dict
  3. EVERY ex-date: which ticker, amount, close on ex-date, running
     cumulative cash divs paid
  4. FINAL DAY: full per-asset summary + dollar-math decomposition
"""
from datetime import datetime

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


TICKERS = ["SPY", "BIL"]
TARGET_WEIGHTS = {"SPY": 0.50, "BIL": 0.50}


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
        self._data_list = [Dividend(t) for t in self.tickers]
        # Diagnostics state
        self._init_dumped = False
        self._first_closes = {}              # close on day 1, per ticker
        self._last_month = None
        self._cumulative_div_cash = {t: 0.0 for t in self.tickers}  # $ per share, hypothetical
        self._seen_ex = {t: set() for t in self.tickers}
        self._n_bars = 0

    @property
    def assets(self):
        return self.tickers

    @property
    def interval(self):
        return "1day"

    @property
    def data(self):
        return self._data_list

    def _today_and_closes(self, data):
        ohlcv = data.get("ohlcv") or []
        if not ohlcv:
            return None, {}
        last_bar = ohlcv[-1] or {}
        out = {}
        today = None
        for t in self.tickers:
            bar = last_bar.get(t) or {}
            d = _parse_date(bar.get("date"))
            if today is None and d is not None:
                today = d
            try:
                out[t] = float(bar.get("close"))
            except (TypeError, ValueError):
                pass
        return today, out

    def _emit_init(self, today, closes, data):
        top_keys = list(data.keys()) if hasattr(data, "keys") else []
        log(f"[DIAG-INIT @ {today}] data_keys={[str(k)[:50] for k in top_keys]}")
        log(f"[DIAG-INIT] start closes: SPY=${closes.get('SPY', 0):.2f}  BIL=${closes.get('BIL', 0):.4f}")
        holdings = data.get("holdings") or {}
        log(f"[DIAG-INIT] initial holdings dict: {dict(holdings)}")
        for t in self.tickers:
            divs = data.get(("dividend", t)) or []
            n = len(divs)
            first = divs[0] if divs else None
            last = divs[-1] if divs else None
            log(f"[DIAG-INIT] {t} dividend feed: {n} records | "
                f"first={first} | last={last}")

    def _track_new_dividends(self, today, closes, data):
        """Detect ex-dates that just appeared in the feed for the first time
        and accumulate hypothetical $/share dividend cash. Logs each new ex-date."""
        today_str = today.strftime("%Y-%m-%d")
        for t in self.tickers:
            divs = data.get(("dividend", t)) or []
            for d in divs:
                ex = str(d.get("date") or "")[:10]
                if not ex or ex > today_str:
                    continue
                if ex in self._seen_ex[t]:
                    continue
                self._seen_ex[t].add(ex)
                cash = d.get("adjDividend")
                if cash is None:
                    cash = d.get("dividend")
                try:
                    cash = float(cash or 0.0)
                except (TypeError, ValueError):
                    cash = 0.0
                if cash <= 0:
                    continue
                # Only count ex-dates AFTER we started running (the strategy
                # didn't own shares before its first bar).
                if not self._first_closes.get(t):
                    continue
                self._cumulative_div_cash[t] += cash
                px = closes.get(t)
                px_str = f"${px:.4f}" if px else "n/a"
                log(f"[DIAG-EX-DATE @ {today}] {t}: ex={ex} cash=${cash:.4f} "
                    f"px_today={px_str}  "
                    f"cum_div_per_share=${self._cumulative_div_cash[t]:.4f}")

    def _monthly_snapshot(self, today, closes, data):
        cur_m = (today.year, today.month)
        if cur_m == self._last_month:
            return
        self._last_month = cur_m
        holdings = data.get("holdings") or {}
        # Compute price-only return-since-start and total-since-start (price + cum_div)
        rep = []
        for t in self.tickers:
            c0 = self._first_closes.get(t)
            c = closes.get(t)
            if c0 and c:
                price_ret = (c / c0 - 1) * 100
                div_ret = (self._cumulative_div_cash[t] / c0) * 100
                total_ret = price_ret + div_ret
                rep.append(f"{t}: px=${c:.4f} ret_price={price_ret:+.3f}% "
                           f"ret_div={div_ret:+.3f}% ret_total={total_ret:+.3f}%")
        log(f"[DIAG-MONTH @ {today}] " + " | ".join(rep))
        log(f"[DIAG-MONTH @ {today}] holdings dict: {dict(holdings)}")

    def _final_summary(self, today, closes):
        """Emit a sum-up block that lets the reader (or me) decompose the gap.
        Surmount appears to feed bars cumulatively, so we lean on the last bar."""
        log("[DIAG-SUMMARY] " + "=" * 70)
        log(f"[DIAG-SUMMARY] @ {today}  ({self._n_bars} bars processed)")
        for t in self.tickers:
            c0 = self._first_closes.get(t, 0.0)
            c = closes.get(t, 0.0)
            div_ps = self._cumulative_div_cash[t]
            if c0 > 0:
                price_ret = (c / c0 - 1) * 100
                div_ret = (div_ps / c0) * 100
                total_ret = price_ret + div_ret
                log(f"[DIAG-SUMMARY] {t}: start=${c0:.4f}  end=${c:.4f}  "
                    f"price_ret={price_ret:+.3f}%  divs_per_share=${div_ps:.4f}  "
                    f"div_ret={div_ret:+.3f}%  TR={total_ret:+.3f}%")
        # Predicted 50/50 portfolio outcomes:
        weights = TARGET_WEIGHTS
        price_only = sum(weights[t] * ((closes.get(t, 0) / self._first_closes[t] - 1))
                          for t in self.tickers if self._first_closes.get(t))
        full_drip = sum(weights[t] * ((closes.get(t, 0) / self._first_closes[t] - 1) +
                                       (self._cumulative_div_cash[t] / self._first_closes[t]))
                         for t in self.tickers if self._first_closes.get(t))
        log(f"[DIAG-SUMMARY] Predicted 50/50 portfolio total return:")
        log(f"[DIAG-SUMMARY]   no-divs (price-only):  {price_only * 100:+.3f}%")
        log(f"[DIAG-SUMMARY]   full DRIP:             {full_drip * 100:+.3f}%")
        log(f"[DIAG-SUMMARY]   gap (divs effect):     {(full_drip - price_only) * 100:+.3f}pp")
        log(f"[DIAG-SUMMARY] Compare Surmount's reported TOTAL RETURN to those numbers.")
        log("[DIAG-SUMMARY] " + "=" * 70)

    def run(self, data):
        today, closes = self._today_and_closes(data)
        if today is None or len(closes) < 2:
            return TargetAllocation(TARGET_WEIGHTS)
        self._n_bars += 1

        if not self._init_dumped:
            self._init_dumped = True
            self._first_closes = dict(closes)
            self._emit_init(today, closes, data)

        self._track_new_dividends(today, closes, data)
        self._monthly_snapshot(today, closes, data)

        # Emit summary every 60 bars near end-of-year so it always lands in log
        # without needing an explicit end-of-backtest hook.
        if self._n_bars % 60 == 0 or today.month == 12 and today.day >= 28:
            self._final_summary(today, closes)

        return TargetAllocation(TARGET_WEIGHTS)


if __name__ == "__main__":
    print("Surmount diagnostic strategy module OK")
