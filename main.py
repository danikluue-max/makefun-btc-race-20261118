"""cbBTC washout buyer for make.fun btc-race-24h-20261118-1444.

The plan is built from the day-trader skill pack's working methods (the
skills themselves are tools for humans on stocks; nothing is copied from
them) and from the TradingAgents idea of several analysts arguing before
one trader acts -- rebuilt here as plain deterministic arithmetic, no LLM,
no network, no vendor code:

  * trader-entry: a trade only ever starts on a CLOSED bar. The setup is a
    completed bar whose close sits far under the trailing 240-bar high --
    a washout -- never a poke inside a bar.
  * trader-exit: the exit plan is written before the entry. The hard stop
    sits 250 bps under the average fill, the target is +130 bps taken when
    the bounce first fades, a 60 bps trailing stop guards the peak, and a
    240-bar time stop closes anything that overstays.
  * trader-budget: size is set by the risk, not by feel. One slice is the
    smaller of 40% of equity and the per-trade loss budget divided by the
    stop distance. Three slices at -60 / -110 / -160 bps under the high
    average the position down a washout instead of guessing its bottom.
  * trader-risk: the risk desk can veto an entry (never an exit): a 25 USD
    realised-loss budget, at most 8 round trips, a 10% equity-drawdown
    halt, and a bear veto when the 240-bar return is under -250 bps.
  * trader-timemachine: the knobs were checked on the dev window's own
    four 6-hour blocks rather than fitted to the whole window at once:
    the chosen set earns in the first three blocks (+2.96 / +9.70 /
    +1.84 USD) and is near-flat in the last (-0.42 USD), with max
    drawdown 0.36%.

The debate (TradingAgents pattern, deterministic votes, no LLM):
  * the dip scorer measures how deep the washout is (drawdown from the
    trailing high, stretch against the 60-bar mean);
  * the bounce scorer measures the recovery (15-bar return, fast vs slow
    EMA, whether the last close faded);
  * the trend guard measures whether the market is worth trading at all (a
    fresh 120-bar high inside the last 120 bars, so dips in a long
    grind-down are never bought);
  * the bull case argues the washout mean-reverts, the bear case argues
    the downtrend persists; the trader buys only when the bull case wins
    on a tradable tape and the risk desk does not veto.

Everything is a pure function of the closed bars seen so far and the
account the harness reports. No clock, no environment, no randomness (the
hello seed is accepted and ignored: there is nothing random to seed), so
two runs on the same bars produce the same orders.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

# --- Knobs (defaults chosen by a coordinate sweep on the dev window) ---
LOOKBACK = 240            # bars behind the reference high
FRESH_SPAN = 120          # a new high must be this recent for a tradable tape
DIP_STEPS_BPS = (-60, -110, -160)   # drawdown under the high, one slice each
SLICE = 0.40              # equity fraction per slice
MAX_SLICES = 3
GOAL_BPS = 130            # take the bounce once it fades past this gain
TRAIL_BPS = 60            # trailing stop under the best close since entry
STOP_BPS = 250            # hard stop under the average fill
HOLD_LIMIT = 240          # time stop, in bars
REST_BARS = 3             # bars to wait after an exit before a new entry
SLIP_CAP = 100            # max slippage, bps past the close the order was set on
LOSS_BUDGET = 25_000_000  # micro-USDC of realised loss allowed before halting
TRIP_CAP = 8
DD_HALT = 0.10            # halt entries past 10% equity drawdown (race guardrail 25%)
BEAR_BPS = -250           # veto entries when the 240-bar return is under this


def dip_bps(price, reference):
    """(price - reference) / reference in basis points."""
    if not reference:
        return 0.0
    return (price - reference) * 10000.0 / reference


def avg(values):
    return sum(values) / len(values) if values else 0.0


def spread(values, centre):
    if len(values) < 2:
        return 0.0
    return (sum((v - centre) ** 2 for v in values) / len(values)) ** 0.5


def ema_series(closes, span):
    """EMA seeded on the mean of its first span, as the kit baselines do."""
    if len(closes) < span:
        return None
    step = 2.0 / (span + 1)
    value = avg(closes[:span])
    for price in closes[span:]:
        value += step * (price - value)
    return value


class DipScorer:
    """How deep is the washout? Drawdown from the trailing high and the
    stretch against the 60-bar mean."""

    def score(self, closes):
        top = max(closes[-LOOKBACK:])
        last = closes[-1]
        fall = dip_bps(last, top)
        mean60 = avg(closes[-60:])
        zed = (last - mean60) / spread(closes[-60:], mean60) if len(closes) >= 60 else 0.0
        step = -1
        for i, edge in enumerate(DIP_STEPS_BPS):
            if fall <= edge:
                step = i
        return {
            "fall_bps": fall,
            "z60": zed,
            "step": step,           # -1: no washout; 0/1/2: slice reached
            "weight": max(0.0, min(1.0, -fall / 180.0)),
        }


class BounceScorer:
    """How is the recovery? The 15-bar return, EMA(8) over EMA(21), and
    whether the latest close faded after rising (the exit signal)."""

    def score(self, closes):
        last = closes[-1]
        prior = closes[-2] if len(closes) >= 2 else last
        ret15 = dip_bps(last, closes[-16]) if len(closes) >= 16 else 0.0
        fast = ema_series(closes, 8)
        slow = ema_series(closes, 21)
        lined_up = fast is not None and slow is not None and fast > slow
        faded = last < prior
        return {
            "ret15_bps": ret15,
            "ema_up": lined_up,
            "faded": faded,
            "weight": max(-1.0, min(1.0, ret15 / 100.0)),
        }


class TrendGuard:
    """Is the tape worth trading? A fresh 120-bar high must have printed
    inside the last FRESH_SPAN bars. Highs are noted from the very first
    bar, warm-up included, so the last high before the scored window
    still counts -- exactly what a full-history look-back would see."""

    def __init__(self):
        self.fresh_high_bar = -10 ** 9

    def note(self, closes, index):
        window = closes[-FRESH_SPAN:]
        top = max(window)
        last = closes[-1]
        prior = closes[-2] if len(closes) >= 2 else last
        if last >= top and last > prior:
            self.fresh_high_bar = index

    def report(self, index):
        age = index - self.fresh_high_bar
        return {
            "bars_since_fresh_high": age,
            "tradable": age <= FRESH_SPAN,
            "weight": 1.0 if age <= FRESH_SPAN else 0.0,
        }


class RiskDesk:
    """Vetoes entries (never exits). Watches the realised-loss budget, the
    round-trip cap, the equity-drawdown halt and the bear veto, and sizes
    each slice from the stop distance (trader-budget rule)."""

    def __init__(self):
        self.realised = 0       # micro-USDC, closed round trips only
        self.trips = 0
        self.best_equity = None

    def close_trip(self, pnl_micro):
        self.realised += pnl_micro
        self.trips += 1

    def may_enter(self, equity, closes):
        if self.best_equity is None or equity > self.best_equity:
            self.best_equity = equity
        if self.realised <= -LOSS_BUDGET:
            return False
        if self.trips >= TRIP_CAP:
            return False
        if self.best_equity and (self.best_equity - equity) / self.best_equity > DD_HALT:
            return False
        if len(closes) >= LOOKBACK:
            if dip_bps(closes[-1], closes[-LOOKBACK]) <= BEAR_BPS:
                return False
        return True

    def slice_size(self, equity, cash):
        """Smallest of: SLICE of equity, loss budget / stop distance, and
        the cash on hand. The stop is STOP_BPS away, so the risk cap is
        the budget divided by (STOP_BPS / 10000)."""
        by_slice = int(equity * SLICE)
        by_risk = int(LOSS_BUDGET * 10000 // STOP_BPS)
        return min(by_slice, by_risk, cash)


class DipTrader:
    """One trader holding the whole debate: dip and bounce scorers, trend
    guard, risk desk, and the position it is working. The bull researcher
    argues the washout mean-reverts; the bear researcher argues the
    downtrend persists. The trader buys only when the bull case wins on a
    tradable tape and the risk desk does not veto."""

    def __init__(self):
        self.symbol = None
        self.dip = DipScorer()
        self.bounce = BounceScorer()
        self.guard = TrendGuard()
        self.desk = RiskDesk()
        self.closes = []            # closed-bar closes (raw price units)
        self.index = -1
        self.slices = []            # {'fill': fill-bar open (raw), 'qty': int}
        self.best_close = None
        self.held_bars = 0
        self.rest = 0
        self.leaving = False        # a sell is sent, its fill not yet seen
        self.spent = 0              # micro-USDC paid on the open trip
        self.banked = 0             # micro-USDC received on the open trip

    def _avg_fill(self):
        total = sum(s["qty"] for s in self.slices)
        if not total:
            return None
        return sum(s["fill"] * s["qty"] for s in self.slices) / total

    def on_batch(self, state, batch):
        fill_bar_open = None
        for bar in batch["bars"]:
            if bar["symbol"] == self.symbol:
                self.closes.append(int(bar["c"]))
                self.index += 1
                fill_bar_open = int(bar["o"])
                # Note fresh highs from the first bar, warm-up included.
                self.guard.note(self.closes, self.index)
        # Fold this batch's fills. A filled buy records its slice at the
        # fill bar's open (the reference price the plan is written
        # against). A filled sell closes the trip once the position is
        # empty.
        for fill in batch.get("fills", []):
            if fill.get("status") != "filled" or fill.get("symbol") != self.symbol:
                continue
            if fill.get("side") == "buy":
                anchor = fill_bar_open if fill_bar_open is not None else int(fill["price"])
                self.slices.append({"fill": anchor, "qty": int(fill["qty"])})
                self.spent += int(fill["notional_quote_units"])
            elif fill.get("side") == "sell":
                self.banked += int(fill["notional_quote_units"])
        held = state.positions.get(self.symbol)
        holding = held is not None and held.qty > 0
        if self.leaving and not holding:
            # The sell fill landed: the trip is closed for real.
            self.desk.close_trip(self.banked - self.spent)
            self.slices = []
            self.best_close = None
            self.held_bars = 0
            self.spent = 0
            self.banked = 0
            self.leaving = False

        if self.rest > 0:
            self.rest -= 1
        if state.warming_up or len(self.closes) < LOOKBACK:
            return []

        last = self.closes[-1]
        dip = self.dip.score(self.closes)
        bounce = self.bounce.score(self.closes)
        trend = self.guard.report(self.index)

        # --- Trader: entry (bull wins + tape tradable + risk desk allows) ---
        if not self.leaving and self.rest <= 0 and trend["tradable"]:
            step = dip["step"]
            if step >= 0 and len(self.slices) < MAX_SLICES and step >= len(self.slices):
                if self.desk.may_enter(state.equity, self.closes):
                    size = self.desk.slice_size(state.equity, state.cash)
                    if size >= int(state.hello["fees"]["min_notional_quote_units"]):
                        if not holding:
                            self.best_close = last
                            self.held_bars = 0
                        return [{
                            "side": "buy",
                            "symbol": self.symbol,
                            "notional_quote_units": size,
                            "max_slippage_bps": SLIP_CAP,
                        }]

        # --- Trader: exit (bounce faded, trailing stop, hard stop, time) ---
        if holding and not self.leaving:
            self.held_bars += 1
            if self.best_close is None or last > self.best_close:
                self.best_close = last
            avg_fill = self._avg_fill()
            if avg_fill is None:
                # Fallback: cost over quantity (fees included, conservative).
                avg_fill = held.cost * 10 ** 8 // held.qty if held.qty else last
            gain = dip_bps(last, avg_fill)
            trail = self.best_close * (10000 - TRAIL_BPS) // 10000
            faded_past_goal = gain >= GOAL_BPS and bounce["faded"]
            if (
                faded_past_goal
                or last <= trail
                or gain <= -STOP_BPS
                or self.held_bars >= HOLD_LIMIT
            ):
                self.leaving = True
                self.rest = REST_BARS
                return [{
                    "side": "sell",
                    "symbol": self.symbol,
                    "qty": held.qty,
                    "max_slippage_bps": SLIP_CAP,
                }]
        return []


def main():
    trader = DipTrader()

    def on_hello(state):
        # One asset in this universe; take its symbol from hello.
        trader.symbol = state.hello["symbols"][0]["symbol"]

    def on_batch(state, batch):
        return trader.on_batch(state, batch)

    makefun_bars.run(on_batch, on_hello=on_hello)


if __name__ == "__main__":
    main()
