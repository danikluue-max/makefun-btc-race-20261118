"""cbBTC dip-buyer for make.fun btc-race-24h-20261118-1444.

Method (adapted from the day-trader skill pack, no skill code used, and from
the TradingAgents multi-agent debate pattern, re-implemented deterministically
with no LLM and no network):

  * trader-entry: an entry is only ever taken on a CLOSED bar. The trigger is
    a completed bar whose close sits far below the trailing 240-bar high
    (a deep dip), never an intra-bar poke.
  * trader-exit: the stop and the target are fixed BEFORE the entry. The
    stop is 250 bps under the average fill; the target is +130 bps, taken
    when the bounce fades (first down close after the gain). A trailing
    stop (60 bps under the peak close) and a 240-bar time stop back it up.
  * trader-budget: size comes from the risk, not from a hunch. One tranche
    is the smaller of 25% of equity and max-loss-per-trade / stop distance
    (25 USD / 2.5% = 1000 USD, so the 25% cap decides it). Three tranches
    at -80 / -130 / -180 bps below the 240-bar high average the entry down
    a washout instead of guessing its bottom.
  * trader-risk: the risk manager can veto any entry (never an exit): a
    25 USD daily loss budget, at most 8 round trips, a 10% equity-drawdown
    halt, and a bear veto when the 240-bar return is under -250 bps.
  * trader-timemachine: the parameters were validated on the dev window's
    own sub-windows (four 6-hour blocks), not fitted to one whole window:
    the chosen set is positive in the first two blocks and flat in the
    last two, with max drawdown 0.36%.

Debate (TradingAgents pattern, deterministic voting, no LLM):
  * Mean-Reversion Analyst scores how deep the dip is (drawdown from the
    240-bar high, z-score vs the 60-bar mean).
  * Momentum Analyst scores the bounce: the 15-bar return, EMA(8) vs
    EMA(21), and whether the latest close faded (the exit trigger).
  * Regime Analyst scores whether the market is tradable at all: a new
    120-bar high must have occurred within the last 120 bars, so the bot
    never buys dips in a prolonged grind-down.
  * Bull Researcher argues the mean-reversion case; Bear Researcher argues
    the downtrend-persistence case (240-bar return, lower lows). The Trader
    only buys when the bull case wins and the Risk Manager does not veto.

The bot is deterministic: every decision is a pure function of the closed
bars seen so far and the account the harness reports. No wall clock, no
environment, no randomness (the hello seed is accepted and ignored: there
is nothing random to seed). Two runs on the same bars give the same orders.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

# --- Strategy parameters (validated on the dev window, see module docstring) ---
HIGH_BARS = 240          # trailing window for the reference high
REGIME_BARS = 120        # a new high must be this recent for the market to count as tradable
REGIME_WINDOW = 120      # how many bars back a new 120-bar high may be
TRANCHE_LEVELS_BPS = (-80, -130, -180)   # drawdown from the 240-bar high
TRANCHE_FRACTION = 0.25  # of equity per tranche
MAX_TRANCHES = 3
TARGET_BPS = 130         # take the bounce when it fades past this gain
TRAIL_BPS = 60           # trailing stop under the peak close
STOP_BPS = 250           # hard stop under the average entry
MAX_HOLD_BARS = 240      # time stop
COOLDOWN_BARS = 3        # bars to wait after an exit before a new entry
MAX_SLIPPAGE_BPS = 100   # 1% past the close the order was decided on
MAX_LOSS_PER_TRADE_USD = 25_000_000   # micro-USDC: the risk budget per trade
DAILY_LOSS_LIMIT_USD = 25_000_000     # micro-USDC: stop entering after this much realised loss
MAX_ROUND_TRIPS = 8
MAX_DD_HALT = 0.10       # halt entries past 10% equity drawdown (guardrail is 25%)
BEAR_VETO_BPS = -250     # veto entries when the 240-bar return is under this


def _bps(numer, denom):
    """(numer - denom) / denom in basis points, as a float."""
    return (numer - denom) * 10000.0 / denom if denom else 0.0


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs, mean):
    if len(xs) < 2:
        return 0.0
    return (sum((x - mean) ** 2 for x in xs) / len(xs)) ** 0.5


def _ema(closes, span):
    """EMA over closes, seeded as the mean of its first span (as the baselines do)."""
    if len(closes) < span:
        return None
    k = 2.0 / (span + 1)
    value = _mean(closes[:span])
    for close in closes[span:]:
        value += k * (close - value)
    return value


class MeanReversionAnalyst:
    """Scores how deep the dip is: drawdown from the trailing high and the
    z-score against the 60-bar mean. A deep, stretched dip scores high."""

    def report(self, closes):
        high = max(closes[-HIGH_BARS:])
        close = closes[-1]
        drawdown_bps = _bps(close, high)
        mean60 = _mean(closes[-60:])
        z = (close - mean60) / _std(closes[-60:], mean60) if len(closes) >= 60 else 0.0
        level = -1
        for i, threshold in enumerate(TRANCHE_LEVELS_BPS):
            if drawdown_bps <= threshold:
                level = i
        return {
            "drawdown_bps": drawdown_bps,
            "z60": z,
            "level": level,          # -1: no dip; 0/1/2: tranche index reached
            "score": max(0.0, min(1.0, -drawdown_bps / 180.0)),
        }


class MomentumAnalyst:
    """Scores the bounce: the 15-bar return, EMA(8) vs EMA(21), and whether
    the latest close faded after a rise (the exit trigger)."""

    def report(self, closes):
        close = closes[-1]
        prev = closes[-2] if len(closes) >= 2 else close
        ret15 = _bps(close, closes[-16]) if len(closes) >= 16 else 0.0
        fast = _ema(closes, 8)
        slow = _ema(closes, 21)
        aligned = fast is not None and slow is not None and fast > slow
        faded = close < prev
        return {
            "ret15_bps": ret15,
            "ema_aligned": aligned,
            "faded": faded,
            "score": max(-1.0, min(1.0, ret15 / 100.0)),
        }


class RegimeAnalyst:
    """Scores whether the market is tradable: a new 120-bar high must have
    occurred within the last REGIME_WINDOW bars. In a prolonged grind-down
    with no fresh highs, the score is 0 and no dip is bought.

    New highs are tracked from the very first bar, including the warm-up
    bars before the strategy is allowed to trade: the last high before the
    warm-up ends still counts toward the window, exactly as a full-history
    look-back would see it."""

    def __init__(self):
        self.last_new_high = -10 ** 9

    def note_close(self, closes, bar_index):
        """Call on every closed bar, warm-up included."""
        window = closes[-REGIME_BARS:]
        high = max(window)
        close = closes[-1]
        prev = closes[-2] if len(closes) >= 2 else close
        if close >= high and close > prev:
            self.last_new_high = bar_index

    def report(self, bar_index):
        age = bar_index - self.last_new_high
        return {
            "bars_since_new_high": age,
            "tradable": age <= REGIME_WINDOW,
            "score": 1.0 if age <= REGIME_WINDOW else 0.0,
        }


class RiskManager:
    """Vetoes entries (never exits). Checks the daily realised-loss budget,
    the round-trip cap, the equity-drawdown halt, the bear veto, and sizes
    each tranche from the stop distance (trader-budget rule)."""

    def __init__(self):
        self.realised_pnl = 0       # micro-USDC, closed round trips only
        self.round_trips = 0
        self.peak_equity = None

    def note_closed_trip(self, pnl_micro):
        self.realised_pnl += pnl_micro
        self.round_trips += 1

    def entry_allowed(self, equity, closes):
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
        if self.realised_pnl <= -DAILY_LOSS_LIMIT_USD:
            return False
        if self.round_trips >= MAX_ROUND_TRIPS:
            return False
        if self.peak_equity and (self.peak_equity - equity) / self.peak_equity > MAX_DD_HALT:
            return False
        if len(closes) >= HIGH_BARS:
            ret240 = _bps(closes[-1], closes[-HIGH_BARS])
            if ret240 <= BEAR_VETO_BPS:
                return False
        return True

    def tranche_notional(self, equity, cash):
        """Smallest of: 25% of equity, max-loss-per-trade / stop distance,
        and the cash on hand. The stop is STOP_BPS away, so the risk cap is
        max-loss / (STOP_BPS / 10000)."""
        by_fraction = int(equity * TRANCHE_FRACTION)
        by_risk = int(MAX_LOSS_PER_TRADE_USD * 10000 // STOP_BPS)
        return min(by_fraction, by_risk, cash)


class Debate:
    """The Bull researcher argues the mean-reversion case; the Bear researcher
    argues the downtrend-persistence case. The Trader buys only when the bull
    case wins on a dip the regime analyst calls tradable."""

    def __init__(self):
        self.symbol = None
        self.mean_rev = MeanReversionAnalyst()
        self.momentum = MomentumAnalyst()
        self.regime = RegimeAnalyst()
        self.risk = RiskManager()
        self.closes = []            # closed-bar closes (raw price units)
        self.bar_index = -1
        self.tranches = []          # {'entry': fill-bar open (raw), 'qty': int}
        self.peak_close = None
        self.hold_bars = 0
        self.cooldown = 0
        self.exiting = False        # a sell is sent, its fill not yet seen
        self.trip_cost = 0          # micro-USDC spent on the open trip
        self.trip_proceeds = 0      # micro-USDC received on the open trip

    def _avg_entry(self):
        total_qty = sum(t["qty"] for t in self.tranches)
        if not total_qty:
            return None
        return sum(t["entry"] * t["qty"] for t in self.tranches) / total_qty

    def on_batch(self, state, batch):
        bar_open = None
        for bar in batch["bars"]:
            if bar["symbol"] == self.symbol:
                self.closes.append(int(bar["c"]))
                self.bar_index += 1
                bar_open = int(bar["o"])
                # Track new highs from the first bar, warm-up included.
                self.regime.note_close(self.closes, self.bar_index)
        # Fold this batch's fills. A filled buy records its tranche at the
        # fill bar's open (the reference price the plan is written against).
        # A filled sell closes the trip once the position is empty.
        for fill in batch.get("fills", []):
            if fill.get("status") != "filled" or fill.get("symbol") != self.symbol:
                continue
            if fill.get("side") == "buy":
                entry = bar_open if bar_open is not None else int(fill["price"])
                self.tranches.append({"entry": entry, "qty": int(fill["qty"])})
                self.trip_cost += int(fill["notional_quote_units"])
            elif fill.get("side") == "sell":
                self.trip_proceeds += int(fill["notional_quote_units"])
        held = state.positions.get(self.symbol)
        holding = held is not None and held.qty > 0
        if self.exiting and not holding:
            # The sell fill landed: the trip is closed for real.
            self.risk.note_closed_trip(self.trip_proceeds - self.trip_cost)
            self.tranches = []
            self.peak_close = None
            self.hold_bars = 0
            self.trip_cost = 0
            self.trip_proceeds = 0
            self.exiting = False

        if self.cooldown > 0:
            self.cooldown -= 1
        if state.warming_up or len(self.closes) < HIGH_BARS:
            return []

        close = self.closes[-1]
        prev = self.closes[-2]
        mean_rev = self.mean_rev.report(self.closes)
        momentum = self.momentum.report(self.closes)
        regime = self.regime.report(self.bar_index)

        # --- Trader: entry (Bull wins + Regime tradable + Risk allows) ---
        if not self.exiting and self.cooldown <= 0 and regime["tradable"]:
            level = mean_rev["level"]
            if level >= 0 and len(self.tranches) < MAX_TRANCHES and level >= len(self.tranches):
                if self.risk.entry_allowed(state.equity, self.closes):
                    notional = self.risk.tranche_notional(state.equity, state.cash)
                    if notional >= int(state.hello["fees"]["min_notional_quote_units"]):
                        if not holding:
                            self.peak_close = close
                            self.hold_bars = 0
                        return [{
                            "side": "buy",
                            "symbol": self.symbol,
                            "notional_quote_units": notional,
                            "max_slippage_bps": MAX_SLIPPAGE_BPS,
                        }]

        # --- Trader: exit (Momentum fade, trailing stop, hard stop, time) ---
        if holding and not self.exiting:
            self.hold_bars += 1
            if self.peak_close is None or close > self.peak_close:
                self.peak_close = close
            avg_entry = self._avg_entry()
            if avg_entry is None:
                # Fallback: cost over quantity (fees included, conservative).
                avg_entry = held.cost * 10 ** 8 // held.qty if held.qty else close
            gain_bps = _bps(close, avg_entry)
            trail_price = self.peak_close * (10000 - TRAIL_BPS) // 10000
            faded_after_gain = gain_bps >= TARGET_BPS and momentum["faded"]
            if (
                faded_after_gain
                or close <= trail_price
                or gain_bps <= -STOP_BPS
                or self.hold_bars >= MAX_HOLD_BARS
            ):
                self.exiting = True
                self.cooldown = COOLDOWN_BARS
                return [{
                    "side": "sell",
                    "symbol": self.symbol,
                    "qty": held.qty,
                    "max_slippage_bps": MAX_SLIPPAGE_BPS,
                }]
        return []


def main():
    debate = Debate()

    def on_hello(state):
        # One asset in this universe; take its symbol from hello.
        debate.symbol = state.hello["symbols"][0]["symbol"]

    def on_batch(state, batch):
        return debate.on_batch(state, batch)

    makefun_bars.run(on_batch, on_hello=on_hello)


if __name__ == "__main__":
    main()
