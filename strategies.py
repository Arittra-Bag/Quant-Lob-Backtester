from typing import Dict, Tuple, List, Optional
import numpy as np
from collections import deque


class MarketMaker:
    """
    Simple market making strategy that continuously quotes bid and ask orders
    around the mid price.
    """

    def __init__(self, spread_bps: float = 2.0, position_limit: float = 100.0,
                 order_size: float = 1.0):
        """
        Args:
            spread_bps: Spread to quote around mid price in basis points
            position_limit: Maximum absolute position size
            order_size: Size of each order
        """
        self.spread_bps = spread_bps
        self.position_limit = position_limit
        self.order_size = order_size

        # State
        self.cash = 0.0
        self.position = 0.0
        self.trades = []

    def update(self, lob, timestamp: float) -> Dict:
        """
        Update strategy based on current LOB state.

        Args:
            lob: Current LimitOrderBook
            timestamp: Current timestamp

        Returns:
            Dict with strategy state and actions taken
        """
        mid_price = lob.mid_price()
        if mid_price is None:
            return {"action": "no_op", "cash": self.cash, "position": self.position}

        # Calculate spread offset
        spread_offset = mid_price * (self.spread_bps / 10000)

        # Quote prices
        bid_price = mid_price - spread_offset
        ask_price = mid_price + spread_offset

        # Check if we can place orders (within position limits)
        can_buy = self.position < self.position_limit
        can_sell = self.position > -self.position_limit

        actions = []

        # Place bid order if we can buy more
        if can_buy and bid_price <= lob.best_ask():
            # Simulate immediate fill at best ask (simplified)
            fill_price = min(bid_price, lob.best_ask() or bid_price)
            self.position += self.order_size
            self.cash -= fill_price * self.order_size
            self.trades.append({
                "timestamp": timestamp,
                "side": "buy",
                "price": fill_price,
                "size": self.order_size
            })
            actions.append(f"buy {self.order_size}@{fill_price:.2f}")

        # Place ask order if we can sell more
        if can_sell and ask_price >= lob.best_bid():
            # Simulate immediate fill at best bid (simplified)
            fill_price = max(ask_price, lob.best_bid() or ask_price)
            self.position -= self.order_size
            self.cash += fill_price * self.order_size
            self.trades.append({
                "timestamp": timestamp,
                "side": "sell",
                "price": fill_price,
                "size": self.order_size
            })
            actions.append(f"sell {self.order_size}@{fill_price:.2f}")

        return {
            "action": "traded" if actions else "quoted",
            "cash": self.cash,
            "position": self.position,
            "mid_price": mid_price,
            "actions": actions
        }

    def get_pnl(self, current_price: float) -> float:
        """Calculate current PnL including unrealized position value."""
        return self.cash + self.position * current_price


class MomentumStrategy:
    """
    Momentum strategy that detects price trends and trades accordingly.
    """

    def __init__(self, lookback_period: int = 20, threshold: float = 0.001,
                 position_size: float = 10.0):
        """
        Args:
            lookback_period: Number of periods to look back for momentum
            threshold: Minimum momentum threshold to trigger trade
            position_size: Size of position to take
        """
        self.lookback_period = lookback_period
        self.threshold = threshold
        self.position_size = position_size

        # State
        self.cash = 0.0
        self.position = 0.0
        self.price_history = deque(maxlen=lookback_period)
        self.trades = []

    def update(self, lob, timestamp: float) -> Dict:
        """
        Update strategy based on current LOB state.

        Args:
            lob: Current LimitOrderBook
            timestamp: Current timestamp

        Returns:
            Dict with strategy state and actions taken
        """
        mid_price = lob.mid_price()
        if mid_price is None:
            return {"action": "no_op", "cash": self.cash, "position": self.position}

        self.price_history.append(mid_price)

        if len(self.price_history) < self.lookback_period:
            return {"action": "waiting", "cash": self.cash, "position": self.position}

        # Calculate momentum (rate of change)
        momentum = (mid_price - self.price_history[0]) / self.price_history[0]

        action = "hold"
        actions = []

        # Trend detection
        if momentum > self.threshold and self.position <= 0:
            # Bullish momentum - go long
            if self.position < 0:
                # Close short position first
                fill_price = lob.best_ask() or mid_price
                self.cash += fill_price * abs(self.position)
                self.position = 0
                actions.append(f"close_short {abs(self.position)}@{fill_price:.2f}")

            # Go long
            fill_price = lob.best_ask() or mid_price
            self.position += self.position_size
            self.cash -= fill_price * self.position_size
            self.trades.append({
                "timestamp": timestamp,
                "side": "buy",
                "price": fill_price,
                "size": self.position_size
            })
            action = "long"
            actions.append(f"long {self.position_size}@{fill_price:.2f}")

        elif momentum < -self.threshold and self.position >= 0:
            # Bearish momentum - go short
            if self.position > 0:
                # Close long position first
                fill_price = lob.best_bid() or mid_price
                self.cash += fill_price * self.position
                self.position = 0
                actions.append(f"close_long {self.position}@{fill_price:.2f}")

            # Go short
            fill_price = lob.best_bid() or mid_price
            self.position -= self.position_size
            self.cash += fill_price * self.position_size
            self.trades.append({
                "timestamp": timestamp,
                "side": "sell",
                "price": fill_price,
                "size": self.position_size
            })
            action = "short"
            actions.append(f"short {self.position_size}@{fill_price:.2f}")

        return {
            "action": action,
            "cash": self.cash,
            "position": self.position,
            "mid_price": mid_price,
            "momentum": momentum,
            "actions": actions
        }

    def get_pnl(self, current_price: float) -> float:
        """Calculate current PnL including unrealized position value."""
        return self.cash + self.position * current_price
