from sortedcontainers import SortedDict
from typing import Optional, Dict, List, Tuple
import numpy as np


class LimitOrderBook:
    """
    A limit order book implementation using sorted dictionaries for efficient
    order management and price level queries.
    """

    def __init__(self):
        # Use SortedDict for automatic sorting by price
        # bids: price -> total size (descending order for bids)
        # asks: price -> total size (ascending order for asks)
        self.bids = SortedDict()  # price -> size
        self.asks = SortedDict()  # price -> size

    def update(self, order: Dict[str, any]) -> None:
        """
        Update the order book with a new order or order update.

        Args:
            order: Dict containing 'side' ('buy'/'sell'), 'price' (float), 'size' (float)
                   Positive size = add/update, negative size = cancel/reduce
        """
        side = order['side']
        price = order['price']
        size = order['size']

        if side == 'buy':
            book = self.bids
        elif side == 'sell':
            book = self.asks
        else:
            raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'")

        # Update the price level
        current_size = book.get(price, 0)
        new_size = current_size + size

        if new_size <= 0:
            # Remove price level if size becomes zero or negative
            if price in book:
                del book[price]
        else:
            book[price] = new_size

    def update_batch(self, sides: np.ndarray, prices: np.ndarray, sizes: np.ndarray) -> None:
        """
        Update the order book with a batch of orders for maximum performance.
        
        Args:
            sides: Array of 'buy'/'sell' strings
            prices: Array of price values
            sizes: Array of size values
        """
        for i in range(len(prices)):
            side = sides[i]
            price = prices[i]
            size = sizes[i]

            if side == 'buy':
                book = self.bids
            elif side == 'sell':
                book = self.asks
            else:
                continue  # Skip invalid sides for performance

            # Update the price level
            current_size = book.get(price, 0)
            new_size = current_size + size

            if new_size <= 0:
                # Remove price level if size becomes zero or negative
                if price in book:
                    del book[price]
            else:
                book[price] = new_size

    def best_bid(self) -> Optional[float]:
        """Return the best (highest) bid price, or None if no bids."""
        return self.bids.peekitem(-1)[0] if self.bids else None

    def best_ask(self) -> Optional[float]:
        """Return the best (lowest) ask price, or None if no asks."""
        return self.asks.peekitem(0)[0] if self.asks else None

    def mid_price(self) -> Optional[float]:
        """Return the mid price between best bid and ask."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is not None and ask is not None:
            return (bid + ask) / 2
        return None

    def spread(self) -> Optional[float]:
        """Return the bid-ask spread."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is not None and ask is not None:
            return ask - bid
        return None

    def bid_depth(self, levels: int = 5) -> List[Tuple[float, float]]:
        """Return top N bid levels as (price, size) tuples."""
        return [(price, size) for price, size in self.bids.items()[-levels:]]

    def ask_depth(self, levels: int = 5) -> List[Tuple[float, float]]:
        """Return top N ask levels as (price, size) tuples."""
        return [(price, size) for price, size in self.asks.items()[:levels]]

    def order_imbalance(self) -> Optional[float]:
        """
        Calculate order imbalance: (total bid size - total ask size) / (total bid size + total ask size)
        Returns value between -1 and 1, where positive indicates buy pressure.
        """
        total_bid_size = sum(self.bids.values())
        total_ask_size = sum(self.asks.values())
        total_size = total_bid_size + total_ask_size

        if total_size == 0:
            return None

        return (total_bid_size - total_ask_size) / total_size

    def __str__(self) -> str:
        """String representation of the order book."""
        bid_str = " | ".join([f"{p:.2f}:{s:.2f}" for p, s in self.bid_depth(3)])
        ask_str = " | ".join([f"{p:.2f}:{s:.2f}" for p, s in self.ask_depth(3)])
        return f"LOB - Bids: [{bid_str}] | Asks: [{ask_str}] | Mid: {self.mid_price():.2f}"
