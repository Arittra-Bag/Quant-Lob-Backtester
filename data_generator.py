import numpy as np
import pandas as pd
from typing import Generator, Dict, List
import random


class SyntheticLOBGenerator:
    """
    Generates synthetic limit order book updates for testing and demonstration.
    """

    def __init__(self, base_price: float = 100.0, volatility: float = 0.001,
                 order_frequency: int = 10, num_levels: int = 5):
        """
        Args:
            base_price: Starting mid price
            volatility: Daily volatility (fraction)
            order_frequency: Average orders per timestamp
            num_levels: Number of price levels to maintain
        """
        self.base_price = base_price
        self.volatility = volatility
        self.order_frequency = order_frequency
        self.num_levels = num_levels

        # State
        self.current_price = base_price
        self.timestamp = 0

        # Track current book state for realistic updates
        self.bid_levels = {}
        self.ask_levels = {}

    def _generate_price_levels(self, mid_price: float) -> Dict[str, Dict[float, float]]:
        """Generate realistic bid and ask levels around mid price."""
        spread = mid_price * 0.001  # 10bps spread
        tick_size = mid_price * 0.0001  # 1bps tick size

        bids = {}
        asks = {}

        # Generate bid levels (below mid)
        for i in range(self.num_levels):
            price = mid_price - spread/2 - i * tick_size
            size = random.uniform(0.1, 5.0)
            bids[round(price, 4)] = size

        # Generate ask levels (above mid)
        for i in range(self.num_levels):
            price = mid_price + spread/2 + i * tick_size
            size = random.uniform(0.1, 5.0)
            asks[round(price, 4)] = size

        return {"bids": bids, "asks": asks}

    def _generate_market_orders(self, levels: Dict[str, Dict[float, float]]) -> List[Dict]:
        """Generate market orders that consume liquidity."""
        orders = []

        # Small chance of market orders
        if random.random() < 0.1:
            if random.random() < 0.5 and levels["bids"]:
                # Market buy (hit asks)
                price = max(levels["asks"].keys())
                size = random.uniform(0.1, 2.0)
                orders.append({"side": "buy", "price": price, "size": -size})
            elif levels["asks"]:
                # Market sell (hit bids)
                price = min(levels["bids"].keys())
                size = random.uniform(0.1, 2.0)
                orders.append({"side": "sell", "price": price, "size": -size})

        return orders

    def _generate_limit_orders(self, levels: Dict[str, Dict[float, float]]) -> List[Dict]:
        """Generate limit orders that add liquidity."""
        orders = []

        # Add limit orders
        num_orders = np.random.poisson(self.order_frequency)

        for _ in range(num_orders):
            if random.random() < 0.5:
                # Add bid
                base_price = self.current_price * (1 - random.uniform(0.0005, 0.002))
                price = round(base_price, 4)
                size = random.uniform(0.1, 3.0)
                orders.append({"side": "buy", "price": price, "size": size})
            else:
                # Add ask
                base_price = self.current_price * (1 + random.uniform(0.0005, 0.002))
                price = round(base_price, 4)
                size = random.uniform(0.1, 3.0)
                orders.append({"side": "sell", "price": price, "size": size})

        return orders

    def generate_snapshot(self) -> Dict[str, Dict[float, float]]:
        """Generate a full LOB snapshot."""
        return self._generate_price_levels(self.current_price)

    def generate_updates(self, num_timestamps: int = 1000) -> Generator[Dict, None, None]:
        """
        Generate a sequence of LOB updates.

        Yields:
            Dict with 'timestamp' and 'orders' (list of order updates)
        """
        for t in range(num_timestamps):
            self.timestamp = t

            # Drift price with random walk
            # Assume ~1000 timestamps per day, so scale volatility accordingly
            timestamps_per_day = 1000
            price_change = np.random.normal(0, self.volatility / np.sqrt(252 * timestamps_per_day))
            self.current_price *= (1 + price_change)

            # Generate current levels
            levels = self._generate_price_levels(self.current_price)

            # Generate orders
            orders = []
            orders.extend(self._generate_market_orders(levels))
            orders.extend(self._generate_limit_orders(levels))

            # Occasionally add large orders to create volatility
            if random.random() < 0.05:
                side = "buy" if random.random() < 0.5 else "sell"
                price_offset = random.uniform(0.0002, 0.001)
                if side == "buy":
                    price = self.current_price * (1 - price_offset)
                else:
                    price = self.current_price * (1 + price_offset)

                size = random.uniform(5, 15)
                orders.append({"side": side, "price": round(price, 4), "size": size})

            yield {
                "timestamp": self.timestamp,
                "orders": orders,
                "mid_price": self.current_price
            }

    def save_to_csv(self, filename: str, num_timestamps: int = 10000) -> None:
        """
        Generate synthetic data and save to CSV format.

        CSV format: timestamp,side,price,size
        """
        data = []

        for update in self.generate_updates(num_timestamps):
            for order in update["orders"]:
                data.append({
                    "timestamp": update["timestamp"],
                    "side": order["side"],
                    "price": order["price"],
                    "size": order["size"]
                })

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Generated {len(df)} orders, saved to {filename}")


if __name__ == "__main__":
    print("Data Generator - Use the web UI: python run_ui.py")
