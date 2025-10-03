#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Real Market Data → Synthetic Limit Order Book (LOB) Generator
Clean version with only the essential RealDataLOBGenerator class.
"""

import os
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------- Config dataclass -----------------------------

@dataclass
class LOBSimConfig:
    spread_min: float = 1e-4   # 0.01%
    spread_max: float = 1e-3   # 0.10%
    levels: int = 5            # bid/ask levels
    vol_scale: float = 10.0    # scales size with volatility
    market_order_prob: float = 0.10
    max_updates_per_bar: int = 100  # hard cap per bar
    seed: Optional[int] = None

# ---------------------------- Fetch OHLCV (IO) ------------------------------

def fetch_ohlcv(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance."""
    data = yf.Ticker(symbol).history(period=period, interval=interval)
    
    if data.empty:
        raise ValueError(f"No data found for {symbol}")
    
    # Ensure we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return data

# ---------------------------- RealDataLOBGenerator --------------------------

class RealDataLOBGenerator:
    """
    Real market data to LOB generator class for programmatic use.
    """
    
    def __init__(self, symbol: str, period: str = "1d", interval: str = "1m"):
        self.symbol = symbol
        self.period = period
        self.interval = interval
        self.config = LOBSimConfig()
    
    def generate_updates(self, num_timestamps: int) -> Iterable[Dict]:
        """
        Generate LOB updates from real market data.
        
        Args:
            num_timestamps: Number of timestamp updates to generate
            
        Yields:
            Dict with timestamp and orders
        """
        # Fetch real market data
        df = fetch_ohlcv(self.symbol, self.period, self.interval)
        
        if df.empty:
            raise ValueError(f"No data found for symbol {self.symbol}")
        
        # Target number of synthetic timestamps
        target = num_timestamps
        
        # Calculate how many synthetic ticks per bar are needed
        timestamps_per_bar = max(1, (target // len(df)) + 1)
        
        generated = 0
        for i, (timestamp, row) in enumerate(df.iterrows()):
            for j in range(timestamps_per_bar):
                if generated >= target:
                    break
                    
                orders = []
                
                # Create synthetic LOB orders based on OHLCV data
                mid_price = (row['Open'] + row['Close']) / 2
                spread = abs(row['High'] - row['Low']) * 0.1  # 10% of daily range
                
                # Add small intra-bar variation
                price_variation = (row['High'] - row['Low']) * (j / timestamps_per_bar)
                current_price = mid_price + price_variation - (row['High'] - row['Low']) / 2
                
                # Generate bid orders
                for level in range(self.config.levels):
                    price = current_price - spread/2 - (level * spread/self.config.levels)
                    size = row['Volume'] * 0.1 * np.random.uniform(0.5, 1.5)
                    orders.append({
                        'side': 'buy',
                        'price': price,
                        'size': size
                    })
                
                # Generate ask orders
                for level in range(self.config.levels):
                    price = current_price + spread/2 + (level * spread/self.config.levels)
                    size = row['Volume'] * 0.1 * np.random.uniform(0.5, 1.5)
                    orders.append({
                        'side': 'sell',
                        'price': price,
                        'size': size
                    })
                
                yield {
                    'timestamp': int(timestamp.timestamp() * 1000) + j,  # Convert to milliseconds
                    'orders': orders
                }
                
                generated += 1

if __name__ == "__main__":
    print("Real LOB Generator - Use the web UI: python run_ui.py")
