#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modern LOB Trading System UI
Inspired by Nothing Phone's minimalist aesthetic with grid patterns and clean typography.

Features:
- Sleek dark/light theme with grid backgrounds
- Real-time LOB visualization with heatmaps
- Interactive strategy performance dashboards
- Modern typography and spacing
- Responsive design elements
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lob import LimitOrderBook
from strategies import MarketMaker, MomentumStrategy
from ml_strategy import MLMarketMaker, MLMomentumStrategy
from backtest import LOBBacktester
from data_generator import SyntheticLOBGenerator


class ModernLOBUI:
    """
    Modern, sleek UI for LOB trading system inspired by Nothing Phone design.
    """
    
    def __init__(self):
        self.setup_page_config()
        self.setup_custom_css()
        
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="LOB Trading System",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def setup_custom_css(self):
        """Apply Nothing Phone-inspired custom CSS."""
        st.markdown("""
        <style>
        /* Nothing Phone Inspired Design */
        
        /* Main container with grid background */
        .main .block-container {
            background: linear-gradient(90deg, #f8f9fa 1px, transparent 1px),
                        linear-gradient(#f8f9fa 1px, transparent 1px);
            background-size: 20px 20px;
            background-color: #ffffff;
            padding: 2rem;
            max-width: 1400px;
        }
        
        /* Dark theme option */
        .dark-theme {
            background: linear-gradient(90deg, #2a2a2a 1px, transparent 1px),
                        linear-gradient(#2a2a2a 1px, transparent 1px);
            background-size: 20px 20px;
            background-color: #1a1a1a;
            color: #ffffff;
        }
        
        /* Typography - Nothing Phone style */
        .main-title {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 3rem;
            font-weight: 300;
            letter-spacing: -0.02em;
            color: #000000;
            margin-bottom: 0.5rem;
            text-align: center;
        }
        
        .subtitle {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 1.2rem;
            font-weight: 400;
            color: #666666;
            text-align: center;
            margin-bottom: 3rem;
        }
        
        /* Card containers */
        .metric-card {
            background: rgba(255, 255, 255, 0.8);
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        }
        
        .metric-card h3 {
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            font-weight: 500;
            color: #666666;
            margin: 0 0 0.5rem 0;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .metric-value {
            font-family: 'Inter', sans-serif;
            font-size: 2rem;
            font-weight: 300;
            color: #000000;
            margin: 0;
        }
        
        .metric-change {
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            font-weight: 500;
            margin-top: 0.5rem;
        }
        
        .positive {
            color: #00d4aa;
        }
        
        .negative {
            color: #ff6b6b;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            border-right: 1px solid #e0e0e0;
        }
        
        /* Button styling */
        .stButton > button {
            background: #000000;
            color: #ffffff;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .stButton > button:hover {
            background: #333333;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        /* Plotly chart styling */
        .js-plotly-plot {
            border-radius: 12px;
            overflow: hidden;
        }
        
        /* Grid layout helpers */
        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }
        
        /* Status indicators */
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }
        
        .status-live {
            background: #00d4aa;
            animation: pulse 2s infinite;
        }
        
        .status-paused {
            background: #ff6b6b;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 3px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render the main header with Nothing Phone styling."""
        st.markdown("""
        <div style="text-align: center; margin-bottom: 3rem;">
            <h1 class="main-title">LOB TRADING SYSTEM</h1>
            <p class="subtitle">Advanced Limit Order Book Analysis & Strategy Backtesting</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_metrics_cards(self, results: Dict):
        """Render performance metrics in modern card layout."""
        if not results:
            return
            
        st.markdown("### 📊 Performance Metrics")
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total PnL</h3>
                <div class="metric-value">${results.get('total_pnl', 0):,.2f}</div>
                <div class="metric-change {'positive' if results.get('total_pnl', 0) >= 0 else 'negative'}">
                    {results.get('sharpe_ratio', 0):.2f} Sharpe
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Max Drawdown</h3>
                <div class="metric-value">${results.get('max_drawdown', 0):,.2f}</div>
                <div class="metric-change {'negative' if results.get('max_drawdown', 0) < 0 else 'positive'}">
                    Risk Metric
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Win Rate</h3>
                <div class="metric-value">{results.get('win_rate', 0):.1%}</div>
                <div class="metric-change">
                    {len(results.get('trades', []))} Trades
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Volume</h3>
                <div class="metric-value">${sum(abs(t.get('size', 0) * t.get('price', 0)) for t in results.get('trades', [])):,.0f}</div>
                <div class="metric-change">
                    Volume Traded
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def render_lob_heatmap(self, lob_data: pd.DataFrame):
        """Render LOB heatmap visualization."""
        st.markdown("### 🔥 Order Book Heatmap")
        
        # Create sample heatmap data
        if lob_data.empty:
            st.warning("No LOB data available for heatmap")
            return
        
        # Group by price levels and calculate volume
        price_levels = lob_data.groupby(['price', 'side'])['size'].sum().reset_index()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=price_levels['size'].values.reshape(-1, 1),
            x=['Volume'],
            y=price_levels['price'].values,
            colorscale='RdYlBu_r',
            showscale=True,
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Order Book Depth Heatmap",
            xaxis_title="Side",
            yaxis_title="Price Level",
            height=400,
            font=dict(family="Inter", size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_strategy_comparison(self, results_dict: Dict[str, Dict]):
        """Render strategy comparison charts."""
        st.markdown("### 📈 Strategy Performance Comparison")
        
        if not results_dict:
            st.warning("No strategy results to compare")
            return
        
        # Create comparison chart
        fig = go.Figure()
        
        colors = ['#000000', '#666666', '#00d4aa', '#ff6b6b']
        
        for i, (strategy_name, results) in enumerate(results_dict.items()):
            if 'pnls' in results and len(results['pnls']) > 0:
                fig.add_trace(go.Scatter(
                    x=results['timestamps'],
                    y=results['pnls'],
                    mode='lines',
                    name=strategy_name.replace('_', ' ').title(),
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f'<b>{strategy_name}</b><br>' +
                                'Time: %{x}<br>' +
                                'PnL: $%{y:,.2f}<extra></extra>'
                ))
        
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Time",
            yaxis_title="PnL ($)",
            height=500,
            font=dict(family="Inter", size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_real_time_lob(self):
        """Render real-time LOB visualization."""
        st.markdown("### ⚡ Live Market Data Generation")
        st.info("Real-time market data will be fetched and processed during backtesting.")
    
    def render_sidebar(self):
        """Render sidebar with controls."""
        st.sidebar.markdown("## ⚙️ Controls")
        
        # Data source selection
        data_source = st.sidebar.selectbox(
            "Data Source",
            ["Synthetic", "Real Market Data"],
            help="Choose between synthetic or real market data"
        )
        
        # Symbol selection for real data
        symbol = st.sidebar.selectbox(
            "Market Symbol",
            ["BTC-USD", "AAPL", "TSLA", "MSFT", "GOOGL"],
            help="Select market symbol for real data",
            disabled=(data_source == "Synthetic")
        )
        
        # Strategy selection
        strategies = st.sidebar.multiselect(
            "Strategies to Run",
            ["Market Maker", "Momentum", "ML Market Maker", "ML Momentum"],
            default=["Market Maker", "Momentum"]
        )
        
        
        # Advanced settings
        st.sidebar.markdown("### 🔧 Advanced Settings")
        
        spread_bps = st.sidebar.slider(
            "Spread (bps)",
            min_value=1,
            max_value=50,
            value=5,
            help="Bid-ask spread in basis points"
        )
        
        position_limit = st.sidebar.slider(
            "Position Limit",
            min_value=10,
            max_value=1000,
            value=100,
            help="Maximum position size"
        )
        
        return {
            'data_source': data_source,
            'symbol': symbol,
            'strategies': strategies,
            'spread_bps': spread_bps,
            'position_limit': position_limit
        }
    
    def run_app(self):
        """Main application runner."""
        # Render header
        self.render_header()
        
        # Render sidebar
        controls = self.render_sidebar()
        
        # Main content area
        if st.button("🚀 Run Backtest", key="run_backtest"):
            with st.spinner("Running backtest analysis..."):
                # Initialize backtester
                if controls['data_source'] == "Synthetic":
                    generator = SyntheticLOBGenerator(
                        base_price=50000.0,
                        volatility=0.05,
                        order_frequency=8
                    )
                    backtester = LOBBacktester(generator)
                else:
                    # Fetch real market data dynamically
                    st.info("🔄 Fetching real market data from yfinance...")
                    
                    # Import real data generator
                    from real_lob_generator import RealDataLOBGenerator
                    
                    # Create a temporary CSV file for real data
                    os.makedirs("data", exist_ok=True)
                    temp_csv = "data/temp_real_data.csv"
                    
                    # Generate real data with 5000 units limit using selected symbol
                    real_generator = RealDataLOBGenerator(controls['symbol'], period="1d", interval="1m")
                    
                    # Show progress while generating
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Generate data with progress updates
                    data_points = []
                    for i, update in enumerate(real_generator.generate_updates(5000)):
                        if i % 100 == 0:  # Update progress every 100 timestamps
                            progress_bar.progress(min(i / 5000, 1.0))
                            status_text.text(f"Generating live data... {i}/5000 timestamps")
                        
                        # Convert to CSV format
                        for order in update["orders"]:
                            data_points.append({
                                "timestamp": update["timestamp"],
                                "side": order["side"],
                                "price": order["price"],
                                "size": order["size"]
                            })
                    
                    # Save to temporary CSV
                    df = pd.DataFrame(data_points)
                    df.to_csv(temp_csv, index=False)
                    
                    # Show sample of generated data
                    st.success(f"✅ Generated {len(data_points)} orders from {i+1} timestamps")
                    st.markdown("**Sample of generated data:**")
                    st.dataframe(df.head(5))
                    
                    # Initialize backtester with generated data
                    backtester = LOBBacktester(temp_csv)
                    
                    # Store temp file path for cleanup after backtesting
                    backtester.temp_file = temp_csv
                
                # Run selected strategies
                results_dict = {}
                for strategy in controls['strategies']:
                    strategy_name = strategy.lower().replace(' ', '_')
                    if strategy_name == 'market_maker':
                        results = backtester.run_backtest('market_maker',
                                                        spread_bps=controls['spread_bps'],
                                                        position_limit=controls['position_limit'],
                                                        order_size=2.0)
                    elif strategy_name == 'momentum':
                        results = backtester.run_backtest('momentum',
                                                        lookback_period=20,
                                                        threshold=0.0005,
                                                        position_size=5.0)
                    elif strategy_name == 'ml_market_maker':
                        results = backtester.run_backtest('ml_market_maker',
                                                        base_spread_bps=controls['spread_bps'],
                                                        position_limit=controls['position_limit'],
                                                        order_size=2.0)
                    elif strategy_name == 'ml_momentum':
                        results = backtester.run_backtest('ml_momentum',
                                                        position_size=5.0,
                                                        confidence_threshold=0.6)
                    
                    results_dict[strategy] = results
                
                # Display results
                for strategy, results in results_dict.items():
                    st.markdown(f"### {strategy} Results")
                    self.render_metrics_cards(results)
                
                # Strategy comparison
                if len(results_dict) > 1:
                    self.render_strategy_comparison(results_dict)
                
                # Clean up temporary file if it exists
                if hasattr(backtester, 'temp_file') and os.path.exists(backtester.temp_file):
                    os.remove(backtester.temp_file)
        
        # Real-time LOB section
        st.markdown("---")
        self.render_real_time_lob()
        
        # Footer
        st.markdown("""
        <div style="text-align: center; margin-top: 3rem; padding: 2rem; color: #666666;">
            <p>Built for quantitative trading by Arittra Bag</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main entry point for the Streamlit app."""
    ui = ModernLOBUI()
    ui.run_app()


if __name__ == "__main__":
    main()
