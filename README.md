# LOB Trading System

**Production-ready algorithmic trading platform with machine learning integration**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Render](https://img.shields.io/badge/Deplpy-Render-green.svg)](https://render.com)

## Overview

A sophisticated limit order book (LOB) simulation and backtesting platform that combines traditional quantitative finance with modern machine learning. Built for institutional trading, hedge funds, and fintech companies.

## Key Features

### **Core Trading Engine**
- **High-Performance LOB**: O(log n) order matching with sorted data structures
- **Real-time Simulation**: Live market data integration via yfinance
- **Multi-Asset Support**: Equities, crypto, forex, commodities

### **Advanced Strategies**
- **Market Making**: Dynamic spread adjustment with risk management
- **Momentum Trading**: Trend-following with ML-enhanced signals
- **ML Integration**: Random Forest & Gradient Boosting for price prediction
- **Risk Controls**: Position limits, drawdown protection, volatility scaling

### **Professional Analytics**
- **Performance Metrics**: Sharpe ratio, max drawdown, win rate, P&L attribution
- **Risk Analytics**: VaR, stress testing, correlation analysis
- **Real-time Visualization**: Interactive dashboards with Plotly
- **Backtesting Engine**: Historical strategy validation with parallel processing

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Strategy Layer │    │  Analytics Layer│
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • yfinance API  │    │ • Market Making │    │ • Performance   │
│ • LOB Generator │    │ • Momentum      │    │ • Risk Metrics  │
│ • Real-time     │    │ • ML Strategies │    │ • Visualization │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Business Value

### **For Trading Firms**
- **Reduced Development Time**: 6+ months of quant development in days
- **Risk Management**: Built-in controls and monitoring
- **Scalability**: Cloud-ready architecture for production deployment

### **For Hedge Funds**
- **Strategy Research**: Rapid prototyping and backtesting
- **ML Integration**: Advanced signal generation and portfolio optimization
- **Compliance**: Audit trails and performance attribution

### **For Fintech Companies**
- **API Ready**: RESTful endpoints for integration
- **White-label**: Customizable UI and branding
- **Multi-tenant**: Support for multiple strategies and users

## Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend** | Python 3.12+ | Core trading engine |
| **ML/AI** | scikit-learn, NumPy | Strategy enhancement |
| **Data** | pandas, yfinance | Market data processing |
| **Frontend** | Streamlit, Plotly | Interactive dashboards |
| **Deployment** | Render, Docker | Cloud infrastructure |

## Quick Start

### **Local Development**
```bash
git clone https://github.com/Arittra-Bag/Quant-Lob-Backtester.git
cd Quant-Lob-Backtester
pip install -r requirements.txt
python run_ui.py
```

## Performance Benchmarks

| Metric | Value |
|--------|-------|
| **Order Processing** | 10,000+ orders/second |
| **Backtesting Speed** | 1M+ timestamps in <30s |
| **Memory Usage** | <512MB (free tier) |
| **Latency** | <100ms (local) |

## Use Cases

### **Institutional Trading**
- Algorithm development and testing
- Risk management and compliance
- Performance attribution and reporting

### **Hedge Funds**
- Strategy research and optimization
- Portfolio management and rebalancing
- Client reporting and transparency

### **Fintech Startups**
- MVP development and validation
- Investor demos and presentations
- Product-market fit testing

## Configuration

### **Strategy Parameters**
```python
# Market Making
spread_bps = 5.0          # Bid-ask spread
position_limit = 100.0     # Max position size
order_size = 2.0          # Order quantity

# Momentum
lookback_period = 20       # Signal window
threshold = 0.0005        # Entry threshold
position_size = 5.0       # Trade size
```

### **Risk Management**
```python
# Portfolio limits
max_drawdown = 0.15       # 15% max drawdown
volatility_target = 0.20  # 20% target volatility
correlation_limit = 0.70  # Max correlation
```

### **Operational Benefits**
- **Reduced Risk**: Built-in controls and monitoring
- **Faster Iteration**: Rapid strategy testing and deployment
- **Scalability**: Cloud-native architecture
