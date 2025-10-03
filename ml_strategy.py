import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from lob import LimitOrderBook


class MLFeatures:
    """
    Feature engineering for LOB data suitable for ML models.
    """

    def __init__(self, lookback_window: int = 10):
        self.lookback_window = lookback_window
        self.price_history = []
        self.spread_history = []
        self.imbalance_history = []
        self.scaler = StandardScaler()

    def update(self, lob: LimitOrderBook) -> np.ndarray:
        """
        Extract features from current LOB state.

        Returns:
            Feature vector for ML model
        """
        mid_price = lob.mid_price()
        spread = lob.spread()
        imbalance = lob.order_imbalance()

        if mid_price is None or spread is None or imbalance is None:
            return None

        # Update histories
        self.price_history.append(mid_price)
        self.spread_history.append(spread)
        self.imbalance_history.append(imbalance)

        # Keep only recent history
        if len(self.price_history) > self.lookback_window:
            self.price_history.pop(0)
            self.spread_history.pop(0)
            self.imbalance_history.pop(0)

        if len(self.price_history) < self.lookback_window:
            return None  # Not enough history

        # Calculate features
        features = []

        # Price-based features
        prices = np.array(self.price_history)
        returns = np.diff(prices) / prices[:-1]
        features.extend([
            prices[-1],  # current price
            np.mean(returns),  # avg return
            np.std(returns),   # volatility
            np.max(returns),   # max return
            np.min(returns),   # min return
        ])

        # Spread features
        spreads = np.array(self.spread_history)
        features.extend([
            spreads[-1],  # current spread
            np.mean(spreads),  # avg spread
            np.std(spreads),   # spread volatility
        ])

        # Imbalance features
        imbalances = np.array(self.imbalance_history)
        features.extend([
            imbalances[-1],  # current imbalance
            np.mean(imbalances),  # avg imbalance
            np.std(imbalances),   # imbalance volatility
        ])

        # Depth features
        bid_depth = sum(size for _, size in lob.bid_depth(5))
        ask_depth = sum(size for _, size in lob.ask_depth(5))
        features.extend([
            bid_depth,
            ask_depth,
            bid_depth / (ask_depth + 1e-6),  # bid/ask depth ratio
        ])

        return np.array(features)

    def get_feature_names(self) -> List[str]:
        """Return names of features for interpretability."""
        return [
            'current_price', 'avg_return', 'return_volatility', 'max_return', 'min_return',
            'current_spread', 'avg_spread', 'spread_volatility',
            'current_imbalance', 'avg_imbalance', 'imbalance_volatility',
            'bid_depth', 'ask_depth', 'depth_ratio'
        ]


class MLPricePredictor:
    """
    ML model for predicting short-term price movements.
    """

    def __init__(self, prediction_horizon: int = 5, model_type: str = 'rf'):
        """
        Args:
            prediction_horizon: How many timestamps ahead to predict
            model_type: 'rf' (Random Forest) or 'gb' (Gradient Boosting)
        """
        self.prediction_horizon = prediction_horizon
        self.model_type = model_type
        self.model = None
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the ML model.

        Args:
            X: Feature matrix
            y: Target price movements
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gb':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Calculate training score
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        print(f"ML Model trained - Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
    def predict(self, features: np.ndarray) -> float:
        """
        Predict price movement.

        Returns:
            Predicted price change (positive = bullish, negative = bearish)
        """
        if not self.is_trained or features is None:
            return 0.0

        return self.model.predict(features.reshape(1, -1))[0]

    def feature_importance(self) -> Dict[str, float]:
        """Return feature importance if available."""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return {}

        feature_names = MLFeatures().get_feature_names()
        importance = self.model.feature_importances_

        return dict(zip(feature_names, importance))


class MLMarketMaker:
    """
    ML-enhanced market making strategy that dynamically adjusts spreads.
    """

    def __init__(self, base_spread_bps: float = 5.0, position_limit: float = 100.0,
                 order_size: float = 2.0):
        self.base_spread_bps = base_spread_bps
        self.position_limit = position_limit
        self.order_size = order_size

        # ML components
        self.features = MLFeatures(lookback_window=15)
        self.spread_predictor = MLPricePredictor(prediction_horizon=3, model_type='rf')

        # State
        self.cash = 0.0
        self.position = 0.0
        self.trades = []

        # Training data collection
        self.training_data = []

    def collect_training_data(self, features: np.ndarray, future_price_change: float):
        """Collect data for training the ML model."""
        if features is not None:
            self.training_data.append((features, future_price_change))

    def train_models(self):
        """Train the ML models using collected data."""
        if len(self.training_data) < 100:  # Need minimum data
            return

        X = np.array([x for x, y in self.training_data])
        y = np.array([y for x, y in self.training_data])

        print(f"Training ML model with {len(X)} samples...")
        self.spread_predictor.train(X, y)

        # Clear training data after training
        self.training_data = []

    def update(self, lob: LimitOrderBook, timestamp: float) -> Dict:
        """
        Update strategy with ML-enhanced decision making.
        """
        mid_price = lob.mid_price()
        if mid_price is None:
            return {"action": "no_op", "cash": self.cash, "position": self.position}

        # Extract features
        features = self.features.update(lob)

        # Get ML prediction for optimal spread adjustment (with fallback)
        if self.spread_predictor.is_trained:
            predicted_move = self.spread_predictor.predict(features)
        else:
            # Fallback to simple momentum if ML not trained
            if len(self.features.price_history) >= 5:
                recent_prices = self.features.price_history[-5:]
                predicted_move = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            else:
                predicted_move = 0.0

        # Dynamically adjust spread based on ML prediction and market conditions
        base_spread = mid_price * (self.base_spread_bps / 10000)

        # Increase spread when expecting high volatility
        volatility_multiplier = 1.0 + abs(predicted_move) * 2.0

        # Adjust spread based on position (inventory management)
        position_adjustment = abs(self.position) / self.position_limit * 0.5

        dynamic_spread = base_spread * (volatility_multiplier + position_adjustment)

        # Calculate quote prices
        spread_offset = dynamic_spread / 2
        bid_price = mid_price - spread_offset
        ask_price = mid_price + spread_offset

        # Check position limits
        can_buy = self.position < self.position_limit
        can_sell = self.position > -self.position_limit

        actions = []

        # ML-enhanced trading logic with lower thresholds for more activity
        # If ML predicts upward movement, be more aggressive on buy side
        if predicted_move > 0.0005 and can_buy:  # Reduced threshold from 0.001 to 0.0005
            # Buy more aggressively
            fill_price = min(bid_price, lob.best_ask() or bid_price)
            self.position += self.order_size
            self.cash -= fill_price * self.order_size
            self.trades.append({
                "timestamp": timestamp,
                "side": "buy",
                "price": fill_price,
                "size": self.order_size,
                "reason": "ml_bullish"
            })
            actions.append(f"ML-BUY {self.order_size}@{fill_price:.2f}")

        # If ML predicts downward movement, be more aggressive on sell side
        elif predicted_move < -0.0005 and can_sell:  # Reduced threshold
            fill_price = max(ask_price, lob.best_bid() or ask_price)
            self.position -= self.order_size
            self.cash += fill_price * self.order_size
            self.trades.append({
                "timestamp": timestamp,
                "side": "sell",
                "price": fill_price,
                "size": self.order_size,
                "reason": "ml_bearish"
            })
            actions.append(f"ML-SELL {self.order_size}@{fill_price:.2f}")

        # Store data for future training
        if features is not None:
            # We'll collect the actual price movement in the next update
            pass

        return {
            "action": "ml_traded" if actions else "ml_quoted",
            "cash": self.cash,
            "position": self.position,
            "mid_price": mid_price,
            "dynamic_spread": dynamic_spread,
            "predicted_move": predicted_move,
            "actions": actions,
            "ml_trained": self.spread_predictor.is_trained
        }

    def get_pnl(self, current_price: float) -> float:
        """Calculate current PnL including unrealized position value."""
        return self.cash + self.position * current_price


class MLMomentumStrategy:
    """
    ML-enhanced momentum strategy using pattern recognition.
    """

    def __init__(self, position_size: float = 5.0, confidence_threshold: float = 0.3):
        self.position_size = position_size
        self.confidence_threshold = confidence_threshold

        # ML components
        self.features = MLFeatures(lookback_window=20)
        self.momentum_predictor = MLPricePredictor(prediction_horizon=5, model_type='gb')

        # State
        self.cash = 0.0
        self.position = 0.0
        self.trades = []

    def update(self, lob: LimitOrderBook, timestamp: float) -> Dict:
        """
        ML-based momentum detection and trading.
        """
        mid_price = lob.mid_price()
        if mid_price is None:
            return {"action": "no_op", "cash": self.cash, "position": self.position}

        features = self.features.update(lob)
        if features is None:
            return {"action": "waiting", "cash": self.cash, "position": self.position}

        # Get ML prediction (with fallback)
        if self.momentum_predictor.is_trained:
            predicted_move = self.momentum_predictor.predict(features)
        else:
            # Fallback to simple momentum if ML not trained
            if len(self.features.price_history) >= 5:
                recent_prices = self.features.price_history[-5:]
                predicted_move = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            else:
                predicted_move = 0.0

        action = "hold"
        actions = []
        confidence = abs(predicted_move) / 0.005  # Scale confidence

        # ML-based trend detection with lower confidence threshold
        if predicted_move > 0.001 and confidence > 0.3:  # Reduced thresholds for more activity
            # Strong bullish signal
            if self.position <= 0:
                # Close short if open
                if self.position < 0:
                    fill_price = lob.best_ask() or mid_price
                    self.cash += fill_price * abs(self.position)
                    actions.append(f"close_short {abs(self.position)}@{fill_price:.2f}")

                # Go long
                fill_price = lob.best_ask() or mid_price
                self.position += self.position_size
                self.cash -= fill_price * self.position_size
                self.trades.append({
                    "timestamp": timestamp,
                    "side": "buy",
                    "price": fill_price,
                    "size": self.position_size,
                    "confidence": confidence,
                    "predicted_move": predicted_move
                })
                action = "ml_long"
                actions.append(f"ML-LONG {self.position_size}@{fill_price:.2f} (conf: {confidence:.2f})")

        elif predicted_move < -0.001 and confidence > 0.3:  # Reduced thresholds
            # Strong bearish signal
            if self.position >= 0:
                # Close long if open
                if self.position > 0:
                    fill_price = lob.best_bid() or mid_price
                    self.cash += fill_price * self.position
                    actions.append(f"close_long {self.position}@{fill_price:.2f}")

                # Go short
                fill_price = lob.best_bid() or mid_price
                self.position -= self.position_size
                self.cash += fill_price * self.position_size
                self.trades.append({
                    "timestamp": timestamp,
                    "side": "sell",
                    "price": fill_price,
                    "size": self.position_size,
                    "confidence": confidence,
                    "predicted_move": predicted_move
                })
                action = "ml_short"
                actions.append(f"ML-SHORT {self.position_size}@{fill_price:.2f} (conf: {confidence:.2f})")

        return {
            "action": action,
            "cash": self.cash,
            "position": self.position,
            "mid_price": mid_price,
            "predicted_move": predicted_move,
            "confidence": confidence,
            "actions": actions,
            "ml_trained": self.momentum_predictor.is_trained
        }

    def get_pnl(self, current_price: float) -> float:
        """Calculate current PnL including unrealized position value."""
        return self.cash + self.position * current_price


def create_training_data_from_historical(lob_data: pd.DataFrame, prediction_horizon: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create training dataset from historical LOB data.

    Args:
        lob_data: DataFrame with LOB updates
        prediction_horizon: How many timestamps ahead to predict

    Returns:
        X, y for ML training
    """
    print("Creating training data from historical LOB data...")

    # This is a simplified version - in practice you'd need to replay the LOB
    # and collect features at each timestamp

    # For demo purposes, create synthetic training data
    np.random.seed(42)
    n_samples = 1000

    # Generate synthetic features (14 features as defined in MLFeatures)
    X = np.random.randn(n_samples, 14)

    # Generate synthetic targets (price movements)
    y = np.random.randn(n_samples) * 0.005  # ~0.5% typical moves

    return X, y


if __name__ == "__main__":
    print("ML Strategy - Use the web UI: python run_ui.py")
