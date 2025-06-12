from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import json
from datetime import datetime

app = Flask(__name__)
# Updated CORS to allow your IP address
CORS(app, origins=["http://localhost:3000", "http://192.168.1.222:3000", "http://127.0.0.1:3000"])


def calculate_enhanced_indicators(df):
    """Calculate comprehensive technical indicators with safe Series handling"""
    data = df.copy()

    print("📊 Calculating enhanced technical indicators...")

    # Basic Moving Averages
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_12'] = data['Close'].ewm(span=12).mean()
    data['EMA_26'] = data['Close'].ewm(span=26).mean()

    # MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_histogram'] = data['MACD'] - data['MACD_signal']

    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    data['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands - FIXED
    try:
        bb_period = min(20, len(data) // 2) if len(data) < 40 else 20
        data['BB_middle'] = data['Close'].rolling(window=bb_period).mean()
        bb_std = data['Close'].rolling(window=bb_period).std()

        # Ensure single values, not Series
        bb_upper = data['BB_middle'] + (bb_std * 2)
        bb_lower = data['BB_middle'] - (bb_std * 2)

        data['BB_upper'] = bb_upper
        data['BB_lower'] = bb_lower
        data['BB_width'] = bb_upper - bb_lower

        # Safe division for BB position
        bb_width_safe = data['BB_width'].replace(0, np.nan).fillna(1e-10)
        data['BB_position'] = (data['Close'] - data['BB_lower']) / bb_width_safe
    except Exception as e:
        print(f"⚠️ BB calculation issue: {e}")
        data['BB_position'] = 0.5
        data['BB_width'] = data['Close'] * 0.02
        data['BB_upper'] = data['Close'] * 1.01
        data['BB_lower'] = data['Close'] * 0.99

    # Stochastic Oscillator - FIXED
    try:
        if 'High' in data.columns and 'Low' in data.columns:
            stoch_period = min(14, len(data) // 2)
            low_min = data['Low'].rolling(window=stoch_period).min()
            high_max = data['High'].rolling(window=stoch_period).max()
            high_low_diff = high_max - low_min
            high_low_diff_safe = high_low_diff.replace(0, np.nan).fillna(1e-10)
            data['Stoch_K'] = 100 * ((data['Close'] - low_min) / high_low_diff_safe)
            data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()
        else:
            # Use Close-based approximation
            stoch_period = min(14, len(data) // 2)
            close_low = data['Close'].rolling(window=stoch_period).min()
            close_high = data['Close'].rolling(window=stoch_period).max()
            close_range = close_high - close_low
            close_range_safe = close_range.replace(0, np.nan).fillna(1e-10)
            data['Stoch_K'] = 100 * ((data['Close'] - close_low) / close_range_safe)
            data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()
    except Exception as e:
        print(f"⚠️ Stochastic calculation issue: {e}")
        data['Stoch_K'] = pd.Series([50.0] * len(data), index=data.index)
        data['Stoch_D'] = pd.Series([50.0] * len(data), index=data.index)

    # ATR (Volatility) - FIXED
    try:
        atr_period = min(14, len(data) // 2)
        if 'High' in data.columns and 'Low' in data.columns:
            data['ATR'] = (data['High'] - data['Low']).rolling(window=atr_period).mean()
        else:
            data['ATR'] = data['Close'].rolling(window=atr_period).std()
    except Exception as e:
        print(f"⚠️ ATR calculation issue: {e}")
        data['ATR'] = data['Close'].rolling(window=5).std().fillna(0.001)

    # Price Momentum - FIXED
    try:
        data['Momentum_5'] = data['Close'] / data['Close'].shift(5)
        data['Momentum_10'] = data['Close'] / data['Close'].shift(10)
        data['ROC'] = ((data['Close'] - data['Close'].shift(12)) / data['Close'].shift(12)) * 100
    except Exception as e:
        print(f"⚠️ Momentum calculation issue: {e}")
        data['Momentum_5'] = pd.Series([1.0] * len(data), index=data.index)
        data['ROC'] = pd.Series([0.0] * len(data), index=data.index)

    # Support and Resistance - FIXED
    try:
        sr_period = min(20, len(data) // 2)
        data['Resistance'] = data['Close'].rolling(window=sr_period).max()
        data['Support'] = data['Close'].rolling(window=sr_period).min()
        support_resistance_range = data['Resistance'] - data['Support']
        support_resistance_safe = support_resistance_range.replace(0, np.nan).fillna(1e-10)
        data['Price_position'] = (data['Close'] - data['Support']) / support_resistance_safe
    except Exception as e:
        print(f"⚠️ S/R calculation issue: {e}")
        data['Price_position'] = pd.Series([0.5] * len(data), index=data.index)

    # Volume indicators (simulated for forex) - FIXED
    try:
        if 'Volume' not in data.columns or data['Volume'].isna().all():
            data['Volume'] = pd.Series([1000000] * len(data), index=data.index)

        data['Volume_SMA'] = data['Volume'].rolling(window=10).mean()
        volume_sma_safe = data['Volume_SMA'].replace(0, 1).fillna(1)
        data['Volume_ratio'] = data['Volume'] / volume_sma_safe
    except Exception as e:
        print(f"⚠️ Volume calculation issue: {e}")
        data['Volume'] = pd.Series([1000000] * len(data), index=data.index)
        data['Volume_SMA'] = pd.Series([1000000] * len(data), index=data.index)
        data['Volume_ratio'] = pd.Series([1.0] * len(data), index=data.index)

    print("✅ Enhanced technical indicators calculated")
    return data


def preprocess_enhanced_data(df, look_back=60, feature_set='standard'):
    """Enhanced preprocessing with feature selection - SAFE Series handling"""
    print(f"🔄 Preprocessing data with feature_set: {feature_set}")

    try:
        df_with_indicators = calculate_enhanced_indicators(df)

        # Feature sets
        if feature_set == 'basic':
            features = ['Close', 'SMA_10', 'SMA_20', 'EMA_12']
        elif feature_set == 'standard':
            features = ['Close', 'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26', 'MACD', 'RSI']
        elif feature_set == 'advanced':
            features = ['Close', 'SMA_10', 'SMA_20', 'EMA_12', 'MACD', 'RSI', 'BB_position', 'Stoch_K', 'Momentum_5']
        else:  # comprehensive
            features = [col for col in df_with_indicators.columns
                        if col not in ['Open', 'High', 'Low', 'Volume']
                        and df_with_indicators[col].dtype in ['float64', 'int64']]

        # Ensure features exist and filter valid ones
        available_features = [f for f in features if f in df_with_indicators.columns]
        if not available_features or 'Close' not in available_features:
            print("⚠️ Falling back to basic features")
            available_features = ['Close']
            # Add any available MA
            for ma in ['SMA_10', 'SMA_20', 'EMA_12']:
                if ma in df_with_indicators.columns:
                    available_features.append(ma)

        print(f"📈 Using features: {available_features}")

        # Clean data - handle NaN values safely
        feature_data = df_with_indicators[available_features].copy()

        # Fill NaN values with forward fill, then backward fill, then 0
        feature_data = feature_data.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Remove any remaining infinite values
        feature_data = feature_data.replace([np.inf, -np.inf], np.nan).fillna(0)

        print(f"📊 Clean data shape: {feature_data.shape}")

        if len(feature_data) < look_back + 5:
            raise ValueError(f"Insufficient data after cleaning: {len(feature_data)} < {look_back + 5}")

        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(feature_data.values)  # Use .values to ensure numpy array

        # Create sequences
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i - look_back:i])
            y.append(scaled_data[i, 0])  # Predict Close price (first column)

        X = np.array(X)
        y = np.array(y)

        print(f"✅ Sequences created: X{X.shape}, y{y.shape}")

        return X, y, scaler, feature_data, df_with_indicators

    except Exception as e:
        print(f"💥 Preprocessing error: {str(e)}")
        raise e


def build_enhanced_model(input_shape):
    """Enhanced LSTM model"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def generate_enhanced_prediction(symbol, timeframe, period, look_back, epochs, feature_set='standard'):
    """Enhanced prediction with comprehensive error handling"""
    try:
        print(f"🔄 Step 1: Fetching data for {symbol}")
        # Fetch data
        df = yf.download(symbol, period=period, interval=timeframe)
        df = df.dropna()

        print(f"📊 Downloaded {len(df)} data points")

        if len(df) < look_back + 20:
            return {"error": f"Insufficient data: {len(df)} points, need {look_back + 20}", "data_points": len(df)}

        print(f"🔄 Step 2: Processing indicators (feature_set: {feature_set})")
        # Preprocess with enhanced indicators
        X, y, scaler, feature_data, df_indicators = preprocess_enhanced_data(df, look_back, feature_set)

        print(f"📈 Training data: {X.shape[0]} samples, {X.shape[2]} features")

        if len(X) < 10:
            return {"error": f"Insufficient training data: {len(X)} samples", "samples": len(X)}

        print(f"🔄 Step 3: Building LSTM model")
        # Train enhanced model
        model = build_enhanced_model((X.shape[1], X.shape[2]))

        print(f"🔄 Step 4: Training model ({epochs} epochs)")
        model.fit(X, y, epochs=epochs, batch_size=16, verbose=0, validation_split=0.2)

        print(f"🔄 Step 5: Making prediction")
        # Make prediction
        recent_data = feature_data[-look_back:].values
        recent_scaled = scaler.transform(recent_data)
        X_pred = recent_scaled.reshape(1, look_back, -1)

        pred_scaled = model.predict(X_pred, verbose=0)

        # Inverse transform
        dummy = np.zeros((1, feature_data.shape[1]))
        dummy[0, 0] = pred_scaled[0, 0]
        predicted_price = scaler.inverse_transform(dummy)[0, 0]

        current_price = float(df.iloc[-1]['Close'])
        change = predicted_price - current_price
        change_pct = (change / current_price) * 100

        print(f"💰 Current: {current_price:.5f}, Predicted: {predicted_price:.5f}")

        # Generate signal with confirmations
        signal = "HOLD"
        confirmations = []

        if abs(change_pct) >= 0.01:
            signal = "BUY" if change > 0 else "SELL"

        print(f"🔄 Step 6: Processing technical indicators")
        # Get confirmations from indicators (simplified)
        try:
            latest = df_indicators.iloc[-1]

            # RSI confirmation
            if 'RSI' in df_indicators.columns and pd.notna(latest['RSI']):
                rsi_val = float(latest['RSI'])
                if signal == "BUY" and rsi_val < 70:
                    confirmations.append("RSI_OK")
                elif signal == "SELL" and rsi_val > 30:
                    confirmations.append("RSI_OK")

            # MACD confirmation
            if 'MACD' in df_indicators.columns and pd.notna(latest['MACD']):
                macd = float(latest['MACD'])
                if signal == "BUY" and macd > 0:
                    confirmations.append("MACD_BULLISH")
                elif signal == "SELL" and macd < 0:
                    confirmations.append("MACD_BEARISH")
        except Exception as conf_error:
            print(f"⚠️ Confirmation error (non-critical): {conf_error}")
            confirmations = ["BASIC_ANALYSIS"]

        if not confirmations:
            confirmations = ["PRICE_BASED"]

        # Build technical indicators dict (simplified)
        tech_indicators = {}
        try:
            latest = df_indicators.iloc[-1]

            for col in ['RSI', 'MACD', 'SMA_10', 'SMA_20']:
                if col in df_indicators.columns and pd.notna(latest[col]):
                    tech_indicators[col.lower()] = round(float(latest[col]), 6)
        except Exception as tech_error:
            print(f"⚠️ Technical indicators error (non-critical): {tech_error}")
            tech_indicators = {"rsi": 50.0, "macd": 0.0}

        print(f"✅ Prediction complete: {signal}")

        result = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "period": period,
            "feature_set": feature_set,
            "current_price": round(current_price, 5),
            "predicted_price": round(predicted_price, 5),
            "change": round(change, 5),
            "change_percent": round(change_pct, 2),
            "signal": signal,
            "confirmations": confirmations,
            "confidence": min(100, abs(change_pct) * 20),
            "data_points": len(df),
            "training_samples": len(X),
            "features_used": X.shape[2],
            "technical_indicators": tech_indicators,
            "model_performance": {
                "epochs_trained": epochs,
                "look_back_periods": look_back
            },
            "timestamp": datetime.now().isoformat()
        }

        return result

    except Exception as e:
        print(f"💥 EXCEPTION in generate_enhanced_prediction: {str(e)}")
        import traceback
        print(f"💥 Full traceback: {traceback.format_exc()}")
        return {"error": f"Prediction error: {str(e)}", "success": False}


# Routes
@app.route('/')
def home():
    return jsonify({
        "message": "Enhanced Forex LSTM API with Technical Indicators",
        "version": "2.0",
        "features": [
            "20+ Technical Indicators",
            "Multiple Feature Sets",
            "Enhanced LSTM Model",
            "Signal Confirmations",
            "Comprehensive Analysis"
        ],
        "endpoints": [
            "/api/predict (POST)",
            "/api/predict/<symbol> (GET)",
            "/api/pairs (GET)",
            "/api/indicators (GET)",
            "/api/health (GET)"
        ]
    })


@app.route('/api/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


@app.route('/api/pairs')
def get_pairs():
    pairs = [
        {"symbol": "EURUSD=X", "name": "EUR/USD", "category": "Major"},
        {"symbol": "GBPUSD=X", "name": "GBP/USD", "category": "Major"},
        {"symbol": "USDJPY=X", "name": "USD/JPY", "category": "Major"},
        {"symbol": "AUDUSD=X", "name": "AUD/USD", "category": "Major"},
        {"symbol": "USDCAD=X", "name": "USD/CAD", "category": "Major"},
        {"symbol": "USDCHF=X", "name": "USD/CHF", "category": "Major"},
        {"symbol": "NZDUSD=X", "name": "NZD/USD", "category": "Major"},
        {"symbol": "EURGBP=X", "name": "EUR/GBP", "category": "Cross"},
        {"symbol": "EURJPY=X", "name": "EUR/JPY", "category": "Cross"},
        {"symbol": "GBPJPY=X", "name": "GBP/JPY", "category": "Cross"},
        {"symbol": "BTCUSD=X", "name": "Bitcoin/USD", "category": "Crypto"},
        {"symbol": "ETHUSD=X", "name": "Ethereum/USD", "category": "Crypto"}
    ]
    return jsonify(pairs)


@app.route('/api/indicators')
def get_indicators():
    indicators = {
        "basic": ["Close", "SMA_10", "SMA_20", "EMA_12"],
        "standard": ["Close", "SMA_10", "SMA_20", "EMA_12", "EMA_26", "MACD", "RSI"],
        "advanced": ["Close", "SMA_10", "SMA_20", "EMA_12", "MACD", "RSI", "BB_position", "Stoch_K", "Momentum_5"],
        "comprehensive": [
            "Close", "SMA_10", "SMA_20", "SMA_50", "EMA_12", "EMA_26",
            "MACD", "MACD_signal", "RSI", "BB_position", "BB_width",
            "Stoch_K", "Stoch_D", "ATR", "Momentum_5", "ROC", "Price_position"
        ]
    }
    return jsonify(indicators)


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Parameters with defaults
        symbol = data.get('symbol', 'EURUSD=X')
        timeframe = data.get('timeframe', '1d')
        period = data.get('period', '1y')
        look_back = int(data.get('look_back', 60))
        epochs = int(data.get('epochs', 10))
        feature_set = data.get('feature_set', 'standard')

        # Validation
        valid_timeframes = ['1h', '4h', '1d', '1wk']
        valid_feature_sets = ['basic', 'standard', 'advanced', 'comprehensive']

        if timeframe not in valid_timeframes:
            return jsonify({"error": "Invalid timeframe", "valid": valid_timeframes}), 400

        if feature_set not in valid_feature_sets:
            return jsonify({"error": "Invalid feature set", "valid": valid_feature_sets}), 400

        if look_back < 10 or look_back > 200:
            return jsonify({"error": "Look back must be between 10 and 200"}), 400

        if epochs < 1 or epochs > 100:
            return jsonify({"error": "Epochs must be between 1 and 100"}), 400

        # Generate prediction
        result = generate_enhanced_prediction(symbol, timeframe, period, look_back, epochs, feature_set)

        if result.get("success"):
            return jsonify(result)
        else:
            return jsonify(result), 500

    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/predict/<symbol>')
def quick_predict(symbol):
    """Quick prediction with optimal parameters"""
    result = generate_enhanced_prediction(
        symbol=symbol + '=X',
        timeframe='1d',
        period='1y',
        look_back=60,
        epochs=10,
        feature_set='standard'
    )
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9800)