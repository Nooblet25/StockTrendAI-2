from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import os
from flask_cors import CORS  # Allow frontend JS to access backend

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"]
    }
})

# Available stock symbols (based on data folder)
AVAILABLE_STOCKS = [f.replace('.csv', '') for f in os.listdir('data') 
                   if f.endswith('.csv') and not f.endswith('.csv:Zone.Identifier')]

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Add route to serve the index.html file
@app.route('/')
def home():
    return send_file('index.html')

# Add route to serve the CSS file
@app.route('/styles.css')
def styles():
    return send_file('styles.css')

# Add route to serve the logo
@app.route('/logo.svg')
def logo():
    return send_file('logo.svg')

# === Fetch live data from local CSV files ===
def fetch_live_data(ticker, start="2010-01-01"):
    try:
        print(f"\nFetching data for {ticker}...")
        
        # Check if ticker exists in our data
        if ticker not in AVAILABLE_STOCKS:
            raise ValueError(f"Stock data not available for {ticker}")
            
        # Read the CSV file
        file_path = os.path.join('data', f'{ticker}.csv')
        if not os.path.exists(file_path):
            raise ValueError(f"Data file not found for {ticker}")
            
        try:
            df = pd.read_csv(file_path)
            
            # Convert date column
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Filter by start date
            df = df[df['Date'] >= start]
            
            if df.empty:
                raise ValueError(f"No data available for {ticker} after {start}")
                
            if len(df) < 60:  # Minimum required data points
                raise ValueError(f"Insufficient historical data for {ticker} (minimum 60 days required)")
            
            # Select required columns
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            print(f"Successfully loaded {len(df)} days of data")
            return df
            
        except Exception as e:
            print(f"Error reading data file: {str(e)}")
            raise
            
    except Exception as e:
        print(f"Error in fetch_live_data: {str(e)}")
        raise Exception(f"Error fetching data for {ticker}: {str(e)}")

@app.route('/available_stocks', methods=['GET'])
def get_available_stocks():
    """Return list of available stock symbols"""
    return jsonify(AVAILABLE_STOCKS)

# === Main prediction endpoint ===
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Debug: Print complete request information
        print("\n=== Request Debug Info ===")
        print("Form data:", request.form.to_dict())
        print("Content-Type:", request.headers.get('Content-Type'))
        print("Request method:", request.method)
        
        # Check if form data is present
        if not request.form:
            print("Error: No form data received")
            return jsonify({'error': 'No form data received'}), 400
        
        # Validate input parameters
        if 'ticker' not in request.form or not request.form['ticker'].strip():
            print("Error: Missing ticker symbol")
            return jsonify({'error': 'Ticker symbol is required'}), 400
        
        ticker = request.form['ticker'].strip().upper()
        model_choice = request.form.get('model', 'logistic').lower()
        
        # Set default values if not provided
        current_year = datetime.now().year
        end_year = current_year  # Default to current year
        
        try:
            if 'end_year' in request.form and request.form['end_year']:
                end_year = int(request.form['end_year'])
        except ValueError as e:
            print(f"Error parsing parameters: {e}")
            return jsonify({'error': 'Invalid year format'}), 400

        print(f"\nProcessing prediction:")
        print(f"Ticker: {ticker}")
        print(f"Model: {model_choice}")
        print(f"End Year: {end_year}")

        # Fetch and process data
        try:
            df = fetch_live_data(ticker)
            if df.empty:
                return jsonify({'error': f'No data available for ticker symbol {ticker}'}), 404
            print(f"Data fetched for {ticker}, shape: {df.shape}")
        except Exception as e:
            print(f"Error fetching data: {e}")
            return jsonify({'error': f'Could not fetch data for {ticker}. Please verify the symbol.'}), 404

        # Limit data to last 5 years to prevent memory issues
        df = df.tail(252 * 5)  # Approximately 5 years of trading days
        df = df[df['Date'].dt.year <= end_year]
        
        if df.empty:
            return jsonify({'error': f'No data available for {ticker} up to year {end_year}'}), 404
        
        if len(df) < 60:  # Require at least 60 days of data
            return jsonify({'error': f'Insufficient historical data for {ticker}. Need at least 60 days.'}), 400
        
        print(f"Filtered data shape: {df.shape}")
        
        df = df.dropna()
        if df.empty:
            return jsonify({'error': f'No valid data available for {ticker} after cleaning'}), 404
            
        print(f"Data shape after dropna: {df.shape}")

        # Feature engineering with error checking
        try:
            df['day'] = df['Date'].dt.day
            df['month'] = df['Date'].dt.month
            df['year'] = df['Date'].dt.year
            df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)
            df['open-close'] = df['Open'] - df['Close']
            df['low-high'] = df['Low'] - df['High']
            df['SMA_10'] = df['Close'].rolling(10).mean()
            df['SMA_50'] = df['Close'].rolling(50).mean()
            df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).mean()))
            df['Volatility'] = df['Close'].rolling(10).std()
            df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
            df.dropna(inplace=True)
            print("Feature engineering completed successfully")
        except Exception as e:
            print(f"Error in feature engineering: {e}")
            return jsonify({'error': 'Error processing stock data'}), 500

        if df.empty:
            return jsonify({'error': 'Insufficient data after feature engineering'}), 400

        features = df[['open-close', 'low-high', 'is_quarter_end', 'SMA_10', 'SMA_50', 'RSI', 'Volatility']]
        target = df['target']

        try:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            print("Data scaling completed successfully")
        except Exception as e:
            print(f"Error in data scaling: {e}")
            return jsonify({'error': 'Error in data preprocessing'}), 500

        X_train = features_scaled[:-1]
        y_train = target[:-1]
        X_input = features_scaled[-1:].reshape(1, -1)

        print(f"Training data shapes - X: {X_train.shape}, y: {y_train.shape}")

        # Model selection and training
        try:
            if model_choice == 'logistic':
                model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
                prob = model.predict_proba(X_input)[0][1]
                pred = int(prob > 0.5)

            elif model_choice == 'linear':
                model = LinearRegression().fit(X_train, y_train)
                prob = model.predict(X_input)[0]
                pred = int(prob > 0.5)

            elif model_choice == 'rf':
                model = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
                prob = model.predict_proba(X_input)[0][1]
                pred = int(prob > 0.5)

            elif model_choice == 'xgboost':
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(X_train, y_train)
                prob = model.predict_proba(X_input)[0][1]
                pred = int(prob > 0.5)

            elif model_choice == 'lstm':
                X_lstm = features_scaled.reshape(features_scaled.shape[0], 1, features_scaled.shape[1])
                X_train_lstm = X_lstm[:-1]
                y_train_lstm = target[:-1]

                lstm_model = Sequential([
                    LSTM(50, return_sequences=False, input_shape=(1, X_lstm.shape[2])),
                    Dropout(0.2),
                    Dense(1, activation='sigmoid')
                ])
                lstm_model.compile(optimizer='adam', loss='binary_crossentropy')
                lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=16, verbose=0)
                prob = float(lstm_model.predict(X_lstm[-1:])[0][0])
                pred = int(prob > 0.5)

            elif model_choice == 'arima':
                series = df['Close']
                model = ARIMA(series, order=(5, 1, 0)).fit()
                forecast = model.forecast(steps=1).tolist()  # Only predict next day
                return jsonify({
                    'model': 'arima',
                    'forecast': forecast,
                    'last_price': float(df['Close'].iloc[-1]),
                    'price_change': float(df['Close'].iloc[-1] - df['Close'].iloc[-2]),
                    'price_change_percent': float((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100)
                })

            else:
                return jsonify({'error': 'Invalid model selected'}), 400

            print(f"Model {model_choice} trained successfully")
            print(f"Prediction: {pred}, Confidence: {prob}")

        except Exception as e:
            print(f"Error in model training/prediction: {e}")
            return jsonify({'error': 'Error in model training/prediction'}), 500

        # Return result JSON
        result = {
            'ticker': ticker,
            'model': model_choice,
            'prediction': int(pred),
            'confidence': round(float(prob), 4),
            'last_price': float(df['Close'].iloc[-1]),
            'price_change': float(df['Close'].iloc[-1] - df['Close'].iloc[-2]),
            'price_change_percent': float((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100)
        }
        print("Returning result:", result)
        return jsonify(result)

    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({'error': str(e)}), 500

# === Run the server ===
if __name__ == '__main__':
    app.run(debug=True)