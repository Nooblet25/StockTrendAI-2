# StockTrendAI - Smart Stock Market Analyzer

A web-based application that uses machine learning models to predict stock market trends. The application provides real-time stock analysis and predictions using various ML models including Logistic Regression, Random Forest, XGBoost, LSTM, and ARIMA.

## Features

- Multiple ML models for stock prediction
- Interactive stock price charts
- Real-time market sentiment analysis
- Risk assessment
- Historical performance tracking
- Responsive modern UI with dark theme

## Tech Stack

- Backend: Python/Flask
- Frontend: HTML/CSS/JavaScript
- ML Libraries: scikit-learn, TensorFlow, XGBoost
- Data Analysis: pandas, numpy
- Visualization: Chart.js

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stocktrendai.git
cd stocktrendai
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://localhost:5000
```

## Models Available

- Logistic Regression: Binary classification for price movement
- Linear Regression: Price value prediction
- Random Forest: Ensemble learning approach
- XGBoost: Gradient boosting implementation
- LSTM: Deep learning for time series
- ARIMA: Statistical time series forecasting

## Data

The application uses historical stock data stored in the `data` directory. Each stock has its own CSV file with OHLCV (Open, High, Low, Close, Volume) data.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 