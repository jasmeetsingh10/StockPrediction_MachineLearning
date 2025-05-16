# Stock Price Predictor with LSTM

This project predicts the future stock prices using historical data and an LSTM (Long Short-Term Memory) neural network.

## Features

- Downloads historical data from Yahoo Finance using `yfinance`
- Preprocesses and scales data using MinMaxScaler
- Uses a 3-layer LSTM model with dropout for training
- Predicts future prices for user-defined days
- Prints prediction accuracy and future price estimates

## Project Structure

```
stock-price-predictor/
├── src/ # Source code
│ └── predict_stock.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup

```bash
git clone https://github.com/yourusername/stock-price-predictor.git
cd stock-price-predictor
pip install -r requirements.txt
python src/predict_stock.py
```

## Requirements
- numpy
- pandas
- yfinance
- scikit-learn
- tensorflow
