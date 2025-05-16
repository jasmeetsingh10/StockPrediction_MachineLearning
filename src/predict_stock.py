import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from datetime import datetime, timedelta

def download_stock_data(symbol, user_input_date):
    end_date = user_input_date
    start_date = end_date - timedelta(days=90)

    if end_date.weekday() >= 5:
        end_date -= timedelta(days=end_date.weekday() - 4)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    print(f"Downloading data from {start_str} to {end_str}")
    data = yf.download(symbol, start=start_str, end=end_str)

    if len(data) < 30:
        raise ValueError(f"Not enough data points. Got only {len(data)} rows.")
    return data

def prepare_data(data, look_back=10):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:(i + look_back), 0])
        y.append(scaled_data[i + look_back, 0])

    X = np.array(X).reshape(-1, look_back, 1)
    y = np.array(y)
    return X, y, scaler

def build_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape, 1)),
        LSTM(128, activation="tanh", return_sequences=True),
        Dropout(0.2),
        LSTM(64, activation="tanh", return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation="tanh"),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def predict_future_days(model, recent_data, look_back, future_days, scaler):
    predicted_prices = []
    current_sequence = recent_data[-look_back:].copy()

    for _ in range(future_days):
        x_input = current_sequence.reshape(1, look_back, 1)
        next_val = model.predict(x_input, verbose=0)[0, 0]
        predicted_price = scaler.inverse_transform([[next_val]])[0, 0]
        predicted_prices.append(round(float(predicted_price), 2))
        current_sequence = np.append(current_sequence[1:], next_val)

    return predicted_prices

def return_prediction(stock_symbol, user_input_date, no_of_days):
    data = download_stock_data(stock_symbol, user_input_date)
    hist_data = [{"date": d.strftime("%Y-%m-%d"), "price": p} for d, p in data["Close"].items()]

    look_back = 15
    X, y, scaler = prepare_data(data["Close"], look_back)

    if len(X) < 5:
        raise ValueError(f"Not enough samples after preprocessing. Got only {len(X)} samples.")

    test_size = min(0.2, 1 / len(X))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)

    model = build_model(input_shape=look_back)
    model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=1)

    predicted = model.predict(X_test)
    mse_value = mse(y_test, predicted)
    accuracy = (1 - mse_value) * 100

    scaled_data = scaler.transform(data["Close"].values.reshape(-1, 1)).flatten()
    future_prices = predict_future_days(model, scaled_data, look_back, no_of_days, scaler)

    return future_prices, accuracy, hist_data

if __name__ == "__main__":
    stock_symbol = "AAPL"
    future_days = 3
    user_date = datetime.today()

    try:
        predictions, acc, history = return_prediction(stock_symbol, user_date, future_days)
        print(f"Historical data points: {len(history)}")
        print(f"Predicted Prices for the next {future_days} days: {predictions}")
        print(f"Model Accuracy: {acc:.2f}%")
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
