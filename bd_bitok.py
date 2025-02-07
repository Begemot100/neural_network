import sqlite3
import requests
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from telegram import Bot
import matplotlib.pyplot as plt
import joblib
import time
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score


bot = Bot(token=bot_token)


def send_telegram_message(message):
    bot.send_message(chat_id=chat_id, text=message)


def create_table():
    connection = sqlite3.connect("bitcoin_prices.db")
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS bitcoin_prices
                      (timestamp TEXT PRIMARY KEY,
                       open REAL,
                       high REAL,
                       low REAL,
                       close REAL,
                       volume REAL,
                       close_time TEXT,
                       quote_asset_volume REAL,
                       number_of_trades INTEGER,
                       taker_buy_base_asset_volume REAL,
                       taker_buy_quote_asset_volume REAL,
                       ignore INTEGER,
                       trend TEXT,
                       price_change_percent REAL DEFAULT NULL)''')
    connection.commit()
    connection.close()


def preprocess_data_for_nn(data):
    X = data[['open', 'high', 'low', 'close', 'volume']]
    y = (data['trend'] == 'Up').astype(int)  # Convert 'Up' to 1, 'Down' to 0
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


def train_model(data):
    X, y = preprocess_data_for_nn(data)
    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
    last_timestamp = datetime.strptime(data.index[-1], "%Y-%m-%d %H:%M:%S")
    model.save(f'bitcoin_nn_model_{last_timestamp.strftime("%Y-%m-%d")}.h5')
    model = train_model_with_cross_validation(data)

    return model


def save_data_to_db(data):
    connection = sqlite3.connect("bitcoin_prices.db")
    cursor = connection.cursor()

    cursor.execute("PRAGMA table_info(bitcoin_prices)")
    columns = cursor.fetchall()
    has_price_change_percent = any("price_change_percent" in column for column in columns)

    if not has_price_change_percent:
        cursor.execute("ALTER TABLE bitcoin_prices ADD COLUMN price_change_percent REAL")
        connection.commit()

    data['price_change_percent'] = data['close'].pct_change() * 100
    data.to_sql("bitcoin_prices", connection, if_exists="append", index=False, index_label='timestamp', method='multi')

    connection.commit()
    connection.close()


def analyze_trend(data):
    data['trend'] = 'Neutral'
    data.loc[data['close'] > data['close'].shift(1), 'trend'] = 'Up'
    data.loc[data['close'] < data['close'].shift(1), 'trend'] = 'Down'
    return data


def calculate_entry_exit_percent(data, current_price, current_trend_prob):
    try:
        current_trend_prob = float(current_trend_prob)
    except ValueError:
        current_trend_prob = 0.0

    current_trend = 'Up' if current_trend_prob >= 0.5 else 'Down'

    if current_trend == 'Up':
        previous_price = data['close'].shift(1).iloc[-1]
        percent_change = ((current_price - previous_price) / previous_price) * 100
    elif current_trend == 'Down':
        previous_price = data['close'].shift(1).iloc[-1]
        percent_change = ((current_price - previous_price) / previous_price) * 100
    else:
        percent_change = 0.0

    return round(percent_change, 4)


def make_decision(model, data, current_data, interval):
    X_current = preprocess_data_for_nn(current_data)[0]
    current_trend_prob = model.predict(X_current)[0, 0]

    current_trend = 'Up' if current_trend_prob >= 0.5 else 'Down'
    percent_change = calculate_entry_exit_percent(data, current_data['close'].iloc[-1], current_trend)
    calculate_entry_exit_percent(data, current_price, current_trend_prob)
    if current_trend != previous_trend:
        reversal_message = f"Rеversal in trend detected for {interval} timeframe! New trend: {current_trend}"
        send_telegram_message(reversal_message)
        plot_price_and_forecast(data, percent_change, interval)
    if abs(percent_change) >= 1:
        notification_message = f"Neural Network: Trend for {interval}: {current_trend}\nPercent Change: {percent_change}"
        send_telegram_message_async(notification_message)
        plot_price_and_forecast(data, percent_change, interval)

        buy_signal_threshold = 0.05
        sell_signal_threshold = -0.05

        if current_trend == "Up" and percent_change > buy_signal_threshold:
            print("Generate Buy Signal")

        elif current_trend == "Down" and percent_change < sell_signal_threshold:
            print("Generate Sell Signal")

    return current_trend, percent_change





def plot_price_and_forecast(data, percent_change, interval):
    try:
        plt.figure(figsize=(8, 4))
        data['timestamp'] = pd.to_datetime(data.index)  
        plt.plot(data['timestamp'], data['close'], label='Closing Price', color='blue')
        last_timestamp = data['timestamp'].iloc[-1]
        plt.scatter(last_timestamp, data['close'].iloc[-1], color='red', marker='o', label='Last Price')

        forecast_interval_seconds = 3600  
        forecasted_price = data['close'].iloc[-1] * (1 + percent_change / 100)
        forecast_timestamp = last_timestamp + pd.Timedelta(seconds=forecast_interval_seconds)

        plt.plot([last_timestamp, forecast_timestamp], [data['close'].iloc[-1], forecasted_price],
                 linestyle='dashed', color='green', label='Forecast')
        plt.title(f'Цена и прогноз для таймфрейма {interval}')
        plt.xlabel('Метка времени', fontsize=12)
        plt.ylabel('Цена')
        plt.legend(fontsize=12)
        plt.xticks(rotation=45, ha = 'right', fontsize = 10)
        plt.grid(True)
        plt.yticks(fontsize = 10)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Произошла ошибка при построении графика: {e}")


def train_model(data):
    features = data.drop(['trend', 'price_change_percent'], axis=1)
    target = data['trend']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f'Model Accuracy: {accuracy}')

    last_timestamp = datetime.strptime(data.index[-1], "%Y-%m-%d %H:%M:%S")
    joblib.dump(model, f'bitcoin_trend_model_{last_timestamp.strftime("%Y-%m-%d")}.joblib')

    return model


def evaluate_model(X_test, y_test, model):
    y_pred = model.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy}")


def get_binance_data(symbol, interval, limit):
    base_url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    response = requests.get(base_url, params=params)
    data = response.json()

    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                     'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                     'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df.set_index('timestamp', inplace=True)
    df['close'] = df['close'].astype(float)

    return df

def make_decision(model, data, current_data, interval):
    current_data_frame = current_data.to_frame().T
    current_trend = model.predict(current_data_frame)[0]
    current_price = current_data_frame['close'].iloc[-1]
    percent_change = calculate_entry_exit_percent(data, current_price, current_trend)
    if abs(percent_change) >= 1:
        notification_message = f"Прогноз тренда для таймфрейма {interval}: {current_trend}\nПроцент изменения цены: {percent_change}"
        send_telegram_message(notification_message)
        # plot_price_and_forecast(data, percent_change, interval)

    return current_trend, percent_change


def get_and_save_bitcoin_prices(symbol, intervals, limit):
    create_table()

    for interval in intervals:
        data = get_binance_data(symbol, interval, limit)
        data = analyze_trend(data)
        save_data_to_db(data)

        model = train_model(data)
        current_data = data.iloc[-1].drop(['trend', 'price_change_percent'])

        current_trend, percent_change = make_decision(model, data, current_data, interval)
        print(f"Прогноз тренда для таймфрейма {interval}: {current_trend}")
        print(f"Процент изменения цены: {percent_change} %")


def create_model(X_shape):
    model = Sequential()
    model.add(Dense(64, input_dim=X_shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model_with_cross_validation(data):
    X, y = preprocess_data_for_nn(data)

    # Use StratifiedKFold for classification tasks
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = KerasClassifier(build_fn=create_model, X_shape=X.shape, epochs=10, batch_size=32, verbose=0)

    # Define the scoring metric
    scoring = {'accuracy': make_scorer(accuracy_score)}

    # Perform cross-validation
    results = cross_validate(model, X, y, cv=kfold, scoring=scoring, return_train_score=False)

    print(f'Cross-Validation Accuracy: {results["test_accuracy"].mean()}')
    print(f'Standard Deviation: {results["test_accuracy"].std()}')

    # Now train the model on the entire dataset
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

    last_timestamp = datetime.strptime(data.index[-1], "%Y-%m-%d %H:%M:%S")
    model.model.save(f'bitcoin_nn_model_{last_timestamp.strftime("%Y-%m-%d")}.h5')

    return model


start_time = time.time()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Время обучения: {elapsed_time:.2f} секунд")



if __name__ == "__main__":
    symbol = "BTCUSDT"
    intervals = ["5m", "15m", "1h", "4h", "1d"]
    limit = 1000  
    previous_trends = {interval: 'Neutral' for interval in intervals}

    while True:
        for interval in intervals:

            data = get_binance_data(symbol, interval, limit)
            data = analyze_trend(data)
            save_data_to_db(data)
            model = train_model_with_cross_validation(data)
            model = train_model(data)
            current_data = data.iloc[-1].drop(['trend', 'price_change_percent'])
            previous_trend = previous_trends[interval]
            current_trend, percent_change = make_decision(model, data, current_data, interval)
            previous_trends[interval] = current_trend
            print(f"Neural Network: Trend for {interval}: {current_trend}")
            print(f"Percent Change: {percent_change} %")

            time.sleep(60)  



