import pandas as pd
import numpy as np
import os
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Models
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import pickle

# -----------------------------
# Settings
# -----------------------------
df = pd.read_csv("training_dataset.csv")
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date','units_sold'])
df = df.sort_values(['product_id','date']).reset_index(drop=True)

window_size = 90
forecast_horizon_weeks = 4
forecast_horizon = forecast_horizon_weeks * 7  # Convert weeks to days

model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# -----------------------------
# LSTM Function
# -----------------------------
def train_lstm(product_df, product_id):
    sales_values = product_df['units_sold'].values.reshape(-1,1)
    
    if len(sales_values) <= window_size:
        print(f"Skipping {product_id} - not enough data for LSTM (window_size={window_size})")
        return None, None
    
    scaler = MinMaxScaler()
    sales_scaled = scaler.fit_transform(sales_values)

    # Prepare sequences
    X, y = [], []
    for i in range(window_size, len(sales_scaled)):
        X.append(sales_scaled[i-window_size:i,0])
        y.append(sales_scaled[i,0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(X.shape[1],1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)

    # Recursive forecast
    input_seq = sales_scaled[-window_size:].reshape(1, window_size,1)
    preds_scaled = []
    for _ in range(forecast_horizon):
        pred = model.predict(input_seq, verbose=0)
        preds_scaled.append(pred[0,0])
        input_seq = np.concatenate([input_seq[:,1:,:], pred.reshape(1,1,1)], axis=1)
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1))
    return model, preds

# -----------------------------
# ARIMA Function
# -----------------------------
def train_arima(product_df):
    sales_values = product_df['units_sold'].values
    model = ARIMA(sales_values, order=(5,1,0))
    model_fit = model.fit()
    preds = model_fit.forecast(steps=forecast_horizon)
    return model_fit, preds

# -----------------------------
# Prophet Function
# -----------------------------
def train_prophet(product_df):
    df_prophet = product_df[['date','units_sold']].rename(columns={'date':'ds','units_sold':'y'})
    df_prophet = df_prophet.dropna(subset=['ds','y'])
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=forecast_horizon)
    forecast = model.predict(future)
    preds = forecast['yhat'][-forecast_horizon:].values
    return model, preds

# -----------------------------
# Train All Products
# -----------------------------
forecast_results = []

for product_id in df['product_id'].unique():
    print(f"\nProcessing {product_id}")
    product_df = df[df['product_id']==product_id].sort_values('date')
    
    # LSTM
    lstm_model, lstm_preds = train_lstm(product_df, product_id)
    
    # Skip product if LSTM could not train (not enough data)
    if lstm_model is None:
        print(f"Skipping {product_id} entirely due to insufficient data.")
        continue
    
    # ARIMA
    arima_model, arima_preds = train_arima(product_df)
    
    # Prophet
    prophet_model, prophet_preds = train_prophet(product_df)
    
    # Evaluation
    sales_values = product_df['units_sold'].values
    eval_days = min(forecast_horizon, len(sales_values))
    actual = sales_values[-eval_days:]

    lstm_rmse = sqrt(mean_squared_error(actual, lstm_preds[:eval_days]))
    lstm_mape = mean_absolute_percentage_error(actual, lstm_preds[:eval_days])*100

    arima_rmse = sqrt(mean_squared_error(actual, arima_preds[:eval_days]))
    arima_mape = mean_absolute_percentage_error(actual, arima_preds[:eval_days])*100

    prophet_rmse = sqrt(mean_squared_error(actual, prophet_preds[:eval_days]))
    prophet_mape = mean_absolute_percentage_error(actual, prophet_preds[:eval_days])*100

    print(f"LSTM RMSE: {lstm_rmse:.2f}, ARIMA RMSE: {arima_rmse:.2f}, Prophet RMSE: {prophet_rmse:.2f}")

    # Choose best model
    metrics = {'LSTM': lstm_rmse, 'ARIMA': arima_rmse, 'Prophet': prophet_rmse}
    best_model_name = min(metrics, key=lambda k: metrics[k] if metrics[k] is not None else np.inf)
    print(f"Best model for {product_id}: {best_model_name}")

    # Save best model
    if best_model_name == 'LSTM':
        lstm_model.save(os.path.join(model_dir, f"best_model_{product_id}.keras"))
        best_preds = lstm_preds
    elif best_model_name == 'ARIMA':
        with open(os.path.join(model_dir, f"best_model_{product_id}_arima.pkl"), 'wb') as f:
            pickle.dump(arima_model, f)
        best_preds = arima_preds
    else:
        with open(os.path.join(model_dir, f"best_model_{product_id}_prophet.pkl"), 'wb') as f:
            pickle.dump(prophet_model, f)
        best_preds = prophet_preds

    # Save forecast for next N weeks
    future_dates = pd.date_range(start=product_df['date'].iloc[-1]+pd.Timedelta(days=1), periods=forecast_horizon)
    forecast_results.append(pd.DataFrame({
        'product_id': product_id,
        'date': future_dates,
        'forecast_units': best_preds.flatten()
    }))

# Combine all forecasts
all_forecasts = pd.concat(forecast_results, axis=0).reset_index(drop=True)
all_forecasts.to_csv("forecast_next_n_weeks.csv", index=False)
print("\nForecast for next N weeks saved as 'forecast_next_n_weeks.csv'")



