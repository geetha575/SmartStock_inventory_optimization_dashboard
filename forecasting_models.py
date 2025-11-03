import pandas as pd
import numpy as np
import os
import pickle
from keras.models import load_model
from datetime import timedelta

# Inventory optimization logic
def inventory_optimization_eoq(df, demand_col='forecast_units', ordering_cost=50, holding_cost_per_unit=2, lead_time_days=7, Z=1.65,forecast_horizon=28):
    
   
    df['EOQ'] = np.sqrt((2 * df[demand_col] * ordering_cost) / holding_cost_per_unit)
    
   
    df['safety_stock'] = Z * df[demand_col].std() * np.sqrt(lead_time_days)
    
  
    df['reorder_point'] = df[demand_col].mean() * lead_time_days + df['safety_stock']
    

    conditions = [
        df[demand_col] * lead_time_days < df['reorder_point'],
        df[demand_col] * lead_time_days > df['reorder_point']
    ]
    choices = ['Restock', 'Reduce']
    df['action'] = np.select(conditions, choices, default='Hold')
 
    # ABC classification
   
    total_demand = df.groupby('product_id')[demand_col].sum().sort_values(ascending=False)
    total = total_demand.sum()
    total_demand_cumsum = total_demand.cumsum() / total
    
    abc_class = {}
    for product_id, pct in total_demand_cumsum.items():
        if pct <= 0.8:
            abc_class[product_id] = 'A'
        elif pct <= 0.95:
            abc_class[product_id] = 'B'
        else:
            abc_class[product_id] = 'C'
    
    df['ABC_class'] = df['product_id'].map(abc_class)
 
    # Top-selling products (highest 5 forecasted demand)
    
    top_products = total_demand.head(5).index.tolist()
    df['top_selling'] = df['product_id'].apply(lambda x: 'Top Seller' if x in top_products else '')
    df['avg_daily_demand']= df[demand_col]/ forecast_horizon
    return df

# Forecasting function

def generate_forecasts(uploaded_df, model_dir="models", forecast_horizon=28):
    
    df = uploaded_df.copy()#Normalize columns
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Detect date column
    possible_date_cols = [c for c in df.columns if 'date' in c]
    if not possible_date_cols:
        raise KeyError("No date column found in uploaded file.")
    df.rename(columns={possible_date_cols[0]: 'date'}, inplace=True)

    # Detect product column
    possible_product_cols = [c for c in df.columns if 'product' in c]
    if not possible_product_cols:
        raise KeyError("No product_id column found in uploaded file.")
    df.rename(columns={possible_product_cols[0]: 'product_id'}, inplace=True)

    # Detect units sold
    possible_sales_cols = [c for c in df.columns if 'unit' in c or 'sales' in c]
    if not possible_sales_cols:
        raise KeyError("No units_sold column found in uploaded file.")
    df.rename(columns={possible_sales_cols[0]: 'units_sold'}, inplace=True)

    # Optional columns
    optional_cols_defaults = {
        'lead_time': 7,
        'discount': 0,
        'promotion_flag': 0,
        'seasonality': 1,
        'holiday': 0
    }
    for col, default in optional_cols_defaults.items():
        if col not in df.columns:
            df[col] = default

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'product_id', 'units_sold'])
    df = df.sort_values(['product_id', 'date'])

    forecasts = []

    for product_id in df['product_id'].unique():
        product_df = df[df['product_id'] == product_id].sort_values('date')

        keras_path = os.path.join(model_dir, f"best_model_{product_id}.keras")
        arima_path = os.path.join(model_dir, f"best_model_{product_id}_arima.pkl")
        prophet_path = os.path.join(model_dir, f"best_model_{product_id}_prophet.pkl")

        future_dates = pd.date_range(
            start=product_df['date'].iloc[-1] + timedelta(days=1),
            periods=forecast_horizon
        )

        # Forecast using pre-trained model
       
        preds = np.zeros(forecast_horizon)

        if os.path.exists(keras_path):
            # LSTM
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            sales_values = product_df['units_sold'].values.reshape(-1,1)
            sales_scaled = scaler.fit_transform(sales_values)

            window_size = min(90, len(sales_scaled))
            input_seq = sales_scaled[-window_size:].reshape(1, window_size,1)
            model = load_model(keras_path)

            for i in range(forecast_horizon):
                pred = model.predict(input_seq, verbose=0)
                preds[i] = scaler.inverse_transform(pred.reshape(-1,1))[0,0]
                input_seq = np.append(input_seq[:,1:,:], pred.reshape(1,1,1), axis=1)

        elif os.path.exists(arima_path):
            with open(arima_path, 'rb') as f:
                model = pickle.load(f)
            preds = model.forecast(steps=forecast_horizon)

        elif os.path.exists(prophet_path):
            with open(prophet_path, 'rb') as f:
                model = pickle.load(f)
            future = model.make_future_dataframe(periods=forecast_horizon)
            forecast = model.predict(future)
            preds = forecast['yhat'][-forecast_horizon:].values

        else:
            # Fallback: simple last value repeat
            last_value = product_df['units_sold'].iloc[-1]
            preds = np.array([last_value]*forecast_horizon)
        preds=np.array(preds).flatten() 
        last_date=product_df['date'].max() 
        future_dates=pd.date_range(start=last_date+pd.Timedelta(days=1),periods=forecast_horizon) 
         
        forecast_df = pd.DataFrame({
            'product_id': [product_id]*len(future_dates),
            'date': future_dates,
            'forecast_units': preds[:len(future_dates)]
        })
                
        forecasts.append(forecast_df)

    all_forecasts = pd.concat(forecasts, axis=0).reset_index(drop=True)
    optimized_df = inventory_optimization_eoq(all_forecasts)
    return optimized_df
    print(optimized_df)


