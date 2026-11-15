# multi_stock_model_training.py

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# --- Config ---
TICKERS = {
    'TATAMOTORS.NS': 'Tata Motors',
    'UNITDSPR.NS': 'United Spirits',
    'PETRONET.NS': 'Petronet LNG',
    'COLPAL.NS': 'Colgate-Palmolive',
    'BEL.NS': 'Bharat Electronics'
}

TARGETS = ['Open', 'Close', 'High', 'Low']

MODELS = {
    "lr": LinearRegression(),
    "rf": RandomForestRegressor(random_state=42),
    "xgb": XGBRegressor(random_state=42, verbosity=0),
    "lgb": LGBMRegressor(random_state=42)
}

PARAM_GRID = {
    "rf": {
        "n_estimators": [100],
        "max_depth": [3, 5]
    },
    "xgb": {
        "n_estimators": [100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5]
    },
    "lgb": {
        "n_estimators": [100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5]
    }
}

# --- Feature Engineering ---
def create_features(df):
    df = df.copy()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_30'] = df['Close'].rolling(30).mean()
    df['Price_Change'] = df['Close'].diff()
    df['High_Low_Diff'] = df['High'] - df['Low']
    df['Volume_Change'] = df['Volume'].diff()
    df['Close_Shift_1'] = df['Close'].shift(1)
    df.dropna(inplace=True)
    return df

# --- Train Models ---
all_models = {}
all_scalers = {}
all_errors = {}

for ticker in TICKERS:
    print(f"\n▶️ Training models for {ticker}...")

    df = yf.download(ticker, period="5y", interval="1d", auto_adjust=True)
    df = create_features(df)

    models_by_target = {}
    scalers_by_target = {}
    errors_by_target = {}

    for target in TARGETS:
        print(f"\n   ➤ Target: {target}")

        features = df.drop(columns=TARGETS)
        labels = df[target]

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        best_model = None
        best_score = -np.inf
        best_mae = np.inf
        best_r2 = -np.inf

        for name, model in MODELS.items():
            print(f"      🔍 Trying: {name}", end='')
            if name in PARAM_GRID:
                tscv = TimeSeriesSplit(n_splits=3)
                grid = GridSearchCV(model, PARAM_GRID[name], cv=tscv, scoring='r2', n_jobs=-1, verbose=1)
                grid.fit(X_train_scaled, y_train)
                model = grid.best_estimator_
                print(" ✅ GridSearch applied")
            else:
                model.fit(X_train_scaled, y_train)
                print(" ✅")

            preds = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            print(f" → MAE: {mae:.4f}, R²: {r2:.4f}")

            if r2 > best_score:
                best_model = model
                best_score = r2
                best_mae = mae
                best_r2 = r2

        models_by_target[target.lower()] = best_model
        scalers_by_target[target.lower()] = scaler
        errors_by_target[target.lower()] = {"mae": best_mae, "r2": best_r2}

        print(f"      ✅ Best → MAE: {best_mae:.4f}, R²: {best_r2:.4f}")

    all_models[ticker] = models_by_target
    all_scalers[ticker] = scalers_by_target
    all_errors[ticker] = errors_by_target

# --- Save Models, Scalers, Errors ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
joblib.dump(all_models, f"multi_stock_models_{timestamp}.pkl")
joblib.dump(all_scalers, f"multi_stock_scalers_{timestamp}.pkl")
joblib.dump(all_errors, f"multi_stock_errors_{timestamp}.pkl")

# --- Export error metrics to CSV ---
error_df = pd.DataFrame.from_dict({
    (ticker, target): metrics
    for ticker, target_errors in all_errors.items()
    for target, metrics in target_errors.items()
}, orient='index')
error_df.index.names = ['Ticker', 'Target']
error_df.reset_index(inplace=True)
error_df.to_csv(f"multi_stock_errors_{timestamp}.csv", index=False)

print("\n✅ All models and metrics saved successfully!")


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import yfinance as yf
import os
from datetime import datetime, timedelta

# --- Constants ---
tickers = [
    'TATAMOTORS.NS',
    'UNITDSPR.NS',
    'PETRONET.NS',
    'COLPAL.NS',
    'BEL.NS'
]
features = ['Open', 'High', 'Low', 'Close', 'SMA_10', 'SMA_30', 'Price_Change']

# --- Feature Engineering ---
def create_features(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df_feat = df.copy()
    df_feat['SMA_10'] = df_feat['Close'].rolling(window=10).mean()
    df_feat['SMA_30'] = df_feat['Close'].rolling(window=30).mean()
    df_feat['Price_Change'] = df_feat['Close'].diff()
    df_feat['High_Low_Diff'] = df_feat['High'] - df_feat['Low'] # Added this line as it's used in app.py features
    df_feat['Volume_Change'] = df_feat['Volume'].diff() # Added this line as it's used in app.py features
    df_feat['Close_Shift_1'] = df_feat['Close'].shift(1) # Added this line as it's used in app.py features
    df_feat.dropna(inplace=True)
    return df_feat

# --- Model Training ---
def train_and_save_models():
    models = {}
    scalers = {}
    errors = {}
    backtest = {}

    for ticker in tickers:
        print(f"🔄 Downloading data for {ticker}...")
        # Download more data to ensure at least 60 days for backtesting + features
        df = yf.download(ticker, period="2y") # Increased period to ensure enough data for 60-day backtest after feature creation
        df_feat = create_features(df)

        # Ensure enough data after feature creation for backtesting
        if len(df_feat) < 60:
            print(f"⚠️ Not enough historical data for {ticker} to perform 60-day backtest after feature creation. Skipping backtest for this ticker.")
            # Still proceed with model training if possible
            X_full = df_feat[features]
            y_targets_full = {
                'open': ((df_feat['Open'].shift(-1) - df_feat['Open']) / df_feat['Open']) * 100,
                'close': ((df_feat['Close'].shift(-1) - df_feat['Open']) / df_feat['Open']) * 100,
                'high': ((df_feat['High'].shift(-1) - df_feat['Open']) / df_feat['Open']) * 100,
                'low': ((df_feat['Low'].shift(-1) - df_feat['Open']) / df_feat['Open']) * 100,
            }
            for key in y_targets_full:
                y = y_targets_full[key]
                temp_df = pd.concat([X_full, y.rename('target')], axis=1).dropna()
                X_clean = temp_df[features]
                y_clean = temp_df['target']

                if X_clean.empty:
                    print(f"❌ Not enough clean data for {ticker} - {key} to train model. Skipping.")
                    continue

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_clean)

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_scaled, y_clean)

                y_pred = model.predict(X_scaled)
                mae = np.mean(np.abs(y_clean - y_pred))

                models[ticker][key] = model
                scalers[ticker][key] = scaler
                errors[ticker][key] = {"mae": float(mae)}
                print(f"✅ {ticker} - {key} MAE (from training data): {mae:.2f}")
            continue # Skip backtest if not enough data

        X = df_feat[features]
        y_targets = {
            'open': ((df_feat['Open'].shift(-1) - df_feat['Open']) / df_feat['Open']) * 100,
            'close': ((df_feat['Close'].shift(-1) - df_feat['Open']) / df_feat['Open']) * 100,
            'high': ((df_feat['High'].shift(-1) - df_feat['Open']) / df_feat['Open']) * 100,
            'low': ((df_feat['Low'].shift(-1) - df_feat['Open']) / df_feat['Open']) * 100,
        }

        models[ticker] = {}
        scalers[ticker] = {}
        errors[ticker] = {}
        backtest[ticker] = []

        for key in y_targets:
            y = y_targets[key]
            temp_df = pd.concat([X, y.rename('target')], axis=1).dropna()
            # Ensure X_clean and y_clean are aligned for training
            X_clean = temp_df[features]
            y_clean = temp_df['target']

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y_clean)

            y_pred = model.predict(X_scaled)
            mae = np.mean(np.abs(y_clean - y_pred))

            models[ticker][key] = model
            scalers[ticker][key] = scaler
            errors[ticker][key] = {"mae": float(mae)}

            print(f"✅ {ticker} - {key} MAE (from training data): {mae:.2f}")

        # --- Backtest Predictions (last 60 rows) ---
        # Ensure we have at least 60 days of data for backtesting
        backtest_data_raw = df_feat.tail(60).copy() 
        
        # Drop rows where the next day's actuals would be NaN (due to shift(-1))
        # This aligns the backtest data with available actuals for error calculation
        df_shifted_next_day = df.shift(-1)

        for idx in range(len(backtest_data_raw)):
            row = backtest_data_raw.iloc[[idx]]
            current_date = backtest_data_raw.index[idx]
            next_day_date = current_date + timedelta(days=1) # Placeholder for next trading day

            # Find the actual next trading day and its data
            actual_next_day_data = df_shifted_next_day.loc[current_date] if current_date in df_shifted_next_day.index else None

            if actual_next_day_data is None or actual_next_day_data.isnull().all():
                # If no next day data, or all NaN, skip this row for backtest
                # print(f"Skipping backtest for {current_date.strftime('%Y-%m-%d')} due to missing next day actuals.")
                continue

            row_date_str = current_date.strftime("%Y-%m-%d")
            
            try:
                # Scale features for prediction using the correct scaler for each target
                preds = {}
                for key in models[ticker]:
                    # Ensure the input to transform has the correct features
                    input_features = row[features]
                    row_scaled_for_pred = scalers[ticker][key].transform(input_features)
                    
                    # Predict the percentage change
                    predicted_percentage_change = models[ticker][key].predict(row_scaled_for_pred)[0]
                    
                    # Calculate the predicted price
                    preds[key] = float(row['Open'].iloc[0]) * (1 + float(predicted_percentage_change) / 100)

                actuals = {
                    'open': float(actual_next_day_data['Open']),
                    'close': float(actual_next_day_data['Close']),
                    'high': float(actual_next_day_data['High']),
                    'low': float(actual_next_day_data['Low']),
                }
                
                errors_day = {
                    key: abs(preds[key] - actuals[key])
                    for key in preds
                }
                
                backtest[ticker].append({
                    "Date": row_date_str,
                    "Predictions": preds,
                    "Actuals": actuals,
                    "Errors": errors_day
                })
            except Exception as e:
                print(f"⚠️ Backtest failed on {row_date_str} for {ticker}: {e}")

    # Removed timestamp from filenames
    joblib.dump(models, "multi_stock_models.pkl")
    joblib.dump(scalers, "multi_stock_scalers.pkl")
    joblib.dump(errors, "multi_stock_errors.pkl")
    joblib.dump(backtest, "multi_stock_backtest.pkl")
    print("✅ Training complete. Models and backtest saved.")

if __name__ == "__main__":
    train_and_save_models()


import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- Configuration ---
MODEL_DIR = "."  # Directory where your models, scalers, errors, and backtest files are saved

# --- Evaluation Function ---
def evaluate_models(model_dir=MODEL_DIR):
    """
    Loads saved model errors and backtest data to display MAE and calculate
    a simple prediction accuracy.
    """
    print("🚀 Starting model evaluation...")

    # Find the most recent files
    list_of_files = os.listdir(model_dir)
    
    error_files = sorted([f for f in list_of_files if "multi_stock_errors_" in f and f.endswith(".pkl")], reverse=True)
    backtest_files = sorted([f for f in list_of_files if "multi_stock_backtest_" in f and f.endswith(".pkl")], reverse=True)

    if not error_files or not backtest_files:
        print("❌ No model error or backtest files found. Please ensure 'train_and_save_models()' has been run.")
        return

    latest_error_file = os.path.join(model_dir, error_files[0])
    latest_backtest_file = os.path.join(model_dir, backtest_files[0])

    print(f"Loading errors from: {latest_error_file}")
    print(f"Loading backtest from: {latest_backtest_file}")

    try:
        errors = joblib.load(latest_error_file)
        backtest_results = joblib.load(latest_backtest_file)
    except Exception as e:
        print(f"An error occurred while loading files: {e}")
        return

    print("\n--- Model Mean Absolute Errors (MAE) ---")
    for ticker, targets in errors.items():
        print(f"\n📈 Ticker: {ticker}")
        for target_type, error_values in targets.items():
            mae = error_values.get("mae", "N/A")
            print(f"  Target '{target_type}': MAE = {mae:.4f}")

    print("\n--- Backtest Accuracy ---")
    print("Accuracy is calculated as the percentage of predictions where the predicted value is within a certain threshold (e.g., 2%) of the actual value.")

    # Define a threshold for "accuracy" - you can adjust this
    accuracy_threshold_percentage = 2.0 

    for ticker, daily_results in backtest_results.items():
        print(f"\n📊 Ticker: {ticker}")
        total_predictions = 0
        correct_predictions_within_threshold = 0

        for day_data in daily_results:
            predictions = day_data.get("Predictions", {})
            actuals = day_data.get("Actuals", {})
            
            for key, predicted_value in predictions.items():
                actual_value = actuals.get(key)

                if predicted_value is not None and actual_value is not None:
                    total_predictions += 1
                    # Calculate the percentage difference
                    if actual_value != 0: # Avoid division by zero
                        percentage_diff = (abs(predicted_value - actual_value) / actual_value) * 100
                        if percentage_diff <= accuracy_threshold_percentage:
                            correct_predictions_within_threshold += 1
                    elif predicted_value == 0 and actual_value == 0: # Both are zero, consider it accurate
                         correct_predictions_within_threshold += 1

        accuracy = (correct_predictions_within_threshold / total_predictions * 100) if total_predictions > 0 else 0
        print(f"  Overall Backtest Accuracy (within {accuracy_threshold_percentage:.1f}%): {accuracy:.2f}% ({correct_predictions_within_threshold}/{total_predictions} correct)")

    print("\n✅ Evaluation complete.")

if __name__ == "__main__":
    evaluate_models()

import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- Configuration ---
MODEL_DIR = "."  # Directory where your models, scalers, errors, and backtest files are saved

# --- Evaluation Function ---
def evaluate_models(model_dir=MODEL_DIR):
    """
    Loads saved model errors and backtest data to display MAE and calculate
    a simple prediction accuracy.
    """
    print("🚀 Starting model evaluation...")

    # Find the most recent files
    list_of_files = os.listdir(model_dir)
    
    error_files = sorted([f for f in list_of_files if "multi_stock_errors" in f and f.endswith(".pkl")], reverse=True)
    backtest_files = sorted([f for f in list_of_files if "multi_stock_backtest" in f and f.endswith(".pkl")], reverse=True)

    if not error_files or not backtest_files:
        print("❌ No model error or backtest files found. Please ensure 'train_and_save_models()' has been run.")
        return

    latest_error_file = os.path.join(model_dir, error_files[0])
    latest_backtest_file = os.path.join(model_dir, backtest_files[0])

    print(f"Loading errors from: {latest_error_file}")
    print(f"Loading backtest from: {latest_backtest_file}")

    try:
        errors = joblib.load(latest_error_file)
        backtest_results = joblib.load(latest_backtest_file)
    except Exception as e:
        print(f"An error occurred while loading files: {e}")
        return

    print("\n--- Model Mean Absolute Errors (MAE) from Saved Errors File ---")
    for ticker, targets in errors.items():
        print(f"\n📈 Ticker: {ticker}")
        for target_type, error_values in targets.items():
            mae = error_values.get("mae", "N/A")
            # Ensure MAE is a number before formatting
            if isinstance(mae, (int, float)):
                print(f"  Target '{target_type}': MAE = {mae:.4f}")
            else:
                print(f"  Target '{target_type}': MAE = {mae}")

    print("\n--- True Mean Absolute Errors (MAE) from Backtest Results ---")
    print("This MAE is calculated directly from the predictions and actuals in the backtest data.")
    
    # Store true MAE values for later use in accuracy calculation
    true_maes_calculated = {} 

    for ticker, daily_results in backtest_results.items():
        print(f"\n📈 Ticker: {ticker}")
        
        # Dictionary to store sum of absolute errors and count for each target type
        target_abs_errors = {}
        target_counts = {}

        for day_data in daily_results:
            predictions = day_data.get("Predictions", {})
            actuals = day_data.get("Actuals", {})
            
            for key, predicted_value in predictions.items():
                actual_value = actuals.get(key)

                if predicted_value is not None and actual_value is not None:
                    abs_error = abs(predicted_value - actual_value)
                    
                    target_abs_errors[key] = target_abs_errors.get(key, 0.0) + abs_error
                    target_counts[key] = target_counts.get(key, 0) + 1
        
        if not target_abs_errors:
            print("  No valid predictions/actuals found for MAE calculation.")
            continue

        true_maes_calculated[ticker] = {} # Initialize for the current ticker
        for target_type, total_abs_error in target_abs_errors.items():
            count = target_counts.get(target_type, 0)
            if count > 0:
                true_mae = total_abs_error / count
                true_maes_calculated[ticker][target_type] = true_mae # Store the calculated MAE
                print(f"  Target '{target_type}': True MAE = {true_mae:.4f}")
            else:
                print(f"  Target '{target_type}': No predictions found for True MAE calculation.")


    print("\n--- Backtest Accuracy ---")
    # Changed accuracy threshold to 1.0% as requested
    accuracy_threshold_percentage = 1.0 
    print(f"Accuracy is calculated as the percentage of predictions where the predicted value is within a certain threshold (e.g., {accuracy_threshold_percentage:.1f}%) of the actual value.")

    for ticker, daily_results in backtest_results.items():
        print(f"\n📊 Ticker: {ticker}")
        total_predictions = 0
        correct_predictions_within_threshold = 0

        for day_data in daily_results:
            predictions = day_data.get("Predictions", {})
            actuals = day_data.get("Actuals", {})
            
            for key, predicted_value in predictions.items():
                actual_value = actuals.get(key)

                if predicted_value is not None and actual_value is not None:
                    total_predictions += 1
                    # Calculate the percentage difference
                    if actual_value != 0: # Avoid division by zero
                        percentage_diff = (abs(predicted_value - actual_value) / actual_value) * 100
                        if percentage_diff <= accuracy_threshold_percentage:
                            correct_predictions_within_threshold += 1
                    elif predicted_value == 0 and actual_value == 0: # Both are zero, consider it accurate
                         correct_predictions_within_threshold += 1

        accuracy = (correct_predictions_within_threshold / total_predictions * 100) if total_predictions > 0 else 0
        print(f"  Overall Backtest Accuracy (within {accuracy_threshold_percentage:.1f}%): {accuracy:.2f}% ({correct_predictions_within_threshold}/{total_predictions} correct)")

    print("\n--- Backtest Accuracy within True MAE Range ---")
    print("This accuracy checks if the actual value falls within [predicted_value - True MAE, predicted_value + True MAE].")

    for ticker, daily_results in backtest_results.items():
        print(f"\n📊 Ticker: {ticker}")
        total_predictions_mae_range = 0
        correct_predictions_mae_range = 0

        if ticker not in true_maes_calculated:
            print(f"  True MAE data not found for {ticker}. Skipping this accuracy calculation.")
            continue

        for day_data in daily_results:
            predictions = day_data.get("Predictions", {})
            actuals = day_data.get("Actuals", {})
            
            for key, predicted_value in predictions.items():
                actual_value = actuals.get(key)

                if predicted_value is not None and actual_value is not None:
                    if key in true_maes_calculated[ticker]:
                        mae_for_target = true_maes_calculated[ticker][key]
                        lower_bound = predicted_value - mae_for_target
                        upper_bound = predicted_value + mae_for_target

                        total_predictions_mae_range += 1
                        if lower_bound <= actual_value <= upper_bound:
                            correct_predictions_mae_range += 1
                    else:
                        print(f"  Warning: True MAE for {ticker} - {key} not found. Skipping this prediction.")

        accuracy_mae_range = (correct_predictions_mae_range / total_predictions_mae_range * 100) if total_predictions_mae_range > 0 else 0
        print(f"  Overall Backtest Accuracy (within True MAE range): {accuracy_mae_range:.2f}% ({correct_predictions_mae_range}/{total_predictions_mae_range} correct)")

    print("\n✅ Evaluation complete.")

if __name__ == "__main__":
    evaluate_models()




