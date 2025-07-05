import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from pandas import Timestamp
import glob
import os
import gdown # Import gdown library

# --- App Config and Style (MUST BE AT THE VERY TOP) ---
st.set_page_config(page_title="Stock Prediction App", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    /* General body and app styling */
    body, .stApp {
        background-color: #121212; /* Dark background */
        color: #FFFFFF; /* White text */
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #00acc1; /* Cyan color for main headers */
    }
    
    /* Sidebar styling */
    .stSidebar {
        background-color: #1e1e1e; /* Slightly lighter dark for sidebar */
        color: #FFFFFF;
        padding-top: 20px; /* Add some padding at the top */
    }
    .stSidebar h2 {
        color: #FFFFFF; /* White color for sidebar header */
    }

    /* Metric labels (if used) */
    .metric-label { 
        font-weight: bold; 
    }
    
    /* Highlight for prediction results */
    .prediction-highlight {
        background-color: #1e1e1e; /* Slightly lighter dark */
        border-left: 5px solid #00acc1; /* Cyan border on left */
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3); /* More prominent shadow */
        height: 100%; /* Ensure equal height in columns */
    }

    /* Prediction value colors based on implied change from current price (simple assumption) */
    .positive-pred { color: #4CAF50; font-weight: bold; } /* Green for positive */
    .negative-pred { color: #F44336; font-weight: bold; } /* Red for negative */
    .neutral-pred { color: #FFFFFF; } /* White for neutral/no significant change */

    /* Strategy insights bullet points */
    .prediction-highlight ul {
        list-style: none; /* Remove default bullet points */
        padding-left: 0;
    }
    .prediction-highlight li {
        margin-bottom: 8px;
    }

    /* Styling for accuracy blocks */
    .accuracy-block {
        padding: 10px 15px;
        margin: 5px;
        border-radius: 6px;
        color: white;
        font-weight: bold;
        display: inline-block;
        text-align: center;
        min-width: 120px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .accuracy-green { background-color: #2e7d32; } /* Dark green */
    .accuracy-red { background-color: #c62828; } /* Dark red */
    .accuracy-blue { background-color: #007bb6; } /* Blue for general MAE display */

    /* Streamlit widgets styling */
    .stSelectbox > div > div > div, .stTextInput > div > div > input, .stRadio > label, .stCheckbox > label {
        background-color: #282828;
        color: #FFFFFF;
        border: 1px solid #444444; /* Darker border */
        border-radius: 5px;
        padding: 8px 12px; /* Add padding to inputs/selects */
    }
    .stButton > button {
        background-color: #00acc1;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.2s; /* Smooth transition on hover */
    }
    .stButton > button:hover {
        background-color: #008fa7;
        color: white;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        color: #FFFFFF;
        background-color: #1e1e1e;
        border-radius: 8px; /* Rounded corners for dataframe */
        overflow: hidden; /* Ensures content stays within rounded corners */
        box-shadow: 0 4px 8px rgba(0,0,0,0.3); /* Added shadow for all dataframes */
    }
    /* Target cells within the dataframe for styling */
    .stDataFrame table td, .stDataFrame table th { /* Added th here */
        vertical-align: middle; /* Align content vertically in the middle */
        text-align: center; /* Center text horizontally */
    }
    .stDataFrame table { /* Target the table inside the dataframe */
        color: #FFFFFF !important; /* Ensure text color within table */
    }
    .stDataFrame .css-1dp5vir { /* Header background */
        background-color: #333333;
        color: #FFFFFF;
    }
    .stDataFrame .css-1l02zmt { /* Header text color */
        color: #FFFFFF;
    }
    
    /* Specific styling for the 'Last 5 Days + Prediction' table */
    #last-5-days-table .stDataFrame {
        font-size: 1.1em; /* Increased font size for this specific table */
    }
    /* Highlight the last row (predicted row) specifically */
    #last-5-days-table .stDataFrame tbody tr:last-child {
        background-color: #2a3a4a !important; /* Dark blue/grey background for prediction */
        font-weight: bold; /* Make predicted row bold */
        color: #E0E0E0 !important; /* Slightly lighter text for predicted row */
        border-top: 2px solid #00acc1; /* Cyan line above predicted row */
    }
    #last-5-days-table .stDataFrame thead th {
        font-size: 1.2em; /* Larger header font for this table */
    }
    
</style>
""", unsafe_allow_html=True)


# --- Google Drive File ID for the large models file ---
# IMPORTANT: Make sure this file is publicly accessible (Anyone with the link -> Viewer)
MODELS_FILE_ID = "1tR8pR4miv5Gfbvd5bxurBdIshUvhkRBp" 

# --- Function to download files from Google Drive ---
@st.cache_resource(ttl=3600) # Cache the download to avoid re-downloading on every rerun
def download_from_google_drive(file_id, output_path):
    """Downloads a file from Google Drive given its file ID."""
    if not os.path.exists(output_path):
        try:
            st.info(f"Downloading {output_path} from Google Drive...")
            gdown.download(id=file_id, output=output_path, quiet=False)
            st.success(f"Successfully downloaded {output_path}.")
        except Exception as e:
            st.error(f"❌ Error downloading {output_path} from Google Drive: {e}. Please check the file ID and sharing permissions.")
            st.stop()
    else:
        st.info(f"{output_path} already exists. Skipping download.")

# --- Ensure models file is downloaded before loading ---
download_from_google_drive(MODELS_FILE_ID, "multi_stock_models.pkl")

# --- Define paths for all files ---
# multi_stock_models.pkl will be downloaded
models_file = "multi_stock_models.pkl"
# Other files are expected to be in the same Git repository
scalers_file = "multi_stock_scalers.pkl"
errors_file = "multi_stock_errors.pkl"
backtest_file = "multi_stock_backtest.pkl" 

# --- Check if all necessary files exist (downloaded or from Git) ---
if not (os.path.exists(models_file) and os.path.exists(scalers_file) and os.path.exists(errors_file)):
    st.error("❌ Required model/scaler/error files not found. Ensure 'multi_stock_models.pkl' is downloadable from Google Drive and others are committed to your Git repository.")
    st.stop()

try:
    models = joblib.load(models_file)
    scalers = joblib.load(scalers_file)
    errors = joblib.load(errors_file)
    backtest = joblib.load(backtest_file) if os.path.exists(backtest_file) else {}
except Exception as e:
    st.error(f"❌ Error loading model/data files: {e}. Ensure all files are valid pickle files and accessible. If running on Streamlit Cloud, ensure 'scikit-learn' is in your requirements.txt!")
    st.stop()

# Initialize session state for mode if not already set
if "simulate_profit_mode" not in st.session_state:
    st.session_state.simulate_profit_mode = False
if "show_monthly_min_margin" not in st.session_state: # New session state for monthly aggregation
    st.session_state.show_monthly_min_margin = False

# --- Sidebar buttons to toggle modes ---
if st.sidebar.button("💰 Toggle 60-Day Profit Simulation"):
    st.session_state.simulate_profit_mode = not st.session_state.simulate_profit_mode
    st.session_state.show_monthly_min_margin = False # Turn off other mode
if st.sidebar.button("📊 Show Monthly Min Margins"): # New button
    st.session_state.show_monthly_min_margin = not st.session_state.show_monthly_min_margin
    st.session_state.simulate_profit_mode = False # Turn off other mode

# --- Main App Logic based on mode ---
if st.session_state.simulate_profit_mode:
    st.title("📊 Smart Min-Margin Backtest (Auto Stock Picker)")
    
    # Ensure backtest data is loaded and available for simulation
    if not backtest:
        st.warning("No backtest data loaded to run the simulation. Please ensure 'multi_stock_backtest.pkl' is correctly loaded.")
        st.stop()

    invest_thousands = st.number_input("Daily Investment (in ₹000)", min_value=1, step=1)
    invest_amount = invest_thousands * 1000

    # --- Initialize ---
    total_profit = 0
    win_days = 0
    loss_days = 0
    flat_days = 0
    records = []

    # --- Collect All Backtest Data by Date ---
    data_by_date = {}
    for ticker, data in backtest.items():
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.tail(60)
        for _, row in df.iterrows():
            date = row['Date'].strftime('%Y-%m-%d')
            if date not in data_by_date:
                data_by_date[date] = []
            data_by_date[date].append({
                'ticker': ticker,
                'actual': row['Actuals'],
                'pred': row['Predictions'],
                'mae': errors.get(ticker, {})
            })

    # --- Loop Over Each Day ---
    for date, stocks in sorted(data_by_date.items()):
        best_margin = -1
        selected = None
        for stock in stocks:
            pred = stock['pred']
            mae = stock['mae']
            if not pred or not mae:
                continue
            try:
                high = pred['high']
                low = pred['low']
                mae_high = mae['high']['mae']
                mae_low = mae['low']['mae']
                margin = (high - mae_high) - (low + mae_low)
                if margin > best_margin:
                    best_margin = margin
                    selected = {
                        'ticker': stock['ticker'],
                        'actual': stock['actual'],
                        'pred': stock['pred'],
                        'mae_high': mae_high,
                        'mae_low': mae_low
                    }
            except:
                continue

        if not selected:
            continue

        act = selected['actual']
        pred = selected['pred']
        mae_high = selected['mae_high']
        mae_low = selected['mae_low']
        ticker = selected['ticker']

        entry_price = pred['low'] + mae_low
        stoploss = pred['low'] 
        target = pred['high'] - mae_high

        result = "No Entry"
        profit = 0
        exit_price = 0 # Initialize exit_price

        if act['low'] <= entry_price:  # Entry triggered when actual low touches entry
            if act['high'] >= target:
                exit_price = target
                result = "Profit"
                win_days += 1
            elif act['low'] <= stoploss:
                exit_price = stoploss
                result = "Loss"
                loss_days += 1
            else:
                exit_price = act['close']
                result = "Flat"
                flat_days += 1

            pct = (exit_price - entry_price) / entry_price
            profit = pct * invest_amount

            records.append({
                'Date': date,
                'Stock': ticker,
                'Entry (Low + MAE)': round(entry_price, 2),
                'Stoploss (Low)': round(stoploss, 2),
                'Target (High - MAE)': round(target, 2),
                'Actual High': act['high'],
                'Actual Low': act['low'],
                'Exit Price': round(exit_price, 2),
                'Result': result,
                'P/L (₹)': round(profit, 2)
            })

            total_profit += profit

    # --- Display Results ---
    if records:
        df_results = pd.DataFrame(records)
        st.subheader("📅 Daily Trades")
        st.dataframe(df_results, use_container_width=True)

        st.subheader("📈 Summary")
        st.success(f"Total Profit/Loss: ₹{total_profit:,.2f}")
        st.info(f"Profitable Days: {win_days} | Loss Days: {loss_days} | Flat Days: {flat_days} | Total Trades: {len(records)}")
    else:
        st.warning("No valid trades found in backtest data for the selected criteria.")

    st.stop() # Stop rendering the rest of the script if in simulation mode

elif st.session_state.show_monthly_min_margin: # New block for monthly min margin aggregation
    st.title("📊 Monthly Min-Margin Aggregation")

    if not backtest:
        st.warning("No backtest data loaded to calculate monthly min margins. Please ensure 'multi_stock_backtest.pkl' is correctly loaded.")
        st.stop()

    monthly_margins = []

    # Collect all backtest data by date, similar to the simulation logic
    data_by_date = {}
    for ticker, data in backtest.items():
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.tail(60) # Consider last 60 days for aggregation
        for _, row in df.iterrows():
            date = row['Date'].strftime('%Y-%m-%d')
            if date not in data_by_date:
                data_by_date[date] = []
            data_by_date[date].append({
                'ticker': ticker,
                'actual': row['Actuals'],
                'pred': row['Predictions'],
                'mae': errors.get(ticker, {})
            })

    daily_min_margins_data = []
    for date_str, stocks in sorted(data_by_date.items()):
        best_margin_for_day = -float('inf') # Initialize with negative infinity
        selected_stock_for_day = None

        for stock in stocks:
            pred = stock['pred']
            mae = stock['mae']
            if not pred or 'high' not in pred or 'low' not in pred or \
               not mae or 'high' not in mae or 'low' not in mae or \
               'mae' not in mae['high'] or 'mae' not in mae['low']:
                continue
            try:
                high_pred = pred['high']
                low_pred = pred['low']
                mae_high = mae['high']['mae']
                mae_low = mae['low']['mae']
                
                # Calculate margin using the exact logic from simulation
                margin = (high_pred - mae_high) - (low_pred + mae_low)
                
                if margin > best_margin_for_day:
                    best_margin_for_day = margin
                    selected_stock_for_day = stock
            except Exception as e:
                # print(f"Error processing stock {stock.get('ticker')} on {date_str}: {e}") # For debugging
                continue

        if selected_stock_for_day:
            pred_s = selected_stock_for_day['pred']
            mae_s = selected_stock_for_day['mae']

            # Re-calculate min_margin_pct using the selected stock's data and MAE
            # Ensure MAE values are extracted correctly from the 'mae' dictionary
            current_mae_high = mae_s['high']['mae'] if 'high' in mae_s and 'mae' in mae_s['high'] else 0.0
            current_mae_low = mae_s['low']['mae'] if 'low' in mae_s and 'mae' in mae_s['low'] else 0.0

            target_val = pred_s['high'] - current_mae_high
            stop_loss_val = pred_s['low'] + current_mae_low # This is the correct stoploss for margin calculation
            
            # Use actual open from the selected stock's actual data for percentage base
            actual_open_for_pct = selected_stock_for_day['actual']['open']

            if actual_open_for_pct != 0:
                min_margin_pct_val = ((target_val - stop_loss_val) / actual_open_for_pct) * 100
            else:
                min_margin_pct_val = 0.0

            daily_min_margins_data.append({
                'Date': datetime.strptime(date_str, '%Y-%m-%d'),
                'Min Margin %': min_margin_pct_val
            })
    
    if daily_min_margins_data:
        daily_df = pd.DataFrame(daily_min_margins_data)
        daily_df['Month'] = daily_df['Date'].dt.to_period('M')

        # Changed from .mean() to .sum() for aggregation
        monthly_total_margins = daily_df.groupby('Month')['Min Margin %'].sum().reset_index()
        monthly_total_margins['Month'] = monthly_total_margins['Month'].astype(str) # Convert Period to string for display
        monthly_total_margins.rename(columns={'Min Margin %': 'Total Min Margin %'}, inplace=True) # Renamed column

        st.subheader("📈 Monthly Total Min Margin % (Last 60 Days)") # Updated subheader
        st.dataframe(monthly_total_margins.style.format({'Total Min Margin %': "{:.2f}%"}), use_container_width=True) # Updated format
    else:
        st.warning("No daily min margins could be calculated for the last 60 days to aggregate monthly.")

    st.stop() # Stop rendering the rest of the script if in this mode

else:
    # --- Original App Code Starts Here ---
    
    # --- Sidebar Inputs ---
    st.sidebar.header("🔧 Configuration")
    tickers = {
        'TATAMOTORS.NS': 'Tata Motors',
        'UNITDSPR.NS': 'United Spirits',
        'PETRONET.NS': 'Petronet LNG',
        'COLPAL.NS': 'Colgate-Palmolive',
        'BEL.NS': 'Bharat Electronics'
    }
    selected_ticker = st.sidebar.selectbox("Choose a stock:", list(tickers.keys()), format_func=lambda x: tickers[x])
    manual_open = st.sidebar.text_input("**(Optional) Enter today's Open price (₹):**", value="")
    # Removed chart-related sidebar options
    # date_range = st.sidebar.selectbox("**Select Date Range for Chart:**", ["1M", "3M", "6M", "1Y", "5Y"], index=4)
    # chart_type = st.sidebar.radio("**Select Chart Type:**", options=["Line Chart", "Candlestick"], index=0)
    # show_chart = st.sidebar.checkbox("📊 Show Price Chart", value=False)
    # Checkbox to control display of the backtest table
    backtest_mode = st.sidebar.checkbox("📈 Show Backtest Results (Last 60 Days)", value=False) 
    # Checkbox for showing original MAE
    show_original_mae = st.sidebar.checkbox("ℹ️ Show Original Model MAE (from training)", value=False)

    # --- Fetch Data ---
    @st.cache_data(ttl=3600)
    def get_data(ticker, period_key="1y"): # Changed to default to 1 year period for data fetching
        end_date = datetime.now()
        # Using a fixed period for data fetching as chart date range is removed
        start_date = end_date - timedelta(days=365) # Fetch 1 year of data by default
        
        try:
            data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
            if data.empty:
                st.warning(f"⚠️ No data fetched for {ticker} for the selected range. Check ticker symbol or date range.")
            return data
        except Exception as e:
            st.error(f"❌ Error fetching data for {ticker}: {e}. Please check your internet connection or the ticker symbol.")
            return pd.DataFrame()

    def create_features(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df_feat = df.copy()
        
        df_feat['SMA_10'] = df_feat['Close'].rolling(window=10).mean()
        df_feat['SMA_30'] = df_feat['Close'].rolling(window=30).mean()
        df_feat['Price_Change'] = df_feat['Close'].diff()
        df_feat['High_Low_Diff'] = df_feat['High'] - df_feat['Low']
        df_feat['Volume_Change'] = df_feat['Volume'].diff()
        df_feat['Close_Shift_1'] = df_feat['Close'].shift(1)

        df_feat.dropna(inplace=True)
        return df_feat

    # --- Prediction Logic ---
    with st.spinner("🔄 Fetching data and generating predictions..."):
        # Call get_data without date_range, using default period
        df = get_data(selected_ticker) 

        if df.empty:
            st.stop()

        feat_df = create_features(df)
        if feat_df.empty:
            st.error("❌ Not enough historical data to generate features for prediction.")
            st.stop()

        last_data_date = feat_df.index[-1].date()
        future_day = last_data_date + timedelta(days=1)
        while future_day.weekday() >= 5 or pd.Timestamp(future_day) in feat_df.index:
            future_day += timedelta(days=1)

        latest = feat_df.loc[[feat_df.index[-1]]].copy()
        current_open_price_actual = float(latest['Open'].iloc[0])
        
        open_price_for_prediction = current_open_price_actual 
        if manual_open.strip():
            try:
                open_price_for_prediction = float(manual_open.strip())
            except ValueError:
                st.warning("⚠️ Invalid open price entered. Using actual last known open price for prediction base.")

        scaler_features = ['Open', 'High', 'Low', 'Close', 'SMA_10', 'SMA_30', 'Price_Change']
        features = latest[scaler_features].copy()
        features['Open'] = open_price_for_prediction

        if selected_ticker not in models or selected_ticker not in scalers:
            st.error(f"❌ Models or scalers not found for {tickers[selected_ticker]}. Please ensure training was successful for this stock.")
            st.stop()

        model_open = models[selected_ticker]['open']
        model_close = models[selected_ticker]['close']
        model_high = models[selected_ticker].get('high')
        model_low = models[selected_ticker].get('low')

        X_scaled_open = scalers[selected_ticker]['open'].transform(features)
        X_scaled_close = scalers[selected_ticker]['close'].transform(features)
        X_scaled_high = scalers[selected_ticker]['high'].transform(features) if 'high' in scalers[selected_ticker] else None
        X_scaled_low = scalers[selected_ticker]['low'].transform(features) if 'low' in scalers[selected_ticker] else None

        raw_open = model_open.predict(X_scaled_open)[0].item()
        raw_close = model_close.predict(X_scaled_close)[0].item()
        raw_high = model_high.predict(X_scaled_high)[0].item() if model_high and X_scaled_high is not None else 0.0
        raw_low = model_low.predict(X_scaled_low)[0].item() if model_low and X_scaled_low is not None else 0.0

        pred_open_price = open_price_for_prediction * (1 + raw_open / 100)
        pred_close_price = open_price_for_prediction * (1 + raw_close / 100)
        pred_high_price = open_price_for_prediction * (1 + raw_high / 100)
        pred_low_price = open_price_for_prediction * (1 + raw_low / 100)

        # --- Initialize MAE values for display (DEFAULT to Backtest MAE for last 60 days) ---
        display_mae_open = 0.0
        display_mae_close = 0.0
        display_mae_high = 0.0
        display_mae_low = 0.0

        # Attempt to load and calculate MAE from last 60 days of backtest data
        if selected_ticker in backtest and backtest[selected_ticker]:
            try:
                bt_data_for_mae_calc = pd.DataFrame(backtest[selected_ticker])
                bt_data_for_mae_calc['Date'] = pd.to_datetime(bt_data_for_mae_calc['Date'])
                
                # Filter for the last 60 days of backtest data for MAE calculation
                end_date_bt = bt_data_for_mae_calc['Date'].max()
                start_date_bt = end_date_bt - timedelta(days=60)
                bt_data_for_mae_calc = bt_data_for_mae_calc[bt_data_for_mae_calc['Date'] >= start_date_bt]

                if not bt_data_for_mae_calc.empty:
                    temp_err_df = bt_data_for_mae_calc['Errors'].apply(pd.Series)
                    expected_inner_cols_err = ['open', 'close', 'high', 'low']
                    
                    if all(col in temp_err_df.columns for col in expected_inner_cols_err):
                        if not temp_err_df['open'].dropna().empty:
                            display_mae_open = temp_err_df['open'].abs().mean()
                        if not temp_err_df['close'].dropna().empty:
                            display_mae_close = temp_err_df['close'].abs().mean()
                        if not temp_err_df['high'].dropna().empty:
                            display_mae_high = temp_err_df['high'].abs().mean()
                        if not temp_err_df['low'].dropna().empty:
                            display_mae_low = temp_err_df['low'].abs().mean()
                        st.info(f"Defaulting to Backtest MAE (last 60 days) for display (e.g., Open MAE: ₹{display_mae_open:.2f}).")
                    else:
                        st.warning("⚠️ Backtest error data structure mismatch for MAE calculation. Defaulting to original model MAE.")
                        # Fallback to original model MAE if backtest structure is bad
                        err_original = errors[selected_ticker]
                        display_mae_open = err_original['open']['mae']
                        display_mae_close = err_original['close']['mae']
                        display_mae_high = err_original['high']['mae'] if 'high' in err_original else 0.0
                        display_mae_low = err_original['low']['mae'] if 'low' in err_original else 0.0
                else:
                    st.warning("⚠️ Backtest data is empty for the last 60 days. Defaulting to original model MAE.")
                    # Fallback to original model MAE if backtest data is empty
                    err_original = errors[selected_ticker]
                    display_mae_open = err_original['open']['mae']
                    display_mae_close = err_original['close']['mae']
                    display_mae_high = err_original['high']['mae'] if 'high' in err_original else 0.0
                    display_mae_low = err_original['low']['mae'] if 'low' in err_original else 0.0
            except Exception as e:
                st.warning(f"Could not calculate backtest MAE for display due to an error: {e}. Defaulting to original model MAE.")
                # Fallback to original model MAE if an error occurs during backtest MAE calculation
                err_original = errors[selected_ticker]
                display_mae_open = err_original['open']['mae']
                display_mae_close = err_original['close']['mae']
                display_mae_high = err_original['high']['mae'] if 'high' in err_original else 0.0
                display_mae_low = err_original['low']['mae'] if 'low' in err_original else 0.0
        else:
            st.warning("⚠️ No backtest data found for selected stock. Defaulting to original model MAE.")
            # Fallback to original model MAE if no backtest data at all
            err_original = errors[selected_ticker]
            display_mae_open = err_original['open']['mae']
            display_mae_close = err_original['close']['mae']
            display_mae_high = err_original['high']['mae'] if 'high' in err_original else 0.0
            display_mae_low = err_original['low']['mae'] if 'low' in err_original else 0.0

        target = pred_high_price - display_mae_high
        stop_loss = pred_low_price + display_mae_low
        margin_range = target - stop_loss
        optimal_margin_abs = pred_high_price - pred_low_price
        optimal_margin_pct = (optimal_margin_abs / pred_open_price) * 100 if pred_open_price else 0
        min_margin_pct = (margin_range / pred_open_price) * 100 if pred_open_price else 0

        st.title(f"📈 {tickers[selected_ticker]} Forecast for {future_day.strftime('%Y-%m-%d')}")
        
        def get_price_color_class(current_price, predicted_price, threshold_pct=0.1): # threshold in percentage
            if predicted_price > current_price * (1 + threshold_pct / 100):
                return "positive-pred"
            elif predicted_price < current_price * (1 - threshold_pct / 100):
                return "negative-pred"
            else:
                return "neutral-pred"

        open_color = get_price_color_class(current_open_price_actual, pred_open_price)
        close_color = get_price_color_class(current_open_price_actual, pred_close_price)
        high_color = get_price_color_class(current_open_price_actual, pred_high_price)
        low_color = get_price_color_class(current_open_price_actual, pred_low_price)

        # --- Split Layout for Predictions and Strategy ---
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="prediction-highlight">
                <h3>🎯 Predictions:</h3>
                <ul>
                    <li><b>Predicted Open:</b> <span class="{open_color}">₹{pred_open_price:.2f}</span> ± ₹{display_mae_open:.2f}</li>
                    <li><b>Predicted Close:</b> <span class="{close_color}">₹{pred_close_price:.2f}</span> ± ₹{display_mae_close:.2f}</li>
                    <li><b>Predicted High:</b> <span class="{high_color}">₹{pred_high_price:.2f}</span> ± ₹{display_mae_high:.2f}</li>
                    <li><b>Predicted Low:</b> <span class="{low_color}">₹{pred_low_price:.2f}</span> ± ₹{display_mae_low:.2f}</li>
                </ul>
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="prediction-highlight">
                <h4>📌 Strategy Insights:</h4>
                <ul>
                    <li><b>Target:</b> ₹{target:.2f}</li>
                    <li><b>Stop Loss:</b> ₹{stop_loss:.2f}</li>
                    <li><b>Optimal Margin:</b> ₹{optimal_margin_abs:.2f} ({optimal_margin_pct:.2f}%)</li>
                    <li><b>Min Margin:</b> ₹{min_margin_pct:.2f}%)</li>
                </ul>
            </div>""", unsafe_allow_html=True)


        # --- Last 5 Trading Days + Next Predicted Day Table ---
        st.subheader(f"🗓️ Historical & Predicted Prices")
        
        display_df = df[['Open', 'High', 'Low', 'Close']].copy().tail(5)
        display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d')
        future_day_str = future_day.strftime('%Y-%m-%d')
        
        # Append the predicted day's data with a special index for highlighting
        predicted_row = pd.DataFrame([[pred_open_price, pred_high_price, pred_low_price, pred_close_price]], 
                                     index=[f"Predicted ({future_day_str})"], columns=display_df.columns)
        display_df = pd.concat([display_df, predicted_row])
        
        # Use an empty container with a specific ID for custom CSS targeting
        st.markdown('<div id="last-5-days-table">', unsafe_allow_html=True)
        st.dataframe(display_df.style.format("₹{:.2f}"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


        # --- Display Original Model MAE (Optional) ---
        if show_original_mae:
            st.subheader("ℹ️ Original Model Accuracy (Percentage & MAE from training data)")
            original_err = errors[selected_ticker]

            # Calculate percentage accuracy based on predicted price and the original MAE
            original_acc_high_pct = (100 - (original_err['high']['mae'] / pred_high_price * 100)) if pred_high_price else 0
            original_acc_low_pct = (100 - (original_err['low']['mae'] / pred_low_price * 100)) if pred_low_price else 0
            original_acc_open_pct = (100 - (original_err['open']['mae'] / pred_open_price * 100)) if pred_open_price else 0
            original_acc_close_pct = (100 - (original_err['close']['mae'] / pred_close_price * 100)) if pred_close_price else 0

            col_acc_top1, col_acc_top2 = st.columns(2)
            col_acc_bottom1, col_acc_bottom2 = st.columns(2)

            with col_acc_top1:
                st.markdown(f"""
                <div style="padding: 10px; background-color: #1f1f1f; border-radius: 8px; text-align: center;">
                    <div class="accuracy-block accuracy-green"><b>High:</b><br>₹{original_err['high']['mae'] if 'high' in original_err else 0.0:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_acc_top2:
                st.markdown(f"""
                <div style="padding: 10px; background-color: #1f1f1f; border-radius: 8px; text-align: center;">
                    <div class="accuracy-block accuracy-red"><b>Low:</b><br>₹{original_err['low']['mae'] if 'low' in original_err else 0.0:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_acc_bottom1:
                st.markdown(f"""
                <div style="padding: 10px; background-color: #1f1f1f; border-radius: 8px; text-align: center;">
                    <div class="accuracy-block accuracy-green"><b>Open:</b><br>₹{original_err['open']['mae']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_acc_bottom2:
                st.markdown(f"""
                <div style="padding: 10px; background-color: #1f1f1f; border-radius: 8px; text-align: center;">
                    <div class="accuracy-block accuracy-red"><b>Close:</b><br>₹{original_err['close']['mae']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)


        if backtest_mode and selected_ticker in backtest:
            st.subheader("🔢 Backtest Results (Last 60 Days)") # Changed title to reflect 60 days
            
            # Load backtest data into a DataFrame
            bt = pd.DataFrame(backtest[selected_ticker])

            # Convert 'Date' column to datetime objects
            bt['Date'] = pd.to_datetime(bt['Date'])

            # Filter for the last 60 days for display in the backtest table
            end_date_display = bt['Date'].max()
            start_date_display = end_date_display - timedelta(days=60)
            bt_filtered_for_display = bt[bt['Date'] >= start_date_display].sort_values(by='Date', ascending=True) # Sort by date

            if bt_filtered_for_display.empty:
                st.warning(f"⚠️ Backtest data for {tickers[selected_ticker]} is empty for the last 60 days. No results to display.")
            else:
                # Check if 'Predictions', 'Actuals', 'Errors' columns exist and are dictionaries
                if all(col in bt_filtered_for_display.columns for col in ['Predictions', 'Actuals', 'Errors']) and \
                   isinstance(bt_filtered_for_display['Predictions'].iloc[0], dict): # Check first element type
                    
                    # Expand the dictionary columns into new Dataframes
                    pred_df = bt_filtered_for_display['Predictions'].apply(pd.Series)
                    act_df = bt_filtered_for_display['Actuals'].apply(pd.Series)
                    err_df = bt_filtered_for_display['Errors'].apply(pd.Series)

                    # Ensure the expanded DataFrames have the expected columns (open, close, high, low)
                    expected_inner_cols = ['open', 'close', 'high', 'low']
                    if not all(col in pred_df.columns for col in expected_inner_cols) or \
                       not all(col in act_df.columns for col in expected_inner_cols) or \
                       not all(col in err_df.columns for col in expected_inner_cols):
                        st.error("❌ Backtest data structure mismatch: Inner dictionaries (Predictions, Actuals, Errors) do not contain expected keys (open, close, high, low). Displaying raw data for inspection.")
                        st.dataframe(bt_filtered_for_display, use_container_width=True, height=700)
                    else:
                        # Rename columns for clarity in the combined DataFrame
                        pred_df.columns = [f'Pred_{col.capitalize()}' for col in expected_inner_cols]
                        act_df.columns = [f'Actual_{col.capitalize()}' for col in expected_inner_cols]
                        err_df.columns = [f'Error_{col.capitalize()}' for col in expected_inner_cols]

                        # Define the desired column order
                        ordered_cols = ['Date']
                        for metric in ['Open', 'Close', 'High', 'Low']:
                            ordered_cols.append(f'Actual_{metric}')
                            ordered_cols.append(f'Pred_{metric}')
                            ordered_cols.append(f'Error_{metric}')

                        # Combine all into a single DataFrame and reorder columns
                        combined_df = pd.concat([bt_filtered_for_display['Date'], act_df, pred_df, err_df], axis=1)[ordered_cols]

                        # Explicitly format Date column to string and round numeric columns
                        combined_df['Date'] = combined_df['Date'].dt.strftime('%Y-%m-%d')
                        numeric_cols_to_format = [col for col in combined_df.columns if col not in ['Date']]
                        combined_df[numeric_cols_to_format] = combined_df[numeric_cols_to_format].round(2)

                        # --- Apply conditional styling for backtest table errors ---
                        def highlight_backtest_errors(row):
                            # Initialize styles Series with empty strings for no style
                            styles = pd.Series('', index=row.index) 
                            for col_prefix in ['Open', 'Close', 'High', 'Low']:
                                error_col = f'Error_{col_prefix}'
                                actual_col = f'Actual_{col_prefix}'

                                # Ensure columns exist and values are not NaN for calculations
                                if error_col in row.index and pd.notna(row[error_col]) and \
                                   actual_col in row.index and pd.notna(row[actual_col]):
                                    
                                    error_value = abs(row[error_col])
                                    actual_value = row[actual_col]

                                    # Apply full CSS style string based on error logic
                                    if actual_value != 0: # Avoid division by zero
                                        if error_value >= 0.01 * actual_value: # Red if error is >= 1% of actual value
                                            styles[error_col] = 'background-color: #641E16; color: white; font-weight: bold;'  # Dark Red
                                        elif error_value >= 2.0: # Blue if error is >= ₹2.0 but < 1% of actual value
                                            styles[error_col] = 'background-color: #154360; color: white; font-weight: bold;'  # Dark Blue
                                        else: # Green if error is < ₹2.0
                                            styles[error_col] = 'background-color: #145A32; color: white; font-weight: bold;' # Dark Green
                                    else: # Handle case where actual_value is 0 (unlikely for stock prices, but for robustness)
                                        if error_value > 0: # Any error on 0 actual is considered large
                                            styles[error_col] = 'background-color: #641E16; color: white; font-weight: bold;' # Dark Red
                                        else: # Error is also 0
                                            styles[error_col] = 'background-color: #145A32; color: white; font-weight: bold;' # Dark Green (Perfect prediction on 0 actual)
                            return styles
                            
                        # Apply styling by returning full CSS style strings (axis=1 applies function row-wise)
                        styled_df = combined_df.style \
                            .apply(highlight_backtest_errors, axis=1) \
                            .format({col: "₹{}" for col in numeric_cols_to_format}) # Apply currency symbol after rounding
                        
                        st.dataframe(
                            styled_df,
                            use_container_width=True,
                            height=700
                        )

                        # --- Calculate and Display Backtest MAE for the backtest section ---
                        st.markdown("<br>", unsafe_allow_html=True) # Add some space
                        st.subheader("📊 Backtest Mean Absolute Error (Last 60 Days)")
                        
                        # Corrected: Use the RENAMED columns for MAE calculation
                        backtest_mae_open_display = err_df['Error_Open'].abs().mean() if not err_df['Error_Open'].dropna().empty else 0.0
                        backtest_mae_close_display = err_df['Error_Close'].abs().mean() if not err_df['Error_Close'].dropna().empty else 0.0
                        backtest_mae_high_display = err_df['Error_High'].abs().mean() if not err_df['Error_High'].dropna().empty else 0.0
                        backtest_mae_low_display = err_df['Error_Low'].abs().mean() if not err_df['Error_Low'].dropna().empty else 0.0

                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-around; padding: 10px; background-color: #1f1f1f; border-radius: 8px;">
                            <div class="accuracy-block accuracy-blue"><b>Open MAE:</b><br>₹{backtest_mae_open_display:.2f}</div>
                            <div class="accuracy-block accuracy-blue"><b>Close MAE:</b><br>₹{backtest_mae_close_display:.2f}</div>
                            <div class="accuracy-block accuracy-blue"><b>High MAE:</b><br>₹{backtest_mae_high_display:.2f}</div>
                            <div class="accuracy-block accuracy-blue"><b>Low MAE:</b><br>₹{backtest_mae_low_display:.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    # Fallback to displaying raw DataFrame if structure is not as expected for expansion
                    st.warning(f"⚠️ Backtest data for {tickers[selected_ticker]} has an unexpected structure. Displaying raw data.")
                    st.dataframe(bt_filtered_for_display, use_container_width=True, height=700)

        elif backtest_mode and selected_ticker not in backtest:
            st.warning(f"⚠️ No backtest data found for {tickers[selected_ticker]} for the last 60 days. Ensure a backtest file exists and contains data for this stock.")

    # --- New: Min Margin Analysis Across Stocks ---
    st.subheader("📊 Min Margin Analysis Across Stocks")
    all_stocks_min_margins = []
    
    # Define a helper function to get prediction data for a single ticker
    @st.cache_data(ttl=3600) # Cache this function as well for efficiency
    def get_prediction_data_for_ticker(ticker_symbol, _models, _scalers, _errors, _tickers_dict, _backtest): # Added _backtest
        df = get_data(ticker_symbol) # This is already cached
        if df.empty:
            return None, None # Return None for future_day too

        feat_df = create_features(df)
        if feat_df.empty:
            return None, None # Return None for future_day too

        last_data_date = feat_df.index[-1].date()
        future_day = last_data_date + timedelta(days=1)
        while future_day.weekday() >= 5 or pd.Timestamp(future_day) in feat_df.index:
            future_day += timedelta(days=1)


        latest = feat_df.loc[[feat_df.index[-1]]].copy()
        current_open_price_actual = float(latest['Open'].iloc[0])
        open_price_for_prediction = current_open_price_actual # Using actual for this calculation

        scaler_features = ['Open', 'High', 'Low', 'Close', 'SMA_10', 'SMA_30', 'Price_Change']
        features = latest[scaler_features].copy()
        features['Open'] = open_price_for_prediction

        if ticker_symbol not in _models or ticker_symbol not in _scalers:
            return None, None

        model_high = _models[ticker_symbol].get('high')
        model_low = _models[ticker_symbol].get('low')

        X_scaled_high = _scalers[ticker_symbol]['high'].transform(features) if 'high' in _scalers[ticker_symbol] else None
        X_scaled_low = _scalers[ticker_symbol]['low'].transform(features) if 'low' in _scalers[ticker_symbol] else None

        raw_high = model_high.predict(X_scaled_high)[0].item() if model_high and X_scaled_high is not None else 0.0
        raw_low = model_low.predict(X_scaled_low)[0].item() if model_low and X_scaled_low is not None else 0.0

        pred_open_price = open_price_for_prediction # Use the actual open price as base for percentage
        pred_high_price = open_price_for_prediction * (1 + raw_high / 100)
        pred_low_price = open_price_for_prediction * (1 + raw_low / 100)

        # --- Replicate MAE selection logic from main app ---
        display_mae_high = 0.0
        display_mae_low = 0.0

        if ticker_symbol in _backtest and _backtest[ticker_symbol]:
            try:
                bt_data_for_mae_calc = pd.DataFrame(_backtest[ticker_symbol])
                bt_data_for_mae_calc['Date'] = pd.to_datetime(bt_data_for_mae_calc['Date'])
                end_date_bt = bt_data_for_mae_calc['Date'].max()
                start_date_bt = end_date_bt - timedelta(days=60)
                bt_data_for_mae_calc = bt_data_for_mae_calc[bt_data_for_mae_calc['Date'] >= start_date_bt]

                if not bt_data_for_mae_calc.empty:
                    temp_err_df = bt_data_for_mae_calc['Errors'].apply(pd.Series)
                    expected_inner_cols_err = ['open', 'close', 'high', 'low']
                    if all(col in temp_err_df.columns for col in expected_inner_cols_err):
                        if not temp_err_df['high'].dropna().empty:
                            display_mae_high = temp_err_df['high'].abs().mean()
                        if not temp_err_df['low'].dropna().empty:
                            display_mae_low = temp_err_df['low'].abs().mean()
                    else:
                        # Fallback to original model MAE if backtest structure is bad
                        err_original = _errors[ticker_symbol]
                        display_mae_high = err_original['high']['mae'] if 'high' in err_original else 0.0
                        display_mae_low = err_original['low']['mae'] if 'low' in err_original else 0.0
                else:
                    # Fallback to original model MAE if backtest data is empty
                    err_original = _errors[ticker_symbol]
                    display_mae_high = err_original['high']['mae'] if 'high' in err_original else 0.0
                    display_mae_low = err_original['low']['mae'] if 'low' in err_original else 0.0
            except Exception as e:
                # Fallback to original model MAE if an error occurs during backtest MAE calculation
                err_original = _errors[ticker_symbol]
                display_mae_high = err_original['high']['mae'] if 'high' in err_original else 0.0
                display_mae_low = err_original['low']['mae'] if 'low' in err_original else 0.0
        else:
            # Fallback to original model MAE if no backtest data at all
            err_original = _errors[ticker_symbol]
            display_mae_high = err_original['high']['mae'] if 'high' in err_original else 0.0
            display_mae_low = err_original['low']['mae'] if 'low' in err_original else 0.0

        mae_high = display_mae_high # Use the selected MAE
        mae_low = display_mae_low   # Use the selected MAE
        # --- End MAE selection logic ---

        target = pred_high_price - mae_high
        stop_loss = pred_low_price + mae_low
        
        if pred_open_price == 0: # Avoid division by zero
            min_margin_pct = 0
        else:
            min_margin_pct = ((target - stop_loss) / pred_open_price) * 100

        return {
            'Ticker': ticker_symbol,
            'Stock Name': _tickers_dict[ticker_symbol],
            'Predicted Open': pred_open_price,
            'Predicted High': pred_high_price,
            'Predicted Low': pred_low_price,
            'Target (Adj.)': target,
            'Stop Loss (Adj.)': stop_loss,
            'Min Margin %': min_margin_pct
        }, future_day # Return the prediction data dictionary and future_day
        

    with st.spinner("Calculating min margins for all stocks..."):
        for ticker_symbol in tickers.keys():
            pred_data, future_day = get_prediction_data_for_ticker(ticker_symbol, models, scalers, errors, tickers, backtest) # Passed backtest
            if pred_data:
                pred_data['Prediction Date'] = future_day.strftime('%Y-%m-%d') # Add the date
                all_stocks_min_margins.append(pred_data)
    
    if all_stocks_min_margins:
        min_margin_df = pd.DataFrame(all_stocks_min_margins)
        min_margin_df = min_margin_df.sort_values(by='Min Margin %', ascending=False).reset_index(drop=True)

        # Find the stock with the highest min margin
        highest_margin_stock = min_margin_df.iloc[0]

        st.info(f"The stock with the highest predicted Min Margin % for **{highest_margin_stock['Prediction Date']}** is **{highest_margin_stock['Stock Name']} ({highest_margin_stock['Ticker']})** with a margin of **{highest_margin_stock['Min Margin %']:.2f}%**.")

        # Styling for the table
        def highlight_max_margin(s):
            is_max = s == s.max()
            return ['background-color: #004d40; color: white; font-weight: bold;' if v else '' for v in is_max]

        st.dataframe(
            min_margin_df.style
            .apply(highlight_max_margin, subset=['Min Margin %'])
            .format({
                'Predicted Open': "₹{:.2f}",
                'Predicted High': "₹{:.2f}",
                'Predicted Low': "₹{:.2f}",
                'Target (Adj.)': "₹{:.2f}",
                'Stop Loss (Adj.)': "₹{:.2f}",
                'Min Margin %': "{:.2f}%"
            }),
            use_container_width=True
        )
    else:
        st.warning("Could not calculate min margins for any stock. Ensure data and models are correctly loaded.")
