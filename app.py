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
import requests # Import the requests library

# --- Configuration for Google Drive Download ---
# Google Drive File ID for multi_stock_models.pkl
GOOGLE_DRIVE_FILE_ID = "1tR8pR4miv5Gfbvd5bxurBdIshUvhkRBp"
# Constructed direct download URL
GOOGLE_DRIVE_DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"
LOCAL_MODELS_FILENAME = "multi_stock_models.pkl"

# --- Function to download file from Google Drive ---
@st.cache_resource
def download_file_from_google_drive(file_id, destination):
    """
    Downloads a file from Google Drive to a specified local destination.
    Uses a direct download link.
    """
    st.info(f"⏳ Downloading {destination} from Google Drive...")
    try:
        response = requests.get(GOOGLE_DRIVE_DOWNLOAD_URL, stream=True)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        # Handle Google Drive's virus scan warning for large files
        if "confirm" in response.url:
            id_match = re.search(r"id=([\w-]+)", response.url)
            if id_match:
                file_id = id_match.group(1)
                download_url = f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
                response = requests.get(download_url, stream=True)
                response.raise_for_status()

        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success(f"✅ Downloaded {destination} successfully.")
        return True
    except Exception as e:
        st.error(f"❌ Error downloading {destination} from Google Drive: {e}")
        return False

# --- Initial Download Call (runs only once per Streamlit Cloud deployment) ---
# Ensure models file is downloaded before trying to load it
if not os.path.exists(LOCAL_MODELS_FILENAME):
    download_success = download_file_from_google_drive(GOOGLE_DRIVE_FILE_ID, LOCAL_MODELS_FILENAME)
    if not download_success:
        st.error("Could not download necessary model files. Please check the Google Drive link or file permissions.")
        st.stop() # Stop the app if download fails

# --- Load Latest Files ---
def get_latest_file(pattern):
    """Finds the most recently modified file matching a pattern."""
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)

# Now models_file points to the locally downloaded file
models_file = LOCAL_MODELS_FILENAME 
scalers_file = get_latest_file("multi_stock_scalers.pkl")
errors_file = get_latest_file("multi_stock_errors.pkl")
backtest_file = get_latest_file("multi_stock_backtest.pkl") 

if not (models_file and scalers_file and errors_file):
    st.error("❌ Required model/scaler/error files not found even after download attempt. Please check your repository and Google Drive.")
    st.stop()

try:
    models = joblib.load(models_file)
    scalers = joblib.load(scalers_file)
    errors = joblib.load(errors_file)
    backtest = joblib.load(backtest_file) if backtest_file else {}
except Exception as e:
    st.error(f"❌ Error loading model/data files: {e}. Ensure all files are valid pickle files and accessible.")
    st.stop()

# --- App Config and Style ---
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
    .accuracy-blue { background-color: #007bb6; } /* Blue for backtest MAE */

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
    
    /* Styling for backtest table errors */
    .error-small { color: #4CAF50; font-weight: bold; } /* Green for small errors */
    .error-large { color: #F44336; font-weight: bold; } /* Red for large errors */

</style>
""", unsafe_allow_html=True)

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
date_range = st.sidebar.selectbox("**Select Date Range for Chart:**", ["1M", "3M", "6M", "1Y", "5Y"], index=4)
chart_type = st.sidebar.radio("**Select Chart Type:**", options=["Line Chart", "Candlestick"], index=0)
show_chart = st.sidebar.checkbox("📊 Show Price Chart", value=False)
backtest_mode = st.sidebar.checkbox("📈 Show Backtest Last Month", value=False)

# --- Fetch Data ---
@st.cache_data(ttl=3600)
def get_data(ticker, range_key):
    end_date = datetime.now()
    start_dict = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "5Y": 1825}
    days = start_dict.get(range_key, 1825)
    start_date = end_date - timedelta(days=days)
    
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
    df = get_data(selected_ticker, date_range)

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

    err = errors[selected_ticker]
    mae_open = err['open']['mae']
    mae_close = err['close']['mae']
    mae_high = err['high']['mae'] if 'high' in err else 0.0
    mae_low = err['low']['mae'] if 'low' in err else 0.0

    target = pred_high_price - mae_high
    stop_loss = pred_low_price + mae_low
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
                <li><b>Predicted Open:</b> <span class="{open_color}">₹{pred_open_price:.2f}</span> ± ₹{mae_open:.2f}</li>
                <li><b>Predicted Close:</b> <span class="{close_color}">₹{pred_close_price:.2f}</span> ± ₹{mae_close:.2f}</li>
                <li><b>Predicted High:</b> <span class="{high_color}">₹{pred_high_price:.2f}</span> ± ₹{mae_high:.2f}</li>
                <li><b>Predicted Low:</b> <span class="{low_color}">₹{pred_low_price:.2f}</span> ± ₹{mae_low:.2f}</li>
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
                <li><b>Min Margin:</b> ₹{margin_range:.2f} ({min_margin_pct:.2f}%)</li>
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


    # --- Accuracy Display Redesign ---
    acc_open = 100 - mae_open
    acc_close = 100 - mae_close
    acc_high = 100 - mae_high
    acc_low = 100 - mae_low

    st.subheader("✅ Model Accuracy (100 - MAE)")
    col_acc_top1, col_acc_top2 = st.columns(2)
    col_acc_bottom1, col_acc_bottom2 = st.columns(2)

    with col_acc_top1:
        st.markdown(f"""
        <div style="padding: 10px; background-color: #1f1f1f; border-radius: 8px; text-align: center;">
            <div class="accuracy-block accuracy-green"><b>High:</b><br>{acc_high:.2f}% ± ₹{mae_high:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_acc_top2:
        st.markdown(f"""
        <div style="padding: 10px; background-color: #1f1f1f; border-radius: 8px; text-align: center;">
            <div class="accuracy-block accuracy-red"><b>Low:</b><br>{acc_low:.2f}% ± ₹{mae_low:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_acc_bottom1:
        st.markdown(f"""
        <div style="padding: 10px; background-color: #1f1f1f; border-radius: 8px; text-align: center;">
            <div class="accuracy-block accuracy-green"><b>Open:</b><br>{acc_open:.2f}% ± ₹{mae_open:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_acc_bottom2:
        st.markdown(f"""
        <div style="padding: 10px; background-color: #1f1f1f; border-radius: 8px; text-align: center;">
            <div class="accuracy-block accuracy-red"><b>Close:</b><br>{acc_close:.2f}% ± ₹{mae_close:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Adding a small gap between accuracy blocks and the next section
    st.markdown("<br>", unsafe_allow_html=True)


    if show_chart:
        st.subheader("📊 Historical Price Chart with Next Day Prediction")
        chart_df = df[['Open', 'High', 'Low', 'Close']].copy()
        chart_df.loc[Timestamp(future_day)] = [pred_open_price, pred_high_price, pred_low_price, pred_close_price]
        chart_df.sort_index(inplace=True)

        fig = go.Figure()
        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=chart_df.index,
                open=chart_df['Open'],
                high=chart_df['High'],
                low=chart_df['Low'],
                close=chart_df['Close'],
                name='Candlestick',
                increasing_line_color= '#4CAF50',  # Green for increasing
                decreasing_line_color= '#F44336' # Red for decreasing
            ))
        else:
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['Open'], mode='lines', name='Open Price', line=dict(color='#ADD8E6')))
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['Close'], mode='lines', name='Close Price', line=dict(color='#FFD700')))
            
        # Add highlight for the predicted day
        if pd.Timestamp(future_day) in chart_df.index:
            fig.add_vline(x=pd.Timestamp(future_day), line_width=2, line_dash="dash", line_color="#00acc1",
                          annotation_text=f"Predicted Day: {future_day.strftime('%Y-%m-%d')}", annotation_position="top right",
                          annotation_font_color="white")


        fig.update_layout(
            title=f"{tickers[selected_ticker]} Stock Price Chart",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            hovermode="x unified",
            xaxis=dict(tickformat="%Y-%m-%d", rangeslider_visible=True),
            plot_bgcolor="#1e1e1e",
            paper_bgcolor="#1e1e1e",
            font=dict(color="white"),
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)', font=dict(color='white'))
        )
        st.plotly_chart(fig, use_container_width=True)

    if backtest_mode and selected_ticker in backtest:
        st.subheader("🔢 Backtest Results (Last 30 Days)")
        bt = pd.DataFrame(backtest[selected_ticker])

        if bt.empty:
            st.warning(f"⚠️ Backtest data for {tickers[selected_ticker]} is empty. No results to display.")
        else:
            def expand_cols(df, col):
                return pd.DataFrame(df[col].tolist(), index=df.index)

            bt['Date'] = pd.to_datetime(bt['Date'])
            
            pred_df = expand_cols(bt, 'Predictions')
            act_df = expand_cols(bt, 'Actuals')
            err_df = expand_cols(bt, 'Errors')

            expected_cols_count = 4 
            if (pred_df.shape[1] != expected_cols_count or 
                act_df.shape[1] != expected_cols_count or 
                err_df.shape[1] != expected_cols_count):
                st.error("❌ Backtest data structure mismatch: Expected 4 columns (Open, Close, High, Low) for Predictions, Actuals, and Errors. Displaying raw data for inspection.")
                st.dataframe(bt, use_container_width=True, height=700)
            else:
                pred_df.columns = ['Open', 'Close', 'High', 'Low']
                act_df.columns = ['Open', 'Close', 'High', 'Low']
                err_df.columns = ['Open', 'Close', 'High', 'Low']

                combined_df = pd.DataFrame({
                    'Date': bt['Date'],
                    'Actual_Open': act_df['Open'],
                    'Pred_Open': pred_df['Open'],
                    'Error_Open': err_df['Open'],
                    'Actual_Close': act_df['Close'],
                    'Pred_Close': pred_df['Close'],
                    'Error_Close': err_df['Close'],
                    'Actual_Low': act_df['Low'],
                    'Pred_Low': pred_df['Low'],
                    'Error_Low': err_df['Low'],
                    'Actual_High': act_df['High'],
                    'Pred_High': pred_df['High'],
                    'Error_High': err_df['High'],
                })

                ordered_cols = [
                    'Date',
                    'Actual_Open', 'Pred_Open', 'Error_Open',
                    'Actual_Close', 'Pred_Close', 'Error_Close',
                    'Actual_Low', 'Pred_Low', 'Error_Low',
                    'Actual_High', 'Pred_High', 'Error_High',
                ]

                combined_df = combined_df[ordered_cols]

                # --- Apply conditional styling for backtest table errors ---
                ERROR_THRESHOLD = 1.0 # Adjust this based on what you consider a "small" vs "large" error

                def highlight_error_cells(val):
                    if pd.isna(val):
                        return ''
                    if abs(val) <= ERROR_THRESHOLD:
                        return 'color: #4CAF50; font-weight: bold;' # Green for small error
                    else:
                        return 'color: #F44336; font-weight: bold;' # Red for large error
                    
                styled_df = combined_df.style \
                    .applymap(highlight_error_cells, subset=['Error_Open', 'Error_Close', 'Error_High', 'Error_Low']) \
                    .format({col: "₹{:.2f}" for col in combined_df.columns if col != 'Date'})
                
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=700
                )

                # --- Calculate and Display Backtest MAE ---
                st.markdown("<br>", unsafe_allow_html=True) # Add some space
                st.subheader("📊 Backtest Mean Absolute Error (MAE)")
                
                backtest_mae_open = err_df['Open'].abs().mean() if not err_df['Open'].empty else 0.0
                backtest_mae_close = err_df['Close'].abs().mean() if not err_df['Close'].empty else 0.0
                backtest_mae_high = err_df['High'].abs().mean() if not err_df['High'].empty else 0.0
                backtest_mae_low = err_df['Low'].abs().mean() if not err_df['Low'].empty else 0.0

                st.markdown(f"""
                <div style="display: flex; justify-content: space-around; padding: 10px; background-color: #1f1f1f; border-radius: 8px;">
                    <div class="accuracy-block accuracy-blue"><b>Open MAE:</b><br>₹{backtest_mae_open:.2f}</div>
                    <div class="accuracy-block accuracy-blue"><b>Close MAE:</b><br>₹{backtest_mae_close:.2f}</div>
                    <div class="accuracy-block accuracy-blue"><b>High MAE:</b><br>₹{backtest_mae_high:.2f}</div>
                    <div class="accuracy-block accuracy-blue"><b>Low MAE:</b><br>₹{backtest_mae_low:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

    elif backtest_mode and selected_ticker not in backtest:
        st.warning(f"⚠️ No backtest data found for {tickers[selected_ticker]}. Ensure a backtest file exists and contains data for this stock.")
