import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide")

container_style = """
    <style>
    body{
        text-align: center;
    }
        .container {
            position: absolute;
            top: 10px;
            right: 10px;
            background-color: #262631;
            padding: 10px;
            border-radius: 15px;
            border: 0.5px solid gray;
            
        }
        .link-button {
            text-decoration: none;
            color: white !important;  /* Button color */
            font-weight: bold;
            # background-color: transparent !important;
            border: none !important;
            cursor: pointer;
            outline: none !important;
            
        }
        .container:hover {
            border: 1px solid white;
            border-radius: 15px;
            
        }
    </style>
"""

# Apply the CSS style
st.markdown(container_style, unsafe_allow_html=True)

# Create the container with the link button
st.markdown("""
<div class='container'>
    <a class='link-button' href='https://streamlit.io/gallery'>Log Out</a>
</div>
""", unsafe_allow_html=True)

# CSS for center-aligning the header and styling the line
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url(https://i.ibb.co/02Hj4f0/img2.jpg);
    background-position: center;
    background-attachment: local, fixed;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

header_style = """
    <style>
     .header {
            color: #fff;
            padding: 0;
            margin-top: 0;
            text-align: center;
        }
        .caption {
            color: #fff;
            text-align: center;
            margin-top: 0;
            padding-top: 100px:
            font-size: 60px;
        }
        .line {   
            border-bottom: 2px dashed #f85a40 #ccc;
            margin-bottom: 20px;
            padding-bottom: 80px;
        }
        .block-container st-emotion-cache-z5fcl4 ea3mdgi2 {
            padding: 0;
        }   
    </style>
"""

# Adding the CSS to the Streamlit app
st.markdown(header_style, unsafe_allow_html=True)

# Header with center alignment and line separator
st.markdown("<h1 class='header'>The Curator<span style='color: #f85a40;'>.</span></h1>", unsafe_allow_html=True)
st.markdown("<h1 class='caption'>Stock Recommendation Tool</h1>", unsafe_allow_html=True)
st.markdown("<div class='line'></div>", unsafe_allow_html=True)

head1 = '''<style>
body{
}
</style>
'''
st.markdown(head1, unsafe_allow_html=True)

# Load the model
model = load_model('Stock_Predictions_Model.keras')

stock_input = st.text_input('Enter Stock Symbol', 'GOOG')

def get_stock_name(symbol):
    stock = yf.Ticker(symbol)
    info = stock.info
    return info['longName']

# Define function for predicting price using moving averages
def predict_price_using_moving_averages(data, short_window, long_window):
    # Calculate the moving averages
    data['Short_MA'] = data['Close'].rolling(short_window).mean()
    data['Long_MA'] = data['Close'].rolling(long_window).mean()

    # Create a column of 1s with the same length as the data
    data['Prediction'] = np.where(data['Short_MA'] > data['Long_MA'], 1, 0)

    # Find the last row of the DataFrame
    last_row = data.iloc[-1]

    if last_row['Prediction'] == 1:
        # If the short MA is above the long MA, predict the closing price to be the same as the short MA
        tomorrow_price = last_row['Short_MA']
    else:
        # Otherwise, predict the closing price to be the same as the long MA
        tomorrow_price = last_row['Long_MA']

    return tomorrow_price

# Main Streamlit app
if st.button('Show Predictions'):
    if stock_input:
        stock_name = get_stock_name(stock_input)
        if stock_name:
            st.subheader(f'Stock Name: {stock_name} ({stock_input})')
        else:
            st.subheader(f'Stock Data: {stock_input}')
        start = '2010-01-01'
        end = date.today()

        data = yf.download(stock_input, start, end)

        if data.empty:
            st.error("No data found for the entered stock symbol. Please try again.")
        else:
            data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
            data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

            scaler = MinMaxScaler(feature_range=(0, 1))

            pas_100_days = data_train.tail(100)
            data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
            data_test_scale = scaler.fit_transform(data_test)

            ma_50_days = data.Close.rolling(50).mean()

            # Display stock data solo
            st.subheader(f'Stock Data: {stock_input}')
            st.write(data[::-1])
            st.markdown("""
                <style>
                    table {
                        max-height: 600px;
                        max-width: 600px;
                        overflow: auto;
                    }
                </style>
            """, unsafe_allow_html=True)

            # Organize the layout for the graphs in a 3x1 grid
            col1, col2, col3 = st.columns(3)

            # Display Price vs MA50 graph
            with col1:
                st.subheader('Price vs MA50')
                fig2, ax2 = plt.subplots(figsize=(6.972, 5.528))
                ax2.plot(ma_50_days, 'r', label='50-Day Moving Average')
                ax2.plot(data.Close, 'g', label='Actual data')
                ax2.legend()
                st.pyplot(fig2)

            # Display Price vs MA100 vs MA200 graphs
            with col2:
                st.subheader('Price vs MA100 vs MA200')
                ma_100_days = data.Close.rolling(100).mean()
                ma_200_days = data.Close.rolling(200).mean()
                fig3, ax3 = plt.subplots(figsize=(6.972, 5.528))
                ax3.plot(ma_100_days, 'r', label='100-Day Moving Average')
                ax3.plot(ma_200_days, 'b', label='200-Day Moving Average')
                ax3.plot(data.Close, 'g', label='Actual data')
                ax3.legend()
                st.pyplot(fig3)

            # Display Original Price vs Predicted Price graph
            with col3:
                x = []
                y = []

                for i in range(100, data_test_scale.shape[0]):
                    x.append(data_test_scale[i - 100:i])
                    y.append(data_test_scale[i, 0])

                x, y = np.array(x), np.array(y)

                predict = model.predict(x)

                scale = 1 / scaler.scale_

                predict = predict * scale
                y = y * scale

                st.subheader('Original Price vs Predicted Price')
                fig4, ax4 = plt.subplots(figsize=(6.972, 5.528))
                ax4.plot(predict, 'r', label='Predicted Price')
                ax4.plot(y, 'g', label='Original Price')
                ax4.legend()
                st.pyplot(fig4)

            window_size_short = 2
            window_size_long = 4
            tomorrow_price = predict_price_using_moving_averages(data, window_size_short, window_size_long)
            st.write(f"Predicted Price for tomorrow: ${tomorrow_price:.2f}")

            st.warning(
                'Disclaimer: It is crucial to understand that the predictions generated by this application are for informational purposes only and should not be used for making financial decisions. Stock market predictions are inherently uncertain, and past performance is not necessarily indicative of future results.'
            )

            st.warning(
                '**Important:** Please refrain from using this app or any similar tool to make investment decisions based solely on predicted prices. Stock market predictions are inherently uncertain, and past performance is not a guarantee of future results. Many factors beyond historical data influence stock prices, and relying on predictions for real-world trading can lead to significant losses.'
            )

            st.info(
                'If you are considering investing in any stock, it is essential to conduct thorough research, consulting with a qualified financial advisor, and carefully considering your personal risk tolerance and investment goals.'
            )
