import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Define the time range for the historical data you'll download
start_date = '2022-01-01'
end_date = '2025-01-01'

# Define the stock symbol for Tata Motors in NSE format for yfinance
stock_symbol = 'TATAMOTORS.NS'

# Function to calculate the Exponential Moving Average (EMA)
def calculate_ema(data, period=200):
    # Calculate the EMA for the 'Close' prices and add it as a column to the data
    data['EMA_200'] = data['Close'].ewm(span=period, adjust=False).mean()
    return data

# Function to calculate the Stochastic Oscillator (%K and %D)
def calculate_stochastics(data, k_period=21, d_period=5):
    # Calculate the rolling minimum of the 'Low' prices over k_period
    low_min = data['Low'].rolling(window=k_period).min()
    # Calculate the rolling maximum of the 'High' prices over k_period
    high_max = data['High'].rolling(window=k_period).max()
    # Calculate %K line of the stochastic oscillator
    data['%K'] = ((data['Close'] - low_min) / (high_max - low_min)) * 100
    # Calculate %D line, which is the moving average of %K over d_period
    data['%D'] = data['%K'].rolling(window=d_period).mean()
    return data

# Main trading strategy function implementing EMA and stochastic conditions
def stochastic_strategy(data):
    data = data.copy()  # Create a copy to avoid modifying original data
    data = calculate_ema(data)  # Calculate EMA and add to data
    data = calculate_stochastics(data)  # Calculate stochastic oscillator lines and add to data
    # Drop rows where EMA or stochastic indicator values are NaN (from rolling calculations)
    data = data.dropna(subset=['EMA_200', '%K', '%D'])

    signals = ['']  # Initialize the signal list; first entry has no signal

    # Convert required columns to numpy arrays for speed and easy element access
    close = data['Close'].values
    ema_200 = data['EMA_200'].values
    k = data['%K'].values
    d = data['%D'].values

    # Loop through each data point from the second row onward (since we compare to previous)
    for i in range(1, len(data)):
        price_above_ema = close[i] > ema_200[i]  # Is current close above EMA?
        price_below_ema = close[i] < ema_200[i]  # Is current close below EMA?

        # Buy conditions:
        # 1. Price is above EMA (uptrend)
        # 2. %K is below 20 (oversold region)
        # 3. %K just crossed above %D
        if price_above_ema and k[i] < 20 and k[i-1] < d[i-1] and k[i] > d[i] and k[i] <= 20:
            signals.append('Buy')

        # Sell conditions:
        # 1. Price is below EMA (downtrend)
        # 2. %K is above 80 (overbought region)
        # 3. %K just crossed below %D
        elif price_below_ema and k[i] > 80 and k[i-1] > d[i-1] and k[i] < d[i] and k[i] >= 80:
            signals.append('Sell')

        else:
            signals.append('')  # No signal for this row

    # Add the collected signals back into the data frame
    data['Signal'] = signals
    return data

# Main execution starts here
if __name__ == "__main__":
    # Download historical data for the defined stock symbol and date range
    df = yf.download(stock_symbol, start=start_date, end=end_date, auto_adjust=False)

    # Check if columns have MultiIndex (sometimes yfinance returns that)
    # Flatten to single level columns by taking the first level if necessary
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Select only needed OHLC columns and convert all data to floats, drop any NaNs
    df = df[['Open', 'High', 'Low', 'Close']].astype(float).dropna()

    # Save the cleaned raw data to CSV for reference
    df.to_csv(f'{stock_symbol.replace(":", "_")}_stock_data.csv')

    # Run the trading strategy on the downloaded data
    df_strat = stochastic_strategy(df)

    # Create a figure for plotting with 3 rows and 1 column of subplots
    plt.figure(figsize=(14, 10))

    # Subplot 1: Plot close price, EMA, and buy/sell signals as scatter points
    plt.subplot(3, 1, 1)
    plt.plot(df_strat.index, df_strat['Close'], label='Close Price', color='blue')
    plt.plot(df_strat.index, df_strat['EMA_200'], label='200 EMA', color='orange')

    buy_signals = df_strat[df_strat['Signal'] == 'Buy']  # Filter buy signals
    sell_signals = df_strat[df_strat['Signal'] == 'Sell']  # Filter sell signals

    # Plot buy signals with green upward triangles
    plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal', s=100)
    # Plot sell signals with red downward triangles
    plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal', s=100)

    # Annotate buy signals with text and arrows
    for idx, row in buy_signals.iterrows():
        plt.annotate(f'Buy\n{row.Close:.2f}', xy=(idx, row.Close), xytext=(0, 15), textcoords='offset points',
                     arrowprops=dict(facecolor='green', arrowstyle='->', alpha=0.5), fontsize=8, color='green', ha='center')

    # Annotate sell signals similarly
    for idx, row in sell_signals.iterrows():
        plt.annotate(f'Sell\n{row.Close:.2f}', xy=(idx, row.Close), xytext=(0, -20), textcoords='offset points',
                     arrowprops=dict(facecolor='red', arrowstyle='->', alpha=0.5), fontsize=8, color='red', ha='center')

    plt.title(f'{stock_symbol} Close Price and 200 EMA with Buy/Sell Signals')
    plt.legend()
    plt.grid(True)

    # Subplot 2: Plot stochastic oscillator %K and %D with overbought (80) and oversold (20) lines
    plt.subplot(3, 1, 2)
    plt.plot(df_strat.index, df_strat['%K'], label='%K', color='purple')
    plt.plot(df_strat.index, df_strat['%D'], label='%D', color='magenta')
    plt.axhline(80, color='red', linestyle='--', alpha=0.6)  # Overbought
    plt.axhline(20, color='green', linestyle='--', alpha=0.6)  # Oversold
    plt.title(f'{stock_symbol} Stochastic Oscillator (%K and %D)')
    plt.legend()
    plt.grid(True)

    # Subplot 3: Plot separate bars for buy and sell signal counts
    plt.subplot(3, 1, 3)
    buy_count = len(buy_signals)
    sell_count = len(sell_signals)
    plt.bar(['Buy Signals'], [buy_count], color='green')
    plt.bar(['Sell Signals'], [sell_count], color='red')
    plt.title(f'{stock_symbol} Separate Buy and Sell Signal Counts')

    plt.tight_layout()
    plt.show()
