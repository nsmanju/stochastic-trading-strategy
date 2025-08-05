""" Notes for Beginners:

You need historical price data (with columns: 'Open', 'High', 'Low', 'Close').

Set use_macd=True if you want to add the MACD filter; otherwise, the basic stochastic/EMA logic applies.

The script marks each row with "Buy", "Sell", or blank ("") in the 'Signal' column per strategy rules.

Stop loss and take profit management are not handled by the code, but you can retrieve signals and implement exit rules manually. """


import pandas as pd
import numpy as np

# Function to calculate the 200-period Exponential Moving Average (EMA)
def calculate_ema(data, period=200):
    """
    Calculate the 200-period EMA for the 'Close' column.
    Returns a pandas Series of EMA values.
    """
    return data['Close'].ewm(span=period, adjust=False).mean()

# Function to calculate the Stochastics Oscillator (%K and %D)
def calculate_stochastics(data, k_period=14, d_period=3):
    """
    Calculate Stochastics Oscillator (%K, %D) for the given data.
    %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = Simple Moving Average of %K over d_period
    Adds %K and %D columns to the DataFrame.
    """
    # Lowest low over the lookback window
    low_min = data['Low'].rolling(window=k_period).min()
    # Highest high over the lookback window
    high_max = data['High'].rolling(window=k_period).max()
    # %K calculation
    data['%K'] = ((data['Close'] - low_min) / (high_max - low_min)) * 100
    # %D is the moving average of %K
    data['%D'] = data['%K'].rolling(window=d_period).mean()
    return data

# Function to calculate the MACD and its signal line (optional advanced confirmation)
def calculate_macd(data, fast=12, slow=26, signal=9):
    """
    Calculate the MACD (Moving Average Convergence Divergence).
    MACD = 12-EMA - 26-EMA
    Signal Line = 9-period EMA of the MACD line
    Adds 'MACD' and 'MACD_Signal' columns to the DataFrame.
    """
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    data['MACD'] = macd
    data['MACD_Signal'] = signal_line
    return data

def stochastic_strategy(data, use_macd=False):
    """
    Applies the complete stochastic trading strategy logic:
    - Adds EMA, Stochastics, and optionally MACD to the DataFrame
    - Checks for buy/sell signals per provided logic
    - Returns the DataFrame with a 'Signal' column indicating entries
    Parameters:
      data: A DataFrame with 'Open', 'High', 'Low', 'Close' as columns
      use_macd: If True, require MACD confirmation for entry signals
    """
    # Calculate the 200 EMA and add to dataframe
    data['EMA_200'] = calculate_ema(data)
    # Calculate Stochastics indicators and add to dataframe
    data = calculate_stochastics(data)
    # Optionally, calculate MACD indicators and add
    if use_macd:
        data = calculate_macd(data)

    # List to store 'Buy', 'Sell', or '' (no signal) for each row
    signals = []

    # Loop through DataFrame row by row (starting from 1, since we look back one row)
    for i in range(1, len(data)):
        row = data.iloc[i]
        prev_row = data.iloc[i-1]

        # Conditions for price position relative to 200 EMA
        price_above_ema = row['Close'] > row['EMA_200']
        price_below_ema = row['Close'] < row['EMA_200']

        # --- BUY SIGNAL LOGIC ---
        # 1. Price is above the 200 EMA
        # 2. %K is in the oversold zone (below 20)
        # 3. %K line crosses above %D (was below previous bar, above this bar)
        # 4. (Optional) MACD line crosses above signal line (additional filter)
        # 5. All above trigger Buy signal
        buy_condition = (
            price_above_ema and
            row['%K'] < 20 and             # Stochastics in oversold zone
            prev_row['%K'] < prev_row['%D'] and
            row['%K'] > row['%D'] and      # Crossover of %K above %D
            row['%K'] <= 20                # %K remains inside oversold zone after cross
        )
        if use_macd:
            # If MACD filter, only confirm Buy if MACD > signal line
            buy_condition = buy_condition and (row['MACD'] > row['MACD_Signal'])

        # --- SELL SIGNAL LOGIC ---
        # 1. Price is below the 200 EMA
        # 2. %K is in the overbought zone (above 80)
        # 3. %K line crosses below %D (was above previous bar, below this bar)
        # 4. (Optional) MACD line crosses below signal line (additional filter)
        # 5. All above trigger Sell signal
        sell_condition = (
            price_below_ema and
            row['%K'] > 80 and             # Stochastics in overbought zone
            prev_row['%K'] > prev_row['%D'] and
            row['%K'] < row['%D'] and      # Crossover of %K below %D
            row['%K'] >= 80                # %K remains inside overbought zone after cross
        )
        if use_macd:
            # If MACD filter, only confirm Sell if MACD < signal line
            sell_condition = sell_condition and (row['MACD'] < row['MACD_Signal'])

        # Store signal based on detected condition
        if buy_condition:
            signals.append('Buy')
        elif sell_condition:
            signals.append('Sell')
        else:
            signals.append('')  # No signal this bar

    signals.insert(0, '')  # No signal for first row (due to shift by 1 in loop)
    data['Signal'] = signals  # Add signals column to dataframe
    return data

# --- Sample Usage (uncomment and provide your data to use) ---
df = pd.read_csv('sample_stock_data.csv')   # Your OHLC data
result = stochastic_strategy(df, use_macd=True) # Set to False if you don't want to use MACD
print(result[['Close', 'EMA_200', '%K', '%D', 'MACD', 'MACD_Signal', 'Signal']].tail())
