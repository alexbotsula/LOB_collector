import ccxt
import pandas as pd
import time

# Initialize the Binance exchange
binance = ccxt.binance()

# Function to fetch the limit order book
def fetch_order_book(symbol):
    order_book = binance.fetch_order_book(symbol)
    return order_book

# Function to fetch OHLCV data
def fetch_ohlcv(symbol, timeframe='1m'):
    ohlcv = binance.fetch_ohlcv(symbol, timeframe)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Example usage
symbol = 'BTC/USDT'  # Change this to the trading pair you are interested in

while True:
    try:
        # Fetch the limit order book
        order_book = fetch_order_book(symbol)
        print("Order Book:")
        print("Bids:", order_book['bids'][:5])  # Top 5 bids
        print("Asks:", order_book['asks'][:5])  # Top 5 asks

        # Fetch the OHLCV data
        ohlcv_df = fetch_ohlcv(symbol)
        print("\nOHLCV Data:")
        print(ohlcv_df.tail())  # Print the last few entries

        # Wait for 1 minute
        time.sleep(60)
    except Exception as e:
        print(f"An error occurred: {e}")
        time.sleep(60)  # Wait for 1 minute before trying again