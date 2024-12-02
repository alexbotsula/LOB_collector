import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

# Connect to the SQLite database
engine = create_engine('sqlite:///crypto_data.db')

# Function to fetch OHLCV data from the database
def fetch_ohlcv(symbol, engine):
    query = f"SELECT timestamp, open, high, low, close, volume FROM ohlcv WHERE symbol = '{symbol}'"
    df = pd.read_sql(query, engine)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Function to fetch order book data from the database
def fetch_order_book(symbol, engine):
    query = f"SELECT timestamp, bids, asks FROM order_books WHERE symbol = '{symbol}'"
    df = pd.read_sql(query, engine)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Visualize OHLCV data
def plot_ohlcv(df, symbol):
    plt.figure(figsize=(14, 7))
    plt.plot(df['timestamp'], df['open'], label='Open', color='blue')
    plt.plot(df['timestamp'], df['high'], label='High', color='green')
    plt.plot(df['timestamp'], df['low'], label='Low', color='red')
    plt.plot(df['timestamp'], df['close'], label='Close', color='black')
    plt.fill_between(df['timestamp'], df['low'], df['high'], color='gray', alpha=0.3)
    plt.title(f'OHLCV Data for {symbol}')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Visualize order book data
def plot_order_book(df, symbol):
    plt.figure(figsize=(14, 7))
    for idx, row in df.iterrows():
        bids = pd.DataFrame(row['bids'], columns=['price', 'amount'])
        asks = pd.DataFrame(row['asks'], columns=['price', 'amount'])
        plt.plot(bids['price'], bids['amount'], label=f'Bids at {row["timestamp"]}', color='green')
        plt.plot(asks['price'], asks['amount'], label=f'Asks at {row["timestamp"]}', color='red')
    plt.title(f'Order Book Data for {symbol}')
    plt.xlabel('Price')
    plt.ylabel('Amount')
    plt.legend()
    plt.show()

def main():
    symbols = ['BTC/USDT', 'ETH/USDT']
    
    for symbol in symbols:
        # Fetch and plot data for the symbol
        ohlcv_df = fetch_ohlcv(symbol, engine)
        order_book_df = fetch_order_book(symbol, engine)

        # Plot OHLCV data
        plot_ohlcv(ohlcv_df, symbol)

        # Plot order book data
        plot_order_book(order_book_df, symbol)

if __name__ == '__main__':
    main()
