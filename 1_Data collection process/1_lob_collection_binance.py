import ccxt
import pandas as pd
import time
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# PostgreSQL Database Configuration
DB_HOST = "localhost"
DB_NAME = "crypto_data"
DB_USER = "lob_user"
DB_PORT = "5432"  # Default PostgreSQL port

DB_PASSWORD = os.getenv("DB_PASSWORD", "default_password")

# Initialize the database engine
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}', pool_pre_ping=True)
Base = declarative_base()

# Define the OrderBook table
class OrderBook(Base):
    __tablename__ = 'order_books'
    id = Column(Integer, primary_key=True)
    timestamp = Column(String)
    symbol = Column(String)
    bids = Column(JSON)
    asks = Column(JSON)

# Define the OHLCV table
class OHLCV(Base):
    __tablename__ = 'ohlcv'
    id = Column(Integer, primary_key=True)
    timestamp = Column(String)
    symbol = Column(String)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

# Create all tables in the PostgreSQL database
Base.metadata.create_all(engine)

# Initialize the session
Session = sessionmaker(bind=engine)
session = Session()

# Initialize the Binance exchange
binance = ccxt.binance()

# Function to fetch the limit order book
def fetch_order_book(symbol):
    order_book = binance.fetch_order_book(symbol)
    return order_book

# Function to fetch OHLCV data
def fetch_ohlcv(symbol, timeframe='1m'):
    ohlcv = binance.fetch_ohlcv(symbol, timeframe)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']).iloc[[-1]]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').astype(str)
    return df

# Function to store data in PostgreSQL
def store_data(symbol, order_book, ohlcv_df):
    # Order Book Data
    order_book_data = OrderBook(
        timestamp=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        symbol=symbol,
        bids=order_book['bids'],
        asks=order_book['asks']
    )
    session.add(order_book_data)

    # OHLCV Data
    for index, row in ohlcv_df.iterrows():
        ohlcv_data = OHLCV(
            timestamp=row['timestamp'],
            symbol=symbol,
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume']
        )
        session.add(ohlcv_data)

    session.commit()

# Example usage
symbols = ['BTC/USDT', 'ETH/USDT']  # List of trading pairs

while True:
    for symbol in symbols:
        try:
            # Fetch the limit order book
            order_book = fetch_order_book(symbol)
            print(f"Order Book for {symbol}:")
            print("Bids:", order_book['bids'][:5])  # Top 5 bids
            print("Asks:", order_book['asks'][:5])  # Top 5 asks

            # Fetch the OHLCV data
            ohlcv_df = fetch_ohlcv(symbol)
            print(f"\nOHLCV Data for {symbol}:")
            print(ohlcv_df.tail())  # Print the last few entries

            # Store data in PostgreSQL
            store_data(symbol, order_book, ohlcv_df)

        except Exception as e:
            print(f"An error occurred for {symbol}: {e}")

    # Wait for 1 minute
    time.sleep(60)
