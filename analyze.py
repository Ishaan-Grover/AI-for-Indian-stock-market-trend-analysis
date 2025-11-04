import yfinance as yf
import pandas as pd
import numpy as np
import requests
import torch
import os
import argparse
import matplotlib
matplotlib.use('Agg')  # <-- ADD THIS LINE
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
from typing import Tuple

# Load environment variables from .env file
load_dotenv()

# --- 1. SETUP AND CONFIGURATION ---

print("Attempting to load FinBERT model...")
try:
    MODEL_NAME = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"\n--- CRITICAL ERROR ---")
    print(f"Failed to load FinBERT model. This might be a network issue.")
    print(f"Error details: {e}")
    print("Stopping script.")
    exit() # Stop the script if the model can't load


# --- 2. CORE FUNCTIONS (Ported from your Cells) ---

def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetches historical stock data from yfinance."""
    print(f"\nFetching stock data for {ticker} from {start} to {end}...")
    try:
        stock_data = yf.download(ticker, start=start, end=end)
        if stock_data.empty:
            print(f"⚠️ Warning: yfinance returned an empty table for {ticker}.")
            return pd.DataFrame() # Return the empty frame
        
        # --- ⭐️ THIS IS THE FIX ⭐️ ---
        # Check for MultiIndex (multi-level headers) and flatten it
        if isinstance(stock_data.columns, pd.MultiIndex):
            print("Detected MultiIndex from yfinance. Flattening DataFrame...")
            # Collapse the MultiIndex (e.g., ('Close', 'VBL.NS') -> 'Close')
            # We only care about the first level (Open, High, etc.)
            stock_data.columns = stock_data.columns.get_level_values(0)
            
            # Check for and remove duplicate columns (like 'Adj Close')
            if not stock_data.columns.is_unique:
                stock_data = stock_data.loc[:, ~stock_data.columns.duplicated()]
        # --- ⭐️ END OF FIX ⭐️ ---

        print("Calculating technical indicators...")
        
        # This code will now *always* operate on a single-level index
        stock_data['SMA_5'] = stock_data['Close'].rolling(window=5).mean()
        stock_data['SMA_10'] = stock_data['Close'].rolling(window=10).mean()
        
        delta = stock_data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = np.where(loss == 0, 0, gain / loss) 
        stock_data['RSI_14'] = 100 - (100 / (1 + rs))
        
        print("✅ Technical indicators calculated.")
        return stock_data
    except Exception as e:
        print(f"\n--- CRITICAL ERROR in fetch_stock_data ---")
        print(f"An error occurred: {e}")
        print("This could be a KeyError if the 'Close' column wasn't found after flattening.")
        return pd.DataFrame()

def fetch_news_data(query: str, start: str, end: str) -> pd.DataFrame:
    """Fetches news articles from NewsAPI."""
    print(f"\nFetching news for '{query}'...")
    API_KEY = os.environ.get('NEWS_API_KEY')
    if not API_KEY:
        print(f"\n--- CRITICAL ERROR in fetch_news_data ---")
        print("NEWS_API_KEY not found in .env file.")
        print("Please check your .env file is in the correct folder and named correctly.")
        return pd.DataFrame()

    url = (f'https://newsapi.org/v2/everything?'
           f'q={query}&from={start}&to={end}&language=en&'
           f'sortBy=publishedAt&apiKey={API_KEY}')
    
    try:
        response = requests.get(url)
        news_json = response.json()
        
        if news_json['status'] == 'ok':
            news_df = pd.DataFrame(news_json['articles'])
            print(f"✅ Successfully fetched {len(news_df)} articles.")
            return news_df
        else:
            print(f"\n--- CRITICAL ERROR in fetch_news_data ---")
            print(f"NewsAPI returned an error: {news_json.get('message')}")
            return pd.DataFrame()
    except Exception as e:
        print(f"\n--- CRITICAL ERROR in fetch_news_data ---")
        print(f"An error occurred while making the request: {e}")
        return pd.DataFrame()

def analyze_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """Scores sentiment for each news headline using FinBERT."""
    if news_df.empty:
        print("No news to analyze, skipping sentiment analysis.")
        return pd.DataFrame(columns=['publishedAt', 'sentiment']) # Return empty DF

    print("\nAnalyzing sentiment...")
    
    def score(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return "neutral"
        try:
            inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            sentiment_class_idx = torch.argmax(predictions).item()
            return model.config.id2label[sentiment_class_idx]
        except Exception as e:
            print(f"Warning: Failed to score text '{text}'. Error: {e}")
            return "neutral"

    news_subset = news_df.copy()
    news_subset['sentiment'] = news_subset['title'].apply(score)
    print("✅ Sentiment analysis complete.")
    return news_subset

def create_composite_signal(stock_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Merges stock indicators and sentiment into a final composite signal."""
    print("\nGenerating composite signal...")
    try:
        sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
        sentiment_df['sentiment_numeric'] = sentiment_df['sentiment'].map(sentiment_map)
        
        if 'publishedAt' in sentiment_df.columns and not sentiment_df.empty:
            sentiment_df['publishedAt'] = pd.to_datetime(sentiment_df['publishedAt'])
            sentiment_df.set_index('publishedAt', inplace=True)
            daily_sentiment = sentiment_df['sentiment_numeric'].resample('D').mean().fillna(0)
            
            if daily_sentiment.index.tz is not None:
                daily_sentiment.index = daily_sentiment.index.tz_convert(None)
        else:
            daily_sentiment = pd.Series(index=stock_df.index, data=0, name='sentiment_numeric')

        # This join will now work, as both stock_df and daily_sentiment are single-level
        combined_df = stock_df.join(daily_sentiment.rename('sentiment_score'), how='left').fillna(0)

        scaler = MinMaxScaler()
        features = ['SMA_5', 'SMA_10', 'RSI_14', 'sentiment_score']
        
        combined_df[features] = combined_df[features].fillna(0) 
        
        for col in features:
            # Check for columns that are all the same value (no variance)
            if (combined_df[col].max() - combined_df[col].min()) == 0:
                print(f"Warning: Column '{col}' has no variance. Setting scaled to 0.")
                combined_df[col] = 0.0
            else:
                # Use .values.reshape(-1, 1) to avoid sklearn warning
                combined_df[col] = scaler.fit_transform(combined_df[col].values.reshape(-1, 1))
        
        weights = {'rsi_w': 0.5, 'sentiment_w': 0.5}
        combined_df['composite_signal'] = (weights['rsi_w'] * combined_df['RSI_14'] + 
                                           weights['sentiment_w'] * combined_df['sentiment_score'])
        
        print("✅ Composite signal generated.")
        return combined_df
    except Exception as e:
        print(f"\n--- CRITICAL ERROR in create_composite_signal ---")
        print(f"An error occurred: {e}")
        return pd.DataFrame()


def run_backtest(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Runs the trading strategy backtest based on the composite signal."""
    print("\nRunning backtest...")
    try:
        entry_threshold = 0.75
        exit_threshold = 0.4

        combined_df['position'] = np.where(combined_df['composite_signal'] > entry_threshold, 1, 0)
        combined_df['position'] = np.where(combined_df['composite_signal'] < exit_threshold, 0, combined_df['position'])
        combined_df['position'] = combined_df['position'].ffill().fillna(0) 
        
        combined_df['daily_return'] = combined_df['Close'].pct_change()
        combined_df['strategy_return'] = combined_df['position'].shift(1) * combined_df['daily_return']
        
        combined_df['daily_return'] = combined_df['daily_return'].fillna(0)
        combined_df['strategy_return'] = combined_df['strategy_return'].fillna(0)
        
        combined_df['cumulative_buy_hold'] = (1 + combined_df['daily_return']).cumprod()
        combined_df['cumulative_strategy'] = (1 + combined_df['strategy_return']).cumprod()
        
        print("✅ Backtest complete.")
        return combined_df
    except Exception as e:
        print(f"\n--- CRITICAL ERROR in run_backtest ---")
        print(f"An error occurred: {e}")
        return pd.DataFrame()


def plot_results(combined_df: pd.DataFrame, ticker: str):
    """Plots the backtest results and saves the figure."""
    print("\nPlotting results...")
    try:
        plt.figure(figsize=(14, 7))
        plt.plot(combined_df['cumulative_buy_hold'], label='Buy and Hold')
        plt.plot(combined_df['cumulative_strategy'], label='Multi-Factor Strategy')
        plt.title(f'Strategy Performance vs. Buy and Hold for {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid(True)
        
        output_filename = f"{ticker}_strategy_plot.png"
        plt.savefig(output_filename)
        print(f"✅ Plot saved to {output_filename}")
        #plt.show()
    except Exception as e:
        print(f"\n--- CRITICAL ERROR in plot_results ---")
        print(f"An error occurred: {e}")

# --- 3. MAIN EXECUTION ---

def main(ticker: str, query: str, start: str, end: str):
    """Main function to run the complete analysis pipeline."""
    
    stock_df = fetch_stock_data(ticker, start, end)
    if stock_df.empty:
        print("\nStopping script because no stock data could be processed.")
        return

    news_df = fetch_news_data(query, start, end)
    # We continue even if news_df is empty
    
    sentiment_df = analyze_sentiment(news_df) 
    
    combined_df = create_composite_signal(stock_df, sentiment_df)
    if combined_df.empty:
        print("\nStopping script because composite signal could not be generated.")
        return
    
    backtest_results = run_backtest(combined_df)
    if backtest_results.empty:
        print("\nStopping script because backtest failed.")
        return
    
    plot_results(backtest_results, ticker)
    
    print("\n--- ✅✅✅ Analysis Complete ✅✅✅ ---")
    print("Final Strategy Return:", backtest_results['cumulative_strategy'].iloc[-1])
    print("Final Buy/Hold Return:", backtest_results['cumulative_buy_hold'].iloc[-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a multi-factor stock analysis and backtest.")
    
    # --- YOUR ORIGINAL DEFAULTS (Cell 2 & 3) ---
    parser.add_argument('--ticker', type=str, default='VBL.NS', help='Stock ticker symbol')
    parser.add_argument('--query', type=str, default='Varun Beverages', help='News search query')
    parser.add_argument('--start', type=str, default='2025-10-05', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-11-03', help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    print(f"--- Starting Analysis ---")
    print(f"Ticker: {args.ticker}, Query: {args.query}, Start: {args.start}, End: {args.end}")
    
    main(args.ticker, args.query, args.start, args.end)