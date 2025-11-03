import yfinance as yf
import pandas as pd
import numpy as np
import requests
import torch
import os
import argparse  # Used to accept command-line arguments
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
from typing import Tuple

# Load environment variables from .env file
load_dotenv()

# --- 1. SETUP AND CONFIGURATION ---

# Load the pre-trained FinBERT model and tokenizer (do this once)
print("Loading FinBERT model...")
MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
print("Model loaded successfully.")


# --- 2. CORE FUNCTIONS ---

def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetches historical stock data from yfinance."""
    print(f"Fetching stock data for {ticker} from {start} to {end}...")
    stock_data = yf.download(ticker, start=start, end=end)
    if stock_data.empty:
        print(f"No data found for {ticker}.")
        return pd.DataFrame()
    
    print("Calculating technical indicators...")
    # Calculate SMAs
    stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    
    # Calculate RSI
    delta = stock_data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = np.where(loss == 0, 0, gain / loss) # Avoid division by zero
    stock_data['RSI_14'] = 100 - (100 / (1 + rs))
    
    return stock_data.dropna() # Drop initial NaNs from indicators

def fetch_news_data(query: str, start: str, end: str) -> pd.DataFrame:
    """Fetches news articles from NewsAPI."""
    print(f"Fetching news for '{query}'...")
    API_KEY = os.environ.get('NEWS_API_KEY')
    if not API_KEY:
        print("Error: NEWS_API_KEY not found.")
        return pd.DataFrame()

    url = (f'https://newsapi.org/v2/everything?'
           f'q={query}&'
           f'from={start}&'
           f'to={end}&'
           f'language=en&'
           f'sortBy=publishedAt&'
           f'apiKey={API_KEY}')
    
    response = requests.get(url)
    news_json = response.json()
    
    if news_json['status'] == 'ok':
        news_df = pd.DataFrame(news_json['articles'])
        print(f"Successfully fetched {len(news_df)} articles.")
        return news_df
    else:
        print(f"Failed to fetch news: {news_json.get('message')}")
        return pd.DataFrame()

def analyze_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """Scores sentiment for each news headline using FinBERT."""
    if news_df.empty:
        return pd.DataFrame()

    print("Analyzing sentiment... (this may take a few minutes)")
    
    def score(text: str) -> Tuple[str, float]:
        if not isinstance(text, str) or not text.strip():
            return "neutral"
        
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_class_idx = torch.argmax(predictions).item()
        return model.config.id2label[sentiment_class_idx]

    # Use .copy() to avoid SettingWithCopyWarning
    news_subset = news_df.copy()
    news_subset['sentiment'] = news_subset['title'].apply(score)
    print("Sentiment analysis complete.")
    return news_subset

def create_composite_signal(stock_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Merges stock indicators and sentiment into a final composite signal."""
    print("Generating composite signal...")
    
    # 1. Aggregate sentiment
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    sentiment_df['sentiment_numeric'] = sentiment_df['sentiment'].map(sentiment_map)
    sentiment_df['publishedAt'] = pd.to_datetime(sentiment_df['publishedAt'])
    sentiment_df.set_index('publishedAt', inplace=True)
    
    # Resample to daily average and fill missing days
    daily_sentiment = sentiment_df['sentiment_numeric'].resample('D').mean().fillna(0)
    
    # Ensure timezone-naive index for merging
    if daily_sentiment.index.tz is not None:
        daily_sentiment.index = daily_sentiment.index.tz_convert(None)

    # 2. Merge data
    combined_df = stock_df.join(daily_sentiment.rename('sentiment_score'), how='left').fillna(0)

    # 3. Normalize features
    scaler = MinMaxScaler()
    features = ['SMA_20', 'SMA_50', 'RSI_14', 'sentiment_score']
    combined_df[features] = scaler.fit_transform(combined_df[features])
    
    # 4. Create composite signal
    weights = {'rsi_w': 0.5, 'sentiment_w': 0.5}
    combined_df['composite_signal'] = (weights['rsi_w'] * combined_df['RSI_14'] + 
                                       weights['sentiment_w'] * combined_df['sentiment_score'])
    
    return combined_df

def run_backtest(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Runs the trading strategy backtest based on the composite signal."""
    print("Running backtest...")
    
    entry_threshold = 0.75 # Buy if signal is in top 25%
    exit_threshold = 0.4   # Sell if signal is in bottom 40%

    combined_df['position'] = np.where(combined_df['composite_signal'] > entry_threshold, 1, 0)
    combined_df['position'] = np.where(combined_df['composite_signal'] < exit_threshold, 0, combined_df['position'])
    combined_df['position'] = combined_df['position'].ffill().fillna(0)
    
    combined_df['daily_return'] = combined_df['Close'].pct_change()
    combined_df['strategy_return'] = combined_df['position'].shift(1) * combined_df['daily_return']
    
    combined_df['cumulative_buy_hold'] = (1 + combined_df['daily_return']).cumprod()
    combined_df['cumulative_strategy'] = (1 + combined_df['strategy_return']).cumprod()
    
    return combined_df

def plot_results(combined_df: pd.DataFrame, ticker: str):
    """Plots the backtest results and saves the figure."""
    print("Plotting results...")
    plt.figure(figsize=(14, 7))
    plt.plot(combined_df['cumulative_buy_hold'], label='Buy and Hold')
    plt.plot(combined_df['cumulative_strategy'], label='Multi-Factor Strategy')
    plt.title(f'Strategy Performance vs. Buy and Hold for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a file
    output_filename = f"{ticker}_strategy_plot.png"
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
    plt.show()

# --- 3. MAIN EXECUTION ---

def main(ticker: str, query: str, start: str, end: str):
    """Main function to run the complete analysis pipeline."""
    
    stock_df = fetch_stock_data(ticker, start, end)
    if stock_df.empty:
        return

    news_df = fetch_news_data(query, start, end)
    if news_df.empty:
        print("No news data found. Proceeding with technical analysis only.")
        # We can still proceed, sentiment will be 0
        sentiment_df = pd.DataFrame(columns=['publishedAt', 'sentiment'])
    else:
        sentiment_df = analyze_sentiment(news_df)

    combined_df = create_composite_signal(stock_df, sentiment_df)
    
    backtest_results = run_backtest(combined_df)
    
    plot_results(backtest_results, ticker)
    
    print("\n--- Analysis Complete ---")
    print("Final Strategy Return:", backtest_results['cumulative_strategy'].iloc[-1])
    print("Final Buy/Hold Return:", backtest_results['cumulative_buy_hold'].iloc[-1])

if __name__ == "__main__":
    # This allows you to run the script from the command line
    parser = argparse.ArgumentParser(description="Run a multi-factor stock analysis and backtest.")
    parser.add_argument('--ticker', type=str, default='VBL.NS', help='Stock ticker symbol (e.g., VBL.NS)')
    parser.add_argument('--query', type=str, default='Varun Beverages', help='News search query (e.g., Varun Beverages)')
    parser.add_argument('--start', type=str, default='2023-10-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-10-01', help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    main(args.ticker, args.query, args.start, args.end)