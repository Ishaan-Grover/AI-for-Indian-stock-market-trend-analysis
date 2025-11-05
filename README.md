# AI for Indian stock market trend analysis Tool
### 📄 [View the Full Project Presentation (PDF)](Project_Presentation.pdf)

This project analyzes a given stock by combining traditional technical indicators (like SMA and RSI) with modern AI-based sentiment analysis from news headlines. It backtests a trading strategy based on this composite signal.

This was a project for my Major in AI at IIT Ropar.

## 🚀 Features

* Fetches historical stock data using `yfinance`.
* Pulls relevant news articles using the `NewsAPI`.
* Analyzes news headlines for 'positive', 'negative', or 'neutral' sentiment using the **FinBERT** (a transformer AI model).
* Generates a composite "buy/sell" signal by weighting both technical and sentiment scores.
* Runs a full backtest to compare the strategy's performance against a simple "Buy and Hold" approach.
* Generates a plot visualizing the results.

## 🛠️ Tech Stack

* **Python**
* **Pandas & NumPy:** For data manipulation and analysis.
* **Transformers (Hugging Face):** For loading the FinBERT NLP model.
* **PyTorch:** As the backend for the AI model.
* **scikit-learn:** For data normalization.
* **NewsAPI & yfinance:** For data collection.
* **Matplotlib:** For plotting the final results.

##  How to Run This Project

1.  Clone or download this repository.
2.  Create a `.env` file and add your NewsAPI key: `NEWS_API_KEY="YOUR_KEY_HERE"`
3.  Install the required libraries:
    ```
    pip install -r requirements.txt
    ```
4.  Run the analysis from your terminal:
    ```
    python analyze.py
    ```
5.  You can also analyze a different stock:
    ```
    python analyze.py --ticker "RELIANCE.NS" --query "Reliance Industries"

    ```
