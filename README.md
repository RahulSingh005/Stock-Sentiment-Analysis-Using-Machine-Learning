# Stock Sentiment Analysis Using Machine Learning

## Introduction

This project develops a sentiment analysis model to predict stock price movements by analyzing textual data from news articles, social media posts, and other financial sources. By evaluating the sentiment expressed in these texts, the model provides insights into investor and market sentiment, which are valuable for making informed trading decisions.

## Implementation

- **Data Collection:** News headlines and financial data are collected from websites using web scraping or APIs.
- **Data Preprocessing:** The collected news headlines are cleaned by removing special characters, normalizing text (e.g., converting to lowercase), tokenizing sentences into words, removing stopwords, and applying stemming or lemmatization to standardize word forms.
- **Data Integration:** Financial data and news headlines are merged and stored in a `merged_df` DataFrame.
- **Sentiment Analysis:** Sentiment scores for news headlines on specific dates are calculated using `SentimentIntensityAnalyzer()`. Sentiment categories and price movements are determined.
- **Labeling:** Due to the limited number of neutral sentiments, only two movement labels are used: 1 (positive) and 0 (negative or neutral).
- **Train-Test Split:** The data is split into training (75%) and testing (25%) datasets.
- **Model Training and Evaluation:** Machine learning models are trained on the training dataset and evaluated on the testing dataset to predict stock price movements.
- **Performance Metrics:** Key metrics such as Accuracy, Precision, F1 Score, Recall, ROC AUC, Sharpe Ratio, Maximum Drawdown, Number of Trades Executed, Win Ratio, MAE, and MSE are calculated.
- **Visualization:** Graphs are plotted based on predictions and performance metrics.

## Potential Improvements

- **Incorporate Twitter Data:** Utilize Twitter data to capture additional sentiment signals, as it can significantly influence stock price movements. (Note: Data extraction was limited due to API restrictions.)
- **Adapt to Market Conditions:** Incorporate changes in market conditions, investor behavior, and external events to improve model robustness and accuracy.

## Applications

- **Investment Decision Making:** Supports more informed trading strategies.
- **Risk Management:** Helps identify and mitigate investment risks.
- **Market Research and Strategy:** Provides insights for market research and strategic planning.

## References

- [Sentiment Indicator - Investopedia](https://www.investopedia.com/terms/s/sentimentindicator.asp)
- [Stock Market Sentiment Analysis Using Machine Learning - NCBI](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8659448/)
- [Stock Market Sentiment Prediction with OpenAI and Python - InsightBig](https://www.insightbig.com/post/stock-market-sentiment-prediction-with-openai-and-python)
- Stock Market Sentiment Analysis Using Python & Machine Learning
