# Stock Sentiment Analysis Using Machine Learning
## Introduction
The project aims to develop a sentiment analysis model to predict the movement of stock prices based on textual data from news articles, social media posts, and other sources of financial news and opinions. By analyzing the sentiment expressed in these texts, the model will seek to uncover insights into investor sentiment and market sentiment, which can be valuable indicators for making informed trading decisions.

## Implementation
*	The project starts by collecting news headlines and financial data from websites using web scraping or APIs provided by these websites.
*	Then the required libraries are imported into the code.
*	The collected news headlines data is preprocessed to remove special characters, normalize text (e.g., convert to lowercase), tokenize sentences into words, remove stopwords, and apply stemming or lemmatization to standardize word forms.
*	This preprocessing enhances the quality of data used in the project.
*	Then, financial data and news headlines data are merged and stored in the ‘merged_df’ dataframe.
*	Next, sentiment scores of news headlines on a particular date are calculated using SentimentIntensityAnalyzer(). Based on these values, sentiment categories and movements are determined.
*	As the number of neutral sentiments is much less compared to positive and negative sentiments, only two movements are considered: 1 for positive and 0 for negative and neutral.
*	Now, the data is split into two datasets: a training dataset (75%) and a testing dataset (25%).
*	Models are implemented on the training datasets, and their predictions are made on the testing dataset to predict stock prices and movements.
*	Based on the predictions, various calculations such as Accuracy, Precision, F1 score, Recall, ROC AUC, Sharpe Ratio, Maximum Drawdown, No. of Trades Executed, Win Ratio, MAE, and MSE are performed.
*	Graphs are plotted based on these predictions and calculations accordingly.

### Potential Improvements:
  * Twitter data should also be utilized as it can influence stock price movements. I was unable to extract data from Twitter due to API limits.
  *	Changes in market conditions, investor behavior, and external events should influence stock price movements and should also be considered.
 

### Application
*	Investment Decision Making
*	Risk Management
*	Market Research and Strategy

### References
  *	https://www.investopedia.com/terms/s/sentimentindicator.asp
  *	https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8659448/
  *	https://www.insightbig.com/post/stock-market-sentiment-prediction-with-openai-and-python
  *	Stock Market Sentiment Analysis Using Python & Machine Learning

