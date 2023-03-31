# front end
import streamlit as st
# snsrape is a webscraping tool
from snscrape.modules.twitter import TwitterSearchScraper
# pandas is a data manipulation tool
import pandas as pd
# numpy is a numerical manipulation tool
import numpy as np
# transformers is a natural language processing tool
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# scipy is a scientific computing tool
from scipy.special import softmax
# streamlit components
import streamlit.components.v1 as components

# a title for my app
st.title('Stock Market Sentiment Analysis')
st.subheader('Analyze the sentiment of tweets about your favorite stock!')

# input the crypto ticker
ticker = st.selectbox("Select a stock ticker", ('Click Here to Select',"AAPL", "AMZN", "FB", "GOOG", "MSFT", "TSLA"))

processed_ticker = '#' + ticker
# scrape twitter for tweets that have the ticker
scraper = TwitterSearchScraper(processed_ticker)

# result array
tweets = []

if ticker != 'Click Here to Select':
    # loop through the first 100 tweets and append the content to the result array
    for i, item in enumerate(scraper.get_items()):
        tweets.append(item.content)
        if i == 99:
            break

    # load the tokenizer and model
    model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

    tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

    labels = ['Negative', 'Neutral', 'Positive']

    # preprocess the tweets before passing them to the model
    for i in range(len(tweets)):
        st.write(tweets[i])
        
        tweet_words = []
        
        for word in tweets[i].split():
            if word.startswith('@') and len(word) > 1:
                word = '@user'
            elif word.startswith('http'):
                word = 'http'
            tweet_words.append(word)
        
        processed_tweet = ' '.join(tweet_words)
        
        # sentiment analysis
        encoded_tweet = tokenizer(processed_tweet, return_tensors='pt')

        output = model(**encoded_tweet)

        scores = output[0][0].detach().numpy()

        scores = softmax(scores)

        st.write(labels[2], scores[2], labels[1], scores[1], labels[0], scores[0])

        st.write(":heavy_minus_sign:" * 33)

st.balloons()
