import tweepy
import pandas as pd

def fetch_tweets(api_key, api_key_secret, access_token, access_token_secret, query, max_tweets):
    auth = tweepy.OAuthHandler(api_key, api_key_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    tweets = tweepy.Cursor(api.search_tweets, q=query, lang="en").items(max_tweets)
    
    tweet_data = [[tweet.created_at, tweet.text] for tweet in tweets]
    df = pd.DataFrame(tweet_data, columns=['timestamp', 'text'])
    
    return df

# Add your own Twitter API credentials here
api_key = 'your_api_key'
api_key_secret = 'your_api_key_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'
query = "keyword"
max_tweets = 100

df = fetch_tweets(api_key, api_key_secret, access_token, access_token_secret, query, max_tweets)
df.to_csv('data/tweets.csv', index=False)