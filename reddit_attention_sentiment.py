# this script is to create the attention index and sentiment index of reddit posts
# author: Zicheng Xiao (Leo)
# Date: 2024-09-08
# Usage: python reddit_attention_sentiment.py

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
# Load FinBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

# read the data
data_path = Path(__file__).resolve().parent.parent.parent / "reddit_data"
print(os.path.join(data_path, "Bitcoin_submissions.csv"))
bitcoin_posts = pd.read_csv(os.path.join(data_path, "Bitcoin_submissions.csv")) # read the bitcoin data


# please change the path to relative path
def compute_attention(raw_data):
    # create the attention index by counting the number of unique ids per day
    bitcoin_posts['created_utc'] = pd.to_datetime(bitcoin_posts['created_utc'], unit='s')
    bitcoin_posts['post_date'] = bitcoin_posts['created_utc'].dt.date

    # Group by post_date and count the number of unique ids
    attention_df = bitcoin_posts.groupby(['post_date'])['id'].count().reset_index()
    attention_df.rename(columns={'id': 'attention'}, inplace=True)
    attention_df['attention'] = attention_df['attention'].fillna(0)
    return attention_df


# Define a function to compute sentiment for a chunk of text
def compute_chunk_sentiment(chunk):
    inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(probs, dim=1).item()
    return sentiment

# Define a function to compute sentiment for long texts using parallel processing
def compute_sentiment(text):
    if not isinstance(text, str):
        return None  # or handle non-string values as needed
    
    max_length = 512
    chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
    
    with ThreadPoolExecutor() as executor:
        sentiments = list(executor.map(compute_chunk_sentiment, chunks))
    
    # Aggregate sentiment scores (e.g., majority vote)
    overall_sentiment = max(set(sentiments), key=sentiments.count)
    return overall_sentiment

def aggregate_sentiment(df):
    # Group by post_date and count the number of unique ids
    sentiment_df['created_utc'] = pd.to_datetime(sentiment_df['created_utc'], unit='s')
    bitcoin_posts['post_date'] = bitcoin_posts['created_utc'].dt.date
    sentiment_df = sentiment_df.groupby(['post_date'])['sentiment'].mean().reset_index()
    sentiment_df.rename(columns={'sentiment': 'sentiment'}, inplace=True)
    sentiment_df['sentiment'] = sentiment_df['sentiment'].fillna(0)
    return sentiment_df

if __name__ == "__main__":
    attention_df = compute_attention(bitcoin_posts)
    # Print the head of the dataset to verify
    print(attention_df.head())
    # attention_df.describe()
    # Initialize tqdm progress bar
    tqdm.pandas()
    # Apply the sentiment analysis function to each post
    bitcoin_posts['sentiment'] = bitcoin_posts['selftext'].progress_apply(compute_sentiment)
    # aggregate the sentiment by date and merge it attention_df on date
    sentiment_df = aggregate_sentiment(bitcoin_posts)
    # Merge the attention and sentiment dataframes
    bitcoin_data = attention_df.merge(sentiment_df, on='post_date', how='left')
    # save the data to the data folder
    bitcoin_data.to_csv(os.path.join(data_path, "bitcoin_attention_sentiment.csv"), index=False)
    # Print the head of the dataset to verify
    print(bitcoin_data.head())