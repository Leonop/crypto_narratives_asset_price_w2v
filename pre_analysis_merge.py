# write a file to aggregate daily file to weekly file
# and merge it with bitcoin return data and ltm 3 factors data
# load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import global_options
from pathlib import Path
from datetime import datetime

def datetime_to_yearweek(date):
    # Get the ISO calendar week number
    year, week_number, _ = date.isocalendar()
    
    # Return the year and week number in the format yyyyww
    return f"{year}{week_number:02d}"

# the the path of this file
# this file is in the folder "scripts"
# the root folder is the parent folder of "scripts"
current_folder = Path(__file__).parent
root_folder = current_folder.parent.parent
data_folder = str(Path(root_folder, "reddit_data"))
print(current_folder)

# move the column 'yyww' to the certain column
def move_column(df, column_name, position):
    # get the column names
    columns = df.columns.tolist()
    # remove the column name from the list
    columns.remove(column_name)
    # insert the column name to the certain position
    columns.insert(position, column_name)
    return df[columns]

def merge_all_3methods_df():
    methods = ["TF", "TFIDF", "WFIDF"]
    df_bitcoin = pd.DataFrame()
    for method in methods:
        temp = pd.read_csv(str(Path(current_folder,"outputs", "scores", f"bitcoin_scores_{method}_all.csv")))
        # Ensure 'datetime' column exists and is in the correct format
        if 'datetime' not in temp.columns:
            raise KeyError(f"'datetime' column not found in {method} DataFrame")
        # Print columns for debugging
        temp['datetime'] = pd.to_datetime(temp['datetime'])
        temp['yyww'] = temp['datetime'].apply(datetime_to_yearweek)
        for topic_name in global_options.SEED_WORDS.keys():
            temp.rename(columns={topic_name: f"{topic_name}_{method}"}, inplace=True)
        if df_bitcoin.empty:
            df_bitcoin = temp
        else:
            df_bitcoin = pd.merge(df_bitcoin, temp, on=['datetime', 'yyww', 'document_length'], how='left', suffixes=('', f"_{method}"))
    return df_bitcoin

def load_sentimen_attention_data():
    # read the data from the csv file in the reddit_data folder
    bitcoin_posts = pd.read_csv(os.path.join(data_folder, "bitcoin_attention_sentiment.csv"))
    # global variable for Loughran McDonald's sentiment dictionary
    bitcoin_posts['post_date'] = pd.to_datetime(bitcoin_posts['post_date'])
    bitcoin_posts['yyww'] = bitcoin_posts['post_date'].apply(datetime_to_yearweek)
    return bitcoin_posts

def agg_weekly_w2v_scores(df_daily):
    # Aggregate the daily scores to weekly scores
    # load the daily scores
    df_daily.drop(columns=['datetime'], inplace=True)
    # aggregate the daily scores to weekly scores
    df_weekly = df_daily.groupby('yyww').mean().reset_index()
    return df_weekly

def agg_weekly_scores():
    # Aggregate the daily scores to weekly scores
    # drop column datetime
    df_daily = load_sentimen_attention_data()
    df_daily.drop(columns=['post_date'], inplace=True)
    df_weekly = df_daily.groupby('yyww').mean().reset_index()
    print(df_weekly.head())
    return df_weekly

def compute_average_tone(df, date_col, bubble_col, tone_col):
    # Ensure the date_col is in the DataFrame
    if date_col not in df.columns:
        raise ValueError(f"The column {date_col} does not exist in the DataFrame.")

    # Ensure the bubble_col and tone_col are in the DataFrame
    if bubble_col not in df.columns:
        raise ValueError(f"The column {bubble_col} does not exist in the DataFrame.")
    if tone_col not in df.columns:
        raise ValueError(f"The column {tone_col} does not exist in the DataFrame.")

    # Convert the date column to datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Group by date and bubble category, then calculate the average tone
    average_tone = df.groupby([df[date_col].dt.date, bubble_col])[tone_col].mean().unstack(fill_value=0)

    # Rename the columns
    average_tone = average_tone.rename(columns={0: 'No_Bubble_tone', 0.5: 'Uncertain_tone', 1: 'Bubble_tone'})

    return average_tone.reset_index()

def weekly_post(df, date_col, interval='W'):
    filename = "GPT_labeled_sample_5000_format_cleaned_20240608.csv"
    filepath = data_folder
    df = pd.read_csv(os.path.join(filepath, filename))
    if date_col not in df.columns:
        print(f"Error: The column {date_col} does not exist in the DataFrame.")
        return
    # print(type(df.loc[0,date_col]))
    df[f"{date_col}_ymd"] = pd.to_datetime(df[date_col])
    df[f"{date_col}_ymd"] = df[f"{date_col}_ymd"].dt.strftime('%Y-%m-%d')
    df[f"{date_col}_ymd"] = pd.to_datetime(df[f"{date_col}_ymd"])
    # drop nan if date_col is null
    df = df.dropna(subset=[f"{date_col}_ymd"])
    result = compute_average_tone(df, f"{date_col}_ymd", 'bubble', 'tone')
    if interval == 'W':
        col_date1 = 'week'
        df[col_date1] = df[f"{date_col}_ymd"].apply(datetime_to_yearweek)
        result[col_date1] = result[f"{date_col}_ymd"].apply(datetime_to_yearweek)
        result = result.groupby(col_date1)[['No_Bubble_tone',  'Uncertain_tone',  'Bubble_tone']].mean().reset_index()
        # print(result.head())
    elif interval == 'M':
        col_date1 = 'month'
        df[col_date1] = df[date_col].dt.to_period(interval)
    elif interval == 'D':
        col_date1 = 'day'
        df[col_date1] = df[f"{date_col}_ymd"].dt.date
    # convert col_date1 to the week of the year if interval is 'W'
    # Ensure 'bubble' column is numeric and contains expected values
    if 'bubble' in df.columns:
        df['bubble'] = pd.to_numeric(df['bubble'], errors='coerce')
    else:
        print("Error: 'bubble' column not found.")
        return

    # Handling the data aggregation
    filtered_data = df.groupby([col_date1, 'bubble']).size().unstack(fill_value=0)
    filtered_data = filtered_data.reset_index()
    filtered_data = filtered_data.rename(columns={0: 'No Bubble', 0.5: 'Uncertain', 1: 'Bubble'})
    # Rename the index name

    filtered_data = filtered_data.rename_axis(None, axis=1)
    result = result.rename_axis(None, axis=1)
    result.rename(columns={f"{date_col}_ymd": col_date1}, inplace=True)
    result = pd.merge(filtered_data, result, how='left', on=col_date1)
    return result



def compute_upvote_ratio(df):
    ups = df["ups"]
    downs = df["downs"]
    # Fill missing values with 0 to avoid NaN in upvote_ratio computation
    ups = ups.fillna(0)
    downs = downs.fillna(0)
    # Compute the upvote_ratio
    df["upvote_ratio"] = np.where((ups + downs) > 0, (ups - downs)/(ups+downs), 0.01)
    # fill the missing values with 0.01
    return df["upvote_ratio"]

def compute_weighted_average_tone(df, date_col, bubble_col, tone_col, upvote_col):
    # Ensure the date_col is in the DataFrame
    if date_col not in df.columns:
        raise ValueError(f"The column {date_col} does not exist in the DataFrame.")

    # Ensure the bubble_col and tone_col are in the DataFrame
    if bubble_col not in df.columns:
        raise ValueError(f"The column {bubble_col} does not exist in the DataFrame.")
    if tone_col not in df.columns:
        raise ValueError(f"The column {tone_col} does not exist in the DataFrame.")

    # Convert the date column to datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # computed the weighted average tone
    df['weighted_tone'] = df[upvote_col] * df[tone_col]
    # Group by date and bubble category, then calculate the average tone
    average_tone = df.groupby([df[date_col].dt.date, bubble_col])['weighted_tone'].mean().unstack(fill_value=0)

    # Rename the columns
    average_tone = average_tone.rename(columns={0: 'No_Bubble_tone_upvote', 0.5: 'Uncertain_tone_upvote', 1: 'Bubble_tone_upvote'})

    return average_tone.reset_index()


def weekly_up_down_weighted_post(df, date_col, interval='W'):
    filename = "GPT_labeled_sample_5000_format_cleaned_20240608.csv"
    filepath = data_folder
    df = pd.read_csv(os.path.join(filepath, filename))
    if date_col not in df.columns:
        print(f"Error: The column {date_col} does not exist in the DataFrame.")
        return
    # print(type(df.loc[0,date_col]))
    df[f"{date_col}_ymd"] = pd.to_datetime(df[date_col])
    df[f"{date_col}_ymd"] = df[f"{date_col}_ymd"].dt.strftime('%Y-%m-%d')
    df[f"{date_col}_ymd"] = pd.to_datetime(df[f"{date_col}_ymd"])
    # drop nan if date_col is null
    df = df.dropna(subset=[f"{date_col}_ymd"])
    # print(df.head)
    df['upvote_ratio'] = compute_upvote_ratio(df)
    # print(df.head())
    result = compute_weighted_average_tone(df, f"{date_col}_ymd", 'bubble', 'tone', 'upvote_ratio')
    if interval == 'W':
        col_date1 = 'week'
        df[col_date1] = df[f"{date_col}_ymd"].apply(datetime_to_yearweek)
        result[col_date1] = result[f"{date_col}_ymd"].apply(datetime_to_yearweek)
        result = result.groupby(col_date1)[['No_Bubble_tone_upvote',  'Uncertain_tone_upvote',  'Bubble_tone_upvote']].mean().reset_index()
        # print(result.head())
    elif interval == 'M':
        col_date1 = 'month'
        df[col_date1] = df[date_col].dt.to_period(interval)
    elif interval == 'D':
        col_date1 = 'day'
        df[col_date1] = df[f"{date_col}_ymd"].dt.date
    # convert col_date1 to the week of the year if interval is 'W'
    # Ensure 'bubble' column is numeric and contains expected values
    if 'bubble' in df.columns:
        df['bubble'] = pd.to_numeric(df['bubble'], errors='coerce')
    else:
        print("Error: 'bubble' column not found.")
        return

    # Handling the data aggregation
    filtered_data = df.groupby([col_date1, 'bubble']).size().unstack(fill_value=0)
    filtered_data = filtered_data.reset_index()
    filtered_data = filtered_data.rename(columns={0: 'No Bubble', 0.5: 'Uncertain', 1: 'Bubble'})
    # Rename the index name

    filtered_data = filtered_data.rename_axis(None, axis=1)
    result = result.rename_axis(None, axis=1)
    result.rename(columns={f"{date_col}_ymd": col_date1}, inplace=True)
    result = pd.merge(filtered_data, result, how='left', on=col_date1)
    return result

def crypto_ret(df):
    df['bitret'] = df['price'].pct_change()
    # shift the bitret to the previous week
    df['bitret'] = df['bitret'].shift(-1)
    return df

def bitcoin_ret_interval(interval='W'):
    bitcoin_filename = "btc-usd-max.csv"
    bitcoin_path = data_folder
    full_path = os.path.join(bitcoin_path, bitcoin_filename)
    try:
        df_bitcoin = pd.read_csv(full_path)
    except FileNotFoundError:
        print(f"Error: The file {full_path} does not exist.")
        return None

    df_bitcoin['created_date'] = pd.to_datetime(df_bitcoin['snapped_at'])
    if interval == 'W':
        col_date = 'week'
    elif interval == 'M':
        col_date = 'month'
    df_bitcoin[col_date] = df_bitcoin['created_date'].dt.to_period(interval)
    df_bitcoin = df_bitcoin.groupby(col_date)['price'].first().reset_index() # get the price of the first day of the week or month

    df_bitcoin[col_date] = df_bitcoin[col_date].dt.start_time
    df_bitcoin = crypto_ret(df_bitcoin)
    return df_bitcoin


if __name__ == "__main__":
    filename = "GPT_labeled_sample_5000_format_cleaned_20240608.csv"
    filepath = data_folder
    df = pd.read_csv(os.path.join(filepath, filename))
    weekly_posts = weekly_post(df, 'created_date', 'W')
    weekly_posts.rename(columns = {'week':'yyww'}, inplace = True)
    weekly_posts.head(20)


    df_bitcoin = merge_all_3methods_df()
    df_bitcoin = move_column(df_bitcoin, 'yyww', 1)
    df_bitcoin = move_column(df_bitcoin, 'document_length', 2)
    df_bitcoin_scores_weekly = agg_weekly_w2v_scores(df_bitcoin)
    df_bitcoin_scores_weekly.head(10)

    file_path = str(Path(root_folder, "Crypto_data", "LTW_3factor.xlsx"))
    # Load the Excel file into a DataFrame, skipping the first 5 rows
    LTW_3factors = pd.read_excel(file_path, skiprows=5)
    LTW_3factors['yyww'] = LTW_3factors['yyww'].astype(int).astype(str)
    LTW_3factors.head()

    weekly_weighted_posts = weekly_up_down_weighted_post(df, 'created_date', 'W')
    weekly_weighted_posts.rename(columns = {'week':'yyww'}, inplace = True)
    weekly_weighted_posts.head(20)

    # merge weekly_weighted_posts and weekly_posts by yyww
    weekly_posts = pd.merge(weekly_posts, weekly_weighted_posts[['yyww', 'No_Bubble_tone_upvote', 'Uncertain_tone_upvote', 'Bubble_tone_upvote']], on='yyww', how='left')
    weekly_posts.head()

    # merge df_bitcoin with LTW_3factors on yyww with Weekly data
    df_bitcoin_scores_weekly['yyww'] = df_bitcoin_scores_weekly['yyww'].astype(int).astype(str)
    df_bitcoin_LTM3factors = pd.merge(df_bitcoin_scores_weekly, LTW_3factors, on='yyww', how='left')
    df_bitcoin_LTM3factors.head()

    bitcoin_ret = bitcoin_ret_interval('W')
    bitcoin_ret = bitcoin_ret.drop(0).reset_index(drop=True)
    bitcoin_ret.head()


    bitcoin_ret['week'] = pd.to_datetime(bitcoin_ret['week'])
    # year_week = datetime_to_yearweek(date)
    bitcoin_ret['yyww'] = bitcoin_ret['week'].apply(datetime_to_yearweek)
    bitcoin_ret.head()

    df_bitret_LTM3factors = pd.merge(bitcoin_ret, df_bitcoin_LTM3factors, on='yyww', how='left')
    df_bitret_LTM3factors.head()

    # save the final dataframe to a csv file
    df_bitret_LTM3factors.to_csv(str(Path(current_folder,"outputs", "weekly_posts_narrative_tone_ltm3.csv")), index=False)


    # get the weekly attention and sentiment data
    attention_sentiment_weekly = agg_weekly_scores()

    # merge the weekly attention and sentiment data with the final dataframe
    df_bitret_LTM3factors = pd.merge(df_bitret_LTM3factors, attention_sentiment_weekly, on='yyww', how='left')

    # save the final dataframe to a csv file
    df_bitret_LTM3factors.to_csv(str(Path(current_folder,"outputs", "weekly_posts_narrative_tone_ltm3_attention_sentiment.csv")), index=False)