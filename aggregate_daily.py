# This file is to aggregate the scores by daily.
# Author: Zicheng (Leo) Xiao
# Environment： Python 3.8 
# Virtualenv: word2vec_kai 
# Date : 08/12/2024

import global_options
import pandas as pd
from pathlib import Path
from tqdm import tqdm

print("Aggregating scores to firms and adjusting by document lengths.")

def aggregate_daily(topic_name, method = "TF"):
    id2firm = pd.read_csv(str(Path(global_options.DATA_FOLDER, "input", "document_ids2datetime.csv")))
    # rename the column id to document_id
    id2firm.rename(columns={"id": "Doc_ID", "Unnamed: 0": "index"}, inplace=True)
    scores = pd.read_csv(
        str(
            Path(global_options.OUTPUT_FOLDER, "scores", f"{method}", "scores_{}_{}.csv".format(method, topic_name))
        )
    )
    scores['Doc_ID'] = scores['Doc_ID'].str.replace(r'\"', '', regex=True)
    # print(scores.head())
    # print(id2firm.head())
    # print(id2firm.dtypes)
    scores = scores.merge(id2firm, how="left", on=["Doc_ID"]).drop(["Doc_ID", "index"], axis=1)

    # adjust the scores by document length
    scores[topic_name] = 100 * scores[topic_name] / scores["document_length"]
    # conert the datetime to daily date
    scores["datetime"] = pd.to_datetime(scores["datetime"]).dt.date
    scores = scores.groupby(["datetime"]).mean()
    scores = scores.sort_values(by=["datetime"], ascending=True).reset_index()
    return scores

def save_scores(scores, method, topic_name):
    scores.to_csv(
        str(
            Path(
                global_options.OUTPUT_FOLDER,
                "scores",
                f"bitcoin_scores_{method}_{topic_name}.csv",
            )
        ),
        index=False,
        float_format="%.4f",
    )
if __name__ == "__main__":
    agg_scores = pd.DataFrame()
    methods = ["TF", "TFIDF", "WFIDF"]
    for method in methods:
        for k, v in tqdm(dict(global_options.SEED_WORDS).items()):
            print(f"Aggregating scores for {k} using {method}")
            daily_scores = aggregate_daily(k, method)
            # print(daily_scores.head())  # Check the first few rows of the DataFrame
            if agg_scores.empty:
                agg_scores = daily_scores
            else:
                # combine the scores generated aggregate_daily to one dataframe, keep the common columns, and add the new columns
                agg_scores = agg_scores.merge(daily_scores, on=['datetime','document_length'], how="left", suffixes=("", f"_{method}"))
        save_scores(agg_scores, method, "all")
        agg_scores = pd.DataFrame()

