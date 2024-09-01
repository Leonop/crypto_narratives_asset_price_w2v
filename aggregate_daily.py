# This file is to aggregate the scores by daily.
# Author: Zicheng (Leo) Xiao
# Environment： Python 3.8 
# Virtualenv: word2vec_kai 
# Date : 08/12/2024

import global_options
import pandas as pd
from pathlib import Path

print("Aggregating scores to firms and adjusting by document lengths.")

id2firm = pd.read_csv(str(Path(global_options.DATA_FOLDER, "input", "document_ids2datetime.csv")))
# rename the column id to document_id
id2firm.rename(columns={"id": "Doc_ID", "Unnamed: 0": "index"}, inplace=True)
methods = ["TF", "TFIDF", "WFIDF"]
for method in methods:
    scores = pd.read_csv(
        str(
            Path(global_options.OUTPUT_FOLDER, "scores", "scores_{}_{}.csv".format(method, global_options.DIMS[0]))
        )
    )
    scores['Doc_ID'] = scores['Doc_ID'].str.replace(r'\"', '', regex=True)
    # print(scores.head())
    # print(id2firm.head())
    # print(id2firm.dtypes)
    scores = scores.merge(id2firm, how="left", on=["Doc_ID"]).drop(["Doc_ID", "index"], axis=1)
    for dim in global_options.DIMS:
        scores[dim] = 100 * scores[dim] / scores["document_length"]
    # conert the datetime to daily date
    scores["datetime"] = pd.to_datetime(scores["datetime"]).dt.date
    scores = scores.groupby(["datetime"]).mean()
    scores = scores.sort_values(by=["datetime"], ascending=True).reset_index()
    scores.to_csv(
        str(
            Path(
                global_options.OUTPUT_FOLDER,
                "scores",
                "firm_scores_{}_{}.csv".format(method, global_options.DIMS[0]),
            )
        ),
        index=False,
        float_format="%.4f",
    )