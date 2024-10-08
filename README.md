# Measuring Corporate Culture Using Machine Learning

## Introduction
The repository implements the method described in the paper 

Kai Li, Feng Mai, Rui Shen, Xinyan Yan, [__Measuring Corporate Culture Using Machine Learning__](https://academic.oup.com/rfs/advance-article-abstract/doi/10.1093/rfs/hhaa079/5869446?redirectedFrom=fulltext), _The Review of Financial Studies_, 2020; DOI:[10.1093/rfs/hhaa079](http://dx.doi.org/10.1093/rfs/hhaa079) 
[[Available at SSRN]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3256608)

The code is tested on Ubuntu 18.04 and macOS Catalina, with limited testing on Windows 10.  

## Requirement
The code requres 
- `Python 3.6+`
- The required Python packages can be installed via `pip install -r requirements.txt`
- Download and uncompress [Stanford CoreNLP v3.9.2](http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip). Newer versions may work, but they are not tested. Either [set the environment variable to the location of the uncompressed folder](https://stanfordnlp.github.io/stanfordnlp/corenlp_client.html), or edit the following line in the `global_options.py` to the location of the uncompressed folder, for example: 
> os.environ["CORENLP_HOME"] = "/home/user/stanford-corenlp-full-2018-10-05/"   

- If you are using Windows, use "/" instead of "\\" to separate directories.  
- Make sure [requirements for CoreNLP](https://stanfordnlp.github.io/CoreNLP/) are met. For example, you need to have Java installed (if you are using Windows, install [Windows Offline (64-bit) version](https://java.com/en/download/manual.jsp)). To check if CoreNLP is set up correctly, use command line (terminal) to navigate to the project root folder and run `python -m culture.preprocess`. You should see parsed outputs from a single sentence printed after a moment:

    (['when[pos:WRB] I[pos:PRP] be[pos:VBD] a[pos:DT]....

## Data
We included some example data in the `data/input/` folder. The three files are
- `documents.txt`: Each line is a document (e.g., each earnings call). Each document needs to have line breaks remvoed. The file has no header row. 
- `document_ids.txt`: Each line is document ID (e.g., unique identifier for each earnings call). A document ID cannot have `_` or whitespaces. The file has no header row. 
- (Optional) `id2firms.csv`: A csv file with three columns (`document_id`:str, `firm_id`:str, `time`:int). The file has a header row. 


## Before running the code
You can config global options in the `global_options.py`. The most important options are perhaps:
- The RAM allocated for CoreNLP
- The number of CPU cores for CoreNLP parsing and model training
- The seed words
- The max number of words to include in each dimension. Note that after filtering and deduplication (each word can only be loaded under a single dimension), the number of words will be smaller. 


## Running the code
1. Use `python parse.py` to use Stanford CoreNLP to parse the raw documents. This step is relatvely slow so multiple CPU cores is recommended. The parsed files are output in the `data/processed/parsed/` folder:
    - `documents.txt`: Each line is a *sentence*. 
    - `document_sent_ids.txt`: Each line is a id in the format of `docID_sentenceID` (e.g. doc0_0, doc0_1, ..., doc1_0, doc1_1, doc1_2, ...). Each line in the file corresponds to `documents.txt`. 
    
    Note about performance: This step is time-consuming (~10 min for 100 calls). Using `python parse_parallel.py` can speed up the process considerably (~2 min with 8 cores for 100 calls) but it is not well-tested on all platforms. To not break things, the two implementations are separated. 

2. Use `python clean_and_train.py` to clean, remove stopwords, and named entities in parsed `documents.txt`. The program then learns corpus specific phrases using gensim and concatenate them. Finally, the program trains the `word2vec` model. 

    The options can be configured in the `global_options.py` file. The program outputs the following 3 output files:
    - `data/processed/unigram/documents_cleaned.txt`: Each line is a *sentence*. NERs are replaced by tags. Stopwords, 1-letter words, punctuation marks, and pure numeric tokens are removed. MWEs and compound words are concatenated. 
    - `data/processed/bigram/documents_cleaned.txt`: Each line is a *sentence*. 2-word phrases are concatenated.  
    - `data/processed/trigram/documents_cleaned.txt`: Each line is a *sentence*. 3-word phrases are concatenated. This is the final corpus for training the word2vec model and scoring. 

   The program also saves the following gensim models:
   - `models/phrases/bigram.mod`: phrase model for 2-word phrases
   - `models/phrases/trigram.mod`: phrase model for 3-word phrases
   - `models/w2v/w2v.mod`: word2vec model
   
3. Use `python create_dict.py` to create the expanded dictionary. The program outputs the following files:
    - `outputs/dict/expanded_dict.csv`: A csv file with the number of columns equal to the number of dimensions in the dictionary (five in the paper). The row headers are the dimension names. 
    
    (Optional): It is possible to manually remove or add items to the `expanded_dict.csv` before scoring the documents. 

4. Use `clean_expanded_dict.py` to clean the expanded dictionary. The program outputs the following files:
    - `outputs/dict/expanded_dict_{topic_name}.csv`: A csv file with the number of CSV files. The number of files equal to the number of dimensions in the dictionary (14 in the paper). The row headers are the dimension names. 

    (Optional): It is possible to manually remove or add items to the `expanded_dict.csv` before scoring the documents.

5. Use `python score.py` to score the documents. Note that the output scores for the documents are not adjusted by the document length. The program outputs three sets of scores: 
    - `outputs/scores/TF/scores_TF_{topic}.csv`: using raw term counts or term frequency (TF),
    - `outputs/scores/TFIDF/scores_TFIDF_{topic}.csv`: using TF-IDF weights, 
    - `outputs/scores/WFIDF_scores_WFIDF_{topic}.csv`: TF-IDF with Log normalization (WFIDF). 
    if the folder `TF' does not exist, the program will create it.
    (Optional): It is possible to use additional weights on the words (see `score.score_tf_idf()` for detail).  

<!-- 6. (Optional): Use `python aggregate_firms.py` to aggregate the scores to the firm-time level. The final scores are adjusted by the document lengths.  -->


6. Use `python aggregate_daily.py` to aggregate the reddit post level scores to daily level.

    - `outputs/scores/TF/bitcoin_scores_TF.csv`: This file contains the scores for each daily aggregation of reddit posts related to bitcoin, calculated using raw term counts or term frequency (TF).

    - `outputs/scores/TFIDF/bitcoin_scores_TFIDF.csv`: This file contains the scores for each daily aggregation of reddit posts related to bitcoin, calculated using TF-IDF weights.

    - `outputs/scores/WFIDF/bitcoin_scores_WFIDF.csv`: This file contains the scores for each daily aggregation of reddit posts related to bitcoin, calculated using TF-IDF with Log normalization (WFIDF).


7. Use `reddit_attention_sentiment.py' to obtain the daily attention and sentiment index of social media investors on reddits' subbreddit platform r/bitcoin. The program outputs the following files:

    - `src/reddit_data/bitcoin_attention_sentiment.csv`: This file contains the daily attention index of social media investors on reddits' subbreddit platform r/bitcoin.


8. Use `pre_analysis_merge.py' to aggregate the daily files to weekly files, including the daily narratives, attention of retail investors (number of posts on subreddit r/bitcoin), and sentiment of retail investors (sentiment of posts on subreddit r/bitcoin). The program outputs the following files:

    - `src/kai../output/weekly_posts_narrative_tone_ltm3_attention_sentiment.csv`: This file contains the weekly narratives, attention of retail investors (number of posts on subreddit r/bitcoin), and sentiment of retail investors (sentiment of posts on subreddit r/bitcoin).


9. Use `data_visualization.py' to plot the pattern of bitcoin return with ltm 3factors, narratives, attention and sentiment of retail investors. The program outputs the following files:

    - the graph has been saved as pdf in folder `Crypto/output/figure/{}.pdf'.format('bitcoin_return_ltm3factors_narrative_attention_sentiment')`
    to run this program, you need to cd another virualenv with the following packages:
    - `pandas`
    - `matplotlib`
    - `numpy`
    - `seaborn`
    - `datetime`
    It's relatively hard to run in python 3.8.10. So I use cd /mnt/e/Github_projects/virtualenvs/_nlp_tpm and run the program in python 3.10.12.
    first open another terminal and run the following command:
    1. cd /mnt/e/Github_projects/virtualenvs/_nlp_tpm
    2. .\activate
    3. python path_to_file/data_visualization.py
    It saves four figures in the folder `Crypto/output/figure/`
