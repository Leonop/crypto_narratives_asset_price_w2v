"""Glo
bal options for analysis
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

# Hardware options
N_CORES: int = 8  # max number of CPU cores to use
RAM_CORENLP: str = "32G"  # max RAM allocated for parsing using CoreNLP; increase to speed up parsing
PARSE_CHUNK_SIZE: int = 100 # number of lines in the input file to process uing CoreNLP at once. Increase on workstations with larger RAM (e.g. to 1000 if RAM is 64G)  

# Directory locations
os.environ[
    "CORENLP_HOME"
] = "/mnt/g/Dropbox/XiaoZhang/Crypto/src/Kai_xinyan_corporate_culture_2020RFS/stanford-corenlp-full-2018-10-05/stanford-corenlp-full-2018-10-05"   
# location of the CoreNLP models; use / to seperate folders
DATA_FOLDER: str = "/mnt/g/Dropbox/XiaoZhang/Crypto/src/Kai_xinyan_corporate_culture_2020RFS/data/"
MODEL_FOLDER: str = "/mnt/g/Dropbox/XiaoZhang/Crypto/src/Kai_xinyan_corporate_culture_2020RFS/models/" # will be created if does not exist
OUTPUT_FOLDER: str = "/mnt/g/Dropbox/XiaoZhang/Crypto/src/Kai_xinyan_corporate_culture_2020RFS/outputs/" # will be created if does not exist; !!! WARNING: existing files will be removed !!!

# Parsing and analysis options
STOPWORDS: Set[str] = set(
    Path("resources", "StopWords_Generic.txt").read_text().lower().split()
)  # Set of stopwords from https://sraf.nd.edu/textual-analysis/resources/#StopWords
PHRASE_THRESHOLD: int = 10  # threshold of the phraser module (smaller -> more phrases)
PHRASE_MIN_COUNT: int = 10  # min number of times a bigram needs to appear in the corpus to be considered as a phrase
W2V_DIM: int = 300  # dimension of word2vec vectors
W2V_WINDOW: int = 5  # window size in word2vec
W2V_ITER: int = 20  # number of iterations in word2vec
N_WORDS_DIM: int = 500  # max number of words in each dimension of the dictionary
DICT_RESTRICT_VOCAB = None # change to a fraction number (e.g. 0.2) to restrict the dictionary vocab in the top 20% of most frequent vocab

# Inputs for constructing the expanded dictionary
DIMS: List[str] = ["bubble", "scam", "intrinsic_value", "inflation", "volatility", "regulation", "fear_of_missing_out", "fear_of_loss" "blockchain_technology", "security", "environment", "trading_strategy", "liquidity"]
SEED_WORDS: Dict[str, List[str]] = {
    "bubble": [
                "mania",
                "boom",
                "bust",
                "frenzy",
                "frothy",
                "bubble ",
                "pop",
                "tank",
                "crash",
                "panic",
                "mass"
                "black_monday",
                "bloody",
                "moon",
                "bubbly",
                "fluffy",
                "madness"
                ],
    "scam": [
                "Ponzi",
                "Pyramid",
                "scheme"
                "pump",
                "dump",
                "fraud",
                "scam",  
                "hype",
                "meme",
                "deceit",
                "deception",
                "ripoff",
                "fake",
                "sham",
                ],
    "intrinsic_value": [
                "intrinsic",
                "value",
                "real",
                "true",
                "genuine",
                "cash_flow",
                "inherent",
                "fundamental",
                "essential",
                "intrinsic_value",
                "inherent_value",
                "underlying",
                "core",
                ],
    "inflation": [
                "inflation",
                "fiat",
                "money_printing",
                "money_supply",
                "deflation",
                "purchasing_power",
                "government_debt",
                "debt",
                "deficit",
                "fiscal_policy",
                "monetary_policy",
                "central_bank",
                "interest_rate",
                "inflation_target",
                "price_stability",
                "inflation_rate"
                ],
    "volatility": [
                "volatility",
                "risk",
                "price_swings",
                "fluctuations",
                "unstable",
                "market_turbulence",
                "surge",
                "plunge",
                "exposure",
                "uncertainty",
                "downside",
                "market dip",
                "FUD",
                "fear",
                "correction",
                "pullback",
                "reversal",
                ],
    "security": [
                "security",
                "vulnerability",
                "attack",
                "hack",
                "breach",
                "theft",
                "hacker",
                "ddos",
                "encryption",
                "private",
                "key",
                "phishing",
                "malware",
                "ransomware",
                "2FA",
                "wallet",
                "zero-day",
                "cold_storage",
                ],
    "regulation": [
                "regulation",
                "compliance",
                "government_oversight",
                "legal_risks",
                "crackdown",
                "rules",
                "laws",
                "policy",
                "framework",
                "SEC",
                "supervision",
                "monitoring"
                ],
    "blockchain_technology": [
                "blockchain",
                "technology",
                "innovation",
                "ledger",
                "immutable",
                "distributed",
                "ledger",
                "transparent",
                "immutable",
                "decentralized",
                "smart_contract",
                "block_reward",
                "hash",
                "mining",
                "protocol",
                "fork",
                "scalability",
                "interoperability",
                "consensus",
                "proof_work",
                "proof_stake",
        ],
    "fear_of_missing_out": [
                "FOMO",
                "fear_of_missing_out",
                "missing_out",
                "chasing_gains",
                "emotional_investing",
                "impulse_buying",
                "should_have_bought",
                "left_behind",
                "missed_opportunity",
                "late to party",
                "regret",
                "wish_I_had_bought",
                "BTFD",
                "buy_the_dip",
                "greed",
                "emotions",
                "emotional",
                "jealous",
                "panic_buying",
                ],
    "fear_of_loss": [
                "fear_of_loss",
                "loss_aversion",
                "panic_selling",
                "selling_at_a_loss",
                "cutting losses",
                "stop_winning",
                "take_profit",
                "fear",
                "greed",
                "emotions",
                "sentiment",
                "scared",
                "lock_profit",
                ],
    "environmental": [
                "environmental",
                "sustainability",
                "ESG",
                "ecological",
                "green",
                "eco-friendly",
                "mining",
                "green_mining",
                "green_energy",
                "proof",
                "stake",
                "proof",
                "work",
                "decarbonization",
                "sustainable",
                "climate",
                "greenhouse",
                "gas",
                "emissions"
        ],
    "trading_strategy": [
                "trading",
                "strategy",
                "long",
                "short",
                "margin",
                "leverage",
                "position",
                "entry",
                "exit",
                "stop_loss",
                "take_profit",
                "technical_analysis",
                "fundamental_analysis",
                "sentiment_analysis",
                "trend",
                "breakout",
                "pullback",
                "correction",
                "scalping",
                "day_trading",
                "swing_trading",
                "HODL",
                "DCA",
                "pump_and_dump",
                "whale",
                "arbitrage",
        ],
    "liquidity": [
                "liquidity",
                "market_depth",
                "slippage",
                "spread",
                "bid-ask",
                "bid",
                "ask",
                "volume",
                "market_maker",
                "order_book",
                "swap",
                "whale",
                "price_squeeze",
                "drain",
                "spike",
                "imbalance",    
                ]
}

# Create directories if not exist
Path(DATA_FOLDER, "processed", "parsed").mkdir(parents=True, exist_ok=True)
Path(DATA_FOLDER, "processed", "unigram").mkdir(parents=True, exist_ok=True)
Path(DATA_FOLDER, "processed", "bigram").mkdir(parents=True, exist_ok=True)
Path(DATA_FOLDER, "processed", "trigram").mkdir(parents=True, exist_ok=True)
Path(MODEL_FOLDER, "phrases").mkdir(parents=True, exist_ok=True)
Path(MODEL_FOLDER, "phrases").mkdir(parents=True, exist_ok=True)
Path(MODEL_FOLDER, "w2v").mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER, "dict").mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER, "scores").mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER, "scores", "temp").mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER, "scores", "word_contributions").mkdir(parents=True, exist_ok=True)
