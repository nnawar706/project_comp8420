"""
Central configuration for the Intelligent Customer Service System.

Everything tunable lives here so notebooks, scripts and the Streamlit app all
read from one place. Paths are resolved relative to this file, so the project
runs the same no matter what directory you launch it from.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
KB_DIR = DATA_DIR / "knowledge_base"

MODELS_DIR = ROOT / "models"
VECTOR_DB_DIR = ROOT / "vector_db"
EVAL_DIR = ROOT / "evaluation"
FIG_DIR = EVAL_DIR / "figures"

for _d in (RAW_DIR, PROCESSED_DIR, KB_DIR, MODELS_DIR, VECTOR_DB_DIR, EVAL_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Saved artifacts
CLASSIFIER_PATH = MODELS_DIR / "tfidf_classifier.joblib"
RAW_CSV_PATH = RAW_DIR / "bitext_customer_support.csv"
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
VAL_PATH = PROCESSED_DIR / "val.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
HF_DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
# Direct CSV fallback if the `datasets` library route fails on the user's machine.
HF_CSV_FALLBACK_URL = (
    "https://huggingface.co/datasets/"
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset/"
    "resolve/main/bitext-customer-support-llm-chatbot-training-dataset.csv"
)
# We auto-detect columns, but these are the preferred names from Bitext.
TEXT_COL = "instruction"          # the customer message
CATEGORY_COL = "category"         # 10/11-way coarse label  -> classification target
INTENT_COL = "intent"             # 27-way fine label       -> optional harder target
RESPONSE_COL = "response"         # gold agent reply        -> RAG seed + eval reference

RANDOM_STATE = 42
TEST_SIZE = 0.30                  # 70 / 15 / 15 train/val/test (val=test=half of temp)

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
SPACY_MODEL = "en_core_web_sm"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Transformer sentiment model (optional, lazily loaded).
HF_SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# --------------------------------------------------------------------------- #
# Ollama (local LLM) - override with env vars if needed.
# --------------------------------------------------------------------------- #
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.2")   # `ollama pull llama3.2`
LLM_TEMPERATURE = 0.2
LLM_TIMEOUT = 120                 # seconds per generation call

# --------------------------------------------------------------------------- #
# RAG
# --------------------------------------------------------------------------- #
CHROMA_COLLECTION = "company_knowledge"
RAG_TOP_K = 4
CHUNK_SIZE = 600                  # characters
CHUNK_OVERLAP = 100

# --------------------------------------------------------------------------- #
# Decision thresholds (confidence-gating / escalation)
# --------------------------------------------------------------------------- #
CLASSIFIER_CONF_THRESHOLD = 0.55  # below this -> escalate
SENTIMENT_ESCALATE_COMPOUND = -0.6  # VADER compound below this -> priority/escalate
LLM_CONF_THRESHOLD = 0.6          # self-reported LLM confidence below this -> escalate
