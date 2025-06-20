import openai
import os
import tiktoken
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv

from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Configuration / Constants
# ─────────────────────────────────────────────────────────────────────────────

#  Where to store downloaded notebooks:
SOLUTIONS_DIR = Path("solutions")

# Training data for new competitions
COMP_TRAIN_DIR = Path("train")

#  Where to store TF-IDF indices, pickles, etc.:
INDEX_DIR = Path("index_data")

#  Where to store the structured output Excel:
EXCEL_FILE = Path("notebooks_and_competitions_structured.xlsx")

#  Kaggle API client
kaggle_api = KaggleApi()
kaggle_api.authenticate() 

#  OpenAI settings
load_dotenv()
OPENAI_MODEL = "gpt-4o-mini"
openai.api_key = os.getenv("OPENAI_API_KEY")
#  Tokenizer for truncation
ENCODER = tiktoken.get_encoding("cl100k_base")

#  Maximum tokens from a notebook to send to the LLM
MAX_NOTEBOOK_TOKENS = 1000
