import openai
import os
import tiktoken
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv
from pathlib import Path

# This Config file is tailored to DeepSeek-r1 given it uses OpenRouter
# for API calls. If you want to use OpenAI directly, you can modify the
# `base_url` in the OpenAI client initialization below.#

#  Where to store downloaded notebooks:
SOLUTIONS_DIR = Path("solutions")

# Training data for new competitions
COMP_TRAIN_DIR = Path("train")

#  Where to store TF-IDF indices, pickles, etc.:
INDEX_DIR = Path("index_data")


EXCEL_FILE = Path("notebooks_and_competitions_structured.xlsx")


load_dotenv() # Load environment variables from .env file

# --- Kaggle API Client ---
kaggle_api = KaggleApi()
kaggle_api.authenticate()

# --- OpenRouter/OpenAI API Client ---

# 1. Get the API key from the environment.
#    Ensure your .env file has: OPENAI_API_KEY=sk-or-v1-yourkey...
api_key = os.getenv("OPENAI_API_KEY")

# 2. Define the required headers for OpenRouter.
#    This tells OpenRouter who is making the request.
default_headers = {
    "HTTP-Referer": "https://github.com/IllyaGY/REU.git",
    "X-Title": "REU Project", 
}

# 3. Create a dedicated client for making API calls.
#    This client is configured with the key, the OpenRouter URL, and the required headers.
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    default_headers=default_headers,
)

# 4. Define the model and tokenizer constants
OPENAI_MODEL = "deepseek/deepseek-r1"
ENCODER = tiktoken.get_encoding("cl100k_base")
MAX_NOTEBOOK_TOKENS = 1000


MAX_NOTEBOOK_TOKENS = 1000

MAX_FEATURES = 50
MAX_CLASSES = 50