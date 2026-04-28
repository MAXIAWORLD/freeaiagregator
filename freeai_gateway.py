import os
import sqlite3
import datetime
import logging
from typing import Optional

import litellm
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("FreeIAGateway")

PROVIDERS = [
    {"model": "gemini/gemini-2.0-flash-exp", "key": "google",       "req_limit": 1500, "tok_limit": 1_000_000},
    {"model": "groq/llama-3.1-70b-versatile","key": "groq",         "req_limit": 500,  "tok_limit": 500_000},
    {"model": "mistral/mistral-large-latest", "key": "mistral",      "req_limit": 100,  "tok_limit": 200_000},
    {"model": "huggingface/meta-llama/Llama-3.1-70B-Instruct", "key": "huggingface", "req_limit": 1000, "tok_limit": 500_000},
]
