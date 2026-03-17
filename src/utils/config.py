"""
Configuration module for the Preferred Equity Analysis Swarm.
Loads environment variables and provides shared settings.

LLM priority:
  1. Google Gemini via GOOGLE_API_KEY (primary)
  2. OpenAI-compatible API via OPENAI_API_KEY (fallback)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")  # leave empty to use default

# SEC EDGAR Configuration
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "PreferredEquitySwarm research@example.com")

# FRED Configuration
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
UNIVERSE_DIR = os.path.join(DATA_DIR, "universe")


def get_llm(temperature: float = 0.3):
    """
    Factory function that returns the configured LLM instance.

    Priority:
      1. If GOOGLE_API_KEY is set, uses ChatGoogleGenerativeAI (Gemini).
      2. If OPENAI_API_KEY is set, uses ChatOpenAI (OpenAI-compatible API).
      3. Raises ValueError if neither key is available.

    Args:
        temperature: Controls randomness in responses (0.0 to 1.0).
                     Lower values are more deterministic.

    Returns:
        A LangChain chat model instance.
    """
    # Primary: Google Gemini
    if GOOGLE_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
        )

    # Fallback: OpenAI-compatible API
    if OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": OPENAI_MODEL,
            "api_key": OPENAI_API_KEY,
            "temperature": temperature,
        }
        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL

        return ChatOpenAI(**kwargs)

    raise ValueError(
        "No LLM API key is configured. Set either GOOGLE_API_KEY or "
        "OPENAI_API_KEY in your .env file.\n"
        "  Google: https://aistudio.google.com/apikey\n"
        "  OpenAI: https://platform.openai.com/api-keys"
    )
