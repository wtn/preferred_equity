"""
Configuration module for the Preferred Equity Analysis Swarm.
Loads environment variables and provides shared settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

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
    
    Uses ChatGoogleGenerativeAI which connects directly to Google's API
    using your GOOGLE_API_KEY. This works with any Gemini model available
    in Google AI Studio.
    
    Args:
        temperature: Controls randomness in responses (0.0 to 1.0).
                     Lower values are more deterministic.
    
    Returns:
        A LangChain chat model instance configured for Gemini.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Please add it to your .env file.\n"
            "You can get a key from https://aistudio.google.com/apikey"
        )
    
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
    )
