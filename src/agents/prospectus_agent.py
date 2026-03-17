"""
Prospectus Parsing Agent for Preferred Stock Term Extraction.

This agent uses Gemini to read SEC prospectus filings and extract
structured terms for preferred securities, including:

- Security name and series
- Coupon rate (fixed, floating, or fixed-to-floating)
- Par value and liquidation preference
- Call date and call price
- Maturity date (if any)
- Dividend frequency and type (cumulative vs non-cumulative)
- QDI eligibility (Qualified Dividend Income)
- Seniority in the capital structure
- Conversion features (if any)

The agent is designed to work within the LangGraph swarm architecture
and can be called as a standalone function or as a graph node.

Usage:
    from src.agents.prospectus_agent import extract_terms, extract_terms_from_text

    # From a filing dict (downloads text automatically)
    terms = extract_terms(filing_dict)

    # From raw prospectus text
    terms = extract_terms_from_text(prospectus_text, ticker="JPM-PD")
"""

import json
import os
import re
import sys
from typing import Any, Dict, Optional

# Allow direct script execution via:
#   python3 src/agents/prospectus_agent.py JPM-PD
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import get_llm


# ---------------------------------------------------------------------------
# Extraction Prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are a fixed-income analyst specializing in preferred securities.
You are reading a SEC prospectus filing (Form 424B2 or 424B5) for a preferred stock issuance.

Your task is to extract the key terms of the preferred security from the prospectus text below.
Return ONLY a valid JSON object with the following fields. Use null for any field you cannot determine from the text.

Required JSON fields:
{{
    "security_name": "Full official name of the security (e.g., '6.00% Non-Cumulative Preferred Stock, Series DD')",
    "issuer": "Name of the issuing company",
    "series": "Series designation (e.g., 'Series DD', 'Series L')",
    "ticker": "Trading ticker if mentioned (e.g., 'JPM-PD')",
    "par_value": "Par value or liquidation preference per share in dollars (number only, e.g., 25.00)",
    "coupon_rate": "Annual coupon/dividend rate as a percentage (number only, e.g., 6.00)",
    "coupon_type": "One of: 'fixed', 'floating', 'fixed-to-floating', 'adjustable'",
    "floating_benchmark": "If floating or fixed-to-floating, the benchmark rate (e.g., 'SOFR', '3-month LIBOR'). null if fixed.",
    "floating_spread": "If floating, the spread over the benchmark in basis points (number only). null if fixed.",
    "fixed_to_floating_date": "If fixed-to-floating, the date when it switches to floating (YYYY-MM-DD). null otherwise.",
    "dividend_frequency": "One of: 'quarterly', 'semi-annual', 'monthly', 'annual'",
    "cumulative": "true if cumulative dividends, false if non-cumulative",
    "qdi_eligible": "true if dividends qualify for QDI tax treatment, false if not, null if not mentioned",
    "call_date": "First optional redemption date (YYYY-MM-DD). null if not callable.",
    "call_price": "Redemption price per share in dollars (number only). Usually par value.",
    "maturity_date": "Maturity date (YYYY-MM-DD). null if perpetual.",
    "perpetual": "true if the security has no maturity date, false otherwise",
    "conversion_feature": "Brief description of any conversion feature, or null if none",
    "listing_exchange": "Exchange where the security is listed (e.g., 'NYSE', 'NASDAQ')",
    "deposit_shares": "true if the security is issued as depositary shares representing a fraction of a preferred share, false otherwise",
    "deposit_fraction": "If depositary shares, the fraction each share represents (e.g., '1/400th'). null otherwise.",
    "seniority": "Position in capital structure (e.g., 'senior to common stock, junior to all debt')",
    "use_of_proceeds": "Brief summary of the intended use of proceeds, or null if not stated",
    "total_offering_amount": "Total dollar amount of the offering (e.g., '$1,500,000,000'). null if not stated.",
    "confidence_score": "Your confidence in the extraction accuracy from 0.0 to 1.0"
}}

IMPORTANT RULES:
- Return ONLY the JSON object, no other text before or after it.
- Do not use em dashes in any text fields.
- For dates, use YYYY-MM-DD format. If only month and year are given, use the first of the month.
- For dollar amounts in par_value and call_price, return just the number (e.g., 25.00 not "$25.00").
- For coupon_rate, return just the number (e.g., 6.00 not "6.00%").
- If the prospectus is for depositary shares, extract the terms of the underlying preferred stock but note the depositary structure.
- If you cannot determine a field with confidence, use null rather than guessing.

PROSPECTUS TEXT:
{prospectus_text}
"""


# ---------------------------------------------------------------------------
# Core Extraction Functions
# ---------------------------------------------------------------------------

def extract_terms_from_text(
    prospectus_text: str,
    ticker: str = "",
    max_text_length: int = 40000,
) -> Dict[str, Any]:
    """
    Extract structured terms from raw prospectus text using Gemini.

    Args:
        prospectus_text: Plain text content of the prospectus.
        ticker: Optional ticker symbol for context.
        max_text_length: Maximum characters of prospectus text to send to the LLM.

    Returns:
        Dictionary of extracted terms, or an error dict if extraction fails.
    """
    if not prospectus_text or len(prospectus_text.strip()) < 100:
        return {
            "error": "Prospectus text is too short or empty",
            "ticker": ticker,
            "confidence_score": 0.0,
        }

    # Truncate if needed (prospectuses can be very long)
    text = prospectus_text[:max_text_length]

    # Build the prompt
    prompt = EXTRACTION_PROMPT.format(prospectus_text=text)

    try:
        llm = get_llm(temperature=0.1)  # Low temperature for factual extraction
        response = llm.invoke(prompt)

        # Extract the text content from the response
        content = response.content
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    text_parts.append(block["text"])
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "\n".join(text_parts)

        # Parse the JSON from the response
        terms = _parse_json_response(content)

        # Add the ticker if not already present
        if ticker and not terms.get("ticker"):
            terms["ticker"] = ticker

        return terms

    except Exception as e:
        return {
            "error": f"LLM extraction failed: {str(e)}",
            "ticker": ticker,
            "confidence_score": 0.0,
        }


def extract_terms(
    filing: Dict[str, Any],
    pipeline=None,
) -> Dict[str, Any]:
    """
    Extract terms from a filing dict by downloading the prospectus and
    running the LLM extraction.

    Args:
        filing: A filing dict from the EdgarPipeline.
        pipeline: Optional EdgarPipeline instance for downloading.

    Returns:
        Dictionary of extracted terms.
    """
    if pipeline is None:
        from src.data.edgar_pipeline import EdgarPipeline
        pipeline = EdgarPipeline()

    # Download the filing text
    text = pipeline.download_filing(filing, max_chars=40000)
    if not text:
        return {
            "error": "Could not download filing text",
            "accession_number": filing.get("accession_number", ""),
            "confidence_score": 0.0,
        }

    # Extract tickers from the filing metadata
    tickers = filing.get("tickers", [])
    ticker = tickers[0] if tickers else ""

    # Run the extraction
    terms = extract_terms_from_text(text, ticker=ticker)

    # Add filing metadata
    terms["accession_number"] = filing.get("accession_number", "")
    terms["filing_date"] = filing.get("filing_date", "")
    terms["filing_url"] = filing.get("url", "")
    terms["issuer_cik"] = filing.get("issuer_cik", "")

    return terms


# ---------------------------------------------------------------------------
# LangGraph Node Function
# ---------------------------------------------------------------------------

def prospectus_agent_node(state: dict) -> dict:
    """
    LangGraph node function for the Prospectus Parsing Agent.

    Reads the prospectus text from the state (or downloads it from EDGAR),
    extracts structured terms, and writes them back to the state.

    Expected state keys:
        - ticker: str (the preferred stock ticker)
        - prospectus_text: str (optional, raw prospectus text)
        - prospectus_filing: dict (optional, filing dict from EdgarPipeline)

    Writes to state:
        - prospectus_terms: dict (extracted terms)
        - agent_status: dict (records success or failure under "prospectus")
        - errors: list (appends error if extraction fails)
    """
    ticker = state.get("ticker", "")
    prospectus_text = state.get("prospectus_text", "")
    filing = state.get("prospectus_filing", {})

    status_updates = {}
    error_updates = []

    if prospectus_text:
        # Text already provided, just extract
        terms = extract_terms_from_text(prospectus_text, ticker=ticker)
    elif filing:
        # Download and extract from filing
        terms = extract_terms(filing)
    else:
        # Need to search EDGAR for the filing first
        from src.data.edgar_pipeline import fetch_preferred_prospectus
        filings, text = fetch_preferred_prospectus(ticker)

        if text:
            terms = extract_terms_from_text(text, ticker=ticker)
            if filings:
                best_filing = filings[0]
                terms.setdefault("accession_number", best_filing.get("accession_number", ""))
                terms.setdefault("filing_date", best_filing.get("filing_date", ""))
                terms.setdefault("filing_url", best_filing.get("url", ""))
                terms.setdefault("issuer_cik", best_filing.get("issuer_cik", ""))
        else:
            terms = {
                "error": f"No prospectus found for {ticker} on EDGAR",
                "ticker": ticker,
                "confidence_score": 0.0,
            }

    # Check for extraction errors
    if terms.get("error"):
        error_updates.append(f"Prospectus Agent: {terms['error']}")
        status_updates["prospectus"] = "failed"
    else:
        status_updates["prospectus"] = "success"

    return {
        "prospectus_terms": terms,
        "agent_status": status_updates,
        "errors": error_updates,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_response(text: str) -> Dict[str, Any]:
    """
    Parse a JSON object from an LLM response, handling common formatting issues.

    The LLM sometimes wraps JSON in markdown code blocks or adds extra text.
    This function strips those artifacts and extracts the JSON.
    """
    # Remove markdown code block markers
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    # Try direct parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Last resort: return error
    return {
        "error": "Could not parse JSON from LLM response",
        "raw_response": text[:500],
        "confidence_score": 0.0,
    }


def format_terms_report(terms: Dict[str, Any]) -> str:
    """
    Format extracted terms into a human-readable Markdown report.

    Args:
        terms: Dictionary of extracted terms.

    Returns:
        Formatted Markdown string.
    """
    if terms.get("error"):
        return f"**Extraction Error:** {terms['error']}"

    lines = []
    lines.append(f"## {terms.get('security_name', 'Unknown Security')}")
    lines.append("")

    # Core terms table
    lines.append("| Field | Value |")
    lines.append("|---|---|")

    field_labels = {
        "issuer": "Issuer",
        "series": "Series",
        "ticker": "Ticker",
        "par_value": "Par Value",
        "coupon_rate": "Coupon Rate",
        "coupon_type": "Coupon Type",
        "floating_benchmark": "Floating Benchmark",
        "floating_spread": "Floating Spread (bps)",
        "fixed_to_floating_date": "Fixed-to-Floating Date",
        "dividend_frequency": "Dividend Frequency",
        "cumulative": "Cumulative",
        "qdi_eligible": "QDI Eligible",
        "call_date": "First Call Date",
        "call_price": "Call Price",
        "maturity_date": "Maturity Date",
        "perpetual": "Perpetual",
        "listing_exchange": "Exchange",
        "deposit_shares": "Depositary Shares",
        "deposit_fraction": "Depositary Fraction",
        "seniority": "Seniority",
        "total_offering_amount": "Offering Amount",
        "confidence_score": "Confidence Score",
    }

    for key, label in field_labels.items():
        value = terms.get(key)
        if value is not None:
            if key == "coupon_rate":
                value = f"{value}%"
            elif key == "par_value" or key == "call_price":
                value = f"${value}"
            elif key == "floating_spread":
                value = f"{value} bps"
            elif key == "confidence_score":
                value = f"{value:.0%}" if isinstance(value, (int, float)) else value
            elif isinstance(value, bool):
                value = "Yes" if value else "No"
            lines.append(f"| {label} | {value} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "JPM-PD"
    print(f"\n{'='*60}")
    print(f"Prospectus Term Extraction: {ticker}")
    print(f"{'='*60}")

    # Search EDGAR for the prospectus
    from src.data.edgar_pipeline import fetch_preferred_prospectus

    print(f"\nSearching EDGAR for {ticker} prospectus...")
    filings, text = fetch_preferred_prospectus(ticker)

    if not text:
        print("Could not download prospectus text.")
        print(f"Found {len(filings)} filings but download failed (likely SEC rate limiting).")
        if filings:
            print(f"\nBest match: {filings[0]['filing_date']} | {filings[0]['form_type']}")
            print(f"URL: {filings[0]['url']}")
        sys.exit(1)

    print(f"Downloaded {len(text)} chars of prospectus text")
    print(f"\nExtracting terms with Gemini...")

    terms = extract_terms_from_text(text, ticker=ticker)

    if terms.get("error"):
        print(f"\nExtraction error: {terms['error']}")
    else:
        print(f"\n{format_terms_report(terms)}")

    # Also print raw JSON for debugging
    print(f"\n{'='*60}")
    print("Raw extracted terms (JSON):")
    print(json.dumps(terms, indent=2, default=str))
