"""
Market data fetcher for preferred equity securities.
Uses yfinance for price data and basic security information.
Handles preferred stock ticker variants (e.g., C-PJ, C-PRJ, C-P-J) for better reliability.
"""

import yfinance as yf
import pandas as pd
from typing import Optional, List


def get_ticker_variants(ticker: str) -> List[str]:
    """
    Generate common Yahoo Finance ticker variants for preferred stocks.
    
    Example: 'C-PJ' -> ['C-PJ', 'C-PRJ', 'C-P-J', 'C.PJ']
    """
    variants = [ticker]
    
    # If ticker has a dash (like C-PJ or JPM-PD), try common variations
    if '-' in ticker:
        parts = ticker.split('-')
        if len(parts) == 2:
            base = parts[0]
            series = parts[1]
            
            # Extract series letter (e.g., 'J' from 'PJ' or 'PD' from 'PD')
            series_letter = series[1:] if series.startswith('P') and len(series) > 1 else series
            
            # Standard variants
            variants.append(f"{base}-PR{series_letter}")  # C-PRJ
            variants.append(f"{base}-P-{series_letter}")  # C-P-J
            variants.append(f"{base}-P{series_letter}")   # C-PJ (redundant but safe)
            variants.append(f"{base}.PR{series_letter}")  # C.PRJ
            variants.append(f"{base}.P{series_letter}")   # C.PJ
            variants.append(f"{base}p{series_letter}")    # CpJ
            variants.append(f"{base}pr{series_letter}")   # CprJ
            variants.append(f"{base}-{series_letter}")    # C-J
            variants.append(f"{base}.{series_letter}")    # C.J
            variants.append(f"{base}p{series_letter.lower()}") # Cpj
            
            # Special case for some brokers: C.PR.J
            variants.append(f"{base}.PR.{series_letter}")
    
    # Remove duplicates while preserving order
    seen = set()
    return [x for x in variants if not (x in seen or seen.add(x))]


def get_preferred_info(ticker: str) -> dict:
    """
    Fetch basic information about a preferred stock from Yahoo Finance.
    Attempts multiple ticker variants to handle different platform conventions.
    """
    variants = get_ticker_variants(ticker)
    last_error = None
    
    for variant in variants:
        try:
            stock = yf.Ticker(variant)
            info = stock.info
            
            # Check if we got meaningful data (some variants return empty dicts)
            if not info or ("regularMarketPrice" not in info and "previousClose" not in info):
                continue
                
            result = {
                "ticker": ticker,
                "yahoo_ticker": variant,
                "name": info.get("longName", info.get("shortName", "Unknown")),
                "price": info.get("regularMarketPrice", info.get("previousClose", None)),
                "dividend_rate": info.get("dividendRate", None),
                "dividend_yield": info.get("dividendYield", None),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh", None),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow", None),
                "volume": info.get("averageVolume", None),
                "sector": info.get("sector", None),
                "industry": info.get("industry", None),
                "currency": info.get("currency", "USD"),
            }
            return result
        except Exception as e:
            last_error = str(e)
            continue
            
    return {"ticker": ticker, "error": last_error or "No data found for ticker or variants"}


def get_price_history(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """
    Fetch historical price data for a preferred stock.
    Attempts multiple ticker variants.
    """
    variants = get_ticker_variants(ticker)
    for variant in variants:
        try:
            stock = yf.Ticker(variant)
            hist = stock.history(period=period)
            if not hist.empty:
                return hist
        except Exception:
            continue
    return None


def get_dividend_history(ticker: str) -> Optional[pd.DataFrame]:
    """
    Fetch dividend payment history for a preferred stock.
    Attempts multiple ticker variants to find the one with dividend data.
    """
    variants = get_ticker_variants(ticker)
    for variant in variants:
        try:
            stock = yf.Ticker(variant)
            dividends = stock.dividends
            if not dividends.empty:
                return dividends.to_frame(name="dividend")
        except Exception:
            continue
    return None
