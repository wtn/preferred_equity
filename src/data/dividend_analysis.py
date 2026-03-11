"""
Dividend analysis utilities for preferred equity securities.
Computes dividend consistency, growth patterns, and coverage metrics.
"""

import pandas as pd
import yfinance as yf
from typing import Optional


def analyze_dividend_pattern(ticker: str) -> dict:
    """
    Analyze the dividend payment pattern for a preferred stock.
    
    Computes:
    - Payment frequency (monthly, quarterly, semi-annual, annual)
    - Consistency score (how regular are the payments)
    - Average payment amount and standard deviation
    - Any missed or irregular payments
    
    Args:
        ticker: The preferred stock ticker
    
    Returns:
        Dictionary with dividend pattern analysis
    """
    try:
        stock = yf.Ticker(ticker)
        dividends = stock.dividends
        
        if dividends.empty or len(dividends) < 2:
            return {
                "ticker": ticker,
                "has_dividend_history": False,
                "error": "Insufficient dividend history"
            }
        
        # Convert to DataFrame for analysis
        df = dividends.to_frame(name="amount")
        df.index = pd.to_datetime(df.index)
        
        # Calculate payment intervals
        intervals = df.index.to_series().diff().dropna()
        avg_interval_days = intervals.dt.days.mean()
        
        # Determine payment frequency
        if avg_interval_days < 45:
            frequency = "monthly"
        elif avg_interval_days < 120:
            frequency = "quarterly"
        elif avg_interval_days < 200:
            frequency = "semi-annual"
        else:
            frequency = "annual"
        
        # Calculate consistency score (lower std dev of intervals = more consistent)
        interval_std = intervals.dt.days.std()
        if interval_std < 5:
            consistency = "excellent"
            consistency_score = 1.0
        elif interval_std < 15:
            consistency = "good"
            consistency_score = 0.8
        elif interval_std < 30:
            consistency = "fair"
            consistency_score = 0.6
        else:
            consistency = "irregular"
            consistency_score = 0.3
        
        # Payment amount analysis
        amounts = df["amount"]
        avg_payment = round(float(amounts.mean()), 4)
        payment_std = round(float(amounts.std()), 4)
        min_payment = round(float(amounts.min()), 4)
        max_payment = round(float(amounts.max()), 4)
        
        # Check for payment changes (fixed vs variable rate indicator)
        unique_amounts = amounts.round(4).nunique()
        is_fixed_rate = unique_amounts <= 2  # Allow for minor rounding
        
        # Recent trend (last 8 payments vs prior 8)
        if len(amounts) >= 16:
            recent_avg = float(amounts.tail(8).mean())
            prior_avg = float(amounts.iloc[-16:-8].mean())
            if recent_avg > prior_avg * 1.01:
                trend = "increasing"
            elif recent_avg < prior_avg * 0.99:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_history"
        
        # Annual yield calculation from dividends
        annual_dividends = float(amounts.tail(4).sum()) if frequency == "quarterly" else float(amounts.tail(12).sum()) if frequency == "monthly" else float(amounts.tail(1).sum())
        
        return {
            "ticker": ticker,
            "has_dividend_history": True,
            "total_payments_recorded": len(dividends),
            "first_payment_date": str(df.index[0].date()),
            "last_payment_date": str(df.index[-1].date()),
            "frequency": frequency,
            "avg_interval_days": round(avg_interval_days, 1),
            "consistency": consistency,
            "consistency_score": consistency_score,
            "avg_payment": avg_payment,
            "payment_std": payment_std,
            "min_payment": min_payment,
            "max_payment": max_payment,
            "is_fixed_rate": is_fixed_rate,
            "unique_payment_amounts": unique_amounts,
            "trend": trend,
            "trailing_annual_dividends": round(annual_dividends, 4),
        }
        
    except Exception as e:
        return {
            "ticker": ticker,
            "has_dividend_history": False,
            "error": str(e)
        }
