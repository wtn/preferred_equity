"""
Tax and Yield Agent for Preferred Equity Analysis.

Classifies Qualified Dividend Income (QDI) eligibility, computes
tax-equivalent yields for a given investor tax bracket, and compares
after-tax income across instruments.

Key concepts:
  - QDI eligibility: dividends from domestic C-corporations that meet the
    holding period requirement are generally taxed at the long-term capital
    gains rate (0%, 15%, or 20%) rather than the ordinary income rate.
    REIT dividends, trust-issued preferreds, and most foreign-issuer
    preferreds do NOT qualify for QDI.
  - Tax-equivalent yield (TEY): the pre-tax yield an investor would need
    from a fully-taxable instrument to match the after-tax income of a
    QDI-eligible preferred.
  - After-tax yield: the yield remaining after applying the appropriate
    tax rate to the dividend income.

This agent uses prospectus terms and issuer characteristics to make the
QDI determination. It does not provide tax advice.
"""

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Known issuer classifications
# ---------------------------------------------------------------------------

# Major bank holding companies and C-corporations whose preferred dividends
# are generally QDI-eligible.  This is a curated list for the demo universe;
# a production system would use SEC filings or a reference data provider.
KNOWN_C_CORPS = {
    "bank of america",
    "citigroup",
    "jpmorgan chase",
    "jpmorgan",
    "goldman sachs",
    "morgan stanley",
    "wells fargo",
    "us bancorp",
    "pnc financial",
    "truist financial",
    "capital one",
    "state street",
    "bank of new york mellon",
    "bny mellon",
    "charles schwab",
    "metlife",
    "prudential financial",
    "allstate",
    "american express",
    "general electric",
    "at&t",
    "verizon",
}

# REITs and trusts whose preferred dividends are generally NOT QDI-eligible.
KNOWN_NON_QDI_ISSUERS = {
    "annaly capital",
    "agnc investment",
    "two harbors",
    "chimera investment",
    "new york mortgage trust",
    "mfa financial",
    "arbor realty",
    "ready capital",
    "digital realty",
    "public storage",
    "realty income",
    "simon property",
    "vornado realty",
    "sl green realty",
    "boston properties",
}


# ---------------------------------------------------------------------------
# Default tax brackets (2025/2026 federal rates)
# ---------------------------------------------------------------------------

DEFAULT_TAX_BRACKETS = {
    "top_ordinary_rate": 37.0,       # Top federal ordinary income rate
    "qdi_rate": 20.0,               # Top QDI / LTCG rate
    "niit_rate": 3.8,               # Net Investment Income Tax
    "default_state_rate": 5.0,      # Approximate blended state rate
}


def analyze_tax_and_yield(
    market_data: Dict[str, Any],
    prospectus_terms: Dict[str, Any],
    dividend_data: Dict[str, Any],
    investor_bracket: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Produce a tax and yield analysis for a preferred security.

    Args:
        market_data: output from the Market Data Agent
        prospectus_terms: output from the Prospectus Agent
        dividend_data: output from the Dividend Agent
        investor_bracket: optional override for tax rates; defaults to
            top-bracket federal + NIIT + estimated state

    Returns a dictionary with:
      - qdi_eligible: True / False / None (unknown)
      - qdi_classification_reason: explanation of the determination
      - issuer_type: "c_corp", "reit", "trust", "unknown"
      - current_yield_pct: pre-tax current yield
      - after_tax_yield_pct: yield after applying the appropriate tax rate
      - tax_equivalent_yield_pct: TEY if QDI-eligible, else equals current yield
      - effective_tax_rate_pct: the blended rate applied to dividends
      - tax_advantage_bps: basis-point advantage of QDI vs ordinary treatment
      - annual_dividend_per_share: trailing annual dividend
      - after_tax_income_per_share: annual after-tax income
      - tax_summary: one-paragraph narrative
      - methodology: brief description
    """
    bracket = investor_bracket or DEFAULT_TAX_BRACKETS
    result: Dict[str, Any] = {}

    # --- Extract inputs ---
    price = _to_float(market_data.get("price"))
    div_yield_raw = _to_float(market_data.get("dividend_yield"))
    annual_dividend = _to_float(market_data.get("dividend_rate"))
    issuer = str(prospectus_terms.get("issuer", "")).strip()
    prospectus_qdi = prospectus_terms.get("qdi_eligible")
    cumulative = prospectus_terms.get("cumulative")
    coupon_type = str(prospectus_terms.get("coupon_type", "")).lower()

    # Current yield
    if div_yield_raw is not None:
        current_yield_pct = div_yield_raw * 100 if div_yield_raw < 1 else div_yield_raw
    else:
        current_yield_pct = None
    result["current_yield_pct"] = current_yield_pct

    # Annual dividend
    if annual_dividend is None and current_yield_pct is not None and price is not None and price > 0:
        annual_dividend = round(price * current_yield_pct / 100, 4)
    result["annual_dividend_per_share"] = annual_dividend

    # --- QDI classification ---
    issuer_type, qdi_eligible, qdi_reason = _classify_qdi(issuer, prospectus_qdi)
    result["issuer_type"] = issuer_type
    result["qdi_eligible"] = qdi_eligible
    result["qdi_classification_reason"] = qdi_reason
    result["cumulative"] = cumulative

    # --- Tax rates ---
    ordinary_rate = bracket.get("top_ordinary_rate", 37.0)
    qdi_rate = bracket.get("qdi_rate", 20.0)
    niit_rate = bracket.get("niit_rate", 3.8)
    state_rate = bracket.get("default_state_rate", 5.0)

    if qdi_eligible is True:
        # QDI: federal QDI rate + NIIT + state (state may or may not give QDI benefit)
        effective_federal = qdi_rate + niit_rate
        effective_total = effective_federal + state_rate
        tax_treatment = "QDI (qualified dividend income)"
    elif qdi_eligible is False:
        # Ordinary income: top federal rate + NIIT + state
        effective_federal = ordinary_rate + niit_rate
        effective_total = effective_federal + state_rate
        tax_treatment = "Ordinary income"
    else:
        # Unknown: assume ordinary as the conservative case
        effective_federal = ordinary_rate + niit_rate
        effective_total = effective_federal + state_rate
        tax_treatment = "Unknown (conservatively treated as ordinary income)"

    result["effective_tax_rate_pct"] = round(effective_total, 2)
    result["tax_treatment"] = tax_treatment

    # --- After-tax yield ---
    if current_yield_pct is not None:
        after_tax_yield = round(current_yield_pct * (1 - effective_total / 100), 2)
    else:
        after_tax_yield = None
    result["after_tax_yield_pct"] = after_tax_yield

    # --- After-tax income per share ---
    if annual_dividend is not None:
        after_tax_income = round(annual_dividend * (1 - effective_total / 100), 4)
    else:
        after_tax_income = None
    result["after_tax_income_per_share"] = after_tax_income

    # --- Tax-equivalent yield ---
    # TEY = after_tax_yield / (1 - ordinary_tax_rate)
    # This shows what a fully-taxable instrument would need to yield to match.
    if after_tax_yield is not None:
        ordinary_total = (ordinary_rate + niit_rate + state_rate) / 100
        if ordinary_total < 1:
            tey = round(after_tax_yield / (1 - ordinary_total), 2)
        else:
            tey = None
    else:
        tey = None
    result["tax_equivalent_yield_pct"] = tey

    # --- Tax advantage (QDI vs ordinary) ---
    if current_yield_pct is not None and qdi_eligible is True:
        ordinary_after_tax = current_yield_pct * (1 - (ordinary_rate + niit_rate + state_rate) / 100)
        qdi_after_tax = current_yield_pct * (1 - effective_total / 100)
        advantage_bps = round((qdi_after_tax - ordinary_after_tax) * 100, 0)
    else:
        advantage_bps = 0
    result["tax_advantage_bps"] = advantage_bps

    # --- Summary ---
    result["tax_summary"] = _build_summary(
        issuer=issuer,
        qdi_eligible=qdi_eligible,
        qdi_reason=qdi_reason,
        current_yield_pct=current_yield_pct,
        after_tax_yield=after_tax_yield,
        tey=tey,
        effective_total=effective_total,
        advantage_bps=advantage_bps,
        tax_treatment=tax_treatment,
    )

    result["methodology"] = (
        "QDI eligibility is determined by issuer type (C-corporation vs. REIT/trust) "
        "using a curated reference list and prospectus metadata. Tax-equivalent yield "
        "is computed assuming the top federal bracket (37% ordinary / 20% QDI) plus "
        "the 3.8% NIIT and an estimated 5% state rate. Actual tax treatment depends "
        "on the investor's specific circumstances and holding period."
    )

    return result


# ---------------------------------------------------------------------------
# QDI classification
# ---------------------------------------------------------------------------

def _classify_qdi(
    issuer: str,
    prospectus_qdi: Optional[bool],
) -> tuple:
    """Classify QDI eligibility based on issuer name and prospectus metadata.

    Returns (issuer_type, qdi_eligible, reason).
    """
    # If the prospectus explicitly states QDI eligibility, use that
    if prospectus_qdi is True:
        return ("c_corp", True, "Prospectus indicates QDI eligibility.")
    if prospectus_qdi is False:
        return ("unknown", False, "Prospectus indicates dividends are not QDI eligible.")

    # Check against known issuer lists
    issuer_lower = issuer.lower()

    for corp in KNOWN_C_CORPS:
        if corp in issuer_lower:
            return (
                "c_corp",
                True,
                f"{issuer} is a domestic C-corporation. Preferred dividends from "
                f"C-corporations generally qualify for QDI treatment, subject to "
                f"the 61-day holding period requirement."
            )

    for non_qdi in KNOWN_NON_QDI_ISSUERS:
        if non_qdi in issuer_lower:
            return (
                "reit",
                False,
                f"{issuer} appears to be a REIT or mortgage trust. Dividends from "
                f"REITs and trusts are generally taxed as ordinary income and do "
                f"not qualify for QDI treatment."
            )

    # Cannot determine
    return (
        "unknown",
        None,
        f"Unable to determine QDI eligibility for {issuer} from available data. "
        f"Investors should verify the issuer's corporate structure and consult "
        f"the prospectus or a tax advisor."
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _build_summary(
    issuer: str,
    qdi_eligible: Optional[bool],
    qdi_reason: str,
    current_yield_pct: Optional[float],
    after_tax_yield: Optional[float],
    tey: Optional[float],
    effective_total: float,
    advantage_bps: float,
    tax_treatment: str,
) -> str:
    """Build a one-paragraph tax and yield summary."""
    parts: List[str] = []

    if qdi_eligible is True:
        parts.append(
            f"Dividends from {issuer} preferred stock are likely eligible for "
            f"qualified dividend income (QDI) treatment."
        )
    elif qdi_eligible is False:
        parts.append(
            f"Dividends from {issuer} preferred stock are likely taxed as "
            f"ordinary income and do not qualify for QDI treatment."
        )
    else:
        parts.append(
            f"QDI eligibility for {issuer} preferred stock could not be "
            f"determined from available data."
        )

    if current_yield_pct is not None and after_tax_yield is not None:
        parts.append(
            f"At a pre-tax yield of {current_yield_pct:.2f}%, the estimated "
            f"after-tax yield is {after_tax_yield:.2f}% assuming an effective "
            f"combined tax rate of {effective_total:.1f}% ({tax_treatment})."
        )

    if tey is not None and qdi_eligible is True:
        parts.append(
            f"The tax-equivalent yield is {tey:.2f}%, meaning a fully-taxable "
            f"instrument would need to yield {tey:.2f}% to deliver the same "
            f"after-tax income."
        )

    if advantage_bps > 0:
        parts.append(
            f"The QDI tax advantage adds approximately {advantage_bps:.0f} basis "
            f"points of after-tax yield compared to ordinary income treatment."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(value: Any) -> Optional[float]:
    """Coerce common numeric-like values into float."""
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
