"""
Regulatory and Sector Agent for Preferred Equity Analysis.

Assesses bank-specific regulatory risks (Basel III/IV capital treatment,
stress test context) and sector concentration risk for financial-sector
issuers.

Key concepts:
  - Preferred stock issued by bank holding companies counts as Additional
    Tier 1 (AT1) capital under Basel III.  Regulators can force conversion
    or write-down of AT1 instruments if the issuer's CET1 ratio falls below
    a trigger threshold.
  - Non-cumulative preferred stock is the standard AT1-eligible structure
    because the issuer can skip dividends without triggering a default.
  - The eight U.S. G-SIBs (Global Systemically Important Banks) face
    additional capital surcharges and annual stress tests (CCAR/DFAST).
  - Sector concentration: the preferred market is heavily concentrated in
    financials (~70% of the $25-par universe).  This creates correlated
    risk during banking crises.

This agent uses prospectus terms and issuer characteristics to produce a
regulatory risk profile.  It does not access live regulatory filings.
"""

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Reference data: U.S. G-SIBs and capital context
# ---------------------------------------------------------------------------

# U.S. Global Systemically Important Banks and their approximate CET1
# surcharge buckets (as of 2025).  These are public and updated annually
# by the Federal Reserve.
US_GSIBS = {
    "jpmorgan chase": {"surcharge_pct": 4.5, "bucket": 4, "ticker_prefix": "JPM"},
    "jpmorgan": {"surcharge_pct": 4.5, "bucket": 4, "ticker_prefix": "JPM"},
    "bank of america": {"surcharge_pct": 3.0, "bucket": 3, "ticker_prefix": "BAC"},
    "citigroup": {"surcharge_pct": 3.0, "bucket": 3, "ticker_prefix": "C"},
    "goldman sachs": {"surcharge_pct": 2.5, "bucket": 2, "ticker_prefix": "GS"},
    "morgan stanley": {"surcharge_pct": 2.5, "bucket": 2, "ticker_prefix": "MS"},
    "wells fargo": {"surcharge_pct": 2.0, "bucket": 2, "ticker_prefix": "WFC"},
    "bank of new york mellon": {"surcharge_pct": 1.5, "bucket": 1, "ticker_prefix": "BK"},
    "bny mellon": {"surcharge_pct": 1.5, "bucket": 1, "ticker_prefix": "BK"},
    "state street": {"surcharge_pct": 1.0, "bucket": 1, "ticker_prefix": "STT"},
}

# Approximate minimum CET1 requirements for U.S. G-SIBs (simplified):
# Base 4.5% + Capital Conservation Buffer 2.5% + G-SIB surcharge + SCB (varies)
# The Stress Capital Buffer (SCB) replaced the static CCB for large banks.
# For simplicity, we use 4.5% base + 2.5% CCB + surcharge as the floor.
BASE_CET1_REQUIREMENT = 4.5
CAPITAL_CONSERVATION_BUFFER = 2.5

# Sector classification keywords
FINANCIAL_SECTOR_KEYWORDS = {
    "bank", "bancorp", "financial", "capital", "trust", "savings",
    "insurance", "reinsurance", "underwriter", "brokerage",
}

UTILITY_SECTOR_KEYWORDS = {
    "electric", "power", "energy", "utility", "gas", "water",
}

REIT_SECTOR_KEYWORDS = {
    "realty", "property", "properties", "reit", "real estate",
    "mortgage", "housing",
}


def analyze_regulatory_risk(
    market_data: Dict[str, Any],
    prospectus_terms: Dict[str, Any],
) -> Dict[str, Any]:
    """Produce a regulatory and sector risk assessment for a preferred security.

    Returns a dictionary with:
      - sector: financial / utility / reit / other
      - is_gsib: True if the issuer is a U.S. G-SIB
      - gsib_surcharge_pct: G-SIB capital surcharge if applicable
      - gsib_bucket: G-SIB bucket (1-5) if applicable
      - capital_treatment: AT1 / Tier 2 / not_applicable
      - is_at1_eligible: True if non-cumulative preferred from a bank
      - minimum_cet1_pct: estimated minimum CET1 requirement
      - regulatory_risk_level: low / moderate / elevated / high
      - sector_concentration_risk: description of sector concentration
      - dividend_deferral_risk: assessment of dividend skip risk
      - regulatory_summary: one-paragraph narrative
      - methodology: brief description
    """
    result: Dict[str, Any] = {}

    issuer = str(prospectus_terms.get("issuer", "")).strip()
    cumulative = prospectus_terms.get("cumulative")
    coupon_type = str(prospectus_terms.get("coupon_type", "")).lower()
    perpetual = prospectus_terms.get("perpetual")
    conversion_feature = prospectus_terms.get("conversion_feature")
    seniority = str(prospectus_terms.get("seniority", "")).lower()

    # --- Sector classification ---
    sector = _classify_sector(issuer)
    result["sector"] = sector

    # --- G-SIB identification ---
    gsib_info = _identify_gsib(issuer)
    result["is_gsib"] = gsib_info is not None
    result["gsib_surcharge_pct"] = gsib_info["surcharge_pct"] if gsib_info else None
    result["gsib_bucket"] = gsib_info["bucket"] if gsib_info else None

    # --- Capital treatment ---
    is_at1 = _is_at1_eligible(cumulative, perpetual, sector)
    result["is_at1_eligible"] = is_at1
    if is_at1:
        result["capital_treatment"] = "AT1"
    elif sector == "financial":
        result["capital_treatment"] = "Tier 2"
    else:
        result["capital_treatment"] = "not_applicable"

    # --- Minimum CET1 requirement ---
    if gsib_info:
        min_cet1 = BASE_CET1_REQUIREMENT + CAPITAL_CONSERVATION_BUFFER + gsib_info["surcharge_pct"]
    elif sector == "financial":
        min_cet1 = BASE_CET1_REQUIREMENT + CAPITAL_CONSERVATION_BUFFER
    else:
        min_cet1 = None
    result["minimum_cet1_pct"] = round(min_cet1, 1) if min_cet1 is not None else None

    # --- Dividend deferral risk ---
    deferral_risk = _assess_deferral_risk(cumulative, is_at1, sector)
    result["dividend_deferral_risk"] = deferral_risk

    # --- Sector concentration risk ---
    concentration = _assess_concentration_risk(sector)
    result["sector_concentration_risk"] = concentration

    # --- Conversion / write-down risk ---
    has_conversion = conversion_feature is not None and len(str(conversion_feature)) > 5
    result["has_conversion_feature"] = has_conversion
    if has_conversion:
        result["conversion_risk_note"] = (
            "This security has a conversion feature. In a stress scenario, the issuer "
            "or regulator may trigger conversion to common equity, which could result "
            "in significant dilution or loss of principal."
        )
    else:
        result["conversion_risk_note"] = None

    # --- Overall regulatory risk level ---
    risk_level = _assess_overall_risk(
        is_gsib=gsib_info is not None,
        is_at1=is_at1,
        sector=sector,
        deferral_risk=deferral_risk,
        has_conversion=has_conversion,
    )
    result["regulatory_risk_level"] = risk_level

    # --- Stress test context ---
    if gsib_info is not None:
        result["stress_test_context"] = (
            f"{issuer} is subject to annual Federal Reserve stress tests (CCAR/DFAST). "
            f"Preferred dividends may be restricted if the bank's post-stress capital "
            f"ratios fall below minimum requirements. The G-SIB surcharge of "
            f"{gsib_info['surcharge_pct']:.1f}% adds an additional capital buffer."
        )
    elif sector == "financial":
        result["stress_test_context"] = (
            f"{issuer} is a financial institution that may be subject to regulatory "
            f"capital requirements. Preferred dividends could be restricted under "
            f"stress conditions."
        )
    else:
        result["stress_test_context"] = None

    # --- Summary ---
    result["regulatory_summary"] = _build_summary(
        issuer=issuer,
        sector=sector,
        gsib_info=gsib_info,
        is_at1=is_at1,
        min_cet1=min_cet1,
        deferral_risk=deferral_risk,
        risk_level=risk_level,
        concentration=concentration,
    )

    result["methodology"] = (
        "Regulatory risk is assessed using the issuer's G-SIB designation, "
        "capital treatment of the preferred (AT1 vs. Tier 2), and the structural "
        "features of the security (non-cumulative, perpetual). G-SIB surcharges "
        "and minimum CET1 requirements are based on publicly available Federal "
        "Reserve data. This assessment does not incorporate live capital ratios "
        "or stress test results."
    )

    return result


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def _classify_sector(issuer: str) -> str:
    """Classify the issuer into a broad sector."""
    issuer_lower = issuer.lower()

    # Check G-SIBs first (all are financial)
    for gsib_name in US_GSIBS:
        if gsib_name in issuer_lower:
            return "financial"

    for kw in REIT_SECTOR_KEYWORDS:
        if kw in issuer_lower:
            return "reit"

    for kw in FINANCIAL_SECTOR_KEYWORDS:
        if kw in issuer_lower:
            return "financial"

    for kw in UTILITY_SECTOR_KEYWORDS:
        if kw in issuer_lower:
            return "utility"

    return "other"


def _identify_gsib(issuer: str) -> Optional[Dict[str, Any]]:
    """Return G-SIB metadata if the issuer is a U.S. G-SIB, else None."""
    issuer_lower = issuer.lower()
    for name, info in US_GSIBS.items():
        if name in issuer_lower:
            return info
    return None


def _is_at1_eligible(
    cumulative: Optional[bool],
    perpetual: Optional[bool],
    sector: str,
) -> bool:
    """Determine if the preferred qualifies as AT1 capital.

    AT1 requirements under Basel III:
      - Must be non-cumulative (issuer can skip dividends)
      - Must be perpetual (no maturity date)
      - Must be issued by a regulated financial institution
    """
    if sector != "financial":
        return False
    if cumulative is True:
        return False  # cumulative preferreds are Tier 2, not AT1
    if perpetual is False:
        return False  # dated preferreds are Tier 2
    # If cumulative is None or False, and perpetual is None or True,
    # and the issuer is financial, it is likely AT1-eligible.
    return True


def _assess_deferral_risk(
    cumulative: Optional[bool],
    is_at1: bool,
    sector: str,
) -> str:
    """Assess the risk that the issuer defers dividend payments."""
    if cumulative is True:
        return "low"  # missed dividends accumulate and must eventually be paid
    if is_at1:
        return "moderate"  # non-cumulative AT1: issuer can skip without consequence
    if sector == "financial" and cumulative is False:
        return "moderate"
    if cumulative is False:
        return "elevated"  # non-cumulative, non-financial: higher risk
    return "unknown"


def _assess_concentration_risk(sector: str) -> str:
    """Describe sector concentration risk."""
    if sector == "financial":
        return (
            "The preferred equity market is heavily concentrated in the financial "
            "sector (approximately 70% of the $25-par universe). This creates "
            "correlated risk: a systemic banking event would affect most preferred "
            "securities simultaneously. Diversification across sectors is limited "
            "within the preferred asset class."
        )
    if sector == "reit":
        return (
            "REIT preferreds are concentrated in the real estate sector and are "
            "sensitive to both interest rate changes and property market conditions. "
            "They provide some diversification from bank preferreds but carry "
            "their own sector-specific risks."
        )
    if sector == "utility":
        return (
            "Utility preferreds offer some diversification from the dominant "
            "financial sector. They tend to be more rate-sensitive but carry "
            "lower credit risk due to the regulated nature of utility businesses."
        )
    return (
        "This issuer is outside the dominant financial sector, which provides "
        "diversification benefits within a preferred equity portfolio."
    )


def _assess_overall_risk(
    is_gsib: bool,
    is_at1: bool,
    sector: str,
    deferral_risk: str,
    has_conversion: bool,
) -> str:
    """Assign an overall regulatory risk level."""
    risk_score = 0

    # G-SIBs are well-capitalized but face more regulatory scrutiny
    if is_gsib:
        risk_score += 1  # moderate: well-capitalized but complex regulation
    elif sector == "financial":
        risk_score += 2  # smaller banks may have thinner capital buffers

    # AT1 instruments can be written down or converted
    if is_at1:
        risk_score += 1

    # Conversion feature adds tail risk
    if has_conversion:
        risk_score += 1

    # Deferral risk
    deferral_map = {"low": 0, "moderate": 1, "elevated": 2, "unknown": 1}
    risk_score += deferral_map.get(deferral_risk, 1)

    if risk_score <= 1:
        return "low"
    if risk_score <= 3:
        return "moderate"
    if risk_score <= 5:
        return "elevated"
    return "high"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _build_summary(
    issuer: str,
    sector: str,
    gsib_info: Optional[Dict[str, Any]],
    is_at1: bool,
    min_cet1: Optional[float],
    deferral_risk: str,
    risk_level: str,
    concentration: str,
) -> str:
    """Build a one-paragraph regulatory risk summary."""
    parts: List[str] = []

    if gsib_info:
        parts.append(
            f"{issuer} is a U.S. Global Systemically Important Bank (G-SIB) "
            f"in bucket {gsib_info['bucket']} with a capital surcharge of "
            f"{gsib_info['surcharge_pct']:.1f}%."
        )
    elif sector == "financial":
        parts.append(f"{issuer} is a financial institution subject to regulatory capital requirements.")
    else:
        parts.append(f"{issuer} operates in the {sector} sector.")

    if is_at1:
        parts.append(
            "This preferred security qualifies as Additional Tier 1 (AT1) capital, "
            "meaning it is non-cumulative and perpetual. The issuer can defer dividends "
            "without triggering a default, and regulators can require write-down or "
            "conversion under severe stress."
        )

    if min_cet1 is not None:
        parts.append(
            f"The estimated minimum CET1 requirement is {min_cet1:.1f}%, including "
            f"the base requirement, capital conservation buffer, and any G-SIB surcharge."
        )

    deferral_text = deferral_risk.replace("_", " ")
    parts.append(f"Dividend deferral risk is assessed as {deferral_text}.")

    risk_text = risk_level.replace("_", " ")
    parts.append(f"Overall regulatory risk is {risk_text}.")

    return " ".join(parts)
