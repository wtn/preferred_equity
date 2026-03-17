"""
End-to-end test for Phase 3 swarm.
Runs the full pipeline on BAC-PL and validates that all new agent outputs
are populated.
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.advanced_swarm import analyze_preferred_advanced


def test_ticker(ticker: str):
    print(f"\n{'#'*70}")
    print(f"  TESTING: {ticker}")
    print(f"{'#'*70}\n")

    result = analyze_preferred_advanced(ticker)

    # Validate all state keys exist
    expected_keys = [
        "ticker", "market_data", "rate_data", "rate_sensitivity",
        "dividend_data", "prospectus_terms", "call_analysis",
        "tax_analysis", "regulatory_analysis", "relative_value",
        "quality_report", "synthesis", "errors", "agent_status",
    ]
    for key in expected_keys:
        assert key in result, f"Missing state key: {key}"
        print(f"  [OK] State key '{key}' present")

    # Validate quality report
    qr = result["quality_report"]
    print(f"\n  Quality score: {qr.get('overall_score')}")
    print(f"  Passed: {qr.get('passed')}")
    print(f"  Decision: {qr.get('decision')}")

    # Validate call analysis
    ca = result["call_analysis"]
    print(f"\n  Call probability: {ca.get('call_probability')}")
    print(f"  YTC: {ca.get('yield_to_call_pct')}")
    print(f"  YTW: {ca.get('yield_to_worst_pct')}")
    print(f"  Refinancing incentive: {ca.get('refinancing_incentive')}")
    assert ca.get("call_probability") is not None, "Call probability should not be None"

    # Validate tax analysis
    ta = result["tax_analysis"]
    print(f"\n  QDI eligible: {ta.get('qdi_eligible')}")
    print(f"  After-tax yield: {ta.get('after_tax_yield_pct')}")
    print(f"  TEY: {ta.get('tax_equivalent_yield_pct')}")
    print(f"  Tax advantage: {ta.get('tax_advantage_bps')} bps")
    assert ta.get("qdi_eligible") is not None, "QDI should be classified"

    # Validate regulatory analysis
    ra = result["regulatory_analysis"]
    print(f"\n  Sector: {ra.get('sector')}")
    print(f"  G-SIB: {ra.get('is_gsib')}")
    print(f"  Capital treatment: {ra.get('capital_treatment')}")
    print(f"  Regulatory risk: {ra.get('regulatory_risk_level')}")
    assert ra.get("sector") is not None, "Sector should be classified"

    # Validate relative value
    rv = result["relative_value"]
    print(f"\n  Value assessment: {rv.get('value_assessment')}")
    print(f"  Peer count: {rv.get('peer_count')}")
    print(f"  Yield rank: {rv.get('yield_rank')}")
    print(f"  Spread to Treasury: {rv.get('spread_to_treasury_bps')} bps")

    # Validate synthesis
    synthesis = result["synthesis"]
    print(f"\n  Synthesis length: {len(synthesis)} chars")
    assert len(synthesis) > 100, "Synthesis should be substantial"

    # Agent status
    status = result["agent_status"]
    print(f"\n  Agent status: {json.dumps(status, indent=2)}")

    print(f"\n  [PASS] {ticker} end-to-end test passed!")
    return result


if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["BAC-PL"]
    for t in tickers:
        test_ticker(t)
