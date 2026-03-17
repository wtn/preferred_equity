"""
Preferred Equity Analysis Swarm: Streamlit Demo
=================================================
A web interface for the multi-agent preferred equity analysis system.
Demonstrates parallel execution, conditional routing, and quality gating.
Phase 3: Full analytical swarm with 12 agent nodes across 5 layers.
"""

import sys
import os
import json
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.advanced_swarm import analyze_preferred_advanced
from src.data.market_data import get_price_history
from src.data.prospectus_inventory import (
    get_inventory_lookup,
    get_quick_analysis_tickers,
    load_cached_prospectus_inventory,
)


cached_inventory = load_cached_prospectus_inventory()
inventory_lookup = get_inventory_lookup()
quick_tickers = get_quick_analysis_tickers(limit=8)


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Preferred Equity Swarm",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Preferred Equity Swarm")
    st.caption("MSBA Capstone Project")

    st.markdown("---")

    st.subheader("About")
    st.write(
        "This demo showcases a multi-agent AI swarm that analyzes "
        "preferred equity securities. The swarm coordinates specialized "
        "agents running in parallel across three layers, combining live "
        "market data with SEC prospectus extraction, call probability, "
        "tax analysis, regulatory risk, and relative value before a "
        "quality gate determines whether data is sufficient for AI synthesis."
    )

    st.markdown("---")

    st.subheader("Swarm Architecture")
    st.markdown("""
    **Layer 1: Data Collection (Parallel)**
    1. Market Data Agent
    2. Rate Context Agent
    3. Dividend Analysis Agent
    4. Prospectus Parsing Agent

    **Layer 2: Deterministic Analysis**
    5. Interest Rate Sensitivity Agent

    **Layer 3: Analytical Agents (Parallel)**
    6. Call Probability Agent
    7. Tax & Yield Agent
    8. Regulatory & Sector Agent
    9. Relative Value Agent

    **Layer 4: Quality Gate**
    10. Quality Check Agent

    **Layer 5: Conditional Routing**
    11a. Synthesis Agent (Gemini) *or*
    11b. Error Report Agent
    """)

    st.markdown("---")

    st.subheader("LangGraph Patterns")
    st.markdown("""
    **Fan-Out:** 4 data agents run in parallel from START, then 4 analytical agents run in parallel from Layer 2

    **Fan-In:** Layer 1 feeds Layer 2, Layer 3 feeds Quality Gate

    **Conditional Edge:** Routes to Synthesis or Error based on data quality score
    """)

    st.markdown("---")
    st.subheader("Quick Analysis")
    st.write(
        f"{len(cached_inventory)} cached prospectus profiles are available for fast analysis. "
        "Any other preferred ticker can still be searched live."
    )

    if cached_inventory:
        inventory_df = pd.DataFrame(
            [
                {
                    "Ticker": row["ticker"],
                    "Series": row["series"],
                    "Cache": row["cache_label"],
                }
                for row in cached_inventory
            ]
        )
        st.dataframe(inventory_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption("Phase 3: Full Analytical Swarm")


# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------

st.title("Preferred Equity Analysis Swarm")
st.markdown("Enter a preferred stock ticker to run the multi-agent analysis with parallel execution, SEC prospectus extraction, and quality gating.")

# Sample tickers for easy access
col1, col2 = st.columns([3, 1])

with col1:
    ticker = st.text_input(
        "Preferred Stock Ticker",
        value="BAC-PL",
        placeholder="e.g., BAC-PL, JPM-PD, WFC-PL",
        help="Enter a preferred stock ticker symbol. Use the format ISSUER-P[SERIES] (e.g., BAC-PL for Bank of America Series L). Cached tickers analyze faster; uncached tickers fall back to live SEC search and will join your local inventory after successful extraction."
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_button = st.button("Analyze", type="primary", use_container_width=True)

# Quick-pick buttons
st.caption("Quick picks:")
quick_cols = st.columns(max(len(quick_tickers), 1))
for i, qt in enumerate(quick_tickers):
    with quick_cols[i]:
        if st.button(qt, use_container_width=True):
            ticker = qt
            analyze_button = True

selected_inventory_entry = inventory_lookup.get(ticker.strip().upper())
if selected_inventory_entry:
    st.caption(
        f"{ticker.strip().upper()} is available from {selected_inventory_entry['cache_label'].lower()} "
        f"for quick analysis."
    )
else:
    st.caption(
        "This ticker is not cached yet. The app will search live SEC filings and add it to your local "
        "inventory after a successful extraction."
    )

st.markdown("---")


# ---------------------------------------------------------------------------
# Analysis Execution
# ---------------------------------------------------------------------------

if analyze_button and ticker:
    # Agent execution with progress tracking
    progress_container = st.container()

    with progress_container:
        st.subheader(f"Analyzing: {ticker}")

        progress_bar = st.progress(0, text="Initializing swarm...")
        status_placeholder = st.empty()

        try:
            status_placeholder.info("Running Layer 1 (data collection), Layer 2 (rate sensitivity), Layer 3 (analytical agents)...")
            progress_bar.progress(10, text="Agents running across 3 layers...")

            result = analyze_preferred_advanced(ticker)

            quality = result.get("quality_report", {})
            if quality.get("passed", False):
                progress_bar.progress(100, text="Analysis complete!")
                status_placeholder.success(
                    f"All agents completed. Quality score: {quality.get('overall_score', 0):.0%}. "
                    f"Route: Synthesis Agent."
                )
            else:
                progress_bar.progress(100, text="Analysis complete (with issues)")
                status_placeholder.warning(
                    f"Quality score: {quality.get('overall_score', 0):.0%} (below threshold). "
                    f"Route: Error Report Agent."
                )

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.stop()

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Agent Status Dashboard
    # ---------------------------------------------------------------------------

    st.subheader("Agent Execution Status")

    agent_status = result.get("agent_status", {})
    quality_report = result.get("quality_report", {})
    prospectus_terms = result.get("prospectus_terms", {})

    # Row 1: Layer 1 + Layer 2 agents
    st.caption("Layer 1 (Data Collection) + Layer 2 (Analysis)")
    status_cols_1 = st.columns(7)

    agent_labels_1 = [
        ("market_data", "Market Data"),
        ("rate_context", "Rate Context"),
        ("dividend", "Dividend"),
        ("prospectus", "Prospectus"),
        ("interest_rate", "Rate Sensitivity"),
    ]

    for i, (key, label) in enumerate(agent_labels_1):
        with status_cols_1[i]:
            status = agent_status.get(key, "unknown")
            if status == "success":
                st.metric(label, "OK", delta="success", delta_color="normal")
            elif status == "failed":
                st.metric(label, "FAIL", delta="failed", delta_color="inverse")
            else:
                st.metric(label, "?", delta="unknown", delta_color="off")

    with status_cols_1[5]:
        qscore = quality_report.get("overall_score", 0)
        passed = quality_report.get("passed", False)
        st.metric(
            "Quality Gate",
            f"{qscore:.0%}",
            delta="passed" if passed else "failed",
            delta_color="normal" if passed else "inverse"
        )

    with status_cols_1[6]:
        route = quality_report.get("decision", "unknown")
        if route == "proceed_to_synthesis":
            st.metric("Route Taken", "Synthesis", delta="AI analysis", delta_color="normal")
        else:
            st.metric("Route Taken", "Error Report", delta="fallback", delta_color="inverse")

    # Row 2: Layer 3 agents
    st.caption("Layer 3 (Analytical Agents)")
    status_cols_2 = st.columns(4)

    agent_labels_2 = [
        ("call_probability", "Call Probability"),
        ("tax_yield", "Tax & Yield"),
        ("regulatory", "Regulatory"),
        ("relative_value", "Relative Value"),
    ]

    for i, (key, label) in enumerate(agent_labels_2):
        with status_cols_2[i]:
            status = agent_status.get(key, "unknown")
            if status == "success":
                st.metric(label, "OK", delta="success", delta_color="normal")
            elif status == "failed":
                st.metric(label, "FAIL", delta="failed", delta_color="inverse")
            else:
                st.metric(label, "?", delta="unknown", delta_color="off")

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Key Metrics
    # ---------------------------------------------------------------------------

    market_data = result.get("market_data", {})
    rate_data = result.get("rate_data", {})
    rate_sensitivity = result.get("rate_sensitivity", {})
    dividend_data = result.get("dividend_data", {})
    call_analysis = result.get("call_analysis", {})
    tax_analysis = result.get("tax_analysis", {})
    regulatory_analysis = result.get("regulatory_analysis", {})
    relative_value = result.get("relative_value", {})

    st.subheader("Key Metrics")

    metric_cols = st.columns(5)

    with metric_cols[0]:
        price = market_data.get("price", None)
        st.metric("Current Price", f"${price:,.2f}" if isinstance(price, (int, float)) else "N/A")

    with metric_cols[1]:
        div_rate = market_data.get("dividend_rate", None)
        st.metric("Annual Dividend", f"${div_rate:,.2f}" if div_rate else "N/A")

    with metric_cols[2]:
        div_yield = market_data.get("dividend_yield", None)
        if div_yield:
            yield_pct = div_yield if div_yield > 1 else div_yield * 100
            st.metric("Current Yield", f"{yield_pct:.2f}%")
        else:
            yield_pct = None
            st.metric("Current Yield", "N/A")

    with metric_cols[3]:
        ten_yr = rate_data.get("10Y", rate_data.get("20Y", None))
        if ten_yr and yield_pct:
            spread = (yield_pct - ten_yr) * 100
            st.metric("Spread vs Treasury", f"{spread:.0f} bps")
        else:
            st.metric("Spread vs Treasury", "N/A")

    with metric_cols[4]:
        consistency = dividend_data.get("consistency", "N/A")
        frequency = dividend_data.get("frequency", "N/A")
        st.metric("Dividend Pattern", f"{frequency.title()}", delta=consistency, delta_color="normal" if consistency in ("excellent", "good") else "off")

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Call Risk Analysis (Phase 3)
    # ---------------------------------------------------------------------------

    if call_analysis and call_analysis.get("call_probability") is not None:
        st.subheader("Call Risk Analysis")

        call_cols = st.columns(5)

        with call_cols[0]:
            prob = str(call_analysis.get("call_probability", "N/A")).replace("_", " ").title()
            score = call_analysis.get("call_probability_score")
            st.metric("Call Probability", prob, delta=f"score: {score:.2f}" if score is not None else None)

        with call_cols[1]:
            ytc = call_analysis.get("yield_to_call_pct")
            st.metric("Yield-to-Call", f"{ytc:.2f}%" if isinstance(ytc, (int, float)) else "N/A")

        with call_cols[2]:
            ytw = call_analysis.get("yield_to_worst_pct")
            st.metric("Yield-to-Worst", f"{ytw:.2f}%" if isinstance(ytw, (int, float)) else "N/A")

        with call_cols[3]:
            years = call_analysis.get("years_to_call")
            if years is not None and years > 0:
                st.metric("Years to Call", f"{years:.1f}")
            elif years is not None and years <= 0:
                st.metric("Years to Call", "Currently Callable")
            else:
                st.metric("Years to Call", "N/A")

        with call_cols[4]:
            incentive = str(call_analysis.get("refinancing_incentive", "N/A")).replace("_", " ").title()
            st.metric("Refinancing Incentive", incentive)

        premium = call_analysis.get("premium_to_call_pct")
        if premium is not None:
            call_price = call_analysis.get("call_price_per_share")
            if premium > 0:
                st.caption(f"Trading at a {premium:.1f}% premium to the ${call_price:.2f} call price.")
            elif premium < 0:
                st.caption(f"Trading at a {abs(premium):.1f}% discount to the ${call_price:.2f} call price.")
            else:
                st.caption(f"Trading at the ${call_price:.2f} call price.")

        if call_analysis.get("call_analysis_summary"):
            with st.expander("Call Analysis Detail"):
                st.write(call_analysis["call_analysis_summary"])

        st.markdown("---")

    # ---------------------------------------------------------------------------
    # Interest Rate Sensitivity
    # ---------------------------------------------------------------------------

    if rate_sensitivity and not rate_sensitivity.get("error"):
        st.subheader("Interest Rate Sensitivity")

        rate_cols = st.columns(5)

        with rate_cols[0]:
            regime = str(rate_sensitivity.get("regime", "N/A")).replace("_", " ").title()
            st.metric("Rate Regime", regime)

        with rate_cols[1]:
            primary_measure = rate_sensitivity.get("primary_measure", "Primary Measure")
            primary_value = rate_sensitivity.get("primary_value")
            if isinstance(primary_value, (int, float)):
                if "Duration" in primary_measure or "Reset" in primary_measure:
                    primary_text = f"{primary_value:.2f} yrs"
                elif "Coupon" in primary_measure:
                    primary_text = f"{primary_value:.2f}%"
                else:
                    primary_text = f"{primary_value:.2f}"
            else:
                primary_text = str(primary_value or "N/A")
            st.metric(primary_measure, primary_text)

        with rate_cols[2]:
            duration = rate_sensitivity.get("effective_duration")
            st.metric(
                "Effective Duration",
                f"{duration:.2f} yrs" if isinstance(duration, (int, float)) else "N/A",
            )

        with rate_cols[3]:
            dv01 = rate_sensitivity.get("effective_dv01_per_share")
            st.metric(
                "DV01 / Share",
                f"${dv01:.4f}" if isinstance(dv01, (int, float)) else "N/A",
            )

        with rate_cols[4]:
            st.metric("Confidence", str(rate_sensitivity.get("confidence", "N/A")).title())

        if rate_sensitivity.get("summary"):
            st.caption(rate_sensitivity["summary"])
        if rate_sensitivity.get("benchmark_note"):
            st.caption(f"Note: {rate_sensitivity['benchmark_note']}")

        benchmark_fields_present = any(
            rate_sensitivity.get(key) is not None
            for key in (
                "contractual_benchmark",
                "live_benchmark_label",
                "benchmark_replacement_method",
                "all_in_floating_coupon_pct",
                "projected_post_reset_coupon_pct",
            )
        )
        if benchmark_fields_present:
            st.markdown("**Benchmark Context**")
            benchmark_cols = st.columns(4)

            with benchmark_cols[0]:
                st.metric(
                    "Contractual Benchmark",
                    str(rate_sensitivity.get("contractual_benchmark") or "N/A"),
                )

            with benchmark_cols[1]:
                st.metric(
                    "Live Benchmark Used",
                    str(rate_sensitivity.get("live_benchmark_label") or "N/A"),
                )

            with benchmark_cols[2]:
                st.metric(
                    "Replacement Method",
                    str(rate_sensitivity.get("benchmark_replacement_method") or "N/A"),
                )

            coupon_estimate = rate_sensitivity.get("all_in_floating_coupon_pct")
            coupon_label = "All-In Reset Coupon"
            if coupon_estimate is None:
                coupon_estimate = rate_sensitivity.get("projected_post_reset_coupon_pct")
                coupon_label = "Projected Post-Reset Coupon"

            with benchmark_cols[3]:
                st.metric(
                    coupon_label,
                    (
                        f"{coupon_estimate:.2f}%"
                        if isinstance(coupon_estimate, (int, float))
                        else "N/A"
                    ),
                )

            next_reset_tenor = rate_sensitivity.get("next_reset_tenor_years")
            if isinstance(next_reset_tenor, (int, float)):
                st.caption(f"Next reset tenor assumption: {next_reset_tenor:.2f} years")

        scenario_table = rate_sensitivity.get("scenario_table", [])
        if scenario_table:
            if rate_sensitivity.get("scenario_table_type") == "benchmark_coupon":
                benchmark_label = rate_sensitivity.get("live_benchmark_label") or "Benchmark"
                scenario_df = pd.DataFrame(
                    [
                        {
                            "Shock (bps)": row["shock_bps"],
                            f"{benchmark_label} Rate": (
                                f"{row['benchmark_rate_pct']:.2f}%"
                                if isinstance(row.get("benchmark_rate_pct"), (int, float))
                                else "N/A"
                            ),
                            "All-In Coupon": (
                                f"{row['all_in_coupon_pct']:.2f}%"
                                if isinstance(row.get("all_in_coupon_pct"), (int, float))
                                else "N/A"
                            ),
                        }
                        for row in scenario_table
                    ]
                )
            else:
                scenario_df = pd.DataFrame(
                    [
                        {
                            "Shock (bps)": row["shock_bps"],
                            "Estimated Price Change": row["estimated_price_change"],
                            "Estimated Price": row["estimated_price"],
                        }
                        for row in scenario_table
                    ]
                )
            st.dataframe(scenario_df, use_container_width=True, hide_index=True)

        st.markdown("---")
    elif rate_sensitivity.get("error"):
        st.subheader("Interest Rate Sensitivity")
        st.warning(rate_sensitivity["error"])
        st.markdown("---")

    # ---------------------------------------------------------------------------
    # Tax and Yield Profile (Phase 3)
    # ---------------------------------------------------------------------------

    if tax_analysis and tax_analysis.get("qdi_eligible") is not None:
        st.subheader("Tax and Yield Profile")

        tax_cols = st.columns(5)

        with tax_cols[0]:
            qdi = tax_analysis.get("qdi_eligible")
            if qdi is True:
                qdi_text = "QDI Eligible"
                qdi_delta = "qualified"
            elif qdi is False:
                qdi_text = "Not QDI"
                qdi_delta = "ordinary income"
            else:
                qdi_text = "Unknown"
                qdi_delta = "not determined"
            st.metric("QDI Status", qdi_text, delta=qdi_delta, delta_color="normal" if qdi else "off")

        with tax_cols[1]:
            after_tax = tax_analysis.get("after_tax_yield_pct")
            st.metric("After-Tax Yield", f"{after_tax:.2f}%" if isinstance(after_tax, (int, float)) else "N/A")

        with tax_cols[2]:
            tey = tax_analysis.get("tax_equivalent_yield_pct")
            st.metric("Tax-Equivalent Yield", f"{tey:.2f}%" if isinstance(tey, (int, float)) else "N/A")

        with tax_cols[3]:
            eff_rate = tax_analysis.get("effective_tax_rate_pct")
            st.metric("Effective Tax Rate", f"{eff_rate:.1f}%" if isinstance(eff_rate, (int, float)) else "N/A")

        with tax_cols[4]:
            advantage = tax_analysis.get("tax_advantage_bps", 0)
            st.metric("QDI Advantage", f"{advantage:.0f} bps" if advantage else "N/A")

        if tax_analysis.get("qdi_classification_reason"):
            st.caption(tax_analysis["qdi_classification_reason"])

        if tax_analysis.get("tax_summary"):
            with st.expander("Tax Analysis Detail"):
                st.write(tax_analysis["tax_summary"])

        st.markdown("---")

    # ---------------------------------------------------------------------------
    # Regulatory and Sector Risk (Phase 3)
    # ---------------------------------------------------------------------------

    if regulatory_analysis and regulatory_analysis.get("sector") is not None:
        st.subheader("Regulatory and Sector Risk")

        reg_cols = st.columns(5)

        with reg_cols[0]:
            sector = str(regulatory_analysis.get("sector", "N/A")).title()
            st.metric("Sector", sector)

        with reg_cols[1]:
            is_gsib = regulatory_analysis.get("is_gsib", False)
            gsib_text = "Yes" if is_gsib else "No"
            bucket = regulatory_analysis.get("gsib_bucket")
            gsib_delta = f"Bucket {bucket}" if bucket else None
            st.metric("G-SIB", gsib_text, delta=gsib_delta)

        with reg_cols[2]:
            capital = str(regulatory_analysis.get("capital_treatment", "N/A")).upper()
            st.metric("Capital Treatment", capital)

        with reg_cols[3]:
            risk_level = str(regulatory_analysis.get("regulatory_risk_level", "N/A")).replace("_", " ").title()
            st.metric("Regulatory Risk", risk_level)

        with reg_cols[4]:
            deferral = str(regulatory_analysis.get("dividend_deferral_risk", "N/A")).replace("_", " ").title()
            st.metric("Deferral Risk", deferral)

        min_cet1 = regulatory_analysis.get("minimum_cet1_pct")
        if min_cet1 is not None:
            st.caption(f"Estimated minimum CET1 requirement: {min_cet1:.1f}%")

        if regulatory_analysis.get("stress_test_context"):
            with st.expander("Stress Test Context"):
                st.write(regulatory_analysis["stress_test_context"])

        if regulatory_analysis.get("regulatory_summary"):
            with st.expander("Regulatory Analysis Detail"):
                st.write(regulatory_analysis["regulatory_summary"])

        st.markdown("---")

    # ---------------------------------------------------------------------------
    # Relative Value (Phase 3)
    # ---------------------------------------------------------------------------

    if relative_value and relative_value.get("value_assessment") is not None:
        st.subheader("Relative Value")

        rv_cols = st.columns(5)

        with rv_cols[0]:
            value = str(relative_value.get("value_assessment", "N/A")).title()
            st.metric("Value Assessment", value)

        with rv_cols[1]:
            peer_count = relative_value.get("peer_count", 0)
            rank = relative_value.get("yield_rank")
            rank_text = f"#{rank} of {peer_count}" if rank else "N/A"
            st.metric("Yield Rank", rank_text)

        with rv_cols[2]:
            spread_tsy = relative_value.get("spread_to_treasury_bps")
            st.metric("Spread to Treasury", f"{spread_tsy:.0f} bps" if spread_tsy is not None else "N/A")

        with rv_cols[3]:
            spread_common = relative_value.get("spread_to_common_bps")
            st.metric("Spread to Common", f"{spread_common:.0f} bps" if spread_common is not None else "N/A")

        with rv_cols[4]:
            common_yield = relative_value.get("common_dividend_yield_pct")
            st.metric("Common Div Yield", f"{common_yield:.2f}%" if isinstance(common_yield, (int, float)) else "N/A")

        # Peer comparison table
        peers = relative_value.get("peer_universe", [])
        if peers:
            st.markdown("**Peer Comparison**")
            peer_df = pd.DataFrame(
                [
                    {
                        "Ticker": p.get("ticker", "N/A"),
                        "Issuer": p.get("issuer", "N/A"),
                        "Coupon": f"{p['coupon_rate']:.2f}%" if p.get("coupon_rate") is not None else "N/A",
                        "Type": str(p.get("coupon_type", "N/A")).title(),
                        "Cumulative": "Yes" if p.get("cumulative") else "No" if p.get("cumulative") is False else "N/A",
                        "Relationship": p.get("relationship", "N/A"),
                    }
                    for p in peers
                ]
            )
            st.dataframe(peer_df, use_container_width=True, hide_index=True)

        if relative_value.get("relative_value_summary"):
            with st.expander("Relative Value Detail"):
                st.write(relative_value["relative_value_summary"])

        st.markdown("---")

    # ---------------------------------------------------------------------------
    # Charts Row
    # ---------------------------------------------------------------------------

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Treasury Yield Curve vs Preferred Yield")

        if rate_data:
            maturities = list(rate_data.keys())
            yields = list(rate_data.values())

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=maturities,
                y=yields,
                mode='lines+markers',
                name='Treasury Yields',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=8),
            ))

            if yield_pct:
                fig.add_hline(
                    y=yield_pct,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"{ticker} Yield: {yield_pct:.2f}%",
                    annotation_position="top right",
                )

            fig.update_layout(
                xaxis_title="Maturity",
                yaxis_title="Yield (%)",
                height=400,
                template="plotly_white",
                showlegend=True,
            )

            st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        st.subheader("Price History (1 Year)")

        price_hist = get_price_history(ticker, period="1y")
        if price_hist is not None and not price_hist.empty:
            fig2 = go.Figure()

            fig2.add_trace(go.Scatter(
                x=price_hist.index,
                y=price_hist["Close"],
                mode='lines',
                name='Close Price',
                line=dict(color='#2ca02c', width=2),
                fill='tozeroy',
                fillcolor='rgba(44, 160, 44, 0.1)',
            ))

            fig2.update_layout(
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400,
                template="plotly_white",
            )

            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Price history not available for this ticker.")

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Dividend Analysis Details
    # ---------------------------------------------------------------------------

    if dividend_data.get("has_dividend_history"):
        st.subheader("Dividend Analysis")

        div_cols = st.columns(4)

        with div_cols[0]:
            st.metric("Total Payments", dividend_data.get("total_payments_recorded", "N/A"))
        with div_cols[1]:
            st.metric("Avg Payment", f"${dividend_data.get('avg_payment', 0):.4f}")
        with div_cols[2]:
            is_fixed = dividend_data.get("is_fixed_rate", None)
            st.metric("Rate Type", "Fixed" if is_fixed else "Variable" if is_fixed is not None else "N/A")
        with div_cols[3]:
            st.metric("Trend", dividend_data.get("trend", "N/A").replace("_", " ").title())

        st.caption(
            f"History from {dividend_data.get('first_payment_date', 'N/A')} "
            f"to {dividend_data.get('last_payment_date', 'N/A')}. "
            f"Consistency score: {dividend_data.get('consistency_score', 0):.0%}."
        )

        st.markdown("---")

    # ---------------------------------------------------------------------------
    # Prospectus Terms
    # ---------------------------------------------------------------------------

    if prospectus_terms and not prospectus_terms.get("error"):
        st.subheader("Prospectus Terms (SEC)")

        security_name = prospectus_terms.get("security_name", "Unknown security")
        series = prospectus_terms.get("series", "N/A")
        filing_url = prospectus_terms.get("filing_url")
        filing_date = prospectus_terms.get("filing_date", "N/A")
        accession_number = prospectus_terms.get("accession_number", "N/A")
        source_label = str(prospectus_terms.get("source", "live")).replace("_", " ").title()
        resolution_label = str(prospectus_terms.get("resolution_source", "live")).replace("_", " ").title()
        mismatch_warning = prospectus_terms.get("mismatch_warning")

        st.markdown(f"**{security_name}**")
        st.caption(
            f"Series: {series} | Filing date: {filing_date} | Accession: {accession_number}"
        )
        st.caption(f"Prospectus source: {source_label} | Resolution: {resolution_label}")
        if filing_url:
            st.markdown(f"[Open filing on SEC.gov]({filing_url})")
        if mismatch_warning:
            st.warning(mismatch_warning)

        prospectus_cols = st.columns(5)

        with prospectus_cols[0]:
            coupon_rate = prospectus_terms.get("coupon_rate")
            st.metric("Coupon Rate", f"{coupon_rate:.2f}%" if isinstance(coupon_rate, (int, float)) else "N/A")
        with prospectus_cols[1]:
            st.metric("Coupon Type", str(prospectus_terms.get("coupon_type", "N/A")).title())
        with prospectus_cols[2]:
            st.metric("First Call Date", prospectus_terms.get("call_date", "N/A") or "N/A")
        with prospectus_cols[3]:
            qdi_flag = prospectus_terms.get("qdi_eligible")
            if qdi_flag is True:
                qdi_text = "Yes"
            elif qdi_flag is False:
                qdi_text = "No"
            else:
                qdi_text = "Unknown"
            st.metric("QDI Eligible", qdi_text)
        with prospectus_cols[4]:
            perpetual = prospectus_terms.get("perpetual")
            if perpetual is True:
                perpetual_text = "Yes"
            elif perpetual is False:
                perpetual_text = "No"
            else:
                perpetual_text = "Unknown"
            st.metric("Perpetual", perpetual_text)

        st.caption(
            f"Depositary shares: {prospectus_terms.get('deposit_fraction', 'No')} | "
            f"Cumulative: {prospectus_terms.get('cumulative', 'Unknown')} | "
            f"Exchange: {prospectus_terms.get('listing_exchange', 'N/A')}"
        )

        st.markdown("---")
    elif prospectus_terms.get("error"):
        st.subheader("Prospectus Terms (SEC)")
        st.warning(prospectus_terms["error"])
        st.markdown("---")

    # ---------------------------------------------------------------------------
    # AI Synthesis or Error Report
    # ---------------------------------------------------------------------------

    synthesis = result.get("synthesis", "")

    if quality_report.get("passed", False):
        st.subheader("AI Synthesis (Gemini)")
    else:
        st.subheader("Error Report")

    st.markdown(synthesis)

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Quality Check Details (expandable)
    # ---------------------------------------------------------------------------

    with st.expander("View Quality Check Details"):
        checks = quality_report.get("checks", {})

        # Display in rows of 4 to handle the expanded check set
        check_items = list(checks.items())
        for row_start in range(0, len(check_items), 4):
            row_items = check_items[row_start:row_start + 4]
            qc_cols = st.columns(len(row_items))
            for i, (source, details) in enumerate(row_items):
                with qc_cols[i]:
                    st.markdown(f"**{source.replace('_', ' ').title()}**")
                    score = details.get("score", 0)
                    st.progress(score, text=f"Score: {score:.0%}")
                    for check_name, check_val in details.items():
                        if check_name != "score":
                            st.write(f"{'✓' if check_val else '✗'} {check_name.replace('_', ' ').title()}")

    # ---------------------------------------------------------------------------
    # Raw Agent Outputs (expandable)
    # ---------------------------------------------------------------------------

    with st.expander("View Raw Agent Outputs"):
        raw_tabs = st.tabs([
            "Market Data", "Rate Data", "Rate Sensitivity", "Dividend Data",
            "Prospectus Terms", "Call Analysis", "Tax Analysis",
            "Regulatory Analysis", "Relative Value", "Agent Status",
        ])

        with raw_tabs[0]:
            st.json(market_data)
        with raw_tabs[1]:
            st.json(rate_data)
        with raw_tabs[2]:
            st.json(rate_sensitivity)
        with raw_tabs[3]:
            st.json(dividend_data)
        with raw_tabs[4]:
            st.json(prospectus_terms)
        with raw_tabs[5]:
            st.json(call_analysis)
        with raw_tabs[6]:
            st.json(tax_analysis)
        with raw_tabs[7]:
            st.json(regulatory_analysis)
        with raw_tabs[8]:
            st.json(relative_value)
        with raw_tabs[9]:
            st.json(agent_status)

elif not ticker:
    st.info("Enter a preferred stock ticker above and click 'Analyze' to begin.")


# ---------------------------------------------------------------------------
# Cached Inventory
# ---------------------------------------------------------------------------

inventory_rows = load_cached_prospectus_inventory()

st.subheader("Available for Quick Analysis")
st.caption(
    "These issues already have cached structured prospectus terms, so they load faster. "
    "You can still enter any other preferred ticker above for live SEC search. "
    "When a new issue is extracted successfully, it is added to your local runtime inventory."
)

if inventory_rows:
    inventory_df = pd.DataFrame(
        [
            {
                "Ticker": row["ticker"],
                "Issuer": row["issuer"],
                "Series": row["series"],
                "Cache": row["cache_label"],
                "Filing Date": row["filing_date"],
                "Accession": row["accession_number"],
            }
            for row in inventory_rows
        ]
    )
    st.dataframe(inventory_df, use_container_width=True, hide_index=True)
else:
    st.info("No cached prospectus inventory found yet.")
