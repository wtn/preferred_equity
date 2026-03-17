# Phase 3 Completion Report: Advanced Analytical Agents

This document summarizes the completion of Phase 3 of the Preferred Equity Analysis Swarm capstone project.

## Architecture Evolution

In Phase 3, the swarm architecture evolved from a 2-layer parallel pipeline to a 5-layer pipeline with 12 distinct agent nodes.

1.  **Layer 1: Parallel Data Collection** (Market, Rate, Dividend, Prospectus)
2.  **Layer 2: Deterministic Analysis** (Interest Rate Sensitivity)
3.  **Layer 3: Parallel Analytical Agents** (Call, Tax, Regulatory, Relative Value)
4.  **Layer 4: Quality Gate** (Scoring all 8 upstream agents)
5.  **Layer 5: Conditional Routing** (Synthesis or Error Report)

## New Agents Implemented

### 1. Call Probability Agent (`src/data/call_analysis.py`)
*   **Purpose:** Evaluates the risk of early redemption by the issuer.
*   **Key Metrics:** Yield-to-call (YTC), yield-to-worst (YTW), years to call, refinancing incentive, premium to call.
*   **Heuristic Logic:** Calculates the refinancing incentive by comparing the security's current yield to current prevailing rates (using the Treasury curve + historical spread assumptions). Assigns a categorical probability (Very High, High, Moderate, Low, Very Low, Not Callable) based on the incentive and the premium to par.

### 2. Tax and Yield Agent (`src/data/tax_analysis.py`)
*   **Purpose:** Determines the tax treatment of the dividend and calculates the tax-equivalent yield.
*   **Key Metrics:** QDI eligibility, after-tax yield, tax-equivalent yield (TEY), effective tax rate.
*   **Heuristic Logic:** Uses a rules-based approach to determine Qualified Dividend Income (QDI) eligibility based on the issuer type (financial/bank vs. REIT) and security structure. Computes TEY using standard top-bracket federal tax rates (37% ordinary income, 20% capital gains + 3.8% NIIT).

### 3. Regulatory and Sector Agent (`src/data/regulatory_analysis.py`)
*   **Purpose:** Assesses the regulatory capital treatment and sector-specific risks, primarily for financial issuers.
*   **Key Metrics:** Sector classification, G-SIB status, capital treatment (e.g., AT1), regulatory risk level, dividend deferral risk.
*   **Heuristic Logic:** Identifies G-SIB (Global Systemically Important Bank) issuers using a hardcoded registry of major US banks. Assesses whether the security qualifies as Additional Tier 1 (AT1) capital based on the prospectus terms (e.g., non-cumulative, perpetual). Evaluates dividend deferral risk based on the cumulative/non-cumulative nature of the preferred stock.

### 4. Relative Value Agent (`src/data/relative_value.py`)
*   **Purpose:** Compares the security to a universe of peers to determine its relative attractiveness.
*   **Key Metrics:** Value assessment (Rich, Fair, Cheap), peer count, yield rank, spread to Treasury.
*   **Heuristic Logic:** Filters the cached universe for securities with similar credit profiles (e.g., other bank preferreds). Ranks the target security's yield against the peer group. Assigns a value assessment based on its quartile ranking (top quartile = Cheap, bottom quartile = Rich).

## Integration and Testing

*   **Graph Updates:** The `advanced_swarm.py` module was completely rewritten to integrate the new agents as a parallel fan-out layer after the Interest Rate Sensitivity agent.
*   **State Schema:** The LangGraph state dictionary was expanded to include `call_analysis`, `tax_analysis`, `regulatory_analysis`, and `relative_value`.
*   **Quality Gate:** The `quality_check_agent` was updated to score the completeness of the four new analytical outputs.
*   **Synthesis Prompt:** The Gemini prompt in `institutional_synthesis_prompt.md` and `advanced_swarm.py` was updated to instruct the LLM to synthesize the new analytical findings into its final report.
*   **UI Updates:** The Streamlit dashboard (`app.py`) was expanded with four new sections to visualize the outputs of the Phase 3 agents.
*   **Testing:** End-to-end tests were run on both fixed-rate (`BAC-PL`) and floating-rate (`MS-PA`) securities. Both tests passed with a 1.00 quality score, confirming that all 12 agents execute correctly and the LIBOR-to-SOFR transition logic functions as expected within the expanded pipeline.

## Configuration Fallback

During testing, a fallback was added to `src/utils/config.py` to allow the use of OpenAI-compatible models (`gpt-4.1-mini`) when a Google Gemini API key is not present in the environment. This improves the portability of the project.

## Next Steps

The project is now ready for Phase 4, which will introduce an Orchestrator Agent to manage workflow conflicts and expand the system to handle multi-security portfolio analysis.
