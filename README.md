# Preferred Equity Analysis Swarm

**MSBA Capstone Project**

A multi-agent AI swarm system that analyzes preferred equity securities using LangGraph and Google Gemini. The swarm coordinates specialized agents to evaluate credit risk, interest rate sensitivity, call probability, tax treatment, regulatory exposure, and relative value across the preferred securities universe.

## Project Structure

```
preferred-equity-swarm/
├── src/
│   ├── agents/          # LangGraph agent definitions (prospectus_agent, advanced_swarm)
│   ├── data/            # Data pipeline modules (edgar_pipeline, market_data, rate_data)
│   └── utils/           # Shared utilities (config)
├── streamlit_app/       # Streamlit demo interface
├── tests/               # Unit and integration tests
├── notebooks/           # Jupyter notebooks for exploration
├── docs/                # Project documentation and architectural walkthroughs
├── data/
│   ├── edgar_cache/     # Local cache for SEC filings to reduce API calls
│   ├── prospectus_terms/# Extracted and structured prospectus data (demo/runtime)
│   └── preferred_filing_registry.json # Curated registry of known preferred filings
└── requirements.txt     # Python dependencies
```

## Technology Stack

| Component | Technology |
|---|---|
| Agent Orchestration | LangGraph |
| LLM Backend | Google Gemini (`gemini-2.5-flash`) via `langchain-google-genai` |
| SEC Filings | SEC EDGAR EFTS and Submissions APIs |
| Market Data | yfinance |
| Rate Data | FRED API and Yahoo Finance ETF proxies |
| Demo UI | Streamlit |
| Visualization | Plotly |

## Setup

```bash
# Clone the repository
git clone https://github.com/jkriertran/preferred-equity-swarm.git
cd preferred-equity-swarm

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

## Running the Application

To run the interactive Streamlit dashboard:

```bash
cd preferred-equity-swarm
streamlit run streamlit_app/app.py
```

## Current Status: Phase 3 Completed

The project has successfully completed Phase 3, expanding the swarm to 12 agent nodes operating across 5 distinct layers.

### Completed Capabilities

**Layer 1: Parallel Data Collection**
*   **Market Data Agent:** Fetches real-time pricing and dividend data.
*   **Rate Context Agent:** Pulls live Treasury yield curves and SOFR benchmark rates.
*   **Dividend Analysis Agent:** Computes dividend consistency, payment frequency, and trailing yield.
*   **Prospectus Parsing Agent:** A staged extraction pipeline that uses deterministic parsing for standard terms and falls back to Gemini for complex legal language.

**Layer 2: Deterministic Analysis**
*   **Interest Rate Sensitivity Agent:** Computes duration, DV01, and handles the transition from LIBOR to SOFR for legacy floating-rate securities.

**Layer 3: Parallel Analytical Agents (Phase 3 additions)**
*   **Call Probability Agent:** Estimates yield-to-call, yield-to-worst, and heuristic call probability based on refinancing incentives and premium to par.
*   **Tax and Yield Agent:** Classifies Qualified Dividend Income (QDI) eligibility and computes tax-equivalent yields.
*   **Regulatory and Sector Agent:** Assesses Basel III/IV AT1 capital treatment, G-SIB surcharges, and dividend deferral risk.
*   **Relative Value Agent:** Ranks the security against peers by yield, spread to Treasury, and structure.

**Layer 4 & 5: Routing and Synthesis**
*   **Quality Gate:** Evaluates the outputs of all 8 upstream agents to determine if the data is sufficient for synthesis.
*   **Synthesis Agent:** Synthesizes the outputs into a professional, institutional-grade research note.
*   **Error Report Agent:** Generates a structured diagnostic report if the quality gate fails.

**User Interface**
*   A polished Streamlit dashboard that visualizes yield curves, price history, benchmark context, and the outputs of all Phase 3 analytical agents.

## Next Phase: Phase 4 (Orchestration & Refinement)

The upcoming final phase will focus on:
1.  **Orchestrator Agent:** Adding a supervisor node for workflow management and conflict resolution.
2.  **Portfolio Analysis:** Expanding the swarm to analyze multiple securities simultaneously.
3.  **Final Polish:** Optimizing prompts, token usage, and UI presentation for the final Capstone deliverable.
