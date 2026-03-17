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

## Current Status: Phase 2 Completed

The project has successfully completed Phase 2, delivering a robust vertical slice of the swarm architecture.

### Completed Capabilities
*   **EDGAR Pipeline:** A full data layer that searches and downloads prospectus supplements from the SEC EDGAR database.
*   **Prospectus Parsing Agent:** A staged extraction pipeline that uses deterministic parsing for standard terms and falls back to Gemini for complex legal language. It includes a cache-first approach for rapid demo execution.
*   **Market Data Agent:** Fetches real-time pricing and dividend data.
*   **Rate Context Agent:** Pulls live Treasury yield curves and SOFR benchmark rates. It includes sophisticated logic to handle the transition from LIBOR to SOFR for legacy floating-rate securities.
*   **Dividend Analysis Agent:** Computes dividend consistency, payment frequency, and trailing yield.
*   **Synthesis Agent:** Synthesizes the outputs of all data agents into a professional, institutional-grade research note.
*   **Orchestration:** A LangGraph workflow featuring parallel execution, quality gating, and conditional routing.
*   **User Interface:** A polished Streamlit dashboard that visualizes yield curves, price history, and the newly added benchmark context for floating-rate securities.

## Next Phase: Phase 3 (Advanced Agents)

The upcoming phase will expand the swarm from four agents to eight, adding the following capabilities:
1.  **Call Probability Agent:** To estimate the likelihood of early redemption based on rate environments.
2.  **Tax and Yield Agent:** To classify Qualified Dividend Income eligibility.
3.  **Regulatory and Sector Agent:** To monitor bank capital requirements.
4.  **Relative Value Agent:** To rank securities against peers.
