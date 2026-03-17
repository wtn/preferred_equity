# Local Running Instructions: Preferred Equity Swarm (Phase 3)

Follow these steps to run the full Phase 3 swarm on your local machine.

## 1. Update Your Local Repository

Pull the latest changes from GitHub to get the four new agents, the updated graph, and the Streamlit UI:

```bash
cd preferred-equity-swarm
git pull origin master
```

## 2. Update Your Environment

I have added new dependencies for the Phase 3 agents and the LLM fallback. Re-install the requirements:

```bash
pip install -r requirements.txt
```

*Note: If you encounter any missing modules, you may need to manually install the new ones:*
`pip install langgraph langchain-core langchain-google-genai langchain-openai yfinance`

## 3. Configure Your LLM API Key

The system now supports **both** Google Gemini (primary) and OpenAI (fallback). Open your `.env` file and ensure at least one of these is set:

```bash
# Option A: Google Gemini (Primary)
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# Option B: OpenAI (Fallback)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

*The system will automatically use the OpenAI key if the Google key is missing.*

## 4. Run the End-to-End Tests

Before launching the UI, you can verify that all 12 agents are working correctly by running the same test script I used:

```bash
# Test a fixed-rate security (Bank of America Series L)
python3 tests/test_phase3_e2e.py BAC-PL

# Test a floating-rate security (Morgan Stanley Series A)
python3 tests/test_phase3_e2e.py MS-PA
```

## 5. Launch the Streamlit Dashboard

Run the updated Streamlit app to see the new Call Risk, Tax, Regulatory, and Relative Value sections:

```bash
streamlit run streamlit_app/app.py
```

## What's New in the UI?

- **Layered Status Dashboard:** You'll now see status indicators for all 12 agent nodes.
- **Call Risk Analysis:** New metrics for Yield-to-Call, Yield-to-Worst, and heuristic call probability.
- **Tax and Yield Profile:** Displays QDI eligibility and Tax-Equivalent Yield (TEY).
- **Regulatory and Sector Risk:** Shows G-SIB status and Basel III/IV capital treatment.
- **Relative Value:** Ranks the security against its peers in the cached universe.
- **Expanded Raw Outputs:** You can now inspect the raw JSON from all 9 analytical agents in the bottom expander.
