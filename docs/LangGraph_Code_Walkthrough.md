# LangGraph Code Walkthrough: Understanding the Preferred Equity Swarm

**Author:** Manus AI
**Date:** March 2026
**Project:** Preferred Equity Analysis Swarm (MSBA Capstone)

This document walks through the codebase, explains the LangGraph concepts behind each design decision, and maps the patterns you are learning now to the full eight-agent swarm you will build by December. It covers both the introductory "Hello World" concepts and the advanced features implemented in Phase 2.

---

## 1. The Big Picture: What Is LangGraph?

LangGraph is a framework for building **stateful, multi-step AI applications** as directed graphs. Think of it as a flowchart engine where each box in the flowchart is an "agent" (a Python function), and the arrows between boxes define the order of execution. The key insight is that LangGraph manages a **shared state object** that every agent can read from and write to, which is how agents communicate with each other without needing to call each other directly.

There are three core concepts you need to internalize:

| Concept | What It Is | Analogy |
|---|---|---|
| **State** | A typed dictionary that flows through the graph | A shared clipboard that every agent can read and write |
| **Node** | A Python function that receives state, does work, and returns state updates | A worker at a station on an assembly line |
| **Edge** | A connection between nodes that defines execution order | The conveyor belt between stations |

The power of LangGraph over simpler approaches (like chaining LLM calls in a loop) is that it gives you **conditional routing** (if/else logic between agents), **parallel execution** (multiple agents running at the same time), and **cycles** (agents that can loop back and re-run based on results). These are the patterns that make a true "swarm" possible.

---

## 2. File-by-File Walkthrough

### 2.1 Configuration: `src/utils/config.py`

```python
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "PreferredEquitySwarm research@example.com")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
```

**What this does:** This is the central configuration file. It loads API keys from a `.env` file (which is gitignored so your secrets never get committed) and defines file paths. Every other module imports from here rather than hardcoding values.

**Why it matters for LangGraph:** In a multi-agent system, you want a single source of truth for configuration. If you later switch from Gemini to GPT-4, you change one line here and every agent picks it up automatically. This is especially important when you have eight agents that all need LLM access.

---

### 2.2 Data Layer: `src/data/market_data.py`

```python
def get_preferred_info(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    info = stock.info
    result = {
        "ticker": ticker,
        "name": info.get("longName", info.get("shortName", "Unknown")),
        "price": info.get("regularMarketPrice", info.get("previousClose", None)),
        "dividend_rate": info.get("dividendRate", None),
        "dividend_yield": info.get("dividendYield", None),
        # ... more fields
    }
    return result
```

**What this does:** This is a pure data-fetching module. It wraps the `yfinance` library to pull real-time preferred stock data from Yahoo Finance. Notice that it returns a plain Python dictionary, not a LangGraph-specific object. This is intentional.

**Design principle: Keep agents thin, keep data modules reusable.** The `market_data.py` module knows nothing about LangGraph. It is a standalone utility that could be used in a Jupyter notebook, a Flask API, or a command-line script. The LangGraph agent calls this module and then packages the result into the graph state. This separation means you can test your data pipelines independently of your agent logic.

---

### 2.3 Advanced Data Layer: `src/data/rate_sensitivity.py`

In Phase 2, we introduced complex benchmark resolution for floating-rate securities.

```python
def _resolve_benchmark_context(prospectus_terms: dict, rate_data: dict) -> dict:
    # Determines if a legacy LIBOR security needs to be mapped to a SOFR proxy
    # Returns a rich dictionary explaining the substitution
```

**Why this matters:** Financial analysis requires handling edge cases gracefully. By encapsulating the LIBOR-to-SOFR transition logic in a dedicated data module, the downstream Rate Context Agent remains clean and focused solely on state management.

---

### 2.4 The Core: `src/agents/advanced_swarm.py`

This is the heart of the Phase 2 project, demonstrating parallel execution and conditional routing. Let us walk through it section by section.

#### Section A: The State Schema

```python
class SwarmState(TypedDict):
    """State that is passed between agents in the swarm."""
    ticker: str
    market_data: dict
    rate_data: dict
    prospectus_terms: dict
    dividend_data: dict
    synthesis: str
    errors: list
    agent_status: dict
```

**This is the most important design decision in the entire project.** The `SwarmState` defines the shared data structure that every agent reads from and writes to. Think of it as a contract: every agent promises to read certain fields and write certain fields.

**Key LangGraph concept:** When an agent returns a dictionary like `{"market_data": info}`, LangGraph **merges** that dictionary into the existing state. It does not replace the entire state. This merge behavior is what allows agents to work independently without stepping on each other's data.

#### Section B: Agent Node Functions

```python
def market_data_agent(state: SwarmState) -> dict:
    ticker = state["ticker"]                          # READ from state
    info = get_preferred_info(ticker)                  # DO work
    return {"market_data": info}                       # WRITE to state
```

**The pattern is always the same: Read, Do, Write.** Every agent function follows this three-step pattern:

1. **Read** the fields it needs from the state dictionary
2. **Do** its specialized work (fetch data, call an API, run an LLM)
3. **Write** its results back by returning a dictionary with the fields it owns

The Market Data Agent, Rate Context Agent, and Prospectus Parsing Agent are **tool agents**: they execute deterministic operations (API calls or regex parsing) with occasional LLM fallbacks. The Synthesis Agent is a **reasoning agent**: it uses Gemini to interpret and synthesize the data collected by the tool agents.

#### Section C: Conditional Routing

```python
def quality_gate(state: SwarmState) -> str:
    """Determine whether to proceed to synthesis or fail gracefully."""
    errors = state.get("errors", [])
    if errors:
        return "fail"
    return "pass"
```

**This is a crucial pattern.** Instead of letting the Synthesis Agent try to write a report with missing data, the quality gate inspects the state. If any parallel agent reported a failure, the graph routes execution to an error handler instead.

#### Section D: Building the Graph

```python
def build_advanced_graph() -> StateGraph:
    workflow = StateGraph(SwarmState)

    # Add agent nodes
    workflow.add_node("market_data_agent", market_data_agent)
    workflow.add_node("rate_context_agent", rate_context_agent)
    workflow.add_node("prospectus_agent", prospectus_agent_node)
    workflow.add_node("dividend_agent", dividend_agent_node)
    workflow.add_node("synthesis_agent", synthesis_agent)
    workflow.add_node("error_handler", error_handler_node)

    # Define execution order (Parallel Fan-Out)
    workflow.set_entry_point("prospectus_agent")
    workflow.add_edge("prospectus_agent", "market_data_agent")
    workflow.add_edge("prospectus_agent", "rate_context_agent")
    workflow.add_edge("prospectus_agent", "dividend_agent")

    # Fan-In to Quality Gate
    workflow.add_conditional_edges(
        ["market_data_agent", "rate_context_agent", "dividend_agent"],
        quality_gate,
        {
            "pass": "synthesis_agent",
            "fail": "error_handler"
        }
    )

    workflow.add_edge("synthesis_agent", END)
    workflow.add_edge("error_handler", END)

    return workflow.compile()
```

**This is where LangGraph shines.** You are building a directed graph programmatically. The Phase 2 graph uses **parallel branches** (market data, rate context, and dividend analysis run simultaneously after the prospectus is parsed) and **conditional edges** (the quality gate).

#### Section E: Running the Graph

```python
def analyze_preferred(ticker: str) -> dict:
    graph = build_advanced_graph()

    initial_state = {
        "ticker": ticker,
        # ... initialize other fields
    }

    result = graph.invoke(initial_state)
    return result
```

**`graph.invoke(initial_state)` is the single line that runs the entire swarm.** You pass in the initial state, and LangGraph executes each node in order, merging each agent's output into the state as it goes. You do not write any orchestration loops yourself.

---

## 3. Visual Graph Diagram

The diagram below shows the current Phase 2 execution flow.

```
                    +-- market_data_agent --+
                    |                       |
prospectus_agent ---+-- rate_context_agent -+--[fan-in]--> quality_check
                    |                       |                    |
                    +-- dividend_agent -----+              [conditional]
                                                           /         \
                                                     [pass]         [fail]
                                                       |               |
                                                synthesis_agent    error_handler
                                                       |               |
                                                      END             END
```

---

## 4. Mapping Phase 2 to the Full Swarm

Every pattern in the Phase 2 code maps directly to a pattern in the full eight-agent swarm you will build next:

| Phase 2 Pattern | Full Swarm Pattern |
|---|---|
| `SwarmState` with 8 fields | `SwarmState` with 20+ fields covering all analysis dimensions |
| 4 tool agents | 5 tool agents (adding tax and regulatory) |
| 1 reasoning agent (synthesis) | 3 reasoning agents (adding call probability and relative value) |
| Parallel fan-out/fan-in | Deeper parallel branches with complex inter-agent signals |
| Quality gate | Dynamic conflict resolution between contradictory agent outputs |

The architecture scales because the fundamental pattern never changes: define state, define agents that read/do/write, define edges that control flow, compile, invoke.

---

## 5. Key Takeaways

**State is the backbone.** Design your `SwarmState` carefully because it defines what agents can communicate. Every inter-agent dependency is expressed through shared state fields.

**Agents are just functions.** There is no special agent class or decorator. Any Python function that takes state and returns a partial state update is a valid LangGraph node. This makes testing trivial.

**Edges are the orchestration.** The graph structure (not the agent code) determines execution order, parallelism, and conditional branching. This separation of concerns is what makes the system maintainable as it grows.

**The graph compiles to a runnable.** Once you call `compile()`, you get an object with an `invoke()` method that handles all the execution mechanics.

---

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) [1]
- [LangGraph Conceptual Guide: State Management](https://langchain-ai.github.io/langgraph/concepts/low_level/) [2]
- [LangChain Google Generative AI Integration](https://python.langchain.com/docs/integrations/chat/google_generative_ai/) [3]
- [yfinance Documentation](https://github.com/ranaroussi/yfinance) [4]
- [Streamlit Documentation](https://docs.streamlit.io/) [5]

[1]: https://langchain-ai.github.io/langgraph/
[2]: https://langchain-ai.github.io/langgraph/concepts/low_level/
[3]: https://python.langchain.com/docs/integrations/chat/google_generative_ai/
[4]: https://github.com/ranaroussi/yfinance
[5]: https://docs.streamlit.io/
