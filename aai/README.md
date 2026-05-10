# aai — Agentic AI examples

This folder contains small, self-contained examples and mini-labs that demonstrate agent workflows using LangGraph, LLMs (Google Gemini), and simple tool integrations.

Table of contents

- ch2/
  - `agent-search-summary.py` — Two-agent LangGraph workflow: Wikipedia search + Gemini summarizer.

- ch3/
  - `01-react-agent.py` — Minimal React-style agent that can call tools (example: calculate_area).
  - `02-retry-tool.py` — Retry / resilience example: node retried up to 3 times via conditional edges.
  - `03-stream-output.py` — Shows streaming LLM output and printing chunks live.
  - `04-decision-evaluation.py` — Reflection example: have the LLM evaluate and correct a prior answer.
  - `05-minilab.py` — Mini lab: Alpha Vantage stock fetch + LLM summary + reflection check.
  - `06-p-tuning.py` — Small prompt-search demo (plus optional guarded PEFT init helper).

- ch4/
  - `01-langsmith-eval.py` — LangChain / LangSmith evaluation example.
  - `02-tuning-workflow.py` — Tuning workflow examples.
  - `04-minilab.py` — Additional mini-lab in chapter 4.

- ch5/
  - `01-langgraph-planner.py` — Planner + executor workflow using LangGraph and Gemini.
  - `02-langgraph-memory.py` — Conversation memory with SQLite checkpointer and LangGraph.
  - `03-reflection-llm.py` — Multi-stage generate → reflect → improve using the LLM.
  - `05-minilab.py` — Another mini-lab showing tool usage and a reflection/completeness node.

Quick notes

- Most examples expect a `.env` file with `GOOGLE_API_KEY` for Gemini. Some examples also use `ALPHA_VANTAGE_API_KEY` for live stock data.
- The `06-p-tuning.py` file includes a helper `init_peft_prompt_tuning(...)` that will initialize a PEFT prompt-tuning model only when explicitly enabled via environment variables (`USE_PEFT=1` and `PEFT_BASE_MODEL`).

How to run an example

1. Create a `.env` file with any required keys (e.g. `GOOGLE_API_KEY`):

```bash
# .env
GOOGLE_API_KEY=sk-...
ALPHA_VANTAGE_API_KEY=your_key_here
```

2. Run a script:

```bash
python aai/ch3/05-minilab.py
```

Contributing

If you add examples, please include a short top-level docstring explaining the purpose and any required environment variables.
