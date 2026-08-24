# GlobalCart Operations Resolver Agent

Quest #04 Part A implementation: a single technical support agent that resolves
complex GlobalCart refund tickets using the provided starter-kit tools.

The assignment pages are treated as context for this implementation. The
starter-kit data and services are not modified.

## Architecture

The solution has two modes:

- `openai`: main agent mode. It exposes the four starter-kit functions as
  OpenAI function tools and executes the model-requested calls.
- `grok`: provider mode using xAI's OpenAI-compatible Responses API and custom
  function calling.
- `gemini`: provider mode using Google Gemini's Interactions API and custom
  function calling.
- `langgraph-local`: deterministic LangGraph workflow. It uses shared state and
  explicit nodes to orchestrate the same local business logic without an LLM.
- `multi-agent`: default and official Part 2 implementation. It uses LangGraph
  to coordinate three specialized LLM-assisted agents: Researcher & Fraud
  Auditor, Decision Maker / Operations Lead, and Communications & Escalation
  Manager. Each agent has its own role prompt, structured handoff, and provider
  call through `globalcart_agent.multi_agent_provider.call_multi_agent_llm` by
  default when an API key is configured. Deterministic tools and validators keep
  authority over decisions, refunds, fraud rules, policy ids, escalation status,
  and the final `action_taken` contract.
- `local`: deterministic fallback. It performs the same order -> user -> policy
  -> refund investigation without an API key, useful for demos and regression
  tests.
- `auto`: aliases the official `multi-agent` path.

Part 1 modes call only the starter-kit tools:

- `get_order_details`
- `get_user_profile`
- `check_return_policy`
- `process_refund`

Part 2 `multi-agent` adds deterministic mock tools for:

- `audit_fraud_risk`
- `send_slack_alert`

## Setup

From this folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For OpenAI mode, copy `.env.example` to `.env` or export the variables in your
shell:

```bash
set OPENAI_API_KEY=your_api_key_here
set OPENAI_MODEL=gpt-5
set XAI_API_KEY=your_xai_api_key_here
set GROK_MODEL=grok-4.6
set GEMINI_API_KEY=your_gemini_api_key_here
set GEMINI_MODEL=gemini-3.6-flash
set MULTI_AGENT_LLM_PROVIDER=auto
set MULTI_AGENT_MODEL=
```

For the Part 2 `multi-agent` path, provider selection is independent from the
CLI mode and lives in `multi_agent_provider.py`. Set
`MULTI_AGENT_LLM_PROVIDER=openai`, `grok`, `gemini`, or leave it as `auto`.
Auto prefers OpenAI, then Grok, then Gemini, and falls back to deterministic
agent behavior when no usable provider is configured. Part 2 agents should not
call the Part 1 provider resolvers directly.

The OpenAI SDK reads `OPENAI_API_KEY` from the environment, as described in the
official OpenAI quickstart:
https://platform.openai.com/docs/quickstart/make-your-first-api-request

Grok mode uses xAI's OpenAI-compatible function-calling flow with
`base_url="https://api.x.ai/v1"`:
https://docs.x.ai/developers/tools/function-calling

Gemini mode uses Google Gemini custom function calling through the Interactions
API:
https://ai.google.dev/gemini-api/docs/function-calling

## Run

Local deterministic mode:

```bash
python run_agent.py --mode local "Hi, I'm Maya. My earbuds from order ORD-1001 arrived cracked right out of the box."
```

LangGraph deterministic workflow mode:

```bash
python run_agent.py --mode langgraph-local "Hi, I'm Maya. My earbuds from order ORD-1001 arrived cracked right out of the box."
```

Part 2 multi-agent mode:

```bash
python run_agent.py --mode multi-agent "This is Ronen, order ORD-1005. The tablet screen was smashed on arrival. Refund me, this keeps happening."
```

OpenAI tool-calling mode:

```bash
python run_agent.py --mode openai "Order ORD-1002. The espresso machine is dented and leaking. I paid 150 dollars for this."
```

Grok tool-calling mode:

```bash
python run_agent.py --mode grok "Hi, I'm Maya. My earbuds from order ORD-1001 arrived cracked right out of the box."
```

Gemini tool-calling mode:

```bash
python run_agent.py --mode gemini "Hi, I'm Maya. My earbuds from order ORD-1001 arrived cracked right out of the box."
```

Default multi-agent mode:

```bash
python run_agent.py "My order ORD-2222 never arrived and I want the $300 back."
```

The output is parseable JSON with exactly three top-level fields:

- `reasoning_chain`
- `action_taken`
- `customer_response`

`reasoning_chain` is an audit trail based on tool outputs, not private hidden
chain-of-thought.

In `multi-agent` mode, `action_taken` also reports multi-agent execution
metadata:

- `agent3_response_mode`: `llm` or `deterministic`
- `agent3_llm_provider`: `openai`, `grok`, `gemini`, or `null`
- `agent3_llm_error`: present only when an attempted provider call fails
- `agent_execution`: per-agent metadata for `researcher`, `decision_maker`, and
  `communications`, showing the provider/model or deterministic fallback plus
  the `multi_agent_provider.call_multi_agent_llm` entrypoint.

Agent 1 may call only order, customer, and fraud-audit tools. Agent 2 may call
only policy and refund tools. Agent 3 may call only the mock escalation tool.
The LLMs can summarize, propose, and write, but verified tools remain the source
of truth for all business facts.

## Guardrails And Edge Cases

Business errors are handled as data. If a tool returns an `error` key, the agent
stops, reports the issue honestly, and does not retry in a loop.

The agent never says a refund was issued unless `process_refund` returns
`APPROVED`. If `process_refund` returns `ESCALATION_REQUIRED`, the customer
response says the case was escalated and that no automatic refund was paid.

Known regression cases:

- `ORD-1001`: approved VIP refund under cap.
- `ORD-1002`: escalated because amount is above Standard cap.
- `ORD-1003`: rejected outside return window, citing `POL-RET-01`.
- `ORD-1008`: rejected non-returnable category, citing `POL-REF-03`.
- `ORD-1010`: approved at 48 USD.
- `ORD-1011`: escalated at 52 USD.
- `ORD-1005`: escalated due to risk and repeat claims.
- `ORD-1007`: rejected because order has not shipped.
- `ORD-2222`: reports missing order instead of inventing data.

## Verify

First verify the starter kit:

```bash
cd "..\Stage 1\starter-kit"
python examples\verify_scenarios.py
```

Then run this solution's default scenario suite. It uses `auto`, which now
aliases the official `multi-agent` path:

```bash
cd ..\..\solution
python run_scenarios.py
```

For deterministic baseline checks that never call an LLM provider:

```bash
python run_scenarios.py --mode local
python run_scenarios.py --mode langgraph-local
```

For the Part 2 multi-agent regression suite:

```bash
python run_part2_scenarios.py
```

This script now uses the same provider-preferred Part 2 behavior as
`multi-agent`: OpenAI, then Grok, then Gemini, then deterministic fallback.
For a stable no-LLM baseline, run:

```bash
python run_part2_scenarios.py --deterministic
```
