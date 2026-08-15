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
- `local`: deterministic fallback. It performs the same order -> user -> policy
  -> refund investigation without an API key, useful for demos and regression
  tests.
- `auto`: uses OpenAI when `OPENAI_API_KEY` is set, then Grok when
  `XAI_API_KEY` is set, then Gemini when `GEMINI_API_KEY` is set, otherwise
  local mode.

Both modes call only:

- `get_order_details`
- `get_user_profile`
- `check_return_policy`
- `process_refund`

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
```

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

Auto mode:

```bash
python run_agent.py "My order ORD-2222 never arrived and I want the $300 back."
```

The output is parseable JSON with exactly three top-level fields:

- `reasoning_chain`
- `action_taken`
- `customer_response`

`reasoning_chain` is an audit trail based on tool outputs, not private hidden
chain-of-thought.

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
cd ..\starter-kit
python examples\verify_scenarios.py
```

Then run this solution's local regression suite:

```bash
cd ..\solution
python run_scenarios.py
```
