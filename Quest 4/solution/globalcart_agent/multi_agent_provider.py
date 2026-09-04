from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
from typing import Any

from .resolver import _interaction_text, _parse_json_output, _response_text


PROVIDER_OPENAI = "openai"
PROVIDER_GROK = "grok"
PROVIDER_GEMINI = "gemini"
PROVIDER_DETERMINISTIC = "deterministic"
MULTI_AGENT_LLM_ENTRYPOINT = "multi_agent_provider.call_multi_agent_llm"


def call_multi_agent_llm(
    *,
    agent_name: str,
    system_prompt: str,
    payload: dict[str, Any],
    output_schema: dict[str, Any],
    text_only: bool = False,
) -> dict[str, Any]:
    """Call the configured Part 2 LLM provider for one specialized agent."""
    provider = select_multi_agent_provider()
    if provider is None:
        return _fallback_packet()
    if os.environ.get("GLOBALCART_TEST_UNSAFE_AGENT3") == "1" and agent_name == "communications":
        return {
            "mode": "llm",
            "provider": "test-unsafe",
            "model": "test",
            "entrypoint": MULTI_AGENT_LLM_ENTRYPOINT,
            "text": "I approved your refund of 999.00 USD and it has already been issued.",
            "parsed": None,
        }
    try:
        text = _call_provider_with_timeout(provider, system_prompt, payload, output_schema, text_only)
        parsed = None if text_only else _parse_json_output(text)
        return {
            "mode": "llm",
            "provider": provider,
            "model": _model_for_provider(provider),
            "entrypoint": MULTI_AGENT_LLM_ENTRYPOINT,
            "text": text.strip(),
            "parsed": parsed,
        }
    except BaseException as exc:
        return {
            "mode": "deterministic_fallback",
            "provider": provider,
            "model": _model_for_provider(provider),
            "entrypoint": MULTI_AGENT_LLM_ENTRYPOINT,
            "text": "",
            "parsed": None,
            "error": str(exc),
        }


def select_multi_agent_provider() -> str | None:
    configured = os.environ.get("MULTI_AGENT_LLM_PROVIDER", "").strip().lower()
    if configured in {"", "auto"}:
        if os.environ.get("OPENAI_API_KEY"):
            return PROVIDER_OPENAI
        if os.environ.get("XAI_API_KEY"):
            return PROVIDER_GROK
        if os.environ.get("GEMINI_API_KEY"):
            return PROVIDER_GEMINI
        return None
    if configured in {"none", "local", PROVIDER_DETERMINISTIC}:
        return None
    if configured == PROVIDER_OPENAI and os.environ.get("OPENAI_API_KEY"):
        return PROVIDER_OPENAI
    if configured == PROVIDER_GROK and os.environ.get("XAI_API_KEY"):
        return PROVIDER_GROK
    if configured == PROVIDER_GEMINI and os.environ.get("GEMINI_API_KEY"):
        return PROVIDER_GEMINI
    return None


def _fallback_packet() -> dict[str, Any]:
    return {
        "mode": "deterministic_fallback",
        "provider": None,
        "model": None,
        "entrypoint": MULTI_AGENT_LLM_ENTRYPOINT,
        "text": "",
        "parsed": None,
    }


def _model_for_provider(provider: str) -> str:
    if os.environ.get("MULTI_AGENT_MODEL"):
        return os.environ["MULTI_AGENT_MODEL"]
    if provider == PROVIDER_OPENAI:
        return os.environ.get("OPENAI_MODEL", "gpt-5")
    if provider == PROVIDER_GROK:
        return os.environ.get("GROK_MODEL", "grok-4.6")
    return os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")


def _timeout_seconds() -> float:
    raw = os.environ.get("MULTI_AGENT_TIMEOUT_SECONDS", "20").strip()
    try:
        timeout = float(raw)
    except ValueError:
        return 20.0
    return timeout if timeout > 0 else 20.0


def _call_provider_with_timeout(
    provider: str,
    system_prompt: str,
    payload: dict[str, Any],
    output_schema: dict[str, Any],
    text_only: bool,
) -> str:
    timeout = _timeout_seconds()
    result_queue: mp.Queue = mp.Queue(maxsize=1)
    process = mp.Process(
        target=_provider_worker,
        args=(result_queue, provider, system_prompt, payload, output_schema, text_only),
        daemon=True,
    )
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(2)
        if process.is_alive():
            process.kill()
            process.join(2)
        raise TimeoutError(f"Multi-agent {provider} provider call timed out after {timeout:g} seconds.")
    try:
        message = result_queue.get_nowait()
    except queue.Empty as exc:
        raise RuntimeError(f"Multi-agent {provider} provider call ended without a result.") from exc
    if message["ok"]:
        return message["text"]
    raise RuntimeError(message["error"])


def _provider_worker(
    result_queue: mp.Queue,
    provider: str,
    system_prompt: str,
    payload: dict[str, Any],
    output_schema: dict[str, Any],
    text_only: bool,
) -> None:
    try:
        text = _call_provider(provider, system_prompt, payload, output_schema, text_only)
        result_queue.put({"ok": True, "text": text})
    except BaseException as exc:
        result_queue.put({"ok": False, "error": str(exc)})


def _call_provider(
    provider: str,
    system_prompt: str,
    payload: dict[str, Any],
    output_schema: dict[str, Any],
    text_only: bool,
) -> str:
    user_prompt = _user_prompt(payload, output_schema, text_only)
    model = _model_for_provider(provider)
    if provider == PROVIDER_OPENAI:
        from openai import OpenAI

        client = OpenAI()
        response = client.responses.create(
            model=model,
            input=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        )
        return _response_text(response)
    if provider == PROVIDER_GROK:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ.get("XAI_API_KEY"), base_url="https://api.x.ai/v1")
        response = client.responses.create(
            model=model,
            input=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        )
        return _response_text(response)

    from google import genai

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    interaction = client.interactions.create(
        model=model,
        store=False,
        input=[
            {
                "type": "user_input",
                "content": [{"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"}],
            }
        ],
    )
    return _interaction_text(interaction)


def _user_prompt(payload: dict[str, Any], output_schema: dict[str, Any], text_only: bool) -> str:
    if text_only:
        return (
            "Use the structured case data below. Return only the customer-facing message text.\n\n"
            f"Structured case data:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
    return (
        "Use the structured case data below. Return only parseable JSON matching this schema shape. "
        "Do not wrap the JSON in markdown.\n\n"
        f"Expected JSON shape:\n{json.dumps(output_schema, ensure_ascii=False, indent=2)}\n\n"
        f"Structured case data:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
