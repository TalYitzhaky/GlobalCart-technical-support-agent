"""HTTP entrypoint for the GlobalCart ticket-resolution workflow."""

from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from globalcart_agent import resolve_ticket


app = FastAPI(title="GlobalCart Resolver API", version="1.0.0")


class ResolveTicketRequest(BaseModel):
    ticket: str = Field(min_length=1, description="Customer support ticket text.")
    mode: Literal[
        "auto", "local", "langgraph-local", "multi-agent", "openai", "grok", "gemini"
    ] = "multi-agent"

    @field_validator("ticket")
    @classmethod
    def validate_ticket(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Ticket must contain non-whitespace text.")
        return value


class ResolveTicketResponse(BaseModel):
    reasoning_chain: list[str]
    action_taken: dict[str, Any]
    customer_response: str


@app.post("/resolve-ticket", response_model=ResolveTicketResponse)
def resolve_ticket_endpoint(request: ResolveTicketRequest) -> dict[str, Any]:
    # A synchronous route runs blocking workflow/provider calls in FastAPI's thread pool.
    return resolve_ticket(request.ticket, mode=request.mode)
