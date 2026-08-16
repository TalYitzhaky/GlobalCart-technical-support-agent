#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from globalcart_agent import resolve_ticket


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the GlobalCart Operations Resolver Agent.")
    parser.add_argument("ticket", nargs="?", help="Customer ticket text.")
    parser.add_argument("--ticket-file", help="Path to a text file containing the customer ticket.")
    parser.add_argument("--mode", choices=["auto", "local", "langgraph-local", "openai", "grok", "gemini"], default="auto")
    args = parser.parse_args()

    if args.ticket_file:
        ticket = Path(args.ticket_file).read_text(encoding="utf-8").strip()
    elif args.ticket:
        ticket = args.ticket
    else:
        ticket = sys.stdin.read().strip()

    if not ticket:
        parser.error("Provide a ticket argument, --ticket-file, or stdin text.")

    result = resolve_ticket(ticket, mode=args.mode)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
