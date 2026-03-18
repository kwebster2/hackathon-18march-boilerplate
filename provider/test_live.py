"""
Live test — sends real messages to the running provider agent.

Usage:
    # 1. Start the provider in dev mode (separate terminal):
    ORCA_DEV_MODE=true ANTHROPIC_API_KEY=sk-ant-... python main.py

    # 2. Run this script with your Anthropic key:
    ANTHROPIC_API_KEY=sk-ant-... python test_live.py

The script uses response_mode=sync so it blocks and prints the full response.
"""

import os
import sys
import uuid
import httpx

BASE_URL = "http://localhost:8000"
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

if not ANTHROPIC_KEY:
    print("ERROR: Set ANTHROPIC_API_KEY environment variable")
    sys.exit(1)


def send(message: str, label: str = "") -> str:
    payload = {
        "thread_id": str(uuid.uuid4()),
        "model": "claude-sonnet-4-6",
        "message": message,
        "conversation_id": 1,
        "response_uuid": str(uuid.uuid4()),
        "message_uuid": str(uuid.uuid4()),
        "channel": str(uuid.uuid4()),
        "variables": [
            {"id": "1", "name": "MADHACK-ANTHROPIC-KEY", "value": ANTHROPIC_KEY, "type": "string"}
        ],
        "url": "http://localhost",
        "response_mode": "sync",
        "stream_mode": False,
        "chat_history": [],
    }

    print(f"\n{'='*60}")
    print(f"TEST: {label or message}")
    print("="*60)

    with httpx.Client(timeout=30) as client:
        r = client.post(f"{BASE_URL}/api/v1/send_message", json=payload)
        r.raise_for_status()
        data = r.json()

    content = data.get("content", data)
    print(content)
    return content


def run_tests():
    # Health check first
    with httpx.Client(timeout=5) as c:
        health = c.get(f"{BASE_URL}/api/v1/health").json()
    print(f"Agent health: {health['status']} — {health['service']}")

    send("What tours do you have available?", "List all tours")

    send(
        "Show me nature tours suitable for families with kids.",
        "Filter tours by category"
    )

    send(
        "Check availability for tour ID 1 on 2026-04-20 for 3 guests.",
        "Availability check"
    )

    send(
        "What's the price for tour ID 2 for 8 people? Can we get a group discount?",
        "Group discount negotiation"
    )

    send(
        "Book tour ID 1 on 2026-04-25 for 2 guests. "
        "Name: Jane Smith, email: jane@example.com",
        "Full booking"
    )


if __name__ == "__main__":
    run_tests()
