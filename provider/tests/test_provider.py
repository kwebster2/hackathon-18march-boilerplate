"""
Unit tests for the provider agent logic.
All external calls (Anthropic + Tour Guide API) are mocked.

Run: pytest tests/test_provider.py -v
"""

import asyncio
import json
import pytest
from unittest.mock import MagicMock, patch

from orca.domain.models import ChatMessage, Variable


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_chat_message(text: str) -> ChatMessage:
    return ChatMessage(
        thread_id="test-thread",
        model="claude-sonnet-4-6",
        message=text,
        conversation_id=1,
        response_uuid="test-response-uuid",
        message_uuid="test-message-uuid",
        channel="test-channel",
        variables=[Variable(id="1", name="MADHACK-ANTHROPIC-KEY", value="test-key", type="string")],
        url="http://localhost",
    )


def make_anthropic_response(text: str = "", tool_calls: list = None):
    """Build a minimal fake Anthropic response."""
    response = MagicMock()
    response.usage.input_tokens = 100
    response.usage.output_tokens = 50

    if tool_calls:
        response.stop_reason = "tool_use"
        content_blocks = []
        for call in tool_calls:
            block = MagicMock()
            block.type = "tool_use"
            block.id = call["id"]
            block.name = call["name"]
            block.input = call["input"]
            content_blocks.append(block)
        response.content = content_blocks
    else:
        response.stop_reason = "end_turn"
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = text
        response.content = [text_block]

    return response


# ── execute_tool unit tests ───────────────────────────────────────────────────

class TestExecuteTool:
    """Tests for execute_tool() — mocks httpx so no real network calls."""

    def _mock_response(self, data: dict):
        mock = MagicMock()
        mock.json.return_value = data
        mock.raise_for_status = MagicMock()
        return mock

    @patch("main.httpx.Client")
    def test_list_tours(self, MockClient):
        MockClient.return_value.__enter__.return_value.request.return_value = self._mock_response(
            [{"id": 1, "name": "City Walk", "category": "cultural"}]
        )
        from main import execute_tool
        result = execute_tool("list_tours", {"category": "cultural"})
        assert isinstance(result, list)
        assert result[0]["name"] == "City Walk"

    @patch("main.httpx.Client")
    def test_get_tour(self, MockClient):
        MockClient.return_value.__enter__.return_value.request.return_value = self._mock_response(
            {"id": 3, "name": "Jungle Trek", "difficulty": "challenging"}
        )
        from main import execute_tool
        result = execute_tool("get_tour", {"tour_id": 3})
        assert result["id"] == 3

    @patch("main.httpx.Client")
    def test_check_availability(self, MockClient):
        MockClient.return_value.__enter__.return_value.request.return_value = self._mock_response(
            {"available_spots": 8, "tour_id": 2, "date": "2026-04-15"}
        )
        from main import execute_tool
        result = execute_tool("check_availability", {"tour_id": 2, "date": "2026-04-15", "guests": 4})
        assert result["available_spots"] == 8

    @patch("main.httpx.Client")
    def test_get_pricing(self, MockClient):
        MockClient.return_value.__enter__.return_value.request.return_value = self._mock_response(
            {"tour_id": 1, "guests": 2, "price_per_person": 45.0, "total": 90.0}
        )
        from main import execute_tool
        result = execute_tool("get_pricing", {"tour_id": 1, "guests": 2})
        assert result["total"] == 90.0

    @patch("main.httpx.Client")
    def test_book_tour(self, MockClient):
        MockClient.return_value.__enter__.return_value.request.return_value = self._mock_response(
            {"id": 42, "status": "confirmed", "tour_id": 1}
        )
        from main import execute_tool
        result = execute_tool("book_tour", {
            "tour_id": 1,
            "tour_date": "2026-04-20",
            "guest_name": "Alice",
            "guest_email": "alice@example.com",
            "num_guests": 2,
        })
        assert result["id"] == 42
        assert result["status"] == "confirmed"

    @patch("main.httpx.Client")
    def test_cancel_booking(self, MockClient):
        MockClient.return_value.__enter__.return_value.request.return_value = self._mock_response(
            {"id": 42, "status": "cancelled"}
        )
        from main import execute_tool
        result = execute_tool("cancel_booking", {"booking_id": 42})
        assert result["status"] == "cancelled"

    def test_unknown_tool_returns_error(self):
        from main import execute_tool
        result = execute_tool("fly_to_moon", {})
        assert "error" in result


# ── process_message unit tests ────────────────────────────────────────────────

class TestProcessMessage:
    """Tests for the full process_message() handler with all mocks."""

    @patch("main.httpx.Client")
    @patch("main.Anthropic")
    def test_simple_text_response(self, MockAnthropic, MockClient):
        """Claude returns a plain text answer — no tool calls."""
        mock_llm = MagicMock()
        MockAnthropic.return_value = mock_llm
        mock_llm.messages.create.return_value = make_anthropic_response(
            text="We have 12 tours available across 6 categories!"
        )

        mock_session = MagicMock()
        mock_handler = MagicMock()
        mock_handler.begin.return_value = mock_session

        with patch("main.OrcaHandler", return_value=mock_handler):
            from main import process_message
            asyncio.run(process_message(make_chat_message("What tours do you have?")))

        mock_session.stream.assert_called_once()
        streamed = mock_session.stream.call_args[0][0]
        assert "tours" in streamed.lower() or len(streamed) > 0
        mock_session.close.assert_called_once()
        mock_session.error.assert_not_called()

    @patch("main.httpx.Client")
    @patch("main.Anthropic")
    def test_tool_call_then_text(self, MockAnthropic, MockClient):
        """Claude calls list_tours tool, then gives a text answer."""
        mock_llm = MagicMock()
        MockAnthropic.return_value = mock_llm

        tool_response = make_anthropic_response(tool_calls=[{
            "id": "tool_abc",
            "name": "list_tours",
            "input": {"category": "adventure"},
        }])
        text_response = make_anthropic_response(text="Here are the adventure tours I found!")
        mock_llm.messages.create.side_effect = [tool_response, text_response]

        MockClient.return_value.__enter__.return_value.request.return_value = MagicMock(
            json=MagicMock(return_value=[{"id": 1, "name": "Jungle Trek", "category": "adventure"}]),
            raise_for_status=MagicMock(),
        )

        mock_session = MagicMock()
        mock_handler = MagicMock()
        mock_handler.begin.return_value = mock_session

        with patch("main.OrcaHandler", return_value=mock_handler):
            from main import process_message
            asyncio.run(process_message(make_chat_message("Show me adventure tours")))

        assert mock_llm.messages.create.call_count == 2
        mock_session.stream.assert_called_once()
        mock_session.close.assert_called_once()

    @patch("main.Anthropic")
    def test_exception_calls_session_error(self, MockAnthropic):
        """If Anthropic raises, session.error() is called."""
        MockAnthropic.return_value.messages.create.side_effect = Exception("API down")

        mock_session = MagicMock()
        mock_handler = MagicMock()
        mock_handler.begin.return_value = mock_session

        with patch("main.OrcaHandler", return_value=mock_handler):
            from main import process_message
            asyncio.run(process_message(make_chat_message("Hello")))

        mock_session.error.assert_called_once()
        mock_session.stream.assert_not_called()

    @patch("main.httpx.Client")
    @patch("main.Anthropic")
    def test_usage_is_tracked(self, MockAnthropic, MockClient):
        """Token usage should be reported via session.usage.track()."""
        mock_llm = MagicMock()
        MockAnthropic.return_value = mock_llm
        mock_llm.messages.create.return_value = make_anthropic_response(
            text="4 spots available on April 15."
        )

        mock_session = MagicMock()
        mock_handler = MagicMock()
        mock_handler.begin.return_value = mock_session

        with patch("main.OrcaHandler", return_value=mock_handler):
            from main import process_message
            asyncio.run(process_message(make_chat_message("Is tour 1 available on April 15?")))

        mock_session.usage.track.assert_called_once()
        call_kwargs = mock_session.usage.track.call_args[1]
        assert call_kwargs["tokens"] > 0


# ── Negotiation system prompt check ──────────────────────────────────────────

class TestNegotiationPrompt:
    """Verify the system prompt encodes our discount policy."""

    def test_discount_tiers_mentioned_in_system_prompt(self):
        from main import SYSTEM_PROMPT
        assert "5%" in SYSTEM_PROMPT
        assert "10%" in SYSTEM_PROMPT
        assert "15%" in SYSTEM_PROMPT

    def test_negotiation_keyword_in_system_prompt(self):
        from main import SYSTEM_PROMPT
        assert "discount" in SYSTEM_PROMPT.lower() or "negotiat" in SYSTEM_PROMPT.lower()
