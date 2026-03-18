import json
import logging
import httpx
from anthropic import Anthropic
from orca import create_agent_app, ChatMessage, OrcaHandler, Variables

logger = logging.getLogger(__name__)

TOUR_API_BASE = "https://hacketon-18march-api.orcaplatform.ai/tour-guide-1/api"
TOUR_API_KEY = "tour-guide-1-key-vwx234"

TOOLS = [
    {
        "name": "list_tours",
        "description": "List available tours, optionally filtered by category, difficulty, max_price, or location.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": [
                        "cultural",
                        "adventure",
                        "food",
                        "nature",
                        "nightlife",
                        "historical",
                    ],
                },
                "difficulty": {
                    "type": "string",
                    "enum": ["easy", "moderate", "challenging"],
                },
                "max_price": {
                    "type": "number",
                    "description": "Maximum price per person",
                },
                "location": {
                    "type": "string",
                    "description": "Partial match on location name",
                },
            },
        },
    },
    {
        "name": "get_tour",
        "description": "Get full details of a specific tour by ID.",
        "input_schema": {
            "type": "object",
            "properties": {"tour_id": {"type": "integer"}},
            "required": ["tour_id"],
        },
    },
    {
        "name": "check_availability",
        "description": "Check how many spots are available for a tour on a specific date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tour_id": {"type": "integer"},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "guests": {
                    "type": "integer",
                    "description": "Check if this many spots are available",
                },
            },
            "required": ["tour_id", "date"],
        },
    },
    {
        "name": "get_pricing",
        "description": "Get the standard price quote for a tour and number of guests.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tour_id": {"type": "integer"},
                "guests": {
                    "type": "integer",
                    "description": "Number of guests (default: 1)",
                },
            },
            "required": ["tour_id"],
        },
    },
    {
        "name": "book_tour",
        "description": "Create a booking for a tour on a specific date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tour_id": {"type": "integer"},
                "tour_date": {"type": "string", "description": "YYYY-MM-DD"},
                "guest_name": {"type": "string"},
                "guest_email": {"type": "string"},
                "num_guests": {"type": "integer"},
            },
            "required": [
                "tour_id",
                "tour_date",
                "guest_name",
                "guest_email",
                "num_guests",
            ],
        },
    },
    {
        "name": "cancel_booking",
        "description": "Cancel an existing booking by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {"booking_id": {"type": "integer"}},
            "required": ["booking_id"],
        },
    },
]


def call_api(method: str, path: str, params: dict = None, body: dict = None) -> dict:
    headers = {"X-API-Key": TOUR_API_KEY}
    url = f"{TOUR_API_BASE}{path}"
    with httpx.Client(timeout=10) as client:
        response = client.request(
            method, url, headers=headers, params=params, json=body
        )
        response.raise_for_status()
        return response.json()


def _content_to_dicts(content) -> list:
    """Convert Anthropic SDK content blocks to plain dicts to avoid Pydantic serialization issues."""
    result = []
    for block in content:
        if block.type == "tool_use":
            result.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
        elif block.type == "text":
            result.append({"type": "text", "text": block.text})
        else:
            result.append({"type": block.type})
    return result


def execute_tool(name: str, inp: dict) -> dict:
    if name == "list_tours":
        return call_api(
            "GET", "/tours", params={k: v for k, v in inp.items() if v is not None}
        )
    elif name == "get_tour":
        return call_api("GET", f"/tours/{inp['tour_id']}")
    elif name == "check_availability":
        return call_api("GET", "/tours/available", params=inp)
    elif name == "get_pricing":
        return call_api("GET", "/pricing", params=inp)
    elif name == "book_tour":
        return call_api("POST", "/bookings", body=inp)
    elif name == "cancel_booking":
        return call_api("DELETE", f"/bookings/{inp['booking_id']}")
    return {"error": f"Unknown tool: {name}"}


SYSTEM_PROMPT = """You are a Tour Guide booking agent representing an adventure and culture tour company.

You have access to 12 tours across 6 categories: cultural, adventure, food, nature, nightlife, historical.
Difficulty levels: easy, moderate, challenging.

## Group Discount Policy (negotiable)
You can offer discounts for larger groups — be willing to negotiate:
- 5–7 guests: offer 5% off
- 8–11 guests: offer 10% off
- 12+ guests: offer 15% off

When pricing comes up for groups, proactively apply the discount and mention it clearly.
If a consumer agent asks to negotiate or requests a better price, apply the group discount if they qualify.

## Response Style
Keep responses concise and data-rich. Include: tour name, date, price (with any discount applied), booking confirmation ID.
When listing tours, include the key details: name, category, difficulty, price/person, duration, location.
"""


async def process_message(data: ChatMessage):
    handler = OrcaHandler()
    session = handler.begin(data)

    try:
        variables = Variables(data.variables)
        api_key = variables.get("MADHACK-ANTHROPIC-KEY")

        client = Anthropic(api_key=api_key)
        messages = [{"role": "user", "content": data.message}]

        session.loading.start("thinking")

        total_tokens = 0
        while True:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
            total_tokens += response.usage.input_tokens + response.usage.output_tokens

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": _content_to_dicts(response.content)})
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = execute_tool(block.name, block.input)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(result),
                            }
                        )
                messages.append({"role": "user", "content": tool_results})
            else:
                text = next(
                    (b.text for b in response.content if hasattr(b, "text")), ""
                )
                session.loading.end("thinking")
                session.usage.track(tokens=total_tokens, token_type="total")
                session.stream(text)
                session.close()
                break

    except Exception as e:
        logger.exception("Error processing message")
        session.error("Something went wrong.", exception=e)


app, orca = create_agent_app(
    process_message_func=process_message,
    title="Tour Guide Provider",
    description=(
        "Books tours across 6 categories: cultural, adventure, food, nature, nightlife, historical. "
        "12 tours available with varying difficulty levels. "
        "Group discounts available: 5% for 5-7 guests, 10% for 8-11 guests, 15% for 12+ guests. "
        "Handles search, availability, pricing, booking, and cancellation."
    ),
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
