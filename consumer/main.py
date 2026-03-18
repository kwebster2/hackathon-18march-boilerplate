import logging
import json
import anthropic
from orca import create_agent_app, ChatMessage, OrcaHandler, Variables, ChatHistoryHelper

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a fabulous, enthusiastic travel concierge with a vibrant personality. \
You LIVE for travel and helping people plan unforgettable experiences. \
You match the energy and vibe of whatever the user is into — if they want nightlife, you're hyped; \
if they want a chill nature retreat, you're zen; if they want culture, you're a passionate nerd about it.

You have access to provider agents that can search tours, check availability, get pricing, and make bookings. \
When the user wants something, figure out what they need and delegate to the right provider.

Keep responses concise but full of personality. Use your voice — be warm, be fun, be helpful. \
Mirror the user's energy and style. If someone is clearly excited about something, match that excitement. \
Add flavor and recommendations but don't be overbearing.

When presenting tour options, format them nicely but keep it conversational, not robotic."""


def build_chat_messages(history, user_message: str, provider_context: str) -> list[dict]:
    """Convert Orca chat history + new message into Anthropic-format messages."""
    messages = []

    if history:
        recent = history.get_last_n_messages(10)
        for msg in recent:
            if isinstance(msg, dict):
                role = "user" if msg.get("role") == "user" else "assistant"
                content = msg.get("content", "")
            else:
                role = "user" if msg.role == "user" else "assistant"
                content = msg.content
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})
    return messages


async def process_message(data: ChatMessage):
    handler = OrcaHandler()
    session = handler.begin(data)

    try:
        variables = Variables(data.variables)
        api_key = variables.get("MADHACK-ANTHROPIC-KEY")

        if not api_key:
            logger.error("MADHACK-ANTHROPIC-KEY not found in variables. Available: %s", list(data.variables.keys()) if data.variables else "None")
            session.stream("I'm missing my API key configuration. Please check that MADHACK-ANTHROPIC-KEY is set in Orca variables.")
            session.close()
            return

        client = anthropic.Anthropic(api_key=api_key)
        history = ChatHistoryHelper(data.chat_history)

        # Discover connected providers
        available = {agent.slug: agent for agent in session.available_agents}
        provider_list = ", ".join(
            f"{a.name} ({a.slug}): {a.description}" for a in available.values()
        ) or "No providers connected yet."

        provider_context = (
            f"\n\nAvailable provider agents you can delegate to:\n{provider_list}\n\n"
            "When you need data from a provider, respond with a JSON block like:\n"
            '{"delegate": "agent-slug", "query": "your question for the provider"}\n'
            "You can include multiple delegate blocks if needed. "
            "If you can answer directly without a provider, just respond normally."
        )

        system_prompt = SYSTEM_PROMPT + provider_context
        messages = build_chat_messages(history, data.message, provider_context)

        session.loading.start("thinking")

        # First LLM call — decide what to do
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
            temperature=0.9,
        )

        reply = response.content[0].text
        session.loading.end()

        # Check if the LLM wants to delegate to a provider
        if '{"delegate"' in reply:
            session.loading.start("search")

            # Extract delegation requests
            provider_responses = []
            for line in reply.split("\n"):
                line = line.strip()
                if line.startswith("{") and '"delegate"' in line:
                    try:
                        req = json.loads(line)
                        slug = req["delegate"]
                        query = req["query"]
                        if slug in available:
                            resp = session.ask_agent(slug, query)
                            provider_responses.append(
                                f"Response from {available[slug].name}:\n{resp}"
                            )
                    except (json.JSONDecodeError, KeyError):
                        continue

            session.loading.end("search")

            if provider_responses:
                # Second LLM call — synthesize provider data with personality
                provider_data = "\n\n".join(provider_responses)
                messages.append({"role": "assistant", "content": reply})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Here are the results from the providers:\n\n{provider_data}\n\n"
                        "Now synthesize this into a fabulous, personality-filled response for the user. "
                        "Don't mention 'providers' or 'agents' — just present the info naturally."
                    ),
                })

                synth = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    system=system_prompt,
                    messages=messages,
                    temperature=0.9,
                )
                reply = synth.content[0].text

        session.stream(reply)
        session.close()

    except Exception as e:
        logger.exception("Error processing message")
        session.error("Something went wrong.", exception=e)


app, orca = create_agent_app(
    process_message_func=process_message,
    title="Vibes Travel Assistant",
    description="Your personal travel concierge — matching your energy, finding your perfect experiences",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
