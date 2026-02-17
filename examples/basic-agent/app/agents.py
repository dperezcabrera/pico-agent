from pico_agent import agent


@agent(
    name="assistant",
    instructions="You are a helpful assistant. Use the available tools to answer questions. Be concise.",
    tools=["calculator", "get_weather"],
)
class Assistant:
    """A simple assistant agent with calculator and weather tools."""
    pass
