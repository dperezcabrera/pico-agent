import asyncio

from pico_boot import init
from pico_ioc import configuration, YamlTreeSource
from pico_agent import AgentRunner


async def main():
    config = configuration(YamlTreeSource("config.yml"))

    container = init(
        modules=[
            "app.tools",
            "app.agents",
        ],
        config=config,
    )

    runner = await container.aget(AgentRunner)

    # Run the agent with a sample query
    queries = [
        "What is 25 * 17 + 83?",
        "What's the weather in Tokyo?",
    ]

    for query in queries:
        print(f"\nUser: {query}")
        response = await runner.run("assistant", query)
        print(f"Agent: {response}")

    await container.ashutdown()


if __name__ == "__main__":
    asyncio.run(main())
