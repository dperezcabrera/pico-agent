from pico_agent import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate (e.g., '2 + 3 * 4')

    Returns:
        The result of the calculation.
    """
    try:
        result = eval(expression)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city to get weather for.

    Returns:
        A weather description for the city.
    """
    # Simulated weather data for demonstration
    weather_data = {
        "London": "Cloudy, 15°C",
        "New York": "Sunny, 22°C",
        "Tokyo": "Rainy, 18°C",
        "Paris": "Partly cloudy, 20°C",
    }
    return weather_data.get(city, f"Weather data not available for {city}")
