# Basic Agent Example

A simple conversational agent using pico-agent with tool definitions.

## Requirements

- Python 3.11+
- An LLM API key (OpenAI, Anthropic, or Google)

## Setup

```bash
pip install -r requirements.txt
```

## Configure

Set your LLM provider API key:

```bash
# For OpenAI
export OPENAI_API_KEY=your-key-here

# For Anthropic
export ANTHROPIC_API_KEY=your-key-here

# For Google
export GOOGLE_API_KEY=your-key-here
```

## Run

```bash
python -m app.main
```

This will start an interactive agent that can use the defined tools to answer questions.
