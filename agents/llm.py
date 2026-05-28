"""LLM factory – swap between Anthropic and OpenAI via the LLM_PROVIDER env var."""

import os

from dotenv import load_dotenv

load_dotenv()


def get_llm(temperature: float = 0):
    """Return a chat model instance based on the LLM_PROVIDER environment variable.

    Supported values for LLM_PROVIDER:
        - "openai"    (default) – uses ChatOpenAI
        - "anthropic"           – uses ChatAnthropic

    The corresponding API key must also be set:
        OPENAI_API_KEY   for OpenAI
        ANTHROPIC_API_KEY for Anthropic
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        return ChatAnthropic(model=model, temperature=temperature)

    # Default: OpenAI
    from langchain_openai import ChatOpenAI

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=temperature)
