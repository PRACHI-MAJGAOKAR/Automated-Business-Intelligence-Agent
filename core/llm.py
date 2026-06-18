"""

Thin wrapper around ChatGoogleGenerativeAI (Gemini Flash).
All agents import get_llm() from here when model is instantiated once.

Structured JSON calls go through call_structured() which:
  1. Appends a "respond ONLY in JSON" instruction
  2. Strips markdown fences from the response
  3. Parses and returns the dict — raises ValueError on bad JSON
"""

from __future__ import annotations
import time
import random
import json
import re
from functools import lru_cache
from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI
from core.config import GEMINI_API_KEY, GEMINI_MODEL
from functools import lru_cache
#from langchain_openai import ChatOpenAI
import os
"""
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set.\n"
            "Set it before running."
        )

    return ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0.1,  # low temp → consistent structured output
        max_completion_tokens=2048,
    )"""

@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set.\n"
            "Export it before running:  export GEMINI_API_KEY=your_key_here"
        )
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.1,          # low temp → consistent structured output
        max_output_tokens=2048,
    )


def call_llm(prompt: str, retries: int = 4) -> str:
    """Plain text call with exponential backoff on 429."""
    llm = get_llm()
    for attempt in range(retries):
        try:
            resp = llm.invoke(prompt)
            return resp.content.strip()
        except Exception as exc:
            error_str = str(exc)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt == retries - 1:
                    raise
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limited. Waiting {wait:.1f}s before retry {attempt + 2}/{retries}...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded.")


def call_structured(prompt: str, retries: int = 4) -> dict[str, Any]:
    """Structured JSON call with exponential backoff on 429."""
    enforced_prompt = (
        prompt
        + "\n\nIMPORTANT: Respond ONLY with a valid JSON object. "
        "No explanation, no markdown, no code fences. Raw JSON only."
    )
    for attempt in range(retries):
        try:
            raw = call_llm(enforced_prompt, retries=1)
            raw = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip().strip("`").strip()
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM returned invalid JSON.\nRaw response:\n{raw}\nError: {exc}"
            )
        except Exception as exc:
            error_str = str(exc)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt == retries - 1:
                    raise
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limited. Waiting {wait:.1f}s before retry {attempt + 2}/{retries}...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded.")
