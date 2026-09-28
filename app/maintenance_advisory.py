"""Groq maintenance advisory from the dashboard's displayed results."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

SYSTEM_PROMPT = """You are an aero-piston engine maintenance assistant.
Write a concise maintenance advisory using the supplied Edge ML diagnosis and
engine risk/RUL results. Explain the suspected fault, relate it to the confidence,
anomaly score, health state, subsystem risks and remaining life, then recommend
relevant inspection priorities. Treat model predictions as suspected faults.
Do not invent readings, limits, remaining-life estimates or manufacturer procedures.
If the results disagree or evidence is missing, explain that briefly. Recommend
confirming defects against approved engine maintenance data before replacement.
Return only the final advisory in plain text, with short paragraphs."""


def env_value(name: str, default: str = "") -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    root = Path(__file__).resolve().parents[1]
    for path in (root / ".env", root.parent / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if entry.startswith("export "):
                entry = entry[7:].strip()
            key, separator, raw = entry.partition("=")
            if separator and key.strip() == name:
                return raw.strip().strip("\"'")
    return default


def draft_with_groq(results: dict[str, Any], *, post=None) -> dict[str, str]:
    key = env_value("GROQ_API_KEY")
    if not key:
        raise ValueError("Set GROQ_API_KEY in the project .env file.")
    response = (post or requests.post)(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": env_value("GROQ_MODEL", "openai/gpt-oss-20b"),
            "temperature": 0.2,
            "max_completion_tokens": 1800,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Generate a maintenance advisory from these dashboard results:\n" + json.dumps(results, allow_nan=False)},
            ],
        },
        timeout=30,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Groq returned an empty maintenance advisory.")
    return {"text": content.strip()}
