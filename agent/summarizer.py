"""
summarizer.py — On-demand document summarization via llama-server.

summarize_file() reads the full content of the active .md file and asks
the LLM for a concise overview.  It streams the response and returns the
plain-text summary as a string.

This is intentionally a standalone function (not part of agent_worker)
so it can be triggered from the GUI on demand without interfering with
the transcription queue.
"""

import json
import requests

from agent.config import LLAMA_SERVER_URL, LLAMA_MODEL

SUMMARY_SYSTEM_PROMPT = """You are a helpful assistant that summarizes study notes.
Given the full content of a Markdown note file, write a concise overview (5–10 bullet points)
that captures the main topics, key concepts, and any open threads.
Reply in the same language as the document.
Use plain text only — no Markdown formatting in your response.
"""


def summarize_file(file_path: str) -> str:
    """
    Read `file_path` and return a plain-text summary from the LLM.
    Raises on connection errors or if the file cannot be read.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if not content.strip():
        return "(file is empty)"

    user_message = (
        f"Here are the full notes from the file '{file_path}':\n\n"
        f"{content}\n\n"
        "Please provide a concise overview of these notes."
    )

    raw_chunks: list[str] = []
    with requests.post(
        LLAMA_SERVER_URL,
        json={
            "model": LLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            "temperature": 0.4,   # lower temp for more factual summaries
            "max_tokens": 1024,
            "stream": True,
        },
        stream=True,
        timeout=30,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                data_str = line_str[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        raw_chunks.append(delta)
                except (json.JSONDecodeError, KeyError):
                    continue

    return "".join(raw_chunks).strip()
