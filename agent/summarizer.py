"""
summarizer.py — On-demand document summarization via llama-server.
"""

import json
import re
import requests

from agent.config import LLAMA_SERVER_URL, LLAMA_MODEL

MAX_CONTENT_CHARS = 24_000


SUMMARY_SYSTEM_PROMPT = (
    "You are a helpful assistant that summarizes study notes.\n"
    "Given the content of a Markdown note file, write a concise overview "
    "(5-10 bullet points) that captures the main topics, key concepts, and "
    "any open threads. Reply in the same language as the document. "
    "Use plain text only, no Markdown formatting in your response."
)


def summarize_file(file_path: str, log=None) -> str:
    """
    Read `file_path` and return a plain-text summary from the LLM.
    `log` is an optional callable(str) to emit debug lines to the GUI console.
    Raises ValueError if the LLM returns an empty response.
    """
    def _log(msg: str) -> None:
        if log:
            log(msg)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if not content.strip():
        return "(file is empty)"

    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS]
        _log(f"⚠️  File truncated to {MAX_CONTENT_CHARS} chars for summarization.")

    _log(f"📤 Sending {len(content)} chars to LLM for summarization…")

    user_message = (
        "Here is the content of the note file:\n\n"
        f"{content}\n\n"
        "Please provide a concise overview of these notes."
    )

    raw_chunks: list[str] = []
    lines_seen = 0

    with requests.post(
        LLAMA_SERVER_URL,
        json={
            "model": LLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            "temperature": 0.4,
            "max_tokens": 4096,   # raised: gemma-4 uses thinking tokens before answering
            "stream": True,
        },
        stream=True,
        timeout=30,
    ) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            lines_seen += 1
            line_str = line.decode("utf-8")

            if not line_str.startswith("data: "):
                continue

            data_str = line_str[6:]
            if data_str.strip() == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # path 1: standard streaming delta
            text = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            # path 2: some llama.cpp builds use "text" instead of delta.content
            if not text:
                text = chunk.get("choices", [{}])[0].get("text", "")
            # path 3: top-level content (non-standard)
            if not text:
                text = chunk.get("content", "")

            if text:
                raw_chunks.append(text)

    _log(f"📥 Stream ended. SSE lines: {lines_seen}, chunks: {len(raw_chunks)}")

    full_text = "".join(raw_chunks)

    # gemma-4 wraps its reasoning in <think>...</think> before the real answer.
    # Strip the entire thinking block so only the actual summary remains.
    full_text = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL)
    result = full_text.strip()

    if result:
        _log(f"🔍 Preview: {result[:200].replace(chr(10), ' ')}…")
    else:
        _log(f"❌ Empty result after stripping think tags. Raw length: {len(full_text)}")
        _log(f"   Raw preview: {full_text[:300]!r}")
        raise ValueError(
            "LLM returned an empty summary after stripping thinking tokens. "
            "Check the log console for the raw output."
        )

    return result
