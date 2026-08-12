"""
agent.py — LLM agent worker.

agent_worker() runs on a background thread and processes transcriptions
from text_queue one at a time, calling the local LLM and writing Markdown
to disk.

The `app` parameter is any object that satisfies this interface:
    app.log(message: str) -> None
    app.set_status(text: str, color: str) -> None
    app.root.after(delay_ms: int, fn: callable) -> None
    app.lbl_file  (object with a .config(**kwargs) method)

Both NotesApp (GUI) and StubApp (test runner) satisfy this contract.
"""

import json
import os

import requests

from agent.config import (
    LLAMA_SERVER_URL,
    LLAMA_MODEL,
    BASE_DIR,
    agent_state,
    text_queue,
)
from agent.session import save_session

# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an agent that manages a Markdown note file.
You receive voice transcriptions (in any language) and MUST always reply with valid JSON only — no text before or after it.

Core rules:
1. If the user is DESCRIBING or EXPLAINING a concept (e.g. "a load balancer is...", "replication serves to..."),
   the action is "append" and you write the content as expanded, well-structured Markdown.

2. If the user is giving an EXPLICIT COMMAND (e.g. "create a table with columns X Y Z",
   "add a bullet list of...", "add a heading that says...", "write a paragraph about..."),
   execute the command exactly. The action is still "append".

3. If the user explicitly asks to start a new topic or new file
   (e.g. "new topic", "start a file about", "create a file for"),
   the action is "new_file".

4. NEVER repeat content already present in the file. Always continue from where it left off.

5. If the user asks for a definition of a term they don't know, add it as a highlighted blockquote:
   > **Term**: definition here.

JSON response schema:
{
  "action": "append" | "new_file",
  "filename": "filename_without_spaces.md",
  "markdown_content": "## Heading\\n\\nMarkdown text..."
}

The "filename" field must be a sensible .md filename derived from the topic.
If the action is "append" and a file is already open, always use that file.
"""


# ---------------------------------------------------------------------------
# AGENT WORKER
# ---------------------------------------------------------------------------

def agent_worker(app) -> None:
    """
    Infinite loop that reads transcriptions from the queue and processes them.
    Runs on a separate thread so it never blocks the GUI.
    Exits cleanly when it receives None from the queue.
    """
    while True:
        text = text_queue.get()  # blocks until an item is available
        if text is None:         # None is the shutdown signal
            break

        preview = f'"{text[:60]}..."' if len(text) > 60 else f'"{text}"'
        app.log(f"📥 Text queued → LLM: {preview}")
        app.set_status("🤖 Agent processing...", "#bb86fc")

        raw = ""
        try:
            # Build context: send the last 3000 chars of the current file so the
            # agent can write in continuity without repeating itself.
            file_context = ""
            if agent_state["current_file"] and os.path.exists(agent_state["current_file"]):
                with open(agent_state["current_file"], "r", encoding="utf-8") as f:
                    existing = f.read()
                    file_context = existing[-3000:] if len(existing) > 3000 else existing

            user_message = (
                f"Currently open file: {agent_state['current_file'] or 'none'}\n\n"
                f"Last lines of the file (context):\n"
                f"{file_context if file_context else '(file is empty or no file is open)'}\n\n"
                f"---\n"
                f"New voice transcription:\n\"{text}\"\n"
            )

            # Call llama-server using streaming to avoid read timeouts.
            # The 30s connect/first-byte timeout keeps the connection alive
            # for the full generation regardless of response length.
            raw_chunks: list[str] = []
            with requests.post(
                LLAMA_SERVER_URL,
                json={
                    "model": LLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_message},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2048,
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

            raw = "".join(raw_chunks).strip()

            # Strip possible ```json ... ``` fences from the response
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed   = json.loads(raw)
            action   = parsed.get("action", "append")
            filename = parsed.get("filename", "notes.md")
            markdown = parsed.get("markdown_content", "")

            if not markdown.strip():
                app.log("⚠️ LLM returned empty content — skipping.")
                app.set_status("⚠️ No content generated.", "#ff9800")
                continue

            # Sanitise filename: strip any path component, ensure .md extension
            filename = os.path.basename(filename)
            if not filename.endswith(".md"):
                filename += ".md"
            file_path = os.path.join(BASE_DIR, filename)

            # Write to disk
            if action == "new_file" or agent_state["current_file"] is None:
                agent_state["current_file"] = file_path
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(markdown + "\n\n")
                app.log(f"🆕 New file created: {file_path}")
            else:
                with open(agent_state["current_file"], "a", encoding="utf-8") as f:
                    f.write(markdown + "\n\n")
                app.log(f"📝 Appended to: {agent_state['current_file']}")

            # Notify the GUI from the main thread
            app.root.after(0, lambda fn=agent_state["current_file"]: app.lbl_file.config(
                text=f"📄 {os.path.basename(fn)}", fg="#ce93d8"
            ))
            save_session(agent_state["current_file"])
            app.log(f"\n--- WRITTEN CONTENT ---\n{markdown}\n{'─' * 50}")
            app.set_status("✅ Notes updated.", "#4caf50")

        except json.JSONDecodeError as e:
            app.log(f"❌ Invalid JSON from LLM: {e}\nRaw (first 200 chars): {raw[:200]}")
            app.set_status("❌ LLM response parse error", "#f44336")
        except requests.exceptions.ConnectionError:
            app.log(
                "❌ Cannot connect to llama-server on port 12345.\n"
                "Make sure llama-server.exe is running before starting this app."
            )
            app.set_status("❌ llama-server unreachable", "#f44336")
        except Exception as e:
            app.log(f"❌ Agent error: {e}")
            app.set_status("❌ Error", "#f44336")
        finally:
            text_queue.task_done()
