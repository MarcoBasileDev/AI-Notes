import os
import json
import queue
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import requests
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
LLAMA_SERVER_URL = "http://localhost:12345/v1/chat/completions"
LLAMA_MODEL      = "gemma"   # llama-server ignores this field but the schema requires it
SAMPLE_RATE      = 16000     # 16 kHz — optimal for Whisper

# All files are written to the same directory as this script,
# regardless of the working directory from which Python is launched.
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
AUDIO_TMP = os.path.join(BASE_DIR, "rec_temp.wav")

# ---------------------------------------------------------------------------
# AGENT STATE
# ---------------------------------------------------------------------------
# Tracks which .md file is currently open so the agent can append in context.
agent_state = {
    "current_file": None,   # absolute path of the active .md file
    "last_section": None,   # last heading/topic written (for continuity)
}

# In-memory queue: Whisper produces → agent worker consumes
text_queue: queue.Queue[str] = queue.Queue()


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
# AGENT WORKER (runs on a dedicated background thread)
# ---------------------------------------------------------------------------

def agent_worker(app: "NotesApp"):
    """
    Infinite loop that reads transcriptions from the queue and processes them.
    Runs on a separate thread so it never blocks the GUI.
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
            # The 30s timeout applies only to receiving the first byte;
            # after that, tokens arrive continuously with no long silence.
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
                timeout=30,  # timeout for the first byte only
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

            # Robust cleanup: strip possible ```json ... ``` fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = json.loads(raw)

            action        = parsed.get("action", "append")
            filename      = parsed.get("filename", "notes.md")
            markdown      = parsed.get("markdown_content", "")

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
                # Always append to the currently open file
                with open(agent_state["current_file"], "a", encoding="utf-8") as f:
                    f.write(markdown + "\n\n")
                app.log(f"📝 Appended to: {agent_state['current_file']}")

            # Update GUI label (must be called from the main thread)
            app.root.after(0, lambda fn=agent_state["current_file"]: app.lbl_file.config(text=f"📄 {fn}"))
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


# ---------------------------------------------------------------------------
# GUI APPLICATION
# ---------------------------------------------------------------------------

class NotesApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🎙️ Voice Notes Agent")
        self.root.geometry("680x580")
        self.root.configure(bg="#121212")

        self.recording = False
        self.audio_data: list = []
        self.stream = None

        # Load Whisper on CPU — keeps VRAM free for the LLM
        print("Loading Whisper (small, CPU, int8)...")
        self.whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        print("Whisper ready.")

        self._build_gui()

        # Start the agent worker on a background thread
        self.agent_thread = threading.Thread(target=agent_worker, args=(self,), daemon=True)
        self.agent_thread.start()

    # -----------------------------------------------------------------------
    # GUI LAYOUT
    # -----------------------------------------------------------------------

    def _build_gui(self):
        top_frame = tk.Frame(self.root, bg="#121212")
        top_frame.pack(fill=tk.X, padx=15, pady=10)

        # Currently open file label
        self.lbl_file = tk.Label(
            top_frame, text="📄 No file open", fg="#888888", bg="#121212",
            font=("Consolas", 10, "italic"), anchor="w"
        )
        self.lbl_file.pack(side=tk.LEFT)

        # Microphone selector
        mic_frame = tk.Frame(self.root, bg="#121212")
        mic_frame.pack(fill=tk.X, padx=15, pady=2)

        tk.Label(mic_frame, text="🎤 Microphone:", fg="#aaaaaa", bg="#121212", font=("Arial", 9)).pack(side=tk.LEFT)
        self.mic_map = self._get_microphones()
        self.combo_mic = ttk.Combobox(mic_frame, values=list(self.mic_map.keys()), width=55, state="readonly")
        self.combo_mic.pack(side=tk.LEFT, padx=8)
        if self.mic_map:
            self.combo_mic.current(0)

        # Record / Stop button
        self.btn_rec = tk.Button(
            self.root, text="🔴  START RECORDING",
            font=("Arial", 12, "bold"), bg="#b71c1c", fg="white",
            command=self._toggle_recording, height=2, width=28,
            relief=tk.FLAT, cursor="hand2"
        )
        self.btn_rec.pack(pady=12)

        # Status label
        self.lbl_status = tk.Label(
            self.root, text="Ready.", fg="#4caf50", bg="#121212",
            font=("Arial", 11, "bold")
        )
        self.lbl_status.pack()

        # Scrollable log console
        self.txt_log = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, width=76, height=20,
            bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 9),
            insertbackground="white"
        )
        self.txt_log.pack(pady=10, padx=15)

    # -----------------------------------------------------------------------
    # MICROPHONE HELPERS
    # -----------------------------------------------------------------------

    def _get_microphones(self) -> dict:
        """Returns a dict of {display_name: device_id} for all input devices."""
        mic_map = {}
        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                mic_map[f"[{idx}] {dev['name']}"] = idx
        return mic_map

    # -----------------------------------------------------------------------
    # RECORDING
    # -----------------------------------------------------------------------

    def _toggle_recording(self):
        if not self.recording:
            selected = self.combo_mic.get()
            if not selected:
                self.set_status("⚠️ Please select a microphone first.", "#ff9800")
                return

            self.mic_id = self.mic_map[selected]
            self.recording = True
            self.audio_data = []
            self.btn_rec.config(text="⏹️  STOP & TRANSCRIBE", bg="#e65100")
            self.set_status("🎙️ Recording... speak now!", "#ffeb3b")
            threading.Thread(target=self._record_audio, daemon=True).start()
        else:
            self.recording = False
            self.btn_rec.config(text="🔴  START RECORDING", bg="#b71c1c")
            self.set_status("⏳ Transcribing with Whisper...", "#2196f3")
            if self.stream:
                self.stream.stop()
                self.stream.close()

    def _record_audio(self):
        """Captures microphone input and saves it to a temporary WAV file."""
        def callback(indata, frames, time, status):
            if self.recording:
                self.audio_data.append(indata.copy())

        self.stream = sd.InputStream(
            device=self.mic_id, samplerate=SAMPLE_RATE, channels=1, callback=callback
        )
        with self.stream:
            while self.recording:
                sd.sleep(100)

        if self.audio_data:
            audio_np = np.concatenate(self.audio_data, axis=0)
            wav.write(AUDIO_TMP, SAMPLE_RATE, audio_np)
            threading.Thread(target=self._transcribe, daemon=True).start()

    # -----------------------------------------------------------------------
    # WHISPER TRANSCRIPTION
    # -----------------------------------------------------------------------

    def _transcribe(self):
        """Runs Whisper on the recorded audio and pushes the transcript to the queue."""
        try:
            # language=None enables automatic language detection.
            # task="transcribe" keeps the original language (no translation).
            # This handles Italian, English, or mixed speech correctly.
            segments, info = self.whisper_model.transcribe(AUDIO_TMP, language=None, task="transcribe")
            text = " ".join(s.text for s in segments).strip()
            self.log(f"🌐 Detected language: {info.language} (confidence: {info.language_probability:.2f})")

            if os.path.exists(AUDIO_TMP):
                os.remove(AUDIO_TMP)

            if not text:
                self.set_status("⚠️ No speech detected.", "#ff9800")
                return

            self.log(f"🗣️ Whisper transcript:\n\"{text}\"\n{'─' * 50}")
            self.set_status("📬 Queued for agent...", "#ce93d8")

            # Hand off to the agent worker via the in-memory queue
            text_queue.put(text)

        except Exception as e:
            self.log(f"❌ Whisper error: {e}")
            self.set_status("❌ Transcription error", "#f44336")

    # -----------------------------------------------------------------------
    # THREAD-SAFE GUI HELPERS
    # -----------------------------------------------------------------------

    def log(self, message: str):
        """Appends a line to the log console from any thread."""
        def _write():
            self.txt_log.insert(tk.END, message + "\n")
            self.txt_log.see(tk.END)
        self.root.after(0, _write)

    def set_status(self, text: str, color: str):
        """Updates the status label from any thread."""
        self.root.after(0, lambda: self.lbl_status.config(text=text, fg=color))


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = NotesApp(root)
    root.mainloop()
    # Signal the agent worker thread to exit cleanly
    text_queue.put(None)
