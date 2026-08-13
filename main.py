"""
main.py — Voice Notes Agent: GUI entry point.

All logic lives in the agent/ package:
    agent/config.py       — constants and shared state
    agent/session.py      — session persistence (last open file)
    agent/agent.py        — LLM agent worker thread
    agent/transcriber.py  — audio recording + Whisper transcription
    agent/summarizer.py   — on-demand document summarization
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk

from agent.agent import agent_worker
from agent.config import BASE_DIR, agent_state, text_queue
from agent.session import load_last_session, save_session
from agent.summarizer import summarize_file
from agent.transcriber import Transcriber


class NotesApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("🎙️ AI Notes")
        self.root.geometry("680x640")
        self.root.minsize(420, 440)
        self.root.configure(bg="#121212")

        self.transcriber = Transcriber(self)
        self._build_gui()

        # Start the agent worker on a background thread
        self.agent_thread = threading.Thread(
            target=agent_worker, args=(self,), daemon=True
        )
        self.agent_thread.start()

    # -----------------------------------------------------------------------
    # GUI LAYOUT
    # -----------------------------------------------------------------------

    def _build_gui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(6, weight=1)  # log console expands

        # ── Row 0: active file label ────────────────────────────────────────
        self.lbl_file = tk.Label(
            self.root, text="📄 No file open", fg="#888888", bg="#121212",
            font=("Consolas", 12, "italic"), anchor="w",
        )
        self.lbl_file.grid(row=0, column=0, sticky="ew", padx=15, pady=(10, 2))

        # ── Row 1: file management toolbar ──────────────────────────────────
        file_bar = tk.Frame(self.root, bg="#121212")
        file_bar.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 4))
        file_bar.columnconfigure(0, weight=1)
        file_bar.columnconfigure(1, weight=1)
        file_bar.columnconfigure(2, weight=1)

        self.btn_open = tk.Button(
            file_bar, text="📂  Open note",
            font=("Arial", 9, "bold"), bg="#1e3a5f", fg="#90caf9",
            relief=tk.FLAT, cursor="hand2", padx=10, pady=5,
            command=self._open_existing_note,
        )
        self.btn_open.grid(row=0, column=0, sticky="ew", padx=(0, 3))

        last = load_last_session()
        last_label = f"⏮  {os.path.basename(last)}" if last else "⏮  No previous session"
        self.btn_resume = tk.Button(
            file_bar, text=last_label,
            font=("Arial", 9, "bold"),
            bg="#1a3a2a" if last else "#1e1e1e",
            fg="#a5d6a7" if last else "#555555",
            relief=tk.FLAT,
            cursor="hand2" if last else "arrow",
            padx=10, pady=5,
            command=self._resume_last_session if last else (lambda: None),
            state=tk.NORMAL if last else tk.DISABLED,
        )
        self.btn_resume.grid(row=0, column=1, sticky="ew", padx=3)
        self._last_session_path = last

        self.btn_summarize = tk.Button(
            file_bar, text="📋  Summarize",
            font=("Arial", 9, "bold"), bg="#3a2a1a", fg="#ffcc80",
            relief=tk.FLAT, cursor="hand2", padx=10, pady=5,
            command=self._request_summary,
            state=tk.DISABLED,  # enabled once a file is open
        )
        self.btn_summarize.grid(row=0, column=2, sticky="ew", padx=(3, 0))

        # ── Row 2: summary panel (hidden until a summary is available) ───────
        self.summary_frame = tk.Frame(self.root, bg="#1a1a2e", bd=0)
        self.summary_frame.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 4))
        self.summary_frame.columnconfigure(0, weight=1)
        self.summary_frame.grid_remove()  # hidden by default

        summary_header = tk.Frame(self.summary_frame, bg="#1a1a2e")
        summary_header.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 2))
        summary_header.columnconfigure(0, weight=1)

        tk.Label(
            summary_header, text="📋 Document Overview",
            fg="#ffcc80", bg="#1a1a2e", font=("Arial", 9, "bold"), anchor="w",
        ).grid(row=0, column=0, sticky="w")

        tk.Button(
            summary_header, text="✕",
            font=("Arial", 8), bg="#1a1a2e", fg="#666666",
            relief=tk.FLAT, cursor="hand2", padx=4,
            command=self._hide_summary,
        ).grid(row=0, column=1, sticky="e")

        self.txt_summary = tk.Text(
            self.summary_frame, wrap=tk.WORD, height=7,
            bg="#1a1a2e", fg="#e0c97f", font=("Consolas", 9),
            relief=tk.FLAT, cursor="arrow", state=tk.DISABLED,
            insertbackground="#1a1a2e",
        )
        self.txt_summary.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))

        # ── Row 3: microphone selector ──────────────────────────────────────
        mic_frame = tk.Frame(self.root, bg="#121212")
        mic_frame.grid(row=3, column=0, sticky="ew", padx=15, pady=2)
        mic_frame.columnconfigure(1, weight=1)

        tk.Label(
            mic_frame, text="🎤 Microphone:", fg="#aaaaaa", bg="#121212",
            font=("Arial", 9),
        ).grid(row=0, column=0, sticky="w")

        self.mic_map = self.transcriber.get_microphones()
        self.combo_mic = ttk.Combobox(
            mic_frame, values=list(self.mic_map.keys()), state="readonly",
        )
        self.combo_mic.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        if self.mic_map:
            self.combo_mic.current(0)

        # ── Row 4: record / stop button ─────────────────────────────────────
        self.btn_rec = tk.Button(
            self.root, text="🔴  START RECORDING",
            font=("Arial", 12, "bold"), bg="#b71c1c", fg="white",
            command=self._toggle_recording, height=2,
            relief=tk.FLAT, cursor="hand2",
        )
        self.btn_rec.grid(row=4, column=0, sticky="ew", padx=15, pady=10)

        # ── Row 5: status label ─────────────────────────────────────────────
        self.lbl_status = tk.Label(
            self.root, text="Ready.", fg="#4caf50", bg="#121212",
            font=("Arial", 11, "bold"),
        )
        self.lbl_status.grid(row=5, column=0, pady=(0, 4))

        # ── Row 6: scrollable log console ───────────────────────────────────
        self.txt_log = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD,
            bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 9),
            insertbackground="white",
        )
        self.txt_log.grid(row=6, column=0, sticky="nsew", padx=15, pady=(0, 15))

    # -----------------------------------------------------------------------
    # FILE MANAGEMENT
    # -----------------------------------------------------------------------

    def _set_active_file(self, path: str) -> None:
        """Set the active note file, update UI and persist the session."""
        agent_state["current_file"] = path
        agent_state["file_summary"] = None  # clear any stale summary
        save_session(path)
        short = os.path.basename(path)
        self.lbl_file.config(text=f"📄 {short}", fg="#ce93d8")
        self.btn_summarize.config(state=tk.NORMAL)
        self._hide_summary()
        self.log(f"📂 Active file set: {path}")
        self.set_status(f"✅ Continuing: {short}", "#4caf50")

    def _open_existing_note(self) -> None:
        path = filedialog.askopenfilename(
            title="Open note",
            initialdir=BASE_DIR,
            filetypes=[("Markdown files", "*.md"), ("All files", "*.*")],
        )
        if path:
            self._set_active_file(path)

    def _resume_last_session(self) -> None:
        if self._last_session_path and os.path.isfile(self._last_session_path):
            self._set_active_file(self._last_session_path)
        else:
            self.set_status("⚠️ Previous session file no longer exists.", "#ff9800")

    # -----------------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------------

    def _request_summary(self) -> None:
        """Trigger summarization on a background thread to keep the GUI responsive."""
        path = agent_state.get("current_file")
        if not path or not os.path.isfile(path):
            self.set_status("⚠️ No file open to summarize.", "#ff9800")
            return

        self.btn_summarize.config(state=tk.DISABLED, text="⏳  Summarizing…")
        self.set_status("🤖 Generating summary…", "#bb86fc")

        threading.Thread(target=self._run_summary, args=(path,), daemon=True).start()

    def _run_summary(self, path: str) -> None:
        """Background thread: call the LLM, then update the UI."""
        try:
            summary = summarize_file(path)
            agent_state["file_summary"] = summary  # store for agent_worker to use once
            self.root.after(0, lambda: self._show_summary(summary))
            self.set_status("✅ Summary ready.", "#4caf50")
        except Exception as e:
            self.log(f"❌ Summary error: {e}")
            self.set_status("❌ Summary failed.", "#f44336")
        finally:
            self.root.after(0, lambda: self.btn_summarize.config(
                state=tk.NORMAL, text="📋  Summarize"
            ))

    def _show_summary(self, text: str) -> None:
        self.txt_summary.config(state=tk.NORMAL)
        self.txt_summary.delete("1.0", tk.END)
        self.txt_summary.insert(tk.END, text)
        self.txt_summary.config(state=tk.DISABLED)
        self.summary_frame.grid()  # make the panel visible

    def _hide_summary(self) -> None:
        self.summary_frame.grid_remove()

    # -----------------------------------------------------------------------
    # RECORDING
    # -----------------------------------------------------------------------

    def _toggle_recording(self) -> None:
        if not self.transcriber.recording:
            selected = self.combo_mic.get()
            if not selected:
                self.set_status("⚠️ Please select a microphone first.", "#ff9800")
                return
            device_id = self.mic_map[selected]
            self.btn_rec.config(text="⏹️  STOP & TRANSCRIBE", bg="#e65100")
            self.set_status("🎙️ Recording... speak now!", "#ffeb3b")
            self.transcriber.start_recording(device_id)
        else:
            self.btn_rec.config(text="🔴  START RECORDING", bg="#b71c1c")
            self.set_status("⏳ Transcribing with Whisper...", "#2196f3")
            self.transcriber.stop_recording()

    # -----------------------------------------------------------------------
    # THREAD-SAFE GUI HELPERS  (interface consumed by agent_worker / Transcriber)
    # -----------------------------------------------------------------------

    def log(self, message: str) -> None:
        """Append a line to the log console from any thread."""
        def _write() -> None:
            self.txt_log.insert(tk.END, message + "\n")
            self.txt_log.see(tk.END)
        self.root.after(0, _write)

    def set_status(self, text: str, color: str) -> None:
        """Update the status label from any thread."""
        self.root.after(0, lambda: self.lbl_status.config(text=text, fg=color))


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = NotesApp(root)
    root.mainloop()
    text_queue.put(None)  # signal agent worker to exit cleanly
