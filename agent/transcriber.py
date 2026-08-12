"""
transcriber.py — Audio recording and Whisper transcription.

Transcriber owns the microphone stream and the WhisperModel instance.
It is deliberately decoupled from the GUI: all UI feedback goes through
the same app interface used by agent_worker (log / set_status).
"""

import os
import threading

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
from faster_whisper import WhisperModel

from agent.config import SAMPLE_RATE, AUDIO_TMP, text_queue


class Transcriber:
    """
    Handles microphone capture and Whisper transcription.

    After transcription the resulting text is pushed onto `text_queue`
    for the agent worker to consume — the Transcriber never calls the
    LLM directly.
    """

    def __init__(self, app) -> None:
        """
        app: any object satisfying the log / set_status interface.
        """
        self.app = app
        self.recording = False
        self.audio_data: list = []
        self.stream = None

        app.log("Loading Whisper (small, CPU, int8)...")
        self.model = WhisperModel("small", device="cpu", compute_type="int8")
        app.log("Whisper ready.")

    # -----------------------------------------------------------------------
    # PUBLIC
    # -----------------------------------------------------------------------

    def get_microphones(self) -> dict[str, int]:
        """Return {display_name: device_id} for every input-capable device."""
        mic_map: dict[str, int] = {}
        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                mic_map[f"[{idx}] {dev['name']}"] = idx
        return mic_map

    def start_recording(self, device_id: int) -> None:
        """Begin capturing audio from the given device on a background thread."""
        self.recording = True
        self.audio_data = []
        threading.Thread(
            target=self._record_loop, args=(device_id,), daemon=True
        ).start()

    def stop_recording(self) -> None:
        """Stop capture; transcription starts automatically when audio is ready."""
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()

    # -----------------------------------------------------------------------
    # PRIVATE
    # -----------------------------------------------------------------------

    def _record_loop(self, device_id: int) -> None:
        """Capture loop — runs on a background thread."""
        def callback(indata, frames, time, status):
            if self.recording:
                self.audio_data.append(indata.copy())

        self.stream = sd.InputStream(
            device=device_id, samplerate=SAMPLE_RATE, channels=1, callback=callback
        )
        with self.stream:
            while self.recording:
                sd.sleep(100)

        if self.audio_data:
            audio_np = np.concatenate(self.audio_data, axis=0)
            wav.write(AUDIO_TMP, SAMPLE_RATE, audio_np)
            threading.Thread(target=self._transcribe, daemon=True).start()

    def _transcribe(self) -> None:
        """Run Whisper on the saved WAV and push the transcript to the queue."""
        try:
            # language=None → automatic detection; task="transcribe" keeps original language
            segments, info = self.model.transcribe(AUDIO_TMP, language=None, task="transcribe")
            text = " ".join(s.text for s in segments).strip()
            self.app.log(
                f"🌐 Detected language: {info.language} "
                f"(confidence: {info.language_probability:.2f})"
            )

            if os.path.exists(AUDIO_TMP):
                os.remove(AUDIO_TMP)

            if not text:
                self.app.set_status("⚠️ No speech detected.", "#ff9800")
                return

            self.app.log(f"🗣️ Whisper transcript:\n\"{text}\"\n{'─' * 50}")
            self.app.set_status("📬 Queued for agent...", "#ce93d8")
            text_queue.put(text)

        except Exception as e:
            self.app.log(f"❌ Whisper error: {e}")
            self.app.set_status("❌ Transcription error", "#f44336")
