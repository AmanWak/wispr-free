"""
Wispr Free — Hold-to-Dictate
A free, open-source, local speech-to-text tool for macOS.
Hold a trigger key, speak, release — text is transcribed and pasted instantly.

Usage:
    python3 wispr_free.py                         # defaults (tiny model, Right Option key)
    python3 wispr_free.py --model base             # use a larger Whisper model
    python3 wispr_free.py --trigger right_ctrl      # change trigger key
    python3 wispr_free.py --language ja             # transcribe Japanese
    python3 wispr_free.py --no-paste               # clipboard only, no auto-paste

See README.md for full documentation.
"""

import argparse
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
from pynput import keyboard


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit these or override via command-line flags
# ─────────────────────────────────────────────────────────────────────────────

# Available Whisper models (see README.md for full comparison):
#   tiny   — 39M params,  ~1 GB RAM, fastest,  good accuracy
#   base   — 74M params,  ~1 GB RAM, fast,     better accuracy
#   small  — 244M params, ~2 GB RAM, moderate, great accuracy
#   medium — 769M params, ~5 GB RAM, slower,   excellent accuracy
#   large  — 1550M params,~10 GB RAM, slowest, best accuracy
DEFAULT_MODEL = "tiny"

# Audio sample rate (Whisper expects 16 kHz)
SAMPLE_RATE = 16000

# Trigger key mapping — maps CLI name → pynput Key object
TRIGGER_KEYS = {
    "right_option": keyboard.Key.alt_r,
    "left_option":  keyboard.Key.alt_l,
    "right_cmd":    keyboard.Key.cmd_r,
    "left_ctrl":    keyboard.Key.ctrl_l,
    "right_ctrl":   keyboard.Key.ctrl_r,
    "caps_lock":    keyboard.Key.caps_lock,
    "f13":          keyboard.KeyCode.from_vk(105),
    "f14":          keyboard.KeyCode.from_vk(107),
    "f15":          keyboard.KeyCode.from_vk(113),
    "f16":          keyboard.KeyCode.from_vk(106),
    "f17":          keyboard.KeyCode.from_vk(64),
    "f18":          keyboard.KeyCode.from_vk(79),
    "f19":          keyboard.KeyCode.from_vk(80),
    "f20":          keyboard.KeyCode.from_vk(90),
}

DEFAULT_TRIGGER = "right_option"

# Default transcription language (ISO 639-1 code, e.g. "en", "es", "ja", "de")
DEFAULT_LANGUAGE = "en"


# ─────────────────────────────────────────────────────────────────────────────
# Recorder — streams mic audio while the trigger key is held
# ─────────────────────────────────────────────────────────────────────────────

class Recorder:
    """Manages microphone streaming via sounddevice."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._is_recording = False

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"  ⚠️  Audio: {status}", flush=True)
        self._frames.append(indata.copy())

    def start(self):
        """Start recording from the default microphone."""
        if self._is_recording:
            return
        self._is_recording = True
        self._frames = []
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        print("🎤 Recording...", flush=True)

    def stop(self) -> np.ndarray | None:
        """Stop recording and return the captured audio as a numpy array."""
        if not self._is_recording:
            return None
        self._is_recording = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._frames:
            print("  ⚠️  No audio captured.", flush=True)
            return None

        return np.concatenate(self._frames, axis=0)

    @property
    def is_recording(self) -> bool:
        return self._is_recording


# ─────────────────────────────────────────────────────────────────────────────
# Transcriber — wraps OpenAI Whisper for local speech-to-text
# ─────────────────────────────────────────────────────────────────────────────

class Transcriber:
    """Loads a Whisper model and transcribes audio arrays to text."""

    def __init__(self, model_name: str = DEFAULT_MODEL, language: str = DEFAULT_LANGUAGE):
        self.model_name = model_name
        self.language = language
        self._model = None

    def load(self):
        """Download (if needed) and load the Whisper model into memory."""
        print(f"⏳ Loading Whisper '{self.model_name}' model...", flush=True)
        self._model = whisper.load_model(self.model_name)
        print("✅ Model loaded!", flush=True)

    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe a numpy audio array to text. Returns stripped string."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        print("⚙️  Transcribing...", flush=True)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_data, SAMPLE_RATE)
            tmp_path = tmp.name

        try:
            result = self._model.transcribe(tmp_path, language=self.language)
            return result["text"].strip()
        finally:
            os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# OutputHandler — pastes text via clipboard + simulated ⌘V
# ─────────────────────────────────────────────────────────────────────────────

class OutputHandler:
    """Copies transcribed text to clipboard and optionally pastes it."""

    def __init__(self, auto_paste: bool = True):
        self.auto_paste = auto_paste
        self._kb = keyboard.Controller()

    def deliver(self, text: str):
        """Copy text to clipboard and paste into the focused text field."""
        # Copy to clipboard
        subprocess.run("pbcopy", text=True, input=text, check=True)

        if not self.auto_paste:
            print("📋 Copied to clipboard!", flush=True)
            return

        # Small delay so clipboard settles and trigger key is fully released
        time.sleep(0.05)

        # Simulate ⌘V
        self._kb.press(keyboard.Key.cmd)
        self._kb.press("v")
        self._kb.release("v")
        self._kb.release(keyboard.Key.cmd)

        print("📋 Pasted!", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# App — wires everything together
# ─────────────────────────────────────────────────────────────────────────────

class WisprFree:
    """Main application: key listener → recorder → transcriber → output."""

    def __init__(self, config: argparse.Namespace):
        self.trigger_key = TRIGGER_KEYS.get(config.trigger, keyboard.Key.alt_r)
        self.recorder = Recorder(sample_rate=SAMPLE_RATE)
        self.transcriber = Transcriber(model_name=config.model, language=config.language)
        self.output = OutputHandler(auto_paste=not config.no_paste)
        self._work_queue: queue.Queue = queue.Queue()

    # ── Key callbacks ────────────────────────────────────────────────────

    def _on_press(self, key):
        if key == self.trigger_key:
            self.recorder.start()

    def _on_release(self, key):
        if key == self.trigger_key:
            audio = self.recorder.stop()
            if audio is not None:
                self._work_queue.put(audio)

    # ── Worker thread ────────────────────────────────────────────────────

    def _worker(self):
        """Background thread: dequeue audio → transcribe → paste."""
        while True:
            audio_data = self._work_queue.get()
            if audio_data is None:
                break  # Shutdown signal

            text = self.transcriber.transcribe(audio_data)
            if not text:
                print("  ⚠️  No speech detected.", flush=True)
                continue

            print(f'✅ "{text}"', flush=True)
            try:
                self.output.deliver(text)
            except Exception as e:
                print(f"  ⚠️  Paste failed ({e}), text is on your clipboard.", flush=True)

    # ── Run ──────────────────────────────────────────────────────────────

    def run(self):
        """Start the app: load model, spawn threads, listen for keys."""
        trigger_name = next(
            (k for k, v in TRIGGER_KEYS.items() if v == self.trigger_key),
            "unknown",
        )

        print()
        print("🚀 Wispr Free — Hold-to-Dictate")
        print("=" * 50)
        print(f"  Trigger key  : {trigger_name.replace('_', ' ').title()}")
        print(f"  Whisper model: {self.transcriber.model_name}")
        print(f"  Language     : {self.transcriber.language}")
        print(f"  Auto-paste   : {'on' if self.output.auto_paste else 'off (clipboard only)'}")
        print()
        print("  Hold trigger key → speak → release to transcribe & paste")
        print("  Press Ctrl+C to quit")
        print("=" * 50)
        print()

        # Permission reminder
        print("📌 macOS permissions required (System Settings → Privacy & Security):")
        print("     • Input Monitoring  → add your terminal app")
        print("     • Accessibility     → add your terminal app")
        print("     • Microphone        → prompted automatically")
        print()

        # Load Whisper model
        self.transcriber.load()
        print()

        # Start background worker
        worker_thread = threading.Thread(target=self._worker, daemon=True)
        worker_thread.start()

        # Start key listener (blocks main thread)
        print(f"👂 Listening for {trigger_name.replace('_', ' ').title()} key...\n")
        with keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        ) as listener:
            try:
                listener.join()
            except KeyboardInterrupt:
                print("\n👋 Shutting down...")
                self._work_queue.put(None)
                worker_thread.join(timeout=5)


# ─────────────────────────────────────────────────────────────────────────────
# CLI — argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="wispr_free",
        description="Wispr Free — free, local, hold-to-dictate speech-to-text for macOS.",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        choices=["tiny", "base", "small", "medium", "large"],
        help=f"Whisper model size (default: {DEFAULT_MODEL}). "
             "Larger = more accurate but slower and more RAM.",
    )
    parser.add_argument(
        "--trigger", "-t",
        default=DEFAULT_TRIGGER,
        choices=list(TRIGGER_KEYS.keys()),
        help=f"Trigger key to hold while speaking (default: {DEFAULT_TRIGGER}).",
    )
    parser.add_argument(
        "--language", "-l",
        default=DEFAULT_LANGUAGE,
        help=f"Transcription language as ISO 639-1 code (default: {DEFAULT_LANGUAGE}). "
             "Examples: en, es, fr, de, ja, zh, ko, hi.",
    )
    parser.add_argument(
        "--no-paste",
        action="store_true",
        default=False,
        help="Disable auto-paste. Text will only be copied to your clipboard.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    config = parse_args()
    app = WisprFree(config)
    app.run()


if __name__ == "__main__":
    main()
