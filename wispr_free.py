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

    python3 wispr_free.py vocab add "Aman" "FAANG"  # add custom words
    python3 wispr_free.py vocab list                 # show custom words
    python3 wispr_free.py vocab remove "FAANG"       # remove a word
    python3 wispr_free.py vocab clear                # remove all words

See README.md for full documentation.
"""

import argparse
import json
import os
import pathlib
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Optional

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
DEFAULT_MODEL = "base"

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
# Configuration — paths and API settings
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_DIR = pathlib.Path.home() / ".wispr_free"
VOCAB_FILE = CONFIG_DIR / "custom_words.json"

# API providers for smart command detection
API_PROVIDERS = ["gemini", "purdue", "none"]
DEFAULT_API_PROVIDER = "none"

# Purdue GenAI Studio (OpenAI-compatible endpoint)
PURDUE_API_URL = "https://genai.rcac.purdue.edu/api/chat/completions"
PURDUE_DEFAULT_MODEL = "llama3.1:latest"

# Google Gemini REST API
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"

# Prompt used to classify transcribed text as a voice command
COMMAND_DETECTION_PROMPT = """You are a voice command classifier for a speech-to-text dictation tool.
Determine if the following transcribed text is a DELETION/UNDO command
(meaning the user wants to erase what they just typed).

Examples of deletion commands:
- "scratch that", "delete that", "undo that", "never mind"
- "remove that", "erase that", "take that back", "oops"

Examples that are NOT deletion commands:
- "scratch the surface of the problem"
- "delete the file from the server"
- "I never mind the cold"
- "go back to the office"

Respond with ONLY valid JSON, no other text:
{{"is_delete_command": true}} or {{"is_delete_command": false}}

Transcribed text: "{text}"
"""


# ─────────────────────────────────────────────────────────────────────────────
# Custom Vocabulary — biases Whisper toward user-specified words
# ─────────────────────────────────────────────────────────────────────────────

class CustomVocabulary:
    """Manages a persistent list of custom words/names/acronyms.

    Stores words in ~/.wispr_free/custom_words.json.
    Words are injected into Whisper's initial_prompt to bias transcription
    toward recognizing these terms.
    """

    def __init__(self, vocab_file: pathlib.Path = VOCAB_FILE):
        self._file = vocab_file
        self._words: list[str] = []
        self._load()

    def _load(self):
        """Load words from disk."""
        if self._file.exists():
            try:
                self._words = json.loads(self._file.read_text())
            except (json.JSONDecodeError, OSError):
                self._words = []
        else:
            self._words = []

    def _save(self):
        """Persist words to disk."""
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(json.dumps(self._words, indent=2))

    def add(self, words: list[str]) -> list[str]:
        """Add words. Returns list of newly added words (skips duplicates)."""
        added = []
        for w in words:
            w = w.strip()
            if w and w not in self._words:
                self._words.append(w)
                added.append(w)
        if added:
            self._save()
        return added

    def remove(self, words: list[str]) -> list[str]:
        """Remove words. Returns list of actually removed words."""
        removed = []
        for w in words:
            w = w.strip()
            if w in self._words:
                self._words.remove(w)
                removed.append(w)
        if removed:
            self._save()
        return removed

    def clear(self) -> int:
        """Remove all words. Returns count of removed words."""
        count = len(self._words)
        self._words = []
        self._save()
        return count

    def list_words(self) -> list[str]:
        """Return a copy of all custom words."""
        return list(self._words)

    def get_prompt(self) -> str:
        """Build a Whisper initial_prompt string from custom words.

        Whisper uses this to bias its decoder toward recognizing these tokens.
        """
        if not self._words:
            return ""
        return "Vocabulary: " + ", ".join(self._words) + "."


# ─────────────────────────────────────────────────────────────────────────────
# Command Detector — identifies voice commands like "scratch that"
# ─────────────────────────────────────────────────────────────────────────────

class CommandDetector:
    """Detects voice commands (e.g. 'scratch that') using local regex
    matching with optional LLM verification via Gemini or Purdue GenAI Studio.

    Flow:
      1. Fast local regex for obvious patterns → instant, no API call
      2. If API is configured, ambiguous text is sent to LLM for classification
      3. If no API, only exact local patterns are caught
    """

    # Regex patterns that are almost certainly deletion commands
    LOCAL_DELETE_PATTERNS = [
        r"^scratch\s+that\.?$",
        r"^delete\s+that\.?$",
        r"^undo\s+that\.?$",
        r"^undo\.?$",
        r"^never\s*mind\.?$",
        r"^remove\s+that\.?$",
        r"^erase\s+that\.?$",
        r"^take\s+that\s+back\.?$",
        r"^go\s+back\.?$",
        r"^backspace\.?$",
        r"^clear\s+that\.?$",
        r"^oops\.?$",
    ]

    def __init__(self, api_provider: str = "none", api_key: Optional[str] = None,
                 api_model: Optional[str] = None):
        self.api_provider = api_provider
        self.api_key = api_key
        self.api_model = api_model
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.LOCAL_DELETE_PATTERNS]

    def _local_detect(self, text: str) -> Optional[dict]:
        """Fast regex check for obvious deletion commands."""
        cleaned = text.strip()
        for pattern in self._compiled:
            if pattern.match(cleaned):
                return {"action": "delete", "method": "local"}
        return None

    def _call_gemini(self, text: str) -> Optional[dict]:
        """Classify text using Google Gemini API."""
        import urllib.request
        import urllib.error

        url = GEMINI_API_URL.format(model=self.api_model or GEMINI_DEFAULT_MODEL)
        url += f"?key={self.api_key}"

        payload = json.dumps({
            "contents": [{"parts": [{"text": COMMAND_DETECTION_PROMPT.format(text=text)}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 50},
        }).encode()

        req = urllib.request.Request(
            url, data=payload, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read())
                content = body["candidates"][0]["content"]["parts"][0]["text"]
                # Parse the JSON from LLM response
                cleaned = content.strip().strip("`").removeprefix("json").strip()
                result = json.loads(cleaned)
                if result.get("is_delete_command"):
                    return {"action": "delete", "method": "gemini"}
        except Exception as e:
            print(f"  \u26a0\ufe0f  Gemini API error: {e}", flush=True)
        return None

    def _call_purdue(self, text: str) -> Optional[dict]:
        """Classify text using Purdue GenAI Studio API."""
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": self.api_model or PURDUE_DEFAULT_MODEL,
            "messages": [{"role": "user", "content": COMMAND_DETECTION_PROMPT.format(text=text)}],
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            PURDUE_API_URL, data=payload, method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read())
                content = body["choices"][0]["message"]["content"]
                cleaned = content.strip().strip("`").removeprefix("json").strip()
                result = json.loads(cleaned)
                if result.get("is_delete_command"):
                    return {"action": "delete", "method": "purdue"}
        except Exception as e:
            print(f"  \u26a0\ufe0f  Purdue API error: {e}", flush=True)
        return None

    def detect(self, text: str) -> Optional[dict]:
        """Check if transcribed text is a voice command.

        Returns a dict like {"action": "delete", "method": "local"|"gemini"|"purdue"}
        or None if the text is normal dictation.
        """
        # 1. Fast local match for obvious patterns
        local = self._local_detect(text)
        if local:
            return local

        # 2. If API is configured, ask LLM to classify ambiguous text
        if self.api_provider != "none" and self.api_key:
            if self.api_provider == "gemini":
                return self._call_gemini(text)
            elif self.api_provider == "purdue":
                return self._call_purdue(text)

        return None


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

    def transcribe(self, audio_data: np.ndarray, initial_prompt: str = "") -> str:
        """Transcribe a numpy audio array to text.

        Args:
            audio_data: Audio as a numpy array.
            initial_prompt: Optional prompt to bias Whisper toward specific words.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        print("⚙️  Transcribing...", flush=True)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_data, SAMPLE_RATE)
            tmp_path = tmp.name

        try:
            kwargs = {"language": self.language}
            if initial_prompt:
                kwargs["initial_prompt"] = initial_prompt
            result = self._model.transcribe(tmp_path, **kwargs)
            return result["text"].strip()
        finally:
            os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# OutputHandler — pastes text via clipboard + simulated ⌘V
# ─────────────────────────────────────────────────────────────────────────────

class OutputHandler:
    """Copies transcribed text to clipboard and optionally pastes it.

    Tracks the last pasted text so it can be deleted on voice command.
    """

    def __init__(self, auto_paste: bool = True):
        self.auto_paste = auto_paste
        self._kb = keyboard.Controller()
        self._last_text: str = ""

    def deliver(self, text: str):
        """Copy text to clipboard and paste into the focused text field."""
        # Copy to clipboard
        subprocess.run("pbcopy", text=True, input=text, check=True)

        if not self.auto_paste:
            print("📋 Copied to clipboard!", flush=True)
            self._last_text = text
            return

        # Small delay so clipboard settles and trigger key is fully released
        time.sleep(0.05)

        # Simulate ⌘V
        self._kb.press(keyboard.Key.cmd)
        self._kb.press("v")
        self._kb.release("v")
        self._kb.release(keyboard.Key.cmd)

        self._last_text = text
        print("📋 Pasted!", flush=True)

    def delete_last(self):
        """Delete the last pasted text by simulating ⌘Z (undo)."""
        if not self._last_text:
            print("  ⚠️  Nothing to delete.", flush=True)
            return

        if not self.auto_paste:
            print("  ⚠️  Auto-paste is off — nothing was typed to delete.", flush=True)
            return

        time.sleep(0.05)

        # Undo the last paste with ⌘Z
        self._kb.press(keyboard.Key.cmd)
        self._kb.press("z")
        self._kb.release("z")
        self._kb.release(keyboard.Key.cmd)

        deleted_text = self._last_text
        self._last_text = ""
        print(f'🗑️  Deleted: "{deleted_text}"', flush=True)

    @property
    def has_last_text(self) -> bool:
        return bool(self._last_text)


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
        self.vocabulary = CustomVocabulary()
        self.command_detector = CommandDetector(
            api_provider=config.api_provider,
            api_key=config.api_key,
            api_model=getattr(config, "api_model", None),
        )
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
        """Background thread: dequeue audio → transcribe → detect commands → paste/delete."""
        while True:
            audio_data = self._work_queue.get()
            if audio_data is None:
                break  # Shutdown signal

            # Transcribe with custom vocabulary bias
            vocab_prompt = self.vocabulary.get_prompt()
            text = self.transcriber.transcribe(audio_data, initial_prompt=vocab_prompt)
            if not text:
                print("  ⚠️  No speech detected.", flush=True)
                continue

            print(f'✅ "{text}"', flush=True)

            # Check for voice commands (e.g. "scratch that")
            command = self.command_detector.detect(text)
            if command and command["action"] == "delete":
                print(f'🔍 Command detected ({command["method"]}): delete last', flush=True)
                try:
                    self.output.delete_last()
                except Exception as e:
                    print(f"  ⚠️  Delete failed: {e}", flush=True)
                continue

            # Normal text — paste it
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

        # Count custom vocab words
        vocab_words = self.vocabulary.list_words()
        vocab_display = f"{len(vocab_words)} words" if vocab_words else "none"

        print()
        print("🚀 Wispr Free — Hold-to-Dictate")
        print("=" * 50)
        print(f"  Trigger key  : {trigger_name.replace('_', ' ').title()}")
        print(f"  Whisper model: {self.transcriber.model_name}")
        print(f"  Language     : {self.transcriber.language}")
        print(f"  Auto-paste   : {'on' if self.output.auto_paste else 'off (clipboard only)'}")
        print(f"  Custom vocab : {vocab_display}")
        print(f"  Command API  : {self.command_detector.api_provider}")
        print()
        print("  Hold trigger key → speak → release to transcribe & paste")
        print('  Say "scratch that" to delete the last transcription')
        print("  Press Ctrl+C to quit")
        print("=" * 50)
        print()

        # Show custom words if any
        if vocab_words:
            print(f"📖 Custom vocabulary: {', '.join(vocab_words)}")
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
    subparsers = parser.add_subparsers(dest="command")

    # ── vocab subcommand ─────────────────────────────────────────────
    vocab_parser = subparsers.add_parser(
        "vocab", help="Manage custom vocabulary words for better transcription.",
    )
    vocab_sub = vocab_parser.add_subparsers(dest="vocab_action")

    add_p = vocab_sub.add_parser("add", help="Add custom words.")
    add_p.add_argument("words", nargs="+", help="Words to add (names, acronyms, etc.).")

    rm_p = vocab_sub.add_parser("remove", help="Remove custom words.")
    rm_p.add_argument("words", nargs="+", help="Words to remove.")

    vocab_sub.add_parser("list", help="List all custom words.")
    vocab_sub.add_parser("clear", help="Remove all custom words.")

    # ── dictation flags (when no subcommand) ─────────────────────────
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        choices=["tiny", "base", "small", "medium", "large"],
        help=f"Whisper model size (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--trigger", "-t",
        default=DEFAULT_TRIGGER,
        choices=list(TRIGGER_KEYS.keys()),
        help=f"Trigger key (default: {DEFAULT_TRIGGER}).",
    )
    parser.add_argument(
        "--language", "-l",
        default=DEFAULT_LANGUAGE,
        help=f"ISO 639-1 language code (default: {DEFAULT_LANGUAGE}).",
    )
    parser.add_argument(
        "--no-paste",
        action="store_true",
        default=False,
        help="Disable auto-paste (clipboard only).",
    )
    parser.add_argument(
        "--api-provider",
        default=DEFAULT_API_PROVIDER,
        choices=API_PROVIDERS,
        help="LLM API for smart command detection (default: none). "
             "Set to 'gemini' or 'purdue' to enable AI-powered detection "
             "of ambiguous commands like 'scratch that'. Without an API, "
             "only exact phrase matching is used.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the chosen provider. Can also be set via "
             "WISPR_GEMINI_API_KEY or WISPR_PURDUE_API_KEY environment variables.",
    )
    parser.add_argument(
        "--api-model",
        default=None,
        help=f"Override the default LLM model for command detection. "
             f"Purdue default: {PURDUE_DEFAULT_MODEL}, "
             f"Gemini default: {GEMINI_DEFAULT_MODEL}.",
    )

    args = parser.parse_args()

    # Resolve API key from environment if not passed via CLI
    if not hasattr(args, "api_key") or args.api_key is None:
        provider = getattr(args, "api_provider", "none")
        if provider == "gemini":
            args.api_key = os.environ.get("WISPR_GEMINI_API_KEY")
        elif provider == "purdue":
            args.api_key = os.environ.get("WISPR_PURDUE_API_KEY")
        else:
            args.api_key = None

    # Ensure api_provider exists
    if not hasattr(args, "api_provider"):
        args.api_provider = "none"

    return args


# ─────────────────────────────────────────────────────────────────────────────
# Vocab CLI handlers
# ─────────────────────────────────────────────────────────────────────────────

def handle_vocab(args: argparse.Namespace):
    """Handle the 'vocab' subcommand."""
    vocab = CustomVocabulary()

    if args.vocab_action == "add":
        added = vocab.add(args.words)
        if added:
            print(f"✅ Added: {', '.join(added)}")
        else:
            print("ℹ️  All words already exist.")
        all_words = vocab.list_words()
        if all_words:
            print(f"📖 Current vocabulary ({len(all_words)}): {', '.join(all_words)}")

    elif args.vocab_action == "remove":
        removed = vocab.remove(args.words)
        if removed:
            print(f"✅ Removed: {', '.join(removed)}")
        else:
            print("ℹ️  None of those words were in your vocabulary.")
        all_words = vocab.list_words()
        if all_words:
            print(f"📖 Current vocabulary ({len(all_words)}): {', '.join(all_words)}")

    elif args.vocab_action == "list":
        words = vocab.list_words()
        if words:
            print(f"📖 Custom vocabulary ({len(words)} words):")
            for w in words:
                print(f"   • {w}")
        else:
            print("📖 No custom words configured.")
            print('   Add some with: python3 wispr_free.py vocab add "YourName" "ACRONYM"')

    elif args.vocab_action == "clear":
        count = vocab.clear()
        print(f"🗑️  Cleared {count} word(s).")

    else:
        print("Usage: python3 wispr_free.py vocab {add|remove|list|clear}")
        print("  add    <words...>  — Add custom words for better recognition")
        print("  remove <words...>  — Remove custom words")
        print("  list               — Show all custom words")
        print("  clear              — Remove all custom words")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    config = parse_args()

    # Handle vocab subcommand
    if config.command == "vocab":
        handle_vocab(config)
        return

    # Start dictation
    app = WisprFree(config)
    app.run()


if __name__ == "__main__":
    main()
