# Wispr Free

**A free, open-source, fully local speech-to-text tool for macOS.**

Hold a key, speak, release — your words are transcribed and pasted instantly into any app. No cloud, no subscription, no API keys. Everything runs on your machine.

> Inspired by [Wispr Flow](https://wisprflow.ai/) — this is a free alternative that works entirely offline.

I didn't want to pay $12/mo so I made a free, jank version that runs locally.

---

## Demo

```
$ python3 wispr_free.py

🚀 Wispr Free — Hold-to-Dictate
==================================================
  Trigger key  : Right Option
  Whisper model: tiny
  Language     : en
  Auto-paste   : on
==================================================

✅ Model loaded!
👂 Listening for Right Option key...

🎤 Recording...
⚙️  Transcribing...
✅ "Hey this is a test of the speech to text tool"
📋 Pasted!
```

---

## How It Works

1. **You hold down Right Option (⌥)** — the mic starts streaming audio
2. **You speak** — audio frames are captured in real time
3. **You release the key** — recording stops, audio is sent to Whisper
4. **Whisper transcribes** your speech locally (no internet needed)
5. **Text is auto-pasted** into whatever app/text field you're using (Notes, Chrome, Slack, anywhere)

If no text field is focused, the text stays on your clipboard — just ⌘V wherever you want.

### Architecture

```
┌─────────────────────┐
│  Key Listener Thread │  ← pynput detects Right Option press/release
│  (pynput)           │
└──────┬──────────────┘
       │ on_press → start mic
       │ on_release → stop mic, enqueue audio
       ▼
┌─────────────────────┐
│  Worker Thread       │  ← dequeues audio, runs Whisper, pastes result
│  (threading)        │
└──────┬──────────────┘
       │
       ├─→ Whisper transcribe (local, offline)
       ├─→ pbcopy (clipboard)
       └─→ simulated ⌘V (pynput Controller)
```

All three threads (main, listener, worker) run concurrently so the key listener never blocks during transcription.

---

## Installation

### Prerequisites

- **macOS** (uses `pbcopy` for clipboard and `pynput` for global hotkeys)
- **Python 3.10+**
- **ffmpeg** (required by Whisper for audio decoding)

### Steps

```bash
# 1. Install ffmpeg (if you don't have it)
brew install ffmpeg

# 2. Install Python dependencies
pip3 install openai-whisper sounddevice soundfile numpy pynput

# 3. Fix SSL certificates (if you get SSL errors on first run)
#    Run the certificate installer that ships with your Python version:
/Applications/Python\ 3.13/Install\ Certificates.command

# 4. Clone this repo (or just download wispr_free.py)
git clone https://github.com/YOUR_USERNAME/wispr-free.git
cd wispr-free
```

### macOS Permissions

On first run, macOS will ask for permissions. Grant these in **System Settings → Privacy & Security**:

| Permission | What to add | Why |
|---|---|---|
| **Input Monitoring** | Your terminal app (Terminal.app, iTerm2, etc.) | So pynput can detect global key presses |
| **Accessibility** | Your terminal app | So pynput can simulate ⌘V to paste |
| **Microphone** | Prompted automatically | So sounddevice can record audio |

> **After granting permissions, restart your terminal** for them to take effect.

---

## Usage

### Basic (defaults)

```bash
python3 wispr_free.py
```

This uses the `tiny` model, Right Option (⌥) trigger key, English language, and auto-paste enabled.

### Command-Line Options

```
python3 wispr_free.py [OPTIONS]

Options:
  -m, --model {tiny,base,small,medium,large}
        Whisper model size (default: tiny)

  -t, --trigger {right_option,left_option,right_cmd,left_ctrl,right_ctrl,caps_lock,f13..f20}
        Key to hold while speaking (default: right_option)

  -l, --language LANG
        ISO 639-1 language code (default: en)

  --no-paste
        Disable auto-paste, only copy to clipboard

  -h, --help
        Show help message
```

### Examples

```bash
# Use the base model for better accuracy
python3 wispr_free.py --model base

# Use Right Command as trigger key
python3 wispr_free.py --trigger right_cmd

# Transcribe Spanish
python3 wispr_free.py --language es

# Japanese, medium model, clipboard only
python3 wispr_free.py -m medium -l ja --no-paste

# Set up a shell alias for quick access
echo 'alias wispr="python3 /path/to/wispr_free.py"' >> ~/.zshrc
source ~/.zshrc
wispr  # now just type this!
```

---

## Whisper Models — Which One Should I Use?

All models run fully offline after the first download.

| Model | Parameters | Download | RAM Usage | Speed (M1 Air) | Accuracy | Best For |
|---|---|---|---|---|---|---|
| **tiny** | 39M | ~75 MB | ~1 GB | ~1–2 sec | Good | Quick notes, casual dictation |
| **base** | 74M | ~142 MB | ~1 GB | ~2–3 sec | Better | Daily use, recommended starting point |
| **small** | 244M | ~466 MB | ~2 GB | ~4–6 sec | Great | Meetings, detailed transcription |
| **medium** | 769M | ~1.5 GB | ~5 GB | ~10–15 sec | Excellent | Professional, multi-language |
| **large** | 1550M | ~2.9 GB | ~10 GB | ~20–30 sec | Best | Maximum accuracy, complex audio |

> **Recommendation:** Start with `tiny` or `base`. Switch to `small` if you need better accuracy. Only use `medium`/`large` if accuracy is critical and you don't mind waiting.

### Switching Models

```bash
# Try base model
python3 wispr_free.py --model base

# Or edit the DEFAULT_MODEL variable in wispr_free.py:
DEFAULT_MODEL = "base"  # Change this line
```

The model is downloaded automatically on first use and cached at `~/.cache/whisper/`.

---

## Performance on Apple Silicon (M1 MacBook Air)

Wispr Free runs **great** on M1/M2/M3 Macs. Here's what to expect:

### Resource Usage (tiny model — default)

| Resource | Idle (listening) | Recording | Transcribing |
|---|---|---|---|
| **CPU** | ~0% | ~1–2% | ~80–100% (brief burst) |
| **RAM** | ~150 MB (Python + model) | ~160 MB | ~180 MB peak |
| **Total with model** | ~1 GB | ~1 GB | ~1 GB |
| **Battery impact** | Negligible | Negligible | Minimal (1–2 sec bursts) |

### Resource Usage (base model)

| Resource | Idle | Recording | Transcribing |
|---|---|---|---|
| **CPU** | ~0% | ~1–2% | ~100% (2–3 sec) |
| **RAM** | ~200 MB | ~210 MB | ~250 MB peak |
| **Total with model** | ~1 GB | ~1 GB | ~1.1 GB |

### Resource Usage (small model)

| Resource | Idle | Recording | Transcribing |
|---|---|---|---|
| **CPU** | ~0% | ~1–2% | ~100% (4–6 sec) |
| **RAM** | ~500 MB | ~510 MB | ~600 MB peak |
| **Total with model** | ~2 GB | ~2 GB | ~2.2 GB |

### Key Performance Notes

- **M1/M2/M3 Macs** benefit hugely from Apple's Neural Engine — Whisper runs faster than on equivalent x86 CPUs
- The `tiny` model transcribes ~30 seconds of audio in **1–2 seconds** on M1
- **Battery impact is minimal** — the model only activates in short bursts when you release the trigger key. Between dictations, CPU usage is essentially zero
- 8 GB RAM Macs can comfortably run `tiny`, `base`, or `small` models alongside normal apps
- 16 GB RAM Macs can run `medium` without issues
- `large` model needs ~10 GB and is only recommended for 16+ GB machines with nothing else heavy running

---

## Code Structure

The codebase is organized into clean, modular classes so you can easily extend or modify it:

```
wispr_free.py
│
├── Configuration         ← Model, sample rate, trigger keys, language
│
├── class Recorder        ← Microphone streaming (sounddevice)
│   ├── start()           ← Opens mic InputStream with callback
│   └── stop() → audio    ← Closes stream, returns numpy array
│
├── class Transcriber     ← Speech-to-text (Whisper)
│   ├── load()            ← Downloads & loads model into RAM
│   └── transcribe(audio) ← Returns text string
│
├── class OutputHandler   ← Clipboard + paste (pbcopy + pynput)
│   └── deliver(text)     ← Copies to clipboard, simulates ⌘V
│
├── class WisprFree       ← Main app (wires everything together)
│   ├── _on_press()       ← Trigger key pressed → start recording
│   ├── _on_release()     ← Trigger key released → stop → enqueue
│   ├── _worker()         ← Background thread: transcribe → paste
│   └── run()             ← Entry point: loads model, starts threads
│
└── parse_args()          ← CLI argument parsing (argparse)
```

### Extending It

**Want to add post-processing (grammar fix, summarization)?**
```python
# Add a method to WisprFree or create a new Processor class:
class Processor:
    def process(self, raw_text: str) -> str:
        # Your logic here (local LLM, API call, regex, etc.)
        return cleaned_text

# Then in _worker(), after transcription:
text = self.processor.process(text)
```

**Want to add a GUI notification?**
```python
# Use osascript to show a macOS notification:
import subprocess
subprocess.run([
    "osascript", "-e",
    f'display notification "{text}" with title "Wispr Free"'
])
```

**Want to support Linux?**
- Replace `pbcopy` with `xclip` or `xdotool` in `OutputHandler`
- Replace `pynput` keyboard simulation with `xdotool type`

**Want to log transcriptions?**
```python
# Add to OutputHandler.deliver():
with open("transcriptions.log", "a") as f:
    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {text}\n")
```

---

## Supported Languages

Whisper supports 99 languages. Pass any [ISO 639-1 code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes):

```bash
python3 wispr_free.py --language es   # Spanish
python3 wispr_free.py --language fr   # French
python3 wispr_free.py --language de   # German
python3 wispr_free.py --language ja   # Japanese
python3 wispr_free.py --language zh   # Chinese
python3 wispr_free.py --language ko   # Korean
python3 wispr_free.py --language hi   # Hindi
python3 wispr_free.py --language ar   # Arabic
python3 wispr_free.py --language pt   # Portuguese
python3 wispr_free.py --language ru   # Russian
```

> For non-English languages, `small` or `medium` models give significantly better results than `tiny`.

---

## Trigger Key Options

| Key Name | Flag | Physical Key | Notes |
|---|---|---|---|
| `right_option` | `--trigger right_option` | Right ⌥ | **Default.** Rarely used standalone |
| `left_option` | `--trigger left_option` | Left ⌥ | May conflict with special characters |
| `right_cmd` | `--trigger right_cmd` | Right ⌘ | Good if you only use left ⌘ |
| `left_ctrl` | `--trigger left_ctrl` | Left ⌃ | Easy reach, may conflict in terminal |
| `right_ctrl` | `--trigger right_ctrl` | Right ⌃ | Rarely used |
| `caps_lock` | `--trigger caps_lock` | Caps Lock | Toggles caps — use with caution |
| `f13`–`f20` | `--trigger f18` | F13–F20 | Requires key remapping (Karabiner) |

> **Why not the `fn` key?** The `fn` key is handled at the hardware level by Apple's keyboard controller. macOS never receives it as a discrete key event, so no software can detect it.

---

## Troubleshooting

### "No audio captured" or silence

- Check **System Settings → Privacy & Security → Microphone** — your terminal must be allowed
- Run `python3 -c "import sounddevice; print(sounddevice.query_devices())"` to verify your mic is detected
- Make sure your mic isn't muted or used by another app

### Key press not detected

- Grant **Input Monitoring** permission to your terminal app
- **Restart your terminal** after granting permissions
- Try a different trigger key: `python3 wispr_free.py --trigger right_cmd`

### Auto-paste not working

- Grant **Accessibility** permission to your terminal app
- **Restart your terminal** after granting
- Use `--no-paste` as a workaround (text will be on your clipboard)

### SSL error on first model download

```bash
# Run the Python certificate installer:
/Applications/Python\ 3.13/Install\ Certificates.command
# Adjust "3.13" to match your Python version
```

### Slow transcription

- Use a smaller model: `--model tiny`
- Close RAM-heavy apps if using `medium` or `large`
- Check Activity Monitor — if Python is swapping to disk, you need a smaller model

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| [openai-whisper](https://github.com/openai/whisper) | Latest | Local speech-to-text AI model |
| [sounddevice](https://python-sounddevice.readthedocs.io/) | Latest | Real-time microphone streaming |
| [soundfile](https://pysoundfile.readthedocs.io/) | Latest | WAV file writing for Whisper |
| [numpy](https://numpy.org/) | Latest | Audio buffer management |
| [pynput](https://pynput.readthedocs.io/) | Latest | Global hotkey detection + key simulation |
| [ffmpeg](https://ffmpeg.org/) | Latest | Audio decoding (system dependency) |

---

## License

MIT — do whatever you want with it. Free as in beer, free as in speech.

---

## Contributing

PRs welcome! Some ideas:

- [ ] Linux support (`xclip` / `xdotool` backend)
- [ ] Windows support (`pyperclip` / `pyautogui` backend)
- [ ] System tray icon with status indicator
- [ ] GUI settings panel
- [ ] macOS notification on transcription complete
- [ ] Transcription history / log file
- [ ] Custom Whisper model fine-tuning support
- [ ] Auto-punctuation post-processor
- [ ] Noise gate / silence trimming before transcription
