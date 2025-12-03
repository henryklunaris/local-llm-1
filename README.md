# Local Voice Assistant

A fully local, voice ai that runs entirely on your machine. No cloud APIs, no subscriptions.

## Features

- 🗣️ **Speech-to-Text** - Fast transcription with faster-whisper
- 🤖 **Local LLM** - Uses LM Studio for inference (OpenAI-compatible API)
- 🔊 **Text-to-Speech** - Lightning-fast Supertonic TTS (ONNX-based)
- 💬 **Conversation Memory** - Maintains chat history across turns

## Tech Stack 

| Component | Technology |
|-----------|------------|
| **STT** | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (small.en) |
| **TTS** | [Supertonic](https://github.com/supertone-inc/supertonic) (ONNX) |
| **LLM** | [LM Studio](https://lmstudio.ai/) (any model) |
| **VAD** | [Silero VAD](https://github.com/snakers4/silero-vad) (neural network) |

## Prerequisites

- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** - Fast Python package manager
- **[LM Studio](https://lmstudio.ai/)** - Local LLM inference server

## Installation

### 1. Install uv (if not already installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with Homebrew
brew install uv
```

### 2. Clone and setup

```bash
git clone https://github.com/henryklunaris/local-talking-llm.git
cd local-talking-llm

# Install dependencies
uv sync

# Download TTS models (~260MB)
uv run python download_models.py
```

### 3. Setup LM Studio

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Download a model (e.g., Qwen3-8B, Llama 3, Mistral)
3. Start the local server (default: `http://localhost:1234`)
4. (Optional) Create a system prompt preset for personality

## Usage

### Basic Usage

```bash
uv run python app.py
```

Just start speaking - the assistant will automatically detect when you talk and respond.

### Command-Line Options

```bash
# Different voice styles (F1, F2, M1, M2)
uv run python app.py --voice-style assets/voice_styles/M1.json

# Faster speech
uv run python app.py --speed 1.3

# Adjust VAD sensitivity (0.0-1.0, higher = stricter speech detection)
uv run python app.py --vad-threshold 0.6

# Save voice responses to files
uv run python app.py --save-voice
```

### All Options

| Option | Default | Description |
|--------|---------|-------------|
| `--voice-style` | `assets/voice_styles/F1.json` | TTS voice (F1, F2, M1, M2) |
| `--speed` | `1.05` | Speech speed (higher = faster) |
| `--steps` | `5` | TTS quality steps (higher = better) |
| `--model` | `qwen/qwen3-8b` | LLM model name in LM Studio |
| `--vad-threshold` | `0.5` | Silero VAD threshold (0.0-1.0) |
| `--silence-duration` | `0.8` | Seconds of silence to stop recording |
| `--min-speech-duration` | `0.5` | Minimum speech length to process |
| `--save-voice` | `false` | Save responses to `voices/` |
| `--use-gpu` | `false` | Use GPU for TTS (if available) |

## LM Studio Configuration

### Disabling "Thinking" Mode (Qwen3)

If using Qwen3 and seeing `<think>` tags in responses:

1. In LM Studio, go to the model settings
2. Find "Enable Thinking" toggle and turn it **OFF**
3. Or create a preset with your system prompt

### System Prompt

The assistant uses LM Studio's system prompt preset. Create a preset in LM Studio with your desired personality, for example:

```
You are a helpful and friendly AI assistant. Keep responses concise (under 20 words).
```

## Troubleshooting

### "No module named 'xyz'"
```bash
uv sync  # Reinstall dependencies
```

### Microphone not detected
- Check system permissions for microphone access
- Verify your default audio input device

### Slow transcription
- The first run downloads the Whisper model (~460MB)
- Consider using `--vad-threshold 0.6` to filter more noise

### TTS model not found
```bash
uv run python download_models.py  # Re-download models
```

### LM Studio connection refused
- Ensure LM Studio is running with the server started
- Check the server is on `http://localhost:1234`

## Project Structure

```
local-talking-llm/
├── app.py              # Main application
├── tts.py              # Supertonic TTS wrapper
├── download_models.py  # Model downloader
├── pyproject.toml      # Dependencies
├── assets/
│   ├── onnx/           # TTS models (downloaded)
│   └── voice_styles/   # Voice presets
└── voices/             # Saved audio (if --save-voice)
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## About the Tech

### What is ONNX?

[ONNX](https://onnx.ai/) (Open Neural Network Exchange) is an open format for representing machine learning models. It allows models trained in one framework (like PyTorch) to run efficiently in a lightweight runtime without needing the full framework installed.

**Why ONNX for this project?**
- No PyTorch dependency (~2GB saved)
- Faster inference on CPU
- Smaller installation footprint
- Cross-platform compatibility

### What is Supertonic?

[Supertonic](https://github.com/supertone-inc/supertonic) is a lightning-fast, on-device text-to-speech system by Supertone Inc. 

**Key features:**
- **Speed**: Up to 167x real-time synthesis
- **Size**: Only ~66M parameters (~260MB model)
- **Quality**: Handles complex text normalization (dates, numbers, abbreviations)
- **Privacy**: Runs 100% locally, no cloud API needed
- **Voices**: Multiple pre-built voice styles (male/female)

The ONNX version we use runs efficiently on CPU without requiring a GPU.

## Acknowledgments

- [Supertonic](https://github.com/supertone-inc/supertonic) - Lightning-fast TTS
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Optimized Whisper
- [LM Studio](https://lmstudio.ai/) - Local LLM inference
- [LangChain](https://langchain.com/) - LLM orchestration
