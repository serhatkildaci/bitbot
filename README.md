# BitBot

**Local Real-Time AI Assistant with Speech-to-Text, Large Language Model, and Text-to-Speech Integration**

## Overview

BitBot is a local, real-time, audio-enabled AI assistant designed to run entirely on consumer hardware without requiring cloud services or internet connectivity for core functionality. The project implements a streaming STT → LLM → TTS pipeline with wake word detection, optimized for responsive conversational interactions on daily-driver laptops and desktop computers.

## 🚀 Quick Start

### Current Development Status

**MVP Status**: 🔧 **In Active Development**
- ✅ Core architecture implemented
- ✅ Simplified TTS engine working
- ✅ Text chat interface ready
- ✅ Component integration designed
- 🔧 Dependency installation needed
- 🔧 Model management in progress

### Prerequisites

1. **Python 3.8+** with pip
2. **Ollama** for local LLM serving (future)
3. **Porcupine Access Key** for wake word detection (future)

### Installation

1. **Clone and navigate to project:**
   ```bash
   git clone <repository-url>
   cd bitbot
   ```

2. **Install core dependencies:**
   ```bash
   pip install -r requirements-core.txt
   ```

3. **Test the architecture:**
   ```bash
   python test_mvp.py
   ```

4. **Try the text chat interface:**
   ```bash
   python main.py chat
   ```

### Development Setup (Advanced)

1. **Install Ollama (for LLM integration):**
   ```bash
   # Download from https://ollama.com/
   ollama pull mistral:7b-instruct
   ollama serve
   ```

2. **Get Porcupine access key (for wake word):**
   - Visit https://console.picovoice.ai/
   - Create account and get access key
   - Copy `.env.example` to `.env` and add your key:
   ```bash
   cp .env.example .env
   # Edit .env and add your PORCUPINE_ACCESS_KEY
   ```

3. **Test full setup:**
   ```bash
   python main.py test
   ```

4. **Start BitBot (when ready):**
   ```bash
   python main.py start
   ```

## 🎙️ Usage

### Text Chat Mode (Available Now)
```bash
python main.py chat
```
- Interactive text conversation with BitBot
- Rich terminal interface with commands
- Conversation history and status

### Voice Mode (Coming Soon)
```bash
python main.py start
```
- **Wake BitBot:** Say "Hey BitBot"
- **Talk naturally:** BitBot processes speech and responds
- **End conversation:** Say "goodbye" or "bye"

## 📋 Current Commands

### Available Commands
```bash
python main.py chat           # Start text chat interface
python main.py config         # Show current configuration
python main.py test           # Test all components (when ready)
python main.py setup          # Show setup guide
python main.py version        # Show version info
```

### Development Commands
```bash
python test_mvp.py           # Test architecture without dependencies
```

## ⚙️ Configuration

BitBot automatically detects your hardware and configures accordingly:

### Hardware Tiers

- **BitBotS (Small)**: Standard PCs, older laptops
  - STT: Whisper tiny.en
  - LLM: Mistral 7B (Q4_K_M)
  - TTS: Simple pyttsx3

- **BitBotM (Medium)**: Modern laptops, 4GB+ VRAM
  - STT: Whisper small.en
  - LLM: Llama 3.1 8B (Q4_K_M)
  - TTS: Simple pyttsx3 + gTTS

- **BitBotL (Large)**: High-end PCs, 8GB+ VRAM
  - STT: Whisper large-v3
  - LLM: Llama 3.1 8B (Q5_K_M)
  - TTS: Simple pyttsx3 + gTTS

### Environment Variables

Edit `.env` file to customize:

```bash
# Required (for voice mode)
PORCUPINE_ACCESS_KEY=your_key_here

# Optional
OLLAMA_BASE_URL=http://localhost:11434
AUDIO_INPUT_DEVICE=0
AUDIO_OUTPUT_DEVICE=0
LOG_LEVEL=INFO
```

## 🏗️ Architecture

BitBot follows a modular, asynchronous architecture:

### Core Components

- **Audio Manager**: Cross-platform audio I/O with sounddevice
- **STT Engine**: Faster Whisper for speech recognition
- **LLM Engine**: Ollama client with OpenAI-compatible API
- **TTS Engine**: Simple pyttsx3-based synthesis (MVP)
- **Wake Word**: Picovoice Porcupine for trigger detection
- **Pipeline**: Asyncio orchestration of STT → LLM → TTS
- **Chat Interface**: Rich text-based interaction

### Data Flow

```
Audio Input → Wake Word Detection → STT → LLM → TTS → Audio Output
     ↑                ↓                              ↓
   Microphone    "Hey BitBot"                   Speakers

Text Input → LLM → Text Output
     ↑              ↓
  Chat CLI      Terminal
```

## 🛠️ Development

### Project Structure

```
bitbot/
├── audio/             # Audio I/O management
├── cli/               # Command-line interfaces  
├── config/            # Configuration management
├── core/              # Pipeline orchestration
├── llm/               # LLM integration
├── stt/               # Speech-to-text engines
├── tts/               # Text-to-speech engines
│   ├── simple_engine.py    # MVP TTS (current)
│   └── realtime_engine.py  # Advanced TTS (future)
├── wake_word/         # Wake word detection
└── __init__.py
main.py                # CLI entry point
test_mvp.py           # Architecture tests
requirements-core.txt  # Core dependencies (MVP)
requirements.txt      # Full dependencies (future)
```

### Development Status

See [DEVELOPMENT_STATUS.md](DEVELOPMENT_STATUS.md) for detailed progress tracking.

### Git Branching Strategy

- `main` - Production-ready releases
- `develop` - Integration branch for features
- `feature/*` - Individual component development
- `release/*` - Release preparation

### Key Technologies

- **Python asyncio**: Non-blocking I/O and pipeline orchestration
- **Faster Whisper**: High-performance STT inference
- **Ollama**: Local LLM deployment platform
- **pyttsx3**: Cross-platform text-to-speech (MVP)
- **Picovoice Porcupine**: Accurate wake word detection
- **sounddevice**: Cross-platform audio handling
- **Rich**: Beautiful terminal interfaces

## 🚨 Troubleshooting

### Common Issues

1. **Architecture test failures:**
   ```bash
   pip install psutil loguru
   python test_mvp.py
   ```

2. **Missing dependencies:**
   ```bash
   pip install -r requirements-core.txt
   ```

3. **Ollama not running (future):**
   ```bash
   ollama serve
   ```

4. **Audio device issues (future):**
   ```bash
   python main.py test  # Check audio devices
   ```

### Getting Help

Check detailed documentation:
- [Development Status](DEVELOPMENT_STATUS.md) - Current progress
- [MVP Development Plan](MVP_DEVELOPMENT_PLAN.md) - Complete roadmap

## 🎯 Current MVP Status

### ✅ Working Features
- ✅ Text chat interface with rich formatting
- ✅ LLM integration architecture
- ✅ Configuration system with hardware tiers
- ✅ Modular component design
- ✅ Async pipeline orchestration
- ✅ CLI commands and help system

### 🔧 In Development
- 🔧 Model downloading and management
- 🔧 Audio pipeline optimization
- 🔧 Integration testing
- 🔧 Dependency resolution completion

### 📋 Coming Soon
- Wake word detection integration
- Speech-to-text processing
- Text-to-speech synthesis
- End-to-end voice pipeline
- Performance optimization

## 🔮 Roadmap

### Next Release (v0.1.0 - MVP)
- [ ] Complete dependency installation
- [ ] Model management system
- [ ] End-to-end voice pipeline
- [ ] Integration testing
- [ ] Performance optimization

### Future Releases
- **v0.2.0**: Advanced TTS, web interface, custom wake words
- **v0.3.0**: MCP integration, RAG, multi-language
- **v1.0.0**: Production deployment, enterprise features

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

This project is in active development. See [DEVELOPMENT_STATUS.md](DEVELOPMENT_STATUS.md) for current priorities and how to contribute.

---

**BitBot**: Your local AI assistant, private by design. 🤖

*Current Status: MVP in development with core architecture complete*