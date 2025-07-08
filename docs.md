## **BitBot: Official Build Rules & Technology Stack**

This document outlines the official technology stack and architectural principles for building **BitBot**, a local, real-time, audio-enabled AI assistant. The stack is designed to be open-source, modular, scalable, and optimized for performance on consumer hardware.

### **I. Core Architectural Principles**

1.  **Asynchronous Streaming Pipeline:** The entire system MUST operate on a concurrent, streaming `STT → LLM → TTS` pipeline. Data should flow between components in real-time to minimize perceived latency.
2.  **Centralized Orchestration:** All I/O-bound operations (audio, model inference, tool calls) MUST be managed by **Python `asyncio`**. This is critical for non-blocking execution and creating a responsive, conversational feel.
3.  **Aggressive Model Quantization:** To ensure performance on "daily driver" laptops and PCs, all models (STT, LLM, TTS) SHOULD be quantized. The **GGUF** format, particularly `Q4_K_M`, is the recommended standard for LLMs.

---

### **II. Technology Stack & Build Rules**

#### **1. Wake Word Detection: "Hey BitBot!"**

*   **Primary Technology:** **Picovoice Porcupine**
*   **Build Rule:** Use Porcupine for a highly accurate, extremely lightweight, and cross-platform wake word engine that runs efficiently on-device to trigger the assistant.
*   **Alternative:** `openWakeWord` for a pure Python, easily customizable option.
*   **Documentation:**
    *   **Porcupine GitHub:** [https://github.com/Picovoice/porcupine](https://github.com/Picovoice/porcupine)
    *   **Porcupine Website:** [https://picovoice.ai/platform/porcupine/](https://picovoice.ai/platform/porcupine/)

#### **2. Audio Input/Output (I/O)**

*   **Primary Technology:** **`sounddevice`**
*   **Build Rule:** Use the `sounddevice` library for low-latency, cross-platform audio recording and playback. It MUST be integrated into the `asyncio` event loop for non-blocking operation.
*   **Alternative:** `PyAudio`
*   **Documentation:**
    *   **`sounddevice` Docs:** [https://python-sounddevice.readthedocs.io/](https://python-sounddevice.readthedocs.io/)

#### **3. Speech-to-Text (STT) Engine**

*   **Primary Technology:** **Faster Whisper**
*   **Build Rule:** Use `Faster Whisper` to achieve real-time, high-accuracy transcription. It must be paired with a streaming implementation to process audio chunks as they arrive.
*   **Alternative:** **Whisper.cpp** for a highly efficient, pure C/C++ implementation with excellent Apple Silicon and Intel hardware support.
*   **Documentation:**
    *   **Faster Whisper GitHub:** [https://github.com/guillaumekln/faster-whisper](https://github.com/guillaumekln/faster-whisper)
    *   **Whisper.cpp GitHub:** [https://github.com/ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp)

#### **4. Local LLM Platform & Models**

*   **LLM Platform:** **Ollama**
*   **Build Rule:** Use Ollama as the exclusive platform for deploying and serving local LLMs. Its hardware acceleration, streaming tool-call support, and OpenAI-compatible API are essential for BitBot's modular architecture.
*   **Recommended Models:**
    *   **Mistral 7B Instruct (`mistral:7b-instruct`):** The recommended default for its balance of efficiency and capability.
    *   **Llama 3.1 8B Instruct (`llama3.1:8b`):** A powerful alternative for general-purpose tasks and a large context window.
*   **Documentation:**
    *   **Ollama Website:** [https://ollama.com/](https://ollama.com/)
    *   **Ollama GitHub:** [https://github.com/ollama/ollama](https://github.com/ollama/ollama)
    *   **Mistral Model Card:** [https://ollama.com/library/mistral:7b-instruct](https://ollama.com/library/mistral:7b-instruct)
    *   **Llama 3.1 Model Card:** [https://ollama.com/library/llama3.1:8b](https://ollama.com/library/llama3.1:8b)

#### **5. Text-to-Speech (TTS) Engine**

*   **TTS Orchestration Library:** **RealtimeTTS**
*   **Build Rule:** Use the `RealtimeTTS` library to manage the TTS pipeline. Its ability to abstract different backend engines makes the system modular and its low-latency focus is critical for real-time responses.
*   **Recommended Engines (via RealtimeTTS):**
    *   **Kyutai TTS (Moshi):** The primary choice for cutting-edge, ultra-low-latency, and natural-sounding speech.
    *   **Piper TTS:** The primary choice for low-resource systems where efficiency is paramount.
*   **Documentation:**
    *   **RealtimeTTS GitHub:** [https://github.com/KoljaB/RealtimeTTS](https://github.com/KoljaB/RealtimeTTS)
    *   **Kyutai TTS GitHub:** [https://github.com/kyutai/kyutai](https://github.com/kyutai/kyutai)
    *   **Piper TTS GitHub:** [https://github.com/rhasspy/piper](https://github.com/rhasspy/piper)

#### **6. Tool & External Data Integration**

*   **Tooling Protocol:** **Model Context Protocol (MCP)**
*   **Build Rule:** All external tools (e.g., weather APIs, calendars) MUST be exposed to the LLM via an MCP server. This open standard ensures a decoupled, modular, and future-proof architecture.
*   **MCP Server Framework:** **FastMCP**
*   **Build Rule:** Use the `FastMCP` Python framework to build custom MCP servers. Its decorator-based API simplifies the creation of tools.
*   **Local Data Retrieval (RAG):** **LanceDB**
*   **Build Rule:** For Retrieval-Augmented Generation, use `LanceDB` as the local vector database. Its embedded, serverless design is ideal for desktop applications and avoids network latency.
*   **Documentation:**
    *   **MCP Official Website:** [https://modelcontext.dev/](https://modelcontext.dev/)
    *   **FastMCP Docs:** [https://gofastmcp.com/](https://gofastmcp.com/)
    *   **LanceDB Docs:** [https://lancedb.github.io/lancedb/](https://lancedb.github.io/lancedb/)

---

### **III. BitBot Release Tier Configurations**

This section defines the model configurations for different target hardware profiles.

*   #### **BitBotS (Small | Standard PCs, Older Laptops)**
    *   **STT Model:** `Faster Whisper` (or `Whisper.cpp`) using the `tiny.en` or `base.en` models.
    *   **LLM Model:** `Mistral 7B Instruct` quantized to `Q4_K_M`.
    *   **TTS Engine:** `Piper TTS` for maximum efficiency and low resource use.

*   #### **BitBotM (Medium | Capable PCs, Modern Laptops with 8GB VRAM)**
    *   **STT Model:** `Faster Whisper` using the `small` or `medium` quantized models.
    *   **LLM Model:** `Llama 3.1 8B Instruct` quantized to `Q4_K_M`.
    *   **TTS Engine:** `Kyutai TTS` for a balance of quality and very low latency.

*   #### **BitBotL (Large | High-end PCs, Apple Silicon, >8GB VRAM)**
    *   **STT Model:** `Faster Whisper` (with GPU acceleration) using the `large` model for highest accuracy.
    *   **LLM Model:** `Llama 3.1` (or other high-parameter model) quantized to `Q5_K_M` or higher, depending on available VRAM.
    *   **TTS Engine:** `Kyutai TTS` to provide the most natural and responsive voice.