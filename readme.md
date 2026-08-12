# 🎙️ Voice Notes Agent

> **Stop writing notes. Start thinking out loud.**

A fully local, offline-first note-taking agent that replaces typing with speech.  
You dictate or read content aloud — the agent writes clean, structured Markdown notes for you, in continuity, on the same file, without you ever touching a keyboard for formatting.

---

## The Problem It Solves

Writing notes while studying or reviewing technical material has a hidden cost: **the act of formatting slows down learning**.

When you stop to think about how to structure a table, what heading level to use, or how to phrase a bullet point, you are spending cognitive resources on _presentation_ instead of _understanding_. The context switch kills your flow.

This tool removes that overhead entirely:

- You speak naturally — explaining a concept, giving a command, or asking for a definition
- The agent figures out whether you are _describing_ something or _giving an instruction_
- It writes properly structured Markdown and appends to the active file in continuity
- The file stays open across recordings — no new file created on every session

### Key benefits

- **Speed** — speaking is 3–5× faster than typing for dense technical content
- **Zero formatting overhead** — you never write a single Markdown character
- **Stateful continuity** — the agent reads the tail of the current file before writing, so it never repeats itself and always picks up where you left off
- **Built-in learning aid** — say _"add a definition of X"_ and the agent inserts an inline blockquote with the explanation, directly in your notes
- **Fully private and offline** — Whisper runs on CPU, the LLM runs locally via llama-server; no data leaves your machine

---

## Architecture

```
Microphone
    │
    ▼
┌──────────────────────────────┐
│   faster-whisper  (CPU)      │  — transcribes speech, auto-detects language
│   model: small, int8         │  — ~2–5s latency, leaves all VRAM for the LLM
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│   In-memory Queue            │  — decouples transcription from generation
└──────────────────────────────┘    you can record again while the LLM is still writing
    │
    ▼
┌──────────────────────────────┐
│   Agent Worker Thread        │  — reads last 3000 chars of the active file as context
│                              │  — sends system prompt + context + transcript to LLM
└──────────────────────────────┘
    │  (streaming HTTP / SSE)
    ▼
┌──────────────────────────────┐
│   llama-server  (GPU)        │  — OpenAI-compatible API, port 12345
│   gemma-4 26B Q4_K_XL        │  — ~17–20 t/s on RTX 3070 8GB VRAM
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│   Markdown file on disk      │  — created once, appended forever
└──────────────────────────────┘
```

### Design decisions

**Why a queue and not a direct call?**  
Whisper runs on CPU and finishes in a few seconds. The LLM runs on GPU and takes longer. Without a queue, the GUI would block between recordings. With the queue, you can finish a recording, immediately start the next one, and the agent catches up in the background, the two processes never step on each other.

**Why streaming?**  
At ~17 t/s, a 2048-token response takes around 2 minutes. A standard blocking HTTP call would time out waiting for the full body. Streaming sets a 30-second timeout only on receiving the _first_ byte, after that the connection stays alive for the full generation, regardless of length.

**Why CPU for Whisper?**  
The RTX 3070 has 8 GB of VRAM. A quantized 26B model uses most of that. Running Whisper on the GPU would either not fit or force a smaller LLM. The Ryzen 9 3900x handles `faster-whisper small int8` in 2–5 seconds — fast enough that the bottleneck is always the LLM, not the transcription.

---

## Benchmarks (reference hardware)

| Component | Spec                                      |
| --------- | ----------------------------------------- |
| CPU       | AMD Ryzen 9 3900X (12 cores / 24 threads) |
| RAM       | 32 GB DDR4                                |
| GPU       | NVIDIA RTX 3070 - 8 GB GDDR6              |
| OS        | Windows 11                                |
| CUDA      | 12.x                                      |

| Metric                                           | Value                           |
| ------------------------------------------------ | ------------------------------- |
| Whisper transcription latency (small, int8, CPU) | ~2–5 s depending on clip length |
| LLM generation speed (gemma-4 26B Q4_K_XL, GPU)  | ~17–20 tokens/s                 |
| Time to first token (streaming)                  | < 5 s                           |
| VRAM usage (LLM only)                            | ~7.2 GB                         |
| RAM usage (Whisper small + Python)               | ~1.5 GB                         |
| Response length for a typical note block         | 300–600 tokens (~15–35 s)       |

These numbers are from real usage with the configuration above. Smaller models (e.g. 7B or 12B) will be significantly faster if you have less VRAM or want lower latency.

---

## Requirements

### Python packages

```
Python >= 3.10
faster-whisper
sounddevice
scipy
numpy
requests
tkinter        # included in standard Python on Windows
```

```cmd
pip install faster-whisper sounddevice scipy numpy requests
```

### CUDA (for GPU inference with llama-server)

You need a working CUDA installation to run llama-server on GPU.

1. **NVIDIA driver** — version 525 or later  
   Download: https://www.nvidia.com/drivers

2. **CUDA Toolkit** — version 12.x recommended  
   Download: https://developer.nvidia.com/cuda-downloads

3. **Verify your setup:**

```cmd
nvidia-smi
```

Expected output (example):

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.03    Driver Version: 560.35.03    CUDA Version: 12.6                |
+-----------------------------------------------------------------------------------------+
| GPU  Name                 | RTX 3070          | 8192 MiB VRAM                          |
```

If `nvidia-smi` is not found, your driver is not installed or not on PATH.

4. **llama-server CUDA build**  
   Download the pre-built CUDA binary from the [llama.cpp releases page](https://github.com/ggml-org/llama.cpp/releases).  
   Look for a file named `llama-b<version>-bin-win-cuda-<version>-x64.zip`.  
   Extract and run `llama-server.exe` from that folder.

> **Note:** Do not use the CPU-only build — it will work but generation speed will drop to ~1–2 t/s, making the tool impractical for real-time use.

### Model

The default model is `unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL`, downloaded automatically by llama-server from HuggingFace on first run.

If you want to pre-download it manually:

```cmd
pip install huggingface_hub
huggingface-cli download unsloth/gemma-4-26B-A4B-it-GGUF --include "UD-Q4_K_XL*"
```

---

## Setup & Usage

### 1. Start llama-server

```cmd
.\llama-server.exe -hf unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL ^
  --cache-type-k q4_0 ^
  --cache-type-v q4_0 ^
  --ctx-size 64000 ^
  --port 12345 ^
  --cache-ram 4096 ^
  --temp 1.0 --top-p 0.95 --top-k 64 --repeat-penalty 1.0 ^
  --no-mmap
```

Wait until the console shows:

```
I srv  llama_server: listening on http://127.0.0.1:12345
```

> **Tip:** The `--cache-ram 4096` flag offloads KV cache to RAM when VRAM is tight. Remove it if you have a larger GPU and want maximum speed.

### 2. Start the app

```cmd
python main.py
```

Whisper downloads the `small` model automatically on first run (~240 MB).

### 3. Record

1. Select your microphone from the dropdown
2. Click **START RECORDING** and speak
3. Click **STOP & TRANSCRIBE** when done
4. Whisper transcribes → text enters the queue → agent writes to the `.md` file

The active file path is shown at the top of the window.  
You can start a new recording immediately, the agent will process entries from the queue sequentially in the background.

---

## What You Can Say

| Intent                     | Example                                                                                                                             | Result                                      |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| Describe a concept         | _"A load balancer distributes incoming requests across multiple backend servers to prevent any single node from being overwhelmed"_ | Expanded Markdown section with heading      |
| New topic / new file       | _"New topic: digital wallet system design"_                                                                                         | Creates a new `.md` file                    |
| Add a table                | _"Create a table with columns TPS and Node Count, first row 120k and 1, second row 1.2 million and 10"_                             | Formatted Markdown table                    |
| Add a bullet list          | _"Add a bullet list of functional requirements: bank transfer support, one million TPS, 99.9% availability, transaction rollback"_  | `- item` bullet list                        |
| Add a heading              | _"Add a section heading: Introduction and Problem Scope"_                                                                           | `## Introduction and Problem Scope`         |
| Ask for a definition       | _"I don't know what a consensus algorithm is, add a definition"_                                                                    | `> **Consensus algorithm**: ...` blockquote |
| Continue the current topic | _(just keep talking)_                                                                                                               | Appended to the current section             |

The agent infers intent from natural speech — you do not need to use exact keywords.

---

## Configuration

All tuneable constants live in `agent/config.py`:

| Constant           | Default                                      | Description                                                                          |
| ------------------ | -------------------------------------------- | ------------------------------------------------------------------------------------ |
| `LLAMA_SERVER_URL` | `http://localhost:12345/v1/chat/completions` | llama-server OpenAI-compatible endpoint                                              |
| `LLAMA_MODEL`      | `"gemma"`                                    | Model name in the request body (ignored by llama-server, required by the API schema) |
| `SAMPLE_RATE`      | `16000`                                      | Audio capture sample rate (keep at 16 kHz for Whisper)                               |
| `BASE_DIR`         | Project root directory                       | Output directory for all generated `.md` files                                       |

**Swap the model**: replace the `-hf` argument in the llama-server command with any GGUF model.

**Use a lighter Whisper model**: change `"small"` to `"tiny"` or `"base"` in `agent/transcriber.py` for faster transcription on slower CPUs (slight accuracy tradeoff).

**Force output language**: edit the `SYSTEM_PROMPT` constant in `agent/agent.py` to instruct the agent to always write in a specific language regardless of what language you speak.

---

## Project Structure

```
AI-Notes/
├── main.py                   # GUI (NotesApp) + entry point
├── README.md                 # this file
├── agent/
│   ├── config.py             # constants, shared agent_state and text_queue
│   ├── session.py            # save_session / load_last_session
│   ├── agent.py              # SYSTEM_PROMPT + agent_worker thread
│   └── transcriber.py        # Transcriber class (mic capture + Whisper)
└── tests/
    ├── run_tests.py          # headless integration test runner
    ├── test_scenarios.json   # test scenarios (editable, no code required)
    └── output/               # .md files generated by the test runner (git-ignored)
```

Generated `.md` files from normal usage are saved in the project root directory.

---

## Testing

### Why a test suite

Testing the agent manually means dictating the same content over and over every time you change a prompt, swap a model, or touch the agent logic. The test runner solves this by injecting fixed text transcriptions directly into the agent queue, bypassing Whisper and the GUI entirely. You get the same `.md` output you would get from real dictation, without touching a microphone.

The output is intentionally **not auto-validated**: there is no assert checking whether a table looks nice or a definition is well-written. That judgment call is yours. The runner's job is to produce the files; your job is to open them and read them.

### How it works

`tests/run_tests.py` imports the agent logic from the `agent/` package and redirects all file output to `tests/output/`. It replaces the GUI with a minimal stub that prints log lines to the terminal. The agent worker thread runs exactly as in the real app — same system prompt, same LLM call, same queue — so what you test is the real thing.

Each scenario is a list of steps. Steps are processed one at a time, in order, and the runner waits for the LLM to finish each one before sending the next. Agent state (current open file) is reset between scenarios.

### Prerequisites

llama-server must be running before launching the test runner, exactly as for normal usage:

```cmd
.\llama-server.exe -hf unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL ^
  --cache-type-k q4_0 --cache-type-v q4_0 ^
  --ctx-size 64000 --port 12345
```

### Running the tests

```cmd
# run all scenarios
python tests\run_tests.py

# run a single scenario by name
python tests\run_tests.py --scenario "01 - System Design Notes"

# list all available scenarios
python tests\run_tests.py --list
```

Output files land in `tests\output\`. Open them in any Markdown viewer (VS Code preview, Obsidian, Typora) to verify the result.

### Adding or editing scenarios

Open `tests/test_scenarios.json`. Each entry is an object with a name, a description, and a list of steps, plain text strings, exactly what you would say out loud:

```json
{
  "scenario": "04 - My new scenario",
  "description": "What I want to verify",
  "steps": [
    "Create a new file about neural networks",
    "Add a definition for backpropagation",
    "Create a table with columns Layer and Activation Function. First row: input and none. Second row: hidden and ReLU. Third row: output and Softmax"
  ]
}
```

No code changes required. The runner picks up the file automatically on the next run.

### Included scenarios

| Scenario                   | What it covers                                                                                                   |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `01 - System Design Notes` | New file creation, concept description, definition blockquote, table, bullet list, new heading, paragraph append |
| `02 - New Topic Switch`    | Switching to a new file mid-session, continuity across steps                                                     |
| `03 - Resume and Extend`   | Resume the document produced by scenario 01 and append a new section, verifying cross-session continuity         |

> Note on language testing: Whisper's language support is best verified with real microphone input using `wishper.py`. The test runner bypasses Whisper entirely, so a dedicated language scenario would only test the LLM — not the transcription layer where language handling actually matters.

---

## License

MIT — use it, fork it, extend it.
