# Aura Model Card

*The default settings for these models are found in `core/config.py:LLMConfig` (verified on August 1, 2026).
You can override any of these settings by using environment variables like `AURA_MODEL`, `AURA_DEEP_MODEL`, `AURA_BRAINSTEM_MODEL`, `AURA_FALLBACK_MODEL`, or any variable starting with `AURA_LLM__`.*

## Primary Model (Cortex)

| Field | Value |
|-------|-------|
| **Role** | Primary reasoning and conversation |
| **Architecture** | Transformer LLM (27B parameters, fused Qwen3.8-27B / `Aura-Cortex`, migrated from historical 32B) |
| **Runtime** | MLX on Apple Silicon |
| **Quantization** | MLX fused native weights (`training/fused-model/active.json`); legacy 8-bit/4-bit profiles supported |
| **Context Window** | 8192–262,144 tokens (configurable) |
| **Inference** | Local, on-device |
| **Fine-tuning** | Promoted fused LoRA delta (`training/fused-model/active.json`) as the live Cortex without a re-quantize |

### Intended Use
This is the main model you interact with. It handles all direct conversations, thinks through problems, plans how to use tools, and performs complex mental tasks.

### Limitations
- May make up information (hallucinate) if it doesn't know the answer.
- The amount of past conversation it can remember (its context window) limits how deeply it can think through long, multi-step problems.
- Uses a compressed format (4-bit quantization) to save memory, which slightly reduces its precision.
- Cannot process images or audio directly.

### Ethical Considerations
- Built on publicly available model foundations.
- Trained without any private or personal data.
- Includes built-in safeguards to prevent users from manipulating it with harmful instructions (prompt injection).

---

## Deep Model (Solver)

| Field | Value |
|-------|-------|
| **Role** | Deep-reasoning hot-swap tier for hard problems |
| **Architecture** | Transformer LLM (72B parameters, Qwen2.5-72B-Instruct) |
| **Runtime** | MLX on Apple Silicon |
| **Quantization** | 4-bit (MLX native) |
| **Inference** | Local, on-device |

### Intended Use
This model is swapped in automatically when a problem requires extremely deep thinking. It runs on high-end desktop computers (like those with 64GB of memory). It is the smartest local model but also the slowest (taking about 84 seconds per step). Because of this, it is not used for normal conversation. The 27B Cortex model handles regular chats, and the Solver only steps in when a task is especially difficult. You can enable or detect it using `AURA_DEEP_MODEL`.

---

## Background Model (Brainstem)

| Field | Value |
|-------|-------|
| **Role** | Background maintenance, classification, lightweight tasks |
| **Architecture** | Transformer LLM (9B parameters, Qwen3.5-9B) |
| **Runtime** | MLX on Apple Silicon |
| **Quantization** | 4-bit (MLX native) |
| **Context Window** | Dynamically measured from model config (default 32,768 tokens) |
| **Inference** | Local, on-device |
| **Reasoning mode** | Explicitly controlled |

We upgraded this model from Qwen2.5-7B to Qwen3.5-9B on August 12, 2026. Because no other parts of the system relied on the exact structure of the old model, we were able to upgrade it easily. (The Reflex model below could not be upgraded for exactly this reason.) For more details, see [docs/MODEL_ROSTER.md](docs/MODEL_ROSTER.md).

### Intended Use
Used for behind-the-scenes work: organizing memory, sorting information, checking system health, and doing routine maintenance. It never speaks directly to the user in normal operation.

### Limitations
- Not as smart as the primary model.
- Cannot handle complex, multi-step problems.
- Operates strictly in the background so it doesn't slow down or interfere with user conversations.

---

## Reflex Model

| Field | Value |
|-------|-------|
| **Role** | Fast reflex lane: sub-second acknowledgements, routing, guards |
| **Architecture** | Transformer LLM (1.5B parameters, Qwen2.5-1.5B-Instruct) |
| **Runtime** | MLX on Apple Silicon |
| **Quantization** | 4-bit (MLX native) |
| **Inference** | Local, on-device |

### Intended Use
This is the fastest local model. It handles quick, automatic responses, basic routing, and simple safety checks while the larger 27B Cortex model is busy or warming up. This allows the system to reply instantly instead of making the user wait. It is never used to write deep or complex answers.

---

## Speech-to-Text Model

| Field | Value |
|-------|-------|
| **Role** | Primary ASR; serves both duplex stages (480 ms partials and the final) |
| **Architecture** | Parakeet TDT 0.6B v3 (`parakeet_mlx`) |
| **Runtime** | MLX on Apple Silicon |
| **Inference** | Local, on-device. Audio does not leave the machine |
| **Fallback** | `faster_whisper` on CPU |

This model replaced an older two-step Whisper setup on August 12, 2026. In our local tests on 12.4 seconds of real speech, the Parakeet model processed the audio in just **166 milliseconds**. This was significantly faster than the old models (`whisper-small.en` took 193 ms and `whisper-large-v3-turbo` took 317 ms). Because this new model is so fast, we now use it for both quick partial transcriptions and final polished text, saving system resources without losing any accuracy.

### Limitations
- Primarily designed for English. The official tests show a very low error rate (between 6.32% and 7.83%), but our small local tests were too short to show any errors at all.

---

## Embedding Model

| Field | Value |
|-------|-------|
| **Role** | Semantic memory retrieval — the dense half of hybrid scoring |
| **Architecture** | `Qwen/Qwen3-Embedding-0.6B` |
| **Dimensions** | 384 |
| **Inference** | Local, on-device |

This model helps the system search through its memory by turning text into searchable numbers. It replaced `all-MiniLM-L6-v2` on August 12, 2026. The old model had a severe flaw: it could only read 256 words at a time, but we were feeding it 800-word chunks. As a result, **77% of every long document was silently ignored**. When important information was at the end of a document, the old model almost always failed to find it, whereas the new Qwen3 model finds it reliably. The new model takes slightly longer (20.2 milliseconds vs 10.7 milliseconds), but we now size our text chunks to perfectly match what the model can actually read.

### Limitations
- Takes about twice as long per search query as the model it replaced.
- It can sometimes be tricky to tell the difference between a good search result and a bad one. To fix this, we measure what a "bad" result looks like in practice (`core/memory/retrieval_calibration.py`) and set our quality standards based on those tests, rather than just guessing.

---

## Model Verification

The system identifies a model by actually inspecting its files, not just by looking at its folder name. We built `core/brain/llm/model_artifact_profile.py` to do this because relying on folder names was dangerous. For example, if someone renamed a massive model's folder to "cortex" or included "32b" in the name, the system would get confused about how much memory it needed.

When loading a model, the system reads:

- `model.safetensors.index.json` to find the exact number of parameters and file size.
- `config.json` to understand the model's architectural shape.
- The list of actual model files and their sizes.

From this information, the system creates a unique digital fingerprint (using a SHA-256 hash) based on the configuration and file sizes. **This is not a hash of the entire 20 GB model** — reading 20 GB every time would take way too long. Instead, it proves the model's identity based on its structure. If the model's declared shape changes, the fingerprint changes.

The system also keeps a record of how it identified the model. This lets us know if it successfully measured the real files, or if it had to guess based on the folder name (which we only do during testing or before a model is fully downloaded).

Finally, before starting the model, the system checks if your computer actually has enough memory to run it. For instance, the `AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB` setting acts as a safety limit; if your computer doesn't have that much memory available, the system will refuse to load the model to prevent a crash.
