# Aura Architecture Specification

> **Scope note (reviewed 2026-08-01).** This document provides a high-level summary of system rules and a component map. The authoritative technical specification is [../ARCHITECTURE.md](../ARCHITECTURE.md) (§0–§18); if the two disagree, that specification takes precedence. This file is kept for its **invariants list**, which clearly states the core operational rules that must always hold true.

## System Overview

Aura is an AI cognitive system organized around discrete execution steps called "ticks." Every input passes through a step-by-step processing pipeline, saves its resulting state to persistent storage, and maintains continuous background processing through a once-per-second pulse (a 1Hz heartbeat).

## Invariants (must always be true)

1. **Every tick is atomic (all-or-nothing)**: acquire lock → run phases → save state → release lock. Incomplete states are never saved.
2. **State is event-sourced**: each phase creates a new, unchangeable state record derived from the previous one.
3. **The kernel never crashes on a single phase failure**: errors in an individual phase are caught and logged, and the tick continues running.
4. **Vault commit failures are non-fatal**: if saving to long-term storage fails, the system still returns a response to the user.
5. **Identity instructions stay closer to generation than conversation history**: core personality prompts are placed near the end of the prompt context, so long chats cannot push the system's personality out of the language model's active memory window.
6. **System prompts never exceed 5000 tokens**: a strict cap prevents overflowing the language model's input limit.
7. **No raw numeric metrics in user-facing responses**: internal emotional and cognitive scores influence the tone of voice, but are never recited directly to the user.
8. **Mock fallbacks exist for every organ**: the kernel completes its tick even if underlying hardware or AI models are unavailable.
9. **Working memory caps at 40 turns**: older conversation history is summarized before memory and context quality degrade.
10. **The substrate pauses when idle**: background processing pauses to avoid wasting CPU power when there is no user interaction.

## Component Map

```
aura_main.py                    Entry point (--desktop, --headless)
├── interface/server.py          FastAPI + WebSocket server
│   ├── routes/chat.py           Chat API (user messages)
│   ├── routes/system.py         Health/status API
│   └── routes/subsystems.py     Telemetry, code graph, voice APIs
├── core/kernel/
│   ├── aura_kernel.py           The tick loop (AuraKernel.tick())
│   ├── kernel_interface.py      Public API for the kernel
│   ├── organs.py                Lazy-loading organ stubs
│   └── organ_fallbacks.py       Fallback implementations
├── core/phases/                 Linear pipeline phases
│   ├── phi_consciousness.py     IIT 4.0 phi computation
│   ├── response_generation_unitary.py  LLM response with cognitive context
│   ├── cognitive_routing_unitary.py    Intent classification
│   └── ...                      Affect, motivation, memory consolidation
├── core/consciousness/          136 consciousness modules (2026-08-01)
│   ├── affective_steering.py    Residual stream injection (CAA)
│   ├── phi_core.py              IIT 4.0 (TPM, KL-divergence, MIP)
│   ├── liquid_substrate.py      20Hz dynamical system
│   ├── stdp_learning.py         Reward-modulated plasticity
│   ├── global_workspace.py      Competitive broadcast (Baars)
│   └── ...                      Heartbeat, attention, temporal binding
├── core/memory/
│   ├── conceptual_gravitation.py  Memory drift (C.O.R.E. inspired)
│   ├── knowledge_compression.py   Three-layer DTU compression
│   ├── navigating_graph.py        NSG proximity search
│   └── episodic_memory.py         SQLite episodic store
├── core/brain/llm/
│   ├── mlx_client.py             MLX model client
│   ├── mlx_worker.py             Isolated model process (loads LoRA)
│   ├── llm_router.py             Multi-tier failover
│   └── inference_gate.py         Request routing + cloud fallback
├── core/affect/
│   ├── damasio_v2.py             Plutchik + somatic markers
│   └── affective_circumplex.py   Valence/arousal → sampling params
├── core/introspection/
│   └── code_graph.py             AST-based codebase self-knowledge
├── training/
│   ├── personality_spec.py       Character fusion training data
│   ├── build_dataset.py          JSONL generator
│   └── finetune_lora.py          MLX LoRA fine-tune
├── benchmarks/
│   ├── cognitive_stack_comparison.py  10-turn personality benchmark
│   └── long_horizon_stress.py         30-turn drift detection
└── tests/                        181+ tests
```

## Data Flow: User Message → Response

```
1. HTTP POST /api/chat {"message": "..."}
2. chat.py: fast-path checks (identity challenge, architecture query)
3. KernelInterface.process(message)
4. AuraKernel.tick(message):
   a. Acquire lock (90s timeout, 3 retries)
   b. Derive new state version
   c. For each phase in pipeline:
      - Execute with timeout (85s foreground, 45s background)
      - On failure: log, continue (never crash the tick)
   d. Commit state to vault (catch BrokenPipeError)
   e. Release lock
5. Response extracted from state.cognition.last_response
6. chat.py: _stabilize_user_facing_reply (prefer LLM text over templates)
7. Return JSON response
```

## Inference Modulation (3 levels)

```
Level 1: Activation Steering (affective_steering.py)
  - Hooks into MLX transformer forward pass
  - Injects direction vectors into residual stream: h = h + alpha * v
  - Computed via contrastive activation addition (CAA)

Level 2: Sampling Parameters (affective_circumplex.py)
  - Valence/arousal → temperature, max_tokens, repetition_penalty
  - Neurochemical modulation: dopamine → temp, serotonin → tokens, cortisol → brevity

Level 3: System Prompt (response_generation_unitary.py)
  - Personality spec, mood description, neurochemical cues, phi integration level
  - Hard-capped at 20K chars (~5000 tokens)
  - Identity anchor injected after 10+ turns
```

## State Persistence

- **SQLite** via `StateRepository`: records system state as an immutable history of events.
- **Working memory**: an active in-memory conversation list, capped at 40 turns.
- **Rolling summary**: compressed conversation history managed by `KnowledgeCompressor` (broken down into 3-layer semantic units).
- **Episodic memory**: stores past events in SQLite, using `NavigatingGraph` for fast logarithmic-time (O(log N)) retrieval.
- **Conceptual gravitation**: gradually shifts concept embeddings (semantic associations) during background dream cycles.
- **STDP weights**: updates synaptic learning weights in the neural liquid substrate every 100 ticks.
