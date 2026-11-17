# RCT — Reusable Computation Transformer

**An experimental Transformer inference framework exploring exact prompt deduplication, KV caching, shared-prefix reuse, and adaptive inference-path selection.**

RCT is an independently developed research and engineering project investigating opportunities to avoid redundant computation during Transformer inference. The current implementation includes an approximately 100-million-parameter Transformer and experimental inference paths evaluated on a single NVIDIA GeForce RTX 2050.

> **Project status:** Experimental implementation; performance validation and reproducibility work are ongoing. Results are workload-specific and should not be interpreted as general speedup claims.

## Contents

- [Overview](#1-overview)
- [Current implementation](#2-current-implementation)
- [Model configuration](#3-model-configuration)
- [Inference modes](#4-inference-modes)
- [Adaptive policies](#5-adaptive-policies)
- [Architecture](#6-architecture)
- [Benchmark methodology and results](#7-benchmark-methodology-and-results)
- [GPU profiling](#8-gpu-profiling)
- [Correctness and validation](#9-correctness-and-validation)
- [Environment](#10-environment)
- [Repository layout](#11-repository-layout)
- [Reproduction notes](#12-reproduction-notes)
- [Limitations](#13-limitations)
- [Future work](#14-future-work)

## 1. Overview

Autoregressive Transformer inference repeatedly uses previously processed context. RCT explores whether selected computations can be reused when requests contain identical prompts or shared prefixes.

The project investigates three questions:

1. Can identical complete prompts in a batch avoid redundant processing?
2. Can cached computation for a common prompt prefix be reused across sequences?
3. Can an adaptive policy choose an optimized path only for workloads where measurements support its use?

RCT treats these as experimental engineering questions. An optimization is not assumed to help every workload: reuse overhead can offset savings, and the fallback path remains important.

## 2. Current implementation

| Component | Reported status |
|---|---|
| RCT ~100M Transformer model | Implemented |
| Standard autoregressive generation | Implemented |
| Incremental generation with KV cache | Implemented |
| Shared-prefix generation | Implemented |
| Automatic shared-prefix detection | Implemented |
| Exact complete-prompt deduplication | Implemented |
| Adaptive deduplication gate | Implemented |
| Benchmark-guided prefix selection | Implemented |
| Adaptive prefix-gate tests | Reported passing (12/12) |
| Prefix and KV benchmarks | Completed for selected workloads |
| PyTorch GPU profiling | Completed for selected workloads |
| Broad multi-GPU validation | Pending |
| Production readiness | Not established |

## 3. Model configuration

| Parameter | Value |
|---|---:|
| Total parameters | 99,028,992 |
| Vocabulary size | 8,192 |
| Hidden size | 512 |
| Transformer layers | 30 |
| Attention heads | 8 |
| FFN intermediate size | 2,048 |
| Maximum sequence length | 512 |

The development checkpoint is reported at `checkpoints/rct100m_continued/best_model.pt` (training step 1,500; validation loss approximately 5.9246). Checkpoints are excluded from the public repository and must be obtained separately for checkpoint-dependent experiments.

## 4. Inference modes

The unified inference interface is reported in `inference.py`.

| Mode | Purpose |
|---|---|
| `standard` | Full-context autoregressive generation |
| `kv` | Incremental autoregressive generation using a KV cache |
| `prefix` | Shared-prefix generation |
| `adaptive` | Exact complete-prompt deduplication with KV caching |
| `auto` | Benchmark-guided prefix selection with adaptive KV fallback |

Example usage (confirm the current function signature and checkpoint path against the checked-out source before running):

```python
from inference import load_rct_inference

inference = load_rct_inference(
    "checkpoints/rct100m_continued/best_model.pt",
    device="cuda",
)

output = inference.generate(
    prompt,
    max_new_tokens=8,
    mode="auto",
)
```

### KV caching

Stores previously computed attention keys and values so that autoregressive decoding can reuse the context state rather than recomputing the entire preceding context at every generation step.

### Shared-prefix generation

Attempts to reuse cached computations associated with a common prompt prefix across multiple sequences.

### Exact prompt deduplication

Detects identical complete prompts within a batch and reuses work where the implementation can safely do so. The described mechanism uses exact equality, not approximate semantic similarity.

## 5. Adaptive policies

### Benchmark-guided prefix selection

The reported automatic policy currently enables the prefix path for a restricted set of workloads:

```python
AUTO_PREFIX_POLICY = {
    4: {24},
    8: {24},
}
```

The evaluated workload described in the project notes uses 32-token prompts and 8 generated tokens. The policy targets batch size 4 or 8 with a shared-prefix length of 24 tokens; other workloads fall back to the adaptive KV path.

**Important qualification:** The earlier eight-session dataset showed improvements for both batch 4 / prefix 24 and batch 8 / prefix 24. The later confirmation run supported batch 8 / prefix 24, while batch 4 / prefix 24 was inconclusive. The policy therefore needs further confirmation before both cases are treated as equally established.

### Exact-prompt deduplication gate

The reported default is:

```python
dedup_threshold = 0.75
```

Deduplication is activated when the duplicate rate reaches the configured threshold. This is an experimentally selected, workload-dependent setting—not a universal optimum.

## 6. Architecture

The following is a **conceptual architecture based on the README's description of the public interface and components**. It is not a verified source-code call graph; update it after checking the actual imports and control flow.

![RCT conceptual architecture](docs/architecture.mmd)

The diagram source is also available at [`docs/architecture.mmd`](docs/architecture.mmd).

## 7. Benchmark methodology and results

### Experimental setup

- GPU: NVIDIA GeForce RTX 2050 (4 GB)
- Model: RCT 100M
- Prompt length: 32 tokens
- Generated tokens: 8
- Timed repetitions: 30 per workload per session
- Compared paths: KV generation and automatic adaptive inference
- Reported multi-session analysis: 8 independent sessions

Speedup is defined as:

`KV median latency / AUTO median latency`

Thus, speedup above 1.0 means AUTO was faster; below 1.0 means KV was faster. A confidence interval containing 1.0 is treated as inconclusive.

### Eight-session hierarchical-bootstrap results

Reported analysis settings: 10,000 hierarchical bootstrap iterations, random seed 42, sessions resampled at the session level and repetitions resampled within selected sessions; 95% percentile confidence intervals.

| Batch | Shared prefix | Median speedup | 95% CI | Interpretation |
|---:|---:|---:|---:|---|
| 2 | 8 | 1.0024× | [0.9659, 1.0257] | Inconclusive |
| 2 | 16 | 0.9916× | [0.9542, 1.0287] | Inconclusive |
| 2 | 24 | 1.0121× | [0.9719, 1.0543] | Inconclusive |
| 4 | 8 | 1.0118× | [0.9852, 1.0442] | Inconclusive |
| 4 | 16 | 1.0213× | [0.9841, 1.1219] | Inconclusive |
| 4 | 24 | 1.0862× | [1.0217, 1.1478] | AUTO faster |
| 8 | 8 | 0.9996× | [0.9808, 1.0191] | Inconclusive |
| 8 | 16 | 0.9867× | [0.9698, 1.0099] | Inconclusive |
| 8 | 24 | 1.1443× | [1.1068, 1.1780] | AUTO faster |

These results are workload-specific evidence from the reported eight-session dataset. They do not establish general speedups across arbitrary prompt structures, batch sizes, or hardware.

### Latest post-implementation confirmation

A later benchmark was performed after a change to the shared-prefix cache-copy implementation.

| Batch | Prefix | KV median (ms) | AUTO median (ms) | Speedup | 95% CI | Interpretation |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 24 | 127.091 | 124.688 | 1.0049× | [0.9783, 1.0405] | Inconclusive |
| 8 | 24 | 142.745 | 130.612 | 1.0781× | [1.0539, 1.1246] | AUTO faster |

The confirmation run supports a batch 8 / prefix 24 improvement; batch 4 / prefix 24 is inconclusive. This is not proof that the cache-copy code change caused the difference. A controlled before-and-after experiment is needed to isolate that change.

### How to read the charts

- `assets/benchmark_multisession.png` shows the eight-session speedups and 95% confidence intervals.
- `assets/benchmark_confirmation.png` shows the later confirmation run separately.
- The two experiments are deliberately not pooled.
- The confidence intervals are reported as unadjusted for testing multiple workloads; interpret them accordingly.

## 8. GPU profiling

The reported profiling workload used batch size 8, 32-token prompts, a 24-token shared prefix, and 8 generated tokens. PyTorch Profiler was used with CPU/CUDA activities, shape recording, and memory profiling.

The investigation examined attention tensor shapes, KV-cache operations, tensor copies and concatenation, CUDA operator activity, and memory allocation behavior.

Reported approximate peak allocated GPU memory:

| Mode | Peak allocated memory |
|---|---:|
| KV | 884.666 MB |
| Prefix | 885.541 MB |
| AUTO | 885.541 MB |

These are profiler-observed values for the specified workload, not general memory requirements. The investigation identified cache-copy and concatenation allocations as possible optimization targets; no general memory reduction is claimed.

## 9. Correctness and validation

The project reports tests covering Transformer behavior, KV caching, shared-prefix generation, adaptive inference, and computation reuse.

Reported adaptive prefix-gate result:

```text
RCT ADAPTIVE PREFIX GATE TEST: PASS (12/12)
```

The described validation includes shared-prefix detection, inference-path selection, fallback behavior for unsupported workloads, output agreement checks, prompt/generation length restrictions, and gate statistics.

Floating-point operations may produce small numerical differences. Tests should distinguish tolerance-based numerical agreement from bitwise equality.

## 10. Environment

| Component | Reported configuration |
|---|---|
| Operating system | Windows |
| Python | 3.12.7 |
| GPU | NVIDIA GeForce RTX 2050 |
| GPU memory | 4 GB |
| PyTorch | 2.14.1+cu126 |
| CUDA build | 12.6 |
| Development shell | Windows PowerShell |

The project has primarily been developed and tested on one local GPU. Multi-GPU and cross-architecture validation remain future work.

## 11. Repository layout

The following is the representative structure supplied in the project notes; it is not an exhaustive directory listing.

```text
RCT/
├── config.py
├── train.py
├── train_100m.py
├── train_100m_continue.py
├── inference.py
├── README.md
├── requirements.txt
├── model/
│   ├── transformer.py
│   ├── reusable_block.py
│   ├── computation_memory.py
│   ├── fingerprint.py
│   ├── dependency.py
│   ├── adaptive_reuse.py
│   ├── adaptive_reusable_ffn.py
│   ├── deduplicated_ffn.py
│   ├── prompt_deduplication.py
│   ├── rct_100m.py
│   ├── rct_100m_kv.py
│   ├── rct_100m_prefix.py
│   └── rct_100m_prefix_generation.py
├── benchmarks/
├── tests/
├── docs/
├── assets/
├── checkpoints/   # excluded from Git
├── data/          # excluded from Git
└── results/       # excluded from Git
```

## 12. Reproduction notes

Clone the repository:

```powershell
git clone https://github.com/ManishaChindam0309/RCT.git
cd RCT
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Obtain a compatible model checkpoint separately, then configure the checkpoint path for the desired experiment. Run the tests and benchmark scripts using the commands documented in those scripts.

**Reproducibility caveat:** Before advertising a one-command reproduction, verify the dependency file, exact benchmark entry points, checkpoint acquisition instructions, random seeds, warm-up procedure, synchronization, and output locations on a clean environment. CUDA-enabled PyTorch installation may require a hardware-compatible installation command.

## 13. Limitations

RCT is an experimental research and engineering prototype. Current results do not establish:

- General speedups across arbitrary inference workloads.
- Improvements across different GPU architectures.
- Better language-model quality than comparable baselines.
- A universal reduction in computational complexity.
- Production readiness.
- Superiority over established inference frameworks.

The benchmark evidence is workload-specific and primarily based on one GPU. The adaptive policy needs validation across more prompt lengths, generation lengths, batch sizes, and hardware configurations. The reported confidence intervals are not adjusted for multiple workload comparisons.

## 14. Future work

1. Revalidate the final adaptive prefix policy with additional independent sessions.
2. Expand tests across all inference modes and edge cases.
3. Evaluate more prompt lengths, generation lengths, batch sizes, and prefix distributions.
4. Investigate GPU memory allocation, cache-copy, and concatenation overhead.
5. Compare against standard Transformer inference under identical conditions.
6. Report throughput, memory, and latency trade-offs.
7. Evaluate generation quality and output behavior.
8. Explore optimized C++/CUDA kernels.
9. Investigate cross-GPU and multi-GPU behavior.
10. Document exact commands and clean-environment reproduction.

---

## Project summary

**RCT — Reusable Computation Transformer** is an independently developed Transformer inference research project exploring exact computation reuse, adaptive deduplication, KV caching, shared-prefix generation, and benchmark-guided inference selection.

The model and experimental inference mechanisms are implemented. Work continues on performance validation, reproducibility, GPU profiling, and broader engineering evaluation.

**Repository:** https://github.com/ManishaChindam0309/RCT
