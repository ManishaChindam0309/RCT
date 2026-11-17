# RCT — Reusable Computation Transformer

**An experimental Transformer inference framework exploring reusable computation, exact prompt deduplication, KV caching, and shared-prefix generation.**

RCT is an independently developed research and engineering project focused on reducing redundant computation during Transformer inference.

The project explores several inference optimization techniques, including:

- Exact duplicate-prompt detection and deduplication.
- Incremental autoregressive generation using KV caching.
- Shared-prefix KV reuse.
- Adaptive inference-path selection.
- Benchmark-guided optimization policies.

The current implementation includes an approximately 100-million-parameter Transformer model and experimental inference optimizations validated on an NVIDIA GeForce RTX 2050.

**Project status:** Experimental implementation and ongoing performance validation.

---

## 1. Project overview

Transformer inference repeatedly performs computation across input tokens and generation steps. RCT investigates opportunities to reuse previously computed information when workloads contain identical prompts or shared prefixes.

The project currently focuses on three engineering questions:

1. Can identical prompts within a batch be processed without unnecessarily repeating computation?
2. Can shared prompt prefixes be reused efficiently across multiple sequences?
3. Can an adaptive inference policy select an optimization only when benchmark evidence supports its use?

RCT is an experimental framework. Its optimizations are evaluated individually and under controlled workloads rather than assumed to provide universal speedups.

## 2. Project status

| Component | Status |
|---|---|
| RCT 100M Transformer model | Implemented |
| Standard autoregressive generation | Implemented |
| Incremental KV-cache generation | Implemented |
| Shared-prefix generation | Implemented |
| Automatic shared-prefix detection | Implemented |
| Exact complete-prompt deduplication | Implemented |
| Adaptive deduplication gate | Implemented |
| Benchmark-guided prefix selection | Implemented |
| Adaptive prefix gate tests | Passing 12/12 |
| Prefix and KV benchmarks | Completed |
| PyTorch GPU profiling | Completed |
| Broad multi-GPU validation | Pending |
| Production readiness | Not established |

## 3. Model configuration

The current RCT 100M model configuration:

| Parameter | Value |
|---|---:|
| Total parameters | 99,028,992 |
| Vocabulary size | 8,192 |
| Hidden size | 512 |
| Transformer layers | 30 |
| Attention heads | 8 |
| FFN intermediate size | 2,048 |
| Maximum sequence length | 512 |

The continued-training checkpoint used during development:

`checkpoints/rct100m_continued/best_model.pt`

Recorded checkpoint information:

- Training step: 1,500
- Validation loss: approximately 5.9246

**Note:** Model checkpoints are excluded from the public Git repository. A compatible checkpoint must be obtained separately to reproduce checkpoint-dependent experiments.

## 4. Inference architecture

The unified inference interface is implemented in `inference.py`.

Five inference modes are available:

| Mode | Description |
|---|---|
| `standard` | Full-context autoregressive generation |
| `kv` | Incremental autoregressive generation with KV caching |
| `prefix` | Shared-prefix generation |
| `adaptive` | Exact complete-prompt deduplication with KV caching |
| `auto` | Benchmark-guided prefix selection with adaptive KV fallback |

Example:

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

### Inference optimization concepts

**KV caching**

Stores previously computed attention keys and values so that autoregressive generation does not need to recompute the entire preceding context at every decoding step.

**Shared-prefix generation**

Reuses cached computations associated with a common prompt prefix across multiple sequences.

**Exact prompt deduplication**

Detects identical complete prompts within a batch and avoids redundant processing where the implementation can safely reuse results.

**Adaptive inference**

Selects between inference paths using workload conditions and benchmark-guided policies.

## 5. Adaptive prefix selection

The automatic inference mode uses a restricted workload policy.

The current implementation uses the following policy:

```python
AUTO_PREFIX_POLICY = {
    4: {24},
    8: {24},
}
```

The intended workload is currently restricted to:

- Prompt length: 32 tokens
- Generated tokens: 8
- Batch size 4 with shared-prefix length 24
- Batch size 8 with shared-prefix length 24

Other workloads fall back to the adaptive KV path.

The restriction is intentional. An optimization that performs well for one batch size or prefix length should not automatically be applied to every workload.

**Validation qualification:** Earlier multi-session experiments supported both selected workloads. A subsequent implementation change and confirmation run continued to support batch 8 / prefix 24, while batch 4 / prefix 24 was inconclusive in that latest run. Further confirmation is needed before treating both workload selections as equally established.

## 6. Adaptive exact-prompt deduplication

The adaptive KV inference path detects identical complete prompts within a batch.

Deduplication is activated only when the duplicate rate reaches a configured threshold.

Current default:

```python
dedup_threshold = 0.75
```

The threshold was selected experimentally to balance potential computation reuse against deduplication overhead.

The mechanism uses exact prompt equality rather than approximate similarity.

The threshold should be regarded as a workload-dependent experimental setting, not a universal optimum.

## 7. Benchmark methodology and results

### Experimental environment

- GPU: NVIDIA GeForce RTX 2050
- GPU memory: 4 GB
- Model: RCT 100M
- Prompt length: 32 tokens
- Generated tokens: 8
- Timed repetitions: 30 per workload per session
- Methods: KV generation and automatic adaptive inference

The benchmark suite uses paired measurements and repeated sessions to evaluate performance differences.

Speedup is defined as:

`KV latency / AUTO latency`

Therefore:

- Speedup greater than 1.0 means AUTO is faster.
- Speedup below 1.0 means KV is faster.
- A confidence interval containing 1.0 is treated as inconclusive.

### Multi-session hierarchical bootstrap

Eight independent benchmark sessions were analyzed using hierarchical bootstrap resampling:

- 10,000 bootstrap iterations.
- Random seed: 42.
- Sessions resampled at the session level.
- Repetitions resampled within selected sessions.
- 95% percentile confidence intervals.

Results:

| Batch | Prefix | Median speedup | Hierarchical 95% CI | Interpretation |
|---:|---:|---:|---:|---|
| 2 | 8 | 1.0024x | [0.9659, 1.0257] | Inconclusive |
| 2 | 16 | 0.9916x | [0.9542, 1.0287] | Inconclusive |
| 2 | 24 | 1.0121x | [0.9719, 1.0543] | Inconclusive |
| 4 | 8 | 1.0118x | [0.9852, 1.0442] | Inconclusive |
| 4 | 16 | 1.0213x | [0.9841, 1.1219] | Inconclusive |
| 4 | 24 | 1.0862x | [1.0217, 1.1478] | AUTO faster |
| 8 | 8 | 0.9996x | [0.9808, 1.0191] | Inconclusive |
| 8 | 16 | 0.9867x | [0.9698, 1.0099] | Inconclusive |
| 8 | 24 | 1.1443x | [1.1068, 1.1780] | AUTO faster |

These results provide workload-specific evidence of improvements for batch 4 / prefix 24 and batch 8 / prefix 24 in the eight-session dataset.

They do not establish general speedups across arbitrary workloads.

### Latest post-implementation confirmation

A subsequent benchmark was performed after modifying the shared-prefix cache-copy implementation.

| Batch | Prefix | KV median (ms) | AUTO median (ms) | Speedup | 95% CI |
|---:|---:|---:|---:|---:|---:|
| 4 | 24 | 127.091 | 124.688 | 1.0049x | [0.9783, 1.0405] |
| 8 | 24 | 142.745 | 130.612 | 1.0781x | [1.0539, 1.1246] |

In this latest run:

- Batch 8 / prefix 24 showed a statistically supported improvement.
- Batch 4 / prefix 24 was inconclusive.
- The other tested workloads were also inconclusive.

This is a confirmation of workload-specific behavior, not proof that the cache-copy modification caused the observed performance difference. A controlled before-and-after experiment is still required to isolate the effect of that code change.

### Benchmark interpretation

The results demonstrate why adaptive inference policies should be conservative.

A workload-specific optimization should only be selected when repeated measurements provide sufficient evidence. Otherwise, the implementation should fall back to a reliable baseline inference path.

The reported confidence intervals are unadjusted for testing multiple workloads and should be interpreted accordingly.

## 8. GPU profiling and memory analysis

RCT inference workloads were profiled using PyTorch Profiler with CPU and CUDA activities, shape recording, and memory profiling.

Profiled workload:

- Batch size: 8
- Prompt length: 32 tokens
- Shared prefix: 24 tokens
- New tokens: 8

The profiler traces include:

- KV inference
- Shared-prefix inference
- Automatic inference

The profiling investigation examined:

- Attention tensor shapes.
- Key/value cache operations.
- Tensor copies and concatenation.
- CUDA operator activity.
- Memory allocation behavior.

### Initial observations

The shared-prefix path processes the non-shared suffix separately from the common prefix.

Profiling also identified cache-copy operations and concatenation allocations that may be worth investigating in future optimization work.

Approximate peak allocated GPU memory during the profiled workloads:

| Mode | Peak allocated memory |
|---|---:|
| KV | 884.666 MB |
| Prefix | 885.541 MB |
| AUTO | 885.541 MB |

These are profiler-observed measurements for the tested workload, not general memory requirements for every configuration.

The investigation suggests possible opportunities to reduce intermediate allocation overhead, but no general memory reduction or additional speedup is claimed.

## 9. Correctness and validation

The project includes tests covering Transformer behavior, KV caching, shared-prefix generation, adaptive inference and computation reuse.

The adaptive prefix gate test reports:

```text
RCT ADAPTIVE PREFIX GATE TEST: PASS (12/12)
```

Validation includes:

- Shared-prefix detection.
- Prefix-versus-KV inference selection.
- Automatic fallback for unsupported workloads.
- Output agreement between automatic and KV generation.
- Prompt-length and generation-length restrictions.
- Gate activation and bypass statistics.

Additional inference tests have checked standard generation, KV generation, shared-prefix generation and exact-prompt behavior.

Floating-point operations may produce small numerical differences even when outputs are numerically equivalent. Tests should distinguish numerical tolerance from bitwise equality.

## 10. Hardware and software environment

Current development environment:

| Component | Configuration |
|---|---|
| Operating system | Windows |
| Python | 3.12.7 |
| GPU | NVIDIA GeForce RTX 2050 |
| GPU memory | 4 GB |
| PyTorch | 2.14.1+cu126 |
| CUDA build | 12.6 |
| Development shell | Windows PowerShell |

The project has primarily been developed and tested on a single local GPU.

Multi-GPU and cross-architecture validation remain future work.

## 11. Repository structure

Representative project structure:

```text
rct-project/
│
├── config.py
├── train.py
├── train_100m.py
├── train_100m_continue.py
├── inference.py
├── README.md
├── requirements.txt
│
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
│
├── benchmarks/
│   ├── test_rct_adaptive_prefix_gate.py
│   ├── benchmark_rct_adaptive_prefix_gate.py
│   └── ...
│
├── tests/
│   ├── test_kv_cache.py
│   ├── test_shared_prefix.py
│   ├── test_adaptive_reuse.py
│   └── ...
│
├── checkpoints/          # Excluded from Git
├── data/                 # Excluded from Git
└── results/              # Excluded from Git
```

The structure is representative rather than an exhaustive listing of every experimental module.

## 12. Reproducibility

The source code and tests are maintained in Git.

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

Install the project dependencies:

```powershell
pip install -r requirements.txt
```

Obtain a compatible model checkpoint separately, then configure the checkpoint path for the desired experiment.

Run the relevant tests and benchmarks using their documented commands.

**Reproducibility note:** The current dependency file and experiment scripts should be checked together before claiming a complete clean-machine reproduction. CUDA-enabled PyTorch installation may require a hardware-compatible installation command.

## 13. Limitations

RCT is an experimental research and engineering prototype.

Current results do not establish:

- General speedups across arbitrary inference workloads.
- Performance improvements across different GPU architectures.
- Better language-model quality than comparable baselines.
- A universal reduction in computational complexity.
- Production readiness.
- Superiority over established inference frameworks.

The current benchmark evidence is workload-specific and primarily based on one GPU.

The adaptive policy requires further validation across prompt lengths, generation lengths, batch sizes, and hardware configurations.

## 14. Future development

Planned engineering and research directions:

1. Consolidate documentation for individual computation-reuse mechanisms.
2. Expand regression testing across all inference modes.
3. Revalidate the final adaptive prefix policy using additional benchmark sessions.
4. Evaluate more prompt lengths and generation lengths.
5. Investigate GPU memory allocation and cache-copy overhead.
6. Compare against standard Transformer inference under identical conditions.
7. Evaluate model quality and generation behavior.
8. Explore modern C++ and CUDA kernel implementations.
9. Investigate cross-GPU and multi-GPU behavior.

---

## Project summary

**RCT — Reusable Computation Transformer**

An independently developed Transformer inference research project exploring exact computation reuse, adaptive deduplication, KV caching, shared-prefix generation, and benchmark-guided inference selection.

The core 100M model and experimental inference mechanisms are implemented. Work continues on performance validation, reproducibility, GPU profiling, and broader engineering evaluation.

**Repository:** https://github.com/ManishaChindam0309/RCT