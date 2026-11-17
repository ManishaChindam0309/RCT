# RCT — Reusable Computation Transformer

RCT (Reusable Computation Transformer) is an experimental Transformer architecture and inference framework focused on reducing repeated computation through reusable computation paths, exact prompt deduplication, KV caching, and shared-prefix generation.

The project currently includes a 100M-parameter Transformer model and several experimental inference optimizations.

## 1. Project status

**Current milestone: RCT 100M inference optimization and validation**

| Component | Status |
|---|---|
| RCT 100M model | Implemented |
| Model checkpoint | Available |
| Standard autoregressive generation | Implemented |
| KV-cache generation | Implemented |
| Shared-prefix generation | Implemented |
| Automatic shared-prefix detection | Implemented |
| Exact complete-prompt deduplication | Implemented |
| Adaptive deduplication gate | Implemented |
| Benchmark-guided adaptive prefix gate | Implemented |
| Adaptive prefix gate tests | Passing 12/12 |
| Prefix and KV performance benchmarks | Completed |
| Broad multi-GPU validation | Not yet completed |

## 2. Model configuration

The current RCT 100M model has:

| Parameter | Value |
|---|---:|
| Parameters | 99,028,992 |
| Vocabulary size | 8,192 |
| Hidden size | 512 |
| Transformer layers | 30 |
| Attention heads | 8 |
| FFN intermediate size | 2,048 |
| Maximum sequence length | 512 |

The current continued-training checkpoint is:

`checkpoints/rct100m_continued/best_model.pt`

The checkpoint contains model weights, training step, validation loss, and model configuration.

Recorded checkpoint information:
- Training step: 1,500
- Validation loss: approximately 5.9246

## 3. Inference modes

The unified inference interface is implemented in `inference.py`.

Five generation modes are available:

| Mode | Description |
|---|---|
| `standard` | Full-context autoregressive generation |
| `kv` | Incremental generation using KV caching |
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

## 4. Current adaptive prefix policy

The `auto` mode uses a deliberately restricted policy based on measured RTX 2050 benchmark results.

Current configuration:

```python
AUTO_PREFIX_POLICY = {
    4: {16, 24},
    8: {24},
}
```

The gate is currently restricted to:
- Prompt length: 32 tokens
- Generated tokens: 8
- Batch size 4 with shared-prefix length 16 or 24
- Batch size 8 with shared-prefix length 24

Other workloads use adaptive KV generation.

This restriction is intentional. A benchmark result from one workload should not automatically be assumed to generalize to every batch size, prefix length, or generation length.

## 5. Adaptive exact-prompt deduplication

The adaptive KV inference path detects identical complete prompts within a batch.

It only activates deduplication when the duplicate rate reaches the configured threshold.

Current default:

```python
dedup_threshold = 0.75
```

This optimization is intended to avoid unnecessary deduplication overhead when prompts are mostly unique.

Deduplication is exact: it does not treat merely similar prompts as duplicates.

## 6. Benchmark findings

### Adaptive prefix gate

The integrated benchmark compared KV generation with automatic adaptive inference.

Test conditions:
- GPU: NVIDIA GeForce RTX 2050
- Model: RCT 100M
- Prompt length: 32
- Generated tokens: 8
- 30 timed paired repetitions per workload
- Randomized method order
- Paired bootstrap confidence intervals

Results:

| Batch | Prefix | Auto speedup vs KV | Conclusion |
|---:|---:|---:|---|
| 2 | 8 | 0.9595x | Inconclusive |
| 2 | 16 | 0.9990x | Inconclusive |
| 2 | 24 | 1.0117x | Inconclusive |
| 4 | 8 | 1.1311x | Inconclusive |
| 4 | 16 | 1.2412x | Auto faster |
| 4 | 24 | 1.2923x | Auto faster |
| 8 | 8 | 1.0047x | Inconclusive |
| 8 | 16 | 0.9719x | Inconclusive |
| 8 | 24 | 1.1741x | Auto faster |

The automatic gate therefore retains only the three workloads with statistically supported speedups.

Benchmark output:

`results/rct_adaptive_prefix_gate_confirmation/run_20261106T163434Z/`

Files:
- `raw_measurements.csv`
- `summary.csv`

### Adaptive deduplication

The duplicate-rate threshold was evaluated using multiple batch sizes and duplicate ratios.

A 75% duplicate threshold was selected because it showed consistent speedups across the tested batch sizes.

The threshold is an experimentally selected setting, not a universal optimum.

## 7. Validation

The adaptive prefix gate test is:

```powershell
python benchmarks\test_rct_adaptive_prefix_gate.py
```

Latest result:

```text
RCT ADAPTIVE PREFIX GATE TEST: PASS (12/12)
```

The test verifies:
- Shared-prefix detection
- Correct prefix-versus-KV selection
- Exact output agreement between automatic and KV modes
- Unsupported prompt-length fallback
- Unsupported generation-length fallback
- No-shared-prefix fallback
- Gate activation and bypass statistics

## 8. Hardware and software environment

Current development environment:

- Operating system: Windows
- Python: 3.12.7
- GPU: NVIDIA GeForce RTX 2050
- GPU memory: 4 GB
- PyTorch: 2.14.1+cu126
- CUDA: 12.6

The project is currently developed and tested locally in Windows PowerShell.

## 9. Project structure

```text
rct-project/
│
├── config.py
├── train.py
├── inference.py
├── test_rct.py
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
│   ├── test_rct_inference.py
│   ├── test_rct_adaptive_gate.py
│   ├── test_rct_adaptive_prefix_gate.py
│   ├── benchmark_rct_adaptive_prefix_gate.py
│   └── ...
│
├── checkpoints/
│   └── rct100m_continued/
│       └── best_model.pt
│
└── results/
```

The tree is representative rather than an exhaustive list of every experimental module and benchmark.

## 10. Next development milestones

Recommended next steps:

1. Update and consolidate documentation for the individual RCT computation-reuse mechanisms.
2. Run regression tests for all existing inference modes.
3. Benchmark the final adaptive policy again to confirm its performance after integration.
4. Evaluate additional prompt lengths and generation lengths.
5. Measure memory usage and throughput across different batch sizes.
6. Compare RCT against a standard Transformer baseline under identical hardware and workload conditions.
7. Evaluate model quality through validation loss and generation-quality tests.
8. Investigate whether the reusable-computation mechanisms can deliver reliable benefits beyond exact duplicate prompts and shared prefixes.

## 11. Research scope and limitations

RCT is an experimental project. The current results establish that selected inference optimizations work under specific measured conditions.

They do not yet establish:
- General performance improvements across arbitrary workloads.
- Improvements across different GPUs.
- Better language-model quality than comparable baselines.
- A universal reduction in computational complexity.
- Production readiness.

The next research phase should focus on reproducibility, broader workload testing, model-quality evaluation, and comparisons against carefully controlled baselines.

---

**RCT development status:** Core 100M model implemented; inference optimizations implemented; benchmark-guided adaptive prefix gate validated; broader research evaluation remains ongoing.