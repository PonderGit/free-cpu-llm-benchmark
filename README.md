# Free CPU LLM Benchmark

A generic, zero-cost CPU benchmark for small open-weight language models using GitHub Actions public hosted runners.

## Scope

- CPU only
- No paid API
- No GPU / ZeroGPU
- No private or unpublished research material
- No production deployment changes

The workflow downloads a public ~1.5B GGUF model, runs prompt-processing / generation throughput benchmarks, and performs two small A/B smoke tests. Results are uploaded as GitHub Actions artifacts.

This repository is intentionally generic and is used only to test whether free CPU inference is practical for small, bounded evaluation workloads.
