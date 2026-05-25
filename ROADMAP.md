# Roadmap

## Stage 1: Core Framework

- [x] CUDA driver interception via `LD_PRELOAD` hook
- [x] Deterministic memory layout via VMM interposition
- [x] CUDA graph capture, save, and reload APIs
- [x] Automatic kernel binary extraction and reload
- [x] Kernel node attributes saving and restoration for cuBLAS on Hopper+

## Stage 2: Performance Optimization

- [x] Memory preallocation fast path (pointer-bump allocations)
- [x] Binary graph format (`.cugraph`) replacing JSON text parsing (84x faster Phase 1)
- [x] Topology-grouped template sharing across batch sizes
- [x] Graph manifest generation with automatic template assignment
- [x] On-demand exec update via `cuGraphExecUpdate` at replay time

## Stage 3: Multi-GPU and Communication Kernel Support

- [x] NVSHMEM kernel auto-initialization at load time
- [x] DeepEP multi-segment fatbin capture and pre-linking
- [ ] Release NVSHMEM stub layer for single-GPU offline template capture

## Stage 4: vLLM/SGLang Integration

- [x] Sync with latest vLLM release
    - [x] EP on vLLM with DeepEP
    - [ ] TP on vLLM
    - [x] Quantized MoE with DeepGemm
- [x] Drop-in integration layer for CUDA graph persistence in vLLM
- [x] Sync with latest SGLang release
    - [ ] EP on SGLang
    - [ ] TP on SGLang

## Stage 5: Disaggregated and Large-Scale Serving

- [ ] Prefill-decode (PD) disaggregated serving engine with Foundry-accelerated cold start
- [ ] Cross-node expert parallelism at scale (multi-host NVLink + InfiniBand)
- [ ] End-to-end fast initialization with weight transfer via RDMA
- [ ] NVIDIA Dynamo compatibility for elastic LLM serving at scale

## Stage 6: Instant-On Elastic Expert Parallelism

- [ ] Dynamic EP size adjustment by reusing templates and updating rank-dependent communication state
- [ ] Instant EP scale-up/scale-down without full graph recapture
