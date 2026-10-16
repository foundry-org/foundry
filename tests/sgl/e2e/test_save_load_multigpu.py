# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Multi-GPU end to end: native graph run, Foundry SAVE, Foundry LOAD, with the
serve settings of the recipes in recipe/sglang/. Assertions:

  (a) greedy token ids of SAVE and LOAD identical to native (8 prompts x 64 tokens, temperature 0);
  (b) LOAD median TPOT not more than TPOT_REL_TOL above native (32 requests, concurrency 8,
      after a throwaway bench on each engine; one-sided, a slow native run is not a failure);
  (c) LOAD graph restore ("[Foundry] Loaded N SGLang graphs in Xs", max over ranks) below
      RESTORE_RATIO x the native decode capture ("Capture ... graph end ... elapsed=Xs", max over ranks);
  (d) Foundry active in the launcher and every scheduler, no "[HOOK] ERROR";
  (e) SAVE wrote one archive directory per rank, and SAVE final_alloc_offset == LOAD after_load_all_graphs per rank.

Cases (args copied from the recipe lines cited in CASES; a recipe change must show up here):
  q35_tp2 / q35_ep2 / q35_tpep2   Qwen3.5-35B-A3B on 2 GPUs (serve_qwen3.5-35ba3b.sh), real weights
  q35_122b_fp8_ep4                Qwen3.5-122B-A10B-FP8 ep4 (serve_qwen3.5-122ba10b.sh), xlarge, dummy weights
  dsv4_flash_fp8_ep4              DeepSeek-V4-Flash-FP8 ep4 (serve_deepseek-v4-flash.sh), xlarge, dummy weights
Native, SAVE and LOAD all run with the variables of recipe/sglang/set_env.sh (a value
already exported wins, as there), plus NCCL_CUMEM_ENABLE=0 and NCCL_NVLS_ENABLE=0.

Qwen3.5 (hybrid gated DeltaNet) and DeepSeek-V4 cannot capture full-backend prefill graphs in plain SGLang (the GDN
backend's out-of-graph metadata is decode-only; the DSV4 backend rejects EXTEND), so every case here is decode-only.

    pytest tests/sgl/e2e/test_save_load_multigpu.py -m "slow and multigpu and not xlarge" -v -s   # 35B, 2 GPUs
    pytest tests/sgl/e2e/test_save_load_multigpu.py -m "slow and xlarge" -v -s                    # 4 GPUs
    pytest tests/sgl/e2e/test_save_load_multigpu.py -m slow -k q35_ep2 -v -s                      # one case

Overrides: FOUNDRY_E2E_MAXBS (decode max batch size per rank, default 64; the recipes use 256),
FOUNDRY_E2E_WEIGHTS=dummy|real (default per case).

Runs on whichever route the installed sglang has (_sglang_helpers.foundry_launch): the --cuda-graph-persistence
flags when sglang has the Foundry adapter, the plugin's environment variable otherwise.
"""

import os
import re

import pytest

pytest.importorskip("sglang")

from _sglang_helpers import ACTIVE as _ACTIVE  # noqa: E402
from _sglang_helpers import foundry_toml, serve  # noqa: E402
from sglang.benchmark.serving import run_benchmark  # noqa: E402
from sglang.test.test_utils import (  # noqa: E402
    DEFAULT_URL_FOR_TEST,
    get_benchmark_args,
)

pytestmark = pytest.mark.multigpu

# H200, 3 real-weight runs per 35B case: tp2 within 1.3%; ep2 -0.4%, +1.2%, +12.3%.
TPOT_REL_TOL = 0.15
# 35B on H200: restore 0.26 s (tp2) / 0.34 s (ep2) against a multi-second native capture.
RESTORE_RATIO = 0.25
LAUNCH_TIMEOUT_S = 3600
MAXBS = int(os.environ.get("FOUNDRY_E2E_MAXBS", "64"))

_PROMPTS = [
    "The capital of France is",
    "1, 2, 3, 4, 5,",
    "def fibonacci(n):",
    "Explain why the sky is blue in one sentence.",
    "Translate to German: The weather is nice today.",
    "The three primary colors are",
    "Write a haiku about the ocean.",
    "In 1969, the first person to walk on the moon was",
]
_SAMPLING = {"temperature": 0, "max_new_tokens": 64, "ignore_eos": True}

_LOADED = re.compile(r"\[Foundry\] Loaded (\d+) SGLang graphs in ([0-9.]+)s")
_CAPTURED = re.compile(
    r"Capture (?:target decode CUDA graph|cuda graph) end\..*?(?:elapsed=|Time elapsed: )([0-9.]+) ?s"
)
_SAVE_OFFSET = re.compile(r"\[Foundry\] SGLang final_alloc_offset=(\d+)")
_LOAD_OFFSET = re.compile(r"\[Foundry\] SGLang alloc_offset\[after_load_all_graphs\]=(\d+)")

# recipe/sglang/set_env.sh, sourced by every recipe for native, --save and --load alike.
_SET_ENV = {
    "TORCHDYNAMO_DISABLE": "1",
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
    "NVSHMEM_DISABLE_NCCL": "1",
    "EP_DISABLE_GIN": "1",
    "NCCL_IB_DISABLE": "1",
    "NVSHMEM_REMOTE_TRANSPORT": "none",
    "PYTORCH_CUDA_ALLOC_CONF": "pinned_use_cuda_host_register:True,pinned_num_register_threads:8",
}
# NCCL buffers through the plain allocator on the native engine too (Foundry pins them on SAVE / LOAD).
_NCCL_ENV = {"NCCL_CUMEM_ENABLE": "0", "NCCL_NVLS_ENABLE": "0"}
_EP_ENV = {"SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "512", "NVSHMEM_QP_DEPTH": "2048"}

# Common to every recipe: --trust-remote-code, --disable-radix-cache, the graph flags and the all-reduce pair.
_COMMON_ARGS = [
    "--trust-remote-code",
    "--disable-radix-cache",
    "--disable-cuda-graph-padding",
    "--cuda-graph-backend-prefill", "disabled",
    "--disable-custom-all-reduce", "--enable-torch-symm-mem",
    "--random-seed", "0",  # test only: greedy parity
]  # fmt: skip
_DEEPEP = [
    "--moe-a2a-backend", "deepep", "--deepep-mode", "low_latency", "--moe-runner-backend", "deep_gemm",
]  # fmt: skip


def _ep(n):
    # DP attention over all n ranks, as the recipes' epN case (--attn-dp-size replaces the
    # deprecated --dp-size N --enable-dp-attention pair).
    return ["--tp-size", str(n), "--attn-dp-size", str(n), "--ep-size", str(n)]


def _decode_max(maxbs):
    return ["--cuda-graph-max-bs-decode", str(maxbs)]


def _decode_step(n, maxbs):
    """tpep: max bs N * MAXBS, captured sizes N, 2N, ..., N * MAXBS."""
    sizes = [str(b) for b in range(n, n * maxbs + 1, n)]
    return ["--cuda-graph-max-bs-decode", str(n * maxbs), "--cuda-graph-bs-decode", *sizes]


def _case(case_id, model, gpus, args, graphs, *, memfrac, weights, env=None, marks=()):
    return pytest.param(
        {
            "id": case_id,
            "model": model,
            "gpus": gpus,
            "args": [*args, *graphs, "--mem-fraction-static", str(memfrac)],
            "weights": weights,
            "env": {**_NCCL_ENV, **(env or {})},
        },
        id=case_id,
        marks=marks,
    )


_Q35 = "Qwen/Qwen3.5-35B-A3B"  # serve_qwen3.5-35ba3b.sh:38
CASES = [
    # serve_qwen3.5-35ba3b.sh (common args :16, :39-42): tp2 TOPO/MEMFRAC :18; ep2 TOPO :20-22,
    # env :23, MEMFRAC :24; tpep2 TOPO :26-28, GRAPHS :29-30, env :31, MEMFRAC :32.
    _case("q35_tp2", _Q35, 2, ["--tp-size", "2"], _decode_max(MAXBS),
          memfrac=0.8, weights="real"),
    _case("q35_ep2", _Q35, 2,
          _ep(2) + _DEEPEP + ["--chunked-prefill-size", "256", "--attention-backend", "fa3",
                              "--max-running-requests", "512"],
          _decode_max(MAXBS), memfrac=0.8, weights="real", env=_EP_ENV),
    _case("q35_tpep2", _Q35, 2,
          ["--tp-size", "2", "--ep-size", "2"] + _DEEPEP
          + ["--chunked-prefill-size", "256", "--attention-backend", "fa3", "--max-running-requests", "512"],
          _decode_step(2, MAXBS), memfrac=0.8, weights="real", env=_EP_ENV),
    # serve_qwen3.5-122ba10b.sh ep4 (common args :49-53): model :8, mamba :17 / :51,
    # GRAPHS :18-19, TOPO :22-24, env :39, MEMFRAC :43.
    _case("q35_122b_fp8_ep4", "Qwen/Qwen3.5-122B-A10B-FP8", 4,
          _ep(4) + _DEEPEP + ["--chunked-prefill-size", "256", "--attention-backend", "fa3",
                              "--max-running-requests", "1024", "--max-mamba-cache-size", "256"],
          _decode_max(MAXBS), memfrac=0.6, weights="dummy", env=_EP_ENV, marks=pytest.mark.xlarge),
    # serve_deepseek-v4-flash.sh ep4 (common args :24, :29-30): env :19-20, model :23,
    # MEMFRAC :25, TOPO :26-28, GRAPHS :29.
    # KV page 256: the prefill chunk divided by dp must stay a multiple of it.
    _case("dsv4_flash_fp8_ep4", "sgl-project/DeepSeek-V4-Flash-FP8", 4,
          _ep(4) + _DEEPEP + ["--chunked-prefill-size", "2048", "--max-running-requests", "1024"],
          _decode_max(MAXBS), memfrac=0.7, weights="dummy",
          env={**_EP_ENV, "SGLANG_DSV4_FP4_EXPERTS": "0"}, marks=pytest.mark.xlarge),
]  # fmt: skip


def _engine_env(case):
    env = {k: os.environ.get(k, v) for k, v in _SET_ENV.items()}
    env.update(case["env"])
    return env


def _toml(path, mode, workspace):
    # 4 GB scratch: TP/EP dist init takes 1.8-2.9 GB before the region opens (1 GB default is too small).
    return foundry_toml(path, mode, workspace, scratch_space_size="4096MB")


def _server_args(case):
    weights = os.environ.get("FOUNDRY_E2E_WEIGHTS", case["weights"])
    args = [*_COMMON_ARGS, *case["args"]]
    if weights == "dummy":
        args += ["--load-format", "dummy"]
    return args


def _bench_tpot(model, out_path):
    args = get_benchmark_args(
        base_url=DEFAULT_URL_FOR_TEST,
        dataset_name="random-ids",
        tokenizer=model,
        num_prompts=32,
        random_input_len=256,
        random_output_len=128,
        max_concurrency=8,
        seed=0,
    )
    # Throwaway pass first: the first requests of a fresh engine carry one-time
    # costs (lazy kernel loading, JIT) that are not what (b) measures.
    args.output_file = str(out_path.with_suffix(".warm.jsonl"))
    run_benchmark(args)
    args.output_file = str(out_path)
    return run_benchmark(args)["median_tpot_ms"]


def _serve(tmp_path, name, case, toml, bench, mode=None):
    def bench_tpot():
        return _bench_tpot(case["model"], tmp_path / f"bench_{name}.jsonl")

    responses, tpot_ms, log = serve(
        tmp_path,
        name,
        case["model"],
        _server_args(case),
        _PROMPTS,
        _SAMPLING,
        mode=mode,
        toml=toml,
        env=_engine_env(case),
        launch_timeout=LAUNCH_TIMEOUT_S,
        request_timeout=600,
        while_up=bench_tpot if bench else None,
    )
    return {"ids": [r["output_ids"] for r in responses], "tpot_ms": tpot_ms, "log": log}


@pytest.mark.parametrize("case", CASES)
def test_save_then_load_multigpu(case, tmp_path):
    workspace = tmp_path / "archive"

    # Native first: also warms the JIT caches before SAVE.
    native = _serve(tmp_path, "native", case, None, bench=True)
    save = _serve(
        tmp_path,
        "save",
        case,
        _toml(tmp_path / "save.toml", "save", workspace),
        bench=False,
        mode="save",
    )
    load = _serve(
        tmp_path,
        "load",
        case,
        _toml(tmp_path / "load.toml", "load", workspace),
        bench=True,
        mode="load",
    )

    # (d) launcher + one scheduler per rank.
    assert not _ACTIVE.search(native["log"]), "Foundry must be inert without being enabled"
    for run in (save, load):
        assert "[HOOK] ERROR" not in run["log"], "hook reported an error"
        assert len(set(_ACTIVE.findall(run["log"]))) >= case["gpus"] + 1, (
            "Foundry did not load in every process"
        )

    # (e)
    rank_dirs = sorted(p.name for p in workspace.glob("rank_*"))
    assert len(rank_dirs) == case["gpus"], rank_dirs
    save_offsets = sorted(int(x) for x in _SAVE_OFFSET.findall(save["log"]))
    load_offsets = sorted(int(x) for x in _LOAD_OFFSET.findall(load["log"]))
    assert len(save_offsets) == case["gpus"] and save_offsets == load_offsets, (
        save_offsets,
        load_offsets,
    )

    # (a)
    assert save["ids"] == native["ids"]
    assert load["ids"] == native["ids"]

    # (c) every rank restores the same decode set; the slowest rank gates the start,
    # on LOAD as on the native capture it replaces.
    loaded = [(int(n), float(t)) for n, t in _LOADED.findall(load["log"])]
    assert len(loaded) == case["gpus"] and len({n for n, _ in loaded}) == 1, loaded
    restore_s = max(t for _, t in loaded)
    captured = [float(t) for t in _CAPTURED.findall(native["log"])]
    assert captured, "no decode capture time in the native log"
    capture_s = max(captured)
    assert restore_s < RESTORE_RATIO * capture_s, (
        f"LOAD restore {restore_s:.3f}s, native capture {capture_s:.3f}s (bound x{RESTORE_RATIO})"
    )
    print(
        f"[e2e {case['id']}] native decode capture {capture_s:.3f} s, LOAD restore {restore_s:.3f} s "
        f"({loaded[0][0]} graphs/rank, ratio {restore_s / capture_s:.4f}); median TPOT native "
        f"{native['tpot_ms']:.3f} ms, LOAD {load['tpot_ms']:.3f} ms"
    )

    # (b)
    # One-sided: LOAD must not be slower than native by more than the tolerance;
    # a stalled native bench must not fail the test.
    rel = (load["tpot_ms"] - native["tpot_ms"]) / native["tpot_ms"]
    assert rel <= TPOT_REL_TOL, (
        f"median TPOT native {native['tpot_ms']:.3f} ms, LOAD {load['tpot_ms']:.3f} ms"
    )
