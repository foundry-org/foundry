# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Multi-GPU end to end: native graph run, Foundry SAVE, Foundry LOAD, for the
models the SGLang base test (test/registered/e2e/plugins/test_foundry_graph_persistence.py
in SGLang, Qwen3.5-2B on 1 GPU) does not cover. Same assertions as the base test:

  (a) greedy token ids of SAVE and LOAD identical to native (8 prompts x 64 tokens, temperature 0);
  (b) LOAD median TPOT within TPOT_REL_TOL of native (32 requests, concurrency 8);
  (c) LOAD graph restore ("[Foundry] Loaded N SGLang graphs in Xs", max over ranks) below the case's bound;
  (d) plugin active in the launcher and every scheduler, no "[HOOK] ERROR";
  (e) SAVE wrote one archive directory per rank, and SAVE final_alloc_offset == LOAD after_load_all_graphs per rank.

Tiers:
  default   Qwen3.5-35B-A3B tp2 and ep2 (2 GPUs). Real weights by default (FOUNDRY_E2E_WEIGHTS=dummy to skip the
            70 GB download); correctness on dummy weights compares deterministic garbage, still a valid parity check.
  xlarge    Qwen3.5-122B-A10B-FP8 ep4 and DeepSeek-V4-Flash-FP8 ep4 (4 GPUs, dummy weights by default). Deselected by
            the project's addopts; run with `-m xlarge`.

Qwen3.5 (hybrid gated DeltaNet) and DeepSeek-V4 cannot capture full-backend prefill graphs in plain SGLang (the GDN
backend's out-of-graph metadata is decode-only; the DSV4 backend rejects EXTEND), so every case here is decode-only.
Each case skips when fewer GPUs are visible than it needs.

    pytest tests/integration/sglang/test_sglang_save_load_large.py -v -s                  # 35B tp2 / ep2
    pytest tests/integration/sglang/test_sglang_save_load_large.py -v -s -m xlarge        # 122B / DSV4 ep4
    FOUNDRY_E2E_CASES=q35_ep2 pytest ... -v -s                                            # one case
"""

import os
import re

import pytest

pytest.importorskip("sglang")
torch = pytest.importorskip("torch")

import requests  # noqa: E402
from foundry.integration.sglang import preflight  # noqa: E402
from foundry.integration.sglang.plugin import CONFIG_ENV  # noqa: E402
from sglang.benchmark.serving import run_benchmark  # noqa: E402
from sglang.test.test_utils import (  # noqa: E402
    DEFAULT_URL_FOR_TEST,
    get_benchmark_args,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

# Placeholder until measured over 3 runs per case.
TPOT_REL_TOL = 0.15
LAUNCH_TIMEOUT_S = 3600

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

_ACTIVE = re.compile(r"\[Foundry\] sglang plugin active: pid=(\d+)")
_LOADED = re.compile(r"\[Foundry\] Loaded (\d+) SGLang graphs in ([0-9.]+)s")
_SAVE_OFFSET = re.compile(r"\[Foundry\] SGLang final_alloc_offset=(\d+)")
_LOAD_OFFSET = re.compile(r"\[Foundry\] SGLang alloc_offset\[after_load_all_graphs\]=(\d+)")

# Identical on native, SAVE and LOAD (recipe/sglang/serve_common.sh): NCCL buffers through the plain allocator, and
# no rank-0-only DeepGEMM precompile inside the deterministic range.
_COMMON_ENV = {
    "NCCL_CUMEM_ENABLE": "0",
    "NCCL_NVLS_ENABLE": "0",
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
}
_EP_ENV = {"SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "512", "NVSHMEM_QP_DEPTH": "2048"}


def _tp(n):
    return ["--tp-size", str(n), "--disable-custom-all-reduce", "--enable-torch-symm-mem"]


def _ep(n, chunk, attn):
    return [
        "--tp-size", str(n), "--dp-size", str(n), "--ep-size", str(n), "--enable-dp-attention",
        "--moe-a2a-backend", "deepep", "--deepep-mode", "low_latency", "--moe-runner-backend", "deep_gemm",
        "--enable-torch-symm-mem", "--disable-custom-all-reduce", "--chunked-prefill-size", str(chunk),
        *attn,
    ]  # fmt: skip


def _case(
    case_id,
    model,
    gpus,
    args,
    *,
    decode_max_bs,
    memfrac,
    weights,
    restore_bound_s,
    env=None,
    marks=(),
):
    return pytest.param(
        {
            "model": model,
            "gpus": gpus,
            "args": args,
            "decode_max_bs": decode_max_bs,
            "memfrac": memfrac,
            "weights": weights,
            "restore_bound_s": restore_bound_s,
            "env": {**_COMMON_ENV, **(env or {})},
        },
        id=case_id,
        marks=marks,
    )


# restore_bound_s values are placeholders (to be set from 3 measured runs); decode_max_bs is per DP rank on EP rows.
CASES = [
    _case("q35_tp2", "Qwen/Qwen3.5-35B-A3B", 2, _tp(2),
          decode_max_bs=64, memfrac=0.8, weights="real", restore_bound_s=2.0),
    _case("q35_ep2", "Qwen/Qwen3.5-35B-A3B", 2,
          _ep(2, 256, ["--attention-backend", "fa3"]) + ["--max-running-requests", "128"],
          decode_max_bs=64, memfrac=0.8, weights="real", restore_bound_s=2.0, env=_EP_ENV),
    _case("q35_122b_fp8_ep4", "Qwen/Qwen3.5-122B-A10B-FP8", 4,
          _ep(4, 256, ["--attention-backend", "fa3"]) + ["--max-mamba-cache-size", "1024"],
          decode_max_bs=64, memfrac=0.7, weights="dummy", restore_bound_s=3.0, env=_EP_ENV,
          marks=pytest.mark.xlarge),
    # dsv4 KV page 256: the prefill chunk divided by dp must stay a multiple of it.
    _case("dsv4_flash_fp8_ep4", "sgl-project/DeepSeek-V4-Flash-FP8", 4, _ep(4, 2048, []),
          decode_max_bs=64, memfrac=0.7, weights="dummy", restore_bound_s=3.0,
          env={**_EP_ENV, "SGLANG_DSV4_FP4_EXPERTS": "0"},
          marks=[
              pytest.mark.xlarge,
              # On SGLang main the restored topk_small_batch_cluster_kernel node lacks its cluster dims.
              pytest.mark.xfail(reason="LOAD: cuGraphAddKernelNode error 912 on a cluster-launch kernel", strict=False),
          ]),
]  # fmt: skip


def _selected(case_id):
    wanted = os.environ.get("FOUNDRY_E2E_CASES")
    return wanted is None or case_id in wanted.split(",")


def _toml(path, mode, workspace):
    # 4 GB scratch: TP/EP dist init takes 1.8-2.9 GB before the region opens (1 GB default is too small).
    path.write_text(
        f'mode = "{mode}"\nbase_addr = 0x600000000000\nregion_size = "256GB"\n'
        f'workspace_root = "{workspace}"\nscratch_space_size = "4096MB"\n'
    )
    return str(path)


def _server_args(case):
    weights = os.environ.get("FOUNDRY_E2E_WEIGHTS", case["weights"])
    args = [
        *case["args"],
        "--cuda-graph-max-bs-decode", str(case["decode_max_bs"]),
        "--disable-cuda-graph-padding",
        "--cuda-graph-backend-prefill", "disabled",
        "--mem-fraction-static", str(case["memfrac"]),
        "--random-seed", "0",
    ]  # fmt: skip
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
    args.output_file = str(out_path)
    return run_benchmark(args)["median_tpot_ms"]


def _serve(tmp_path, name, case, foundry_toml, bench):
    launch_env = {k: v for k, v in os.environ.items() if k != CONFIG_ENV}
    launch_env.update(case["env"])
    if foundry_toml is not None:
        launch_env[CONFIG_ENV] = foundry_toml
        try:
            preflight.run(foundry_toml, env=launch_env)
        except preflight.PreflightError as exc:
            pytest.fail(f"Foundry preflight: {exc}")
    result = {}
    with open(tmp_path / f"{name}.out", "w") as out, open(tmp_path / f"{name}.err", "w") as err:
        process = popen_launch_server(
            case["model"],
            DEFAULT_URL_FOR_TEST,
            timeout=LAUNCH_TIMEOUT_S,
            other_args=_server_args(case),
            env=launch_env,
            return_stdout_stderr=(out, err),
        )
        try:
            ids = []
            for prompt in _PROMPTS:
                r = requests.post(
                    DEFAULT_URL_FOR_TEST + "/generate",
                    json={"text": prompt, "sampling_params": _SAMPLING},
                    timeout=600,
                )
                r.raise_for_status()
                ids.append(r.json()["output_ids"])
            result["ids"] = ids
            if bench:
                result["tpot_ms"] = _bench_tpot(case["model"], tmp_path / f"bench_{name}.jsonl")
        finally:
            terminate_and_kill_process_tree(process)
    result["log"] = (tmp_path / f"{name}.err").read_text() + (tmp_path / f"{name}.out").read_text()
    return result


@pytest.mark.parametrize("case", CASES)
def test_save_then_load_large(case, tmp_path, request):
    case_id = request.node.callspec.id
    if not _selected(case_id):
        pytest.skip(f"not in FOUNDRY_E2E_CASES={os.environ['FOUNDRY_E2E_CASES']}")
    if torch.cuda.device_count() < case["gpus"]:
        pytest.skip(f"needs {case['gpus']} GPUs, {torch.cuda.device_count()} visible")
    workspace = tmp_path / "archive"

    # Native first: also warms the JIT caches before SAVE.
    native = _serve(tmp_path, "native", case, None, bench=True)
    save = _serve(
        tmp_path, "save", case, _toml(tmp_path / "save.toml", "save", workspace), bench=False
    )
    load = _serve(
        tmp_path, "load", case, _toml(tmp_path / "load.toml", "load", workspace), bench=True
    )

    # (d) launcher + one scheduler per rank.
    assert not _ACTIVE.search(native["log"]), "plugin must be inert without the env var"
    for run in (save, load):
        assert "[HOOK] ERROR" not in run["log"], "hook reported an error"
        assert len(set(_ACTIVE.findall(run["log"]))) >= case["gpus"] + 1, (
            "plugin did not load in every process"
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

    # (c) every rank restores the same decode set; the slowest rank gates the start.
    loaded = [(int(n), float(t)) for n, t in _LOADED.findall(load["log"])]
    assert len(loaded) == case["gpus"] and len({n for n, _ in loaded}) == 1, loaded
    restore_s = max(t for _, t in loaded)
    assert restore_s < case["restore_bound_s"], (
        f"restore {restore_s:.3f}s, bound {case['restore_bound_s']}s"
    )

    # (b)
    rel = abs(load["tpot_ms"] - native["tpot_ms"]) / native["tpot_ms"]
    assert rel <= TPOT_REL_TOL, (
        f"median TPOT native {native['tpot_ms']:.3f} ms, LOAD {load['tpot_ms']:.3f} ms"
    )
