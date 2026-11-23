"""Markdown table of chunk_latency results: <dir>/*/latency.json -> stdout.

Usage: python -m delayopsd.eval.latency_table outputs/chunk_latency
"""

import json
import sys
from pathlib import Path


def dtype_note(d):
    total = sum(v["params"] for v in d.values())
    return ", ".join(f"{k} {100 * v['params'] / total:.0f}%" for k, v in sorted(d.items(), key=lambda kv: -kv[1]["params"]))


def main(root):
    rows = [json.loads(p.read_text()) for p in sorted(Path(root).glob("*/latency.json"))]
    if not rows:
        raise SystemExit(f"no latency.json under {root}")
    gpus = sorted({r["gpu"] for r in rows})
    print(f"GPU: {', '.join(gpus)} | torch {rows[0]['torch']} (CUDA {rows[0]['cuda']}) | batch 1 | "
          f"{rows[0]['n_warmup']} warm-up + {rows[0]['n_iters']} timed calls | TF32 matmul {rows[0]['allow_tf32']}\n")
    print("| label | rtc | k | compile | params dtype | denoise steps | chunk | e2e ms (mean ± std) | median | p90 | "
          "model ms | K_min 20 Hz | K_min 50 Hz |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        e, m = r["e2e_ms"], r["model_ms"]
        compile = r["compile_mode"] if r["compile_model"] else "off"
        model = "–" if m is None else f"{m['mean']:.1f} ± {m['std']:.1f}"
        print(f"| {r['label']} | {r['rtc']} | {r['delay'] or '–'} | {compile} | {dtype_note(r['param_dtypes'])} | "
              f"{r['num_denoise_steps']} | {r['chunk_size']} | {e['mean']:.1f} ± {e['std']:.1f} | {e['median']:.1f} | {e['p90']:.1f} | "
              f"{model} | {r['k_min_mean']['20Hz']} | {r['k_min_mean']['50Hz']} |")


if __name__ == "__main__":
    main(sys.argv[1])
