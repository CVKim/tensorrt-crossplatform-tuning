"""
Multi-model Python-side TRT engine load benchmark.

Walks engines/<MODEL>/*.trt, loads each via TensorRT Python API on Windows,
measures deserialize + IExecutionContext creation time, and writes per-model CSV.

Run on Windows side (engines are built with --runtimePlatform=WindowsAMD64).

Configure ROOT below or pass as CLI argument:
    python bench_load_python.py /path/to/workspace
"""
from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorrt as trt


def _parse_root() -> Path:
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default=str(Path.cwd()),
                   help="workspace root (containing engines/ and logs/)")
    return Path(p.parse_args().root)


ROOT = _parse_root()
ENG_DIR = ROOT / "engines"
LOG_DIR = ROOT / "logs"

LOGGER = trt.Logger(trt.Logger.WARNING)


def measure(engine_path: Path) -> dict:
    raw = engine_path.read_bytes()
    disk_bytes = len(raw)
    runtime = trt.Runtime(LOGGER)

    t0 = time.perf_counter()
    engine = runtime.deserialize_cuda_engine(raw)
    t1 = time.perf_counter()
    if engine is None:
        return {"name": engine_path.name, "disk_bytes": disk_bytes,
                "disk_mib": round(disk_bytes / 1024**2, 2),
                "deserialize_ms": -1, "context_ms": -1, "total_ms": -1,
                "num_layers": -1, "num_io": -1, "refittable": -1,
                "device_memory_mib": -1, "status": "FAILED_DESERIALIZE"}

    deserialize_ms = (t1 - t0) * 1000.0
    t2 = time.perf_counter()
    ctx = engine.create_execution_context()
    t3 = time.perf_counter()
    context_ms = (t3 - t2) * 1000.0

    try:
        dev_mem = engine.get_device_memory_size_v2(trt.ExecutionContextAllocationStrategy.STATIC)
    except (AttributeError, TypeError):
        dev_mem = engine.device_memory_size

    info = {
        "name": engine_path.name,
        "disk_bytes": disk_bytes,
        "disk_mib": round(disk_bytes / 1024**2, 2),
        "deserialize_ms": round(deserialize_ms, 1),
        "context_ms": round(context_ms, 1),
        "total_ms": round(deserialize_ms + context_ms, 1),
        "num_layers": engine.num_layers,
        "num_io": engine.num_io_tensors,
        "refittable": int(engine.refittable),
        "device_memory_mib": round(dev_mem / 1024**2, 2),
        "status": "OK",
    }
    del ctx
    del engine
    del runtime
    gc.collect()
    return info


def run_for_model(model_tag: str) -> list[dict]:
    eng_subdir = ENG_DIR / model_tag
    log_subdir = LOG_DIR / model_tag
    log_subdir.mkdir(parents=True, exist_ok=True)
    engines = sorted(eng_subdir.glob("*.trt"))
    if not engines:
        print(f"[{model_tag}] no engines found in {eng_subdir}")
        return []

    print(f"\n========== {model_tag}  ({len(engines)} engines) ==========")
    print(f"{'engine':70s}  {'MiB':>7s}  {'deser_ms':>9s}  {'ctx_ms':>9s}  {'total_ms':>9s}  status")

    rows: list[dict] = []
    for eng in engines:
        try:
            r = measure(eng)
        except Exception as e:
            r = {"name": eng.name, "disk_bytes": eng.stat().st_size,
                 "disk_mib": round(eng.stat().st_size / 1024**2, 2),
                 "deserialize_ms": -1, "context_ms": -1, "total_ms": -1,
                 "num_layers": -1, "num_io": -1, "refittable": -1,
                 "device_memory_mib": -1,
                 "status": f"EXC:{type(e).__name__}:{e}"[:60]}
        r["model"] = model_tag
        rows.append(r)
        print(f"{r['name']:70s}  {r['disk_mib']:>7.2f}  "
              f"{r['deserialize_ms']:>9}  {r['context_ms']:>9}  "
              f"{r['total_ms']:>9}  {r['status']}")

    out_csv = log_subdir / "_load_bench_python.csv"
    fields = ["model", "name", "disk_bytes", "disk_mib", "deserialize_ms",
              "context_ms", "total_ms", "num_layers", "num_io", "refittable",
              "device_memory_mib", "status"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {out_csv}")
    return rows


def main() -> int:
    print(f"TensorRT Python: {trt.__version__}")
    models = sorted([d.name for d in ENG_DIR.iterdir() if d.is_dir()])
    if not models:
        print("No model subdirectories under engines/")
        return 1

    all_rows: list[dict] = []
    for m in models:
        all_rows.extend(run_for_model(m))

    # Combined CSV
    combined = LOG_DIR / "_load_bench_python_all.csv"
    if all_rows:
        fields = list(all_rows[0].keys())
        with combined.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nwrote {combined}")

    # Quick side-by-side summary
    print("\n========== SUMMARY (per-config size + load) ==========")
    by_config: dict[str, dict] = {}
    for r in all_rows:
        if r["status"] != "OK": continue
        # Engine name pattern: <ORIG_ONNX>__<CFG>.trt -> grab CFG
        cfg = r["name"].split("__", 1)[-1].replace(".trt", "")
        by_config.setdefault(cfg, {})[r["model"]] = r

    header_models = sorted({r["model"] for r in all_rows if r["status"] == "OK"})
    print(f"  {'cfg':35s} | " + " | ".join(f"{m:>22s}" for m in header_models))
    print(f"  {'':35s} | " + " | ".join(f"{'disk MiB  total ms':>22s}" for _ in header_models))
    print("  " + "-" * (35 + 25 * len(header_models)))
    for cfg in sorted(by_config):
        cells = []
        for m in header_models:
            r = by_config[cfg].get(m)
            if r:
                cells.append(f"{r['disk_mib']:>9.1f}  {r['total_ms']:>9.0f}")
            else:
                cells.append(f"{'-':>22s}")
        print(f"  {cfg:35s} | " + " | ".join(f"{c:>22s}" for c in cells))

    return 0


if __name__ == "__main__":
    sys.exit(main())
