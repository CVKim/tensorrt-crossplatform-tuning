"""Dump layer names from each engine to confirm Flash Attention / fused MHA presence.

Usage:
    python inspect_engine_layers.py engine1.trt engine2.trt ...

For meaningful output, engines must have been built WITHOUT --profilingVerbosity=none
(otherwise the inspector returns empty layer info).
"""
import sys
from pathlib import Path
from collections import Counter
import tensorrt as trt

LOGGER = trt.Logger(trt.Logger.WARNING)


def attention_keywords(name: str) -> str:
    n = name.lower()
    if "flash" in n: return "FLASH"
    if "fused_multi_head" in n or "fmha" in n or "kmha" in n: return "FMHA"
    if "multihead" in n or "multi_head" in n: return "MHA"
    if "scaled_dot" in n or "sdpa" in n: return "SDPA"
    if "attention" in n: return "ATTN"
    return ""


def inspect(engine_path: Path) -> None:
    if not engine_path.exists():
        print(f"!! not found: {engine_path}")
        return
    runtime = trt.Runtime(LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    inspector = engine.create_engine_inspector()
    info = inspector.get_engine_information(trt.LayerInformationFormat.ONELINE)

    n_layers = engine.num_layers
    layer_lines = info.strip().split("\n")

    attn_hits = Counter()
    norm_hits = 0
    softmax_hits = 0
    fused_names = []
    for line in layer_lines:
        tag = attention_keywords(line)
        if tag:
            attn_hits[tag] += 1
            if len(fused_names) < 3:
                fused_names.append(line[:160])
        lower = line.lower()
        if "layernorm" in lower or "layer_norm" in lower or "_norm_" in lower:
            norm_hits += 1
        if "softmax" in lower:
            softmax_hits += 1

    print(f"\n=== {engine_path.name} ===")
    print(f"  num_layers (post-fusion): {n_layers}")
    print(f"  attention-related: {dict(attn_hits)}")
    print(f"  layernorm-related: {norm_hits}")
    print(f"  softmax-related:   {softmax_hits}")
    if fused_names:
        print(f"  sample attention layer names:")
        for s in fused_names:
            print(f"    {s}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    for arg in sys.argv[1:]:
        inspect(Path(arg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
