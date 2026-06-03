"""
Compare per-defect bounding-box coordinates across multiple engine variants
producing the same inference image.

Use case: you have rendered detection results from N engine builds (each engine
draws predicted bboxes in red on the same input image, saved as JPG/PNG). This
script:

  1. Detects red bbox connected components per image.
  2. Pairs bboxes across images by maximum IoU vs a reference image.
  3. Reports per-defect coordinate deltas and per-image mean / max delta.
  4. Saves cropped defect regions for visual side-by-side review.

Usage:
    python analyze_defect_bboxes.py /path/to/folder --reference opt4_local_fp16.jpg

The script will:
    * read every *.jpg / *.png in the folder
    * use --reference as the baseline for all delta computations
    * write _analysis/ subfolder with cropped defect regions per variant
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # disable decompression-bomb warning for large industrial images


def find_red_bboxes(img_rgb: np.ndarray, min_area: int = 20,
                    col_gap: int = 50) -> list[tuple[int, int, int, int]]:
    """Find connected red regions. Returns list of (x0, y0, x1, y1)."""
    r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
    mask = (r.astype(int) - np.maximum(g, b).astype(int) > 60) & (r > 120)
    if not mask.any():
        return []
    col_has = mask.any(axis=0)
    cols = np.where(col_has)[0]
    if len(cols) == 0:
        return []
    gaps = np.where(np.diff(cols) > col_gap)[0]
    starts = np.concatenate([[cols[0]], cols[gaps + 1]]) if len(gaps) else np.array([cols[0]])
    ends = np.concatenate([cols[gaps], [cols[-1]]]) if len(gaps) else np.array([cols[-1]])
    bboxes = []
    for x0, x1 in zip(starts, ends):
        col_slice = mask[:, x0:x1 + 1]
        rows = np.where(col_slice.any(axis=1))[0]
        if len(rows) == 0:
            continue
        y0, y1 = rows[0], rows[-1]
        if (x1 - x0 + 1) * (y1 - y0 + 1) < min_area:
            continue
        bboxes.append((int(x0), int(y0), int(x1), int(y1)))
    return bboxes


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0 + 1); ih = max(0, iy1 - iy0 + 1)
    inter = iw * ih
    aa = (ax1 - ax0 + 1) * (ay1 - ay0 + 1)
    bb = (bx1 - bx0 + 1) * (by1 - by0 + 1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def crop_union(img: np.ndarray, bb1, bb2, pad: int = 24) -> np.ndarray:
    H, W = img.shape[:2]
    x0 = max(0, min(bb1[0], bb2[0]) - pad)
    y0 = max(0, min(bb1[1], bb2[1]) - pad)
    x1 = min(W - 1, max(bb1[2], bb2[2]) + pad)
    y1 = min(H - 1, max(bb1[3], bb2[3]) + pad)
    return img[y0:y1 + 1, x0:x1 + 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="Folder containing rendered comparison images")
    parser.add_argument("--reference", required=True,
                        help="Filename within folder used as the delta baseline")
    parser.add_argument("--out", default="_analysis",
                        help="Subfolder for cropped defect regions (default: _analysis)")
    parser.add_argument("--no-crops", action="store_true",
                        help="Skip writing cropped defect regions")
    args = parser.parse_args()

    folder = Path(args.folder)
    out_dir = folder / args.out
    out_dir.mkdir(exist_ok=True)

    image_paths = sorted([p for p in folder.iterdir()
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                          and not p.name.startswith("_")])
    if not image_paths:
        print(f"No images found in {folder}", file=sys.stderr)
        return 1

    ref_path = folder / args.reference
    if not ref_path.exists() or ref_path not in image_paths:
        print(f"Reference image {args.reference} not found in {folder}", file=sys.stderr)
        return 1

    print(f"=== Loading {len(image_paths)} images ===")
    imgs: dict[str, np.ndarray] = {}
    for p in image_paths:
        imgs[p.stem] = np.array(Image.open(p).convert("RGB"))
        print(f"  {p.stem}: {imgs[p.stem].shape}")

    print("\n=== Detecting red bboxes ===")
    bboxes_per: dict[str, list[tuple[int, int, int, int]]] = {}
    for k, img in imgs.items():
        bbs = find_red_bboxes(img)
        bboxes_per[k] = bbs
        print(f"  {k}: {len(bbs)} bboxes")

    ref_key = ref_path.stem
    ref_bbs = bboxes_per[ref_key]
    keys = list(imgs.keys())

    print(f"\n=== Per-defect comparison (reference = {ref_key}) ===")
    for i, ref_bb in enumerate(ref_bbs):
        print(f"\n--- defect #{i}  ref bbox (x0,y0,x1,y1) = {ref_bb} ---")
        for k in keys:
            bbs = bboxes_per[k]
            if not bbs:
                print(f"  {k:50s} NO_BBOX")
                continue
            ious = [bbox_iou(ref_bb, bb) for bb in bbs]
            j = int(np.argmax(ious))
            bb = bbs[j]
            d = (bb[0] - ref_bb[0], bb[1] - ref_bb[1],
                 bb[2] - ref_bb[2], bb[3] - ref_bb[3])
            print(f"  {k:50s} iou={ious[j]:.4f}  Δ=({d[0]:+d},{d[1]:+d},{d[2]:+d},{d[3]:+d})")

    print("\n=== Summary (vs reference) ===")
    print(f"{'option':50s} {'mean_iou':>10s} {'mean|Δ|':>10s} {'max|Δ|':>8s}")
    for k in keys:
        if k == ref_key:
            continue
        bbs = bboxes_per[k]
        if not bbs:
            print(f"  {k:50s} NO BBOXES")
            continue
        ious, deltas = [], []
        for ref_bb in ref_bbs:
            ds = [bbox_iou(ref_bb, bb) for bb in bbs]
            j = int(np.argmax(ds))
            best = bbs[j]
            ious.append(ds[j])
            deltas.append(abs(best[0] - ref_bb[0]) + abs(best[1] - ref_bb[1])
                          + abs(best[2] - ref_bb[2]) + abs(best[3] - ref_bb[3]))
        print(f"  {k:50s} {np.mean(ious):>10.4f} {np.mean(deltas):>10.2f} {np.max(deltas):>8d}")

    if not args.no_crops:
        print(f"\n=== Saving cropped defect regions to {out_dir} ===")
        for i, ref_bb in enumerate(ref_bbs):
            for k in keys:
                bbs = bboxes_per[k]
                if not bbs:
                    continue
                ious = [bbox_iou(ref_bb, bb) for bb in bbs]
                j = int(np.argmax(ious))
                crop = crop_union(imgs[k], ref_bb, bbs[j])
                Image.fromarray(crop).save(out_dir / f"defect{i}_{k}.png")
            print(f"  defect #{i}: saved {len(keys)} crops")

    return 0


if __name__ == "__main__":
    sys.exit(main())
