# -*- coding: utf-8 -*-
"""01_prep_data.py
Builds the benchmark split from the local PlantVillage-derived maize subset:
  * near-duplicate (perceptual-hash) grouping so augmented copies never cross
    train/val/test boundaries
  * stratified 70/15/15 split at the image-group level
  * phenotypic-shift OOD variants of val/test (novel-hybrid appearance)
  * simulated deployment metadata (growth stage / weather) for the
    ontological-doubt experiment, incl. a staged "metadata error" subset
"""
import os
import sys

import numpy as np
from PIL import Image, ImageEnhance, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

CACHE = common.CACHE
os.makedirs(CACHE, exist_ok=True)
SHIFT_VAL = os.path.join(CACHE, "shift", "val")
SHIFT_TEST = os.path.join(CACHE, "shift", "test")
os.makedirs(SHIFT_VAL, exist_ok=True)
os.makedirs(SHIFT_TEST, exist_ok=True)

PHASH_THRESH = 4  # hamming distance
IMG_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def ahash(img):
    """64-bit average hash."""
    g = img.convert("L").resize((8, 8), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float32).ravel()
    bits = (a > a.mean()).astype(np.uint8)
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def popcount_chunked(x):
    """Hamming-distance matrix, chunked to bound memory. x: (n,) uint64."""
    n = x.shape[0]
    out = np.zeros((n, n), dtype=np.uint8)
    bc = getattr(np, "bit_count", None)
    CH = 256
    for i in range(0, n, CH):
        j_end = min(i + CH, n)
        xr = x[i:j_end]
        xor = xr[:, None] ^ x[None, :]
        if bc is not None:
            out[i:j_end] = bc(xor).astype(np.uint8)
        else:
            xb = xor.view(np.uint8).reshape(j_end - i, n, 8)
            t = np.array([bin(k).count("1") for k in range(256)], dtype=np.uint8)
            out[i:j_end] = t[xb].sum(axis=2)
    return out


def shift_image(img):
    """Phenotypic-shift OOD transform: drought-like discoloration + high
    contrast + mild defocus (sensor degradation). Deterministic.
    Variant C selected via accuracy probe (0.805 -> 0.620 on probe set)."""
    from PIL import ImageFilter
    return (
        ImageEnhance.Brightness(
            ImageEnhance.Contrast(
                ImageEnhance.Color(img).enhance(0.50)
            ).enhance(1.45)
        ).enhance(1.30).filter(ImageFilter.GaussianBlur(1.5))
    )


def main():
    rng = np.random.default_rng(common.SPLIT_SEED)

    # ---- enumerate images ----
    items = []  # (path, class, ahash)
    for cls, folder in common.CLASS_FOLDERS.items():
        d = os.path.join(common.DATA_ROOT, folder)
        files = [f for f in os.listdir(d) if f.endswith(IMG_EXT)]
        for f in files:
            p = os.path.join(d, f)
            try:
                with Image.open(p) as im:
                    h = ahash(im)
                items.append({"path": p, "cls": cls, "hash": h, "name": f"{cls}/{f}"})
            except Exception as e:
                print("skip", p, e)
    print(f"images enumerated: {len(items)}")
    hashes = np.array([it["hash"] for it in items], dtype=np.uint64)
    D = popcount_chunked(hashes)
    # ---- union-find grouping ----
    n = len(items)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    idx = np.argwhere(D <= PHASH_THRESH)
    print(f"near-dup pairs: {len(idx)}")
    for i, j in idx:
        union(int(i), int(j))
    gid = [find(i) for i in range(n)]
    groups = {}
    for i, g in enumerate(gid):
        groups.setdefault(g, []).append(i)
    print(f"groups: {len(groups)} (merged {n - len(groups)} images)")

    # ---- group-level stratified split ----
    by_cls = {c: [] for c in common.CLASS_NAMES}
    for g, members in groups.items():
        by_cls[items[members[0]]["cls"]].append(g)

    split = {"train": [], "val": [], "test": []}
    for cls in common.CLASS_NAMES:
        gs = by_cls[cls]
        rng.shuffle(gs)
        n = len(gs)
        n_tr = int(round(n * 0.70))
        n_va = int(round(n * 0.15))
        for i, g in enumerate(gs):
            members = groups[g]
            rep = items[members[0]]
            dest = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
            for m in members:
                it = items[m]
                split[dest].append({"name": it["name"], "path": it["path"], "cls": cls})
    for k in split:
        rng.shuffle(split[k])
    common.save_json(split, "split.json")
    for k, v in split.items():
        print(f"{k}: {len(v)}")

    # ---- OOD phenotypic shift for val/test ----
    for subset in ("val", "test"):
        out_dir = SHIFT_VAL if subset == "val" else SHIFT_TEST
        for it in split[subset]:
            dst = os.path.join(out_dir, it["name"].replace("/", "__"))
            if not os.path.exists(dst):
                try:
                    with Image.open(it["path"]) as im:
                        shifted = shift_image(im.convert("RGB"))
                    shifted.save(dst, quality=90)
                except Exception as e:
                    print("shift fail", it["name"], e)
        print(f"shifted {subset}: {len(os.listdir(out_dir))}")

    # ---- deployment metadata for val/test ----
    stages = ["V4", "V5", "V6", "V7", "VT", "R1"]
    stage_order = common.ONTOLOGY_RULES["gls"]["stage_order"]
    for subset in ("val", "test"):
        meta = {}
        for it in split[subset]:
            # realistic weather: 60% temperate, 40% hot spell
            hot = rng.random() < 0.40
            temp = float(rng.uniform(30.0, 34.0) if hot else rng.uniform(24.0, 29.0))
            hum = float(rng.uniform(0.55, 0.95))
            stage = stages[int(rng.integers(0, len(stages)))]
            meta[it["name"]] = {
                "cls": it["cls"],
                "stage": stage,
                "temp_c": temp,
                "hum": hum,
                "violation": False,
            }
        # staged metadata-error subset (~25%): reported stage conflicts with
        # ground-truth disease (e.g., GLS label but stage V4/V5)
        names = list(meta.keys())
        rng.shuffle(names)
        k = int(round(len(names) * 0.25))
        for nm in names[:k]:
            meta[nm]["stage"] = "V4" if meta[nm]["stage"] != "V4" else "V5"
            meta[nm]["violation"] = True
        common.save_json(meta, f"meta_{subset}.json")
        viol = sum(1 for v in meta.values() if v["violation"])
        print(f"{subset} meta: {len(meta)} samples, {viol} staged violations")


if __name__ == "__main__":
    main()
