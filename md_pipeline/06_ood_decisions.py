# -*- coding: utf-8 -*-
"""06_ood_decisions.py — decision-level evaluation ON the phenotypic-shift
(OOD) set: how much does MDDR's doubt-driven escalation reduce incorrect
autonomous actions when the classifier faces shifted inputs?
Reuses per-seed models, saved tau_e and deployment metadata (metadata is a
property of the field scenario, identical for clean and shifted images)."""
import json
import os
import sys

import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

CACHE = common.CACHE
STAGE_ORDER = common.ONTOLOGY_RULES["gls"]["stage_order"]
MIN_STAGE = common.ONTOLOGY_RULES["gls"]["min_stage"]
RUST_TMAX = common.ONTOLOGY_RULES["rust"]["max_temp_c"]
IDX_GLS = common.CLASS2IDX["gls"]
IDX_RUST = common.CLASS2IDX["rust"]
IDX_HEALTHY = common.CLASS2IDX["healthy"]


def H(p):
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(-1)


def mi(probs):
    return H(probs.mean(0)) - H(probs).mean(0)


def pct_rank(x, ref):
    s = np.sort(ref)
    return (np.searchsorted(s, x) + 1) / (len(s) + 1)


def violations(top1, meta):
    viol = np.zeros(len(top1), dtype=bool)
    for i, (t, m) in enumerate(zip(top1, meta)):
        if t == IDX_GLS and STAGE_ORDER.index(m["stage"]) < STAGE_ORDER.index(MIN_STAGE):
            viol[i] = True
        if t == IDX_RUST and m["temp_c"] > RUST_TMAX:
            viol[i] = True
    return viol


def main():
    split = common.load_json("split.json")
    meta_test = common.load_json("meta_test.json")
    names = [it["name"] for it in split["test"]]
    meta = [meta_test[nm] for nm in names]
    feat_train = np.load(os.path.join(CACHE, "feat_train.npy"))
    out_all = {}
    for seed in range(1, common.N_SEEDS + 1):
        out = os.path.join(CACHE, f"seed{seed}")
        with open(os.path.join(out, "mddr.json")) as f:
            md = json.load(f)
        tau_e = md["tau_e"]
        ens = np.load(os.path.join(out, "probs_ens_test_shift.npy"))
        std = np.load(os.path.join(out, "probs_std_test_shift.npy"))
        lab = np.load(os.path.join(CACHE, "lab_test_shift.npy"))
        feat = np.load(os.path.join(CACHE, "feat_test_shift.npy"))
        feat_val = np.load(os.path.join(CACHE, "feat_val_shift.npy"))
        ens_val = np.load(os.path.join(out, "probs_ens_val_shift.npy"))
        lab_val = np.load(os.path.join(CACHE, "lab_val_shift.npy"))
        # reference distributions from validation shift set
        mi_ref = mi(ens_val)
        nn = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(feat_train)
        dist_ref = nn.kneighbors(feat_val, return_distance=True)[0].mean(1)
        mi_te = mi(ens)
        dist_te = nn.kneighbors(feat, return_distance=True)[0].mean(1)
        dE = 0.6 * pct_rank(mi_te, mi_ref) + 0.4 * pct_rank(dist_te, dist_ref)
        p = ens.mean(0)
        top1 = p.argmax(1)
        viol = violations(top1, meta)
        esc = viol | (dE >= tau_e)
        act = ~esc
        true = lab
        n = len(true)
        incorrect_mddr = sum(1 for i in range(n) if act[i] and top1[i] != true[i])
        base_err = (top1 != true)
        warranted = (esc & (base_err | viol)).sum()
        esc_prec = warranted / max(1, int(esc.sum()))
        t1_std = std.argmax(1)
        incorrect_std = int((t1_std != true).sum())
        res = {
            "seed": seed,
            "incorrect_std_rate": incorrect_std / n,
            "incorrect_ens_rate": float((top1 != true).mean()),
            "incorrect_mddr_rate": incorrect_mddr / n,
            "escalation_rate": float(esc.mean()),
            "escalation_precision": float(esc_prec),
            "validity": float(1 - (act & viol).sum() / n),
            "std_accuracy": float((t1_std == true).mean()),
            "ens_accuracy": float((top1 == true).mean()),
        }
        out_all[seed] = res
        print(f"[s{seed}] shift-set: std_inc={res['incorrect_std_rate']:.3f} "
              f"ens_inc={res['incorrect_ens_rate']:.3f} mddr_inc={res['incorrect_mddr_rate']:.3f} "
              f"esc={res['escalation_rate']:.3f} esc_prec={res['escalation_precision']:.3f} "
              f"validity={res['validity']:.3f} std_acc={res['std_accuracy']:.3f}", flush=True)
    with open(os.path.join(CACHE, "ood_decisions.json"), "w") as f:
        json.dump(out_all, f, indent=1)
    print("OOD DECISIONS DONE", flush=True)


if __name__ == "__main__":
    main()
