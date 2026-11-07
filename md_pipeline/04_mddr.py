# -*- coding: utf-8 -*-
"""04_mddr.py
Instantiate the MDDR prototype on top of the deep ensemble and evaluate the
decision pipeline on the simulated deployment set:

  * triadic doubt: delta_a (aleatoric, mean member entropy),
                   delta_e (epistemic, MI + feature-manifold distance),
                   delta_o (ontological, agronomic rule violations)
  * AGCR policy: ontological violation -> escalate; epistemic doubt above
    tau_e -> escalate; extreme aleatoric doubt -> request re-imaging;
    otherwise act (spray / no action)
  * thresholds: tau_e selected on val_shift by Youden's J for detecting
    ensemble top-1 errors (reported in the paper)
  * AVD loop on the label-noise models: doubt-guided relabelling of the
    top-100 highest-doubt training samples vs random-100 baseline

Saves per-seed metrics JSON under cache/seed{s}/mddr.json and cache/
noise_seed{s}/audit.json.
"""
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


def mi(member_probs):
    return H(member_probs.mean(0)) - H(member_probs).mean(0)


def pct_rank(x, ref):
    """percentile rank of x within ref distribution, in [0,1]."""
    s = np.sort(ref)
    return (np.searchsorted(s, x) + 1) / (len(s) + 1)


def load(seed):
    out = os.path.join(CACHE, f"seed{seed}")
    res = {"seed": seed}
    for sub in ["val", "test", "val_shift", "test_shift"]:
        res[f"ens_{sub}"] = np.load(os.path.join(out, f"probs_ens_{sub}.npy"))
        res[f"lab_{sub}"] = np.load(os.path.join(CACHE, f"lab_{sub}.npy"))
        res[f"feat_{sub}"] = np.load(os.path.join(CACHE, f"feat_{sub}.npy"))
    res["ens_train"] = None
    res["feat_train"] = np.load(os.path.join(CACHE, "feat_train.npy"))
    res["lab_train"] = np.load(os.path.join(CACHE, "lab_train.npy"))
    return res


def ontological_violation(top1, meta):
    viol = np.zeros(len(top1), dtype=bool)
    for i, (t, m) in enumerate(zip(top1, meta)):
        if t == IDX_GLS and STAGE_ORDER.index(m["stage"]) < STAGE_ORDER.index(MIN_STAGE):
            viol[i] = True
        if t == IDX_RUST and m["temp_c"] > RUST_TMAX:
            viol[i] = True
    return viol


def compute_doubt(d, subset_key):
    ens = d[f"ens_{subset_key}"]
    feat = d[f"feat_{subset_key}"]
    ent = H(ens).mean(0)  # aleatoric raw
    ep_mi = mi(ens)
    nn = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(d["feat_train"])
    dist = nn.kneighbors(feat, return_distance=True)[0].mean(1)
    return ent, ep_mi, dist


def main():
    all_m = {}
    for seed in range(1, common.N_SEEDS + 1):
        d = load(seed)
        ent_val, mi_val, dist_val = compute_doubt(d, "val")
        ent_vs, mi_vs, dist_vs = compute_doubt(d, "val_shift")
        ent_te, mi_te, dist_te = compute_doubt(d, "test")
        ent_ts, mi_ts, dist_ts = compute_doubt(d, "test_shift")

        dE_val = 0.6 * pct_rank(mi_val, mi_val) + 0.4 * pct_rank(dist_val, dist_val)
        dA_val = pct_rank(ent_val, ent_val)
        dE_vs = 0.6 * pct_rank(mi_vs, mi_val) + 0.4 * pct_rank(dist_vs, dist_val)
        dE_te = 0.6 * pct_rank(mi_te, mi_val) + 0.4 * pct_rank(dist_te, dist_val)
        dE_ts = 0.6 * pct_rank(mi_ts, mi_val) + 0.4 * pct_rank(dist_ts, dist_val)
        dA_te = pct_rank(ent_te, ent_val)

        # ---- tau_e via Youden's J on val_shift error detection ----
        p_vs = d["ens_val_shift"].mean(0)
        top1_vs = p_vs.argmax(1)
        err_vs = top1_vs != d["lab_val_shift"]
        best_j, best_tau = -1.0, 0.5
        for t in np.linspace(0.3, 0.95, 66):
            flag = dE_vs >= t
            if flag.sum() == 0 or err_vs.sum() == 0:
                continue
            tpr = (flag & err_vs).sum() / err_vs.sum()
            fpr = (flag & ~err_vs).sum() / (~err_vs).sum()
            j = tpr - fpr
            if j > best_j:
                best_j, best_tau = j, float(t)
        tau_e = best_tau
        tau_a = 0.95

        # ---- deployment decisions on test ----
        meta_test = common.load_json("meta_test.json")
        names = [it["name"] for it in common.load_json("split.json")["test"]]
        meta = [meta_test[nm] for nm in names]
        p_te = d["ens_test"].mean(0)
        top1 = p_te.argmax(1)
        viol = ontological_violation(top1, meta)
        esc = viol | (dE_te >= tau_e)
        reimg = (~esc) & (dA_te >= tau_a)
        act = ~(esc | reimg)
        true = d["lab_test"]

        def spray_ok(i):
            return p_te[i].argmax() == true[i]  # treated class correct

        n = len(true)
        unnecessary = sum(1 for i in range(n) if act[i] and top1[i] != true[i])
        wrong_spray = sum(1 for i in range(n) if act[i] and top1[i] != IDX_HEALTHY and true[i] == IDX_HEALTHY)
        missed = sum(1 for i in range(n) if act[i] and top1[i] == IDX_HEALTHY and true[i] != IDX_HEALTHY)
        violations_committed = sum(1 for i in range(n) if act[i] and viol[i])
        base_err = (top1 != true)
        warranted = (esc & (base_err | viol)).sum()
        esc_rate = esc.mean()
        esc_prec = warranted / max(1, int(esc.sum()))
        top3_rec = np.mean(np.array([true[i] in p_te[i].argsort()[-3:] for i in range(n)]))
        validity = 1 - violations_committed / n
        unnec_rate = unnecessary / n
        missed_rate = missed / n

        # ---- baselines: always act (no doubt) ----
        def always_act(probs):
            t1 = probs.argmax(1)
            inc = sum(1 for i in range(n) if t1[i] != true[i])
            wsp = sum(1 for i in range(n) if t1[i] != IDX_HEALTHY and true[i] == IDX_HEALTHY)
            msd = sum(1 for i in range(n) if t1[i] == IDX_HEALTHY and true[i] != IDX_HEALTHY)
            vio = sum(1 for i in range(n) if
                      (t1[i] == IDX_GLS and STAGE_ORDER.index(meta[i]["stage"]) < STAGE_ORDER.index(MIN_STAGE))
                      or (t1[i] == IDX_RUST and meta[i]["temp_c"] > RUST_TMAX))
            return {"incorrect": inc, "wrong_spray": wsp, "missed": msd, "violations": vio,
                    "accuracy": float((t1 == true).mean())}

        res = {
            "seed": seed,
            "tau_e": tau_e,
            "tau_a": tau_a,
            "mddr": {
                "accuracy": float((top1 == true).mean()),
                "ece": None,  # computed in 05
                "agronomic_validity": validity,
                "unnecessary_rate": unnec_rate,
                "wrong_spray_rate": wrong_spray / n,
                "missed_rate": missed_rate,
                "escalation_rate": esc_rate,
                "escalation_precision": esc_prec,
                "reimage_rate": float(reimg.mean()),
                "top3_recall": float(top3_rec),
                "violations_committed": int(violations_committed),
            },
            "baselines": {},
            "scores": {
                "mddr_e_test": dE_te.round(4).tolist(),
                "mddr_e_test_shift": dE_ts.round(4).tolist(),
                "mddr_a_test": dA_te.round(4).tolist(),
                "mi_test": mi_te.round(4).tolist(),
                "mi_test_shift": mi_ts.round(4).tolist(),
                "dist_test": dist_te.round(4).tolist(),
                "dist_test_shift": dist_ts.round(4).tolist(),
            },
        }
        for name, key in [("std", "probs_std_test"), ("ens", "probs_ens_test"),
                          ("edl", "probs_edl_test"), ("mc", "probs_mc_test")]:
            pr = np.load(os.path.join(CACHE, f"seed{seed}", f"{key}.npy"))
            if pr.ndim == 3:
                pr = pr.mean(0)
            res["baselines"][name] = always_act(pr)
        json.dump(res, open(os.path.join(CACHE, f"seed{seed}", "mddr.json"), "w"), indent=1)
        all_m[seed] = res
        print(f"[mddr s{seed}] tau_e={tau_e:.3f} validity={validity:.3f} "
              f"unn={unnec_rate:.3f} esc={esc_rate:.3f} esc_prec={esc_prec:.3f}", flush=True)

    # ---- AVD doubt-guided audit on label noise ----
    audits = {}
    for seed in range(1, common.N_SEEDS + 1):
        out = os.path.join(CACHE, f"noise_seed{seed}")
        ens_tr = np.load(os.path.join(out, "probs_ens_train.npy"))
        lab = np.load(os.path.join(out, "noise_lab_train.npy"))
        true_lab = np.load(os.path.join(CACHE, "lab_train.npy"))
        feat_tr = np.load(os.path.join(CACHE, "feat_train.npy"))
        flipped = lab != true_lab
        ep_mi = mi(ens_tr)
        nn = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(feat_tr)
        dist = nn.kneighbors(feat_tr, return_distance=True)[0].mean(1)
        dE = 0.6 * pct_rank(ep_mi, ep_mi) + 0.4 * pct_rank(dist, dist)
        # class-consistency: distance to assigned-class centroid vs nearest
        # other-class centroid (a mislabeled sample sits inside another cluster)
        cent = np.stack([feat_tr[lab == c].mean(0) for c in range(4)])
        d_own = np.linalg.norm(feat_tr - cent[lab], axis=1)
        d_all = np.stack([np.linalg.norm(feat_tr - cent[c], axis=1) for c in range(4)]).T
        d_all.sort(axis=1)
        cons = d_own - 0.5 * d_all[:, 1]
        combined = 0.5 * pct_rank(cons, cons) + 0.5 * dE
        order = np.argsort(-cons)  # MDDR audit: class-consistency doubt
        order_de = np.argsort(-dE)
        order_cb = np.argsort(-combined)
        k = 100
        rng = np.random.default_rng(seed)
        rnd = rng.choice(len(lab), size=k, replace=False)
        audits[seed] = {
            "precision_at_100": float(flipped[order[:k]].mean()),
            "precision_de_at_100": float(flipped[order_de[:k]].mean()),
            "precision_comb_at_100": float(flipped[order_cb[:k]].mean()),
            "precision_random": float(flipped[rnd].mean()),
        }
        print(f"[audit s{seed}] precision@100 MDDR={audits[seed]['precision_at_100']:.3f} "
              f"(dE-only {audits[seed]['precision_de_at_100']:.3f}, "
              f"combo {audits[seed]['precision_comb_at_100']:.3f}) "
              f"random={audits[seed]['precision_random']:.3f} "
              f"(flipped total={int(flipped.sum())})", flush=True)
        json.dump(audits, open(os.path.join(CACHE, "audit.json"), "w"), indent=1)

    common.save_json({"mddr": all_m, "audit": audits}, "mddr_all.json")
    print("MDDR + AUDIT DONE", flush=True)


if __name__ == "__main__":
    main()
