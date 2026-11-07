# -*- coding: utf-8 -*-
"""05_evaluate.py
Aggregate per-seed results, compute headline metrics with statistics
(mean+-SD, Wilcoxon signed-rank + Holm correction, Cliff's delta), and
produce manuscript-ready tables (Markdown) and 300-dpi figures.

Outputs under journal_transfer/results/:
  results.json, summary_tables.md, fig1_framework.png,
  fig2_doubt_distributions.png, fig3_ood_roc.png,
  fig4_decision_utility.png, fig5_noise_audit.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

torch.set_num_threads(min(18, os.cpu_count() or 1))
CACHE = common.CACHE
RES = common.RESULTS
os.makedirs(RES, exist_ok=True)
K = len(common.CLASS_NAMES)
SEEDS = list(range(1, common.N_SEEDS + 1))


class Head(nn.Module):
    def __init__(self, d=2048, k=K):
        super().__init__()
        self.net = nn.Linear(d, k)

    def forward(self, x):
        return self.net(x)


def H(p):
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(-1)


def mi(probs):
    return H(probs.mean(0)) - H(probs).mean(0)


def ece(probs, y, bins=15):
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.sum() > 0:
            e += (m.sum() / len(y)) * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def cliff(x, y):
    """Cliff's delta of x relative to y across paired seeds (x_i vs y_i)."""
    ds = []
    for a, b in zip(x, y):
        ds.append((1 if a > b else (-1 if a < b else 0)))
    return float(np.mean(ds))


def load_mddr():
    with open(os.path.join(CACHE, "mddr_all.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def per_seed_scores():
    """Return dict method -> dict score -> list over seeds for OOD/err detection."""
    S = {"msp": [], "mc_mi": [], "ens_mi": [], "edl_u": [], "mddr_e": []}
    for seed in SEEDS:
        out = os.path.join(CACHE, f"seed{seed}")
        lab_t = np.load(os.path.join(CACHE, "lab_test.npy"))
        lab_s = np.load(os.path.join(CACHE, "lab_test_shift.npy"))
        # MSP (std)
        p = np.load(os.path.join(out, "probs_std_test.npy"))
        ps = np.load(os.path.join(out, "probs_std_test_shift.npy"))
        S["msp"].append((1 - p.max(1), 1 - ps.max(1), lab_t, lab_s, p))
        # MC-MI
        pm = np.load(os.path.join(out, "probs_mc_test.npy"))
        pms = np.load(os.path.join(out, "probs_mc_test_shift.npy"))
        S["mc_mi"].append((mi(pm), mi(pms), lab_t, lab_s, pm.mean(0)))
        # ENS-MI
        pe = np.load(os.path.join(out, "probs_ens_test.npy"))
        pes = np.load(os.path.join(out, "probs_ens_test_shift.npy"))
        S["ens_mi"].append((mi(pe), mi(pes), lab_t, lab_s, pe.mean(0)))
        # EDL uncertainty mass
        m = Head()
        m.load_state_dict(torch.load(os.path.join(out, "edl.pt"), map_location="cpu"))
        m.eval()
        with torch.no_grad():
            for sub, key in (("test", "test"), ("test_shift", "test_shift")):
                X = torch.from_numpy(np.load(os.path.join(CACHE, f"feat_{sub}.npy")))
                ev = F.softplus(m(X)) + 1.0
                u = (K / (ev + 1).sum(1)).numpy()
                p_edl = ((ev + 1) / (ev + 1).sum(1, keepdim=True)).numpy()
                if key == "test":
                    u_t, p_t = u, p_edl
                else:
                    u_s, p_s = u, p_edl
        S["edl_u"].append((u_t, u_s, lab_t, lab_s, p_t))
        # MDDR delta_e
        with open(os.path.join(out, "mddr.json")) as f:
            md = json.load(f)
        S["mddr_e"].append((np.array(md["scores"]["mddr_e_test"]),
                            np.array(md["scores"]["mddr_e_test_shift"]),
                            lab_t, lab_s, pe.mean(0)))
    return S


def auc(a, b):
    y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    s = np.concatenate([a, b])
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else 0.5


def roc_curves(S, key_a, key_b):
    curves = {}
    for name, rows in S.items():
        fprs, tprs = [], []
        for (a, b, *_rest) in rows:
            y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
            s = np.concatenate([a, b])
            fpr, tpr, _ = roc_curve(y, s)
            fprs.append(fpr)
            tprs.append(tpr)
        grid = np.linspace(0, 1, 200)
        interp = [np.interp(grid, f, t) for f, t in zip(fprs, tprs)]
        curves[name] = (grid, np.mean(interp, 0))
    return curves


def main():
    S = per_seed_scores()
    MD = load_mddr()
    results = {}

    # ---------- accuracy + ECE (test) ----------
    rows = {}
    for seed in SEEDS:
        out = os.path.join(CACHE, f"seed{seed}")
        y = np.load(os.path.join(CACHE, "lab_test.npy"))
        for name, key in [("std", "probs_std_test"), ("mc", "probs_mc_test"),
                          ("ens", "probs_ens_test"), ("edl", "probs_edl_test")]:
            p = np.load(os.path.join(out, f"{key}.npy"))
            if p.ndim == 3:
                p = p.mean(0)
            rows.setdefault(name, []).append({
                "acc": float((p.argmax(1) == y).mean()),
                "ece": ece(p, y),
            })
    acc_ece = {}
    for name, vals in rows.items():
        acc_ece[name] = {
            "acc": (np.mean([v["acc"] for v in vals]), np.std([v["acc"] for v in vals])),
            "ece": (np.mean([v["ece"] for v in vals]), np.std([v["ece"] for v in vals])),
        }
    results["accuracy_ece"] = acc_ece

    # ---------- OOD detection + error detection ----------
    ood, err = {}, {}
    for name, rows in S.items():
        a_ood = [auc(a, b) for (a, b, *_r) in rows]
        errs = []
        for (a, _b, lab_t, _lab_s, p) in rows:
            e = (p.argmax(1) != lab_t).astype(int)
            if e.sum() == 0 or (1 - e).sum() == 0:
                errs.append(0.5)
            else:
                errs.append(roc_auc_score(e, a))
        ood[name] = (float(np.mean(a_ood)), float(np.std(a_ood)))
        err[name] = (float(np.mean(errs)), float(np.std(errs)))
    # ablations: MDDR delta_e components (saved per seed in mddr.json)
    mi_ab, dist_ab = [], []
    for seed in SEEDS:
        with open(os.path.join(CACHE, f"seed{seed}", "mddr.json")) as f:
            md = json.load(f)
        s = md["scores"]
        mi_ab.append(auc(np.array(s["mi_test"]), np.array(s["mi_test_shift"])))
        dist_ab.append(auc(np.array(s["dist_test"]), np.array(s["dist_test_shift"])))
    ood["mi_only"] = (float(np.mean(mi_ab)), float(np.std(mi_ab)))
    ood["dist_only"] = (float(np.mean(dist_ab)), float(np.std(dist_ab)))
    err["mi_only"] = (float("nan"), 0.0)
    err["dist_only"] = (float("nan"), 0.0)
    results["ood_auroc"] = ood
    results["err_auroc"] = err

    # ---------- deployment metrics ----------
    dep = {}
    for name in ["std", "ens", "mddr"]:
        dep[name] = {}
    for seed in SEEDS:
        with open(os.path.join(CACHE, f"seed{seed}", "mddr.json")) as f:
            md = json.load(f)
        for name in ["std", "ens"]:
            for k, v in md["baselines"][name].items():
                dep[name].setdefault(k, []).append(v)
        for k, v in md["mddr"].items():
            if v is None:
                continue
            dep["mddr"].setdefault(k, []).append(v)
    # normalize baseline counts to rates, fill policy fields
    n_test = int(len(np.load(os.path.join(CACHE, "lab_test.npy"))))
    for seed in SEEDS:
        y = np.load(os.path.join(CACHE, "lab_test.npy"))
        for nm, key in [("std", "probs_std_test"), ("ens", "probs_ens_test")]:
            p = np.load(os.path.join(CACHE, f"seed{seed}", f"{key}.npy"))
            if p.ndim == 3:
                p = p.mean(0)
            dep[nm].setdefault("top3_recall", []).append(
                float(np.mean([y[i] in p[i].argsort()[-3:] for i in range(len(y))])))
    for name in ["std", "ens"]:
        d = dep[name]
        d["agronomic_validity"] = [1 - v / n_test for v in d["violations"]]
        d["unnecessary_rate"] = [v / n_test for v in d["incorrect"]]
        d["wrong_spray_rate"] = [v / n_test for v in d["wrong_spray"]]
        d["missed_rate"] = [v / n_test for v in d["missed"]]
        d["escalation_rate"] = [0.0] * len(SEEDS)
        d["escalation_precision"] = [float("nan")] * len(SEEDS)
    dep_sum = {}
    for name, d in dep.items():
        dep_sum[name] = {k: (float(np.nanmean(v)), float(np.nanstd(v)))
                         for k, v in d.items()}
    results["deployment"] = dep_sum

    # ---------- audit ----------
    with open(os.path.join(CACHE, "audit.json")) as f:
        aud = json.load(f)
    md_prec = [aud[str(s)]["precision_at_100"] for s in SEEDS]
    de_prec = [aud[str(s)]["precision_de_at_100"] for s in SEEDS]
    cb_prec = [aud[str(s)]["precision_comb_at_100"] for s in SEEDS]
    rnd_prec = [aud[str(s)]["precision_random"] for s in SEEDS]
    results["audit"] = {
        "mddr": (float(np.mean(md_prec)), float(np.std(md_prec))),
        "de_only": (float(np.mean(de_prec)), float(np.std(de_prec))),
        "combo": (float(np.mean(cb_prec)), float(np.std(cb_prec))),
        "random": (float(np.mean(rnd_prec)), float(np.std(rnd_prec))),
    }

    # ---------- statistics ----------
    stats = {}
    def wtest(a, b):
        w, p = wilcoxon(a, b)
        return float(w), float(p)

    for metric, get in [
        ("ood_auroc_mddr_vs_ens", lambda: ([auc(a, b) for (a, b, *_r) in S["mddr_e"]],
                                            [auc(a, b) for (a, b, *_r) in S["ens_mi"]])),
        ("ood_auroc_mddr_vs_msp", lambda: ([auc(a, b) for (a, b, *_r) in S["mddr_e"]],
                                           [auc(a, b) for (a, b, *_r) in S["msp"]])),
        ("unnecessary_mddr_vs_std", lambda: (dep["mddr"]["unnecessary_rate"],
                                             dep["std"]["unnecessary_rate"])),
        ("validity_mddr_vs_std", lambda: (dep["mddr"]["agronomic_validity"],
                                          dep["std"]["agronomic_validity"])),
    ]:
        a, b = get()
        w, p = wtest(a, b)
        stats[metric] = {"W": w, "p": p, "cliff_delta": cliff(a, b)}
    # Holm correction over the p-values
    ps = sorted(stats[m]["p"] for m in stats)
    m = len(ps)
    holm = [min(1.0, p * (m - i)) for i, p in enumerate(ps)]
    for (mk, pv), hv in zip(sorted(stats.items(), key=lambda kv: kv[1]["p"]), holm):
        stats[mk]["holm_p"] = float(hv)
    results["stats"] = stats

    with open(os.path.join(RES, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)

    # ================= tables =================
    md_lines = []
    add = md_lines.append
    add("# MDDR benchmark — headline results (mean ± SD over 5 seeds)\n")
    add("| Method | Test acc (%) | ECE | OOD AUROC (score) | Err-det AUROC |")
    add("|---|---|---|---|---|")
    for name, label in [("std", "Standard CNN (MSP)"), ("mc", "MC-Dropout (MI)"),
                        ("ens", "Deep Ensemble (MI)"), ("edl", "EDL (u)"),
                        ("mddr_e", "MDDR (δe)")]:
        key = {"std": "msp", "mc": "mc_mi", "ens": "ens_mi", "edl": "edl_u",
               "mddr_e": "mddr_e"}[name]
        a = acc_ece.get(name)
        if a:
            acc_t = f"{100*a['acc'][0]:.1f}±{100*a['acc'][1]:.1f}"
            ece_t = f"{a['ece'][0]:.3f}±{a['ece'][1]:.3f}"
        else:
            acc_t = "—"
            ece_t = "—"
        o = ood.get(key, (float("nan"), 0.0))
        e = err.get(key, (float("nan"), 0.0))
        add(f"| {label} | {acc_t} | {ece_t} "
            f"| {o[0]:.3f}±{o[1]:.3f} | {e[0]:.3f}±{e[1]:.3f} |")
    for key, lab in [("mi_only", "— MI alone (ablation)"), ("dist_only", "— manifold dist. alone (ablation)")]:
        o = ood.get(key, (float("nan"), 0.0))
        add(f"| {lab} | — | — | {o[0]:.3f}±{o[1]:.3f} | — |")
    add("\n| Deployment (test) | Standard | Ensemble | MDDR |")
    add("|---|---|---|---|")
    for k, lab in [("accuracy", "Accuracy"), ("agronomic_validity", "Agronomic validity"),
                   ("unnecessary_rate", "Unnecessary actions"),
                   ("missed_rate", "Missed treatments"),
                   ("escalation_rate", "Escalation rate"),
                   ("top3_recall", "Top-3 recall")]:
        cells = []
        for n in ["std", "ens", "mddr"]:
            v = dep_sum[n].get(k, (float("nan"), 0.0))
            cells.append(f"{v[0]:.3f}±{v[1]:.3f}")
        add(f"| {lab} | {' | '.join(cells)} |")
    v = dep_sum["mddr"]["escalation_precision"]
    add(f"| Escalation precision | — | — | {v[0]:.3f}±{v[1]:.3f} |")
    add("\n| Noise audit (20% label flip) | precision@100 |")
    add("|---|---|")
    add(f"| MDDR doubt-guided | {results['audit']['mddr'][0]:.3f}±{results['audit']['mddr'][1]:.3f} |")
    add(f"| Random baseline | {results['audit']['random'][0]:.3f}±{results['audit']['random'][1]:.3f} |")
    add("\n| Statistical test (MDDR vs baseline, 5 seeds) | W | p | Holm p | Cliff δ |")
    add("|---|---|---|---|---|")
    for k, v in stats.items():
        add(f"| {k} | {v['W']} | {v['p']:.3f} | {v['holm_p']:.3f} | {v['cliff_delta']:.2f} |")
    with open(os.path.join(RES, "summary_tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    # ================= figures =================
    plt.rcParams.update({"font.size": 10, "figure.dpi": 300})

    # fig1 framework
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.axis("off")
    boxes = [
        (0.02, 0.62, 0.26, 0.24, "Base classifier\n(deep ensemble, ResNet-50\nfrozen features)"),
        (0.34, 0.72, 0.28, 0.18, "δa aleatoric\n(entropy)"),
        (0.34, 0.47, 0.28, 0.18, "δe epistemic\n(MI + manifold dist.)"),
        (0.34, 0.22, 0.28, 0.18, "δo ontological\n(agronomic rules)"),
        (0.68, 0.55, 0.28, 0.20, "AGCR policy\nrisk-calibrated replanning"),
        (0.68, 0.10, 0.28, 0.32, "Act (spray / no action)\nEscalate (top-3 + explanation)\nRe-image request"),
        (0.02, 0.06, 0.26, 0.20, "Field feedback loop\n(AVD: outcome audit,\nthreshold update)"),
    ]
    for x, y, w, h, t in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, facecolor="#eef3fb",
                                   edgecolor="#2b5fa8", lw=1.4))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(0.66, 0.64), xytext=(0.30, 0.66),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#2b5fa8"))
    for yy in (0.80, 0.55, 0.30):
        ax.annotate("", xy=(0.66, 0.62), xytext=(0.35, yy),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#888888"))
    ax.annotate("", xy=(0.82, 0.52), xytext=(0.84, 0.44),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#2b5fa8"))
    ax.annotate("", xy=(0.03, 0.30), xytext=(0.03, 0.12),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#2b5fa8",
                                connectionstyle="arc3,rad=-0.4"))
    ax.set_title("Metacognitive Doubt-Driven Reasoning (MDDR): instantiated pipeline",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig1_framework.png"))
    plt.close(fig)

    # fig2 doubt distributions (test vs shifted)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    pairs = [("msp", "MSP"), ("ens_mi", "Ensemble MI"), ("mddr_e", "MDDR δe")]
    for ax_, (nm, lab) in zip(axes, pairs):
        id_scores = np.concatenate([r[0] for r in S[nm]])
        ood_scores = np.concatenate([r[1] for r in S[nm]])
        ax_.hist([id_scores, ood_scores], bins=30, density=True, alpha=0.65,
                 label=["In-distribution", "Phenotypic shift (OOD)"])
        ax_.set_title(lab)
        ax_.set_xlabel("uncertainty score")
        ax_.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig2_doubt_distributions.png"))
    plt.close(fig)

    # fig3 ROC curves (mean)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    curves = roc_curves(S, 0, 1)
    labels = {"msp": "Standard (MSP)", "mc_mi": "MC-Dropout (MI)",
              "ens_mi": "Ensemble (MI)", "edl_u": "EDL (u)", "mddr_e": "MDDR (δe)"}
    colors = {"msp": "#999999", "mc_mi": "#c48a1d", "ens_mi": "#2b5fa8",
              "edl_u": "#7a4fa3", "mddr_e": "#c0392b"}
    for nm, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, label=f"{labels[nm]} ({ood[nm][0]:.3f})", color=colors[nm], lw=1.8)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("OOD detection: test vs phenotypic-shift set")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig3_ood_roc.png"))
    plt.close(fig)

    # fig4 decision utility bars
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    names = ["std", "ens", "mddr"]
    labels = ["Standard", "Ensemble", "MDDR"]
    metrs = [("agronomic_validity", "Agronomic\nvalidity"),
             ("unnecessary_rate", "Unnecessary\nactions"),
             ("missed_rate", "Missed\ntreatments"),
             ("escalation_rate", "Escalation\nrate")]
    x = np.arange(len(metrs))
    w = 0.25
    for i, (nm, lb) in enumerate(zip(names, labels)):
        vals = [dep_sum[nm][k][0] for k, _ in metrs]
        errs = [dep_sum[nm][k][1] for k, _ in metrs]
        ax.bar(x + (i - 1) * w, vals, w, yerr=errs, capsize=3, label=lb)
    ax.set_xticks(x)
    ax.set_xticklabels([t for _, t in metrs])
    ax.set_ylabel("rate")
    ax.set_title("Deployment decision utility")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig4_decision_utility.png"))
    plt.close(fig)

    # fig5 audit precision@k curves (pooled over seeds)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ks = [20, 50, 100, 200, 300, 500]
    m_prec, d_prec, r_prec = [], [], []
    for seed in SEEDS:
        out = os.path.join(CACHE, f"noise_seed{seed}")
        ens_tr = np.load(os.path.join(out, "probs_ens_train.npy"))
        lab = np.load(os.path.join(out, "noise_lab_train.npy"))
        true_lab = np.load(os.path.join(CACHE, "lab_train.npy"))
        feat_tr = np.load(os.path.join(CACHE, "feat_train.npy"))
        flipped = lab != true_lab
        nn = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(feat_tr)
        dist = nn.kneighbors(feat_tr, return_distance=True)[0].mean(1)
        dE = 0.6 * _rank(mi(ens_tr)) + 0.4 * _rank(dist)
        cent = np.stack([feat_tr[lab == c].mean(0) for c in range(4)])
        d_own = np.linalg.norm(feat_tr - cent[lab], axis=1)
        d_all = np.stack([np.linalg.norm(feat_tr - cent[c], axis=1) for c in range(4)]).T
        d_all.sort(axis=1)
        cons = d_own - 0.5 * d_all[:, 1]
        order = np.argsort(-cons)
        order_d = np.argsort(-dE)
        rng = np.random.default_rng(seed)
        rnd = rng.choice(len(lab), size=max(ks), replace=False)
        m_prec.append([flipped[order[:k]].mean() for k in ks])
        d_prec.append([flipped[order_d[:k]].mean() for k in ks])
        r_prec.append([flipped[rnd[:k]].mean() for k in ks])
    ax.plot(ks, np.mean(m_prec, 0), "o-", label="MDDR audit (consistency doubt)",
            color="#c0392b")
    ax.plot(ks, np.mean(d_prec, 0), "^:", label="Epistemic doubt only", color="#7a4fa3")
    ax.plot(ks, np.mean(r_prec, 0), "s--", label="Random", color="#999999")
    ax.set_xlabel("top-k audited samples")
    ax.set_ylabel("precision (fraction of flipped labels)")
    ax.set_title("Doubt-guided data audit under 20% label noise")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig5_noise_audit.png"))
    plt.close(fig)

    print("EVALUATION DONE")
    print(json.dumps(results, indent=1, default=float)[:3000])


def _rank(x):
    s = np.sort(x)
    return (np.searchsorted(s, x) + 1) / (len(s) + 1)


if __name__ == "__main__":
    main()
