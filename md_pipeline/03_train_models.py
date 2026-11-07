# -*- coding: utf-8 -*-
"""03_train_models.py
Train the four uncertainty baselines on cached ResNet-50 features, for 5 seeds:
  std : single deterministic linear head (softmax, MSP)
  mc  : MLP head with dropout, MC sampling at inference (T=30)
  ens : deep ensemble of 5 independently-initialised linear heads
  edl : evidential Dirichlet head (Sensoy et al., 2018 style loss)
Also trains a label-noise variant (20% GLS<->NCLB flip) of std/ens for the
doubt-guided data-audit experiment.
Predictions and model states are cached per seed for downstream evaluation.
"""
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

torch.set_num_threads(min(18, os.cpu_count() or 1))
CACHE = common.CACHE
SUBSETS = ["val", "test", "val_shift", "test_shift"]
K = len(common.CLASS_NAMES)
DEV = torch.device("cpu")


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)


class Head(nn.Module):
    def __init__(self, d=2048, k=K, mlp=False, p=0.5):
        super().__init__()
        if mlp:
            self.net = nn.Sequential(
                nn.Linear(d, 512), nn.ReLU(), nn.Dropout(p),
                nn.Linear(512, k), nn.Dropout(p))
        else:
            self.net = nn.Linear(d, k)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def eval_acc(model, X, y, kind):
    model.eval()
    p = predict(model, X, kind, T=1)
    if p.ndim == 3:
        p = p.mean(0)
    return float((p.argmax(1) == y.numpy()).mean())


def train_head(model, X, y, Xv, yv, kind, epochs=150, lr=1e-3, batch=256, patience=15):
    yb = F.one_hot(y, K).float()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = X.shape[0]
    best, best_acc, bad = None, -1.0, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            logits = model(X[idx])
            if kind == "edl":
                evidence = F.softplus(logits) + 1.0
                alpha = evidence + 1.0
                S = alpha.sum(1, keepdim=True)
                yt = yb[idx]
                loss = torch.sum(
                    yt * (torch.special.digamma(S) - torch.special.digamma(alpha)), 1
                ).mean()
                lam = min(1.0, ep / 40.0)
                alpha_tilde = yt + (1 - yt) * alpha
                S_t = alpha_tilde.sum(1, keepdim=True)
                kl = (
                    torch.lgamma(S_t).squeeze(1)
                    - torch.lgamma(torch.tensor(float(K), dtype=alpha.dtype))
                    - torch.lgamma(alpha_tilde).sum(1)
                    + ((alpha_tilde - 1) *
                       (torch.special.digamma(alpha_tilde) - torch.special.digamma(S_t))
                       ).sum(1)
                )
                loss = loss + lam * kl.mean()
            else:
                loss = F.cross_entropy(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = eval_acc(model, Xv, yv, kind)
        if acc > best_acc:
            best_acc, best, bad = acc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best)
    return best_acc


@torch.no_grad()
def predict(model, X, kind, T=1):
    """std/edl -> (n,K); mc -> (T,n,K) stochastic; (ens handled externally)."""
    model.eval()
    if kind == "edl":
        logits = model(X)
        evidence = F.softplus(logits) + 1.0
        alpha = evidence + 1.0
        S = alpha.sum(1, keepdim=True)
        return (alpha / S).numpy()
    if kind == "mc":
        model.train()  # keep dropout active
        out = torch.stack([F.softmax(model(X), 1) for _ in range(T)], 0)
        model.eval()
        return out.numpy()
    return F.softmax(model(X), 1).numpy()


def load_sub(sub):
    X = torch.from_numpy(np.load(os.path.join(CACHE, f"feat_{sub}.npy")))
    y = torch.from_numpy(np.load(os.path.join(CACHE, f"lab_{sub}.npy")))
    return X, y


def run_seed(seed):
    out = os.path.join(CACHE, f"seed{seed}")
    os.makedirs(out, exist_ok=True)
    set_seed(seed)
    Xtr, ytr = load_sub("train")
    Xv, yv = load_sub("val")
    for sub in SUBSETS:
        X, _ = load_sub(sub)
        _ = X  # loaded on demand below

    # --- standard ---
    m = Head()
    print(f"[s{seed}] std val_acc={train_head(m, Xtr, ytr, Xv, yv, 'std'):.4f}", flush=True)
    torch.save(m.state_dict(), os.path.join(out, "std.pt"))
    # --- mc-dropout ---
    m = Head(mlp=True)
    print(f"[s{seed}] mc val_acc={train_head(m, Xtr, ytr, Xv, yv, 'mc'):.4f}", flush=True)
    torch.save(m.state_dict(), os.path.join(out, "mc.pt"))
    # --- ensemble (5 heads) ---
    ens = []
    for i in range(common.ENSEMBLE_SIZE):
        torch.manual_seed(seed * 100 + i)
        np.random.seed(seed * 100 + i)
        h = Head()
        train_head(h, Xtr, ytr, Xv, yv, "std")
        ens.append(h.state_dict())
    torch.save(ens, os.path.join(out, "ens.pt"))
    # --- edl ---
    m = Head()
    print(f"[s{seed}] edl val_acc={train_head(m, Xtr, ytr, Xv, yv, 'edl'):.4f}", flush=True)
    torch.save(m.state_dict(), os.path.join(out, "edl.pt"))

    # --- predictions on all subsets ---
    m_std = Head(); m_std.load_state_dict(torch.load(os.path.join(out, "std.pt")))
    m_mc = Head(mlp=True); m_mc.load_state_dict(torch.load(os.path.join(out, "mc.pt")))
    m_edl = Head(); m_edl.load_state_dict(torch.load(os.path.join(out, "edl.pt")))
    ens_models = [Head() for _ in range(common.ENSEMBLE_SIZE)]
    for h, sd in zip(ens_models, ens):
        h.load_state_dict(sd)
    for sub in SUBSETS:
        X, _ = load_sub(sub)
        np.save(os.path.join(out, f"probs_std_{sub}.npy"),
                predict(m_std, X, "std"))
        np.save(os.path.join(out, f"probs_mc_{sub}.npy"),
                predict(m_mc, X, "mc", T=common.MC_PASSES))
        np.save(os.path.join(out, f"probs_edl_{sub}.npy"),
                predict(m_edl, X, "edl"))
        ens_out = np.stack([predict(h, X, "std") for h in ens_models])
        np.save(os.path.join(out, f"probs_ens_{sub}.npy"), ens_out)
        print(f"[s{seed}] predicted {sub}", flush=True)
    print(f"[s{seed}] done", flush=True)


def run_noise(seed):
    """20% GLS<->NCLB label flip; train std + ensemble; store train probs for audit."""
    out = os.path.join(CACHE, f"noise_seed{seed}")
    os.makedirs(out, exist_ok=True)
    set_seed(seed)
    Xtr, ytr = load_sub("train")
    Xv, yv = load_sub("val")
    y = ytr.clone()
    i_gls = (ytr == common.CLASS2IDX["gls"]).nonzero().flatten()
    i_nclb = (ytr == common.CLASS2IDX["nclb"]).nonzero().flatten()
    nflip = int(round(0.20 * min(len(i_gls), len(i_nclb))))
    g1 = torch.randperm(len(i_gls))
    g2 = torch.randperm(len(i_nclb))
    y[i_gls[g1[:nflip]]] = common.CLASS2IDX["nclb"]
    y[i_nclb[g2[:nflip]]] = common.CLASS2IDX["gls"]
    np.save(os.path.join(out, "noise_lab_train.npy"), y.numpy())

    m = Head()
    train_head(m, Xtr, y, Xv, yv, "std")
    torch.save(m.state_dict(), os.path.join(out, "std.pt"))
    ens = []
    for i in range(common.ENSEMBLE_SIZE):
        torch.manual_seed(seed * 100 + i)
        np.random.seed(seed * 100 + i)
        h = Head()
        train_head(h, Xtr, y, Xv, yv, "std")
        ens.append(h.state_dict())
    torch.save(ens, os.path.join(out, "ens.pt"))
    ens_models = [Head() for _ in range(common.ENSEMBLE_SIZE)]
    for h, sd in zip(ens_models, ens):
        h.load_state_dict(sd)
    Xt, _ = load_sub("test")
    np.save(os.path.join(out, "probs_ens_test.npy"),
            np.stack([predict(h, Xt, "std") for h in ens_models]))
    np.save(os.path.join(out, "probs_ens_train.npy"),
            np.stack([predict(h, Xtr, "std") for h in ens_models]))
    print(f"[n{seed}] noise model done (flips={2 * nflip})", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode != "noise":
        for s in range(1, common.N_SEEDS + 1):
            run_seed(s)
    for s in range(1, common.N_SEEDS + 1):
        run_noise(s)
    print("ALL TRAINING DONE", flush=True)
