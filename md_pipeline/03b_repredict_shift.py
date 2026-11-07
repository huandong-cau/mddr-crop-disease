# -*- coding: utf-8 -*-
"""03b_repredict_shift.py — recompute predictions on the regenerated
val_shift/test_shift features for all methods and seeds (clean-subset
predictions are unchanged and reused)."""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import CACHE, CLASS_NAMES  # noqa: E402

torch.set_num_threads(min(18, os.cpu_count() or 1))


class Head(torch.nn.Module):
    def __init__(self, mlp=False, k=len(CLASS_NAMES), p=0.5):
        super().__init__()
        if mlp:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(2048, 512), torch.nn.ReLU(), torch.nn.Dropout(p),
                torch.nn.Linear(512, k), torch.nn.Dropout(p))
        else:
            self.net = torch.nn.Linear(2048, k)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def predict(model, X, kind, T=1):
    model.eval()
    if kind == "edl":
        ev = F.softplus(model(X)) + 1.0
        alpha = ev + 1.0
        S = alpha.sum(1, keepdim=True)
        return (alpha / S).numpy()
    if kind == "mc":
        model.train()
        out = torch.stack([F.softmax(model(X), 1) for _ in range(T)], 0)
        model.eval()
        return out.numpy()
    return F.softmax(model(X), 1).numpy()


def main():
    for seed in range(1, common.N_SEEDS + 1):
        out = os.path.join(CACHE, f"seed{seed}")
        m_std = Head()
        m_std.load_state_dict(torch.load(os.path.join(out, "std.pt"), map_location="cpu"))
        m_mc = Head(mlp=True)
        m_mc.load_state_dict(torch.load(os.path.join(out, "mc.pt"), map_location="cpu"))
        m_edl = Head()
        m_edl.load_state_dict(torch.load(os.path.join(out, "edl.pt"), map_location="cpu"))
        ens_sd = torch.load(os.path.join(out, "ens.pt"), map_location="cpu")
        ens = [Head() for _ in ens_sd]
        for h, sd in zip(ens, ens_sd):
            h.load_state_dict(sd)
        for sub in ("val_shift", "test_shift"):
            X = torch.from_numpy(np.load(os.path.join(CACHE, f"feat_{sub}.npy")))
            np.save(os.path.join(out, f"probs_std_{sub}.npy"), predict(m_std, X, "std"))
            np.save(os.path.join(out, f"probs_mc_{sub}.npy"),
                    predict(m_mc, X, "mc", T=common.MC_PASSES))
            np.save(os.path.join(out, f"probs_edl_{sub}.npy"), predict(m_edl, X, "edl"))
            np.save(os.path.join(out, f"probs_ens_{sub}.npy"),
                    np.stack([predict(h, X, "std") for h in ens]))
        print(f"[s{seed}] shift predictions refreshed", flush=True)
    print("REPREDICT DONE", flush=True)


if __name__ == "__main__":
    main()
