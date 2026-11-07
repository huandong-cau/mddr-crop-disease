# MDDR benchmark — headline results (mean ± SD over 5 seeds)

| Method | Test acc (%) | ECE | OOD AUROC (score) | Err-det AUROC |
|---|---|---|---|---|
| Standard CNN (MSP) | 82.2±0.3 | 0.095±0.008 | 0.703±0.018 | 0.549±0.022 |
| MC-Dropout (MI) | 82.4±0.7 | 0.200±0.011 | 0.339±0.045 | 0.404±0.036 |
| Deep Ensemble (MI) | 82.3±0.2 | 0.097±0.005 | 0.747±0.023 | 0.597±0.024 |
| EDL (u) | 81.2±0.5 | 0.226±0.010 | 0.759±0.043 | 0.593±0.011 |
| MDDR (δe) | — | — | 0.803±0.021 | 0.593±0.016 |
| — MI alone (ablation) | — | — | 0.748±0.023 | — |
| — manifold dist. alone (ablation) | — | — | 0.739±0.000 | — |

| Deployment (test) | Standard | Ensemble | MDDR |
|---|---|---|---|
| Accuracy | 0.822±0.003 | 0.823±0.002 | 0.823±0.002 |
| Agronomic validity | 0.751±0.004 | 0.751±0.002 | 1.000±0.000 |
| Unnecessary actions | 0.178±0.003 | 0.177±0.002 | 0.095±0.006 |
| Missed treatments | 0.028±0.001 | 0.029±0.001 | 0.026±0.001 |
| Escalation rate | 0.000±0.000 | 0.000±0.000 | 0.370±0.036 |
| Top-3 recall | 0.991±0.001 | 0.991±0.000 | 0.991±0.000 |
| Escalation precision | — | — | 0.747±0.052 |

| Noise audit (20% label flip) | precision@100 |
|---|---|
| MDDR doubt-guided | 0.206±0.015 |
| Random baseline | 0.092±0.035 |

| Statistical test (MDDR vs baseline, 5 seeds) | W | p | Holm p | Cliff δ |
|---|---|---|---|---|
| ood_auroc_mddr_vs_ens | 0.0 | 0.062 | 0.250 | 1.00 |
| ood_auroc_mddr_vs_msp | 0.0 | 0.062 | 0.188 | 1.00 |
| unnecessary_mddr_vs_std | 0.0 | 0.062 | 0.125 | -1.00 |
| validity_mddr_vs_std | 0.0 | 0.062 | 0.062 | 1.00 |
