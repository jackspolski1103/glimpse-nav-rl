"""Figura de los experimentos A y B. Uso: uv run python fig_ab.py -> figures/expAB.png"""
import glob
import json

import matplotlib.pyplot as plt
import numpy as np

ENC = [("base", "original", "#8a8f98"), ("ft_k2_lr3e-6", "LR 3e-6", "#9cc6f2"),
       ("ft_k2_lr1e-5", "LR 1e-5", "#257de0"), ("ft_k2_lr3e-5", "LR 3e-5", "#0f3d75")]
DY = {"ft_k2_lr1e-5": -1.6, "ft_k2_lr3e-5": 1.6}  # separa las dos etiquetas finales que se pisan
last = lambda f: json.loads(open(f).read().strip().splitlines()[-1])
rep = last("results/modal/eval_repr.jsonl")

fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.8))
for key, name, col in ENC:
    hs = [last(f)["history"] for f in sorted(glob.glob(f"results/modal/expA_{key}_s*.jsonl"))]
    x = [m["steps"] / 1e6 for m in hs[0]]
    y = np.array([[m["success"] * 100 for m in h] for h in hs])
    a.fill_between(x, y.min(0), y.max(0), color=col, alpha=0.2, lw=0)
    a.plot(x, y.mean(0), color=col, lw=2)
    a.text(x[-1] + 0.15, y.mean(0)[-1] + DY.get(key, 0), f"{name}  {y.mean(0)[-1]:.1f} %", va="center", fontsize=9)
    knn = rep["original" if key == "base" else key]["knn_imagewoof"]
    b.scatter(knn, y.mean(0)[-1], s=70, color=col, edgecolor="white", linewidth=2, zorder=3)
    b.annotate(name, (knn, y.mean(0)[-1]), textcoords="offset points", xytext=(8, -4), fontsize=9)
a.set_title("A: agente nuevo sobre cada encoder congelado (3 semillas)", loc="left", fontsize=10)
a.set_xlabel("pasos (millones)"); a.set_ylabel("éxito en validación (%)"); a.set_xlim(0, 13.5)
b.set_title("Navegación frente a información general", loc="left", fontsize=10)
b.set_xlabel("B: k-NN en Imagewoof, imagen completa (%)"); b.set_ylabel("A: éxito final (%)")
b.set_xlim(91, 96.5); b.set_ylim(70, 83)
for ax in (a, b):
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig("figures/expAB.png", dpi=160)
