"""Figure for experiments A and B. Usage: uv run python fig_ab.py -> figures/expAB.png"""
import glob
import json

import matplotlib.pyplot as plt
import numpy as np

ENC = [("base", "original", "#8a8f98"), ("ft_k2_lr3e-6", "LR 3e-6", "#9cc6f2"),
       ("ft_k2_lr1e-5", "LR 1e-5", "#257de0"), ("ft_k2_lr3e-5", "LR 3e-5", "#0f3d75")]
DY = {"ft_k2_lr1e-5": -1.6, "ft_k2_lr3e-5": 1.6}  # separates the two end labels that overlap
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
reg = {**last("results/modal/eval_repr_reg.jsonl"), **last("results/modal/eval_repr_reg2.jsonl")}  # no-RL baseline (direct regression), in orange
pts = []
for lr in ("3e-7", "1e-6", "3e-6", "1e-5", "3e-5"):
    fin = [last(f)["final"]["success"] * 100 for f in sorted(glob.glob(f"results/modal/expA_reg_k2_lr{lr}_s*.jsonl"))]
    pts.append((reg[f"reg_k2_lr{lr}"]["knn_imagewoof"], np.mean(fin)))
    b.annotate(lr, pts[-1], textcoords="offset points", xytext=(-6, -13), fontsize=8, color="#b34a12", ha="right")
b.plot(*zip(*pts), "-o", color="#f47230", ms=8, mec="white", mew=2, zorder=2)
b.text(85.6, 77.6, "no RL (regression)", color="#333", fontsize=9)
b.text(92.3, 81.9, "with RL", color="#333", fontsize=9)
a.set_title("A: new agent on each RL fine-tuned encoder (3 seeds)", loc="left", fontsize=10)
a.set_xlabel("steps (millions)"); a.set_ylabel("validation success (%)"); a.set_xlim(0, 13.5)
b.set_title("Navigation versus general information", loc="left", fontsize=10)
b.set_xlabel("B: k-NN on Imagewoof, full image (%)"); b.set_ylabel("A: final success (%)")
b.set_xlim(84.5, 96.8); b.set_ylim(70, 83)
for ax in (a, b):
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig("figures/expAB.png", dpi=160)
