"""Mismos episodios de validación con el encoder original y con el ajustado con RL.

Antes: agente de la fase 1 (frozen_bump_s0, encoder original). Después: encoder ft_k2_lr3e-5 con el agente que
salió de ese ajuste (objetivo codificado con el encoder original, como en el entrenamiento).
Uso: uv run python fig_antes_despues.py -> figures/antes_despues.png
"""
import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

from glimpse import GLIMPSE, GRID, IMG, STRIDE
from ppo import Agent, make_task
from viz import BLUE, INK, INK2, ORANGE, box, rollout

FT, N, SEED = "ft_k2_lr3e-5", 6, 3
load = lambda f: torch.from_numpy(np.load(f)["emb"].astype(np.float32))
_, (emb0, n_va), obs_dim = make_task("embeddings", "", "cpu")
tr = load("out/base_train.npz")
mu, sd = tr.mean((0, 1)), tr.std((0, 1)) + 1e-6
del tr
base, ft = (load("out/base_val.npz") - mu) / sd, (load(f"out/{FT}_val.npz") - mu) / sd
agents = []
for path in (max(glob.glob("checkpoints/frozen_bump_s0_*.pt"), key=lambda p: int(p.rsplit("_", 1)[1][:-3])),
             f"out/{FT}_agent.pt"):
    a = Agent(obs_dim)
    a.load_state_dict(torch.load(path, map_location="cpu"))
    agents.append(a)


def run(n, seed):
    g = torch.Generator().manual_seed(seed)
    img = torch.randint(n_va, (n,), generator=g)
    start = torch.randint(GRID, (n, 2), generator=g)
    goal = (start + torch.randint(1, GRID, (n, 2), generator=g)) % GRID
    before = rollout(agents[0], emb0, n_va, img, start, goal, seed)
    after = rollout(agents[1], lambda i, c: ft[i, c], n_va, img, start, goal, seed, embed_goal=lambda i, c: base[i, c])
    return img, start, goal, before, after


_, _, _, (_, ok0), (_, ok1) = run(5000, 0)
print(f"mismos 5000 episodios: antes {np.mean(ok0) * 100:.1f} %, después {np.mean(ok1) * 100:.1f} %")

img, start, goal, before, after = run(N, SEED)
ds = datasets.ImageFolder("data/imagenette2/val", transforms.Compose(
    [transforms.Resize(IMG), transforms.CenterCrop(IMG), transforms.ToTensor()]))
fig, axes = plt.subplots(2, N, figsize=(3.1 * N, 6.9))
for row, (label, (paths, ok)) in enumerate((("Encoder original", before), ("Encoder ajustado con RL", after))):
    for i in range(N):
        ax = axes[row, i]
        ax.imshow(ds[int(img[i])][0].permute(1, 2, 0))
        box(ax, start[i].tolist(), BLUE)
        box(ax, goal[i].tolist(), ORANGE)
        pts = [(c * STRIDE + GLIMPSE / 2, r * STRIDE + GLIMPSE / 2) for r, c in paths[i]]
        for p0, p1 in zip(pts, pts[1:]):
            if p0 != p1:  # si son iguales, chocó con el borde
                for color, lw in (("white", 4.5), (INK, 2)):
                    ax.annotate("", p1, p0, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0,
                                                            shrinkB=0, mutation_scale=11))
        ax.plot(*pts[-1], "o", ms=9, color=INK, mec="white", mew=2)
        dist = int((start[i] - goal[i]).abs().sum())
        ax.set_title(f"{'Llegó' if ok[i] else 'No llegó'} · {len(paths[i]) - 1} pasos (mínimo {dist})", fontsize=9,
                     loc="left", fontweight="bold" if ok[i] else "normal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[row, 0].set_ylabel(label, fontsize=11, color=INK, fontweight="bold")
fig.suptitle("Mismos episodios de validación · Azul: inicio · Naranja: objetivo · Flechas: recorrido · Punto negro: final",
             x=0.01, ha="left", fontsize=10, color=INK2, y=1.0)
fig.tight_layout()
fig.savefig("figures/antes_despues.png", dpi=100)
