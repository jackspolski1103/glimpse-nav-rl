"""Figuras: curva de aprendizaje y trayectorias sobre imágenes de validación.

Uso: uv run python viz.py --name frozen_s0
Salida: figures/<name>_curva.png y figures/<name>_trayectorias.png
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from torchvision import datasets, transforms

from env import GlimpseNav
from glimpse import GLIMPSE, GRID, IMG, STRIDE
from ppo import Agent, make_task

BLUE, ORANGE, INK, INK2, GRIDC, AXIS = "#257de0", "#f47230", "#191919", "#666666", "#e7e7e7", "#c2c2c2"
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.edgecolor": AXIS, "savefig.dpi": 160, "savefig.bbox": "tight"})


def curve(run, out):
    h = run["history"]
    x = [m["steps"] / 1e6 for m in h]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    series = [("Éxito (%)", [("validación", [m["success"] * 100 for m in h], BLUE),
                             ("entrenamiento", [m["train_success"] * 100 for m in h], ORANGE)],
               run["random"]["success"] * 100, (0, 100)),
              ("SPL", [("validación", [m["spl"] for m in h], BLUE)], run["random"]["spl"], (0, 1))]
    for ax, (title, lines, rand, ylim) in zip(axes, series):
        gap = (ylim[1] - ylim[0]) * 0.07  # separación mínima entre etiquetas finales
        ends = sorted(lines, key=lambda l: l[1][-1])
        label_y = [ends[0][1][-1]]
        for _, y, _ in ends[1:]:
            label_y.append(max(y[-1], label_y[-1] + gap))
        for (label, y, color), ly in zip(ends, label_y):
            ax.plot(x, y, color=color, lw=2)
            ax.plot(x[-1], y[-1], "o", color=color, ms=5, clip_on=False)
            ax.annotate(f"{label}  {y[-1]:.3g}", (x[-1], ly), xytext=(8, 0), textcoords="offset points",
                        va="center", color=INK, fontsize=9)
        ax.axhline(rand, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.annotate(f"paseo aleatorio  {rand:.3g}", (x[-1], rand), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=9)
        ax.set(xlabel="Pasos de entrenamiento (millones)", ylim=ylim, xlim=(0, x[-1]))
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.grid(axis="y", color=GRIDC, lw=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(length=0)
    fig.subplots_adjust(wspace=0.55)
    fig.savefig(out)
    plt.close(fig)


@torch.no_grad()
def rollout(agent, embed, n_images, img, start, goal, seed, embed_goal=None):
    """Mismos episodios (imagen, inicio, objetivo) para cualquier agente. -> lista de caminos, éxitos"""
    env = GlimpseNav(embed, n_images, len(img), auto_reset=False, embed_goal=embed_goal)
    env.img, env.pos, env.goal = img.clone(), start.clone(), goal.clone()
    env.visited[:] = False
    env.visited[env.ar, start[:, 0], start[:, 1]] = True
    env.t[:] = 0
    torch.manual_seed(seed)
    paths = [[tuple(p)] for p in start.tolist()]
    alive, ok = torch.ones(len(img), dtype=torch.bool), torch.zeros(len(img), dtype=torch.bool)
    for _ in range(env.max_steps):
        a = agent.act(env.observe())[0]
        pos_before = env.pos.clone()
        env.step(a)
        env.pos = torch.where(alive[:, None], env.pos, pos_before)  # los ya terminados se quedan quietos
        for i in alive.nonzero().squeeze(1).tolist():
            paths[i].append(tuple(env.pos[i].tolist()))
        ok |= alive & (env.pos == env.goal).all(1)
        alive &= ~ok
    return paths, ok.tolist()


def box(ax, cell, color, lw=2.5):
    r, c = cell
    ax.add_patch(Rectangle((c * STRIDE, r * STRIDE), GLIMPSE, GLIMPSE, fill=False, ec="white", lw=lw + 2))
    ax.add_patch(Rectangle((c * STRIDE, r * STRIDE), GLIMPSE, GLIMPSE, fill=False, ec=color, lw=lw))


def trajectories(name, out, n=6, seed=0):
    ckpts = sorted(glob.glob(f"checkpoints/{name}_*.pt"), key=lambda p: int(p.rsplit("_", 1)[1][:-3]))
    picks = [ckpts[0], ckpts[len(ckpts) // 10], ckpts[len(ckpts) // 2], ckpts[-1]]
    _, (emb_va, n_va), obs_dim = make_task("embeddings", "", "cpu")
    ds = datasets.ImageFolder("data/imagenette2/val", transforms.Compose(
        [transforms.Resize(IMG), transforms.CenterCrop(IMG), transforms.ToTensor()]))

    g = torch.Generator().manual_seed(seed)
    img = torch.randperm(n_va, generator=g)[:n]
    start = torch.randint(GRID, (n, 2), generator=g)
    goal = (start + torch.randint(2, GRID - 1, (n, 2), generator=g)) % GRID  # objetivos a cierta distancia

    fig, axes = plt.subplots(n, len(picks) + 1, figsize=(2.6 * (len(picks) + 1), 2.75 * n))
    for j, ck in enumerate(picks):
        agent = Agent(obs_dim)
        agent.load_state_dict(torch.load(ck))
        paths, ok = rollout(agent, emb_va, n_va, img, start, goal, seed)
        steps = int(ck.rsplit("_", 1)[1][:-3])
        for i in range(n):
            ax = axes[i, j + 1]
            ax.imshow(ds[int(img[i])][0].permute(1, 2, 0))
            box(ax, start[i].tolist(), BLUE)
            box(ax, goal[i].tolist(), ORANGE)
            ys = [r * STRIDE + GLIMPSE / 2 for r, _ in paths[i]]
            xs = [c * STRIDE + GLIMPSE / 2 for _, c in paths[i]]
            ax.plot(xs, ys, color="white", lw=4, solid_capstyle="round")
            ax.plot(xs, ys, color=INK, lw=2, marker="o", ms=4, solid_capstyle="round")
            ax.plot(xs[-1], ys[-1], marker="o", ms=9, color=INK, mec="white", mew=2)
            ax.set_xlabel(f"{'llegó' if ok[i] else 'no llegó'} · {len(paths[i]) - 1} pasos", color=INK, fontsize=9)
            if i == 0:
                label = "sin entrenar" if steps == 0 else f"{steps / 1e6:.1f} M pasos"
                ax.set_title(label, fontsize=11, fontweight="bold", loc="left")
    for i in range(n):
        ax = axes[i, 0]
        r, c = goal[i].tolist()
        ax.imshow(ds[int(img[i])][0].permute(1, 2, 0)[r * STRIDE:r * STRIDE + GLIMPSE, c * STRIDE:c * STRIDE + GLIMPSE])
        if i == 0:
            ax.set_title("glimpse objetivo", fontsize=11, fontweight="bold", loc="left")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Azul: inicio · Naranja: objetivo · Línea: recorrido del agente (el punto grande es donde termina)",
                 x=0.01, ha="left", fontsize=10, color=INK2, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def examples(name, out, rows=3, cols=4, seed=1):
    """Agente final sobre imágenes de validación: una imagen por ejemplo, con el recorrido y su sentido."""
    ck = max(glob.glob(f"checkpoints/{name}_*.pt"), key=lambda p: int(p.rsplit("_", 1)[1][:-3]))
    _, (emb_va, n_va), obs_dim = make_task("embeddings", "", "cpu")
    ds = datasets.ImageFolder("data/imagenette2/val", transforms.Compose(
        [transforms.Resize(IMG), transforms.CenterCrop(IMG), transforms.ToTensor()]))
    n = rows * cols
    g = torch.Generator().manual_seed(seed)
    img = torch.randperm(n_va, generator=g)[:n]
    start = torch.randint(GRID, (n, 2), generator=g)
    goal = (start + torch.randint(1, GRID, (n, 2), generator=g)) % GRID
    agent = Agent(obs_dim)
    agent.load_state_dict(torch.load(ck))
    paths, ok = rollout(agent, emb_va, n_va, img, start, goal, seed)

    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.9 * rows))
    for i, ax in enumerate(axes.flat):
        ax.imshow(ds[int(img[i])][0].permute(1, 2, 0))
        box(ax, start[i].tolist(), BLUE)
        box(ax, goal[i].tolist(), ORANGE)
        pts = [(c * STRIDE + GLIMPSE / 2, r * STRIDE + GLIMPSE / 2) for r, c in paths[i]]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if (x0, y0) == (x1, y1):
                continue  # chocó con el borde
            for color, lw in (("white", 4.5), (INK, 2)):
                ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                            shrinkA=0, shrinkB=0, mutation_scale=11))
        ax.plot(*pts[0], "o", ms=9, color=BLUE, mec="white", mew=2)
        ax.plot(*pts[-1], "o", ms=9, color=INK, mec="white", mew=2)
        dist = int((start[i] - goal[i]).abs().sum())
        ax.set_title(f"{'Llegó' if ok[i] else 'No llegó'} · {len(paths[i]) - 1} pasos (mínimo {dist})",
                     fontsize=10, loc="left", color=INK, fontweight="bold" if ok[i] else "normal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("Recuadro azul: glimpse inicial · Recuadro naranja: glimpse objetivo · Flechas: recorrido · "
                 "Punto negro: dónde termina", x=0.01, ha="left", fontsize=10, color=INK2, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return sum(ok), n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    Path("figures").mkdir(exist_ok=True)
    runs = [json.loads(l) for l in open("results/ppo.jsonl")]
    run = [r for r in runs if r.get("name") == args.name][-1]
    curve(run, f"figures/{args.name}_curva.png")
    trajectories(args.name, f"figures/{args.name}_trayectorias.png")
    for seed in (1, 2):
        k, n = examples(args.name, f"figures/{args.name}_ejemplos_{seed}.png", seed=seed)
        print(f"ejemplos {seed}: llegó en {k} de {n}")
    print("figuras en figures/")
