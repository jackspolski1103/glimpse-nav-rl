"""Same validation episodes with the original encoder and with the RL fine-tuned one (LR 3e-5).

Before: frozen_bump_s0 agent on the original encoder. After: new agent from experiment A on the fine-tuned
encoder (out/expA_ft_k2_lr3e-5_s0_agent.pt). Both trained for 10 M steps from scratch.
Usage: uv run python fig_before_after.py -> figures/before_after.png (summary) and figures/before_after_examples.png
"""
import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

from glimpse import GLIMPSE, GRID, IMG, STRIDE
from ppo import Agent, make_task, table_embed
from viz import BLUE, GRIDC, INK, INK2, ORANGE, box, rollout

FT, GRAY = "ft_k2_lr3e-5", "#8a8f98"
_, (emb0, n_va), obs_dim = make_task("embeddings", "", "cpu")
*_, stats = table_embed("train", "", "cpu", prefix=f"out/{FT}_")
emb1 = table_embed("val", "", "cpu", stats, prefix=f"out/{FT}_")[0]
agents = []
for path in (max(glob.glob("checkpoints/frozen_bump_s0_*.pt"), key=lambda p: int(p.rsplit("_", 1)[1][:-3])),
             f"out/expA_{FT}_s0_agent.pt"):
    a = Agent(obs_dim)
    a.load_state_dict(torch.load(path, map_location="cpu"))
    agents.append(a)

n = 5000
g = torch.Generator().manual_seed(0)
img = torch.randint(n_va, (n,), generator=g)
start = torch.randint(GRID, (n, 2), generator=g)
goal = (start + torch.randint(1, GRID, (n, 2), generator=g)) % GRID
(p0, ok0), (p1, ok1) = (rollout(a, e, n_va, img, start, goal, 0) for a, e in zip(agents, (emb0, emb1)))
ok0, ok1 = np.array(ok0), np.array(ok1)
dist = (start - goal).abs().sum(1).numpy()
cats = [("Both reach it", ok0 & ok1), ("Only with the fine-tuned", ~ok0 & ok1),
        ("Only with the original", ok0 & ~ok1), ("Neither reaches it", ~ok0 & ~ok1)]
print(f"{n} identical episodes: original {ok0.mean() * 100:.1f} %, fine-tuned {ok1.mean() * 100:.1f} %")
for name, m in cats:
    print(f"  {name}: {m.mean() * 100:.1f} %")
both = ok0 & ok1
extra = lambda p: np.array([len(x) - 1 for x in p])[both] - dist[both]
print(f"  extra steps when both reach it: original {extra(p0).mean():.2f}, fine-tuned {extra(p1).mean():.2f}")

# Summary
fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.6))
vals = [m.mean() * 100 for _, m in cats]
a.barh([c for c, _ in cats][::-1], vals[::-1], color=[GRAY, GRAY, BLUE, GRAY], height=0.6)
for y, v in enumerate(vals[::-1]):
    a.text(v + 1, y, f"{v:.1f} %", va="center", fontsize=10)
a.set_xlim(0, 75); a.set_xlabel("% of episodes")
a.set_title(f"{n} identical validation episodes", loc="left", fontsize=11, fontweight="bold")
ds_ = sorted(set(dist))
bins = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 12)]
for ok, color, label in ((ok0, GRAY, "original"), (ok1, BLUE, "RL fine-tuned")):
    y = [ok[(dist >= lo) & (dist <= hi)].mean() * 100 for lo, hi in bins]
    b.plot(range(len(bins)), y, "-o", color=color, lw=2, ms=7, mec="white", mew=2)
    b.text(len(bins) - 0.85, y[-1], f"{label}  {y[-1]:.0f} %", va="center", fontsize=9)
b.set_xticks(range(len(bins)), [f"{lo}–{hi}" for lo, hi in bins]); b.set_xlim(-0.3, len(bins) + 0.9); b.set_ylim(40, 100)
b.set_xlabel("initial distance to the goal (cells)"); b.set_ylabel("success (%)")
b.set_title("Success by initial distance", loc="left", fontsize=11, fontweight="bold")
for ax in (a, b):
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x" if ax is a else "y", color=GRIDC, lw=0.8)
    ax.tick_params(length=0)
fig.tight_layout(); fig.savefig("figures/before_after.png", dpi=160)

# Examples: the first of each case, in the order they came up (not hand-picked)
picks = [(name, int(i)) for (name, m), k in zip(cats, (2, 2, 1, 1)) for i in np.flatnonzero(m)[:k]]
ds = datasets.ImageFolder("data/imagenette2/val", transforms.Compose(
    [transforms.Resize(IMG), transforms.CenterCrop(IMG), transforms.ToTensor()]))
fig, axes = plt.subplots(2, len(picks), figsize=(3.1 * len(picks), 7.2))
for row, (label, paths, ok) in enumerate((("Original encoder", p0, ok0), ("RL fine-tuned encoder", p1, ok1))):
    for col, (name, i) in enumerate(picks):
        ax = axes[row, col]
        ax.imshow(ds[int(img[i])][0].permute(1, 2, 0))
        box(ax, start[i].tolist(), BLUE)
        box(ax, goal[i].tolist(), ORANGE)
        pts = [(c * STRIDE + GLIMPSE / 2, r * STRIDE + GLIMPSE / 2) for r, c in paths[i]]
        for q0, q1 in zip(pts, pts[1:]):
            if q0 != q1:  # if they are equal, it bumped into the border
                for color, lw in (("white", 4.5), (INK, 2)):
                    ax.annotate("", q1, q0, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0,
                                                            shrinkB=0, mutation_scale=11))
        ax.plot(*pts[-1], "o", ms=9, color=INK, mec="white", mew=2)
        ax.set_xlabel(f"{'Reached' if ok[i] else 'Not reached'} · {len(paths[i]) - 1} steps (minimum {dist[i]})",
                      fontsize=9, color=INK, fontweight="bold" if ok[i] else "normal")
        if row == 0:
            ax.set_title(f"{name}\n({dict(zip([c for c, _ in cats], vals))[name]:.0f} % of episodes)",
                         fontsize=10, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[row, 0].set_ylabel(label, fontsize=11, color=INK, fontweight="bold")
fig.suptitle("Blue: start · Orange: goal · Arrows: path · Black dot: end",
             x=0.01, ha="left", fontsize=10, color=INK2, y=1.0)
fig.tight_layout(h_pad=2.5)
fig.savefig("figures/before_after_examples.png", dpi=100)
