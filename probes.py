"""Phase 0: how much spatial information is there in the frozen embeddings?

Usage: uv run python probes.py [--suffix _limit200] [--emb out/ft_k2_lr1e-5_]
Trains on train images and evaluates on val images (unseen).
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from glimpse import GRID, device

DEV = device()
MOVES = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]], device=DEV)  # up, down, left, right


def load(split: str, suffix: str, prefix: str = "cache/") -> torch.Tensor:
    return torch.from_numpy(np.load(f"{prefix}{split}{suffix}.npz")["emb"].astype(np.float32)).to(DEV)


def net(d_in: int, d_out: int, hidden: int = 0) -> nn.Module:
    if not hidden:
        return nn.Linear(d_in, d_out).to(DEV)
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, d_out)).to(DEV)


def rc(n: int):
    return torch.randint(GRID, (n,), device=DEV), torch.randint(GRID, (n,), device=DEV)


def batch_position(E, n):
    i, (r, c) = torch.randint(len(E), (n,), device=DEV), rc(n)
    p = r * GRID + c
    return E[i, p], p


def batch_neighbor(E, n):
    """Pair (current, neighbour) -> in which direction is the neighbour?"""
    i, (r, c) = torch.randint(len(E), (n,), device=DEV), rc(n)
    d = torch.randint(4, (n,), device=DEV)
    r2, c2 = r + MOVES[d, 0], c + MOVES[d, 1]
    ok = (r2 >= 0) & (r2 < GRID) & (c2 >= 0) & (c2 < GRID)
    i, r, c, r2, c2, d = i[ok], r[ok], c[ok], r2[ok], c2[ok], d[ok]
    return torch.cat([E[i, r * GRID + c], E[i, r2 * GRID + c2]], 1), d


def batch_offset(E, n):
    """Any pair from the same image -> offset (rows, columns)."""
    i, (r, c), (r2, c2) = torch.randint(len(E), (n,), device=DEV), rc(n), rc(n)
    x = torch.cat([E[i, r * GRID + c], E[i, r2 * GRID + c2]], 1)
    return x, torch.stack([r2 - r, c2 - c], 1).float()


def fit(model, batch_fn, loss_fn, train, val, steps, bs=4096):
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
    for _ in range(steps):
        x, y = batch_fn(train, bs)
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        x, y = batch_fn(val, 50_000)
        return model(x), y


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--emb", default="cache/", help="embeddings prefix (as in ppo.py)")
    args = ap.parse_args(argv)
    torch.manual_seed(0)
    res = {}

    train, val = load("train", args.suffix, args.emb), load("val", args.suffix, args.emb)
    mu, sd = train.mean((0, 1)), train.std((0, 1)) + 1e-6
    train, val = (train - mu) / sd, (val - mu) / sd
    d = train.shape[-1]
    acc = lambda out, y: (out.argmax(1) == y).float().mean().item() * 100

    print(f"train/val images: {len(train)}/{len(val)}  dim: {d}  device: {DEV}")

    out, y = fit(net(d, GRID * GRID), batch_position, F.cross_entropy, train, val, args.steps)
    res["position"] = acc(out, y)
    print(f"absolute position from 1 glimpse (linear):   {acc(out, y):5.1f} %   chance {100 / GRID**2:.1f} %")

    for name, h in (("linear", 0), ("MLP", 512)):
        out, y = fit(net(2 * d, 4, h), batch_neighbor, F.cross_entropy, train, val, args.steps)
        res[f"neighbour_{name.lower()}"] = acc(out, y)
        print(f"neighbour direction from 2 glimpses ({name}): {acc(out, y):5.1f} %   chance 25.0 %")

    out, y = fit(net(2 * d, 2, 512), batch_offset, F.mse_loss, train, val, args.steps)
    err = (out.round() - y).abs().sum(1).mean().item()
    base = y.abs().sum(1).mean().item()
    exact = (out.round() == y).all(1).float().mean().item() * 100
    print(f"offset between 2 glimpses (MLP): error {err:.2f} cells (no information: {base:.2f}), exact {exact:.1f} %")
    return {**res, "offset_err": err, "offset_exact": exact}


if __name__ == "__main__":
    main()
