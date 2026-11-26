"""Phase 2: fine-tuning the encoder with the PPO gradient (RL only).

Stage 1: the agent head is trained with the encoder frozen (embedding table).
Stage 2: the last encoder blocks are unfrozen; the current glimpse goes through the live encoder
         and the PPO loss reaches its weights. The goal glimpse still comes from the frozen
         original encoder (table).

Self-contained: starts from the pixels, not from cache/*.npz. Runs the same on the Mac or on Modal.
Local usage (smoke test): uv run python finetune.py --limit 64 --s1 60000 --s2 60000 --eval-envs 256 --name smoke
Output in out/: <name>_encoder.pt, <name>_agent.pt, <name>_{train,val}.npz (fine-tuned encoder embeddings), <name>.json
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from env import EGO, GlimpseNav
from glimpse import GLIMPSE, GRID, IMG, MEAN, STD, STRIDE, device, load_encoder
from ppo import evaluate, train


def load_pixels(root, split, limit=None):
    """Images resized to 448x448 as uint8 (N, 3, 448, 448), in ImageFolder order."""
    path = Path("cache") / f"pixels_{split}{f'_limit{limit}' if limit else ''}.npy"
    if path.exists():
        return torch.from_numpy(np.load(path))
    ds = datasets.ImageFolder(Path(root) / split, transforms.Compose(
        [transforms.Resize(IMG), transforms.CenterCrop(IMG), transforms.PILToTensor()]))
    if limit:
        ds = Subset(ds, range(0, len(ds), max(1, len(ds) // limit))[:limit])
    out = torch.cat([x for x, _ in DataLoader(ds, batch_size=64, num_workers=8)])
    path.parent.mkdir(exist_ok=True)
    np.save(path, out.numpy())
    return out


class Glimpses:
    """Crops glimpses from a pixel tensor on the device. The encoder only sees the crop."""

    def __init__(self, pixels):
        self.pix, dev = pixels, pixels.device
        self.ar, self.ch = torch.arange(GLIMPSE, device=dev), torch.arange(3, device=dev)[None, :, None, None]
        self.mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)

    def __call__(self, img, cell, chunk=512):
        if len(img) > chunk:  # the crop indices take a lot of memory: done in chunks
            return torch.cat([self(img[i:i + chunk], cell[i:i + chunk]) for i in range(0, len(img), chunk)])
        rows = (cell // GRID * STRIDE)[:, None] + self.ar
        cols = (cell % GRID * STRIDE)[:, None] + self.ar
        sub = self.pix.index_select(0, img)  # images first; the crop is done on this small copy
        b = torch.arange(len(img), device=img.device)[:, None, None, None]
        x = sub[b, self.ch, rows[:, None, :, None], cols[:, None, None, :]]
        return (x.float() / 255 - self.mean) / self.std


def amp(t):
    """Mixed precision on GPU (about 3x faster on an A10); no effect on CPU/MPS."""
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=t.is_cuda)


class SplitEncoder(torch.nn.Module):
    """ViT split into a frozen trunk + trainable tail (last k blocks and the final norm)."""

    def __init__(self, k_train):
        super().__init__()
        self.m = load_encoder().eval()
        self.split = len(self.m.blocks) - k_train
        self.m.requires_grad_(False)
        self.m.blocks[self.split:].requires_grad_(True)
        self.m.norm.requires_grad_(True)

    @torch.no_grad()
    def trunk(self, x):
        m = self.m
        with amp(x):
            x = m.norm_pre(m.patch_drop(m._pos_embed(m.patch_embed(x))))
            return m.blocks[:self.split](x)

    def tail(self, tok):
        with amp(tok):
            return self.m.norm(self.m.blocks[self.split:](tok))[:, 0].float()  # CLS token

    def forward(self, x):
        return self.tail(self.trunk(x))


class Live:
    """Adapter used by ppo.train: tokens in the rollout, embedding with gradient in the update."""

    def __init__(self, enc, glimpses, stats, half):
        self.enc, self.glimpses, (self.mu, self.sd) = enc, glimpses, stats
        self.dtype = torch.float16 if half else torch.float32
        self.d = len(self.mu)

    def params(self):
        return [p for p in self.enc.parameters() if p.requires_grad]

    def buffer(self, t, n):
        tokens = (GLIMPSE // 14) ** 2 + 1
        return torch.zeros(t, n, tokens, self.d, dtype=self.dtype, device=self.mu.device)

    def tokens(self, img, cell):
        return self.enc.trunk(self.glimpses(img, cell)).to(self.dtype)

    def fill(self, obs, tok):
        z = (self.enc.tail(tok.float()) - self.mu) / self.sd
        return torch.cat([z, obs[:, self.d:]], 1)

    def prep(self, env, obs):  # for evaluate()
        with torch.no_grad():
            return self.fill(obs, self.tokens(env.img, env.cell))


@torch.no_grad()
def embed_all(model, glimpses, n_images, bs=32):
    """Embedding (CLS token) of the 49 glimpses of each image -> (N, 49, D)."""
    dev = glimpses.pix.device
    cells = torch.arange(GRID * GRID, device=dev)
    out = []
    for i in range(0, n_images, bs):
        img = torch.arange(i, min(i + bs, n_images), device=dev)
        x = glimpses(img.repeat_interleave(len(cells)), cells.repeat(len(img)))
        with amp(x):
            z = model(x)
        # clone(): z is a view of the tensor of all tokens; without copying, each chunk would keep all of it alive
        out.append(z.float().reshape(len(img), len(cells), -1).clone())
    return torch.cat(out)


def mem(tag):
    if torch.cuda.is_available():
        print(f"[GPU memory] {tag}: {torch.cuda.memory_allocated() / 2**30:.1f} GB in use, "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/imagenette2")
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, help="only N images per split (smoke test)")
    ap.add_argument("--s1", type=int, default=2_000_000, help="stage 1 steps (frozen encoder)")
    ap.add_argument("--s2", type=int, default=10_000_000, help="stage 2 steps (live encoder)")
    ap.add_argument("--k", type=int, default=2, help="final encoder blocks that are trained")
    ap.add_argument("--lr-enc", type=float, default=1e-5)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--eval-envs", type=int, default=4096)
    ap.add_argument("--evals", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    dev = device()
    out = Path("out")
    out.mkdir(exist_ok=True)
    t0 = time.time()

    g_tr = Glimpses(load_pixels(args.data, "train", args.limit).to(dev))
    g_va = Glimpses(load_pixels(args.data, "val", args.limit).to(dev))
    n_tr, n_va = len(g_tr.pix), len(g_va.pix)
    print(f"pixels: train {n_tr}, val {n_va}  ({time.time() - t0:.0f} s)  device {dev}", flush=True)

    mem("pixels loaded")
    base = load_encoder().eval().to(dev)  # original encoder: goals and statistics
    e_tr, e_va = embed_all(base, g_tr, n_tr), embed_all(base, g_va, n_va)
    mem("original embeddings")
    stats = (e_tr.mean((0, 1)), e_tr.std((0, 1)) + 1e-6)
    t_tr, t_va = (e_tr - stats[0]) / stats[1], (e_va - stats[0]) / stats[1]
    d = e_tr.shape[-1]
    obs_dim = 2 * d + EGO * EGO + 4
    mk = lambda table, n_img, n: GlimpseNav(lambda i, c: table[i, c], n_img, n, dev)
    env, ev_va, ev_tr = mk(t_tr, n_tr, args.envs), mk(t_va, n_va, args.eval_envs), mk(t_tr, n_tr, args.eval_envs)

    print("== stage 1: agent head, frozen encoder ==", flush=True)
    agent, h1 = train(env, ev_va, obs_dim, args.s1, evals=5, train_eval_env=ev_tr)

    mem("end of stage 1")
    print(f"== stage 2: live encoder, last {args.k} blocks, lr {args.lr_enc} ==", flush=True)
    enc = SplitEncoder(args.k).to(dev)
    half = dev == "cuda"
    live_tr, live_va = Live(enc, g_tr, stats, half), Live(enc, g_va, stats, half)
    probe_img = torch.randint(n_va, (2048,), device=dev)
    probe_cell = torch.randint(GRID * GRID, (2048,), device=dev)

    def drift():  # how far the fine-tuned embedding drifted from the original, on validation glimpses
        with torch.no_grad():
            z = enc(g_va(probe_img, probe_cell))
        return {"x_cos": F.cosine_similarity(z, e_va[probe_img, probe_cell]).mean().item()}

    agent, h2 = train(env, ev_va, obs_dim, args.s2, evals=args.evals, train_eval_env=ev_tr, agent=agent,
                      live=live_tr, lr_enc=args.lr_enc, eval_prep=live_va.prep, train_eval_prep=live_tr.prep,
                      extra_log=drift)

    mem("end of stage 2")
    torch.save(enc.m.state_dict(), out / f"{args.name}_encoder.pt")
    torch.save(agent.state_dict(), out / f"{args.name}_agent.pt")
    for split, g, n, e in (("train", g_tr, n_tr, e_tr), ("val", g_va, n_va, e_va)):
        np.savez(out / f"{args.name}_{split}.npz", emb=embed_all(enc, g, n).cpu().numpy().astype(np.float16))
        if not (out / f"base_{split}.npz").exists():  # original encoder through the same path, for experiment A
            np.savez(out / f"base_{split}.npz", emb=e.cpu().numpy().astype(np.float16))
    (out / f"{args.name}.json").write_text(json.dumps(
        {**vars(args), "stage1": h1, "stage2": h2, "seconds": time.time() - t0}))
    print(f"done in {time.time() - t0:.0f} s -> out/{args.name}_*", flush=True)


if __name__ == "__main__":
    main()
