"""Fase 2: fine-tuning del encoder con el gradiente de PPO (solo RL).

Etapa 1: se entrena la cabeza del agente con el encoder congelado (tabla de embeddings).
Etapa 2: se liberan los últimos bloques del encoder; el glimpse actual pasa por el encoder en vivo
         y la pérdida de PPO llega a sus pesos. El glimpse objetivo sigue viniendo del encoder
         original congelado (tabla).

Autocontenido: parte de los píxeles, no de cache/*.npz. Corre igual en el Mac o en Modal.
Uso local (prueba): uv run python finetune.py --limit 64 --s1 60000 --s2 60000 --eval-envs 256 --name prueba
Salida en out/: <name>_encoder.pt, <name>_agent.pt, <name>_{train,val}.npz (embeddings del encoder ajustado), <name>.json
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
    """Imágenes reescaladas a 448x448 como uint8 (N, 3, 448, 448), en el orden de ImageFolder."""
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
    """Recorta glimpses de un tensor de píxeles en el dispositivo. El encoder solo ve el recorte."""

    def __init__(self, pixels):
        self.pix, dev = pixels, pixels.device
        self.ar, self.ch = torch.arange(GLIMPSE, device=dev), torch.arange(3, device=dev)[None, :, None, None]
        self.mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)

    def __call__(self, img, cell):
        rows = (cell // GRID * STRIDE)[:, None] + self.ar
        cols = (cell % GRID * STRIDE)[:, None] + self.ar
        x = self.pix[img[:, None, None, None], self.ch, rows[:, None, :, None], cols[:, None, None, :]]
        return (x.float() / 255 - self.mean) / self.std


class SplitEncoder(torch.nn.Module):
    """ViT partido en tronco congelado + cola entrenable (últimos k bloques y la norma final)."""

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
        x = m.norm_pre(m.patch_drop(m._pos_embed(m.patch_embed(x))))
        return m.blocks[:self.split](x)

    def tail(self, tok):
        return self.m.norm(self.m.blocks[self.split:](tok))[:, 0]  # token CLS

    def forward(self, x):
        return self.tail(self.trunk(x))


class Live:
    """Adaptador que usa ppo.train: tokens en el rollout, embedding con gradiente en la actualización."""

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

    def prep(self, env, obs):  # para evaluate()
        with torch.no_grad():
            return self.fill(obs, self.tokens(env.img, env.cell))


@torch.no_grad()
def embed_all(model, glimpses, n_images, bs=64):
    """Embedding (token CLS) de los 49 glimpses de cada imagen -> (N, 49, D)."""
    dev = glimpses.pix.device
    cells = torch.arange(GRID * GRID, device=dev)
    out = []
    for i in range(0, n_images, bs):
        img = torch.arange(i, min(i + bs, n_images), device=dev)
        z = model(glimpses(img.repeat_interleave(len(cells)), cells.repeat(len(img))))
        out.append(z.reshape(len(img), len(cells), -1))
    return torch.cat(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/imagenette2")
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, help="solo N imágenes por split (prueba)")
    ap.add_argument("--s1", type=int, default=2_000_000, help="pasos de la etapa 1 (encoder congelado)")
    ap.add_argument("--s2", type=int, default=10_000_000, help="pasos de la etapa 2 (encoder en vivo)")
    ap.add_argument("--k", type=int, default=2, help="bloques finales del encoder que se entrenan")
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
    print(f"píxeles: train {n_tr}, val {n_va}  ({time.time() - t0:.0f} s)  dispositivo {dev}", flush=True)

    base = load_encoder().eval().to(dev)  # encoder original: objetivos y estadísticas
    e_tr, e_va = embed_all(base, g_tr, n_tr), embed_all(base, g_va, n_va)
    stats = (e_tr.mean((0, 1)), e_tr.std((0, 1)) + 1e-6)
    t_tr, t_va = (e_tr - stats[0]) / stats[1], (e_va - stats[0]) / stats[1]
    d = e_tr.shape[-1]
    obs_dim = 2 * d + EGO * EGO + 4
    mk = lambda table, n_img, n: GlimpseNav(lambda i, c: table[i, c], n_img, n, dev)
    env, ev_va, ev_tr = mk(t_tr, n_tr, args.envs), mk(t_va, n_va, args.eval_envs), mk(t_tr, n_tr, args.eval_envs)

    print("== etapa 1: cabeza del agente, encoder congelado ==", flush=True)
    agent, h1 = train(env, ev_va, obs_dim, args.s1, evals=5, train_eval_env=ev_tr)

    print(f"== etapa 2: encoder en vivo, últimos {args.k} bloques, lr {args.lr_enc} ==", flush=True)
    enc = SplitEncoder(args.k).to(dev)
    half = dev == "cuda"
    live_tr, live_va = Live(enc, g_tr, stats, half), Live(enc, g_va, stats, half)
    probe_img = torch.randint(n_va, (2048,), device=dev)
    probe_cell = torch.randint(GRID * GRID, (2048,), device=dev)

    def drift():  # cuánto se alejó el embedding ajustado del original, en glimpses de validación
        with torch.no_grad():
            z = enc(g_va(probe_img, probe_cell))
        return {"x_cos": F.cosine_similarity(z, e_va[probe_img, probe_cell]).mean().item()}

    agent, h2 = train(env, ev_va, obs_dim, args.s2, evals=args.evals, train_eval_env=ev_tr, agent=agent,
                      live=live_tr, lr_enc=args.lr_enc, eval_prep=live_va.prep, train_eval_prep=live_tr.prep,
                      extra_log=drift)

    torch.save(enc.m.state_dict(), out / f"{args.name}_encoder.pt")
    torch.save(agent.state_dict(), out / f"{args.name}_agent.pt")
    for split, g, n in (("train", g_tr, n_tr), ("val", g_va, n_va)):
        np.savez(out / f"{args.name}_{split}.npz", emb=embed_all(enc, g, n).cpu().numpy().astype(np.float16))
    (out / f"{args.name}.json").write_text(json.dumps(
        {**vars(args), "stage1": h1, "stage2": h2, "seconds": time.time() - t0}))
    print(f"listo en {time.time() - t0:.0f} s -> out/{args.name}_*", flush=True)


if __name__ == "__main__":
    main()
