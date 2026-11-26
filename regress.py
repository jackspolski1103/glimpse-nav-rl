"""Línea base sin RL: ajustar el encoder prediciendo directamente el desplazamiento al objetivo.

Misma señal que usa el agente (dónde está el objetivo respecto a mí), sin agente ni recompensa: dado el glimpse
actual (encoder en vivo) y el glimpse objetivo (encoder original congelado), una cabeza MLP predice (Δfila, Δcolumna).
Presupuesto igualado al de la etapa 2 de finetune.py: mismos glimpses nuevos (--samples), mismas tandas de 8192
reutilizadas 4 épocas en 4 minilotes, mismos LR, mismo decaimiento, mismo recorte de gradiente, mismos bloques libres.

Uso local (prueba): uv run python regress.py --limit 64 --samples 16384 --warm 50 --name prueba_reg
Salida en out/: <name>_encoder.pt, <name>_{train,val}.npz, <name>.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from finetune import Glimpses, SplitEncoder, embed_all, load_pixels
from glimpse import GRID, device, load_encoder
from ppo import layer


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/imagenette2")
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--warm", type=int, default=2000, help="pasos de la cabeza sola, encoder congelado")
    ap.add_argument("--samples", type=int, default=6_000_000, help="glimpses nuevos (= pasos de la etapa 2 con RL)")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--lr-enc", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--evals", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    dev, out, t0 = device(), Path("out"), time.time()
    out.mkdir(exist_ok=True)

    g_tr = Glimpses(load_pixels(args.data, "train", args.limit).to(dev))
    g_va = Glimpses(load_pixels(args.data, "val", args.limit).to(dev))
    n_tr, n_va = len(g_tr.pix), len(g_va.pix)
    base = load_encoder().eval().to(dev)
    e_tr, e_va = embed_all(base, g_tr, n_tr), embed_all(base, g_va, n_va)
    mu, sd = e_tr.mean((0, 1)), e_tr.std((0, 1)) + 1e-6
    t_tr, t_va = (e_tr - mu) / sd, (e_va - mu) / sd
    d = e_tr.shape[-1]
    head = nn.Sequential(layer(2 * d, 256), nn.Tanh(), layer(256, 256), nn.Tanh(), layer(256, 2)).to(dev)

    def pairs(n_img, n):  # imagen, casilla actual, casilla objetivo (distinta), desplazamiento normalizado
        img = torch.randint(n_img, (n,), device=dev)
        cur = torch.randint(GRID * GRID, (n,), device=dev)
        goal = (cur + torch.randint(1, GRID * GRID, (n,), device=dev)) % (GRID * GRID)
        y = torch.stack([goal // GRID - cur // GRID, goal % GRID - cur % GRID], 1).float() / (GRID - 1)
        return img, cur, goal, y

    opt = torch.optim.Adam(head.parameters(), lr=args.lr, eps=1e-5)
    for _ in range(args.warm):  # cabeza sola sobre la tabla del encoder original
        img, cur, goal, y = pairs(n_tr, 2048)
        loss = F.mse_loss(head(torch.cat([t_tr[img, cur], t_tr[img, goal]], 1)), y)
        opt.zero_grad(); loss.backward(); opt.step()

    enc = SplitEncoder(args.k).to(dev)
    params = [p for p in enc.parameters() if p.requires_grad]
    opt = torch.optim.Adam([{"params": list(head.parameters()), "lr": args.lr, "base": args.lr},
                            {"params": params, "lr": args.lr_enc, "base": args.lr_enc}], eps=1e-5)
    trunk = lambda g, img, cell: torch.cat([enc.trunk(g(img[i:i + 1024], cell[i:i + 1024]))
                                            for i in range(0, len(img), 1024)]).float()
    v_img, v_cur, v_goal, v_y = pairs(n_va, 4096)
    history, updates = [], args.samples // args.batch

    @torch.no_grad()
    def log(u):
        z = enc.tail(trunk(g_va, v_img, v_cur))
        pred = head(torch.cat([(z - mu) / sd, t_va[v_img, v_goal]], 1))
        m = {"samples": u * args.batch, "err_cells": ((pred - v_y).abs().sum(1).mean() * (GRID - 1)).item(),
             "x_cos": F.cosine_similarity(z, e_va[v_img, v_cur]).mean().item()}
        history.append(m)
        print(f"glimpses {m['samples']:>10,}  error {m['err_cells']:.3f} casillas  x_cos {m['x_cos']:.4f}  "
              f"({time.time() - t0:.0f} s)", flush=True)

    log(0)
    for u in range(1, updates + 1):
        for g in opt.param_groups:
            g["lr"] = g["base"] * (1 - (u - 1) / updates)
        img, cur, goal, y = pairs(n_tr, args.batch)
        tok, zg = trunk(g_tr, img, cur), t_tr[img, goal]
        for _ in range(4):
            for mb in torch.randperm(args.batch, device=dev).split(args.batch // 4):
                z = (enc.tail(tok[mb]) - mu) / sd
                loss = F.mse_loss(head(torch.cat([z, zg[mb]], 1)), y[mb])
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), 0.5); nn.utils.clip_grad_norm_(params, 0.5)
                opt.step()
        if u % max(1, updates // args.evals) == 0 or u == updates:
            log(u)

    torch.save(enc.m.state_dict(), out / f"{args.name}_encoder.pt")
    for split, g, n in (("train", g_tr, n_tr), ("val", g_va, n_va)):
        np.savez(out / f"{args.name}_{split}.npz", emb=embed_all(enc, g, n).cpu().numpy().astype(np.float16))
    (out / f"{args.name}.json").write_text(json.dumps({**vars(args), "history": history, "seconds": time.time() - t0}))
    print(f"listo en {time.time() - t0:.0f} s -> out/{args.name}_*", flush=True)


if __name__ == "__main__":
    main()
