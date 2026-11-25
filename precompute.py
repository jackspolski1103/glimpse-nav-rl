"""Fase 0: embeddings de los 49 glimpses de cada imagen con el encoder congelado.

Uso: uv run python precompute.py [--limit N]
Salida: cache/{train,val}.npz con emb (N, 49, 384) float16 y label (N,).
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from glimpse import IMG, MEAN, STD, device, extract_glimpses, load_encoder

DATA, CACHE = Path("data/imagenette2"), Path("cache")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="solo las primeras N imágenes (prueba rápida)")
    ap.add_argument("--batch", type=int, default=16, help="imágenes por lote (x49 glimpses)")
    args = ap.parse_args()

    dev = device()
    enc = load_encoder().eval().to(dev)
    tf = transforms.Compose([
        transforms.Resize(IMG), transforms.CenterCrop(IMG),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    CACHE.mkdir(exist_ok=True)
    for split in ("train", "val"):
        ds = datasets.ImageFolder(DATA / split, tf)
        if args.limit:
            ds = Subset(ds, range(0, len(ds), max(1, len(ds) // args.limit))[: args.limit])
        embs, labels = [], []
        for img, y in tqdm(DataLoader(ds, batch_size=args.batch, num_workers=4), desc=split):
            g = extract_glimpses(img).flatten(0, 1).to(dev)  # (B*49, 3, 112, 112)
            z = enc(g).reshape(len(img), -1, enc.num_features)  # token CLS de cada glimpse
            embs.append(z.cpu().numpy().astype(np.float16))
            labels.append(y.numpy())
        out = CACHE / (f"{split}.npz" if not args.limit else f"{split}_limit{args.limit}.npz")
        np.savez(out, emb=np.concatenate(embs), label=np.concatenate(labels))
        print(out, np.concatenate(embs).shape)


if __name__ == "__main__":
    main()
