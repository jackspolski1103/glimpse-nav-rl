"""Experimento B: ¿perdió información general el encoder ajustado?

Imagen completa (224 px), encoder congelado, como se evalúa DINOv2. Mismo script para ambos encoders.
- k-NN (k=20, coseno, voto ponderado) sobre el token CLS: Imagenette y Imagewoof (no usado en el ajuste).
- Similitud entre los embeddings del encoder original y los del ajustado sobre las mismas imágenes (coseno y CKA lineal).

Uso: uv run python eval_repr.py out/ft_k2_lr1e-5_encoder.pt [más encoders...]
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from glimpse import MEAN, STD, device, load_encoder

DATASETS = {"imagenette": "data/imagenette2", "imagewoof": "data/imagewoof2"}
TF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                         transforms.Normalize(MEAN, STD)])


@torch.no_grad()
def features(models, root, split):
    """Una sola pasada por los datos para todos los encoders. -> {nombre: (N, D)}, etiquetas"""
    dev = device()
    feats, labels = {k: [] for k in models}, []
    for x, y in DataLoader(datasets.ImageFolder(Path(root) / split, TF), batch_size=128, num_workers=8):
        x = x.to(dev)
        for k, m in models.items():
            feats[k].append(m(x).cpu())
        labels.append(y)
    return {k: torch.cat(v) for k, v in feats.items()}, torch.cat(labels)


def knn(train, ytr, test, yte, k=20, temp=0.07):
    train, test = F.normalize(train, dim=1), F.normalize(test, dim=1)
    sim, idx = (test @ train.T).topk(k, dim=1)
    votes = torch.zeros(len(test), int(ytr.max()) + 1).scatter_add_(1, ytr[idx], (sim / temp).exp())
    return (votes.argmax(1) == yte).float().mean().item() * 100


def cka(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    return ((a.T @ b).norm() ** 2 / ((a.T @ a).norm() * (b.T @ b).norm())).item()


def main():
    dev = device()
    models = {"original": load_encoder().eval().to(dev)}
    for path in sys.argv[1:]:
        m = load_encoder()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        models[Path(path).name.replace("_encoder.pt", "")] = m.eval().to(dev)

    results = {k: {} for k in models}
    for ds, root in DATASETS.items():
        if not Path(root).exists():
            print(f"(sin {ds}: falta {root})")
            continue
        ftr, ytr = features(models, root, "train")
        fva, yva = features(models, root, "val")
        for k in models:
            results[k][f"knn_{ds}"] = knn(ftr[k], ytr, fva[k], yva)
            results[k][f"cos_{ds}"] = F.cosine_similarity(fva[k], fva["original"]).mean().item()
            results[k][f"cka_{ds}"] = cka(fva[k], fva["original"])
    for k, r in results.items():
        print(k, json.dumps({m: round(v, 4) for m, v in r.items()}))
    Path("results").mkdir(exist_ok=True)
    with open("results/eval_repr.jsonl", "a") as fh:
        fh.write(json.dumps(results) + "\n")


if __name__ == "__main__":
    main()
