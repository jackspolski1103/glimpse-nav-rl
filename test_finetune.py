"""Comprobaciones de la fase 2. Uso: uv run python test_finetune.py"""
import numpy as np
import torch

from finetune import Glimpses, Live, SplitEncoder, embed_all, load_pixels
from glimpse import GLIMPSE, GRID, load_encoder


def main():
    torch.manual_seed(0)
    enc = SplitEncoder(2)
    x = torch.randn(4, 3, GLIMPSE, GLIMPSE)
    with torch.no_grad():
        assert torch.allclose(enc(x), enc.m(x), atol=1e-4)
    print("ok: tronco + cola reproduce el encoder completo")

    # el recorte en el dispositivo da los mismos embeddings que precompute.py (fase 0)
    g = Glimpses(load_pixels("data/imagenette2", "train", 200))
    e = embed_all(load_encoder().eval(), g, 8)
    ref = torch.from_numpy(np.load("cache/train_limit200.npz")["emb"][:8].astype(np.float32))
    cos = torch.nn.functional.cosine_similarity(e, ref, dim=-1).min().item()
    assert cos > 0.999, cos
    print(f"ok: mismos embeddings que la caché de la fase 0 (coseno mínimo {cos:.5f})")

    # el gradiente de la pérdida llega a la cola del encoder y solo a ella
    live = Live(enc, g, (torch.zeros(384), torch.ones(384)), half=False)
    img, cell = torch.arange(4), torch.arange(4)
    obs = torch.zeros(4, 2 * 384 + 10)
    live.fill(obs, live.tokens(img, cell)).sum().backward()
    assert all(p.grad is not None for p in live.params())
    assert all(p.grad is None for p in enc.m.blocks[:enc.split].parameters())
    print("ok: el gradiente llega solo a los bloques liberados")


if __name__ == "__main__":
    main()
