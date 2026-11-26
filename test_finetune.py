"""Phase 2 checks. Usage: uv run python test_finetune.py"""
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
    print("ok: trunk + tail reproduces the full encoder")

    # the on-device crop gives the same embeddings as precompute.py (phase 0)
    g = Glimpses(load_pixels("data/imagenette2", "train", 200))
    e = embed_all(load_encoder().eval(), g, 8)
    ref = torch.from_numpy(np.load("cache/train_limit200.npz")["emb"][:8].astype(np.float32))
    cos = torch.nn.functional.cosine_similarity(e, ref, dim=-1).min().item()
    assert cos > 0.999, cos
    print(f"ok: same embeddings as the phase 0 cache (minimum cosine {cos:.5f})")

    # the loss gradient reaches the encoder tail and only the tail
    live = Live(enc, g, (torch.zeros(384), torch.ones(384)), half=False)
    img, cell = torch.arange(4), torch.arange(4)
    obs = torch.zeros(4, 2 * 384 + 10)
    live.fill(obs, live.tokens(img, cell)).sum().backward()
    assert all(p.grad is not None for p in live.params())
    assert all(p.grad is None for p in enc.m.blocks[:enc.split].parameters())
    print("ok: the gradient reaches only the unfrozen blocks")


if __name__ == "__main__":
    main()
