"""The embedding of a glimpse must not depend on pixels outside that glimpse.

Usage: uv run python test_no_leak.py
"""
import torch

from glimpse import GLIMPSE, GRID, IMG, STRIDE, extract_glimpses, load_encoder


@torch.no_grad()
def embed(enc, img):
    return enc(extract_glimpses(img).flatten(0, 1))  # same path as precompute.py


def test_no_leak():
    torch.manual_seed(0)
    enc = load_encoder().eval()
    img = torch.randn(1, 3, IMG, IMG)
    base = embed(enc, img)
    for r, c in ((0, 0), (3, 3), (6, 2)):
        k = r * GRID + c
        other = torch.randn(1, 3, IMG, IMG)  # everything different...
        ys, xs = slice(r * STRIDE, r * STRIDE + GLIMPSE), slice(c * STRIDE, c * STRIDE + GLIMPSE)
        other[:, :, ys, xs] = img[:, :, ys, xs]  # ...except the pixels of glimpse k
        z = embed(enc, other)
        assert torch.allclose(z[k], base[k], atol=1e-4), f"leak in glimpse {k}"
        far = (GRID - 1 - r) * GRID + (GRID - 1 - c) if (r, c) != (3, 3) else 0
        assert not torch.allclose(z[far], base[far], atol=1e-2)  # the others do change


if __name__ == "__main__":
    test_no_leak()
    print("ok: the embedding of each glimpse depends only on its own pixels")
