"""El embedding de un glimpse no puede depender de píxeles fuera de ese glimpse.

Uso: uv run python test_no_leak.py
"""
import torch

from glimpse import GLIMPSE, GRID, IMG, STRIDE, extract_glimpses, load_encoder


@torch.no_grad()
def embed(enc, img):
    return enc(extract_glimpses(img).flatten(0, 1))  # mismo camino que precompute.py


def test_no_leak():
    torch.manual_seed(0)
    enc = load_encoder().eval()
    img = torch.randn(1, 3, IMG, IMG)
    base = embed(enc, img)
    for r, c in ((0, 0), (3, 3), (6, 2)):
        k = r * GRID + c
        other = torch.randn(1, 3, IMG, IMG)  # todo distinto...
        ys, xs = slice(r * STRIDE, r * STRIDE + GLIMPSE), slice(c * STRIDE, c * STRIDE + GLIMPSE)
        other[:, :, ys, xs] = img[:, :, ys, xs]  # ...salvo los píxeles del glimpse k
        z = embed(enc, other)
        assert torch.allclose(z[k], base[k], atol=1e-4), f"fuga en el glimpse {k}"
        far = (GRID - 1 - r) * GRID + (GRID - 1 - c) if (r, c) != (3, 3) else 0
        assert not torch.allclose(z[far], base[far], atol=1e-2)  # los demás sí cambian


if __name__ == "__main__":
    test_no_leak()
    print("ok: el embedding de cada glimpse depende solo de sus propios píxeles")
