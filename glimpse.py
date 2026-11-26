"""Glimpse grid and encoder. Everything else imports from here."""
import timm
import torch

IMG = 448      # side of the resized image
GLIMPSE = 112  # glimpse side (8x8 patches of 14 px)
STRIDE = 56    # stride between neighbouring glimpses (50 % overlap)
GRID = (IMG - GLIMPSE) // STRIDE + 1  # 7 -> 49 positions
MODEL = "vit_small_patch14_dinov2.lvd142m"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_encoder() -> torch.nn.Module:
    # dynamic_img_size: the same model accepts 112 px glimpses and full images.
    return timm.create_model(MODEL, pretrained=True, num_classes=0, dynamic_img_size=True)


def extract_glimpses(img: torch.Tensor) -> torch.Tensor:
    """(B, 3, IMG, IMG) -> (B, GRID*GRID, 3, GLIMPSE, GLIMPSE), in row-column order."""
    b = img.shape[0]
    g = img.unfold(2, GLIMPSE, STRIDE).unfold(3, GLIMPSE, STRIDE)  # B,3,GRID,GRID,G,G
    return g.permute(0, 2, 3, 1, 4, 5).reshape(b, GRID * GRID, 3, GLIMPSE, GLIMPSE)


if __name__ == "__main__":
    x = torch.rand(2, 3, IMG, IMG)
    g = extract_glimpses(x)
    assert GRID == 7 and g.shape == (2, 49, 3, GLIMPSE, GLIMPSE)
    r, c = 3, 5
    assert torch.equal(g[1, r * GRID + c], x[1, :, r * STRIDE:r * STRIDE + GLIMPSE, c * STRIDE:c * STRIDE + GLIMPSE])
    print("ok")
