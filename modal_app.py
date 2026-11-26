"""Lanza finetune.py en GPUs de Modal (fase 2).

Uso (varias corridas en paralelo, separadas por ';', cada una 'nombre:argumentos'):
  uv run modal run --detach modal_app.py --runs "ft_k2_lr1e-5:--k 2 --lr-enc 1e-5;ft_k2_lr3e-5:--k 2 --lr-enc 3e-5"

Los resultados quedan en el Volume 'glimpse-nav-rl', carpeta out/:
  uv run modal volume get glimpse-nav-rl out/<archivo> out/
"""
import modal

app = modal.App("glimpse-nav-rl")
vol = modal.Volume.from_name("glimpse-nav-rl", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl")
    .uv_pip_install("torch==2.15.0", "torchvision==0.29.2", "timm==1.0.31", "numpy", "gymnasium", "tqdm")
    .add_local_python_source("glimpse", "env", "ppo", "finetune")
)
DATA_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"


@app.function(image=image, volumes={"/vol": vol}, timeout=3600, cpu=8, memory=32768)
def prepare():
    """Una sola vez: descarga Imagenette y guarda los píxeles reescalados en el Volume."""
    import os
    import subprocess
    from pathlib import Path

    os.chdir("/vol")
    if Path("cache/pixels_train.npy").exists() and Path("cache/pixels_val.npy").exists():
        print("píxeles ya preparados")
        return
    subprocess.run(f"curl -sSfL --retry 10 {DATA_URL} | tar -xz -C /tmp", shell=True, check=True)
    from finetune import load_pixels

    for split in ("train", "val"):
        print(split, load_pixels("/tmp/imagenette2", split).shape, flush=True)
    vol.commit()


@app.function(image=image, gpu="A10", volumes={"/vol": vol}, timeout=4 * 3600, cpu=4, memory=32768)
def run(argv: list[str]):
    import os

    os.chdir("/vol")
    import finetune

    finetune.main(argv)
    vol.commit()


@app.local_entrypoint()
def main(runs: str):
    prepare.remote()
    for spec in runs.split(";"):
        name, _, extra = spec.partition(":")
        call = run.spawn(["--name", name, *extra.split()])
        print(f"lanzado {name}: {call.object_id}")
