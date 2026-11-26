"""Lanza finetune.py en GPUs de Modal (fase 2).

Uso (varias corridas en paralelo, separadas por ';', cada una 'nombre:argumentos'):
  uv run modal run --detach modal_app.py::main --runs "ft_k2_lr1e-5:--k 2 --lr-enc 1e-5;ft_k2_lr3e-5:--k 2 --lr-enc 3e-5"

Experimentos A y B (agentes nuevos sobre cada encoder congelado; k-NN con imagen completa):
  uv run modal run --detach modal_app.py::experiments
  uv run modal run --detach modal_app.py::experiments --encs reg_k2_lr1e-5,reg_k2_lr3e-5 --tag _reg
Línea base sin RL (regress.py): igual que el fine-tuning, con --module regress

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
    .add_local_python_source("glimpse", "env", "ppo", "finetune", "eval_repr", "regress")
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


# Tope de gasto: ~1,4 USD/h por corrida (GPU A10 + 2 CPU + 24 GB). El timeout acota el peor caso.
@app.function(image=image, gpu="A10", volumes={"/vol": vol}, timeout=4500, cpu=2, memory=24576)
def run(argv: list[str], module: str = "finetune"):
    import importlib
    import os

    os.chdir("/vol")
    importlib.import_module(module).main(argv)  # finetune (RL) o regress (línea base sin RL)
    vol.commit()


@app.local_entrypoint()
def main(runs: str, module: str = "finetune"):
    prepare.remote()
    for spec in runs.split(";"):
        name, _, extra = spec.partition(":")
        call = run.spawn(["--name", name, *extra.split()], module)
        print(f"lanzado {name}: {call.object_id}")


ENCODERS = ["base", "ft_k2_lr3e-6", "ft_k2_lr1e-5", "ft_k2_lr3e-5"]


# Experimento A: solo CPU (el encoder está congelado, se usan las tablas de embeddings). ~0,2 USD/h por corrida.
@app.function(image=image, volumes={"/vol": vol}, timeout=2 * 3600, cpu=4, memory=8192)
def exp_a(enc: str, seed: int, steps: int = 10_000_000):
    import os
    import shutil
    import sys

    work = f"/tmp/{enc}_s{seed}"  # carpeta propia: ppo.py escribe results/ppo.jsonl y las corridas van en paralelo
    os.makedirs(work)
    os.chdir(work)
    sys.argv = ["ppo.py", "--emb", f"/vol/out/{enc}_", "--steps", str(steps), "--evals", "20", "--seed", str(seed)]
    import ppo

    ppo.main()
    os.makedirs("/vol/results", exist_ok=True)
    shutil.copy("results/ppo.jsonl", f"/vol/results/expA_{enc}_s{seed}.jsonl")
    vol.commit()


@app.function(image=image, gpu="A10", volumes={"/vol": vol}, timeout=3600, cpu=8, memory=16384)
def exp_b(encs: list[str], tag: str = ""):
    import os
    import shutil
    import subprocess
    import sys

    os.makedirs("/tmp/b/data")
    os.chdir("/tmp/b")
    for ds in ("imagenette2", "imagewoof2"):
        subprocess.run(f"curl -sSfL --retry 10 {DATA_URL.replace('imagenette2', ds)} | tar -xz -C data",
                       shell=True, check=True)
    sys.argv = ["eval_repr.py", *[f"/vol/out/{e}_encoder.pt" for e in encs]]
    import eval_repr

    eval_repr.main()
    os.makedirs("/vol/results", exist_ok=True)
    shutil.copy("results/eval_repr.jsonl", f"/vol/results/eval_repr{tag}.jsonl")
    vol.commit()


@app.local_entrypoint()
def experiments(seeds: int = 3, steps: int = 10_000_000, b: bool = True, encs: str = "", tag: str = ""):
    """encs: encoders separados por coma (por defecto, el original y los tres ajustados con RL)."""
    encs = encs.split(",") if encs else ENCODERS
    if b:
        print("B:", exp_b.spawn([e for e in encs if e != "base"], tag).object_id)
    for enc in encs:
        for seed in range(seeds):
            print(f"A {enc} s{seed}:", exp_a.spawn(enc, seed, steps).object_id)
