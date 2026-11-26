"""Lanza finetune.py en GPUs de Modal (fase 2).

Uso (varias corridas en paralelo, separadas por ';', cada una 'nombre:argumentos'):
  uv run modal run --detach modal_app.py::main --runs "ft_k2_lr1e-5:--k 2 --lr-enc 1e-5;ft_k2_lr3e-5:--k 2 --lr-enc 3e-5"

Experimentos A y B (agentes nuevos sobre cada encoder congelado; k-NN con imagen completa):
  uv run modal run --detach modal_app.py::experiments
  uv run modal run --detach modal_app.py::experiments --encs reg_k2_lr1e-5,reg_k2_lr3e-5 --tag _reg
Sondas espaciales + tarea de transferencia sin solape (todo orquestado en Modal):
  uv run modal run --detach modal_app.py::run_transfer
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
    .add_local_python_source("glimpse", "env", "ppo", "finetune", "eval_repr", "regress", "probes")
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


ALL = ENCODERS + [f"reg_k2_lr{lr}" for lr in ("3e-7", "1e-6", "3e-6", "1e-5", "3e-5")]


@app.function(image=image, gpu="A10", volumes={"/vol": vol}, timeout=3600, cpu=8, memory=16384)
def z0_tables(encs: list[str]):
    """Tarea de transferencia: imagen a 784 px, rejilla 7x7 de glimpses de 112 px SIN solape (zoom 1,75x).
    Guarda out/z0_<enc>_{train,val}.npz. Ningún encoder se ajustó con esta disposición."""
    import os
    import subprocess

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    from finetune import amp
    from glimpse import GLIMPSE, GRID, MEAN, STD, load_encoder

    subprocess.run(f"curl -sSfL --retry 10 {DATA_URL} | tar -xz -C /tmp", shell=True, check=True)
    side = GRID * GLIMPSE
    tf = transforms.Compose([transforms.Resize(side), transforms.CenterCrop(side), transforms.ToTensor(),
                             transforms.Normalize(MEAN, STD)])
    models = {}
    for e in encs:
        m = load_encoder()
        if e != "base":
            m.load_state_dict(torch.load(f"/vol/out/{e}_encoder.pt", map_location="cpu"))
        models[e] = m.eval().cuda()
    for split in ("train", "val"):
        feats = {e: [] for e in encs}
        for x, _ in DataLoader(datasets.ImageFolder(f"/tmp/imagenette2/{split}", tf), batch_size=16, num_workers=8):
            b = len(x)
            x = x.cuda().view(b, 3, GRID, GLIMPSE, GRID, GLIMPSE).permute(0, 2, 4, 1, 3, 5).reshape(-1, 3, GLIMPSE, GLIMPSE)
            with torch.no_grad(), amp(x):
                for e, m in models.items():
                    feats[e].append(m(x).float().view(b, GRID * GRID, -1).half().cpu())
        for e in encs:
            np.savez(f"/vol/out/z0_{e}_{split}.npz", emb=torch.cat(feats[e]).numpy())
        print(split, "listo", flush=True)
    vol.commit()


@app.function(image=image, volumes={"/vol": vol}, timeout=2 * 3600, cpu=4, memory=8192)
def probe(prefixes: list[str], tag: str):
    """Sondas de la fase 0 (posición, dirección del vecino, desplazamiento) sobre tablas de embeddings."""
    import json
    import os

    import probes

    res = {}
    for p in prefixes:
        print("==", p, flush=True)
        res[p] = probes.main(["--emb", p])
    os.makedirs("/vol/results", exist_ok=True)
    with open(f"/vol/results/probes{tag}.json", "w") as fh:
        json.dump(res, fh)
    vol.commit()


# Orquestador en Modal (no depende de que el Mac siga despierto).
@app.function(image=image, volumes={"/vol": vol}, timeout=4 * 3600, cpu=0.25)
def transfer(nav: list[str], seeds: int = 2, steps: int = 5_000_000):
    z0_tables.remote(ALL)
    calls = [probe.spawn([f"/vol/out/{e}_" for e in ALL], ""), probe.spawn([f"/vol/out/z0_{e}_" for e in ALL], "_z0")]
    calls += [exp_a.spawn(f"z0_{e}", s, steps) for e in nav for s in range(seeds)]
    for c in calls:
        c.get()


@app.function(image=image, volumes={"/vol": vol}, timeout=4 * 3600, cpu=0.25)
def chain(name: str, argv: list[str], module: str = "finetune", seeds: int = 3):
    """Un ajuste y, al terminar, sus experimentos A y B. Orquestado en Modal."""
    run.remote(["--name", name, *argv], module)
    calls = [exp_b.spawn([name], f"_{name}")] + [exp_a.spawn(name, s) for s in range(seeds)]
    for c in calls:
        c.get()


@app.local_entrypoint()
def run_chain(name: str, args: str, module: str = "finetune"):
    print("cadena:", chain.spawn(name, args.split(), module).object_id)


@app.local_entrypoint()
def run_transfer(nav: str = "base,ft_k2_lr1e-5,reg_k2_lr3e-6"):
    print("transferencia:", transfer.spawn(nav.split(",")).object_id)


@app.local_entrypoint()
def experiments(seeds: int = 3, steps: int = 10_000_000, b: bool = True, encs: str = "", tag: str = ""):
    """encs: encoders separados por coma (por defecto, el original y los tres ajustados con RL)."""
    encs = encs.split(",") if encs else ENCODERS
    if b:
        print("B:", exp_b.spawn([e for e in encs if e != "base"], tag).object_id)
    for enc in encs:
        for seed in range(seeds):
            print(f"A {enc} s{seed}:", exp_a.spawn(enc, seed, steps).object_id)
