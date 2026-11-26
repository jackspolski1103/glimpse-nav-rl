"""Launches finetune.py on Modal GPUs (phase 2).

Usage (several runs in parallel, separated by ';', each one 'name:arguments'):
  uv run modal run --detach modal_app.py::main --runs "ft_k2_lr1e-5:--k 2 --lr-enc 1e-5;ft_k2_lr3e-5:--k 2 --lr-enc 3e-5"

Experiments A and B (new agents on each frozen encoder; k-NN with the full image):
  uv run modal run --detach modal_app.py::experiments
  uv run modal run --detach modal_app.py::experiments --encs reg_k2_lr1e-5,reg_k2_lr3e-5 --tag _reg
Spatial probes + transfer task without overlap (all orchestrated on Modal):
  uv run modal run --detach modal_app.py::run_transfer
No-RL baseline (regress.py): same as the fine-tuning, with --module regress

Results are stored in the 'glimpse-nav-rl' Volume, folder out/:
  uv run modal volume get glimpse-nav-rl out/<file> out/
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
    """One-off: downloads Imagenette and stores the resized pixels in the Volume."""
    import os
    import subprocess
    from pathlib import Path

    os.chdir("/vol")
    if Path("cache/pixels_train.npy").exists() and Path("cache/pixels_val.npy").exists():
        print("pixels already prepared")
        return
    subprocess.run(f"curl -sSfL --retry 10 {DATA_URL} | tar -xz -C /tmp", shell=True, check=True)
    from finetune import load_pixels

    for split in ("train", "val"):
        print(split, load_pixels("/tmp/imagenette2", split).shape, flush=True)
    vol.commit()


# Spending cap: ~1.4 USD/h per run (A10 GPU + 2 CPU + 24 GB). The timeout bounds the worst case.
@app.function(image=image, gpu="A10", volumes={"/vol": vol}, timeout=4500, cpu=2, memory=24576)
def run(argv: list[str], module: str = "finetune"):
    import importlib
    import os

    os.chdir("/vol")
    importlib.import_module(module).main(argv)  # finetune (RL) or regress (no-RL baseline)
    vol.commit()


@app.local_entrypoint()
def main(runs: str, module: str = "finetune"):
    prepare.remote()
    for spec in runs.split(";"):
        name, _, extra = spec.partition(":")
        call = run.spawn(["--name", name, *extra.split()], module)
        print(f"launched {name}: {call.object_id}")


ENCODERS = ["base", "ft_k2_lr3e-6", "ft_k2_lr1e-5", "ft_k2_lr3e-5"]


# Experiment A: CPU only (the encoder is frozen, the embedding tables are used). ~0.2 USD/h per run.
@app.function(image=image, volumes={"/vol": vol}, timeout=2 * 3600, cpu=4, memory=8192)
def exp_a(enc: str, seed: int, steps: int = 10_000_000):
    import os
    import shutil
    import sys

    work = f"/tmp/{enc}_s{seed}"  # own folder: ppo.py writes results/ppo.jsonl and the runs go in parallel
    os.makedirs(work)
    os.chdir(work)
    sys.argv = ["ppo.py", "--emb", f"/vol/out/{enc}_", "--steps", str(steps), "--evals", "20", "--seed", str(seed), "--name", "a"]
    import ppo

    ppo.main()
    os.makedirs("/vol/results", exist_ok=True)
    shutil.copy("results/ppo.jsonl", f"/vol/results/expA_{enc}_s{seed}.jsonl")
    last = max(os.listdir("checkpoints"), key=lambda f: int(f.rsplit("_", 1)[1][:-3]))  # final agent, for figures
    shutil.copy(f"checkpoints/{last}", f"/vol/out/expA_{enc}_s{seed}_agent.pt")
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
    """Transfer task: image at 784 px, 7x7 grid of 112 px glimpses with NO overlap (1.75x zoom).
    Saves out/z0_<enc>_{train,val}.npz. No encoder was fine-tuned with this layout."""
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
        print(split, "done", flush=True)
    vol.commit()


@app.function(image=image, volumes={"/vol": vol}, timeout=2 * 3600, cpu=4, memory=8192)
def probe(prefixes: list[str], tag: str):
    """Phase 0 probes (position, neighbour direction, offset) on embedding tables."""
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


# Orchestrator on Modal (does not depend on the Mac staying awake).
@app.function(image=image, volumes={"/vol": vol}, timeout=4 * 3600, cpu=0.25)
def transfer(nav: list[str], seeds: int = 2, steps: int = 5_000_000):
    z0_tables.remote(ALL)
    calls = [probe.spawn([f"/vol/out/{e}_" for e in ALL], ""), probe.spawn([f"/vol/out/z0_{e}_" for e in ALL], "_z0")]
    calls += [exp_a.spawn(f"z0_{e}", s, steps) for e in nav for s in range(seeds)]
    for c in calls:
        c.get()


@app.function(image=image, volumes={"/vol": vol}, timeout=4 * 3600, cpu=0.25)
def chain(name: str, argv: list[str], module: str = "finetune", seeds: int = 3):
    """One fine-tuning run and, when it finishes, its experiments A and B. Orchestrated on Modal."""
    run.remote(["--name", name, *argv], module)
    calls = [exp_b.spawn([name], f"_{name}")] + [exp_a.spawn(name, s) for s in range(seeds)]
    for c in calls:
        c.get()


@app.local_entrypoint()
def run_chain(name: str, args: str, module: str = "finetune"):
    print("chain:", chain.spawn(name, args.split(), module).object_id)


@app.local_entrypoint()
def run_transfer(nav: str = "base,ft_k2_lr1e-5,reg_k2_lr3e-6"):
    print("transfer:", transfer.spawn(nav.split(",")).object_id)


@app.local_entrypoint()
def experiments(seeds: int = 3, steps: int = 10_000_000, b: bool = True, encs: str = "", tag: str = ""):
    """encs: comma-separated encoders (by default, the original and the three RL fine-tuned ones)."""
    encs = encs.split(",") if encs else ENCODERS
    if b:
        print("B:", exp_b.spawn([e for e in encs if e != "base"], tag).object_id)
    for enc in encs:
        for seed in range(seeds):
            print(f"A {enc} s{seed}:", exp_a.spawn(enc, seed, steps).object_id)
