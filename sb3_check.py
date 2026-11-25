"""Comprobación cruzada: PPO de Stable-Baselines3 sobre el mismo entorno (vía Gymnasium).

Mismos hiperparámetros que ppo.py y misma función de evaluación, para comparar directamente.
Uso: uv run python sb3_check.py --task embeddings
"""
import argparse
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env import GlimpseNav, GlimpseNavGym
from ppo import evaluate, make_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["positions", "embeddings"], default="embeddings")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    (emb_tr, n_tr), (emb_va, n_va), obs_dim = make_task(args.task, args.suffix, "cpu")
    venv = DummyVecEnv([lambda: GlimpseNavGym(emb_tr, n_tr, obs_dim)] * args.envs)
    model = PPO(
        "MlpPolicy", venv, seed=args.seed, device="cpu", verbose=0,
        n_steps=32, batch_size=args.envs * 32 // 4, n_epochs=4,
        learning_rate=lambda left: 2.5e-4 * left, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs={"net_arch": [256, 256]},
    )
    model.learn(args.steps)

    eval_env = GlimpseNav(emb_va, n_va, 4096)
    final = evaluate(eval_env, lambda o: model.policy(o)[0])
    print(f"SB3 PPO, {args.steps:,} pasos: éxito {final['success'] * 100:.1f} %  SPL {final['spl']:.3f}")
    Path("results").mkdir(exist_ok=True)
    with open("results/sb3.jsonl", "a") as fh:
        fh.write(json.dumps({**vars(args), "final": final}) + "\n")


if __name__ == "__main__":
    main()
