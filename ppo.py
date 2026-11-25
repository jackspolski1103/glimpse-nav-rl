"""PPO para la navegación por glimpses (fase 1: encoder congelado).

Adaptado de CleanRL, ppo.py (https://github.com/vwxyzjn/cleanrl, Huang et al., JMLR 2022):
mismos detalles (init ortogonal, Adam eps=1e-5, LR decreciente, GAE, ventaja normalizada,
pérdida de valor recortada, recorte de gradiente). Diferencia con CleanRL: al agotarse el tiempo
se continúa con el valor del estado final (Pardo et al., 2018), como hace Stable-Baselines3.

Uso:
  uv run python ppo.py --task positions      # comprobación: posiciones dadas, debe llegar a ~100 %
  uv run python ppo.py --task embeddings     # fase 1 real, sobre cache/{train,val}.npz
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from env import EGO, GlimpseNav
from glimpse import GRID


def layer(i, o, std=2 ** 0.5):
    lin = nn.Linear(i, o)
    nn.init.orthogonal_(lin.weight, std)
    nn.init.zeros_(lin.bias)
    return lin


class Agent(nn.Module):
    def __init__(self, obs_dim, hidden=256):
        super().__init__()
        mlp = lambda out, std: nn.Sequential(
            layer(obs_dim, hidden), nn.Tanh(), layer(hidden, hidden), nn.Tanh(), layer(hidden, out, std))
        self.actor, self.critic = mlp(4, 0.01), mlp(1, 1.0)

    def act(self, obs, action=None):
        dist = Categorical(logits=self.actor(obs))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(obs).squeeze(-1)


@torch.no_grad()
def evaluate(env, policy):
    """Un episodio por entorno (el primero), para no sesgar hacia episodios cortos.

    SPL (Anderson et al., 2018) = media de éxito * camino_más_corto / max(pasos, camino_más_corto).
    """
    env.reset_where(torch.ones(env.n, dtype=torch.bool, device=env.dev))
    obs = env.observe()
    seen = torch.zeros(env.n, dtype=torch.bool, device=env.dev)
    succ, steps, dist = [], [], []
    for _ in range(env.max_steps):
        obs, _, done, info = env.step(policy(obs))
        first = ~seen[done]
        succ.append(info["success"][first]); steps.append(info["steps"][first]); dist.append(info["dist0"][first])
        seen |= done
    s, l, d = torch.cat(succ).float(), torch.cat(steps).float(), torch.cat(dist).float()
    assert len(s) == env.n
    return {"success": s.mean().item(), "spl": (s * d / torch.maximum(l, d)).mean().item()}


def train(env, eval_env, obs_dim, total_steps, num_steps=32, lr=2.5e-4, gamma=0.99, lam=0.95,
          epochs=4, minibatches=4, clip=0.2, ent_coef=0.01, vf_coef=0.5, max_grad=0.5, evals=10,
          train_eval_env=None, ckpt=None, clip_vloss=False):
    """train_eval_env: mide también en imágenes de entrenamiento. ckpt: prefijo para guardar el agente en cada evaluación."""
    dev, n = env.dev, env.n
    agent = Agent(obs_dim).to(dev)
    opt = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    batch = n * num_steps
    updates = total_steps // batch
    buf = lambda *s: torch.zeros(num_steps, n, *s, device=dev)
    obs_b, act_b, logp_b, rew_b, done_b, val_b = buf(obs_dim), buf(), buf(), buf(), buf(), buf()
    next_obs, next_done = env.observe(), torch.zeros(n, device=dev)
    sample = lambda o: agent.act(o)[0]
    history, t0 = [], time.time()

    def log(u):
        m = {"steps": u * batch, **evaluate(eval_env, sample)}
        if train_eval_env:
            m["train_success"] = evaluate(train_eval_env, sample)["success"]
        if ckpt:
            torch.save(agent.state_dict(), f"{ckpt}_{u * batch}.pt")
        history.append(m)
        tr = f"  (train {m['train_success'] * 100:5.1f} %)" if train_eval_env else ""
        print(f"pasos {m['steps']:>10,}  éxito {m['success'] * 100:5.1f} %{tr}  SPL {m['spl']:.3f}  "
              f"({u * batch / max(time.time() - t0, 1e-9):,.0f} pasos/s)", flush=True)

    log(0)
    for u in range(1, updates + 1):
        opt.param_groups[0]["lr"] = lr * (1 - (u - 1) / updates)
        for t in range(num_steps):
            obs_b[t], done_b[t] = next_obs, next_done
            with torch.no_grad():
                a, lp, _, v = agent.act(next_obs)
            next_obs, r, d, info = env.step(a)
            if info["timeout"].any():  # tiempo agotado no es un estado terminal: se continúa con V(s)
                with torch.no_grad():
                    r = r + gamma * agent.critic(info["final_obs"]).squeeze(-1) * info["timeout"]
            act_b[t], logp_b[t], val_b[t], rew_b[t], next_done = a, lp, v, r, d.float()

        with torch.no_grad():
            next_val = agent.critic(next_obs).squeeze(-1)
            adv, last = torch.zeros_like(rew_b), 0
            for t in reversed(range(num_steps)):
                nonterm = 1 - (next_done if t == num_steps - 1 else done_b[t + 1])
                nv = next_val if t == num_steps - 1 else val_b[t + 1]
                delta = rew_b[t] + gamma * nv * nonterm - val_b[t]
                adv[t] = last = delta + gamma * lam * nonterm * last
            ret = adv + val_b

        f = lambda x: x.reshape(batch, *x.shape[2:])
        o, a, lp, ad, rt, vl = f(obs_b), f(act_b).long(), f(logp_b), f(adv), f(ret), f(val_b)
        for _ in range(epochs):
            for mb in torch.randperm(batch, device=dev).split(batch // minibatches):
                _, newlp, ent, newv = agent.act(o[mb], a[mb])
                ratio = (newlp - lp[mb]).exp()
                madv = (ad[mb] - ad[mb].mean()) / (ad[mb].std() + 1e-8)
                pg = torch.max(-madv * ratio, -madv * ratio.clamp(1 - clip, 1 + clip)).mean()
                v_loss = (newv - rt[mb]) ** 2
                if clip_vloss:
                    v_clip = vl[mb] + (newv - vl[mb]).clamp(-clip, clip)
                    v_loss = torch.max(v_loss, (v_clip - rt[mb]) ** 2)
                v_loss = 0.5 * v_loss.mean()
                loss = pg - ent_coef * ent.mean() + vf_coef * v_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad)
                opt.step()

        if u % max(1, updates // evals) == 0 or u == updates:
            log(u)
    return agent, history


def table_embed(split, suffix, dev, stats=None):
    e = torch.from_numpy(np.load(f"cache/{split}{suffix}.npz")["emb"].astype(np.float32)).to(dev)
    stats = stats or (e.mean((0, 1)), e.std((0, 1)) + 1e-6)
    e = (e - stats[0]) / stats[1]
    return (lambda i, c: e[i, c]), len(e), e.shape[-1], stats


def make_task(task, suffix, dev):
    """-> (embed_train, n_train), (embed_val, n_val), obs_dim"""
    if task == "positions":  # el "embedding" es la posición misma: la tarea se vuelve trivial
        eye = torch.eye(GRID * GRID, device=dev)
        emb = lambda i, c: eye[c]
        return (emb, 1), (emb, 1), 2 * GRID * GRID + EGO * EGO
    emb_tr, n_tr, d, stats = table_embed("train", suffix, dev)
    emb_va, n_va, _, _ = table_embed("val", suffix, dev, stats)
    return (emb_tr, n_tr), (emb_va, n_va), 2 * d + EGO * EGO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["positions", "embeddings"], default="embeddings")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--evals", type=int, default=10)
    ap.add_argument("--clip-vloss", action="store_true", help="recorte de la pérdida de valor (CleanRL)")
    ap.add_argument("--name", help="guarda checkpoints en checkpoints/<name>_<pasos>.pt")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = args.device

    (emb_tr, n_tr), (emb_va, n_va), obs_dim = make_task(args.task, args.suffix, dev)
    env = GlimpseNav(emb_tr, n_tr, args.envs, dev)
    eval_env = GlimpseNav(emb_va, n_va, 4096, dev)
    rand = evaluate(eval_env, lambda o: torch.randint(4, (len(o),), device=dev))
    print(f"paseo aleatorio: éxito {rand['success'] * 100:.1f} %  SPL {rand['spl']:.3f}")
    ckpt = None
    if args.name:
        Path("checkpoints").mkdir(exist_ok=True)
        ckpt = f"checkpoints/{args.name}"
    _, history = train(env, eval_env, obs_dim, args.steps, evals=args.evals,
                       train_eval_env=GlimpseNav(emb_tr, n_tr, 4096, dev), ckpt=ckpt,
                       clip_vloss=args.clip_vloss)

    Path("results").mkdir(exist_ok=True)
    with open("results/ppo.jsonl", "a") as fh:
        fh.write(json.dumps({**vars(args), "random": rand, "final": history[-1], "history": history}) + "\n")


if __name__ == "__main__":
    main()
