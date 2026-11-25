"""¿Dónde falla el agente? Éxito según distancia inicial y según ambigüedad del glimpse objetivo.

Ambigüedad = similitud coseno máxima entre el embedding objetivo y los de las otras 48 casillas
de la misma imagen (alta = hay otro recorte casi igual en la imagen).
Uso: uv run python analysis.py --name frozen_s0
"""
import argparse
import glob

import torch
import torch.nn.functional as F

from env import GlimpseNav
from glimpse import GRID
from ppo import Agent, make_task


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--episodes", type=int, default=50_000)
    args = ap.parse_args()
    torch.manual_seed(0)

    _, (emb, n_img), obs_dim = make_task("embeddings", "", "cpu")
    ckpt = max(glob.glob(f"checkpoints/{args.name}_*.pt"), key=lambda p: int(p.rsplit("_", 1)[1][:-3]))
    agent = Agent(obs_dim)
    agent.load_state_dict(torch.load(ckpt))

    env = GlimpseNav(emb, n_img, args.episodes)
    img, goal, dist = env.img.clone(), env.goal[:, 0] * GRID + env.goal[:, 1], env.dist0.clone()
    ok = torch.zeros(env.n, dtype=torch.bool)
    seen = torch.zeros(env.n, dtype=torch.bool)
    obs = env.observe()
    for _ in range(env.max_steps):
        obs, r, done, _ = env.step(agent.act(obs)[0])
        ok |= done & ~seen & (r > 0)
        seen |= done

    cells = torch.arange(GRID * GRID)
    allz = F.normalize(emb(img[:, None].expand(-1, GRID * GRID), cells[None].expand(env.n, -1)), dim=-1)
    sim = (allz * allz[env.ar, goal][:, None]).sum(-1)
    sim[env.ar, goal] = -1
    amb = sim.max(1).values

    print(f"checkpoint: {ckpt}   episodios: {env.n}   éxito global: {ok.float().mean() * 100:.1f} %\n")
    print("distancia inicial   episodios   éxito")
    for lo, hi in ((1, 2), (3, 4), (5, 6), (7, 8), (9, 12)):
        m = (dist >= lo) & (dist <= hi)
        print(f"  {lo:>2}-{hi:<2}             {int(m.sum()):>7}   {ok[m].float().mean() * 100:5.1f} %")
    print("\nambigüedad del objetivo (cuartiles)   similitud máx.   éxito")
    q = torch.quantile(amb, torch.tensor([0.25, 0.5, 0.75]))
    edges = [-1.0, *q.tolist(), 1.01]
    for k, label in enumerate(("más distintivo", "", "", "más ambiguo")):
        m = (amb >= edges[k]) & (amb < edges[k + 1])
        print(f"  Q{k + 1} {label:<15}                 {amb[m].mean():.2f}          {ok[m].float().mean() * 100:5.1f} %")
    near, clear = dist <= 4, amb < q[1]
    print(f"\ncerca (<=4) y distintivo: {ok[near & clear].float().mean() * 100:.1f} %   "
          f"lejos (>=7) y ambiguo: {ok[(dist >= 7) & ~clear].float().mean() * 100:.1f} %")


if __name__ == "__main__":
    main()
