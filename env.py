"""Glimpse-navigation environment.

GlimpseNav: tensor core, many episodes in parallel.
GlimpseNavGym: the same environment with the Gymnasium interface (one episode at a time).

Observation = [current embedding | goal embedding | egocentric visited mask | bump signal].
The bump signal (4 values) gives the direction of the last move if the border blocked it.
The mask is relative to the agent (13x13, agent in the centre) so as not to reveal its absolute position.
"""
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from glimpse import GRID

EGO = 2 * GRID - 1  # 13: covers any offset within the grid
MOVES = [[-1, 0], [1, 0], [0, -1], [0, 1]]  # up, down, left, right


class GlimpseNav:
    def __init__(self, embed, n_images, n_envs, dev="cpu", embed_goal=None,
                 max_steps=24, step_cost=0.01, auto_reset=True, walls="bump"):
        """embed(img_idx, cell_idx) -> (B, D). In phase 1 it is a table; in phase 2, the live encoder."""
        self.embed, self.embed_goal = embed, embed_goal or embed
        self.n_images, self.n, self.dev = n_images, n_envs, dev
        self.max_steps, self.step_cost, self.auto_reset, self.walls = max_steps, step_cost, auto_reset, walls
        self.moves = torch.tensor(MOVES, device=dev)
        self.ar = torch.arange(n_envs, device=dev)
        z = lambda *s: torch.zeros(n_envs, *s, dtype=torch.long, device=dev)
        self.img, self.pos, self.goal, self.t, self.dist0 = z(), z(2), z(2), z(), z()
        self.visited = torch.zeros(n_envs, GRID, GRID, dtype=torch.bool, device=dev)
        self.bump = torch.zeros(n_envs, 4, device=dev)
        self.reset_where(torch.ones(n_envs, dtype=torch.bool, device=dev))

    def reset_where(self, mask):
        idx = mask.nonzero().squeeze(1)
        n = len(idx)
        if not n:
            return
        pos = torch.randint(GRID, (n, 2), device=self.dev)
        goal = torch.randint(GRID, (n, 2), device=self.dev)
        same = (goal == pos).all(1)  # the goal never coincides with the start
        goal[same, 0] = (goal[same, 0] + torch.randint(1, GRID, (int(same.sum()),), device=self.dev)) % GRID
        self.img[idx] = torch.randint(self.n_images, (n,), device=self.dev)
        self.pos[idx], self.goal[idx], self.t[idx] = pos, goal, 0
        self.dist0[idx] = (pos - goal).abs().sum(1)
        self.visited[idx] = False
        self.visited[idx, pos[:, 0], pos[:, 1]] = True
        self.bump[idx] = 0

    @property
    def cell(self):
        return self.pos[:, 0] * GRID + self.pos[:, 1]

    def observe(self):
        z = self.embed(self.img, self.cell)
        g = self.embed_goal(self.img, self.goal[:, 0] * GRID + self.goal[:, 1])
        pad = F.pad(self.visited.float(), (GRID - 1,) * 4)
        win = torch.arange(EGO, device=self.dev)
        rows = (self.pos[:, 0, None] + win)[:, :, None]
        cols = (self.pos[:, 1, None] + win)[:, None, :]
        ego = pad[self.ar[:, None, None], rows, cols]  # centre (6,6) = current cell
        parts = [z, g, ego.flatten(1)]
        if self.walls == "sensor":  # is there an image border up, down, left, right?
            r, c = self.pos[:, 0], self.pos[:, 1]
            parts.append(torch.stack([r == 0, r == GRID - 1, c == 0, c == GRID - 1], 1).float())
        elif self.walls == "bump":  # only learns of the border after bumping: direction of the last blocked move
            parts.append(self.bump)
        return torch.cat(parts, 1)

    def step(self, action):
        old = self.pos
        self.pos = (self.pos + self.moves[action]).clamp(0, GRID - 1)  # against the border: does not move
        blocked = (self.pos == old).all(1)
        self.bump = F.one_hot(action, 4).float() * blocked[:, None]
        self.t += 1
        self.visited[self.ar, self.pos[:, 0], self.pos[:, 1]] = True
        reached = (self.pos == self.goal).all(1)
        done = reached | (self.t >= self.max_steps)
        reward = reached.float() - self.step_cost
        timeout = done & ~reached
        info = {"success": reached[done], "steps": self.t[done], "dist0": self.dist0[done], "timeout": timeout}
        if self.auto_reset:
            if timeout.any():  # terminal state, before resetting
                info.update(final_obs=self.observe(), final_img=self.img.clone(), final_cell=self.cell)
            self.reset_where(done)
        return self.observe(), reward, done, info


class GlimpseNavGym(gym.Env):
    def __init__(self, embed, n_images, obs_dim, **kw):
        self.core = GlimpseNav(embed, n_images, 1, auto_reset=False, **kw)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = gym.spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            torch.manual_seed(seed)
        self.core.reset_where(torch.ones(1, dtype=torch.bool))
        return self.core.observe()[0].numpy().astype(np.float32), {}

    def step(self, action):
        obs, r, done, info = self.core.step(torch.tensor([int(action)]))
        success = bool(info["success"][0]) if done[0] else False
        return (obs[0].numpy().astype(np.float32), float(r[0]), success, bool(done[0]) and not success,
                {"is_success": success, "dist0": int(self.core.dist0[0])})


def oracle_action(env):
    """Shortest path: rows first, then columns."""
    d = env.goal - env.pos
    a = torch.where(d[:, 0] < 0, 0, 1)
    return torch.where(d[:, 0] == 0, torch.where(d[:, 1] < 0, 2, 3), a)


if __name__ == "__main__":
    torch.manual_seed(0)
    table = torch.randn(5, GRID * GRID, 8)
    env = GlimpseNav(lambda i, c: table[i, c], 5, 512)
    obs = env.observe()
    assert obs.shape == (512, 2 * 8 + EGO * EGO + 4)
    assert (obs[:, 16:-4].reshape(-1, EGO, EGO)[:, GRID - 1, GRID - 1] == 1).all()  # the centre is always visited
    succ, steps, dist = [], [], []
    for _ in range(40):
        _, r, done, info = env.step(oracle_action(env))
        assert ((r > 0) == (done & (r > 0))).all()
        succ.append(info["success"]); steps.append(info["steps"]); dist.append(info["dist0"])
    succ, steps, dist = map(torch.cat, (succ, steps, dist))
    assert succ.all() and (steps == dist).all() and dist.min() >= 1  # the oracle arrives in exactly the distance

    env = GlimpseNav(lambda i, c: table[i, c], 5, 4)
    env.pos[:] = 0
    env.goal[:] = GRID - 1
    obs, *_ = env.step(torch.zeros(4, dtype=torch.long))  # up from row 0
    assert (env.pos == 0).all() and (obs[:, -4:] == torch.tensor([1.0, 0, 0, 0])).all()  # signals the bump
    obs, *_ = env.step(torch.ones(4, dtype=torch.long))  # down: it moves, the signal clears
    assert (obs[:, -4:] == 0).all()

    g = GlimpseNavGym(lambda i, c: table[i, c], 5, 2 * 8 + EGO * EGO + 4)
    o, _ = g.reset(seed=0)
    assert g.observation_space.contains(o)
    for _ in range(30):
        o, r, term, trunc, _ = g.step(int(oracle_action(g.core)[0]))
        if term or trunc:
            break
    assert term and not trunc
    print("ok")
