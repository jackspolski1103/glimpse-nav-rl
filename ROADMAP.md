# Roadmap: fine-tuning an image encoder with glimpse-navigation RL

## Idea

An RL agent learns to move around an image seeing only crops (glimpses) until it reaches a target glimpse. The RL gradient fine-tunes the encoder. Questions:

1. Does the encoder fine-tuned with RL navigate better than the original? (Experiment A)
2. How much general information does the encoder lose from that fine-tuning? (Experiment B)

No human labels in training: glimpses, positions, target and reward all come from the image itself.

## Decisions made

| Piece | Decision |
|---|---|
| Encoder | DINOv2 ViT-S/14 (open). DINOv3 is left as a later improvement |
| Dataset | Imagenette, full-size version |
| Image | Resized to 448×448 |
| Glimpse | 112×112 px (8×8 internal ViT patches), encoded separately; embedding = CLS token |
| Stride | 56 px (50 % overlap) → 7×7 grid = 49 positions. Variant: 84 px → 5×5 |
| Actions | 4 neighbours (up, down, left, right) |
| Target | Glimpse from another cell of the same image, encoded with a frozen copy of the encoder |
| Agent state | Current embedding + target embedding + egocentric visited mask + bump signal (direction of the last move blocked by the border) |
| Algorithm | PPO, starting from the CleanRL implementation (adapted and cited) |
| Environment | Tensor-based core (episodes in parallel) + wrapper with the Gymnasium interface |
| PPO validation | Phase 1 also with Stable-Baselines3 on the same environment; they must agree |
| Policy | Small MLP (2 hidden layers), on purpose: so that the result depends on the embedding |
| Reward | +1 on arrival, −0.01 per step (initial values, to be tuned) |
| Fine-tuning signal | RL only. The world model is left as an optional auxiliary loss |

Resolved in phase 1: the visited mask is egocentric (it does not reveal the absolute position) and the agent perceives the border only when it bumps into it.

## Phases

### Phase 0 — Environment and precomputed embeddings
- Environment with `uv`, PyTorch (MPS), download of DINOv2 and Imagenette.
- Precompute the 49 embeddings per image with the frozen encoder and save them.
- Preliminary measurements (linear probes on frozen embeddings):
  - position of the cell from a single glimpse (expected near chance, 1/49);
  - relative offset between two glimpses of the same image.
- **Done when:** embeddings saved and both numbers measured.

### Phase 1 — PPO agent with frozen encoder
- Grid environment on the precomputed embeddings; random start and target in different cells.
- Baselines: random walk and shortest path (upper bound).
- PPO checks: (1) with its own position and the target's given directly it must reach 100 %; (2) Stable-Baselines3 on the Gymnasium wrapper must give an equivalent success rate.
- **Done when:** the success rate on validation images clearly beats the random walk.

### Phase 2 — Fine-tuning the encoder with RL
- Start from the phase 1 agent. Unfreeze the last blocks of the encoder with an LR 10–100 times lower than the head's.
- The encoder runs at every step (no more precomputation). The target is encoded with the frozen copy.
- **Done when:** training is stable and success is equal to or higher than in phase 1. Save the fine-tuned encoder.

### Phase 3 — Experiment A: does it navigate better?
- Two frozen encoders: original and fine-tuned. A new agent from scratch on each.
- Same architecture, same steps, same images, at least 3 seeds.
- Metrics per position: success rate and steps used / shortest path.
- Do not compare latent distances between encoders (a partial collapse distorts them).

### Phase 4 — Experiment B: what did it lose?
Same script on both encoders, with the full image and a frozen encoder:
- Classification k-NN on a dataset not used in the fine-tuning (Imagewoof; Imagenette is almost saturated for DINOv2).
- Segmentation with a linear layer (Pascal VOC).
- Direct drift: cosine similarity / CKA between embeddings of both encoders on the same images.

### Phase 5 — Extensions (only if phases 3 and 4 show something)
- **Baseline without RL:** fine-tune the encoder with direct regression of the offset to the target (see Related work).
- Degrees of fine-tuning (LoRA, more blocks, everything unfrozen) → "navigation gain vs. general loss" curve.
- World-model auxiliary loss (latent prediction of the next glimpse) and RL / WM / both comparison.
- Less overlap (84 px stride), grid of visited embeddings as memory, DINOv3.

## Compute

- **Phases 0, 1, 3 and 4:** on the Mac (MPS). The encoder is small and the embeddings are precomputed once.
- **Phase 2 (and 5):** the encoder runs with gradient at every step; this is where a rented GPU may be needed.
- **Chosen service:** Modal (launched from the terminal with `modal run`, no machines or SSH to manage, pay per use). Requires an account and a one-time login with `modal setup`.
- Write the training code device-independent (`cuda` / `mps` / `cpu`) from the start, so that the same script runs on the Mac and on Modal.

## Risks

- **Noisy RL signal:** it can damage the encoder. Mitigation: two phases, low LR, few unfrozen blocks.
- **Shortcuts:** with overlap, the encoder can specialize in matching edges. Experiment B detects this.
- **Task too easy or too hard:** adjust overlap and maximum episode length according to phase 1.

## Related work

Search done on 2026-11-25 (six web queries, not exhaustive). **No work with the full combination turned up**: navigation to a target glimpse + fine-tuning a pretrained encoder with RL only + measuring forgetting. The closest:

| Work | What it shares | How it differs |
|---|---|---|
| [seq-JEPA](https://arxiv.org/abs/2505.03176) (NeurIPS 2025) | Glimpses + actions (saccades), latent prediction | Heuristic policy, no target and no RL; trains from scratch |
| [CanViT](https://arxiv.org/abs/2603.22570) (NeurIPS 2026) — read | 128 px glimpses + grid memory ("canvas"); no labels in pretraining | Trains from scratch by distilling DINOv3 with random glimpses, without RL. It then trains policies with RL on the frozen model, with a segmentation reward (uses labels). No task with a target; according to the summary read, goal conditioning and joint policy+model training are listed as future work |
| [Visual Pre-training for Navigation: What Can We Learn from Noise?](https://arxiv.org/abs/2207.00052) (IROS 2023) — read | No labels: given the current crop and the target crop, predict where the target is | Direct one-step regression (ResNet18), with no RL and no moving agent; the downstream policy is trained supervised with actions. Does not analyse the representation |
| [AdaGlimpse](https://arxiv.org/abs/2404.03482) (ECCV 2024) — read | RL (SAC) chooses the position and scale of the next glimpse; ViT-B | Reward = drop in the reconstruction loss or in the distillation loss from a teacher. The encoder is trained with that loss, alternating epochs with the agent. No target to reach, no measurement of forgetting |
| [GRPO-RM](https://arxiv.org/abs/2511.15256) (2025) — read | Fine-tunes DINOv2 ViT-S/14 with an RL-style loss | The reward uses the correct class: in practice it is supervised. One decision per image, no agent. Does not measure forgetting |
| [EVA](https://arxiv.org/abs/2603.27340) (2026) — read | Glimpse policy with REINFORCE | Small CNN, reward = correct classification (labels). No target given to the agent |
| [Foveated ViT with a gaze policy](https://arxiv.org/abs/2604.20392) (2026) | Learned gaze policy on a ViT | Aimed at classification accuracy |
| [DINO-WM](https://arxiv.org/abs/2411.04983) (2024) | World model on DINOv2 to reach a target image | Frozen encoder, planning without RL, control environments |
| [Recurrent Models of Visual Attention](https://arxiv.org/abs/1406.6247) (2014) | RL that decides where to look | Supervised classification, small networks from scratch |
| [Pretrained Visual Representations in RL](https://arxiv.org/abs/2407.17238) (2024) | Pretrained encoders inside RL | Control environments, not navigation over images |

Other background reading: [I-JEPA](https://arxiv.org/abs/2301.08243), [IWM](https://arxiv.org/abs/2403.00504), [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588), [Context Prediction (Doersch 2015)](https://arxiv.org/abs/1505.05192).

After reading the five marked ones (2026-11-25), the combination still does not appear. CanViT is the closest in architecture and places our idea in what it declares as future work.

**Main objection to anticipate:** "What Can We Learn from Noise?" obtains the same signal (where the target is relative to me) with a direct regression, without RL. We have to show what RL adds over that → mandatory baseline in phase 5: fine-tune the encoder by directly predicting the offset to the target, and compare in experiments A and B.

**Borrowed idea:** that paper needs a curriculum (first targets that overlap with the current view, then distant ones) for training to converge. Apply it in phase 1 if the agent does not learn: start with nearby targets and increase the distance.

Before writing a paper, a more thorough search is advisable (Google Scholar, works citing seq-JEPA, AdaGlimpse and CanViT).
