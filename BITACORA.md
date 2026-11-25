# Bitácora

Registro cronológico de lo que se hace y de los resultados. El plan está en [ROADMAP.md](ROADMAP.md). Cada entrada: qué se hizo, cómo reproducirlo, resultado y qué se decide a partir de él.

## 2026-11-25 — Diseño y revisión de trabajo relacionado

- Definido el diseño (ver "Decisiones tomadas" en el roadmap): glimpses de 112 px con paso de 56 px sobre imágenes de 448 px (rejilla 7×7), DINOv2 ViT-S/14, Imagenette, PPO, fine-tuning solo con RL.
- Revisión de trabajo relacionado: seis búsquedas web y lectura de cinco papers (CanViT, "What Can We Learn from Noise?", AdaGlimpse, GRPO-RM, EVA). Ninguno combina navegación a un glimpse objetivo + fine-tuning de un encoder preentrenado solo con RL + medición del olvido. Detalle en el roadmap.
- Consecuencias para el plan: línea base sin RL (regresión directa del desplazamiento al objetivo) y currículo de distancia al objetivo si el agente no aprende.

## 2026-11-25 — Fase 0: entorno

- Repo privado: https://github.com/jackspolski1103/glimpse-nav-rl
- Entorno con `uv`, Python 3.12, PyTorch 2.15.0 (MPS disponible), timm 1.0.31.
- Modal instalado y autenticado (workspace `jackspolski2000`) para cuando haga falta GPU en la fase 2.
- Código:
  - `glimpse.py`: constantes de la rejilla, extracción de glimpses, carga del encoder. `uv run python glimpse.py` ejecuta su comprobación (49 glimpses por imagen y cada uno coincide con el recorte esperado): **pasa**.
  - `precompute.py`: embeddings de los 49 glimpses por imagen con el encoder congelado → `cache/{train,val}.npz`.
  - `probes.py`: probes sobre los embeddings congelados (posición absoluta, dirección del vecino, desplazamiento entre dos glimpses).
- Velocidad medida en el Mac (MPS): **565 glimpses/s** con DINOv2 ViT-S → unos 20 min para Imagenette completo (13 394 imágenes × 49).

## 2026-11-25 — Decisión: estándares de RL

- Entorno con interfaz de Gymnasium; PPO adaptado de CleanRL (no escrito de cero); Stable-Baselines3 como comprobación cruzada en la fase 1. Motivo: para investigación, la corrección del PPO debe poder defenderse, y la fase 2 necesita control directo del gradiente hacia el encoder.
- Métrica de navegación: tasa de éxito y SPL (Anderson et al., 2018), el estándar en navegación.
- Máscara de visitados **egocéntrica** (13×13 centrada en el agente): así el agente no recibe su posición absoluta.

## 2026-11-25 — Fase 1 (adelantada): entorno y PPO, validados sin datos reales

Hecho mientras se descargaba Imagenette; no depende de los embeddings.

- `env.py`: entorno en tensores (`GlimpseNav`) + envoltura Gymnasium (`GlimpseNavGym`). Episodio de 24 pasos como máximo, +1 al llegar, −0,01 por paso, chocar con el borde = no moverse.
  - `uv run python env.py`: **pasa**. Comprueba que el camino más corto llega siempre en exactamente la distancia Manhattan, que el borde bloquea y que la envoltura Gymnasium termina con éxito.
- `ppo.py`: PPO de CleanRL adaptado al entorno en tensores. 256 entornos × 32 pasos por actualización, MLP 256-256 tanh.
- **Comprobación del PPO** (`uv run python ppo.py --task positions --steps 1000000`, semilla 0): el "embedding" es la posición en one-hot, así que la tarea es resoluble con certeza.

  | | Éxito | SPL |
  |---|---|---|
  | Paseo aleatorio | 23,0 % | 0,096 |
  | PPO, 300 mil pasos | 100,0 % | 0,802 |
  | PPO, 1 millón de pasos | 100,0 % | 0,991 |

  Velocidad: ~36 000 pasos/s en CPU. **Conclusión:** el PPO y el entorno funcionan; el paseo aleatorio (23 %) es el suelo contra el que comparar. Una sola semilla.
- Pendiente de esta fase: correrlo sobre embeddings reales y la comprobación cruzada con Stable-Baselines3.

## Fase 0: embeddings y probes

_Pendiente: descarga de Imagenette en curso (se cortó una vez a los 815 MB; relanzada con reintentos)._
