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

## Fase 0: embeddings y probes

_Pendiente: descarga de Imagenette en curso._
