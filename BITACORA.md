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

## 2026-11-25 — Requisito: los embeddings no ven la imagen completa

- Requisito del proyecto: el embedding de un glimpse solo puede depender de los píxeles de ese glimpse. Vale para el glimpse actual y para el objetivo, en todas las fases de entrenamiento.
- Cómo se cumple: `precompute.py` recorta primero y codifica cada recorte como una imagen independiente de 112 px; la atención del ViT solo ocurre entre los 64 parches internos del recorte.
- `uv run python test_no_leak.py`: **pasa**. Cambia todos los píxeles de la imagen salvo los de un glimpse y comprueba que el embedding de ese glimpse no varía (y que los de otros glimpses sí).
- Única excepción, a propósito: el experimento B evalúa el encoder con imagen completa, porque así se evaluó el modelo original. No interviene en ningún entrenamiento.

## 2026-11-25 — Fase 0: embeddings y probes (completada)

- Imagenette descargado: 9 469 imágenes de entrenamiento, 3 925 de validación. (La descarga se cortó una vez; se terminó en 8 tramos paralelos y la descompresión verificó la integridad.)
- `uv run python precompute.py` → `cache/train.npz` (9469×49×384) y `cache/val.npz` (3925×49×384), unos 15 min en el Mac.
- `uv run python probes.py` (entrena en train, evalúa en val; 3000 pasos, semilla 0):

  | Probe | Resultado | Referencia |
  |---|---|---|
  | Posición absoluta desde 1 glimpse (lineal, 49 clases) | 10,9 % | azar 2,0 % |
  | Dirección del vecino desde 2 glimpses (lineal) | 47,2 % | azar 25 % |
  | Dirección del vecino desde 2 glimpses (MLP 512) | 72,9 % | azar 25 % |
  | Desplazamiento entre 2 glimpses cualesquiera (MLP 512) | error 2,92 casillas; exacto 7,4 % | sin información: 4,58 casillas |

- **Lectura:** un glimpse aislado casi no sabe dónde está (lo que se buscaba). Dos glimpses vecinos permiten deducir la dirección mejor que el azar pero lejos del 100 %, y hace falta no linealidad (47 % → 73 %). Para pares lejanos la información es escasa. Ese es el margen que el fine-tuning puede mejorar.
- Limitación: una semilla y probes cortos; los números del MLP pueden subir con más entrenamiento.

## 2026-11-25 — Fase 1: PPO sobre embeddings congelados y comprobación cruzada

Todo en validación, 4096 episodios por evaluación. Paseo aleatorio: 23,0 % de éxito, SPL 0,096.

**Comprobación cruzada con Stable-Baselines3** (`sb3_check.py`; 64 entornos, 2 M de pasos, mismos hiperparámetros, misma función de evaluación):

| Implementación | Semillas 0 / 1 / 2 (éxito %) | Media |
|---|---|---|
| Stable-Baselines3 PPO | 51,6 / 50,0 / 52,1 | 51,2 |
| PPO propio, CleanRL original | 42,9 (solo semilla 0) | — |
| + continuar el valor al agotarse el tiempo | 47,2 / 44,7 / 47,0 | 46,3 |
| + sin recorte de la pérdida de valor (**versión final**) | 49,5 / 50,1 / 47,3 | 49,0 |

- Dos diferencias de implementación explicaban la brecha inicial de ~9 puntos:
  1. **Tiempo agotado:** CleanRL lo trata como estado terminal; lo correcto (Pardo et al., 2018; lo que hace SB3) es continuar con el valor del estado final. `env.py` devuelve ahora la observación terminal y `ppo.py` la usa.
  2. **Recorte de la pérdida de valor:** CleanRL lo aplica por defecto, SB3 no. Ahora está desactivado por defecto (`--clip-vloss` lo reactiva).
- Quedan ~2 puntos de diferencia en la media, con rangos que casi se tocan (47,3–50,1 frente a 50,0–52,1). Con 3 semillas no se puede afirmar que sea cero; se da la implementación por validada para la fase 1 y se anota como diferencia residual.
- En la tarea de control (posiciones dadas) ambas implementaciones llegan al 100 %.

**Entrenamiento largo, versión final** (`uv run python ppo.py --task embeddings --steps 10000000 --evals 20 --name frozen_s0`; 256 entornos, semilla 0, ~7 min en CPU):

| | Éxito (val) | Éxito (train) | SPL (val) |
|---|---|---|---|
| Sin entrenar | 22,7 % | 23,4 % | 0,098 |
| 2 M de pasos | ~54 % | ~55 % | ~0,35 |
| 10 M de pasos | **60,3 %** | 61,4 % | **0,404** |

![Curva de aprendizaje](figures/frozen_s0_curva.png)

La curva sube rápido hasta ~55 % en 2 M de pasos y después muy despacio: se acerca a un techo. **Esta es la línea base del experimento A** (encoder original congelado). Una sola semilla.

**Ejemplos sobre validación** (`uv run python viz.py --name frozen_s0`): seis imágenes de validación, mismo inicio y objetivo, con el agente sin entrenar y tras 1, 5 y 10 M de pasos. La política es estocástica, así que cada recorrido es una muestra.

![Trayectorias](figures/frozen_s0_trayectorias.png)

Observaciones sobre estos seis casos (anecdóticas, no medidas):
- Sin entrenar, el agente deambula cerca del inicio; entrenado, va hacia el objetivo cuando este tiene contenido distintivo (panel del equipo, letras de la carátula, pared gris junto al borde).
- Los fallos se concentran en objetivos sin contenido propio (cielo blanco, fondo blanco): desde el embedding no hay forma de saber dónde está ese recorte. En esos casos el agente termina pegado a un borde.
- Pendiente de medir: éxito según la distancia inicial al objetivo y según lo "informativo" que sea el glimpse objetivo.

**Sin sobreajuste:** el éxito en imágenes de entrenamiento es prácticamente igual al de validación en todas las corridas. El techo no es memorización sino lo que el embedding congelado permite deducir.
