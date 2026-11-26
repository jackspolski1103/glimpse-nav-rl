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

**Más ejemplos** (agente final, 12 imágenes de validación por figura, con flechas de recorrido): llegó en 9 de 12 y en 5 de 12.

![Ejemplos 1](figures/frozen_s0_ejemplos_1.png)
![Ejemplos 2](figures/frozen_s0_ejemplos_2.png)

**Sin sobreajuste:** el éxito en imágenes de entrenamiento es prácticamente igual al de validación en todas las corridas. El techo no es memorización sino lo que el embedding congelado permite deducir.

## 2026-11-25 — Fase 1: ¿dónde falla el agente? (línea base `frozen_s0`)

`uv run python analysis.py --name frozen_s0` (50 000 episodios de validación, éxito global 60,7 %):

| Distancia inicial (casillas) | Éxito |
|---|---|
| 1–2 | 68,6 % |
| 3–4 | 64,1 % |
| 5–6 | 59,0 % |
| 7–8 | 52,8 % |
| 9–12 | 44,5 % |

- **Incluso a 1–2 casillas falla un 31 %**, pese a que el objetivo se solapa con lo que ve. Coincide con el probe de dirección del vecino (73 %): el embedding congelado no deja deducir bien la dirección.
- **La hipótesis "falla cuando el objetivo es ambiguo" no se confirma.** Midiendo la ambigüedad como la similitud máxima del objetivo con otras casillas de la imagen, el éxito por cuartiles es 55,9 / 63,8 / 65,2 / 57,8 %: no hay tendencia. (La medida es imperfecta: con solape, la casilla más parecida suele ser una vecina.)

**Hallazgo: el agente se queda atascado contra el borde.** Al chocar, la observación no cambia, así que repite la acción.

| | En episodios fallidos | En episodios con éxito |
|---|---|---|
| Pasos gastados chocando con el borde | 51,4 % | 7,5 % |

- El 72,1 % de los fallos incluye una racha de ≥5 choques seguidos; el 56,2 %, de ≥10.
- Es un defecto del entorno, no del embedding: el agente no tiene forma de percibir el borde.

**Prueba de dos remedios** (`--walls`, 2 M de pasos, 256 entornos, semilla 0):

| Observación | Éxito (val) | SPL |
|---|---|---|
| Sin información de borde (actual) | 54,2 % | 0,357 |
| `bump`: dirección del último movimiento bloqueado | 59,3 % | 0,391 |
| `sensor`: hay borde arriba/abajo/izquierda/derecha | 64,4 % | 0,431 |

- `sensor` rinde más, pero revela posición parcial (24 de las 49 casillas tocan un borde). `bump` solo informa después de chocar.
- Decisión pendiente del usuario. Una semilla por variante.

## 2026-11-25 — Fase 1: el techo viene del embedding, no del tamaño de la red

- **Política 4 veces más ancha** (`--hidden 1024`, 3 M de pasos, semilla 0): 58,4 % de éxito, SPL 0,368. Con 256 unidades, a los mismos pasos, la corrida larga daba ~56–57 %. Diferencia de 1–2 puntos: ampliar la política no rompe el techo.
- **Probes con casi 7 veces más entrenamiento** (`probes.py --steps 20000`): dirección del vecino con MLP 74,5 % (antes 72,9 %); lineal 47,1 %; posición absoluta 10,6 %. Prácticamente saturados.
- **Conclusión:** ni más capacidad ni más entrenamiento de la cabeza mejoran de forma apreciable. Lo que limita es la información de dirección disponible en el embedding congelado (más el atasco en el borde, ver entrada anterior). Se mantiene la política pequeña (256-256).

## 2026-11-25 — Decisión: aviso de choque, y nueva línea base (experimento A, encoder original)

- Decisión del usuario: el agente percibe el borde con un **aviso de choque** (4 valores: dirección del último movimiento si el borde lo bloqueó). No revela la posición sin actuar. Es el valor por defecto en `env.py`; `--walls none|sensor` quedan para ablación.
- Línea base definitiva del encoder original congelado (`ppo.py --task embeddings --steps 10000000 --evals 20 --seed S --name frozen_bump_sS`; 256 entornos):

  | Semilla | Éxito (val) | Éxito (train) | SPL (val) |
  |---|---|---|---|
  | 0 | 71,2 % | 72,9 % | 0,473 |
  | 1 | 72,3 % | 73,1 % | 0,473 |
  | 2 | 71,0 % | 71,2 % | 0,460 |
  | **Media** | **71,5 %** | 72,4 % | **0,469** |

  Paseo aleatorio: 23 %, SPL 0,10. Sin aviso de choque la misma configuración daba 60,3 % (una semilla).
- Las tres semillas quedan a ~1 punto entre sí: diferencias de más de 2–3 puntos entre encoders serán interpretables.
- `analysis.py --name frozen_bump_s0` (50 000 episodios, éxito 71,4 %): por distancia inicial 77,7 / 73,7 / 70,2 / 66,8 / 54,9 % (1–2, 3–4, 5–6, 7–8, 9–12 casillas). Sigue fallando un 22 % a 1–2 casillas: ese es el margen atribuible al embedding.
- Figuras: `figures/frozen_bump_s0_curva.png`, `_trayectorias.png`, `_ejemplos_1.png` (llegó en 10 de 12), `_ejemplos_2.png` (8 de 12).

![Curva](figures/frozen_bump_s0_curva.png)
![Ejemplos](figures/frozen_bump_s0_ejemplos_1.png)

## 2026-11-25 — Fase 2: implementación del fine-tuning con RL

- `ppo.py`: el entrenador admite un encoder en vivo (`live`). En el rollout se guardan los tokens a la salida del tronco congelado; en la actualización se recalcula el embedding del glimpse actual con gradiente, así la pérdida de PPO llega a los bloques liberados. El camino con encoder congelado no cambia (la tarea de control sigue llegando al 100 %).
- `finetune.py`: etapa 1 (cabeza del agente, encoder congelado, 2 M de pasos) → etapa 2 (últimos `k` bloques + norma final liberados, LR propio). El glimpse objetivo siempre sale del encoder original congelado. Registra `x_cos`: similitud coseno entre el embedding ajustado y el original en glimpses de validación (deriva).
- `uv run python test_finetune.py`: **pasa**. (1) tronco + cola reproduce el encoder completo; (2) el recorte en el dispositivo da los mismos embeddings que la fase 0 (coseno mínimo 1,00000); (3) el gradiente llega solo a los bloques liberados (3,55 M de 22,06 M parámetros con k=2).
- Prueba de punta a punta en el Mac (32 imágenes, pocos pasos): corre completa. Velocidad de la etapa 2 en MPS: **~87 pasos/s** → 10 M de pasos serían ~32 h. Por eso la fase 2 corre en Modal.
- `modal_app.py`: prepara los datos en un Volume de Modal y lanza corridas en GPU A10.
