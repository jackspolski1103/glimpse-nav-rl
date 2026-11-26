# Roadmap: fine-tuning de un encoder de imágenes con RL de navegación por glimpses

## Idea

Un agente de RL aprende a moverse por una imagen viendo solo recortes (glimpses) hasta llegar a un glimpse objetivo. El gradiente de RL ajusta el encoder. Preguntas:

1. ¿El encoder ajustado con RL navega mejor que el original? (Experimento A)
2. ¿Cuánta información general pierde el encoder por ese ajuste? (Experimento B)

Sin etiquetas humanas en el entrenamiento: glimpses, posiciones, objetivo y recompensa salen de la propia imagen.

## Decisiones tomadas

| Pieza | Decisión |
|---|---|
| Encoder | DINOv2 ViT-S/14 (abierto). DINOv3 queda como mejora posterior |
| Dataset | Imagenette, versión de tamaño completo |
| Imagen | Reescalada a 448×448 |
| Glimpse | 112×112 px (8×8 parches internos del ViT), codificado por separado; embedding = token CLS |
| Paso | 56 px (solape 50 %) → rejilla 7×7 = 49 posiciones. Variante: 84 px → 5×5 |
| Acciones | 4 vecinos (arriba, abajo, izquierda, derecha) |
| Objetivo | Glimpse de otra casilla de la misma imagen, codificado con una copia congelada del encoder |
| Estado del agente | Embedding actual + embedding objetivo + máscara egocéntrica de visitados + aviso de choque (dirección del último movimiento bloqueado por el borde) |
| Algoritmo | PPO, partiendo de la implementación de CleanRL (adaptada y citada) |
| Entorno | Núcleo en tensores (episodios en paralelo) + envoltura con la interfaz de Gymnasium |
| Validación del PPO | Fase 1 también con Stable-Baselines3 sobre el mismo entorno; deben coincidir |
| Política | MLP pequeño (2 capas ocultas), a propósito: que el resultado dependa del embedding |
| Recompensa | +1 al llegar, −0,01 por paso (valores iniciales, a ajustar) |
| Señal de fine-tuning | Solo RL. El world model queda como pérdida auxiliar opcional |

Resuelto en la fase 1: la máscara de visitados es egocéntrica (no revela la posición absoluta) y el agente percibe el borde solo al chocar con él.

## Fases

### Fase 0 — Entorno y embeddings precalculados
- Entorno con `uv`, PyTorch (MPS), descarga de DINOv2 e Imagenette.
- Precalcular los 49 embeddings por imagen con el encoder congelado y guardarlos.
- Mediciones previas (probes lineales sobre embeddings congelados):
  - posición de la casilla a partir de un solo glimpse (se espera cerca del azar, 1/49);
  - desplazamiento relativo entre dos glimpses de la misma imagen.
- **Listo cuando:** embeddings guardados y los dos números medidos.

### Fase 1 — Agente PPO con encoder congelado
- Entorno de rejilla sobre los embeddings precalculados; inicio y objetivo al azar en casillas distintas.
- Líneas base: paseo aleatorio y camino más corto (cota superior).
- Comprobaciones del PPO: (1) con las posiciones propia y del objetivo dadas directamente debe llegar al 100 %; (2) Stable-Baselines3 sobre la envoltura de Gymnasium debe dar una tasa de éxito equivalente.
- **Listo cuando:** la tasa de éxito en imágenes de validación supera claramente al paseo aleatorio.

### Fase 2 — Fine-tuning del encoder con RL
- Partir del agente de la fase 1. Liberar los últimos bloques del encoder con LR 10–100 veces menor que el de la cabeza.
- El encoder corre en cada paso (ya no hay precálculo). El objetivo se codifica con la copia congelada.
- **Listo cuando:** el entrenamiento es estable y el éxito es igual o mayor que en la fase 1. Guardar el encoder ajustado.

### Fase 3 — Experimento A: ¿navega mejor?
- Dos encoders congelados: original y ajustado. Un agente nuevo desde cero sobre cada uno.
- Misma arquitectura, mismos pasos, mismas imágenes, al menos 3 semillas.
- Métricas por posición: tasa de éxito y pasos usados / camino más corto.
- No comparar distancias en latente entre encoders (un colapso parcial las falsea).

### Fase 4 — Experimento B: ¿qué perdió?
Mismo script sobre los dos encoders, con imagen completa y encoder congelado:
- k-NN de clasificación en un dataset no usado en el fine-tuning (Imagewoof; Imagenette está casi saturado para DINOv2).
- Segmentación con capa lineal (Pascal VOC).
- Deriva directa: similitud coseno / CKA entre embeddings de ambos encoders sobre las mismas imágenes.

### Fase 5 — Extensiones (solo si las fases 3 y 4 muestran algo)
- **Línea base sin RL:** ajustar el encoder con regresión directa del desplazamiento al objetivo (ver Trabajo relacionado).
- Grados de fine-tuning (LoRA, más bloques, todo libre) → curva "ganancia en navegación vs. pérdida general".
- Pérdida auxiliar de world model (predicción latente del siguiente glimpse) y comparación RL / WM / ambos.
- Menos solape (paso de 84 px), rejilla de embeddings visitados como memoria, DINOv3.

## Cómputo

- **Fases 0, 1, 3 y 4:** en el Mac (MPS). El encoder es pequeño y los embeddings se precalculan una vez.
- **Fase 2 (y la 5):** el encoder corre con gradiente en cada paso; aquí es donde puede hacer falta GPU alquilada.
- **Servicio elegido:** Modal (se lanza desde la terminal con `modal run`, sin gestionar máquinas ni SSH, pago por uso). Requiere cuenta y un inicio de sesión único con `modal setup`.
- Escribir el código de entrenamiento independiente del dispositivo (`cuda` / `mps` / `cpu`) desde el principio, para que el mismo script corra en el Mac y en Modal.

## Riesgos

- **Señal de RL ruidosa:** puede dañar el encoder. Mitigación: dos fases, LR bajo, pocos bloques libres.
- **Atajos:** con solape, el encoder puede especializarse en casar bordes. Lo detecta el experimento B.
- **Tarea demasiado fácil o difícil:** ajustar solape y longitud máxima del episodio según la fase 1.

## Trabajo relacionado

Búsqueda hecha el 2026-11-25 (seis consultas web, no exhaustiva). **No apareció ningún trabajo con la combinación completa**: navegación a un glimpse objetivo + fine-tuning de un encoder preentrenado con solo RL + medición del olvido. Lo más cercano:

| Trabajo | Qué comparte | En qué difiere |
|---|---|---|
| [seq-JEPA](https://arxiv.org/abs/2505.03176) (NeurIPS 2025) | Glimpses + acciones (sacadas), predicción latente | Política heurística, sin objetivo ni RL; entrena desde cero |
| [CanViT](https://arxiv.org/abs/2603.22570) (NeurIPS 2026) — leído | Glimpses de 128 px + memoria en rejilla ("canvas"); sin etiquetas en el preentrenamiento | Entrena desde cero destilando DINOv3 con glimpses al azar, sin RL. Luego entrena políticas con RL sobre el modelo congelado, con recompensa de segmentación (usa etiquetas). Sin tarea con objetivo; según el resumen leído, el condicionamiento a objetivo y el entrenamiento conjunto política+modelo figuran como trabajo futuro |
| [Visual Pre-training for Navigation: What Can We Learn from Noise?](https://arxiv.org/abs/2207.00052) (IROS 2023) — leído | Sin etiquetas: dado el recorte actual y el recorte objetivo, predecir dónde está el objetivo | Regresión directa en un paso (ResNet18), sin RL ni agente que se mueva; la política posterior se entrena supervisada con acciones. No analiza la representación |
| [AdaGlimpse](https://arxiv.org/abs/2404.03482) (ECCV 2024) — leído | RL (SAC) elige posición y escala del siguiente glimpse; ViT-B | Recompensa = bajada de la pérdida de reconstrucción o de destilación de un maestro. El encoder se entrena con esa pérdida, alternando épocas con el agente. Sin objetivo que alcanzar, sin medir olvido |
| [GRPO-RM](https://arxiv.org/abs/2511.15256) (2025) — leído | Ajusta DINOv2 ViT-S/14 con una pérdida tipo RL | La recompensa usa la clase correcta: en la práctica es supervisado. Una decisión por imagen, sin agente. No mide olvido |
| [EVA](https://arxiv.org/abs/2603.27340) (2026) — leído | Política de glimpses con REINFORCE | CNN pequeña, recompensa = clasificación correcta (etiquetas). Sin objetivo dado al agente |
| [ViT foveal con política de mirada](https://arxiv.org/abs/2604.20392) (2026) | Política de mirada aprendida sobre un ViT | Orientado a precisión de clasificación |
| [DINO-WM](https://arxiv.org/abs/2411.04983) (2024) | World model sobre DINOv2 para llegar a una imagen objetivo | Encoder congelado, planificación sin RL, entornos de control |
| [Recurrent Models of Visual Attention](https://arxiv.org/abs/1406.6247) (2014) | RL que decide dónde mirar | Clasificación supervisada, redes pequeñas desde cero |
| [Pretrained Visual Representations in RL](https://arxiv.org/abs/2407.17238) (2024) | Encoders preentrenados dentro de RL | Entornos de control, no navegación sobre imágenes |

Otras lecturas de base: [I-JEPA](https://arxiv.org/abs/2301.08243), [IWM](https://arxiv.org/abs/2403.00504), [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588), [Context Prediction (Doersch 2015)](https://arxiv.org/abs/1505.05192).

Tras leer los cinco marcados (2026-11-25), la combinación sigue sin aparecer. CanViT es el más cercano en arquitectura y sitúa nuestra idea en lo que declara como trabajo futuro.

**Objeción principal a prever:** "What Can We Learn from Noise?" obtiene la misma señal (dónde está el objetivo respecto a mí) con una regresión directa, sin RL. Hay que mostrar qué aporta el RL frente a eso → línea base obligatoria en la fase 5: ajustar el encoder prediciendo directamente el desplazamiento al objetivo, y comparar en los experimentos A y B.

**Idea prestada:** ese paper necesita un currículo (primero objetivos que se solapan con la vista actual, después lejanos) para que el entrenamiento converja. Aplicarlo en la fase 1 si el agente no aprende: empezar con objetivos cercanos y aumentar la distancia.

Antes de escribir un paper conviene una búsqueda más a fondo (Google Scholar, trabajos que citan a seq-JEPA, AdaGlimpse y CanViT).
