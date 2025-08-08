# Project Chimera 🐉: Construyendo un mini-CLIP en Público

Este repositorio documenta el sprint de un mes (agosto 2025) para construir un modelo de Visión-Lenguaje (mini-CLIP) desde un "cero didáctico" sobre hardware de consumidor (NVIDIA GTX 1080). El objetivo es doble: explorar las tripas de la IA multimodal y hacerlo siguiendo principios de ingeniería de software robustos y reproducibles.

**Este proyecto es un experimento en `Building in Public`. Todo el progreso, los errores y los aprendizajes serán documentados.**

---

## Principios Operativos (No Negociables)

1.  **SLOs son Ley:** Los objetivos de rendimiento y tiempo se respetan. Si no se cumplen, se activan planes de contingencia documentados.
2.  **Reproducibilidad Total:** Cada resultado debe ser reproducible. Se versiona el código, la configuración, los datos y el entorno de hardware/software.
3.  **Evaluación Honesta:** Las métricas se reportan con rigor estadístico (`media ± std` en 3 `seeds`) para evitar celebrar picos de suerte.
4.  **Documentar es Parte del Trabajo:** `KNOWN_ISSUES.md` y `CHANGELOG.md` se actualizan a diario.

---

## 🎯 SLOs (Service Level Objectives) y Criterios de Éxito

Estos son los objetivos mínimos para que el proyecto se considere viable.

| Métrica | Objetivo | Cómo se Mide |
|---|---|---|
| **Dataloader Throughput** | ≥ 300 img/s con GPU util ≥ 85% | Promedio de 60-120s con `nvidia-smi dmon -s pucm`. Se acepta <300 img/s si `util` se mantiene >85%. |
| **Prueba de Vida (Sanity)** | `loss[0] - loss[200] ≥ 0.15` en `≤ 25 min` | `wall-clock` time para el script `train_sanity.py`. |
| **Tiempo por Epoch (Real)** | `≤ 2–3 h` (para 10k pares) | `wall-clock` time del script de entrenamiento. |
| **VRAM Pico** | `≤ 7.5 GB` | `torch.cuda.max_memory_allocated()` logueado en MLflow. |
| **Viabilidad del Modelo (GO/NO-GO #2)** | `recall@10 ≥ 0.25` y `recall@1 ≥ 0.05` | Media sobre 3 seeds en el split de validación oficial de Flickr8k. |

---

## 🛠️ Tech Stack y Datos

* **Frameworks:** PyTorch, Transformers, timm
* **Herramientas:** Conda, Conda-Lock, MLflow, Ruff, Black, Pytest
* **CI/CD:** GitHub Actions
* **Datos:** Flickr8k (splits oficiales)
* **Licencia del Código:** Apache 2.0
* **Licencia de Datos:** Se respetan las licencias originales de Flickr8k/COCO.

---

## 🚀 Cómo Reproducir

1.  Clonar el repositorio: `git clone https://github.com/tu-usuario/project-chimera.git`
2.  Instalar dependencias: `make install`
3.  Ejecutar el pipeline de reproducibilidad completo (subset): `make reproduce_small`

---

## 📊 Benchmarks y Costes

*(Esta tabla se rellenará a medida que avance el proyecto)*

| Componente | Latencia (ms/batch) | Throughput | VRAM Pico (GB) |
|---|---|---|---|
| Dataloader | - | | - |
| Image Encoder | | | |
| Text Encoder | | | |

| Tarea | Tiempo de Ejecución (Wall-Clock) |
|---|---|
| Prueba de Vida | |
| Epoch Completo (10k pares) | |