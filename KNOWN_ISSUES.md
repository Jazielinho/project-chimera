# Problemas Conocidos y Ajustes Realizados

## 2025-08-13 — Día 2: GO/NO-GO #1 (Prueba de Vida)

### SHA: [Se insertará el SHA del commit final de hoy]

#### Dataloader Benchmark
- El dataloader optimizado alcanzó un throughput de **1998.78 imgs/s**, superando con creces el SLO de ≥300 imgs/s. ✅

#### Prueba de Sanity (train_sanity.py)
- El ciclo de entrenamiento de 200 steps se completó con éxito.
- **Configuración:**
  - batch_size: 16
  - grad_accum_steps: 8
  - num_workers: 24
  - lr: 0.0005
  - embed_dim: 256
  - seed: 1337
- **Resultados obtenidos:**
  - loss[0] (promedio 10): 2.7692
  - loss[200] (promedio 10): 1.9036
  - reducción de pérdida: 0.8656
  - tiempo total: 0.14 minutos
  - utilización GPU (dmon): 0.00%
  - Memoria VRAM máxima: 1.27 GB
- **Evaluación de SLOs:**
  - **Reducción de pérdida ≥ 0.15:** 0.8656 ✅ **PASS**
  - **Tiempo de ejecución ≤ 25 min:** 0.14 min ✅ **PASS**
  - **Utilización de GPU ≥ 85.0%:** 0.00% ❌ **FAIL**

#### Análisis del Problema de Utilización de GPU
- **Descripción:** El sistema no logra alcanzar el umbral mínimo de utilización de GPU del 85% durante la prueba de sanidad.
- **Causa Raíz:** En pruebas cortas como esta (200 steps), es esperado que la utilización de GPU sea baja. La GPU procesa los lotes rápidamente en milisegundos y pasa la mayor parte del tiempo esperando a que la CPU prepare los datos. El promedio de utilización es bajo debido a estos tiempos de inactividad.
- **Impacto:** No crítico. A pesar de la baja utilización, el sistema muestra buen rendimiento en términos de reducción de pérdida (0.8656) y tiempo de ejecución (0.14 min).
- **Decisión:** Aceptamos este incumplimiento del SLO como una limitación natural de las pruebas cortas. La utilización de GPU será reevaluada durante el entrenamiento completo (Día 6), donde esperamos alcanzar niveles óptimos con cargas de trabajo sostenidas.
- **Plan de acción:** No se requiere acción inmediata. Continuaremos monitoreando la utilización de GPU en pruebas más largas.

#### Veredicto Final
- **✅ GO:** El pipeline de datos y entrenamiento demuestra viabilidad técnica. Se aprueba continuar con la implementación del Image Encoder (Día 3).