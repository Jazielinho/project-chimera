# Registro de Cambios

## 2025-08-10 — Día 2: GO/NO-GO #1 (Prueba de Vida)

### Añadido

- Script de descarga y preparación de datos (`scripts/download_prepare_data.py`)
  - Descarga Flickr8k desde Hugging Face
  - Procesa y guarda en formato Parquet (completo y subset)
  - Genera checksums para reproducibilidad

- Dataloader optimizado (`chimera/data/dataloader.py`)
  - Clase Flickr8kDataset con soporte para diferentes estructuras de datos
  - Función create_dataloader con parámetros optimizados
  - Benchmark para medir rendimiento real

- Script de benchmark para dataloader (`scripts/bench_dataloader.py`)
  - Mide throughput en imgs/s incluyendo transferencia a GPU
  - Genera informe en `reports/bench_dataloader.md`

- Modelos mock para pruebas (`chimera/models/mock_models.py`)
  - MockImageEncoder: Simula encoder de imágenes con salida normalizada de 256 dims
  - MockTextEncoder: Simula encoder de texto con salida normalizada de 256 dims

- Implementación de loss contrastiva (`chimera/losses/contrastive.py`)
  - InfoNCE simétrica (imagen→texto y texto→imagen)
  - Clamp de logit_scale para estabilidad numérica

- Script de entrenamiento para prueba de vida (`chimera/train/train_sanity.py`)
  - Ciclo completo de 200 steps con setup real
  - Medición de wall-clock y métricas de rendimiento
  - Verificación de SLOs (loss_drop, tiempo, GPU util)

### Modificado

- KNOWN_ISSUES.md: Documentación de la prueba de vida y ajustes realizados
- bench_dataloader.md: Informe del benchmark del dataloader

### Eliminado

- N/A