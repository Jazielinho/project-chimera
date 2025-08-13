# Changelog

## [0.1.0] - 2025-08-11

### Added
- **Arquitectura de Dataloader Avanzada:** Implementación de un sistema modular en `chimera/data/dataloader.py` con Factories y Builders.
- **Script de Preparación de Datos:** `scripts/download_prepare_data.py` para descargar y procesar Flickr8k a Parquet.
- **Script de Benchmark de Dataloader:** `scripts/bench_dataloader.py` para medir throughput y uso de GPU.
- **Implementación Real de Encoders:** Versiones iniciales de `ImageEncoderResNet18` y `TextEncoderMiniLM` dentro del script de sanity.
- **Script de Prueba de Vida (`train_sanity.py`):** Implementación completa que ejecuta un ciclo de entrenamiento, loguea en MLflow y genera un reporte de SLOs.

### Fixed
- Se ha añadido lógica para manejar y documentar el `FAIL` esperado en el SLO de utilización de GPU durante la prueba de vida corta.