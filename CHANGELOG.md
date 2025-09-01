# Changelog

## [0.1.3] - 2025-09-01

### Added
- Implemented full training pipeline in `chimera.train.train`
- Added metrics logging for wall-clock time, VRAM usage, gradient norm, and logit_scale
- Integrated checkpoint saving functionality for model recovery
- Created utility script for running training with optimal parameters
- Added GPU memory management and optimization

### Changed
- Evolved train_sanity.py into a production-ready training script
- Improved LR scheduling with warmup and cosine decay
- Enhanced contrastive loss implementation with stability improvements
- Updated documentation for training requirements and parameters

## [0.1.2] - 2025-08-15

### Added
- Implemented contrastive loss function
- created benchmark script for measuring contrastive loss performance
- Added unit tests for contrastive loss function
- Added Makefile target for running contrastive loss benchmarks

## [0.1.1] - 2025-08-14

### Added
- Implemented ImageEncoder model with ResNet18 backbone and projection head
- Created benchmark script for measuring ImageEncoder performance
- Added unit tests for ImageEncoder
- Added Makefile target for running image encoder benchmarks

### Changed
- Updated project structure to include models directory

## [0.0.1] - 2025-08-08

### Added
- Initial project setup
- CI configuration with ruff, black, isort, and pytest
- Basic Makefile and pyproject.toml configuration
- Project structure with placeholder files
## [0.1.1] - 2025-08-13

### Added
- **Documentación de Problemas Conocidos:** Actualización detallada de KNOWN_ISSUES.md con análisis completo de los resultados de la prueba de sanidad.
- **Explicación Mejorada sobre Utilización de GPU:** Documentación ampliada sobre el comportamiento esperado de la GPU en pruebas cortas y plan para su optimización futura.

### Changed
- **Formato Mejorado en Reportes:** Presentación más clara y profesional de los resultados de las pruebas en la documentación.
- **Actualización de la Fecha del Hito GO/NO-GO #1:** Reflejando la fecha actual de finalización (2025-08-13).

## [0.1.0] - 2025-08-11

### Added
- **Arquitectura de Dataloader Avanzada:** Implementación de un sistema modular en `chimera/data/dataloader.py` con Factories y Builders.
- **Script de Preparación de Datos:** `scripts/download_prepare_data.py` para descargar y procesar Flickr8k a Parquet.
- **Script de Benchmark de Dataloader:** `scripts/bench_dataloader.py` para medir throughput y uso de GPU.
- **Implementación Real de Encoders:** Versiones iniciales de `ImageEncoderResNet18` y `TextEncoderMiniLM` dentro del script de sanity.
- **Script de Prueba de Vida (`train_sanity.py`):** Implementación completa que ejecuta un ciclo de entrenamiento, loguea en MLflow y genera un reporte de SLOs.

### Fixed
- Se ha añadido lógica para manejar y documentar el `FAIL` esperado en el SLO de utilización de GPU durante la prueba de vida corta.