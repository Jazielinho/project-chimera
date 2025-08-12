# Problemas Conocidos y Ajustes Realizados

## 2025-08-11 — Día 2: GO/NO-GO #1 (Prueba de Vida)

### SHA: [pendiente insertar SHA después del PR]

#### Dataloader

- Se implementó un dataloader optimizado con los siguientes parámetros:
  - batch_size: 16
  - num_workers: 12
  - pin_memory: True
  - prefetch_factor: 2
  - Transformaciones: resize a 224x224, normalización estándar ImageNet

- El dataloader alcanza un throughput de [se actualizará después de la ejecución] imgs/s, lo cual [cumple/no cumple] el SLO de ≥300 imgs/s.

#### Prueba de Sanity (train_sanity.py)

- Se implementó un ciclo de entrenamiento de 200 steps con:
  - MockImageEncoder y MockTextEncoder para simular el flujo completo
  - ContrastiveLoss (InfoNCE simétrica) con clamp de logit_scale a 4.6
  - AdamW (lr=1e-3)
  - Acumulación de gradientes (4 pasos)
  - Gradient clipping (norm=1.0)

- La prueba de vida [cumple/no cumple] los SLOs:
  - loss_drop: [se actualizará] (SLO: ≥0.15)
  - tiempo: [se actualizará] min (SLO: ≤25 min)
  - utilización GPU: [se actualizará]% (SLO: ≥85%)

#### Ajustes realizados

- Ajustes del Plan B (si fueron necesarios):
  - [Se actualizará según sea necesario]

#### Próximos pasos

- GO: Continuar con la implementación del Image Encoder (Día 3)
- [o si hay NO-GO: documentar plan de rescate]