# Dataloader Benchmark Results

## Configuración

- **Batch Size:** 16
- **Num Workers:** 12
- **Prefetch Factor:** 2
- **Pin Memory:** True
- **Device:** cuda

## Resultados

- **Throughput:** 17076.05 imgs/s
- **Tiempo promedio por batch:** 0.94 ms
- **Tiempo P50 por batch:** 0.88 ms
- **Tiempo P95 por batch:** 1.30 ms

**SLO Status:** ✅ PASS - Throughput 17076.05 imgs/s ≥ 300 imgs/s

## Notas y Observaciones

- El benchmark incluye el tiempo de transferencia a GPU para simular el flujo real de datos.
- Se realizó un warmup previo para estabilizar mediciones.
