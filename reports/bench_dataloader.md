# Dataloader Benchmark Results

## Configuración

- **Parquet:** data/flickr8k/processed/flickr8k_small.parquet
- **Batch Size:** 16
- **Num Workers:** 8
- **Prefetch Factor:** 2
- **Pin Memory:** True
- **Device:** cuda
- **Warmup Iters:** 10
- **Measure Iters:** 200

## Resultados

- **Throughput:** 1998.78 imgs/s
- **Tiempo promedio por batch:** 8.00 ms
- **Tiempo P50 por batch:** 1.07 ms
- **Tiempo P95 por batch:** 43.94 ms
- **GPU Util promedio (dmon):** 0.00%

## SLOs

- Throughput ≥ 300: ✅ PASS
- GPU util ≥ 85.0%: ❌ FAIL

## Notas y Observaciones

- El benchmark **sí** incluye el tiempo de pedir el batch al DataLoader y la transferencia a GPU.
- Se realizó warmup para estabilizar mediciones.
