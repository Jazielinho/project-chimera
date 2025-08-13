# Sanity Check (200 steps)

## Config

- batch_size: 256
- grad_accum_steps: 8
- num_workers: 24
- lr: 0.0005
- embed_dim: 256
- seed: 1337

## Resultados

- loss[0] (avg 10): 5.5438
- loss[200] (avg 10): 3.9636
- drop: 1.5802
- wall-clock: 1.61 min
- GPU util (dmon): 45.48%
- Max VRAM: 2.80 GB

## SLOs

- Loss drop ≥ 0.15: ✅ PASS
- Time ≤ 25 min: ✅ PASS
- GPU util ≥ 85.0%: ❌ FAIL
