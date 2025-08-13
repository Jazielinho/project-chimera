# Sanity Check (200 steps)

## Config

- batch_size: 16
- grad_accum_steps: 8
- num_workers: 8
- lr: 0.0005
- embed_dim: 256
- seed: 1337

## Resultados

- loss[0] (avg 10): 2.7692
- loss[200] (avg 10): 1.9036
- drop: 0.8656
- wall-clock: 0.12 min
- GPU util (dmon): 0.00%
- Max VRAM: 1.27 GB

## SLOs

- Loss drop ≥ 0.15: ✅ PASS
- Time ≤ 25 min: ✅ PASS
- GPU util ≥ 85.0%: ❌ FAIL
