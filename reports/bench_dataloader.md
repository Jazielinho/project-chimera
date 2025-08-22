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

- **Throughput:** 2004.81 imgs/s
- **Tiempo promedio por batch:** 7.98 ms
- **Tiempo P50 por batch:** 1.24 ms
- **Tiempo P95 por batch:** 32.73 ms
- **GPU Util promedio (dmon):** 0.00%

## SLOs

- Throughput ≥ 300: ✅ PASS
- GPU util ≥ 85.0%: ❌ FAIL

## Calidad de datos (QC)

- **Filas:** 1500
- **Decodificación de imagen (muestra 512):** 100.00% OK
- **Duplicados en `image_id`:** 0.00%

**Caption (longitud de string crudo)**

- media: 293.7 | P10: 233 | P50: 290 | P90: 359 | % vacías: 0.00%

**Nulos por columna**

| Columna | % Nulos |
|---|---:|
| `image_id` | 0.00% |
| `caption` | 0.00% |
| `image` | 0.00% |

## Muestras (Original vs Transformada)

> La transformada muestra exactamente lo que entra al modelo (Resize + CenterCrop + Normalize invertido para visualización).

<table>
<tr>
<td style="vertical-align:top; padding:10px; text-align:center;"><div style="font-size:12px; margin-bottom:4px;"><b>2991993027_36ac04e9a0.jpg</b></div><div style="display:inline-block; text-align:center;"><div style="font-size:11px; margin:2px 0;">Original</div><img src="samples/orig_00_2991993027_36ac04e9a0.jpg.png" width="220"></div>&nbsp;&nbsp;<div style="display:inline-block; text-align:center;"><div style="font-size:11px; margin:2px 0;">Transformada</div><img src="samples/xform_00_2991993027_36ac04e9a0.jpg.png" width="220"></div><div style="font-size:12px; margin-top:6px;"><i>A group of people dressed as zom</i></div></td>
<td style="vertical-align:top; padding:10px; text-align:center;"><div style="font-size:12px; margin-bottom:4px;"><b>3712923460_1b20ebb131.jpg</b></div><div style="display:inline-block; text-align:center;"><div style="font-size:11px; margin:2px 0;">Original</div><img src="samples/orig_01_3712923460_1b20ebb131.jpg.png" width="220"></div>&nbsp;&nbsp;<div style="display:inline-block; text-align:center;"><div style="font-size:11px; margin:2px 0;">Transformada</div><img src="samples/xform_01_3712923460_1b20ebb131.jpg.png" width="220"></div><div style="font-size:12px; margin-top:6px;"><i>a bunch of people in camo pants </i></div></td>
</tr>
<tr>
<td style="vertical-align:top; padding:10px; text-align:center;"><div style="font-size:12px; margin-bottom:4px;"><b>617038406_4092ee91dd.jpg</b></div><div style="display:inline-block; text-align:center;"><div style="font-size:11px; margin:2px 0;">Original</div><img src="samples/orig_02_617038406_4092ee91dd.jpg.png" width="220"></div>&nbsp;&nbsp;<div style="display:inline-block; text-align:center;"><div style="font-size:11px; margin:2px 0;">Transformada</div><img src="samples/xform_02_617038406_4092ee91dd.jpg.png" width="220"></div><div style="font-size:12px; margin-top:6px;"><i>A group of people gathering arou</i></div></td>
<td style="vertical-align:top; padding:10px; text-align:center;"><div style="font-size:12px; margin-bottom:4px;"><b>3015898903_70bebb8903.jpg</b></div><div style="display:inline-block; text-align:center;"><div style="font-size:11px; margin:2px 0;">Original</div><img src="samples/orig_03_3015898903_70bebb8903.jpg.png" width="220"></div>&nbsp;&nbsp;<div style="display:inline-block; text-align:center;"><div style="font-size:11px; margin:2px 0;">Transformada</div><img src="samples/xform_03_3015898903_70bebb8903.jpg.png" width="220"></div><div style="font-size:12px; margin-top:6px;"><i>A man and 2 women in dark clothi</i></div></td>
</tr>
<tr>
<td style="vertical-align:top; padding:10px; text-align:center;"><div style="font-size:12px; margin-bottom:4px;"><b>2765747519_2b851e01d6.jpg</b></div><div style="display:inline-block; text-align:center;"><div style="font-size:11px; margin:2px 0;">Original</div><img src="samples/orig_04_2765747519_2b851e01d6.jpg.png" width="220"></div>&nbsp;&nbsp;<div style="display:inline-block; text-align:center;"><div style="font-size:11px; margin:2px 0;">Transformada</div><img src="samples/xform_04_2765747519_2b851e01d6.jpg.png" width="220"></div><div style="font-size:12px; margin-top:6px;"><i>A baby in a green bag peeks out </i></div></td>
<td style="vertical-align:top; padding:10px; text-align:center;"><div style="font-size:12px; margin-bottom:4px;"><b>2218519240_cac5aab53c.jpg</b></div><div style="display:inline-block; text-align:center;"><div style="font-size:11px; margin:2px 0;">Original</div><img src="samples/orig_05_2218519240_cac5aab53c.jpg.png" width="220"></div>&nbsp;&nbsp;<div style="display:inline-block; text-align:center;"><div style="font-size:11px; margin:2px 0;">Transformada</div><img src="samples/xform_05_2218519240_cac5aab53c.jpg.png" width="220"></div><div style="font-size:12px; margin-top:6px;"><i>A man is wearing protective hair</i></div></td>
</tr>
<tr>
<td style="vertical-align:top; padding:10px; text-align:center;"><div style="font-size:12px; margin-bottom:4px;"><b>2369452202_8b0e8e25ca.jpg</b></div><div style="display:inline-block; text-align:center;"><div style="font-size:11px; margin:2px 0;">Original</div><img src="samples/orig_06_2369452202_8b0e8e25ca.jpg.png" width="220"></div>&nbsp;&nbsp;<div style="display:inline-block; text-align:center;"><div style="font-size:11px; margin:2px 0;">Transformada</div><img src="samples/xform_06_2369452202_8b0e8e25ca.jpg.png" width="220"></div><div style="font-size:12px; margin-top:6px;"><i>A crowd of people is standing ar</i></div></td>
<td style="vertical-align:top; padding:10px; text-align:center;"><div style="font-size:12px; margin-bottom:4px;"><b>241346215_037e18403a.jpg</b></div><div style="display:inline-block; text-align:center;"><div style="font-size:11px; margin:2px 0;">Original</div><img src="samples/orig_07_241346215_037e18403a.jpg.png" width="220"></div>&nbsp;&nbsp;<div style="display:inline-block; text-align:center;"><div style="font-size:11px; margin:2px 0;">Transformada</div><img src="samples/xform_07_241346215_037e18403a.jpg.png" width="220"></div><div style="font-size:12px; margin-top:6px;"><i>A horse mascot gives high fives </i></div></td>
</tr>
</table>

## Notas y Observaciones

- El benchmark **sí** incluye el tiempo de pedir el batch al DataLoader y la transferencia a GPU.
- Se realizó warmup para estabilizar mediciones.
