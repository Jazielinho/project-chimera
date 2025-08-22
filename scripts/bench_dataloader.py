#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para benchmarking del dataloader.

Este script evalúa el rendimiento del dataloader midiendo la velocidad de carga
y transferencia de datos a la GPU.
"""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import trange

from chimera.data.dataloader import create_dataloader
from chimera.utils.mlflow_utils import log_reproducibility_passport


def start_dmon(log_path: str = "dmon_dataloader.log"):
    try:
        p = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "pucm", "-d", "1", "-o", "TD", "-f", log_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return p, log_path
    except Exception as e:
        print(f"[WARN] No pude lanzar dmon: {e}")
        return None, None


def stop_dmon(proc, log_path: str | None):
    gpu_util_avg = 0.0
    if proc and log_path and Path(log_path).exists():
        # corta muestreo
        time.sleep(2)
        proc.terminate()
        try:
            lines = Path(log_path).read_text().splitlines()[1:]
            vals = []
            for ln in lines:
                parts = ln.strip().split()
                # columna 5 = "sm" (utilización)
                if len(parts) >= 5:
                    try:
                        vals.append(float(parts[4]))
                    except Exception:
                        pass
            if vals:
                gpu_util_avg = sum(vals) / len(vals)
        except Exception as e:
            print(f"[WARN] Error leyendo {log_path}: {e}")
    return gpu_util_avg


def _sanitize_filename(s: str) -> str:
    keep = "-_.()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(c if c in keep else "_" for c in (s or ""))[:100]


def _denorm_images(imgs: torch.Tensor) -> torch.Tensor:
    # invierte la normalización de ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    imgs = imgs.detach().cpu() * std + mean
    return imgs.clamp(0.0, 1.0)


def _decode_original(x) -> Image.Image:
    # soporta bytes o ruta str
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x)).convert("RGB")
    if isinstance(x, str):
        try:
            return Image.open(x).convert("RGB")
        except Exception:
            raise
    raise TypeError(f"Tipo de imagen no soportado para original: {type(x)}")


def save_sample_pairs(
    parquet_path: str,
    dl: torch.utils.data.DataLoader,
    out_dir: Path,
    max_samples: int = 8,
) -> list[dict]:
    """
    Guarda pares (original, transformada) y devuelve metadatos:
      [{ 'orig': 'orig_00_....png', 'xform':'xform_00_....png',
         'image_id': '...', 'caption': '...' }, ...]
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # primer batch del DL (transformadas)
    it = iter(dl)
    batch = next(it, None)
    if batch is None:
        it = iter(dl)
        batch = next(it)

    ids = [str(x) for x in batch.get("image_id", [])]
    caps = batch.get("caption", [])
    imgs_t = _denorm_images(batch["image"])  # [B, C, H, W]

    k = min(max_samples, imgs_t.size(0))
    ids, caps, imgs_t = ids[:k], caps[:k], imgs_t[:k]

    # dataframe para originales
    df = pd.read_parquet(parquet_path)
    # índice por image_id para acceso O(1)
    if "image_id" not in df.columns or "image" not in df.columns:
        raise ValueError("El Parquet debe contener columnas 'image_id' e 'image'")
    df_idx = df.set_index("image_id")

    to_pil = T.ToPILImage()
    pairs: list[dict] = []

    for i, img_id in enumerate(ids):
        safe = _sanitize_filename(img_id)
        # transformada
        xform_png = f"xform_{i:02d}_{safe}.png"
        to_pil(imgs_t[i]).save(out_dir / xform_png)

        # original (decodifica desde df)
        try:
            row = df_idx.loc[img_id]
        except KeyError:
            # si el image_id no está, salta pero deja constancia
            row = None
        orig_png = f"orig_{i:02d}_{safe}.png"
        if row is not None:
            try:
                pil = _decode_original(row["image"])
                pil.save(out_dir / orig_png)
            except Exception:
                # si falla decodificar, crea marcador vacío
                Image.new("RGB", (224, 224), (220, 220, 220)).save(out_dir / orig_png)
        else:
            Image.new("RGB", (224, 224), (180, 180, 180)).save(out_dir / orig_png)

        pairs.append(
            {
                "orig": orig_png,
                "xform": xform_png,
                "image_id": img_id,
                "caption": caps[i] if i < len(caps) else "",
            }
        )
    return pairs


def run_quality_checks(parquet_path: str, sample_size: int = 512) -> dict:
    """
    QC ligero y rápido sobre el Parquet.
    - tamaños y nulos
    - duplicados de image_id
    - longitud de captions
    - tasa de decodificación de imágenes (muestra)
    """
    df = pd.read_parquet(parquet_path)
    n = len(df)

    # nulos
    nulls = {col: float(df[col].isna().mean()) for col in df.columns}

    # duplicados
    dup_ratio = (
        float(df["image_id"].duplicated().mean()) if "image_id" in df else float("nan")
    )

    # longitudes de caption (como string crudo; robusto ante listas/arrays)
    def _cap_len(x):
        try:
            return len(str(x))
        except Exception:
            return 0

    cap_lens = df["caption"].map(_cap_len) if "caption" in df else pd.Series(dtype=int)
    cap_stats = {
        "len_mean": float(cap_lens.mean()) if not cap_lens.empty else float("nan"),
        "len_p10": (
            float(cap_lens.quantile(0.10)) if not cap_lens.empty else float("nan")
        ),
        "len_p50": (
            float(cap_lens.quantile(0.50)) if not cap_lens.empty else float("nan")
        ),
        "len_p90": (
            float(cap_lens.quantile(0.90)) if not cap_lens.empty else float("nan")
        ),
        "empty_ratio": (
            float((cap_lens == 0).mean()) if not cap_lens.empty else float("nan")
        ),
    }

    # decodificación de imágenes (muestra aleatoria reproducible)
    ok, bad = 0, 0
    if "image" in df.columns:
        ss = min(sample_size, n)
        sample_df = df.sample(n=ss, random_state=0) if ss < n else df
        for x in sample_df["image"].values:
            try:
                _ = _decode_original(x)  # intenta abrir
                ok += 1
            except Exception:
                bad += 1
    decode_ratio_ok = ok / (ok + bad) if (ok + bad) > 0 else float("nan")

    return {
        "rows": n,
        "nulls": nulls,
        "dup_image_id_ratio": dup_ratio,
        "caption_stats": cap_stats,
        "decode_ok_ratio": decode_ratio_ok,
        "decode_checked": ok + bad,
    }


def write_benchmark_report(
    args,
    imgs_per_sec: float,
    avg_time: float,
    p50_time: float,
    p95_time: float,
    gpu_util_avg: float,
    status_thr: bool,
    status_gpu: bool,
    pairs: list[dict] | None = None,  # <-- NUEVO
    qc: dict | None = None,  # <-- NUEVO
):
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "bench_dataloader.md"

    def _trunc(s: str, n: int = 120) -> str:
        return (s or "").replace("\n", " ")[:n]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Dataloader Benchmark Results\n\n")

        # --- Config ---
        f.write("## Configuración\n\n")
        f.write(
            f"- **Parquet:** {os.path.join(args.data_dir, 'processed', 'flickr8k_small.parquet')}\n"
        )
        f.write(f"- **Batch Size:** {args.batch_size}\n")
        f.write(f"- **Num Workers:** {args.num_workers}\n")
        f.write(f"- **Prefetch Factor:** {args.prefetch_factor}\n")
        f.write(f"- **Pin Memory:** {args.pin_memory}\n")
        f.write(
            f"- **Device:** {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\n"
        )
        f.write(f"- **Warmup Iters:** {args.num_warmup}\n")
        f.write(f"- **Measure Iters:** {args.num_iterations}\n\n")

        # --- Resultados ---
        f.write("## Resultados\n\n")
        f.write(f"- **Throughput:** {imgs_per_sec:.2f} imgs/s\n")
        f.write(f"- **Tiempo promedio por batch:** {avg_time*1000:.2f} ms\n")
        f.write(f"- **Tiempo P50 por batch:** {p50_time*1000:.2f} ms\n")
        f.write(f"- **Tiempo P95 por batch:** {p95_time*1000:.2f} ms\n")
        f.write(f"- **GPU Util promedio (dmon):** {gpu_util_avg:.2f}%\n\n")

        # --- SLOs ---
        slo_throughput = 300
        slo_gpu_util = 85.0
        f.write("## SLOs\n\n")
        f.write(
            f"- Throughput ≥ {slo_throughput}: {'✅ PASS' if status_thr else '❌ FAIL'}\n"
        )
        f.write(
            f"- GPU util ≥ {slo_gpu_util}%: {'✅ PASS' if status_gpu else '❌ FAIL'}\n\n"
        )

        # --- Calidad de datos ---
        if qc:
            f.write("## Calidad de datos (QC)\n\n")
            f.write(f"- **Filas:** {qc.get('rows', 'n/a')}\n")
            f.write(
                f"- **Decodificación de imagen (muestra {qc.get('decode_checked', 0)}):** "
                f"{qc.get('decode_ok_ratio', float('nan'))*100:.2f}% OK\n"
            )
            f.write(
                f"- **Duplicados en `image_id`:** {qc.get('dup_image_id_ratio', float('nan'))*100:.2f}%\n\n"
            )

            cap = qc.get("caption_stats", {})
            f.write("**Caption (longitud de string crudo)**\n\n")
            f.write(
                f"- media: {cap.get('len_mean', float('nan')):.1f} | "
                f"P10: {cap.get('len_p10', float('nan')):.0f} | "
                f"P50: {cap.get('len_p50', float('nan')):.0f} | "
                f"P90: {cap.get('len_p90', float('nan')):.0f} | "
                f"% vacías: {cap.get('empty_ratio', float('nan'))*100:.2f}%\n\n"
            )

            f.write("**Nulos por columna**\n\n")
            f.write("| Columna | % Nulos |\n|---|---:|\n")
            for col, ratio in (qc.get("nulls", {}) or {}).items():
                f.write(f"| `{col}` | {ratio*100:.2f}% |\n")
            f.write("\n")

        # --- Muestras: Original vs Transformada ---
        if pairs:
            f.write("## Muestras (Original vs Transformada)\n\n")
            f.write(
                "> La transformada muestra exactamente lo que entra al modelo (Resize + CenterCrop + Normalize invertido para visualización).\n\n"
            )

            # tabla 2 columnas: cada celda contiene original y transformada
            f.write("<table>\n")
            for i, meta in enumerate(pairs):
                if i % 2 == 0:
                    f.write("<tr>\n")

                f.write(
                    '<td style="vertical-align:top; padding:10px; text-align:center;">'
                )
                f.write(
                    f'<div style="font-size:12px; margin-bottom:4px;"><b>{_trunc(meta.get("image_id",""), 80)}</b></div>'
                )
                f.write('<div style="display:inline-block; text-align:center;">')
                f.write('<div style="font-size:11px; margin:2px 0;">Original</div>')
                f.write(f'<img src="samples/{meta["orig"]}" width="220">')
                f.write("</div>")
                f.write("&nbsp;&nbsp;")
                f.write('<div style="display:inline-block; text-align:center;">')
                f.write('<div style="font-size:11px; margin:2px 0;">Transformada</div>')
                f.write(f'<img src="samples/{meta["xform"]}" width="220">')
                f.write("</div>")
                if meta.get("caption"):
                    f.write(
                        f'<div style="font-size:12px; margin-top:6px;"><i>{_trunc(meta["caption"], 160)}</i></div>'
                    )
                f.write("</td>\n")

                if i % 2 == 1:
                    f.write("</tr>\n")
            # si número impar, cierra la fila
            if len(pairs) % 2 == 1:
                f.write("</tr>\n")
            f.write("</table>\n\n")

        # --- Notas ---
        f.write("## Notas y Observaciones\n\n")
        f.write(
            "- El benchmark **sí** incluye el tiempo de pedir el batch al DataLoader y la transferencia a GPU.\n"
        )
        f.write("- Se realizó warmup para estabilizar mediciones.\n")

    print(f"[INFO] Reporte escrito en {report_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Usando dispositivo: {device}")

    mlflow.set_experiment("Dataloader Benchmark")
    with mlflow.start_run(run_name="Dataloader_Benchmark"):
        log_reproducibility_passport()

        # Log de parámetros
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("num_workers", args.num_workers)
        mlflow.log_param("prefetch_factor", args.prefetch_factor)
        mlflow.log_param("pin_memory", args.pin_memory)
        mlflow.log_param("num_iterations", args.num_iterations)
        mlflow.log_param("num_warmup", args.num_warmup)

        data_path = os.path.join(args.data_dir, "processed", "flickr8k_small.parquet")
        dl = create_dataloader(
            parquet_path=data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            shuffle=True,
        )

        # dmon
        proc, log_path = start_dmon()

        # warmup
        it = iter(dl)
        for _ in range(args.num_warmup):
            batch = next(it, None)
            if batch is None:
                it = iter(dl)
                batch = next(it)
            imgs = batch["image"].to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()

        # medida
        times = []
        total_images = 0
        it = iter(dl)
        for _ in trange(args.num_iterations, desc="Benchmark"):
            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()

            batch = next(it, None)
            if batch is None:
                it = iter(dl)
                batch = next(it)
            imgs = batch["image"].to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()

            t1 = time.perf_counter()
            times.append(t1 - t0)
            total_images += imgs.size(0)

        # fin dmon
        gpu_util_avg = stop_dmon(proc, log_path)

        times = np.array(times, dtype=np.float64)
        avg_time = float(times.mean())
        p50_time = float(np.percentile(times, 50))
        p95_time = float(np.percentile(times, 95))
        imgs_per_sec = total_images / float(times.sum())

        # métricas
        mlflow.log_metric("avg_batch_time", avg_time)
        mlflow.log_metric("p50_batch_time", p50_time)
        mlflow.log_metric("p95_batch_time", p95_time)
        mlflow.log_metric("throughput_imgs_per_sec", imgs_per_sec)
        mlflow.log_metric("gpu_util_avg", gpu_util_avg)

        slo_throughput = 300.0
        slo_gpu_util = 85.0
        status_thr = imgs_per_sec >= slo_throughput
        status_gpu = gpu_util_avg >= slo_gpu_util
        mlflow.set_tag("slo.dataloader_throughput", "PASS" if status_thr else "FAIL")
        mlflow.set_tag("slo.dataloader_gpu_util", "PASS" if status_gpu else "FAIL")

        samples_dir = Path("reports") / "samples"
        parquet_path = os.path.join(
            args.data_dir, "processed", "flickr8k_small.parquet"
        )

        pairs = save_sample_pairs(
            parquet_path=parquet_path,
            dl=dl,
            out_dir=samples_dir,
            max_samples=args.num_samples,
        )

        qc = run_quality_checks(
            parquet_path=parquet_path, sample_size=args.qc_sample_size
        )

        write_benchmark_report(
            args,
            imgs_per_sec,
            avg_time,
            p50_time,
            p95_time,
            gpu_util_avg,
            status_thr,
            status_gpu,
            pairs=pairs,
            qc=qc,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark del dataloader")
    parser.add_argument("--data_dir", type=str, default="data/flickr8k")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--num_iterations", type=int, default=200)
    parser.add_argument("--num_warmup", type=int, default=10)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Número de pares original/transformada a mostrar en el reporte",
    )
    parser.add_argument(
        "--qc_sample_size",
        type=int,
        default=512,
        help="Tamaño de muestra para el chequeo de decodificación de imágenes",
    )
    args = parser.parse_args()
    main(args)
