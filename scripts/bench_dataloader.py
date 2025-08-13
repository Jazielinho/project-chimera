#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para benchmarking del dataloader.

Este script evalúa el rendimiento del dataloader midiendo la velocidad de carga
y transferencia de datos a la GPU.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
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


def write_benchmark_report(
    args,
    imgs_per_sec: float,
    avg_time: float,
    p50_time: float,
    p95_time: float,
    gpu_util_avg: float,
    status_thr: bool,
    status_gpu: bool,
):
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "bench_dataloader.md"

    with open(report_path, "w") as f:
        f.write("# Dataloader Benchmark Results\n\n")
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

        f.write("## Resultados\n\n")
        f.write(f"- **Throughput:** {imgs_per_sec:.2f} imgs/s\n")
        f.write(f"- **Tiempo promedio por batch:** {avg_time*1000:.2f} ms\n")
        f.write(f"- **Tiempo P50 por batch:** {p50_time*1000:.2f} ms\n")
        f.write(f"- **Tiempo P95 por batch:** {p95_time*1000:.2f} ms\n")
        f.write(f"- **GPU Util promedio (dmon):** {gpu_util_avg:.2f}%\n\n")

        slo_throughput = 300
        slo_gpu_util = 85.0
        f.write("## SLOs\n\n")
        f.write(
            f"- Throughput ≥ {slo_throughput}: {'✅ PASS' if status_thr else '❌ FAIL'}\n"
        )
        f.write(
            f"- GPU util ≥ {slo_gpu_util}%: {'✅ PASS' if status_gpu else '❌ FAIL'}\n\n"
        )

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

        write_benchmark_report(
            args,
            imgs_per_sec,
            avg_time,
            p50_time,
            p95_time,
            gpu_util_avg,
            status_thr,
            status_gpu,
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
    args = parser.parse_args()
    main(args)
