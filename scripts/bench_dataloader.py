#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para benchmarking del dataloader.

Este script evalúa el rendimiento del dataloader midiendo la velocidad de carga
y transferencia de datos a la GPU.
"""

import argparse
import os
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from tqdm import tqdm

from chimera.data.dataloader import create_dataloader
from chimera.utils.mlflow_utils import log_reproducibility_passport


def main(args):
    """
    Función principal para el benchmark del dataloader.

    Args:
        args: Argumentos de línea de comandos
    """
    # Verificar disponibilidad de GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Iniciar MLflow para logueo
    mlflow.set_experiment("Dataloader Benchmark")
    with mlflow.start_run(run_name="Dataloader_Benchmark"):
        # Loguear el pasaporte de reproducibilidad
        log_reproducibility_passport()

        # Loguear parámetros
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("num_workers", args.num_workers)
        mlflow.log_param("prefetch_factor", args.prefetch_factor)
        mlflow.log_param("pin_memory", args.pin_memory)
        mlflow.log_param("num_iterations", args.num_iterations)
        mlflow.log_param("num_warmup", args.num_warmup)

        # Crear dataloader
        data_path = os.path.join(args.data_dir, "processed", "flickr8k_small.parquet")
        dataloader = create_dataloader(
            parquet_path=data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            shuffle=True,
        )

        # Calentamiento (warmup)
        print(f"Iniciando warmup por {args.num_warmup} iteraciones...")
        for i, batch in enumerate(dataloader):
            if i >= args.num_warmup:
                break
            # Mover datos a GPU para simular el flujo completo
            images = batch["image"].to(device, non_blocking=True)
            # Sincronizar para asegurar que la transferencia se complete
            torch.cuda.synchronize()

        # Benchmark principal
        print(f"Iniciando benchmark por {args.num_iterations} iteraciones...")
        times = []
        total_images = 0

        # Iterar sobre el dataloader y medir tiempos
        for i, batch in tqdm(enumerate(dataloader), total=args.num_iterations):
            if i >= args.num_iterations:
                break

            start_time = time.time()

            # Mover datos a GPU
            images = batch["image"].to(device, non_blocking=True)
            # Sincronizar para medir tiempo preciso
            torch.cuda.synchronize()

            end_time = time.time()
            batch_time = end_time - start_time
            times.append(batch_time)
            total_images += images.size(0)

        # Calcular estadísticas
        times = np.array(times)
        avg_time = np.mean(times)
        p50_time = np.percentile(times, 50)
        p95_time = np.percentile(times, 95)
        imgs_per_sec = total_images / np.sum(times)

        # Imprimir resultados
        print("\nResultados del benchmark:")
        print(f"Tiempo promedio por batch: {avg_time:.4f} s")
        print(f"Tiempo P50 por batch: {p50_time:.4f} s")
        print(f"Tiempo P95 por batch: {p95_time:.4f} s")
        print(f"Throughput: {imgs_per_sec:.2f} imgs/s")

        # Loguear métricas en MLflow
        mlflow.log_metric("avg_batch_time", avg_time)
        mlflow.log_metric("p50_batch_time", p50_time)
        mlflow.log_metric("p95_batch_time", p95_time)
        mlflow.log_metric("throughput_imgs_per_sec", imgs_per_sec)

        # Verificar SLO
        slo_throughput = 300  # imágenes por segundo
        if imgs_per_sec >= slo_throughput:
            print(
                f"✅ SLO CUMPLIDO: Throughput {imgs_per_sec:.2f} imgs/s ≥ {slo_throughput} imgs/s"
            )
            mlflow.set_tag("slo.throughput", "PASS")
        else:
            print(
                f"❌ SLO NO CUMPLIDO: Throughput {imgs_per_sec:.2f} imgs/s < {slo_throughput} imgs/s"
            )
            mlflow.set_tag("slo.throughput", "FAIL")

        # Escribir resultados en un archivo para documentación
        write_benchmark_report(args, imgs_per_sec, avg_time, p50_time, p95_time)


def write_benchmark_report(args, imgs_per_sec, avg_time, p50_time, p95_time):
    """
    Escribe los resultados del benchmark en un archivo markdown.

    Args:
        args: Argumentos de línea de comandos
        imgs_per_sec (float): Imágenes procesadas por segundo
        avg_time (float): Tiempo promedio por batch
        p50_time (float): Tiempo P50 por batch
        p95_time (float): Tiempo P95 por batch
    """
    # Crear directorio reports si no existe
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Ruta al archivo de reporte
    report_path = reports_dir / "bench_dataloader.md"

    # Escribir reporte
    with open(report_path, "w") as f:
        f.write("# Dataloader Benchmark Results\n\n")

        f.write("## Configuración\n\n")
        f.write(f"- **Batch Size:** {args.batch_size}\n")
        f.write(f"- **Num Workers:** {args.num_workers}\n")
        f.write(f"- **Prefetch Factor:** {args.prefetch_factor}\n")
        f.write(f"- **Pin Memory:** {args.pin_memory}\n")
        f.write(
            f"- **Device:** {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\n\n"
        )

        f.write("## Resultados\n\n")
        f.write(f"- **Throughput:** {imgs_per_sec:.2f} imgs/s\n")
        f.write(f"- **Tiempo promedio por batch:** {avg_time*1000:.2f} ms\n")
        f.write(f"- **Tiempo P50 por batch:** {p50_time*1000:.2f} ms\n")
        f.write(f"- **Tiempo P95 por batch:** {p95_time*1000:.2f} ms\n\n")

        # SLO status
        slo_throughput = 300
        if imgs_per_sec >= slo_throughput:
            f.write(
                f"**SLO Status:** ✅ PASS - Throughput {imgs_per_sec:.2f} imgs/s ≥ {slo_throughput} imgs/s\n"
            )
        else:
            f.write(
                f"**SLO Status:** ❌ FAIL - Throughput {imgs_per_sec:.2f} imgs/s < {slo_throughput} imgs/s\n"
            )

        f.write("\n## Notas y Observaciones\n\n")
        f.write(
            "- El benchmark incluye el tiempo de transferencia a GPU para simular el flujo real de datos.\n"
        )
        f.write("- Se realizó un warmup previo para estabilizar mediciones.\n")

    print(f"Reporte escrito en {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark del dataloader")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/flickr8k",
        help="Directorio donde están los datos",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Tamaño del batch")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=12,
        help="Número de workers para la carga paralela",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Factor de prefetch para cada worker",
    )
    parser.add_argument(
        "--pin_memory",
        type=bool,
        default=True,
        help="Si usar pin_memory para transferencia más rápida a GPU",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=200,
        help="Número de iteraciones para el benchmark",
    )
    parser.add_argument(
        "--num_warmup", type=int, default=10, help="Número de iteraciones de warmup"
    )

    args = parser.parse_args()
    main(args)
