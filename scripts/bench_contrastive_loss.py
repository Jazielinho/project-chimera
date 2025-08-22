import argparse
import os
import time

import mlflow
import numpy as np
import torch

from chimera.losses.contrastive import (
    ContrastiveLoss,  # Ajusta el import según tu proyecto
)
from chimera.utils.mlflow_utils import get_git_commit_hash, log_reproducibility_passport


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark de la función de pérdida contrastiva"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Tamaño de batch (default: 256)"
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=256,
        help="Dimensión de los embeddings (default: 256)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Dispositivo (default: cuda)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Iteraciones de benchmark (default: 200)",
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Iteraciones de warmup (default: 10)"
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="reports/bench_contrastive_loss.md",
        help="Ruta del reporte",
    )
    parser.add_argument(
        "--log-mlflow", action="store_true", help="Loguear resultados en MLflow"
    )
    return parser.parse_args()


def generate_fake_embeddings(batch_size, embed_dim, device):
    # Simula dos vistas de embeddings normalizados
    z1 = torch.randn(batch_size, embed_dim, device=device)
    z2 = torch.randn(batch_size, embed_dim, device=device)
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)
    return z1, z2


def run_warmup(loss_fn, batch_size, embed_dim, device, num_warmup):
    print(f"🔥 Warmup ({num_warmup} iteraciones)...")
    for _ in range(num_warmup):
        z1, z2 = generate_fake_embeddings(batch_size, embed_dim, device)
        with torch.no_grad():
            _ = loss_fn(z1, z2)


def run_benchmark(loss_fn, batch_size, embed_dim, device, num_iterations):
    print(f"⏱️  Benchmark ({num_iterations} iteraciones)...")
    times = []
    for i in range(num_iterations):
        z1, z2 = generate_fake_embeddings(batch_size, embed_dim, device)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = loss_fn(z1, z2)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{num_iterations} iteraciones completadas")
    return times


def calculate_metrics(times, batch_size):
    times = np.array(times)
    avg_time = np.mean(times)
    median_time = np.median(times)
    p95_time = np.percentile(times, 95)
    samples_per_sec = batch_size / avg_time
    return {
        "batch_size": batch_size,
        "avg_time_ms": avg_time * 1000,
        "median_time_ms": median_time * 1000,
        "p95_time_ms": p95_time * 1000,
        "samples_per_sec": samples_per_sec,
    }


def format_report(metrics, embed_dim, device, output_path):
    sha = get_git_commit_hash()
    date = time.strftime("%Y-%m-%d %H:%M:%S")
    report = "# Benchmark: Contrastive Loss\n\n"
    report += f"**Fecha**: {date}\n\n"
    report += f"**Commit SHA**: {sha}\n\n"
    report += "## Configuración\n"
    report += f"- **Dispositivo**: {device}\n"
    report += f"- **Batch size**: {metrics['batch_size']}\n"
    report += f"- **Embedding dim**: {embed_dim}\n"
    report += "\n## Métricas de rendimiento\n"
    report += f"- **Throughput**: {metrics['samples_per_sec']:.2f} muestras/seg\n"
    report += f"- **Latencia (media)**: {metrics['avg_time_ms']:.2f} ms/batch\n"
    report += f"- **Latencia (mediana)**: {metrics['median_time_ms']:.2f} ms/batch\n"
    report += f"- **Latencia (P95)**: {metrics['p95_time_ms']:.2f} ms/batch\n"
    return report


def save_report(report, output_path):
    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"\n📄 Reporte guardado en: {output_path}")
    except Exception as e:
        print(f"❌ Error guardando el reporte: {e}")
        print(report)


def log_to_mlflow(args, metrics, total_time):
    print("\n📊 Logueando resultados en MLflow...")
    mlflow.set_experiment("Contrastive Loss Benchmark")
    with mlflow.start_run(
        run_name=f"contrastive_loss_bench_{time.strftime('%Y%m%d_%H%M%S')}"
    ):
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("embed_dim", args.embed_dim)
        mlflow.log_param("device", args.device)
        mlflow.log_param("iterations", args.iterations)
        mlflow.log_metric("throughput_samples_per_sec", metrics["samples_per_sec"])
        mlflow.log_metric("latency_avg_ms", metrics["avg_time_ms"])
        mlflow.log_metric("latency_median_ms", metrics["median_time_ms"])
        mlflow.log_metric("latency_p95_ms", metrics["p95_time_ms"])
        mlflow.log_metric("total_benchmark_time_s", total_time)
        if os.path.exists(args.report_path):
            mlflow.log_artifact(args.report_path, "reports")
        log_reproducibility_passport()
        run_id = mlflow.active_run().info.run_id
        print(f"   ✅ Logging completado (Run ID: {run_id})")
    return run_id


def main():
    print("🚀 Benchmark: Contrastive Loss")
    print("=" * 50)
    args = parse_args()
    device = (
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    loss_fn = ContrastiveLoss().to(device)  # Ajusta si tu loss requiere argumentos

    run_warmup(loss_fn, args.batch_size, args.embed_dim, device, args.warmup)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    times = run_benchmark(
        loss_fn, args.batch_size, args.embed_dim, device, args.iterations
    )
    total_time = time.time() - start
    metrics = calculate_metrics(times, args.batch_size)

    print("\n📊 Resultados:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n📝 Generando reporte...")
    report = format_report(metrics, args.embed_dim, device, args.report_path)
    save_report(report, args.report_path)

    run_id = None
    if args.log_mlflow:
        run_id = log_to_mlflow(args, metrics, total_time)

    print("\n✅ Benchmark finalizado.")
    if args.log_mlflow and run_id:
        print(f"   📈 Resultados en MLflow (Run ID: {run_id})")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
