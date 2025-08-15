import argparse
import os
import time

import mlflow
import numpy as np
import torch

from chimera.data.dataloader import create_dataloader
from chimera.models.image_encoder import ImageEncoder
from chimera.utils.mlflow_utils import get_git_commit_hash, log_reproducibility_passport


def _run_warmup_iterations(dataloader, model, device, num_warmup):
    """Run warmup iterations before benchmarking.

    Args:
        dataloader: DataLoader instance
        model: ImageEncoder instance
        device: Device to run on ('cuda' or 'cpu')
        num_warmup: Number of warmup iterations
    """
    print(f"Running {num_warmup} warmup iterations...")
    for i, batch in enumerate(dataloader):
        if i >= num_warmup:
            break

        images = batch["image"].to(device, non_blocking=True)
        with torch.no_grad():
            _ = model(images)


def _run_benchmark_iterations(dataloader, model, device, num_iterations):
    """Run the main benchmark iterations and measure time.

    Args:
        dataloader: DataLoader instance
        model: ImageEncoder instance
        device: Device to run on ('cuda' or 'cpu')
        num_iterations: Number of iterations to measure

    Returns:
        list: List of time measurements for each iteration
    """
    times = []
    print(f"\nRunning {num_iterations} benchmark iterations...")

    for i, batch in enumerate(dataloader):
        if i >= num_iterations:
            break

        images = batch["image"].to(device, non_blocking=True)

        # Synchronize before timing
        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():
            _ = model(images)

        # Synchronize after forward pass
        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        times.append(end_time - start_time)

        if (i + 1) % 20 == 0:
            print(f"Completed {i + 1}/{num_iterations} iterations")

    return times


def _calculate_metrics(times, batch_size, device):
    """Calculate benchmark metrics from time measurements.

    Args:
        times: List of time measurements
        batch_size: Batch size used in dataloader
        device: Device used for benchmarking

    Returns:
        dict: Dictionary with calculated metrics
    """
    times = np.array(times)
    avg_time = np.mean(times)
    median_time = np.median(times)
    p95_time = np.percentile(times, 95)

    total_imgs = batch_size * len(times)
    imgs_per_sec = total_imgs / np.sum(times)

    # Calculate peak memory usage
    peak_memory_mb = 0
    if device == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    # Create metrics dictionary
    metrics = {
        "batch_size": batch_size,
        "imgs_per_sec": imgs_per_sec,
        "avg_time_ms": avg_time * 1000,
        "median_time_ms": median_time * 1000,
        "p95_time_ms": p95_time * 1000,
        "peak_memory_mb": peak_memory_mb,
    }

    return metrics


def benchmark_encoder(dataloader, model, device, num_warmup=10, num_iterations=200):
    """
    Benchmark the image encoder on the given dataloader.

    Args:
        dataloader: DataLoader instance
        model: ImageEncoder instance
        device: Device to run on ('cuda' or 'cpu')
        num_warmup: Number of warmup iterations
        num_iterations: Number of iterations to measure

    Returns:
        dict: Dictionary with benchmark metrics
    """
    # Ensure model is in evaluation mode
    model.eval()

    # Record peak memory before starting
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Run warmup iterations
    _run_warmup_iterations(dataloader, model, device, num_warmup)

    # Synchronize before starting measurements
    if device == "cuda":
        torch.cuda.synchronize()

    # Run benchmark iterations
    times = _run_benchmark_iterations(dataloader, model, device, num_iterations)

    # Calculate and return metrics
    return _calculate_metrics(times, dataloader.batch_size, device)


def _generate_model_architecture(model, args):
    """
    Generate model architecture summary with proper error handling.

    Args:
        model: ImageEncoder instance
        args: Argument namespace with arch_path and image_size

    Returns:
        bool: True if summary was generated successfully, False otherwise
    """
    print("Generating model architecture summary...")

    # Ensure directory exists before attempting to write summary
    try:
        os.makedirs(os.path.dirname(args.arch_path) or ".", exist_ok=True)
    except OSError as e:
        print(f"Warning: Failed to create directory for architecture summary: {e}")
        return False

    try:
        # Generate the model summary using write_detailed_summary
        model.write_detailed_summary(
            save_path=args.arch_path,
            input_size=(3, args.image_size, args.image_size),
            batch_size=1,
        )

        # Rename the summary file to architecture_image_encoder.md
        summary_file = f"{args.arch_path}_summary.md"
        new_file = "architecture_image_encoder.md"

        if os.path.exists(summary_file):
            # Get the directory from arch_path
            directory = os.path.dirname(args.arch_path)
            new_path = os.path.join(directory, new_file) if directory else new_file

            # Rename the file
            os.rename(summary_file, new_path)
            print(f"✅ Model architecture summary saved: {new_path}")
            return True
        else:
            print("❌ Summary file was not created")
            return False

    except Exception as e:
        print(f"❌ Unexpected error during summary generation: {e}")
        return False


def _display_architecture(model, arch_path):
    """
    Display the generated architecture summary in the output.

    Args:
        model: ImageEncoder instance
        arch_path: Path to the architecture files

    Returns:
        bool: True if architecture summary exists and can be displayed
    """
    # Get directory path
    directory = os.path.dirname(arch_path)
    summary_path = (
        os.path.join(directory, "architecture_image_encoder.md")
        if directory
        else "architecture_image_encoder.md"
    )

    # Check if architecture summary exists
    if not os.path.exists(summary_path):
        return False

    print("\n" + "=" * 60)
    print("🏛️  MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)

    # Display markdown content if it exists
    try:
        with open(summary_path, "r") as f:
            md_content = f.read()
        print(md_content)
    except Exception as e:
        print(f"Error reading architecture summary: {e}")

    # Display additional model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n📋 Model Summary:")
    print("   - Backbone: ResNet18 (ImageNet pretrained)")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Frozen parameters: {total_params - trainable_params:,}")

    return True


def format_report(
    metrics,
    image_size,
    num_workers,
    device,
    output_path,
    gpu_util=None,
    cpu_util=None,
    arch_path=None,
):
    """
    Format a markdown report with benchmark results.

    Args:
        metrics: Dictionary with benchmark metrics
        image_size: Size of input images
        num_workers: Number of dataloader workers
        device: Device used for benchmarking
        output_path: Path where the report will be saved
        gpu_util: GPU utilization percentage (optional)
        cpu_util: CPU utilization percentage (optional)
        arch_path: Path to architecture files (optional)

    Returns:
        str: Markdown formatted report
    """
    sha = get_git_commit_hash()
    date = time.strftime("%Y-%m-%d %H:%M:%S")

    # Format the markdown report
    report = "# Image Encoder Benchmark Results\n\n"
    report += f"**Date**: {date}\n\n"
    report += f"**Git SHA**: `{sha}`\n\n"

    report += "## Configuration\n\n"
    report += f"- **Device**: {device}\n"
    report += f"- **Image size**: {image_size}x{image_size}\n"
    report += f"- **Batch size**: {metrics['batch_size']}\n"
    report += f"- **DataLoader workers**: {num_workers}\n"

    report += "\n## Performance Metrics\n\n"
    report += f"- **Throughput**: {metrics['imgs_per_sec']:.2f} images/sec\n"
    report += f"- **Latency (avg)**: {metrics['avg_time_ms']:.2f} ms/batch\n"
    report += f"- **Latency (median)**: {metrics['median_time_ms']:.2f} ms/batch\n"
    report += f"- **Latency (P95)**: {metrics['p95_time_ms']:.2f} ms/batch\n"
    report += f"- **VRAM peak**: {metrics['peak_memory_mb']:.2f} MB\n"

    if gpu_util is not None:
        report += f"- **GPU utilization**: {gpu_util:.2f}%\n"
    if cpu_util is not None:
        report += f"- **CPU utilization**: ~{cpu_util}%\n"

    report += "\n## Model Details\n\n"
    report += "- **Backbone**: ResNet18 pretrained on ImageNet\n"
    report += "- **Normalization**: ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n"
    report += "- **Training**: Frozen backbone, only projection head trainable\n"

    # Add model architecture section
    if arch_path:
        directory = os.path.dirname(arch_path)
        summary_path = (
            os.path.join(directory, "architecture_image_encoder.md")
            if directory
            else "architecture_image_encoder.md"
        )

        if os.path.exists(summary_path):
            report += "\n## Model Architecture\n\n"

            # Include additional architecture details from the summary file
            try:
                with open(summary_path, "r") as f:
                    summary_content = f.read()

                # Add the summary content to the report
                report += summary_content + "\n"

                # Link to full architecture documentation
                summary_name = os.path.basename(summary_path)
                report += f"\n[View detailed architecture summary]({summary_name})\n"
            except Exception:
                report += "*Error reading architecture summary file.*\n"
        else:
            report += "\n## Model Architecture\n\n"
            report += "*Architecture summary not available. Run with `--visualize-arch` to generate.*\n"
    else:
        report += "\n## Model Architecture\n\n"
        report += "*Architecture summary not generated. Use `--visualize-arch` flag to include summary.*\n"

    return report


def save_report(report, output_path):
    """
    Save the benchmark report to a file.

    Args:
        report: Markdown formatted report string
        output_path: Path to save the report
    """
    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"\n📄 Benchmark report saved to: {output_path}")
    except OSError as e:
        print(f"❌ Failed to save report: {e}")
        # Still print report to console as fallback
        print("\n" + "=" * 50)
        print("BENCHMARK REPORT (console output):")
        print("=" * 50)
        print(report)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark the image encoder")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for benchmark (default: 64)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image size for benchmark (default: 224)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of dataloader workers (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run benchmark on (default: cuda)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of benchmark iterations (default: 200)",
    )
    parser.add_argument(
        "--parquet-path",
        type=str,
        default="data/flickr8k/processed/flickr8k_small.parquet",
        help="Path to parquet dataset",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="reports/bench_image_encoder.md",
        help="Path to save benchmark report",
    )
    parser.add_argument(
        "--gpu-util",
        type=float,
        default=None,
        help="GPU utilization from nvidia-smi dmon (%%)",
    )
    parser.add_argument(
        "--cpu-util", type=str, default=None, help="Approximate CPU utilization (%%)"
    )
    parser.add_argument(
        "--visualize-arch",
        action="store_true",
        help="Generate and display a summary of the model architecture",
    )
    parser.add_argument(
        "--arch-path",
        type=str,
        default="reports/model_architecture",
        help="Path base for saving the architecture summary (default: reports/model_architecture)",
    )
    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="Log benchmark results and reproducibility passport to MLflow",
    )
    return parser.parse_args()


def build_transform(image_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def create_dataloader_from_args(args, transform):
    print(f"📁 Loading dataset from {args.parquet_path}")
    return create_dataloader(
        parquet_path=args.parquet_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
        transform=transform,
        prefetch_factor=2,
    )


def create_model(device: str):
    print("🏗️  Creating ImageEncoder model")
    model = ImageEncoder(embed_dim=256, freeze_backbone=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    return model


def maybe_generate_and_display_architecture(model, args) -> bool:
    if not args.visualize_arch:
        return False
    arch_generated = _generate_model_architecture(model, args)
    if arch_generated:
        _display_architecture(model, args.arch_path)
    return arch_generated


def run_benchmark_and_time(dataloader, model, args):
    print(f"\n⏱️  Running benchmark on {args.device}...")
    print("-" * 30)
    start_total = time.time()
    metrics = benchmark_encoder(
        dataloader=dataloader,
        model=model,
        device=args.device,
        num_warmup=args.warmup,
        num_iterations=args.iterations,
    )
    total_time = time.time() - start_total
    return metrics, total_time


def print_metrics(metrics: dict, total_time: float):
    print("\n📊 Benchmark Results:")
    print("=" * 30)
    print(f"Throughput: {metrics['imgs_per_sec']:.2f} images/sec")
    print(f"Latency (avg): {metrics['avg_time_ms']:.2f} ms/batch")
    print(f"Latency (median): {metrics['median_time_ms']:.2f} ms/batch")
    print(f"Latency (P95): {metrics['p95_time_ms']:.2f} ms/batch")
    print(f"VRAM peak: {metrics['peak_memory_mb']:.2f} MB")
    print(f"Total benchmark time: {total_time:.1f}s")


def log_to_mlflow(args, metrics, total_time, arch_generated):
    print("\n📊 Logging results to MLflow...")
    mlflow.set_experiment("Image Encoder Benchmark")
    with mlflow.start_run(
        run_name=f"image_encoder_benchmark_{time.strftime('%Y%m%d_%H%M%S')}"
    ):
        # Params
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("image_size", args.image_size)
        mlflow.log_param("num_workers", args.num_workers)
        mlflow.log_param("device", args.device)
        mlflow.log_param("warmup_iterations", args.warmup)
        mlflow.log_param("benchmark_iterations", args.iterations)
        # Metrics
        mlflow.log_metric("throughput_imgs_per_sec", metrics["imgs_per_sec"])
        mlflow.log_metric("latency_avg_ms", metrics["avg_time_ms"])
        mlflow.log_metric("latency_median_ms", metrics["median_time_ms"])
        mlflow.log_metric("latency_p95_ms", metrics["p95_time_ms"])
        mlflow.log_metric("peak_memory_mb", metrics["peak_memory_mb"])
        mlflow.log_metric("total_benchmark_time_s", total_time)
        if args.gpu_util is not None:
            mlflow.log_metric("gpu_utilization_percent", args.gpu_util)
        # Artifacts
        if os.path.exists(args.report_path):
            mlflow.log_artifact(args.report_path, "reports")
        if arch_generated:
            directory = os.path.dirname(args.arch_path)
            summary_path = (
                os.path.join(directory, "architecture_image_encoder.md")
                if directory
                else "architecture_image_encoder.md"
            )
            if os.path.exists(summary_path):
                mlflow.log_artifact(summary_path, "architecture")
        # Reproducibility passport
        log_reproducibility_passport()
        run_id = mlflow.active_run().info.run_id
        print(f"   ✅ MLflow logging completed (Run ID: {run_id})")
    return run_id


def print_final_summary(arch_generated: bool, args, run_id: str | None):
    print("\n✅ Benchmark completed successfully!")
    if arch_generated:
        print("   🏛️  Architecture summary generated and displayed")
        directory = os.path.dirname(args.arch_path)
        summary_path = (
            os.path.join(directory, "architecture_image_encoder.md")
            if directory
            else "architecture_image_encoder.md"
        )
        print(f"   📋 View summary: {summary_path}")
    elif args.visualize_arch:
        print("   ⚠️  Architecture summary generation failed (see warnings above)")
        print("   💡 Try: pip install torchinfo")
    if args.log_mlflow and run_id is not None:
        print(f"   📈 Results logged to MLflow (Run ID: {run_id})")


def main():
    print("🚀 Starting ImageEncoder Benchmark")
    print("=" * 50)

    args = parse_args()
    transform = build_transform(args.image_size)
    dataloader = create_dataloader_from_args(args, transform)
    model = create_model(args.device)

    arch_generated = maybe_generate_and_display_architecture(model, args)

    metrics, total_time = run_benchmark_and_time(dataloader, model, args)
    print_metrics(metrics, total_time)

    print("\n📝 Generating report...")
    report = format_report(
        metrics=metrics,
        image_size=args.image_size,
        num_workers=dataloader.num_workers,
        device=args.device,
        output_path=args.report_path,
        gpu_util=args.gpu_util,
        cpu_util=args.cpu_util,
        arch_path=args.arch_path if args.visualize_arch else None,
    )
    save_report(report, args.report_path)

    run_id = None
    if args.log_mlflow:
        run_id = log_to_mlflow(args, metrics, total_time, arch_generated)

    print_final_summary(arch_generated, args, run_id)


if __name__ == "__main__":
    # Enable cuDNN benchmark mode for optimized performance with fixed-size inputs
    torch.backends.cudnn.benchmark = True
    main()
