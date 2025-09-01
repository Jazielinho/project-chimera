#!/usr/bin/env python
import argparse
import os
import subprocess


def parse_args():
    """Parse command line arguments for training configuration"""
    parser = argparse.ArgumentParser(
        description="Run training for one epoch with optimized parameters"
    )

    # Basic configuration
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size per GPU (default: 128)"
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps (default: 2)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/flickr8k/processed/flickr8k_full.parquet",
        help="Path to dataset Parquet file",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--num_workers", type=int, default=12, help="Number of dataloader workers"
    )

    # Model parameters
    parser.add_argument(
        "--embed_dim", type=int, default=256, help="Embedding dimension (default: 256)"
    )
    parser.add_argument(
        "--freeze_encoders",
        action="store_true",
        default=True,
        help="Freeze encoder backbones, train only projections",
    )

    return parser.parse_args()


def get_gpu_info():
    """Get information about available GPUs"""
    try:
        import torch

        if not torch.cuda.is_available():
            return "No GPU available", 0

        device_name = torch.cuda.get_device_name(0)
        device_memory = torch.cuda.get_device_properties(0).total_memory / (
            1024**3
        )  # GB
        return device_name, device_memory
    except Exception as e:
        return f"Error getting GPU info: {e}", 0


def get_optimal_batch_size(gpu_memory):
    """Suggest optimal batch size based on GPU memory"""
    # Conservative estimate: 5GB baseline + ~50MB per sample with 224px images
    if gpu_memory <= 0:
        return 64  # Default for unknown memory

    # Memory-based batch size suggestions
    if gpu_memory < 8:
        return 64
    elif gpu_memory < 12:
        return 96
    elif gpu_memory < 16:
        return 128
    elif gpu_memory < 24:
        return 192
    else:
        return 256


def main():
    """Configure and run the training script with optimal parameters"""
    args = parse_args()

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Get GPU information
    gpu_name, gpu_memory = get_gpu_info()
    print(f"\nDetected GPU: {gpu_name} with {gpu_memory:.1f} GB memory")

    # Adjust batch size if necessary
    optimal_batch = get_optimal_batch_size(gpu_memory)
    if args.batch_size > optimal_batch:
        print(
            f"Warning: Requested batch size {args.batch_size} may be too large for this GPU"
        )
        print(f"Recommending batch size of {optimal_batch} based on GPU memory")
        use_optimal = input(
            f"Use recommended batch size {optimal_batch} instead? [Y/n]: "
        )
        if use_optimal.lower() != "n":
            args.batch_size = optimal_batch

    # Calculate effective batch size
    effective_batch = args.batch_size * args.grad_accum_steps
    print("\nTraining with:")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Gradient accumulation steps: {args.grad_accum_steps}")
    print(f"  - Effective batch size: {effective_batch}")
    print(f"  - Embedding dimension: {args.embed_dim}")
    print(f"  - {'Frozen' if args.freeze_encoders else 'Trainable'} encoder backbones")
    print(f"  - Dataloader workers: {args.num_workers}")

    # Construct command with all parameters
    cmd = [
        "python",
        "-m",
        "chimera.train.train",
        "--batch_size",
        str(args.batch_size),
        "--grad_accum_steps",
        str(args.grad_accum_steps),
        "--data_path",
        args.data_path,
        "--checkpoint_dir",
        args.checkpoint_dir,
        "--num_workers",
        str(args.num_workers),
        "--embed_dim",
        str(args.embed_dim),
        "--log_interval",
        "50",
        "--learning_rate",
        "3e-4",
        "--warmup_ratio",
        "0.05",
        "--prefetch_factor",
        "4",
        "--temperature",
        "0.07",
        "--learnable_temperature",
    ]

    # Add conditional arguments
    if args.freeze_encoders:
        cmd.extend(["--freeze_image_encoder", "--freeze_text_encoder"])

    # Print the command
    print("\nExecuting command:")
    print(" ".join(cmd))
    print("\nStarting training...\n")

    # Run the command
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
