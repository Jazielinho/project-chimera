import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from chimera.data.dataloader import create_dataloader
from chimera.losses.contrastive import ContrastiveLoss
from chimera.models.image_encoder import ImageEncoder
from chimera.models.text_encoder import TextEncoder
from chimera.utils.mlflow_utils import log_reproducibility_passport


def setup_reproducibility(seed: int = 1337) -> None:
    """
    Configure deterministic behavior for reproducibility.

    Args:
        seed: Random seed to use for all random number generators.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Set True in production for speed


def get_gpu_utilization() -> float:
    """
    Get the current GPU utilization percentage.

    Returns:
        float: GPU utilization as a percentage between 0-100
    """
    try:
        import subprocess

        result = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ]
            )
            .decode("utf-8")
            .strip()
        )
        return float(result)
    except Exception:
        return 0.0


def train_step(
    batch: Dict,
    image_encoder: nn.Module,
    text_encoder: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    args: argparse.Namespace,
    device: torch.device,
    step: int,
    grad_norms: List[float],
) -> Tuple[float, Dict, List[float]]:
    """
    Execute a single training step with gradient accumulation.

    Args:
        batch: Dictionary containing image and caption data
        image_encoder: Model for encoding images
        text_encoder: Model for encoding text
        loss_fn: Contrastive loss function
        optimizer: Optimizer for parameter updates
        scheduler: Learning rate scheduler
        args: Command line arguments
        device: Device to run computation on
        step: Current step number
        grad_norms: List to track gradient norms

    Returns:
        Tuple of loss value, loss dictionary, and updated grad_norms list
    """
    # Start timer for step
    start_time = time.time()

    # Get data from batch
    images = batch["image"].to(device, non_blocking=True)
    captions = batch["caption"]

    # Forward pass
    img_f = image_encoder(images)
    txt_f = text_encoder(captions)

    # Calculate loss
    loss_dict = loss_fn(img_f, txt_f)
    loss = loss_dict["loss"] / args.grad_accum_steps

    # Backward pass
    loss.backward()

    # Get actual loss value (before scaling)
    actual_loss = (loss * args.grad_accum_steps).item()

    # Step optimizer if we've accumulated enough gradients
    if (step + 1) % args.grad_accum_steps == 0 or step == args.steps_per_epoch - 1:
        # Clip gradients if specified
        if args.grad_clip_norm > 0:
            trainable_params = (
                list(image_encoder.projection.parameters())
                + list(text_encoder.proj.parameters())
                + list(loss_fn.parameters())
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params, args.grad_clip_norm
            )
            grad_norms.append(
                float(grad_norm.item())
                if hasattr(grad_norm, "item")
                else float(grad_norm)
            )

        # Step optimizer and scheduler
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    # Calculate step time
    step_time = (time.time() - start_time) * 1000  # ms
    loss_dict["step_ms"] = step_time

    # Calculate throughput (images per second)
    pairs_per_second = (args.batch_size / step_time) * 1000
    loss_dict["pairs_per_sec"] = pairs_per_second

    return actual_loss, loss_dict, grad_norms


def log_metrics(
    epoch: int,
    step: int,
    global_step: int,
    loss_dict: Dict,
    lr: float,
    grad_norm: Optional[float] = None,
) -> None:
    """
    Log metrics to MLflow at the current step.

    Args:
        epoch: Current epoch number
        step: Current step within the epoch
        global_step: Global step count across epochs
        loss_dict: Dictionary of loss values and related metrics
        lr: Current learning rate
        grad_norm: Gradient norm after clipping (optional)
    """
    # Log loss components
    mlflow.log_metric("loss", loss_dict["loss"], step=global_step)
    mlflow.log_metric("loss_img2txt", loss_dict["loss_img2txt"], step=global_step)
    mlflow.log_metric("loss_txt2img", loss_dict["loss_txt2img"], step=global_step)

    # Log contrastive details
    mlflow.log_metric("logit_scale", loss_dict["logit_scale_exp"], step=global_step)
    mlflow.log_metric(
        "temperature", loss_dict["effective_temperature"], step=global_step
    )
    mlflow.log_metric(
        "pos_similarity", loss_dict["positive_similarity"], step=global_step
    )
    mlflow.log_metric(
        "neg_similarity", loss_dict["negative_similarity"], step=global_step
    )

    # Log performance metrics
    mlflow.log_metric("step_ms", loss_dict["step_ms"], step=global_step)
    mlflow.log_metric("pairs_per_sec", loss_dict["pairs_per_sec"], step=global_step)
    mlflow.log_metric("gpu_util", get_gpu_utilization(), step=global_step)
    mlflow.log_metric("lr", lr, step=global_step)

    # Log accuracy
    mlflow.log_metric("accuracy", loss_dict["accuracy"], step=global_step)

    # Log gradient norm if available
    if grad_norm is not None:
        mlflow.log_metric("grad_norm", grad_norm, step=global_step)


def main(args: argparse.Namespace) -> None:
    """
    Main training function that runs for one epoch.

    Args:
        args: Command line arguments
    """
    # Setup reproducibility
    setup_reproducibility(args.seed)

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Reset CUDA memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Create directory for checkpoints
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Configurar experimento MLflow
    experiment_name = "chimera-training"
    mlflow.set_experiment(experiment_name)

    # Start MLflow run
    with mlflow.start_run(run_name=f"train_epoch_{args.epoch}"):
        # Log reproducibility passport
        log_reproducibility_passport()

        # Log parameters
        mlflow.log_params(vars(args))

        # Create models
        print("Creating models...")
        image_encoder = ImageEncoder(
            embed_dim=args.embed_dim, freeze_backbone=args.freeze_image_encoder
        ).to(device)

        text_encoder = TextEncoder(
            embed_dim=args.embed_dim,
            freeze_backbone=args.freeze_text_encoder,
            max_length=args.max_text_length,
        ).to(device)

        # Create loss function
        loss_fn = ContrastiveLoss(
            temperature=args.temperature,
            learnable_temperature=args.learnable_temperature,
            label_smoothing=args.label_smoothing,
            assert_normalized=True,
        ).to(device)

        # Set up optimizer with specific trainable parameters
        trainable_params = []

        # Only add projections and loss parameters if encoders are frozen
        if args.freeze_image_encoder:
            trainable_params.extend(list(image_encoder.projection.parameters()))
        else:
            trainable_params.extend(list(image_encoder.parameters()))

        if args.freeze_text_encoder:
            trainable_params.extend(list(text_encoder.proj.parameters()))
        else:
            trainable_params.extend(list(text_encoder.parameters()))

        trainable_params.extend(list(loss_fn.parameters()))

        # Print number of trainable parameters
        num_trainable = sum(p.numel() for p in trainable_params if p.requires_grad)
        print(f"Number of trainable parameters: {num_trainable:,}")

        # Create optimizer
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            eps=args.eps,
        )

        # Create dataloader
        print("Creating dataloader...")
        dataloader = create_dataloader(
            parquet_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=args.prefetch_factor,
            shuffle=True,
            transform=None,  # Use default transforms
            max_text_length=args.max_text_length,
            deterministic=True,
            monitor_speed=True,
        )

        # Calculate steps per epoch and total training steps
        args.steps_per_epoch = len(dataloader)
        total_steps = args.steps_per_epoch // args.grad_accum_steps

        # Create scheduler with warmup
        warmup_steps = int(total_steps * args.warmup_ratio)
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        # Log training setup
        print(f"Total steps in epoch: {args.steps_per_epoch}")
        print(f"Effective batch size: {args.batch_size * args.grad_accum_steps}")
        print(f"Warmup steps: {warmup_steps}")

        # Initialize tracking variables
        running_loss = 0.0
        loss_history = []
        grad_norms = []
        epoch_start_time = time.time()
        global_step = 0

        # Set models to training mode
        image_encoder.train()
        text_encoder.train()
        loss_fn.train()

        # Main training loop
        print(f"Starting epoch {args.epoch}...")
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {args.epoch}")):
            # Train step
            loss, loss_dict, grad_norms = train_step(
                batch=batch,
                image_encoder=image_encoder,
                text_encoder=text_encoder,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                device=device,
                step=step,
                grad_norms=grad_norms,
            )

            # Update running loss and history
            running_loss += loss
            loss_history.append(loss)

            # Log every log_interval steps
            if step % args.log_interval == 0 or step == args.steps_per_epoch - 1:
                # Calculate current learning rate
                current_lr = (
                    scheduler.get_last_lr()[0] if scheduler else args.learning_rate
                )

                # Calculate current gradient norm (if available)
                current_grad_norm = grad_norms[-1] if grad_norms else None

                # Log metrics
                log_metrics(
                    epoch=args.epoch,
                    step=step,
                    global_step=global_step,
                    loss_dict=loss_dict,
                    lr=current_lr,
                    grad_norm=current_grad_norm,
                )

                # Print progress
                avg_loss = running_loss / (step + 1)
                print(
                    f"Step: {step}/{args.steps_per_epoch}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}"
                )

            # Save checkpoint at specified intervals
            if (
                step + 1
            ) % args.checkpoint_interval == 0 or step == args.steps_per_epoch - 1:
                checkpoint_path = (
                    Path(args.checkpoint_dir)
                    / f"checkpoint_epoch{args.epoch}_step{step}.pt"
                )
                torch.save(
                    {
                        "epoch": args.epoch,
                        "step": step,
                        "image_encoder_state_dict": image_encoder.state_dict(),
                        "text_encoder_state_dict": text_encoder.state_dict(),
                        "loss_fn_state_dict": loss_fn.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": (
                            scheduler.state_dict() if scheduler else None
                        ),
                        "args": args,
                    },
                    checkpoint_path,
                )
                print(f"Saved checkpoint to {checkpoint_path}")

            # Update global step
            global_step += 1

        # End of epoch processing
        epoch_time = time.time() - epoch_start_time
        hours, remainder = divmod(epoch_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Get peak memory usage
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            mlflow.log_metric("peak_memory_mb", peak_memory_mb)
            print(f"Peak GPU memory usage: {peak_memory_mb:.2f} MB")

        # Log final metrics
        final_loss = np.mean(loss_history[-10:]) if loss_history else 0
        mlflow.log_metric("epoch_time_seconds", epoch_time)
        mlflow.log_metric("final_loss", final_loss)

        if grad_norms:
            mlflow.log_metric("mean_grad_norm", np.mean(grad_norms))
            mlflow.log_metric("max_grad_norm", np.max(grad_norms))

        # Save final model
        final_checkpoint_path = (
            Path(args.checkpoint_dir) / f"checkpoint_epoch{args.epoch}_final.pt"
        )
        torch.save(
            {
                "epoch": args.epoch,
                "image_encoder_state_dict": image_encoder.state_dict(),
                "text_encoder_state_dict": text_encoder.state_dict(),
                "loss_fn_state_dict": loss_fn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "args": args,
            },
            final_checkpoint_path,
        )

        # Print final summary
        print(
            f"Epoch {args.epoch} completed in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        )
        print(f"Final loss: {final_loss:.4f}")
        print(f"Final checkpoint saved to {final_checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a mini-CLIP model for one epoch"
    )

    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/flickr8k/processed/flickr8k_full.parquet",
        help="Path to the Parquet file containing the training data",
    )
    parser.add_argument(
        "--max_text_length", type=int, default=32, help="Maximum length of text inputs"
    )

    # Model arguments
    parser.add_argument(
        "--embed_dim", type=int, default=256, help="Dimension of the embeddings"
    )
    parser.add_argument(
        "--freeze_image_encoder",
        action="store_true",
        help="Freeze the image encoder backbone",
    )
    parser.add_argument(
        "--freeze_text_encoder",
        action="store_true",
        help="Freeze the text encoder backbone",
    )

    # Training arguments
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=2,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--grad_clip_norm", type=float, default=1.0, help="Gradient clipping norm"
    )
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.98, help="Adam beta2")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--eps", type=float, default=1e-6, help="Adam epsilon")
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Ratio of warmup steps to total steps",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature parameter for contrastive loss",
    )
    parser.add_argument(
        "--learnable_temperature",
        action="store_true",
        help="Make temperature a learnable parameter",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for the contrastive loss",
    )

    # System arguments
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--prefetch_factor", type=int, default=4, help="Prefetch factor for dataloader"
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument(
        "--log_interval", type=int, default=50, help="Interval for logging metrics"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1000,
        help="Interval for saving checkpoints",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints",
    )
    parser.add_argument("--epoch", type=int, default=1, help="Current epoch number")

    # Parse arguments
    args = parser.parse_args()

    # Run training
    main(args)
