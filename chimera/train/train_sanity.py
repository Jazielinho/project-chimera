#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script mejorado de entrenamiento para prueba de vida (sanity check).

Este script ejecuta un ciclo de entrenamiento corto (200 steps) con modelos
más realistas para verificar que el flujo completo funciona correctamente.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import List, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import ResNet18_Weights, resnet18
from transformers import AutoModel, AutoTokenizer

from chimera.data.dataloader import create_dataloader
from chimera.losses.contrastive import ContrastiveLoss
from chimera.utils.mlflow_utils import log_reproducibility_passport


class ImageEncoderResNet18(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for p in self.backbone.parameters():
            p.requires_grad = False

        feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.proj = nn.Sequential(
            nn.Linear(feat, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        z = self.backbone(x)  # [B, feat]
        z = nn.functional.normalize(self.proj(z), dim=-1)
        return z


class TextEncoderMiniLM(nn.Module):
    def __init__(self, embed_dim=256, max_len=32):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        for p in self.model.parameters():
            p.requires_grad = False

        hid = self.model.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hid, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.max_len = max_len

    def forward(self, captions):
        tok = self.tok(
            list(captions),
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        ).to(self.model.device)
        out = self.model(**tok).last_hidden_state
        mask = tok["attention_mask"].unsqueeze(-1)
        emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-6)  # mean pooling
        emb = nn.functional.normalize(self.proj(emb), dim=-1)
        return emb


def setup_reproducibility(seed):
    """
    Configura las semillas para garantizar la reproducibilidad.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def start_dmon(log_path: str = "dmon.log"):
    try:
        p = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "pucm", "-d", "1", "-o", "TD", "-f", log_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("[INFO] Monitor GPU (dmon) iniciado.")
        return p, log_path
    except Exception as e:
        print(f"[WARN] No se pudo iniciar dmon: {e}")
        return None, None


def stop_dmon(proc, log_path: str | None) -> float:
    gpu_util_avg = 0.0
    if not proc or not log_path or not Path(log_path).exists():
        return gpu_util_avg
    time.sleep(1.5)
    try:
        proc.terminate()
    except Exception:
        pass

    try:
        lines = Path(log_path).read_text().splitlines()
        if not lines:
            return gpu_util_avg
        header = lines[0].split()
        # Busca columna 'sm' (streaming multiprocessor util)
        try:
            sm_idx = header.index("sm")
        except ValueError:
            # fallback: última columna numérica
            sm_idx = len(header) - 1

        vals = []
        for ln in lines[1:]:
            parts = ln.split()
            if len(parts) <= sm_idx:
                continue
            try:
                vals.append(float(parts[sm_idx]))
            except Exception:
                pass
        if vals:
            gpu_util_avg = sum(vals) / len(vals)
    except Exception as e:
        print(f"[WARN] Error leyendo {log_path}: {e}")
    return gpu_util_avg


def write_sanity_report(
    args,
    first_10_avg: float,
    last_10_avg: float,
    loss_drop: float,
    total_time_min: float,
    gpu_util_avg: float,
    max_mem_gb: float,
):
    p = Path("reports")
    p.mkdir(exist_ok=True)
    with open(p / "sanity_check.md", "w") as f:
        f.write("# Sanity Check (200 steps)\n\n")
        f.write("## Config\n\n")
        f.write(f"- batch_size: {args.batch_size}\n")
        f.write(f"- grad_accum_steps: {args.grad_accum_steps}\n")
        f.write(f"- num_workers: {args.num_workers}\n")
        f.write(f"- lr: {args.learning_rate}\n")
        f.write(f"- embed_dim: {args.embed_dim}\n")
        f.write(f"- seed: {args.seed}\n\n")
        f.write("## Resultados\n\n")
        f.write(f"- loss[0] (avg 10): {first_10_avg:.4f}\n")
        f.write(f"- loss[200] (avg 10): {last_10_avg:.4f}\n")
        f.write(f"- drop: {loss_drop:.4f}\n")
        f.write(f"- wall-clock: {total_time_min:.2f} min\n")
        f.write(f"- GPU util (dmon): {gpu_util_avg:.2f}%\n")
        f.write(f"- Max VRAM: {max_mem_gb:.2f} GB\n\n")
        ok_loss = loss_drop >= args.min_loss_drop
        ok_time = total_time_min <= 25
        ok_gpu = gpu_util_avg >= args.min_gpu_util
        f.write("## SLOs\n\n")
        f.write(
            f"- Loss drop ≥ {args.min_loss_drop}: {'✅ PASS' if ok_loss else '❌ FAIL'}\n"
        )
        f.write(f"- Time ≤ 25 min: {'✅ PASS' if ok_time else '❌ FAIL'}\n")
        f.write(
            f"- GPU util ≥ {args.min_gpu_util}%: {'✅ PASS' if ok_gpu else '❌ FAIL'}\n"
        )


def train_step(
    batch,
    image_encoder: nn.Module,
    text_encoder: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    args,
    device,
    step: int,
    grad_norms: List[float],
) -> Tuple[float, dict, List[float]]:
    images = batch["image"].to(device, non_blocking=True)
    captions = batch["caption"]

    img_f = image_encoder(images)
    txt_f = text_encoder(captions)

    loss_dict = loss_fn(
        img_f, txt_f
    )  # {"loss", "avg_similarity", "positive_similarity", "logit_scale"}
    loss = loss_dict["loss"] / args.grad_accum_steps
    loss.backward()

    actual_loss = (loss * args.grad_accum_steps).item()

    if (step + 1) % args.grad_accum_steps == 0 or step == args.num_steps - 1:
        if args.grad_clip_norm > 0:
            g = torch.nn.utils.clip_grad_norm_(
                list(image_encoder.parameters())
                + list(text_encoder.parameters())
                + list(loss_fn.parameters()),
                args.grad_clip_norm,
            )
            grad_norms.append(float(g.item()) if hasattr(g, "item") else float(g))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    return actual_loss, loss_dict, grad_norms


def log_metrics(
    step: int, actual_loss: float, loss_dict: dict, grad_norms: List[float]
):
    if step % 10 == 0:
        mlflow.log_metric("loss", actual_loss, step=step)
        mlflow.log_metric("avg_similarity", loss_dict["avg_similarity"], step=step)
        mlflow.log_metric(
            "positive_similarity", loss_dict["positive_similarity"], step=step
        )
        mlflow.log_metric("logit_scale", loss_dict["logit_scale"], step=step)
        if grad_norms:
            mlflow.log_metric("grad_norm", grad_norms[-1], step=step)


def main(args):
    setup_reproducibility(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # acelera convs con tamaños fijos

    print(f"[INFO] device={device}")

    mlflow.set_experiment("Sanity")
    with mlflow.start_run(run_name="Sanity_200steps"):
        log_reproducibility_passport()

        # Hparams
        for k, v in {
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch": args.batch_size * args.grad_accum_steps,
            "num_workers": args.num_workers,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "num_steps": args.num_steps,
            "grad_clip_norm": args.grad_clip_norm,
            "embed_dim": args.embed_dim,
            "max_logit_scale": args.max_logit_scale,
        }.items():
            mlflow.log_param(k, v)

        # Modelos y pérdida (encoders congelados; solo proyecciones se entrenan)
        img_enc = ImageEncoderResNet18(embed_dim=args.embed_dim).to(device)
        txt_enc = TextEncoderMiniLM(embed_dim=args.embed_dim).to(device)
        loss_fn = ContrastiveLoss(max_logit_scale=args.max_logit_scale).to(device)

        # ⬇️ Asegura que TODO está realmente en CUDA
        assert next(img_enc.parameters()).is_cuda, "ImageEncoder no está en CUDA"
        # TextEncoder puede tener params congelados, pero igualmente verifica:
        assert next(txt_enc.parameters()).is_cuda, "TextEncoder no está en CUDA"
        assert next(loss_fn.parameters()).is_cuda, "Loss no está en CUDA"

        trainable = (
            list(img_enc.proj.parameters())
            + list(txt_enc.proj.parameters())
            + list(loss_fn.parameters())
        )
        optimizer = torch.optim.AdamW(
            trainable, lr=args.learning_rate, weight_decay=1e-2
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_steps)

        data_path = os.path.join(args.data_dir, "processed", "flickr8k_small.parquet")
        dl = create_dataloader(
            parquet_path=data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            shuffle=True,
        )

        gpu_proc, log_path = start_dmon("dmon_sanity.log")

        print(f"[INFO] Sanity por {args.num_steps} steps…")
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        losses: List[float] = []
        sims: List[float] = []
        pos_sims: List[float] = []
        logits: List[float] = []
        grad_norms: List[float] = []

        it = iter(dl)
        optimizer.zero_grad(set_to_none=True)

        for step in range(args.num_steps):
            batch = next(it, None)
            if batch is None:
                it = iter(dl)
                batch = next(it)

            actual_loss, loss_dict, grad_norms = train_step(
                batch,
                img_enc,
                txt_enc,
                loss_fn,
                optimizer,
                scheduler,
                args,
                device,
                step,
                grad_norms,
            )
            losses.append(actual_loss)
            sims.append(loss_dict["avg_similarity"])
            pos_sims.append(loss_dict["positive_similarity"])
            logits.append(loss_dict["logit_scale"])
            log_metrics(step, actual_loss, loss_dict, grad_norms)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        wall = time.perf_counter() - t0
        wall_min = wall / 60.0

        first_10_avg = float(np.mean(losses[:10]))
        last_10_avg = float(np.mean(losses[-10:]))
        loss_drop = first_10_avg - last_10_avg

        if torch.cuda.is_available():
            max_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
            torch.cuda.reset_peak_memory_stats()
        else:
            max_mem_gb = 0.0

        gpu_util_avg = stop_dmon(gpu_proc, log_path)

        # Log finales
        mlflow.log_metric("loss0", first_10_avg)
        mlflow.log_metric("loss200", last_10_avg)
        mlflow.log_metric("loss_drop", loss_drop)
        mlflow.log_metric("sanity_wall_clock_min", wall_min)
        mlflow.log_metric("max_memory_allocated_gb", max_mem_gb)
        mlflow.log_metric("gpu_util_avg", gpu_util_avg)

        # SLOs
        pass_loss = loss_drop >= args.min_loss_drop
        pass_time = wall_min <= 25.0
        pass_gpu = gpu_util_avg >= args.min_gpu_util
        mlflow.set_tag("slo.loss_drop", "PASS" if pass_loss else "FAIL")
        mlflow.set_tag("slo.time", "PASS" if pass_time else "FAIL")
        mlflow.set_tag("slo.gpu_util", "PASS" if pass_gpu else "FAIL")

        # Reporte en disco
        write_sanity_report(
            args,
            first_10_avg,
            last_10_avg,
            loss_drop,
            wall_min,
            gpu_util_avg,
            max_mem_gb,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity check (200 steps)")
    parser.add_argument("--data_dir", type=str, default="data/flickr8k")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--max_logit_scale", type=float, default=4.6)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min_loss_drop", type=float, default=0.15)
    parser.add_argument("--min_gpu_util", type=float, default=85.0)
    parser.add_argument("--monitor_gpu", action="store_true")  # compat
    args = parser.parse_args()
    main(args)
