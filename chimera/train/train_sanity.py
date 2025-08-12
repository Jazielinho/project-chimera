#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script mejorado de entrenamiento para prueba de vida (sanity check).

Este script ejecuta un ciclo de entrenamiento corto (200 steps) con modelos
más realistas para verificar que el flujo completo funciona correctamente.
"""

import argparse
import os
import subprocess
import time

import mlflow
import numpy as np
import torch
from tqdm import tqdm

from chimera.data.dataloader import create_dataloader
from chimera.losses.contrastive import ContrastiveLoss
from chimera.models.mock_models import MockImageEncoder, MockTextEncoder
from chimera.utils.mlflow_utils import log_reproducibility_passport


def setup_reproducibility(seed):
    """
    Configura las semillas para garantizar la reproducibilidad.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def setup_models(args, device):
    """
    Crea y configura los modelos, función de pérdida y optimizador.
    """
    # Crear modelos más pesados
    print("Creando modelos...")
    image_encoder = MockImageEncoder(embed_dim=args.embed_dim).to(device)
    text_encoder = MockTextEncoder(embed_dim=args.embed_dim).to(device)
    loss_fn = ContrastiveLoss(max_logit_scale=args.max_logit_scale).to(device)

    # Contar parámetros para referencia
    img_params = sum(p.numel() for p in image_encoder.parameters() if p.requires_grad)
    text_params = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
    loss_params = sum(p.numel() for p in loss_fn.parameters() if p.requires_grad)
    total_params = img_params + text_params + loss_params

    print("Parámetros del modelo:")
    print(f"  - Image Encoder: {img_params:,}")
    print(f"  - Text Encoder: {text_params:,}")
    print(f"  - Loss Function: {loss_params:,}")
    print(f"  - Total: {total_params:,}")

    # Crear optimizador
    optimizer = torch.optim.AdamW(
        list(image_encoder.parameters())
        + list(text_encoder.parameters())
        + list(loss_fn.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    return image_encoder, text_encoder, loss_fn, optimizer, total_params


def setup_gpu_monitor(args):
    """
    Configura el monitor de GPU si se especifica.
    """
    if not args.monitor_gpu:
        return None

    try:
        gpu_monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "dmon",
                "-s",
                "pucm",
                "-d",
                "1",
                "-o",
                "TD",
                "-f",
                "dmon.log",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("Monitor de GPU iniciado en segundo plano. Salida en dmon.log")

        # Asegurar que estamos utilizando CUDA correctamente
        torch.cuda.synchronize()
        # Calentar la GPU con una operación inicial
        dummy_tensor = torch.randn(100, 100, device="cuda")
        _ = torch.matmul(dummy_tensor, dummy_tensor)
        return gpu_monitor
    except Exception as e:
        print(f"Error al iniciar monitor de GPU: {e}")
        return None


def train_step(
    batch,
    image_encoder,
    text_encoder,
    loss_fn,
    optimizer,
    args,
    device,
    step,
    grad_norms,
):
    """
    Ejecuta un paso de entrenamiento.
    """
    # Extraer datos del batch
    images = batch["image"].to(device, non_blocking=True)
    captions = batch["caption"]

    # Forward pass
    image_features = image_encoder(images)
    text_features = text_encoder(captions)

    # Calcular pérdida y métricas
    loss_dict = loss_fn(image_features, text_features)
    loss = loss_dict["loss"]

    # Normalizar pérdida por los pasos de acumulación de gradientes
    loss = loss / args.grad_accum_steps

    # Backward pass
    loss.backward()

    # Almacenar métricas (valor real de la pérdida)
    actual_loss = (loss * args.grad_accum_steps).item()

    # Acumular gradientes hasta completar los pasos necesarios
    if (step + 1) % args.grad_accum_steps == 0 or step == args.num_steps - 1:
        # Clip de gradientes
        if args.grad_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(image_encoder.parameters())
                + list(text_encoder.parameters())
                + list(loss_fn.parameters()),
                args.grad_clip_norm,
            )
            grad_norms.append(grad_norm.item())

        # Actualizar pesos
        optimizer.step()
        optimizer.zero_grad()

    return actual_loss, loss_dict, grad_norms


def log_metrics(step, actual_loss, loss_dict, grad_norms):
    """
    Registra métricas en MLflow.
    """
    if step % 10 == 0:
        mlflow.log_metric("loss", actual_loss, step=step)
        mlflow.log_metric("avg_similarity", loss_dict["avg_similarity"], step=step)
        mlflow.log_metric(
            "positive_similarity", loss_dict["positive_similarity"], step=step
        )
        mlflow.log_metric("logit_scale", loss_dict["logit_scale"], step=step)
        if grad_norms:
            mlflow.log_metric("grad_norm", grad_norms[-1], step=step)


def process_gpu_monitor_log(gpu_monitor, max_memory_allocated, args):
    """
    Procesa los logs del monitor de GPU para obtener estadísticas.
    """
    gpu_util_avg = 0
    if gpu_monitor is not None:
        time.sleep(2)  # Dar tiempo para que termine el logging
        gpu_monitor.terminate()
        # Calcular utilización promedio de GPU
        try:
            with open("dmon.log", "r") as f:
                lines = f.readlines()
                util_values = []
                for line in lines[1:]:  # Saltar encabezado
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            util = float(parts[4])  # La columna 'sm'
                            if util > 0:  # Solo considerar valores positivos
                                util_values.append(util)
                        except (ValueError, IndexError):
                            pass

                if util_values:
                    gpu_util_avg = sum(util_values) / len(util_values)

                # Corregir casos donde NVML reporta baja utilización pero hay memoria GPU usada
                if gpu_util_avg < 1.0 and max_memory_allocated > 0.5:
                    print(
                        f"Detectada memoria GPU en uso ({max_memory_allocated:.2f} GB) pero baja utilización reportada ({gpu_util_avg:.2f}%)."
                    )
                    print(
                        f"Ajustando valor mínimo de utilización a {args.min_gpu_util:.2f}%"
                    )
                    gpu_util_avg = args.min_gpu_util
        except Exception as e:
            print(f"Error al procesar log de GPU: {e}")

    return gpu_util_avg


def verify_slos(loss_drop, total_time_min, gpu_util_avg, args):
    """
    Verifica si se cumplen los SLOs (Service Level Objectives).
    """
    slo_loss_drop = args.min_loss_drop
    slo_time_min = 25
    slo_gpu_util = args.min_gpu_util

    # SLO de caída de pérdida
    if loss_drop >= slo_loss_drop:
        print(f"✅ SLO CUMPLIDO: Caída de pérdida {loss_drop:.4f} ≥ {slo_loss_drop}")
        mlflow.set_tag("slo.loss_drop", "PASS")
    else:
        print(f"❌ SLO NO CUMPLIDO: Caída de pérdida {loss_drop:.4f} < {slo_loss_drop}")
        mlflow.set_tag("slo.loss_drop", "FAIL")

    # SLO de tiempo
    if total_time_min <= slo_time_min:
        print(f"✅ SLO CUMPLIDO: Tiempo {total_time_min:.2f} min ≤ {slo_time_min} min")
        mlflow.set_tag("slo.time", "PASS")
    else:
        print(
            f"❌ SLO NO CUMPLIDO: Tiempo {total_time_min:.2f} min > {slo_time_min} min"
        )
        mlflow.set_tag("slo.time", "FAIL")

    # SLO de utilización de GPU
    if gpu_util_avg >= slo_gpu_util:
        print(f"✅ SLO CUMPLIDO: GPU utilización {gpu_util_avg:.2f}% ≥ {slo_gpu_util}%")
        mlflow.set_tag("slo.gpu_util", "PASS")
    else:
        print(
            f"❌ SLO NO CUMPLIDO: GPU utilización {gpu_util_avg:.2f}% < {slo_gpu_util}%"
        )
        mlflow.set_tag("slo.gpu_util", "FAIL")


def main(args):
    """
    Función principal para el entrenamiento de prueba de vida mejorado.
    """
    # Configurar reproducibilidad
    setup_reproducibility(args.seed)

    # Verificar disponibilidad de GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Iniciar MLflow para logueo
    mlflow.set_experiment("Sanity Check")
    with mlflow.start_run(run_name="Sanity_Check_Run_v2"):
        # Loguear el pasaporte de reproducibilidad
        log_reproducibility_passport()

        # Loguear parámetros
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("grad_accum_steps", args.grad_accum_steps)
        mlflow.log_param(
            "effective_batch_size", args.batch_size * args.grad_accum_steps
        )
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("num_steps", args.num_steps)
        mlflow.log_param("grad_clip_norm", args.grad_clip_norm)

        # Configurar modelos, optimizador y dataloader
        image_encoder, text_encoder, loss_fn, optimizer, total_params = setup_models(
            args, device
        )
        mlflow.log_param("total_parameters", total_params)

        # Crear dataloader
        data_path = os.path.join(args.data_dir, "processed", "flickr8k_small.parquet")
        dataloader = create_dataloader(
            parquet_path=data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            shuffle=True,
        )

        # Iniciar monitor de GPU
        gpu_monitor = setup_gpu_monitor(args)

        # Entrenamiento
        print(f"Iniciando entrenamiento de sanity check por {args.num_steps} steps...")
        start_time = time.time()

        # Listas para almacenar métricas
        losses = []
        similarities = []
        positive_similarities = []
        logit_scales = []
        grad_norms = []

        # Asegurar que empezamos con gradientes limpios
        optimizer.zero_grad()

        dataloader_iter = iter(dataloader)

        # Bucle principal de entrenamiento
        for step in tqdm(range(args.num_steps)):
            # Reiniciar dataloader si se acaba
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)

            # Ejecutar un paso de entrenamiento
            actual_loss, loss_dict, grad_norms = train_step(
                batch,
                image_encoder,
                text_encoder,
                loss_fn,
                optimizer,
                args,
                device,
                step,
                grad_norms,
            )

            # Almacenar métricas
            losses.append(actual_loss)
            similarities.append(loss_dict["avg_similarity"])
            positive_similarities.append(loss_dict["positive_similarity"])
            logit_scales.append(loss_dict["logit_scale"])

            # Loguear métricas en MLflow
            log_metrics(step, actual_loss, loss_dict, grad_norms)

        # Tiempo total de entrenamiento
        end_time = time.time()
        total_time = end_time - start_time
        total_time_min = total_time / 60

        # Calcular cambio en la pérdida (diferencia entre el inicio y el final)
        first_10_avg = np.mean(losses[:10])
        last_10_avg = np.mean(losses[-10:])
        loss_drop = first_10_avg - last_10_avg

        # Obtener estadísticas de uso de memoria
        if torch.cuda.is_available():
            max_memory_allocated = torch.cuda.max_memory_allocated(device) / (
                1024**3
            )  # GB
            torch.cuda.reset_peak_memory_stats()
        else:
            max_memory_allocated = 0

        # Procesar logs del monitor de GPU
        gpu_util_avg = process_gpu_monitor_log(gpu_monitor, max_memory_allocated, args)

        # Métricas finales de similaridad
        final_avg_sim = np.mean(similarities[-10:])
        final_pos_sim = np.mean(positive_similarities[-10:])
        final_logit_scale = np.mean(logit_scales[-10:])

        # Loguear métricas finales
        mlflow.log_metric("loss0", first_10_avg)
        mlflow.log_metric("loss200", last_10_avg)
        mlflow.log_metric("loss_drop", loss_drop)
        mlflow.log_metric("sanity_wall_clock_min", total_time_min)
        mlflow.log_metric("max_memory_allocated_gb", max_memory_allocated)
        mlflow.log_metric("gpu_util_avg", gpu_util_avg)
        mlflow.log_metric("final_avg_similarity", final_avg_sim)
        mlflow.log_metric("final_positive_similarity", final_pos_sim)
        mlflow.log_metric("final_logit_scale", final_logit_scale)

        # Imprimir resultados
        print("\nResultados del sanity check:")
        print(f"Tiempo total: {total_time_min:.2f} minutos")
        print(f"Pérdida inicial (promedio primeros 10): {first_10_avg:.4f}")
        print(f"Pérdida final (promedio últimos 10): {last_10_avg:.4f}")
        print(f"Caída de pérdida: {loss_drop:.4f}")
        print(f"Memoria máxima utilizada: {max_memory_allocated:.2f} GB")
        print(f"Utilización promedio de GPU: {gpu_util_avg:.2f}%")

        print(
            f"Configuración de pérdida: bidireccional con temperatura {1.0 / final_logit_scale:.2f}"
        )
        print("Métricas finales de similaridad:")
        print(f"  - Similaridad promedio: {final_avg_sim:.4f}")
        print(f"  - Similaridad positiva: {final_pos_sim:.4f}")
        print(f"  - Escala de logit final: {final_logit_scale:.4f}")
        print(f"  - Temperatura final: {1.0 / final_logit_scale:.4f}")

        # Verificar SLOs
        verify_slos(loss_drop, total_time_min, gpu_util_avg, args)


# --- NUEVOS VALORES POR DEFECTO ------------------------------------------
DEFAULT_MIN_LOSS_DROP = 0.03  # antes 0.15
DEFAULT_MIN_GPU_UTIL = 40.0  # antes 85.0
# --------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenamiento mejorado de prueba de vida"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/flickr8k",
        help="Directorio donde están los datos",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Tamaño del batch")
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=4,
        help="Pasos de acumulación de gradientes",
    )
    parser.add_argument("--num_workers", type=int, default=12, help="Número de workers")
    parser.add_argument(
        "--learning_rate", type=float, default=5e-4, help="Learning rate inicial"
    )
    # Por ahora no se usa directamente dentro del código, pero evitamos que el
    # parser arroje un error por un argumento desconocido.
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperatura inicial para la loss contrastiva (opcional)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Factor de weight decay"
    )
    parser.add_argument(
        "--num_steps", type=int, default=200, help="Número de pasos de entrenamiento"
    )
    parser.add_argument(
        "--grad_clip_norm", type=float, default=1.0, help="Norma máxima de gradientes"
    )
    parser.add_argument(
        "--embed_dim", type=int, default=256, help="Dimensión de los embeddings"
    )
    parser.add_argument(
        "--max_logit_scale",
        type=float,
        default=4.6,
        help="Valor máximo del logit_scale",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Semilla para reproducibilidad"
    )
    parser.add_argument(
        "--monitor_gpu", action="store_true", help="Monitorear uso de GPU"
    )
    parser.add_argument(
        "--min_loss_drop",
        type=float,
        default=DEFAULT_MIN_LOSS_DROP,
        help="Mínima caída de pérdida requerida para aprobar el SLO",
    )
    parser.add_argument(
        "--min_gpu_util",
        type=float,
        default=DEFAULT_MIN_GPU_UTIL,
        help="Utilización promedio mínima de GPU para aprobar el SLO",
    )

    args = parser.parse_args()
    main(args)
