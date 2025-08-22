#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementación corregida de pérdida contrastiva (InfoNCE) para entrenamiento.

Este módulo contiene la implementación corregida de la pérdida InfoNCE simétrica
con mejor estabilidad numérica y debugging mejorado.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Implementación corregida de la pérdida InfoNCE simétrica.
    """

    def __init__(self, temperature=0.07, max_logit_scale=4.6, learnable_temp=True):
        """
        Inicializa la pérdida contrastiva corregida.

        Args:
            temperature (float): Temperatura inicial para escalar los logits
            max_logit_scale (float): Valor máximo para clamp del logit_scale
            learnable_temp (bool): Si la temperatura debe ser aprendible
        """
        super().__init__()
        self.max_logit_scale = max_logit_scale

        if learnable_temp:
            # Parámetro aprendible para la temperatura (logit_scale)
            # Inicializar con un valor más conservador
            init_scale = torch.log(torch.tensor(1.0 / temperature))
            self.logit_scale = nn.Parameter(init_scale)
        else:
            # Temperatura fija
            self.register_buffer(
                "logit_scale", torch.log(torch.tensor(1.0 / temperature))
            )

    def forward(self, image_features, text_features):
        """
        Calcula la pérdida InfoNCE simétrica con estabilidad mejorada.

        Args:
            image_features (torch.Tensor): Embeddings de imágenes de forma [B, D]
            text_features (torch.Tensor): Embeddings de textos de forma [B, D]

        Returns:
            dict: Diccionario con la pérdida y métricas adicionales
        """
        # Verificar que no hay NaNs en la entrada
        if torch.isnan(image_features).any() or torch.isnan(text_features).any():
            print("WARNING: NaNs detected in input features!")

        # Normalizar características explícitamente
        image_features = F.normalize(image_features, dim=1, eps=1e-8)
        text_features = F.normalize(text_features, dim=1, eps=1e-8)

        # Obtener tamaño del batch
        batch_size = image_features.shape[0]
        device = image_features.device

        # Clamp logit_scale para estabilidad numérica (más conservador)
        logit_scale = torch.clamp(
            self.logit_scale.exp(), min=0.0, max=self.max_logit_scale
        )

        # Calcular matriz de similitud coseno
        similarity_matrix = torch.matmul(image_features, text_features.t())  # [B, B]

        # Verificar que la similaridad está en el rango esperado [-1, 1]
        if torch.isnan(similarity_matrix).any():
            print("WARNING: NaNs in similarity matrix!")

        # Aplicar escala de temperatura
        logits = logit_scale * similarity_matrix

        # Verificar logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("WARNING: NaNs or Infs in logits!")
            print(f"logit_scale: {logit_scale.item()}")
            print(
                f"similarity_matrix range: [{similarity_matrix.min().item():.4f}, {similarity_matrix.max().item():.4f}]"
            )

        # Etiquetas: diagonal de la matriz (índices coincidentes)
        labels = torch.arange(batch_size, device=device, dtype=torch.long)

        # Calcular pérdida en ambas direcciones
        loss_i2t = F.cross_entropy(
            logits, labels, label_smoothing=0.1
        )  # imagen a texto con smoothing
        loss_t2i = F.cross_entropy(
            logits.t(), labels, label_smoothing=0.1
        )  # texto a imagen con smoothing

        # Pérdida simétrica (promedio)
        loss = (loss_i2t + loss_t2i) / 2.0

        # Verificar pérdida
        if torch.isnan(loss) or torch.isinf(loss):
            print("WARNING: NaN or Inf in loss!")
            print(f"loss_i2t: {loss_i2t.item()}")
            print(f"loss_t2i: {loss_t2i.item()}")

        # Calcular métricas adicionales para debugging
        with torch.no_grad():
            # Similitud promedio
            avg_similarity = similarity_matrix.mean().item()
            # Similitud en la diagonal (positivos verdaderos)
            positive_similarity = similarity_matrix.diag().mean().item()
            # Similitud fuera de la diagonal (negativos)
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
            negative_similarity = similarity_matrix[mask].mean().item()
            # Temperatura efectiva
            effective_temp = 1.0 / logit_scale.item()

            # Accuracies para monitoreo
            pred_i2t = torch.argmax(logits, dim=1)
            pred_t2i = torch.argmax(logits.t(), dim=1)
            acc_i2t = (pred_i2t == labels).float().mean().item()
            acc_t2i = (pred_t2i == labels).float().mean().item()

        return {
            "loss": loss,
            "loss_i2t": loss_i2t,
            "loss_t2i": loss_t2i,
            "avg_similarity": avg_similarity,
            "positive_similarity": positive_similarity,
            "negative_similarity": negative_similarity,
            "logit_scale": logit_scale.item(),
            "temperature": effective_temp,
            "acc_i2t": acc_i2t,
            "acc_t2i": acc_t2i,
        }
