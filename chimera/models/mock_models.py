#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modelos mock corregidos para pruebas de sanity.

Este módulo contiene versiones corregidas de los modelos de imagen y texto
con mejor inicialización y normalización.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class MockImageEncoder(nn.Module):
    """
    Encoder de imagen más realista basado en ResNet18 parcial con mejor inicialización.
    """

    def __init__(self, embed_dim=256):
        """
        Inicializa el encoder de imagen mock con mejor inicialización.

        Args:
            embed_dim (int): Dimensión del embedding de salida
        """
        super().__init__()
        # Usar las primeras capas de ResNet18 para mayor realismo
        resnet = models.resnet18(weights=None)  # Sin weights para entrenar desde cero
        # Tomar solo las primeras capas (hasta layer2 para mantener control)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # 64 canales
            resnet.layer2,  # 128 canales
        )

        # Pooling adaptativo y projector con mejor inicialización
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Sequential(
            nn.Linear(128, 512),
            nn.LayerNorm(512),  # Normalización para estabilidad
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embed_dim),
        )

        # Inicialización mejorada
        self._initialize_weights()

    def _initialize_weights(self):
        """Inicializar pesos correctamente."""
        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass del encoder de imagen.

        Args:
            x (torch.Tensor): Tensor de imagen de forma [B, 3, H, W]

        Returns:
            torch.Tensor: Embedding de imagen de forma [B, embed_dim]
        """
        # Extraer características
        x = self.backbone(x)  # [B, 128, H/8, W/8]
        x = self.pool(x)  # [B, 128, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 128]

        # Proyectar (sin normalización aquí, se hace en la loss)
        x = self.projector(x)  # [B, embed_dim]
        return x


class MockTextEncoder(nn.Module):
    """
    Encoder de texto más realista con embeddings y LSTM más profundos y mejor inicialización.
    """

    def __init__(self, vocab_size=10000, embed_dim=256, hidden_dim=256):
        """
        Inicializa el encoder de texto mock con mejor inicialización.

        Args:
            vocab_size (int): Tamaño del vocabulario
            embed_dim (int): Dimensión del embedding final
            hidden_dim (int): Dimensión oculta del LSTM
        """
        super().__init__()
        # Embedding más grande con inicialización normal
        self.embedding = nn.Embedding(vocab_size, 256)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

        # LSTM bidireccional multicapa
        self.lstm = nn.LSTM(
            256,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1,
        )
        # Proyector más complejo con normalización
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),  # *2 por bidireccional
            nn.LayerNorm(512),  # Normalización para estabilidad
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embed_dim),
        )

        # Inicialización mejorada
        self._initialize_weights()

    def _initialize_weights(self):
        """Inicializar pesos correctamente."""
        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        # Inicializar LSTM
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_normal_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

    def forward(self, captions):
        """
        Forward pass del encoder de texto.

        Args:
            captions: Lista de captions (se convertirán a tokens)

        Returns:
            torch.Tensor: Embedding de texto de forma [B, embed_dim]
        """
        device = next(self.parameters()).device

        # Simular tokenización más determinística para mejor convergencia
        seq_length = 32
        tokens_list = []
        for _, caption in enumerate(captions):
            # Usar más variación pero manteniendo cierta estructura
            base_token = (hash(caption) % 1000) + 1000  # Rango 1000-1999
            # Crear secuencia con patrones más variados
            tokens = torch.tensor(
                [base_token + (j % 100) for j in range(seq_length)], device=device
            )
            tokens_list.append(tokens)

        tokens = torch.stack(tokens_list)  # [B, seq_length]

        # Pasar por embedding
        embedded = self.embedding(tokens)  # [B, seq_length, 256]

        # Pasar por LSTM
        output, (hidden, _) = self.lstm(
            embedded
        )  # output: [B, seq_length, hidden_dim*2]

        # Usar attention simple en lugar de promedio
        # Calcular pesos de attention basados en la última hidden state
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [B, hidden_dim*2]
        attention_weights = torch.softmax(
            torch.bmm(output, last_hidden.unsqueeze(2)).squeeze(2), dim=1
        )  # [B, seq_length]

        # Aplicar attention
        attended = torch.sum(
            output * attention_weights.unsqueeze(2), dim=1
        )  # [B, hidden_dim*2]

        # Proyectar (sin normalización aquí, se hace en la loss)
        x = self.projector(attended)  # [B, embed_dim]

        return x
