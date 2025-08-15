import os
from pathlib import Path
from typing import Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """
    Encodes images into fixed-size embeddings using a ResNet18 backbone.

    This class utilizes a ResNet18 backbone model, pre-trained on ImageNet,
    to extract features from input images, and employs a projection head
    to transform these features into fixed-size embeddings. It provides a
    normalized embedding as the output and includes functionality for
    visualizing the architecture.

    :ivar backbone: Pre-trained ResNet18 backbone used for feature extraction.
    :type backbone: timm.models.resnet.ResNet

    :ivar num_features: Number of output features produced by the backbone model.
    :type num_features: int

    :ivar embed_dim: Dimension of the output embeddings.
    :type embed_dim: int

    :ivar projection: Sequential module for projecting backbone features to
        the embedding space, followed by LayerNorm.
    :type projection: torch.nn.Sequential
    """

    def __init__(self, embed_dim: int = 256, freeze_backbone: bool = True):
        """
        Initializes an instance of the class with options for an embedding dimension and freezing the
        backbone model parameters. This class uses a ResNet-18 backbone model which is pre-trained
        and projects the extracted features into the embedding space of desired dimensions.

        :param embed_dim: The dimension for the embedding space, used in the linear projection layer.
        :param freeze_backbone: Indicates whether to freeze the parameters of the backbone model
            so that they do not update during training. Set to True to freeze the backbone.
        """
        super().__init__()

        # Store embed_dim as an instance attribute
        self.embed_dim = embed_dim

        self.backbone = timm.create_model(
            model_name="resnet18", pretrained=True, num_classes=0
        )

        self.num_features = self.backbone.num_features

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(self.num_features, embed_dim), nn.LayerNorm(embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input tensor through the backbone network, applies the projection
        head, and normalizes the resulting embedding. This method is commonly used in
        deep learning models to extract and normalize feature representations for
        further tasks or downstream processing.

        :param x: The input tensor of shape [B, num_channels, height, width] to be
                  passed through the model.
        :type x: torch.Tensor

        :return: A normalized embedding tensor of shape [B, embed_dim].
        :rtype: torch.Tensor
        """

        features = self.backbone(x)  # [B, num_features]

        projection = self.projection(features)  # [B, embed_dim]

        normalized_embedding = F.normalize(projection, p=2, dim=-1)

        return normalized_embedding

    @torch.no_grad()
    def write_detailed_summary(
        self,
        save_path: str,
        input_size: Tuple[int, int, int] = (3, 224, 224),
        batch_size: int = 1,
    ) -> None:
        """
        Guarda un resumen detallado (torchinfo) con tamaños, kernels, params y MACs.
        """
        try:
            from torchinfo import summary
        except Exception:
            Path(os.path.dirname(save_path) or ".").mkdir(parents=True, exist_ok=True)
            with open(f"{save_path}_summary.md", "w") as f:
                f.write("Instala torchinfo para el resumen: `pip install torchinfo`")
            return

        device = next(self.parameters()).device
        s = summary(
            self,
            input_size=(batch_size, *input_size),
            depth=6,  # sube la profundidad para ver capas internas
            device=device,
            verbose=0,
            col_names=(
                "kernel_size",
                "input_size",
                "output_size",
                "num_params",
                "mult_adds",
                "trainable",
            ),
            row_settings=("var_names",),
        )

        Path(os.path.dirname(save_path) or ".").mkdir(parents=True, exist_ok=True)
        with open(f"{save_path}_summary.md", "w") as f:
            f.write("## Model summary (torchinfo)\n\n")
            f.write("```\n")
            f.write(str(s))
            f.write("\n```")
