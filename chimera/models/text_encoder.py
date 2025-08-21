
from typing import Tuple
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class TextEncoder(nn.Module):
    def __init__(
            self,
            embed_dim: int = 256,
            freeze_backbone: bool = True,
            max_length: int = 32
    ):
        """
        TextEncoder is a PyTorch module for encoding text sequences into fixed-size, L2-normalized embeddings.
        It leverages a pretrained transformer backbone (by default, 'sentence-transformers/all-MiniLM-L6-v2')
        and projects the output to a configurable embedding dimension.

        Args:
            embed_dim (int): Output embedding dimension after projection. Default is 256.
            freeze_backbone (bool): If True, the transformer backbone weights are frozen. Default is True.
            max_length (int): Maximum sequence length for tokenization. Default is 32.

        Attributes:
            embed_dim (int): Output embedding dimension.
            checkpoint (str): Name or path of the pretrained transformer model.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for input texts.
            model (transformers.PreTrainedModel): Pretrained transformer model.
            num_features (int): Hidden size of the transformer model.
            proj (nn.Sequential): Linear projection and layer normalization.
            max_length (int): Maximum sequence length for tokenization.
        """

        super().__init__()
        self.embed_dim = embed_dim

        self.checkpoint = "sentence-transformers/all-MiniLM-L6-v2"

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, use_fast=True)
        self.model = AutoModel.from_pretrained(self.checkpoint)

        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        self.num_features = self.model.config.hidden_size

        self.proj = nn.Sequential(
            nn.Linear(self.num_features, embed_dim, bias=False),
            nn.LayerNorm(embed_dim)
        )

        self.max_length = max_length

    def forward(self, texts: list[str] | dict | None = None, **kwargs) -> torch.Tensor:
        """
        Encodes a batch of texts into L2-normalized embeddings.

        Args:
            texts (list[str] | dict): List of input strings or a dictionary with 'input_ids' and 'attention_mask'.

        Returns:
            torch.Tensor: L2-normalized embeddings of shape [batch_size, embed_dim].
        """

        if texts is None:
            if {"input_ids", "attention_mask"}.issubset(kwargs.keys()):
                texts = {"input_ids": kwargs["input_ids"], "attention_mask": kwargs["attention_mask"]}
            else:
                raise ValueError("Provide either `texts` or `input_ids`+`attention_mask`.")

        inputs = self.tokenize(texts)

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        token_embeddings = outputs.last_hidden_state  # [B, L, H]
        attention_mask = inputs['attention_mask'].unsqueeze(-1)  # [B, L, 1]
        masked_embeddings = token_embeddings * attention_mask
        sum_embeddings = masked_embeddings.sum(dim=1)
        lengths = attention_mask.sum(dim=1).clamp(min=1)
        mean_pooled = sum_embeddings / lengths
        projected = self.proj(mean_pooled)
        z_text = F.normalize(projected, p=2, dim=-1)

        return z_text

    def tokenize(self, texts: list[str] | dict) -> dict:
        """
        Tokenizes input texts for the transformer model.

        Args:
            texts (list[str] | dict): List of input strings or a dictionary with 'input_ids' and 'attention_mask'.

        Returns:
            dict: Tokenized inputs suitable for the transformer model.
        """
        # 1) Si ya viene tokenizado, lo devolvemos tal cual (y validamos mínimamente)
        if isinstance(texts, dict):
            required = {"input_ids", "attention_mask"}
            missing = required - set(texts.keys())
            if missing:
                raise ValueError(f"Missing keys in tokenized input: {missing}")
            return texts

        # 2) Si es lista de strings, validamos y tokenizamos
        if not isinstance(texts, (list, tuple)):
            raise TypeError("texts must be list[str] or a tokenized dict")
        if len(texts) == 0:
            raise ValueError("Input texts cannot be empty.")
        if any(not isinstance(t, str) for t in texts):
            raise TypeError("All inputs must be strings")
        if any(t == "" for t in texts):
            raise ValueError("Input texts cannot be empty")

        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors = "pt",
        )

    @torch.no_grad()
    def write_detailed_summary(
        self,
        save_path: str,
        input_size: Tuple[int, ...] = (32,),
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

        # 👉 Construimos un batch sintético ya tokenizado (evitamos usar el tokenizer aquí)
        L = self.max_length
        V = getattr(self.model.config, "vocab_size", 30522)
        input_ids = torch.randint(low=0, high=V, size=(batch_size, L), dtype=torch.long)
        attention_mask = torch.ones((batch_size, L), dtype=torch.long)
        tokenized = {"input_ids": input_ids, "attention_mask": attention_mask}

        s = summary(
            self,
            input_data=(tokenized,),  # pasa el dict como único *arg posicional*
            depth=6,
            device=device,
            verbose=0,
            col_names=("kernel_size", "input_size", "output_size", "num_params", "mult_adds", "trainable"),
            row_settings=("var_names",),
        )

        Path(os.path.dirname(save_path) or ".").mkdir(parents=True, exist_ok=True)
        with open(f"{save_path}_summary.md", "w") as f:
            f.write("## Model summary (torchinfo)\n\n")
            f.write("```\n")
            f.write(str(s))
            f.write("\n```")
