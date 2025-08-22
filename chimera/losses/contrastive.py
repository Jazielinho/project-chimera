import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        logit_scale_min: float = 0.0,
        logit_scale_max: float = 4.6051702,  # ln(100)
        label_smoothing: float = 0.0,
        learnable_temperature: bool = True,
        assert_normalized: bool = False,
    ):
        super().__init__()
        self.logit_scale_min = logit_scale_min
        self.logit_scale_max = logit_scale_max
        self.label_smoothing = label_smoothing
        self.assert_normalized = assert_normalized

        if learnable_temperature:
            init_scale = torch.log(torch.tensor(1.0 / temperature))
            self.logit_scale = nn.Parameter(init_scale)
        else:
            self.register_buffer(
                "logit_scale", torch.log(torch.tensor(1.0 / temperature))
            )

        if not (0.0 <= self.label_smoothing < 1.0):
            raise ValueError("label_smoothing debe estar en [0, 1).")
        if temperature <= 0:
            raise ValueError("temperature debe ser > 0.")

    def forward(self, image_features, text_features):
        if torch.isnan(image_features).any() or torch.isnan(text_features).any():
            print(
                "WARNING: NaNs detected in input features! Replacing non-finite values with zeros."
            )

        if image_features.shape[0] != text_features.shape[0]:
            raise ValueError(
                f"Batch size mismatch: {image_features.shape[0]} vs {text_features.shape[0]}"
            )

        batch_size = image_features.shape[0]
        device = image_features.device
        if batch_size < 2:
            raise ValueError(
                "InfoNCE requiere al menos 2 ejemplos en el batch para funcionar correctamente."
            )

        if self.assert_normalized:
            # --- FIX: crear tensores de comparación en el mismo device ---
            img_norms = image_features.norm(dim=-1)
            txt_norms = text_features.norm(dim=-1)
            ones_img = torch.ones_like(img_norms, device=device)
            ones_txt = torch.ones_like(txt_norms, device=device)
            assert torch.allclose(
                img_norms, ones_img, atol=1e-3
            ), "image_features no están normalizados"
            assert torch.allclose(
                txt_norms, ones_txt, atol=1e-3
            ), "text_features no están normalizados"

        # --- NEW: sanea valores no finitos antes de normalizar ---
        if not torch.isfinite(image_features).all():
            image_features = torch.where(
                torch.isfinite(image_features),
                image_features,
                torch.zeros_like(image_features),
            )
        if not torch.isfinite(text_features).all():
            text_features = torch.where(
                torch.isfinite(text_features),
                text_features,
                torch.zeros_like(text_features),
            )

        # Normalización defensiva (por si acaso)
        image_features = F.normalize(image_features, dim=1, eps=1e-8)
        text_features = F.normalize(text_features, dim=1, eps=1e-8)

        # Clamp logit_scale para estabilidad numérica (más conservador)
        logit_scale = self.logit_scale.clamp(self.logit_scale_min, self.logit_scale_max)
        logit_scale_exp = logit_scale.exp()

        # Similitud coseno (matriz [B, B])
        similarity_matrix = image_features @ text_features.t()
        logits = logit_scale_exp * similarity_matrix  # Escalado por temperatura

        labels = torch.arange(batch_size, device=device)

        # Cross-entropy simétrica
        loss_img2txt = F.cross_entropy(
            logits, labels, label_smoothing=self.label_smoothing
        )
        loss_txt2img = F.cross_entropy(
            logits.t(), labels, label_smoothing=self.label_smoothing
        )
        loss = 0.5 * (loss_img2txt + loss_txt2img)

        # Guardrails: NaN/inf
        if not torch.isfinite(loss):
            max_logit = logits.abs().max().item()
            raise RuntimeError(
                f"Loss NaN/inf detectada. max|logits|={max_logit}, logit_scale_exp={logit_scale_exp.item()}"
            )

        with torch.no_grad():
            # avg_similarity = media global (incluye off-diagonal)
            avg_similarity = similarity_matrix.mean().item()
            positive_similarity = similarity_matrix.diagonal().mean().item()
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
            negative_similarity = similarity_matrix[mask].mean().item()
            effective_temp = 1.0 / logit_scale_exp.item()
            pred_image2text = logits.argmax(dim=1)
            pred_text2image = logits.t().argmax(dim=1)
            accuracy_img2txt = (pred_image2text == labels).float().mean().item()
            accuracy_txt2img = (pred_text2image == labels).float().mean().item()
            accuracy = 0.5 * (accuracy_img2txt + accuracy_txt2img)

        return {
            "loss": loss,
            "loss_img2txt": loss_img2txt,
            "loss_txt2img": loss_txt2img,
            "logit_scale_exp": logit_scale_exp.detach().item(),
            "avg_similarity": avg_similarity,
            "positive_similarity": positive_similarity,
            "negative_similarity": negative_similarity,
            "effective_temperature": effective_temp,
            "accuracy": accuracy,
            "accuracy_img2txt": accuracy_img2txt,
            "accuracy_txt2img": accuracy_txt2img,
        }
