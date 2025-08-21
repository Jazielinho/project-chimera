import torch
import pytest
import transformers
from chimera.models.text_encoder import TextEncoder

@pytest.fixture(scope="module")
def encoder():
    return TextEncoder(embed_dim=256, freeze_backbone=True, max_length=32)


def test_text_encoder_initialization(encoder):
    assert isinstance(encoder, TextEncoder)
    assert encoder.embed_dim == 256
    assert encoder.freeze_backbone is True
    assert encoder.max_length == 32


def test_text_encoder_output_shape(encoder):
    texts = ["hola mundo", "prueba de encoder"]
    z = encoder(texts)
    assert z.shape == (2, 256)


def test_text_encoder_normalization(encoder):
    texts = ["hola mundo", "prueba de encoder"]
    z = encoder(texts)
    norms = torch.norm(z, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_text_encoder_padding(encoder):
    texts = ["a", "esto es una frase mucho más larga que la anterior"]
    z = encoder(texts)
    assert z.shape == (2, 256)


def test_text_encoder_freeze_backbone(encoder):
    backbone_params = list(encoder.model.parameters())
    proj_params = list(encoder.proj.parameters())
    assert all(not p.requires_grad for p in backbone_params)
    assert any(p.requires_grad for p in proj_params)


def test_text_encoder_forward_pass(encoder):
    texts = ["hola mundo", "prueba de encoder"]
    z = encoder(texts)
    assert z is not None
    assert isinstance(z, torch.Tensor)
    assert z.shape[0] == len(texts)
    assert z.shape[1] == encoder.embed_dim
    assert z.dtype == torch.float32


def test_text_encoder_empty_input(encoder):
    texts = []
    with pytest.raises(ValueError, match="Input texts cannot be empty"):
        encoder(texts)


def test_text_encoder_invalid_input(encoder):
    texts = ["hola mundo", 123, None]
    with pytest.raises(TypeError, match="All inputs must be strings"):
        encoder(texts)

    texts = ["hola mundo", "prueba de encoder", ""]
    with pytest.raises(ValueError, match="Input texts cannot be empty"):
        encoder(texts)


def test_text_encoder_max_length(encoder):
    texts = ["a" * 50, "b" * 100]
    z = encoder(texts)
    assert z.shape == (2, 256)
    assert z.dtype == torch.float32
    # Verifica que la longitud de los input_ids no exceda max_length
    tokens = encoder.tokenize(texts)
    assert tokens["input_ids"].shape[1] <= encoder.max_length


def test_text_encoder_device(encoder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    texts = ["hola mundo", "prueba de encoder"]
    z = encoder(texts)
    assert z.device.type == device.type
    if device.type == "cuda":
        assert z.device.index == device.index or z.device.index == 0
    assert z.dtype == torch.float32
    assert z.shape == (2, 256)


def test_text_encoder_batch_size(encoder):
    texts = ["hola mundo"] * 100  # Large batch size
    z = encoder(texts)
    assert z.shape == (100, 256)  # Should handle large batch sizes
    norms = torch.norm(z, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)  # Still normalized


def test_text_encoder_special_characters(encoder):
    texts = ["¡Hola, mundo!", "Prueba de encoder: @#$%^&*()"]
    z = encoder(texts)
    assert z.shape == (2, 256)  # Should handle special characters
    norms = torch.norm(z, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)  # Still normalized


def test_text_encoder_unicode_characters(encoder):
    texts = ["こんにちは世界", "Привет мир"]
    z = encoder(texts)
    assert z.shape == (2, 256)  # Should handle unicode characters
    norms = torch.norm(z, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)  # Still normalized


def test_text_encoder_large_texts(encoder):
    texts = ["a" * 1000, "b" * 2000]  # Very large texts
    z = encoder(texts)
    assert z.shape == (2, 256)  # Should still output the same shape
    norms = torch.norm(z, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)  # Still normalized


def test_text_encoder_different_lengths(encoder):
    texts = ["short", "a bit longer text", "this is a much longer text that exceeds the max length"]
    z = encoder(texts)
    assert z.shape == (3, 256)  # Should handle different lengths
    norms = torch.norm(z, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)  # Still normalized


def test_text_encoder_non_ascii_characters(encoder):
    texts = ["café", "naïve", "résumé"]
    z = encoder(texts)
    assert z.shape == (3, 256)  # Should handle non-ASCII characters
    norms = torch.norm(z, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)  # Still normalized


def test_text_encoder_mixed_languages(encoder):
    texts = ["Hello world", "Hola mundo", "Bonjour le monde"]
    z = encoder(texts)
    assert z.shape == (3, 256)  # Should handle mixed languages
    norms = torch.norm(z, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)  # Still normalized


def test_text_encoder_tokenizer(encoder):
    texts = ["This is a test", "Another test case"]
    tokens = encoder.tokenize(texts)
    assert hasattr(tokens, "input_ids")
    assert hasattr(tokens, "attention_mask")
    assert len(tokens["input_ids"]) == len(texts)
    assert hasattr(tokens["input_ids"], "shape")
    assert hasattr(tokens["attention_mask"], "shape")