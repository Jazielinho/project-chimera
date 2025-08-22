import contextlib
import io
import math

import pytest
import torch
import torch.nn.functional as F

from chimera.losses.contrastive import ContrastiveLoss


@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_normalized_pairs(batch=8, dim=16, noise=0.0, seed=123, device="cpu"):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(batch, dim, generator=g, device=device)
    x = F.normalize(x, dim=1)
    if noise > 0:
        eps = torch.randn(batch, dim, generator=g, device=device) * noise
        y = F.normalize(x + eps, dim=1)
    else:
        y = x.clone()
    return x, y


def _shuffle_rows(t):
    idx = torch.randperm(t.size(0), device=t.device)
    return t[idx], idx


def _default_loss(learnable_temperature=True, **kw):
    return ContrastiveLoss(
        temperature=0.07, learnable_temperature=learnable_temperature, **kw
    )


def test_output_keys_and_finite_values(device):
    loss_fn = _default_loss().to(device)
    i, t = _make_normalized_pairs(batch=8, dim=32, device=device)
    out = loss_fn(i, t)
    expected = {
        "loss",
        "loss_img2txt",
        "loss_txt2img",
        "logit_scale_exp",
        "avg_similarity",
        "positive_similarity",
        "negative_similarity",
        "effective_temperature",
        "accuracy",
        "accuracy_img2txt",
        "accuracy_txt2img",
    }
    assert expected.issubset(out.keys())
    # Tensores principales
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["loss_img2txt"])
    assert torch.isfinite(out["loss_txt2img"])
    # Escala/temperatura
    assert math.isfinite(out["logit_scale_exp"])
    assert out["logit_scale_exp"] > 0
    assert math.isfinite(out["effective_temperature"])
    assert out["effective_temperature"] > 0
    # Rango de accuracies
    for k in ["accuracy", "accuracy_img2txt", "accuracy_txt2img"]:
        v = out[k]
        assert 0.0 <= v <= 1.0


def test_aligned_pairs_loss_lower_than_shuffled(device):
    loss_fn = _default_loss(learnable_temperature=False).to(device)
    i, t = _make_normalized_pairs(batch=16, dim=64, noise=0.05, device=device)
    out_aligned = loss_fn(i, t)["loss"].item()
    t_shuf, _ = _shuffle_rows(t)
    out_shuf = loss_fn(i, t_shuf)["loss"].item()
    assert out_aligned < out_shuf


def test_symmetry_of_bidirectional_loss(device):
    loss_fn = _default_loss(learnable_temperature=False).to(device)
    i, t = _make_normalized_pairs(batch=10, dim=32, noise=0.02, device=device)
    out1 = loss_fn(i, t)["loss"].item()
    # Intercambiar entradas (debería ser esencialmente igual)
    out2 = loss_fn(t, i)["loss"].item()
    assert pytest.approx(out1, rel=1e-4, abs=1e-6) == out2


def test_scale_invariance_due_to_internal_normalization(device):
    loss_fn = _default_loss(learnable_temperature=False).to(device)
    i, t = _make_normalized_pairs(batch=12, dim=48, noise=0.1, device=device)
    out_ref = loss_fn(i, t)["loss"].item()

    # Escalar arbitrariamente antes de la normalización defensiva
    i2 = i * 10.0
    t2 = t * 0.1
    out_scaled = loss_fn(i2, t2)["loss"].item()
    assert pytest.approx(out_ref, rel=1e-5, abs=1e-7) == out_scaled


def test_temperature_effect_monotone(device):
    # Temperatura fija (no aprendible) para probar monotonía:
    # menor T (mayor logit_scale) -> logits más “picudos” -> pérdida menor
    loss_fn_lowT = ContrastiveLoss(temperature=0.05, learnable_temperature=False).to(
        device
    )
    loss_fn_highT = ContrastiveLoss(temperature=0.5, learnable_temperature=False).to(
        device
    )

    i, t = _make_normalized_pairs(batch=16, dim=32, noise=0.1, device=device)
    loss_lowT = loss_fn_lowT(i, t)["loss"].item()
    loss_highT = loss_fn_highT(i, t)["loss"].item()
    assert loss_lowT < loss_highT


def test_gradients_flow_including_learnable_temperature(device):
    loss_fn = _default_loss(learnable_temperature=True).to(device)
    i, t = _make_normalized_pairs(batch=8, dim=16, noise=0.2, device=device)
    i.requires_grad_(True)
    t.requires_grad_(True)

    out = loss_fn(i, t)
    loss = out["loss"]
    loss.backward()

    # Gradientes existen y son finitos
    assert i.grad is not None and torch.isfinite(i.grad).all()
    assert t.grad is not None and torch.isfinite(t.grad).all()
    assert hasattr(loss_fn, "logit_scale") and loss_fn.logit_scale.grad is not None
    assert torch.isfinite(loss_fn.logit_scale.grad).all()


def test_batch_size_mismatch_raises(device):
    loss_fn = _default_loss().to(device)
    i, t = _make_normalized_pairs(batch=8, dim=16, device=device)
    t_mismatch = t[:-1]  # 7
    with pytest.raises(ValueError):
        loss_fn(i, t_mismatch)


def test_batch_too_small_raises(device):
    loss_fn = _default_loss().to(device)
    i, t = _make_normalized_pairs(batch=1, dim=16, device=device)
    with pytest.raises(ValueError):
        loss_fn(i, t)


def test_assert_normalized_true_raises_on_unnormalized(device):
    # Como el assert va antes de la normalización defensiva, debe fallar
    loss_fn = _default_loss(assert_normalized=True).to(device)
    g = torch.Generator(device=device).manual_seed(7)
    i = torch.randn(8, 16, generator=g, device=device)  # no normalizado
    t = torch.randn(8, 16, generator=g, device=device)  # no normalizado
    with pytest.raises(AssertionError):
        loss_fn(i, t)


def test_logit_scale_clamping(device):
    # Fuerza un logit_scale enorme y comprueba que se “clampa”
    loss_fn = _default_loss(
        learnable_temperature=True, logit_scale_min=0.0, logit_scale_max=math.log(100)
    ).to(device)
    with torch.no_grad():
        loss_fn.logit_scale.fill_(10.0)  # muy grande
    i, t = _make_normalized_pairs(batch=8, dim=16, device=device)
    out = loss_fn(i, t)
    # Debe ser exp(clamp(10, max=ln(100))) = 100
    assert pytest.approx(out["logit_scale_exp"], rel=1e-6) == 100.0


def test_label_smoothing_executes(device):
    # Simplemente valida que corre y devuelve valores plausibles
    loss_fn = ContrastiveLoss(
        temperature=0.1, label_smoothing=0.1, learnable_temperature=False
    ).to(device)
    i, t = _make_normalized_pairs(batch=10, dim=32, noise=0.1, device=device)
    out = loss_fn(i, t)
    assert torch.isfinite(out["loss"])
    assert 0.0 <= out["accuracy"] <= 1.0


def test_nans_warning_printed(device, capsys):
    loss_fn = _default_loss().to(device)
    i, t = _make_normalized_pairs(batch=6, dim=8, device=device)
    i = i.clone()
    i[0, 0] = float("nan")
    # Capturar stdout
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        _ = loss_fn(i, t)
    s = buf.getvalue()
    assert "NaNs detected" in s


def test_losses_close_between_img2txt_and_txt2img(device):
    loss_fn = _default_loss(learnable_temperature=False).to(device)
    # Ruido 0 -> similitud simétrica -> pérdidas idénticas
    i, t = _make_normalized_pairs(batch=14, dim=24, noise=0.0, device=device)
    out = loss_fn(i, t)
    l1 = out["loss_img2txt"].item()
    l2 = out["loss_txt2img"].item()
    assert pytest.approx(l1, rel=1e-6, abs=1e-8) == l2
