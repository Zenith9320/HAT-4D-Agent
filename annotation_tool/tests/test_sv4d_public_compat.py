from pathlib import Path
import sys

import pytest
import torch


GM_ROOT = (
    Path(__file__).resolve().parents[1]
    / "model4dgen"
    / "generative-models"
)
sys.path.insert(0, str(GM_ROOT))

from sgm.models.diffusion import DiffusionEngine  # noqa: E402


class _Encoder(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2, bias=False)
        self.quant_conv = torch.nn.Conv2d(2, 2, 1, bias=False)
        torch.nn.init.constant_(self.encoder.weight, value)
        torch.nn.init.constant_(self.quant_conv.weight, value)


class _Embedder:
    def __init__(self, encoder):
        self.encoder = encoder


def _engine():
    engine = object.__new__(DiffusionEngine)
    torch.nn.Module.__init__(engine)
    source = _Encoder(3.0)
    view = _Encoder(0.0)
    motion = _Encoder(0.0)
    engine.conditioner = type(
        "Conditioner",
        (),
        {"embedders": [_Embedder(source), _Embedder(view), _Embedder(motion)]},
    )()
    engine.first_stage_model = _Encoder(0.0)
    return engine, source, view, motion


def test_public_compat_copies_encoder_weights(monkeypatch):
    from sgm.modules.encoders import modules

    monkeypatch.setattr(modules, "VideoPredictionEmbedderWithEncoder", _Embedder)
    engine, source, view, motion = _engine()
    missing = [
        "conditioner.embedders.4.encoder.encoder.weight",
        "conditioner.embedders.5.encoder.encoder.weight",
        "first_stage_model.encoder.weight",
        "first_stage_model.quant_conv.weight",
    ]

    assert engine._initialize_hat4d_public_compat(missing) == []
    assert torch.equal(view.encoder.weight, source.encoder.weight)
    assert torch.equal(motion.encoder.weight, source.encoder.weight)
    assert torch.equal(engine.first_stage_model.encoder.weight, source.encoder.weight)


def test_public_compat_rejects_core_missing_weights(monkeypatch):
    from sgm.modules.encoders import modules

    monkeypatch.setattr(modules, "VideoPredictionEmbedderWithEncoder", _Embedder)
    engine, *_ = _engine()

    with pytest.raises(RuntimeError, match="non-encoder HAT-4D weights"):
        engine._initialize_hat4d_public_compat(["model.diffusion_model.input_blocks.0.weight"])
