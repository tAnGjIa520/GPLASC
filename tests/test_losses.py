"""Numerical regression against archived release code, plus stability edges."""

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from gplasc.losses import AsymmetricSupCon, feature_drift, region_losses, relation_distillation, task_etf

LEGACY = Path(__file__).resolve().parents[1] / "legacy"


def test_contrastive_matches_legacy():
    spec = importlib.util.spec_from_file_location("old_losses", LEGACY / "losses_negative_only.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(7)
    features = F.normalize(torch.randn(8, 2, 12), dim=-1)
    labels = torch.tensor([0, 1, 2, 3, 2, 3, 2, 3])
    expected = module.SupConLoss(temperature=0.5)(features, labels, target_labels=[2, 3])
    actual = AsymmetricSupCon(0.5)(torch.cat(torch.unbind(features, 1)), labels, labels >= 2)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


def test_region_matches_legacy(monkeypatch):
    # Isolate the archived function without importing its CLI, logger, or datasets.
    tree = ast.parse((LEGACY / "main_drift_etf_v2.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "etf_loss")
    namespace = {"torch": torch, "relu": torch.nn.ReLU(), "mse_loss": torch.nn.MSELoss()}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "legacy_region", "exec"), namespace)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    torch.manual_seed(9)
    features = F.normalize(torch.randn(9, 16), dim=1).requires_grad_()
    center = F.normalize(torch.randn(16), dim=0)
    expected = namespace["etf_loss"]([(center[None, :], 0.7)], features, 0)
    actual = region_losses(features, center, 0.7)
    for first, second in zip(actual, expected):
        torch.testing.assert_close(first, second)
    expected_gradient = torch.autograd.grad(sum(expected), features, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(sum(actual), features)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-6, rtol=1e-5)
    assert actual[0] > 0  # Nonzero range penalty is exercised even if a smoke batch has zero hinge.


def test_relation_matches_probability_form():
    torch.manual_seed(2)
    current = F.normalize(torch.randn(8, 16), dim=1)
    previous = F.normalize(torch.randn(8, 16), dim=1)
    mask = ~torch.eye(8, dtype=torch.bool)
    new = (current @ current.T / 0.2)[mask].reshape(8, -1).softmax(1)
    old = (previous @ previous.T / 0.01)[mask].reshape(8, -1).softmax(1)
    expected = -(old * new.log()).sum(1).mean()
    torch.testing.assert_close(relation_distillation(current, previous, 0.2, 0.01), expected)


def test_zero_replay_is_finite_and_differentiable():
    current = torch.randn(4, 16, requires_grad=True)
    value = feature_drift(current, torch.randn_like(current), torch.zeros(4, dtype=torch.bool))
    assert value.item() == 0
    value.backward()
    assert torch.isfinite(current.grad).all()


def test_extreme_temperatures_are_finite():
    features = F.normalize(torch.randn(6, 16), dim=1).requires_grad_()
    value = relation_distillation(features, features.detach(), 1e-4, 1e-4)
    assert torch.isfinite(value)
    value.backward()
    assert torch.isfinite(features.grad).all()


def test_etf_geometry():
    np.random.seed(42)
    vectors = task_etf(32, 5)
    torch.testing.assert_close(vectors.norm(dim=0), torch.ones(5))
    torch.testing.assert_close(vectors.sum(1), torch.zeros(32), atol=1e-6, rtol=0)
    gram = vectors.T @ vectors
    torch.testing.assert_close(gram[~torch.eye(5, dtype=torch.bool)], torch.full((20,), -0.25), atol=1e-6, rtol=0)
    with pytest.raises(ValueError):
        task_etf(32, 1)
