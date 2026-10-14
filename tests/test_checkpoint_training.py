"""Task-boundary resume regression; tiny models keep CPU CI lightweight."""

from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from gplasc import evaluation, training
from gplasc.config import Config
from gplasc.runtime import load_checkpoint


class TinyModel(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 512), nn.BatchNorm1d(512), nn.ReLU())
        self.head = nn.Linear(512, 128)

    def forward(self, images, return_feat=False):
        encoded = self.encoder(images)
        features = F.normalize(self.head(encoded), dim=1)
        return (features, encoded) if return_feat else features


class TinyData(Dataset):
    def __init__(self, two_views=False):
        self.targets = [c for c in range(10) for _ in range(4)]
        self.two_views = two_views

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        # Random augmentation exercises saved RNG restoration.
        value = torch.full((3, 4, 4), index / 40) + torch.rand(3, 4, 4) * 0.1
        return ((value, value + torch.rand_like(value) * 0.1) if self.two_views else value), self.targets[index]


def test_resume_matches_uninterrupted_training_and_evaluation(tmp_path, monkeypatch):
    torch.set_num_threads(1)

    def loader(config, root, train=True, two_views=False, download=False):
        return TinyData(two_views)

    monkeypatch.setattr(training, "load_dataset", loader)
    monkeypatch.setattr(evaluation, "load_dataset", loader)
    monkeypatch.setattr(training, "SupConResNet", TinyModel)
    monkeypatch.setattr(evaluation, "SupConResNet", TinyModel)
    config = Config(
        train_tasks=2,
        workers=0,
        batch_size=4,
        memory_size=4,
        epochs=1,
        first_task_epochs=1,
        warmup_epochs=0,
        samples_per_class=2,
        evaluation_samples_per_class=2,
        evaluation_epochs=1,
    )
    device = torch.device("cpu")
    full, _ = training.train(config, tmp_path, tmp_path / "full", device)
    first, _ = training.train(replace(config, train_tasks=1), tmp_path, tmp_path / "first", device)
    resumed, _ = training.train(config, tmp_path, tmp_path / "resumed", device, first)
    a, b = load_checkpoint(full), load_checkpoint(resumed)
    assert a["memory_indices"] == b["memory_indices"]
    torch.testing.assert_close(a["etf"], b["etf"], rtol=0, atol=0)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    result = evaluation.evaluate(resumed, tmp_path, tmp_path / "evaluation", device)
    assert result["encoder_unchanged"]
    assert result["final"]["samples"] == 8
    assert load_checkpoint(tmp_path / "evaluation/linear_last.pt")["kind"] == "linear"
