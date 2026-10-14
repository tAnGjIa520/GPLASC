"""Validated, serializable configuration shared by training and evaluation."""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

DATASETS = {"cifar10": (10, 2, 32), "cifar100": (100, 20, 32), "tiny-imagenet": (200, 20, 64)}


@dataclass
class Config:
    dataset: str = "cifar10"
    model: str = "resnet18"
    classes_per_task: int | None = None
    train_tasks: int | None = None
    seed: int = 1997
    memory_size: int = 200
    batch_size: int = 256
    workers: int = 4
    epochs: int = 100
    first_task_epochs: int = 500
    learning_rate: float = 0.5
    momentum: float = 0.9
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    temperature: float = 0.1
    current_temperature: float = 0.2
    past_temperature: float = 0.01
    range_weight: float = 1.0
    position_weight: float = 1.0
    drift_weight: float = 1.0
    relation_weight: float = 1.0
    margin: float = 0.15
    evaluation_epochs: int = 100
    evaluation_learning_rate: float = 0.5
    samples_per_class: int | None = None
    evaluation_samples_per_class: int | None = None

    @property
    def num_classes(self):
        return DATASETS[self.dataset][0]

    @property
    def task_size(self):
        return DATASETS[self.dataset][1] if self.classes_per_task is None else self.classes_per_task

    @property
    def total_tasks(self):
        return self.num_classes // self.task_size

    @property
    def tasks_to_train(self):
        return self.train_tasks if self.train_tasks is not None else self.total_tasks

    @property
    def image_size(self):
        return DATASETS[self.dataset][2]

    def validate(self):
        if self.dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        if self.model not in {"resnet18", "resnet34", "resnet50", "resnet101"}:
            raise ValueError(f"Unknown model: {self.model}")
        if self.task_size < 2 or self.num_classes % self.task_size:
            raise ValueError("classes_per_task must be >= 2 and divide the number of classes")
        if self.total_tasks < 2:
            raise ValueError("Task-level ETF requires at least two tasks")
        if not 1 <= self.tasks_to_train <= self.total_tasks:
            raise ValueError("train_tasks must be between 1 and the dataset task count")
        for key in ("batch_size", "epochs", "first_task_epochs", "evaluation_epochs"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        for key in (
            "workers",
            "memory_size",
            "warmup_epochs",
            "weight_decay",
            "range_weight",
            "position_weight",
            "drift_weight",
            "relation_weight",
        ):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative")
        for key in (
            "temperature",
            "current_temperature",
            "past_temperature",
            "learning_rate",
            "evaluation_learning_rate",
        ):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if not 0 <= self.margin <= 1:
            raise ValueError("margin must lie in [0, 1]")
        for key in ("samples_per_class", "evaluation_samples_per_class"):
            if getattr(self, key) is not None and getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive or null")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**values).validate()

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
