"""Class-balanced linear evaluation of frozen representations."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .config import Config
from .data import indices_for_classes, load_dataset, make_loader
from .models import LinearClassifier, SupConResNet
from .runtime import finite_state, load_checkpoint, save_checkpoint, seed_all, write_json


@torch.no_grad()
def validate(model, classifier, loader, config, task, device):
    model.eval()
    classifier.eval()
    classes = (task + 1) * config.task_size
    correct = np.zeros(classes, dtype=np.int64)
    counts = np.zeros(classes, dtype=np.int64)
    task_correct = 0
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = classifier(model.encoder(images))
        total_loss += float(F.cross_entropy(logits, labels)) * len(labels)
        prediction = logits.argmax(1)
        for label in labels.unique():
            mask = labels == label
            key = int(label)
            counts[key] += int(mask.sum())
            correct[key] += int((prediction[mask] == labels[mask]).sum())
            start = key // config.task_size * config.task_size
            task_correct += int(
                (logits[mask, start : start + config.task_size].argmax(1) == labels[mask] - start).sum()
            )
    total = int(counts.sum())
    return {
        "samples": total,
        "loss": total_loss / total,
        "class_il_accuracy": float(correct.sum() / total * 100),
        "task_il_accuracy": task_correct / total * 100,
        "per_class_accuracy": {str(c): float(correct[c] / counts[c] * 100) for c in range(classes) if counts[c]},
    }


def evaluate(checkpoint, data_root, output, device):
    state = load_checkpoint(checkpoint)
    if state.get("kind") != "representation":
        raise ValueError("Evaluate requires a representation checkpoint")
    config = Config.from_dict(state["config"])
    seed_all(config.seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    task = state["task"]
    model = SupConResNet(name=config.model).to(device)
    model.load_state_dict(state["model"])
    model.eval().requires_grad_(False)
    classifier = LinearClassifier(name=config.model, num_classes=config.num_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=config.evaluation_learning_rate, momentum=config.momentum)
    train_data = load_dataset(config, data_root, train=True)
    validation_data = load_dataset(config, data_root, train=False)
    current = indices_for_classes(
        train_data.targets, range(task * config.task_size, (task + 1) * config.task_size), config.samples_per_class
    )
    train_loader = make_loader(train_data, current + state["replay_indices"], config, balanced=True)
    validation_indices = indices_for_classes(
        validation_data.targets, range((task + 1) * config.task_size), config.evaluation_samples_per_class
    )
    validation_loader = make_loader(validation_data, validation_indices, config, train=False)
    history = []
    best_accuracy = -1.0
    original = {name: value.clone() for name, value in model.state_dict().items()}
    for epoch in range(1, config.evaluation_epochs + 1):
        classifier.train()
        for group in optimizer.param_groups:
            group["lr"] = config.evaluation_learning_rate * 0.2 ** sum(epoch > milestone for milestone in (60, 75, 90))
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                encoded = model.encoder(images)
            loss = F.cross_entropy(classifier(encoded), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite linear evaluation loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics = {"epoch": epoch, **validate(model, classifier, validation_loader, config, task, device)}
        history.append(metrics)
        best_accuracy = max(best_accuracy, metrics["class_il_accuracy"])
        print(json.dumps(metrics), flush=True)
    if any(not torch.equal(original[name], value) for name, value in model.state_dict().items()):
        raise RuntimeError("Encoder changed during linear evaluation")
    if not finite_state(classifier):
        raise FloatingPointError("Non-finite classifier checkpoint")
    save_checkpoint(
        output / "linear_last.pt",
        {
            "format_version": 1,
            "kind": "linear",
            "config": config.to_dict(),
            "task": task,
            "model": model.state_dict(),
            "classifier": classifier.state_dict(),
        },
    )
    result = {
        "representation_checkpoint": str(Path(checkpoint).resolve()),
        "history": history,
        "final": history[-1],
        "best_test_accuracy": best_accuracy,
        "encoder_unchanged": True,
        "checkpoint_selection": "last epoch",
    }
    write_json(output / "metrics.json", result)
    return result
