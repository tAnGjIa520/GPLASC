"""Sequential representation training with replay and frozen task teachers."""

import copy
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import indices_for_classes, load_dataset, make_loader, update_memory
from .losses import AsymmetricSupCon, feature_drift, region_losses, region_threshold, relation_distillation, task_etf
from .models import SupConResNet, model_dict
from .runtime import finite_state, load_checkpoint, restore_rng, rng_state, save_checkpoint, seed_all, write_json


def learning_rate(config, epoch, batch, batches, epochs):
    minimum = config.learning_rate * 0.001
    cosine = minimum + (config.learning_rate - minimum) * (1 + math.cos(math.pi * epoch / epochs)) / 2
    if config.warmup_epochs and epoch <= config.warmup_epochs:
        warm_epochs = min(config.warmup_epochs, epochs)
        endpoint = minimum + (config.learning_rate - minimum) * (1 + math.cos(math.pi * warm_epochs / epochs)) / 2
        fraction = (batch + (epoch - 1) * batches) / (warm_epochs * batches)
        return 0.01 + fraction * (endpoint - 0.01)
    return cosine


def train_epoch(model, teacher, loader, optimizer, etf, threshold, task, epoch, epochs, config, device):
    model.train()
    teacher.eval()
    criterion = AsymmetricSupCon(config.temperature)
    totals = {name: 0.0 for name in ("total", "contrastive", "range", "position", "drift", "relation")}
    samples = 0
    old_samples = 0
    for batch, (views, labels) in enumerate(loader):
        labels = labels.to(device)
        images = torch.cat(views).to(device)
        current = (labels >= task * config.task_size) & (labels < (task + 1) * config.task_size)
        old = (labels < task * config.task_size).repeat(2)
        old_samples += int(old.sum()) // 2
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(config, epoch, batch, len(loader), epochs)
        projections, encoded = model(images, return_feat=True)
        range_loss, position_loss = region_losses(
            F.normalize(encoded, dim=1)[current.repeat(2)], etf[:, task], threshold
        )
        losses = {
            "contrastive": criterion(projections, labels, current),
            "range": range_loss,
            "position": position_loss,
            "drift": encoded.sum() * 0,
            "relation": encoded.sum() * 0,
        }
        if task:
            with torch.no_grad():
                previous_projections, previous_encoded = teacher(images, return_feat=True)
            losses["drift"] = feature_drift(encoded, previous_encoded, old)
            losses["relation"] = relation_distillation(
                projections, previous_projections, config.current_temperature, config.past_temperature
            )
        loss = (
            losses["contrastive"]
            + config.range_weight * losses["range"]
            + config.position_weight * losses["position"]
            + config.drift_weight * losses["drift"]
            + config.relation_weight * losses["relation"]
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss: task={task}, epoch={epoch}, batch={batch}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses["total"] = loss
        size = len(labels)
        samples += size
        for name, value in losses.items():
            totals[name] += float(value.detach()) * size
    return {
        **{name: value / samples for name, value in totals.items()},
        "samples": samples,
        "old_samples": old_samples,
        "batches": len(loader),
    }


def train(config, data_root, output, device, resume=None):
    """Train through tasks_to_train; resume only at completed task boundaries."""
    config.validate()
    output = Path(output)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError(f"Output directory is not empty: {output}; choose a new output or --resume")
    output.mkdir(parents=True, exist_ok=True)
    seed_all(config.seed)
    dataset = load_dataset(config, data_root, train=True, two_views=True)
    model = SupConResNet(name=config.model).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay
    )
    start_task = 0
    memory = []
    if resume is None:
        # Allocate for the full task sequence even when only a prefix is trained.
        etf = task_etf(model_dict[config.model][1], config.total_tasks, device)
    else:
        state = load_checkpoint(resume)
        if state.get("kind") != "representation":
            raise ValueError("Resume requires a representation checkpoint")
        prior_config = dict(state["config"])
        requested = config.to_dict()
        for name in ("train_tasks", "workers"):
            prior_config.pop(name)
            requested.pop(name)
        if requested != prior_config:
            raise ValueError("Resume configuration differs; only train_tasks and workers may change")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        etf = state["etf"].to(device)
        memory = state["memory_indices"]
        start_task = state["task"] + 1
        restore_rng(state["rng"])
    if start_task >= config.tasks_to_train:
        raise ValueError("Checkpoint already completed the requested tasks")
    threshold = region_threshold(etf, config.task_size, config.margin)
    write_json(output / "config.json", config.to_dict())
    write_json(
        output / "runtime.json",
        {
            "torch": str(torch.__version__),
            "device": str(device),
            "data_root": str(Path(data_root).resolve()),
            "resume": str(resume) if resume else None,
        },
    )
    history = []
    for task in range(start_task, config.tasks_to_train):
        teacher = copy.deepcopy(model).eval().requires_grad_(False)
        teacher_buffers = {name: value.clone() for name, value in teacher.named_buffers()}
        old_weight = next(model.parameters()).detach().clone()
        replay = list(memory)
        current = indices_for_classes(
            dataset.targets, range(task * config.task_size, (task + 1) * config.task_size), config.samples_per_class
        )
        loader = make_loader(dataset, current + replay, config)
        epochs = config.first_task_epochs if task == 0 else config.epochs
        for epoch in range(1, epochs + 1):
            metrics = train_epoch(
                model, teacher, loader, optimizer, etf, threshold, task, epoch, epochs, config, device
            )
            record = {"task": task, "epoch": epoch, **metrics}
            history.append(record)
            print(json.dumps(record), flush=True)
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        if any(not torch.equal(teacher_buffers[name], value) for name, value in teacher.named_buffers()):
            raise RuntimeError("Frozen teacher buffers changed")
        if not finite_state(model):
            raise FloatingPointError("Non-finite model state after training")
        weights_updated = not torch.equal(old_weight, next(model.parameters()))
        memory = update_memory(dataset.targets, replay, current, config.memory_size)
        checkpoint = output / f"task_{task:02d}.pt"
        save_checkpoint(
            checkpoint,
            {
                "format_version": 1,
                "kind": "representation",
                "config": config.to_dict(),
                "task": task,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "etf": etf.detach().cpu(),
                "memory_indices": memory,
                "replay_indices": replay,
                "rng": rng_state(),
                "checks": {"weights_updated": weights_updated, "teacher_buffers_unchanged": True},
            },
        )
        print(f"Saved {checkpoint}", flush=True)
    return checkpoint, history
