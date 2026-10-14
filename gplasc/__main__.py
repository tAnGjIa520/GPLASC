"""Command-line entry point: download, train, evaluate, and smoke."""

import argparse
import time
from pathlib import Path

import torch

from .config import Config
from .data import load_dataset
from .evaluation import evaluate
from .runtime import device_for, load_checkpoint, write_json
from .training import train


def smoke_config():
    return Config(
        train_tasks=2,
        memory_size=32,
        batch_size=16,
        workers=2,
        epochs=1,
        first_task_epochs=1,
        learning_rate=0.05,
        warmup_epochs=0,
        temperature=0.5,
        drift_weight=0.1,
        evaluation_epochs=1,
        evaluation_learning_rate=0.1,
        samples_per_class=16,
        evaluation_samples_per_class=16,
    ).validate()


def parser():
    root = argparse.ArgumentParser(description="GPLASC continual representation learning")
    commands = root.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download CIFAR data on a networked machine")
    download.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    download.add_argument("--data-root", type=Path, required=True)
    for name in ("train", "evaluate", "smoke"):
        command = commands.add_parser(name)
        command.add_argument("--data-root", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
        command.add_argument("--threads", type=int, default=2, help="PyTorch CPU threads")
        if name == "train":
            command.add_argument("--config", type=Path, required=True)
            command.add_argument("--resume", type=Path, help="Completed task checkpoint")
        if name == "evaluate":
            command.add_argument("--checkpoint", type=Path, required=True)
        if name == "smoke":
            command.add_argument("--config", type=Path, help="Optional override for small CI datasets")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "download":
        config = Config(dataset=args.dataset).validate()
        for split in (True, False):
            load_dataset(config, args.data_root, train=split, download=True)
        print(f"Verified {args.dataset} under {args.data_root.resolve()}")
        return
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    device = device_for(args.device)
    if args.command == "train":
        train(Config.load(args.config), args.data_root, args.output, device, args.resume)
    elif args.command == "evaluate":
        evaluate(args.checkpoint, args.data_root, args.output, device)
    else:
        if args.output.exists():
            raise FileExistsError(f"Smoke output already exists: {args.output}")
        args.output.mkdir(parents=True)
        config = Config.load(args.config) if args.config else smoke_config()
        if config.tasks_to_train < 2 or config.memory_size == 0:
            raise ValueError("Smoke requires at least two tasks and nonzero replay memory")
        started = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        checkpoint, history = train(config, args.data_root, args.output / "train", device)
        state = load_checkpoint(checkpoint)
        if not state["checks"]["weights_updated"] or not state["replay_indices"]:
            raise RuntimeError("Smoke did not exercise learning and replay")
        if not all(bool(torch.isfinite(value).all()) for value in state["model"].values()):
            raise FloatingPointError("Reloaded checkpoint has non-finite parameters")
        if not any(row["old_samples"] > 0 and row["relation"] > 0 for row in history):
            raise RuntimeError("Smoke did not exercise replay and relation distillation")
        metrics = evaluate(checkpoint, args.data_root, args.output / "evaluate", device)
        linear = load_checkpoint(args.output / "evaluate/linear_last.pt")
        if not all(bool(torch.isfinite(value).all()) for value in linear["classifier"].values()):
            raise FloatingPointError("Reloaded classifier has non-finite parameters")
        result = {
            "status": "passed",
            "device": str(device),
            "torch": str(torch.__version__),
            "config": config.to_dict(),
            "training": history,
            "evaluation": metrics["final"],
            "teacher_buffers_unchanged": state["checks"]["teacher_buffers_unchanged"],
            "encoder_unchanged_during_evaluation": metrics["encoder_unchanged"],
            "replay_samples": len(state["replay_indices"]),
            "elapsed_seconds": time.time() - started,
            "max_cuda_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
        }
        write_json(args.output / "smoke_result.json", result)
        print(f"SMOKE PASSED: {args.output / 'smoke_result.json'}", flush=True)


if __name__ == "__main__":
    main()
