# Running GPLASC

[Back to the project overview](../README.md). Run the commands below from the repository root.

## Installation

Python **3.10+** is required. Install a compatible PyTorch/torchvision pair for your CPU or CUDA environment first, then:

```bash
python -m pip install -e .
# For regression tests and formatting checks:
python -m pip install -e '.[dev]'
```

Tested runtime: Python 3.10, PyTorch 2.9.0+cu128, torchvision 0.24.0+cu128, NVIDIA H200. TensorBoard, Apex, Docker, and pretrained weights are not required. Training metrics are written as JSON Lines.

## Prepare data

Download CIFAR-10 on a machine with network access:

```bash
python -m gplasc download --dataset cifar10 --data-root /path/to/data
```

Use `--dataset cifar100` for CIFAR-100. Train, evaluate, and smoke commands **never download data**. On shared storage, download once on the development machine and use the same `--data-root` on the offline GPU machine.

If the official CIFAR-10 download is slow, the [Baidu dataset mirror](https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz) provides the original archive. Its MD5 must be `c58f30108f718f92721af3b95e74349a`; extract it under the data root so that `cifar-10-batches-py/` is directly beneath that root.

Tiny-ImageNet requires the original repository's **processed NumPy format**, with `processed/x_train_01.npy` … `x_train_20.npy`, matching `y_train_*.npy`, and `x_val_*.npy` / `y_val_*.npy`. Images are expected as floating-point NHWC arrays in [0, 1]. The standard raw image directory is not automatically converted or downloaded.

## Quick smoke

```bash
CUDA_VISIBLE_DEVICES=0 python -m gplasc smoke \
  --data-root /path/to/data --output runs/smoke --device cuda
```

This runs two CIFAR-10 tasks, each with at most 16 current examples per class, a 32-example replay buffer, and one epoch per task. It then trains a linear classifier and evaluates 64 test examples. It exercises real ResNet-18 forward/backward passes, replay, distillation, checkpoint loading, and linear evaluation. `--device cpu` is also supported.

Success creates `runs/smoke/smoke_result.json` with `"status": "passed"`. Small-sample accuracy is **not** a reproduction of paper results. Choose a fresh output directory for each run; existing smoke runs are never overwritten.

## Train and evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python -m gplasc train \
  --config configs/cifar10.json --data-root /path/to/data \
  --output runs/cifar10 --device cuda

CUDA_VISIBLE_DEVICES=0 python -m gplasc evaluate \
  --checkpoint runs/cifar10/task_04.pt --data-root /path/to/data \
  --output runs/cifar10-evaluation --device cuda
```

A single visible GPU is recommended. `--device auto` chooses CUDA when available and otherwise CPU. `--threads` controls PyTorch CPU threads; `workers` in the configuration controls data-loader processes. Multi-GPU distributed training is not implemented.

Configuration files are also provided for CIFAR-100 and processed Tiny-ImageNet. They are explicit starting configurations based on the paper and released code; full benchmark accuracy has not been revalidated by this refactor.

| Configuration | Tasks × classes | Representation epochs | Linear evaluation |
|---|---:|---:|---:|
| CIFAR-10 | 5 × 2 | 500 first, 100 subsequent | 100 epochs |
| CIFAR-100 | 5 × 20 | 500 first, 100 subsequent | 100 epochs |
| Tiny-ImageNet | 10 × 20 | 500 first, 50 subsequent | 100 epochs |

`memory_size` is the total replay capacity. Set `train_tasks` to train a prefix of the sequence; ETF construction always uses the full configured task count. `classes_per_task` is validated and respected rather than silently overwritten.

### Resume at a task boundary

```bash
python -m gplasc train --config configs/cifar10.json \
  --data-root /path/to/data --output runs/cifar10-resumed \
  --resume runs/cifar10/task_01.pt --device cuda
```

Checkpoints save the model, SGD state, task ETF, replay indices, and Python/NumPy/PyTorch RNG states. Only `train_tasks` and `workers` may differ on resume. Keep the original dataset, sample ordering, device setup, and runtime versions for comparable results. Mid-epoch resume is not supported. Legacy checkpoints have a different format and are not automatically converted.

## Outputs

- `config.json`, `runtime.json`: effective configuration and execution context.
- `metrics.jsonl`: per-epoch total loss, individual loss components, and replay counts.
- `task_XX.pt`: task-boundary representation checkpoints.
- Evaluation `metrics.json`: final and per-epoch Class-IL / Task-IL accuracies.
- Evaluation `linear_last.pt`: last-epoch classifier and frozen encoder.

The historical best test accuracy is reported for comparison, but the saved classifier is always from the **last epoch**. Test-set maxima are not validation-based model selection. Class-IL uses all classifier outputs; Task-IL restricts predictions using the true task ID, matching the original evaluation protocol.

