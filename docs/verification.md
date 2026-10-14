# Refactor verification — 2026-10-14

Runtime: Python 3.10, PyTorch 2.9.0+cu128, torchvision 0.24.0+cu128; single NVIDIA H200. Full CIFAR-10 was prepared on a networked development machine and read from shared storage on an offline GPU machine.

## Regression tests

11 CPU tests passed, including:

- Numerical agreement of asymmetric SupCon with the archived implementation.
- Numerical and gradient agreement of the region penalties with v2, including a nonzero range hinge.
- Stable relation-distillation equivalence and extreme-temperature behavior.
- ETF geometry and invalid-configuration rejection.
- Empty replay and memory capacity / old-class shrinking.
- Exact tiny-model CPU state equality for uninterrupted vs. task-boundary-resumed training.
- Frozen-encoder linear evaluation and weights-only checkpoint reload.

Ruff lint and format checks passed. These tests use no downloaded data.

## Real-data GPU smoke

The maintained CLI completed two CIFAR-10 tasks with ResNet-18, 16 current examples per class, batch size 16, 32 replay examples, and one epoch per task. The full five-task ETF was allocated throughout.

| Task | Training examples | Replay examples | Batches | Total loss |
|---|---:|---:|---:|---:|
| 0 | 32 | 0 | 2 | 24.54013 |
| 1 | 64 | 32 | 4 | 15.55037 |

Linear evaluation completed on 64 held-out test examples. Class-IL accuracy was 25% and Task-IL accuracy was 50%. Both representation checkpoints and the classifier checkpoint were saved and reloaded. Teacher buffers and the evaluation encoder remained unchanged, and parameters / losses were finite.

## Real-data GPU resume

The task-1 checkpoint was resumed with `train_tasks=5`. Tasks 2, 3, and 4 each completed four batches, keeping replay capacity at 32. Final linear evaluation ran on 160 held-out test examples (16 per class).

| Task | Total loss |
|---|---:|
| 2 | 15.52751 |
| 3 | 15.58145 |
| 4 | 15.56484 |

Final Class-IL accuracy was 10%, Task-IL accuracy was 50%. This verifies the runtime path for GPU optimizer restoration, repeated memory updates, and full-sequence evaluation.

These are deliberately tiny runs. They do not measure convergence or reproduce the paper's benchmark results. The range hinge was zero in these real-data smoke batches; nonzero range loss and its gradient are covered separately by numerical regression tests. CIFAR-100, processed Tiny-ImageNet, and non-ResNet-18 backbones were not benchmarked in this verification.
