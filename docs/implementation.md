# Implementation and migration notes

## Maintained algorithm

The maintained implementation is based on `legacy/main_drift_etf_v2.py`, not the different center-to-sample penalty in `legacy/main_drift_etf.py`.

The default objective is:

```text
asymmetric supervised contrastive loss
+ range_weight × pairwise region hinge
+ position_weight × task prototype MSE
+ drift_weight × old-sample squared feature distance
+ relation_weight × instance relation distillation
```

The contrastive and relation losses use normalized 128-dimensional projections. Geometry and feature drift use normalized encoder outputs (512 dimensions for ResNet-18/34, 2048 for ResNet-50/101). Projection heads are discarded for linear classification.

The range hinge includes **all off-diagonal current-task sample pairs**, including same-class pairs, to preserve released v2 behavior. The paper's equation is written over different-class pairs. This refactor does not claim to resolve that difference. The range and position reductions retain their v2 means. `range_weight=position_weight=1` preserves the released defaults; unlike the old unused `lamda` parameter, these separate weights actually take effect.

ETF prototypes represent **tasks**, not classes. ETF construction uses the configured full task sequence, independent of an early stopping prefix. This fixes early-stop and resume inconsistencies; it means a two-task smoke no longer uses the old smoke's two-center ETF. Exact old smoke losses are therefore not expected to match.

## Intentional engineering changes

- Teacher snapshots are placed in eval mode with gradients disabled; BatchNorm statistics cannot drift.
- Empty replay and replay-only batches yield finite zero-valued unavailable loss terms.
- Relation distributions use log-softmax instead of exponentiation followed by logarithms.
- ETF dimension follows the selected backbone rather than a hardcoded 512.
- Device use is explicit, with CPU supported. There is no hidden `.cuda()` or Apex dependency.
- Task memory is class-balanced, bounded by capacity, and sampled only from prior memory plus the current task. Random selections are not bit-for-bit identical to historical scripts.
- Checkpoints include ETF, optimizer, memory, and RNG states. They use simple containers/tensors and load with `weights_only=True`.
- Existing output directories are protected. Evaluation has its own output directory.
- Dataset paths are supplied explicitly. No training or evaluation command performs network I/O.
- Logging uses JSON/JSONL. Automatic email notifications and editor deployment settings are removed.

Changing `workers` on resume is supported, but exact stochastic equivalence is only asserted in the deterministic CPU regression test with unchanged worker count. CUDA determinism also depends on the same hardware/runtime. Data files are not embedded in checkpoints.

## Legacy migration

The pre-refactor root scripts are under `legacy/`, including the earlier smoke wrapper. They remain available for archaeology, but are not the supported training interface. To inspect or run them, work from that directory and install their historical dependencies separately. Their old machine paths and experiment parameters have not been normalized.

Legacy SMTP helpers are disabled. The original scripts remain in Git history. No repository history was rewritten.

The maintained implementation supports CIFAR-10, CIFAR-100, and the original processed Tiny-ImageNet format. It does not implement all archived ablations, domain-incremental variants, or the paper's integrations into other baselines. These are not silently aliased to the maintained v2 path.

## Reproducibility scope

Numerical tests compare asymmetric SupCon and the region loss (including its gradients) with the archived implementations. Relation distillation is checked against its probability formulation. Tests verify exact model state equality between uninterrupted and resumed tiny-model CPU runs; real ResNet/data integration is covered by the GPU smoke.

A smoke run verifies execution, not convergence or paper-level accuracy. Full CIFAR-100 and Tiny-ImageNet benchmarks must be run before claiming reproduced results.
