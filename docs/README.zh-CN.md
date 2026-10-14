# GPLASC 使用说明

本次整理以原始 `main_drift_etf_v2.py` 为主线，提供统一的训练、线性评估、数据下载和 smoke 命令。旧实验移到 `legacy/`，不会被新训练流程导入。

![GPLASC 方法示意：预先分配任务区域，在区域内学习类别结构。](assets/method-overview.png)

图中颜色表示不同任务，空心点表示任务原型，小点分组表示类别。这是概念示意，不是实验得到的特征可视化。

![GPLASC 两阶段流程：表征学习与冻结编码器后的线性评估。](assets/training-pipeline.png)

## 快速开始

先安装适合机器的 PyTorch / torchvision，再执行：

```bash
python -m pip install -e .
python -m gplasc download --dataset cifar10 --data-root /path/to/data
CUDA_VISIBLE_DEVICES=0 python -m gplasc smoke \
  --data-root /path/to/data --output runs/smoke --device cuda
```

开发机和 GPU 机器共享存储时，只在可联网的开发机下载数据。训练、评估和 smoke 都是离线读取；不需要预训练模型、Docker、Apex 或 TensorBoard。

完整训练与评估：

```bash
python -m gplasc train --config configs/cifar10.json \
  --data-root /path/to/data --output runs/cifar10 --device cuda
python -m gplasc evaluate --checkpoint runs/cifar10/task_04.pt \
  --data-root /path/to/data --output runs/cifar10-eval --device cuda
```

配置文件中 `memory_size` 是所有旧类别共享的回放容量；`classes_per_task` 控制任务划分；`train_tasks` 只限制训练到第几个任务，不改变完整序列的 ETF。任务编号和 checkpoint 文件名从 0 开始。

每任务结束后保存 checkpoint，包含模型、优化器、ETF、回放索引、随机状态。使用 `train --resume path/to/task_01.pt` 可从下一任务继续。恢复时只允许修改 `train_tasks` 和 `workers`；要保持数据内容和顺序不变。不支持中途 epoch 恢复或自动导入旧 checkpoint。

## 方法与目录

- `gplasc/data.py`：数据、增强、回放与分类器平衡采样。
- `gplasc/models.py`：ResNet、投影头和线性分类器。
- `gplasc/losses.py`：对比损失、区域范围、任务中心、旧特征与关系蒸馏。
- `gplasc/training.py`：顺序训练任务。
- `gplasc/evaluation.py`：冻结 encoder 后的线性评估。
- `configs/`：各数据集和 smoke 的显式配置。
- `tests/`：损失数值对照、ETF、空回放、恢复一致性测试。

几何约束与特征蒸馏作用在 encoder 空间，对比学习与关系蒸馏作用在投影空间。ETF 顶点对应任务。

## 注意事项

修复了 teacher 未切换 eval、空回放 NaN、固定设备/维度以及恢复状态不全等问题。保留 v2 默认损失含义：range 项仍包含所有非自身配对（包括同类），这与论文只写不同类配对的公式有所区别；独立权重现在确实生效。详见 [实现说明](implementation.md)。

Smoke 使用真实 CIFAR-10 的小样本，验证两个任务、回放、蒸馏、checkpoint 和线性评估；输出 `smoke_result.json`。它不能说明收敛或复现论文精度。完整基准配置尚需长训练验证。

Tiny-ImageNet 要求原仓库使用的 `processed/*.npy` 格式，不直接支持原始图片目录。

分类器保存最后一轮参数，结果里另外报告历史最高测试准确率；最高测试准确率不等于使用独立验证集选模型。
