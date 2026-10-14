# GPU smoke

项目目录：`/mnt/shared-storage-user/solution/tangjia/continual/GPLASC`。
原始 Git commit：`986fc28`，origin 保留 GitHub 地址。

## 环境

`../.venv-gplasc/bin/python` 是独立 venv，通过 `--system-site-packages` 复用现有共享 Conda 环境 `/mnt/shared-storage-user/tangjia/miniconda3/envs/dual-drift`（Python 3.10，PyTorch 2.9.0+cu128，torchvision 0.24.0+cu128）。原 Conda 环境未修改。TensorBoard 依赖装在 venv 内。

注意：这个轻量环境依赖 dual-drift 环境继续存在，并非可独立迁移的 Conda 环境。依赖快照：`../environment-freeze.txt`。

## 数据

完整 CIFAR-10 Python 数据集存放在 `../data/`，含压缩包和解压目录。镜像源：
https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz

归档 MD5 为官方值 `c58f30108f718f92721af3b95e74349a`。GPU smoke 强制 `download=False`，不会访问网络。

## 运行

在 GPU 服务器执行，先选择有空余显存的卡：

```bash
cd /mnt/shared-storage-user/solution/tangjia/continual/GPLASC
CUDA_VISIBLE_DEVICES=2 bash run_smoke.sh
```

每次写入独立的 `../runs/smoke-<日期时间>/` 和 `../logs/smoke-<日期时间>.log`。时间按 Asia/Taipei 生成。成功标志是日志中的 `SMOKE PASSED` 和 `smoke_result.json` 的 `status: passed`。

## 覆盖范围

复用 `main_drift_etf_v2.main()` 和 `main_linear_buffer.main()` 的真实训练入口，smoke wrapper 只缩小加载的数据量，并进行状态检查。

- CIFAR-10 前两个任务：类别 0/1、2/3。
- ResNet-18 + 128 维 MLP projection，batch=16，seed=1997。
- 每类最多 16 张真实训练图，每任务 1 epoch；任务 0 为 2 batch，任务 1 为 4 batch。
- 回放总容量 32，第二任务验证回放只含旧类 0/1。
- 原实现 `end_task=1` 使本次 ETF 有两个任务中心；不是完整五任务实验配置。
- 两项几何约束、非对称 SupCon、IRD、旧样本特征蒸馏都走原训练路径。
- 冻结 encoder，线性分类器训练 1 epoch（4 batch），在每类 16 张、共 64 张测试图上评估。
- 检查训练 loss 有限、encoder 权重更新、teacher buffers 不变、checkpoint 参数有限且可重载、线性评估输出与分类器保存。

这是运行链路验证，不是精度复现；小样本准确率不应与论文比较。

## 必要修正

- v2 改用 PyTorch SummaryWriter，避开旧 tensorboard_logger 的 protobuf 兼容风险。
- 每任务复制 teacher 后设为 eval 并冻结梯度。
- 空旧样本 batch 的 drift loss 返回零，避免 NaN；总损失增加有限性检查。
- v2 checkpoint 保存 ETF，关闭 TensorBoard writer。
- 线性评估增加 `--data_folder`，读取本项目生成的可信 checkpoint 时显式 `weights_only=False`，适配 PyTorch 2.9。

没有重写论文损失：v2 的 lamda 未生效、pair mask 包含同类 pair 等原有公式差异仍然保留，完整复现前应另行处理。主训练入口仍有硬编码 CUDA/512 维等限制，当前 smoke 使用 ResNet-18。

## 首次运行结果（2026-10-14）

已在指定远程 GPU 服务器的 H200 GPU 2 上通过。

- 结果：`../runs/smoke-20261014-01/smoke_result.json`，`status=passed`。
- 日志：`../logs/smoke-20261014-01.log`。
- 训练 task 0：2 batch，平均 loss=24.5724；task 1：4 batch，平均 loss=15.5143。
- 回放 32 张；两次训练均确认 encoder 更新、teacher buffers 不变。
- 线性评估：64 张图，loss=1.63494，Class-IL=25%，Task-IL=50%。这里只验证执行和指标计算，不评价精度。
- 核心训练/评估及保存检查约 11.4 秒（不含前置导入/数据检查），PyTorch 峰值 allocated 显存约 2718 MiB。
- 本次两个任务中心的 ETF 及默认 margin 使 range penalty 为 0；几何项被正常计算，但本次没有验证非零 range penalty 的收敛行为。
- checkpoints 位于 `../runs/smoke-20261014-01/save_random_32/cifar10_models/smoke/`：`last_random_0.pth`、`last_random_1.pth`、`linear_1.pth`。
