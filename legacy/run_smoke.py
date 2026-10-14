"""Offline, real-CIFAR10 smoke of the existing training and linear-eval entrypoints.

Reduces only dataset size and epochs. Does not replace model/loss/optimizer code.
Run with one visible GPU. Checkpoints contain trusted local argparse objects.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets


def small_loader(loader, per_class, training, balanced=False):
    subset = loader.dataset
    base = subset.dataset
    indices = np.asarray(subset.indices)
    labels = np.asarray(base.targets)[indices]
    selected = np.concatenate([indices[labels == c][:per_class] for c in np.unique(labels)])
    selected_labels = np.asarray(base.targets)[selected]
    sampler = None
    if balanced:
        counts = {int(c): int((selected_labels == c).sum()) for c in np.unique(selected_labels)}
        weights = [1.0 / counts[int(c)] for c in selected_labels]
        sampler = WeightedRandomSampler(weights, len(selected))
    print('SMOKE subset:', dict(zip(*np.unique(selected_labels, return_counts=True))), flush=True)
    return DataLoader(Subset(base, selected.tolist()), batch_size=16,
                      shuffle=training and sampler is None, sampler=sampler,
                      num_workers=2, pin_memory=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    data = Path(args.data_folder).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(output)
    torch.set_num_threads(2)
    assert torch.cuda.is_available(), 'GPU smoke requires CUDA'
    assert torch.cuda.device_count() == 1, 'Select one GPU using CUDA_VISIBLE_DEVICES'
    # Enforce offline dataset access even though the original loaders request download=True.
    cifar_class = datasets.CIFAR10
    def offline_cifar(*a, **kw):
        kw['download'] = False
        return cifar_class(*a, **kw)
    datasets.CIFAR10 = offline_cifar
    offline_cifar(root=str(data), train=True)
    offline_cifar(root=str(data), train=False)
    report = {'status': 'running', 'torch': torch.__version__,
              'gpu': torch.cuda.get_device_name(0), 'data_folder': str(data),
              'training': [], 'evaluation': []}
    started = time.time()
    import main_drift_etf_v2 as train_module
    original_loader = train_module.set_loader
    def train_loader(opt, replay_indices):
        loader, indices = original_loader(opt, replay_indices)
        return small_loader(loader, 16, True), indices
    train_module.set_loader = train_loader
    original_train = train_module.train
    def checked_train(loader, model, teacher, criterion, optimizer, epoch, opt, t=None):
        assert not teacher.training, 'Teacher must remain in eval mode'
        before = {k: v.clone() for k, v in teacher.named_buffers()}
        old_weight = next(model.parameters()).detach().clone()
        loss, teacher = original_train(loader, model, teacher, criterion, optimizer, epoch, opt, t=t)
        assert np.isfinite(loss), loss
        assert not torch.equal(old_weight, next(model.parameters())), 'Model weights did not update'
        assert all(torch.equal(before[k], v) for k, v in teacher.named_buffers()), 'Teacher buffers changed'
        report['training'].append({'task': int(t), 'batches': len(loader),
                                   'samples': len(loader.dataset), 'loss': float(loss),
                                   'weights_updated': True, 'teacher_buffers_unchanged': True})
        return loss, teacher
    train_module.train = checked_train
    sys.argv = ['main_drift_etf_v2.py', '--dataset', 'cifar10', '--data_folder', str(data),
                '--exp_name', 'smoke', '--model', 'resnet18', '--mem_size', '32',
                '--batch_size', '16', '--num_workers', '2', '--epochs', '1', '--start_epoch', '1',
                '--end_task', '1', '--seed', '1997', '--learning_rate', '0.05',
                '--temp', '0.5', '--current_temp', '0.2', '--past_temp', '0.01',
                '--drift_lamda', '0.1', '--print_freq', '1', '--cosine']
    train_module.main()
    ckpt_dir = output / 'save_random_32/cifar10_models/smoke'
    log_dir = output / 'save_random_32/logs/smoke'
    for task in (0, 1):
        ckpt = torch.load(ckpt_dir / f'last_random_{task}.pth', map_location='cpu', weights_only=False)
        assert all(torch.isfinite(v).all() for v in ckpt['model'].values()), 'Nonfinite checkpoint'
        assert ckpt['etf_dict'] is not None
    replay = np.load(log_dir / 'replay_indices_random_1.npy')
    assert len(replay) == 32
    raw = offline_cifar(root=str(data), train=True)
    assert set(np.asarray(raw.targets)[replay]) == {0, 1}
    report['replay_samples'] = len(replay)
    # Release representation-training allocations before classifier training.
    torch.cuda.empty_cache()
    import main_linear_buffer as linear
    original_eval_loader = linear.set_loader
    def eval_loader(opt, replay_indices, target_task):
        tr, val, counts = original_eval_loader(opt, replay_indices, target_task)
        return small_loader(tr, 16, True, balanced=True), small_loader(val, 16, False), counts
    linear.set_loader = eval_loader
    original_validate = linear.validate
    def checked_validate(loader, model, classifier, criterion, opt):
        result = original_validate(loader, model, classifier, criterion, opt)
        loss, _, correct, counts, task_acc = result
        class_acc = float(np.sum(correct) / np.sum(counts) * 100)
        assert np.isfinite(loss) and np.isfinite(class_acc)
        assert sum(counts) == 64
        report['evaluation'].append({'samples': int(sum(counts)), 'loss': float(loss),
                                      'class_il_accuracy': class_acc,
                                      'task_il_accuracy': float(task_acc)})
        return result
    linear.validate = checked_validate
    sys.argv = ['main_linear_buffer.py', '--dataset', 'cifar10', '--data_folder', str(data),
                '--model', 'resnet18', '--target_task', '1', '--epochs', '1', '--num_workers', '2',
                '--batch_size', '16', '--learning_rate', '0.1', '--print_freq', '1',
                '--ckpt', str(ckpt_dir), '--logpt', str(log_dir)]
    linear.main()
    saved = torch.load(ckpt_dir / 'linear_1.pth', map_location='cpu', weights_only=False)
    assert all(torch.isfinite(v).all() for v in saved['linear'].values())
    assert (ckpt_dir / 'acc_buffer_1.txt').is_file()
    report.update(status='passed', elapsed_seconds=time.time()-started,
                  max_cuda_memory_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
                  checkpoint_dir=str(ckpt_dir))
    (output / 'smoke_result.json').write_text(json.dumps(report, indent=2) + '\n')
    print('SMOKE PASSED\n' + json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
