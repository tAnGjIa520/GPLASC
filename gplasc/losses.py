"""GPLASC-v2 losses with the original reductions and stable probability math.

Default range regularization includes all off-diagonal current-task pairs,
including same-class pairs, matching the released v2 implementation.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def task_etf(dimension, tasks, device="cpu"):
    """Construct unit simplex vertices as columns of a [dimension, tasks] tensor."""
    if not 2 <= tasks <= dimension:
        raise ValueError("ETF construction requires 2 <= tasks <= feature dimension")
    basis, _ = np.linalg.qr(np.random.random((dimension, tasks)))
    basis = torch.as_tensor(basis, dtype=torch.float32, device=device)
    center = torch.eye(tasks, device=device) - torch.ones(tasks, tasks, device=device) / tasks
    return math.sqrt(tasks / (tasks - 1)) * basis @ center


def region_threshold(etf, classes_per_task, margin):
    cosine = F.cosine_similarity(etf[:, 0], etf[:, 1], dim=0).clamp(-1, 1)
    minimum = 1 - classes_per_task / (classes_per_task - 1) * (1 - cosine) / 2
    return minimum + (1 - minimum) * margin


class AsymmetricSupCon(nn.Module):
    """Current-task anchors, with replay samples retained as contrasts.

    Input ordering is [view_1 batch, view_2 batch]. Averaging includes masked
    replay anchors, as in the original Co2L/SupContrast implementation.
    """

    def __init__(self, temperature, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, projections, labels, current_mask):
        repeated_labels = labels.repeat(2)
        diagonal = torch.eye(len(projections), dtype=torch.bool, device=projections.device)
        logits = projections @ projections.T / self.temperature
        log_prob = F.log_softmax(logits.masked_fill(diagonal, -torch.inf), dim=1)
        positive = repeated_labels[:, None].eq(repeated_labels[None, :]) & ~diagonal
        positive_log_prob = log_prob.masked_fill(~positive, 0).sum(1) / positive.sum(1).clamp_min(1)
        losses = -self.temperature / self.base_temperature * positive_log_prob
        return (losses * current_mask.repeat(2)).mean()


def region_losses(features, center, threshold):
    """Range hinge and coordinate-averaged prototype MSE on normalized features."""
    if len(features) == 0:
        zero = features.sum() * 0
        return zero, zero
    # Normalize again to match cosine_similarity's treatment of roundoff.
    unit = F.normalize(features, dim=1)
    similarity = unit @ unit.T
    off_diagonal = ~torch.eye(len(features), dtype=torch.bool, device=features.device)
    range_loss = F.relu(threshold - similarity[off_diagonal]).mean() if len(features) > 1 else features.sum() * 0
    position_loss = F.mse_loss(features.mean(0), center)
    return range_loss, position_loss


def feature_drift(current, previous, old_mask):
    """Mean squared Euclidean distance on old samples; empty replay is zero."""
    if not old_mask.any():
        return current.sum() * 0
    delta = F.normalize(current, dim=1)[old_mask] - F.normalize(previous, dim=1)[old_mask]
    return delta.square().sum(1).mean()


def relation_distillation(current, previous, current_temperature, past_temperature):
    """Cross entropy between old and new off-diagonal similarity distributions."""
    mask = ~torch.eye(len(current), dtype=torch.bool, device=current.device)
    student = (current @ current.T / current_temperature)[mask].reshape(len(current), -1)
    teacher = (previous @ previous.T / past_temperature)[mask].reshape(len(previous), -1)
    return -(F.softmax(teacher.detach(), dim=1) * F.log_softmax(student, dim=1)).sum(1).mean()
