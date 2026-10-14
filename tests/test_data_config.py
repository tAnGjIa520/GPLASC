import numpy as np
import pytest

from gplasc.config import Config
from gplasc.data import indices_for_classes, update_memory


def test_task_count_not_tied_to_training_prefix():
    config = Config(train_tasks=1).validate()
    assert config.total_tasks == 5
    assert config.tasks_to_train == 1


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        Config.from_dict({"typo_weight": 1})
    with pytest.raises(ValueError):
        Config(classes_per_task=3).validate()
    with pytest.raises(ValueError):
        Config(temperature=0).validate()
    with pytest.raises(ValueError):
        Config(classes_per_task=0).validate()


def test_memory_uses_only_available_samples_and_obeys_budget():
    labels = np.repeat(np.arange(6), 20)
    old = indices_for_classes(labels, [0, 1], 3)
    current = indices_for_classes(labels, [2, 3], 10)
    np.random.seed(0)
    memory = update_memory(labels, old, current, 12)
    assert len(memory) == len(set(memory)) == 12
    assert set(memory) <= set(old + current)
    assert np.bincount(labels[memory]).tolist() == [3, 3, 3, 3]
    assert update_memory(labels, old, current, 0) == []


def test_memory_shrinks_old_classes_when_new_tasks_arrive():
    labels = np.repeat(np.arange(6), 20)
    memory = indices_for_classes(labels, [0, 1, 2, 3], 3)
    current = indices_for_classes(labels, [4, 5], 10)
    shrunk = update_memory(labels, memory, current, 12)
    assert np.bincount(labels[shrunk]).tolist() == [2, 2, 2, 2, 2, 2]
    assert set(shrunk) <= set(memory + current)
