import pytest
import torch

from collie.interactions import ExplicitInteractions, Interactions


@pytest.fixture()
def test_implicit_interactions():
    return Interactions(
        users=[0, 0, 0, 1, 1, 1, 2, 2],
        items=[0, 1, 2, 1, 2, 3, 0, 2],
        ratings=[1, 1, 1, 1, 1, 1, 1, 1],
        check_num_negative_samples_is_valid=False,
    )


@pytest.fixture()
def test_implicit_predicted_scores():
    return torch.tensor([
        [0.9, 0.4, 0.5, 0.7],
        [0.1, 1.2, 1.7, 0.12],
        [-1, -2, -3, 4],
    ])


@pytest.fixture()
def test_implicit_predicted_scores_negative_sample_user():
    return torch.tensor([
        [0.9, 0.1, -1],
        [0.4, 1.2, -2],
        [0.5, 1.7, -3],
        [0.7, 0.12, 4],
    ])


@pytest.fixture()
def test_explicit_interactions():
    return ExplicitInteractions(
        users=[0, 0, 0, 1, 1, 1, 2, 2],
        items=[0, 1, 2, 1, 2, 3, 0, 2],
        ratings=[1, 2, 3, 4, 5, 4, 3, 2],
    )


@pytest.fixture()
def test_explicit_predicted_scores():
    return torch.tensor([
        [0.9, 1.4, 2.5, -0.7],
        [5.1, 4.2, 3.7, 2.12],
        [-1, 2, -3, 4],
    ])


@pytest.fixture()
def test_implicit_recs():
    return torch.tensor([
        [0, 3, 2, 1],
        [2, 1, 3, 0],
        [3, 0, 1, 2],
    ])


@pytest.fixture()
def test_implicit_recs_negative_sample_user():
    return torch.tensor([
        [0, 1, 2],
        [1, 0, 2],
        [1, 0, 2],
        [2, 0, 1],
    ])


@pytest.fixture()
def test_implicit_labels():
    return torch.tensor([
        [1, 0, 1, 1],
        [1, 1, 1, 0],
        [0, 1, 0, 1],
    ], dtype=float)


@pytest.fixture()
def test_implicit_labels_negative_sample_user():
    return torch.tensor([
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
        [0, 0, 1],
    ], dtype=float)


@pytest.fixture()
def targets(test_implicit_interactions):
    return test_implicit_interactions.mat.tocsr()


@pytest.fixture()
def targets_auc_negative_sample_user():
    # ``sklearn.metrics.roc_auc_score`` throws an error if an item has interacted
    # with every user so changing the final item
    return Interactions(
        users=[0, 0, 0, 1, 1, 1, 2, 2],
        items=[0, 1, 2, 1, 2, 3, 0, 3],  # changed final item
        ratings=[1, 1, 1, 1, 1, 1, 1, 1],
        check_num_negative_samples_is_valid=False,
    ).mat.tocsr()


@pytest.fixture()
def metrics():
    return {
        'mapk': 0.7685185,
        'mrr': 0.8333333,
        'auc': 0.5277777,
        'mse': 5.86055,
        'mae': 1.69750,
    }
