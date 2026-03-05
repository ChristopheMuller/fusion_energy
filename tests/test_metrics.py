import pytest
import torch
import numpy as np

from metrics import optimise_soft_weights, compute_batch_energy, compute_weighted_energy

def test_optimise_soft_weights_no_internal():
    torch.manual_seed(42)
    X_source = torch.randn(10, 2)
    X_target = torch.randn(5, 2)

    logits = optimise_soft_weights(
        X_source=X_source,
        X_target=X_target,
        X_internal=None,
        lr=0.05,
        n_iter=10
    )

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (10,)
    assert not logits.requires_grad
    # Logits shouldn't be all exact zeros after optimization
    assert not torch.allclose(logits, torch.zeros_like(logits))


def test_optimise_soft_weights_with_internal():
    torch.manual_seed(42)
    X_source = torch.randn(10, 2)
    X_target = torch.randn(5, 2)
    X_internal = torch.randn(3, 2)

    logits = optimise_soft_weights(
        X_source=X_source,
        X_target=X_target,
        X_internal=X_internal,
        target_n_aug=2,
        lr=0.05,
        n_iter=10
    )

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (10,)
    assert not logits.requires_grad


def test_optimise_soft_weights_missing_target_n_aug():
    X_source = torch.randn(10, 2)
    X_target = torch.randn(5, 2)
    X_internal = torch.randn(3, 2)

    with pytest.raises(ValueError, match="target_n_aug required when X_internal is provided"):
        optimise_soft_weights(
            X_source=X_source,
            X_target=X_target,
            X_internal=X_internal,
            target_n_aug=None
        )


def test_compute_batch_energy_no_internal():
    torch.manual_seed(42)
    # k=2 batches, n_s=3 source instances per batch, dim=2
    X_source_batch = torch.randn(2, 3, 2)
    X_target = torch.randn(4, 2)

    energies = compute_batch_energy(
        X_source_batch=X_source_batch,
        X_target=X_target,
        X_internal=None
    )

    assert energies.shape == (2,)
    # energy values should be floating point numbers
    assert not torch.isnan(energies).any()


def test_compute_batch_energy_with_internal():
    torch.manual_seed(42)
    # k=2 batches, n_s=3 source instances per batch, dim=2
    X_source_batch = torch.randn(2, 3, 2)
    X_target = torch.randn(4, 2)
    X_internal = torch.randn(2, 2)

    energies = compute_batch_energy(
        X_source_batch=X_source_batch,
        X_target=X_target,
        X_internal=X_internal
    )

    assert energies.shape == (2,)
    assert not torch.isnan(energies).any()


def test_compute_batch_energy_precomputed_consistency():
    torch.manual_seed(42)
    k = 2
    n_s = 3
    dim = 2
    n_t = 4

    # Complete external pool
    X_external = torch.randn(10, dim)
    X_target = torch.randn(n_t, dim)

    # Pick some indices
    batch_idx_tensor = torch.tensor([[0, 1, 2], [3, 4, 5]])

    # Fetch batch explicitly
    X_source_batch = X_external[batch_idx_tensor]

    # Calculate directly
    energies_direct = compute_batch_energy(
        X_source_batch=X_source_batch,
        X_target=X_target,
        X_internal=None
    )

    # Precompute elements
    d_et = torch.cdist(X_external, X_target)
    row_sums_et = d_et.sum(dim=1)
    dist_ee = torch.cdist(X_external, X_external)

    # Calculate using precomputed elements
    energies_precomputed = compute_batch_energy(
        X_source_batch=X_source_batch,
        X_target=X_target,
        X_internal=None,
        row_sums_et=row_sums_et,
        dist_ee=dist_ee,
        batch_idx_tensor=batch_idx_tensor
    )

    assert torch.allclose(energies_direct, energies_precomputed)


def test_compute_weighted_energy_basic():
    np.random.seed(42)
    X_target = np.random.randn(4, 2)
    X_internal = np.random.randn(2, 2)
    X_external = np.random.randn(3, 2)
    weights_external = np.array([0.1, 0.5, 0.4])

    energy = compute_weighted_energy(
        X_target=X_target,
        X_internal=X_internal,
        X_external=X_external,
        weights_external=weights_external
    )

    assert isinstance(energy, float)
    assert not np.isnan(energy)
    assert not np.isinf(energy)


def test_compute_weighted_energy_zero_weight():
    # If there are no internal units and external weights sum to 0
    X_target = np.random.randn(4, 2)
    X_internal = np.empty((0, 2))
    X_external = np.random.randn(3, 2)
    weights_external = np.array([0.0, 0.0, 0.0])

    energy = compute_weighted_energy(
        X_target=X_target,
        X_internal=X_internal,
        X_external=X_external,
        weights_external=weights_external
    )

    assert energy == float('inf')
