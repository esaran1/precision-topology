import numpy as np
import pytest

from src.data import gaussian_blobs, gauss_linking_integral, linked_core_circles, linked_tori


def test_core_circles_have_unit_gauss_linking_number():
    first, second = linked_core_circles(n_points=800)
    linking_number = gauss_linking_integral(first, second)
    print(f"Gauss linking integral Lk = {linking_number:.12f}")
    assert abs(linking_number) == pytest.approx(1.0, abs=2e-4)
    assert abs(linking_number) > 0.9


@pytest.mark.parametrize("generator", [linked_tori, gaussian_blobs])
def test_dataset_shape_balance_and_dtypes(generator):
    dataset = generator(n_per_class=37, seed=123)
    assert dataset.features.shape == (74, 3)
    assert dataset.labels.shape == (74,)
    assert dataset.features.dtype == np.float32
    assert dataset.labels.dtype == np.int64
    assert np.array_equal(np.bincount(dataset.labels), np.array([37, 37]))
    assert np.isfinite(dataset.features).all()


@pytest.mark.parametrize("generator", [linked_tori, gaussian_blobs])
def test_seeding_is_bit_identical(generator):
    first = generator(n_per_class=100, seed=8675309)
    second = generator(n_per_class=100, seed=8675309)
    assert np.array_equal(first.features, second.features)
    assert np.array_equal(first.labels, second.labels)


def test_different_seeds_change_linked_tori_samples():
    first = linked_tori(n_per_class=100, seed=1)
    second = linked_tori(n_per_class=100, seed=2)
    assert not np.array_equal(first.features, second.features)


def test_tube_radius_guard_keeps_solid_tori_disjoint():
    with pytest.raises(ValueError, match="minimum core separation"):
        linked_tori(n_per_class=10, tube_radius=0.5, major_radius=1.0)
    allowed = linked_tori(n_per_class=10, tube_radius=0.49, major_radius=1.0)
    assert allowed.features.shape == (20, 3)
