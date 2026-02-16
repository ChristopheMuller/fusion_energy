import pytest
import numpy as np
from generators import DataGenerator
from structures import PotentialOutcomes

def test_data_generator_shapes():
    dim = 4
    n_rct = 50
    n_ext = 100
    gen = DataGenerator(dim=dim)

    rct_data = gen.generate_rct_pool(
        n=n_rct,
        mean=np.zeros(dim),
        var=1.0,
        treatment_effect=2.0
    )

    assert isinstance(rct_data, PotentialOutcomes)
    assert rct_data.X.shape == (n_rct, dim)
    assert rct_data.Y0.shape == (n_rct,)
    assert rct_data.Y1.shape == (n_rct,)

    ext_data = gen.generate_external_pool(
        n=n_ext,
        mean=np.ones(dim),
        var=1.5
    )

    assert isinstance(ext_data, PotentialOutcomes)
    assert ext_data.X.shape == (n_ext, dim)
    assert ext_data.Y0.shape == (n_ext,)
    assert ext_data.Y1 is None

def test_data_generator_callable_treatment_effect():
    dim = 2
    n = 10
    gen = DataGenerator(dim=dim)

    def treat_eff(X):
        return X[:, 0] * 2.0

    rct_data = gen.generate_rct_pool(
        n=n,
        mean=np.zeros(dim),
        var=1.0,
        treatment_effect=treat_eff
    )

    expected_diff = rct_data.X[:, 0] * 2.0
    actual_diff = rct_data.Y1 - rct_data.Y0
    np.testing.assert_allclose(actual_diff, expected_diff, rtol=1e-5, atol=1e-5)
