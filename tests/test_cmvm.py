import numpy as np
import pytest

from da4ml.cmvm.api import solve
from da4ml.cmvm.util.bit_decompose import csd_decompose
from da4ml.cmvm.util.mat_decompose import kernel_decompose


@pytest.fixture(params=[2, 4, 8])
def n_dim(request) -> int:
    return request.param


@pytest.fixture(params=[2, 4, 8])
def bits(request) -> int:
    return request.param


@pytest.fixture
def kernel(n_dim, bits):
    kernel = np.round((np.random.rand(n_dim, n_dim) - 0.5) * 2 ** (bits + 1)).astype(np.float32)
    return kernel


def test_decompose(kernel):
    csd, shift0, shift1 = csd_decompose(kernel.astype(np.float64))
    shift2 = np.arange(csd.shape[-1])
    recon = csd * (2.0 ** shift0[:, None, None]) * (2.0 ** shift1[None, :, None]) * (2.0 ** shift2[None, None, :])
    recon_sum = np.sum(recon, axis=-1)
    assert np.all(recon_sum == kernel)


@pytest.mark.parametrize('dc', [-2, -1, 0, 1, 2])
def test_kernel_decompose(kernel, dc: int):
    m0, m1 = kernel_decompose(kernel.astype(np.float64), dc=dc)
    recon = m0 @ m1
    assert np.all(recon == kernel)


@pytest.mark.parametrize('hard_dc', [0, 2, -1])
@pytest.mark.parametrize('method0', ['mc', 'wmc'])
@pytest.mark.parametrize('method1', ['mc', 'wmc'])
@pytest.mark.parametrize('decompose_dc', [0, -1, -2])
@pytest.mark.parametrize('search_all_decompose_dc', [False, True])
def test_solve(kernel, method0, method1, hard_dc, decompose_dc, search_all_decompose_dc):
    sol = solve(
        kernel,
        hard_dc=hard_dc,
        method0=method0,
        method1=method1,
        decompose_dc=decompose_dc,
        search_all_decompose_dc=search_all_decompose_dc,
        adder_size=4,
        carry_size=4,
    )

    assert np.all(sol.kernel == kernel)
