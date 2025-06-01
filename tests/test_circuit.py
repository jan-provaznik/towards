# 2025 Jan Provaznik (provaznik@optics.upol.cz)
#

import pytest
import numpy as np
import scipy.special as ss
from circuit import evaluate_circuit_pnrd_pnrd

def fock_kraus_loss (z, d):
    A = np.zeros(shape = (d, d, d))
    for i in np.arange(d):
        k = np.arange(1, d - i)
        v = np.sqrt(np.arange(i + 1, d, dtype = np.float64).cumprod())
        A[k, i, i + k] = v
    A[0] = np.eye(d)
    K = np.arange(d)
    C = np.sqrt(1 - z) ** K / np.sqrt(ss.factorial(K))
    return \
        C[:, np.newaxis, np.newaxis] \
        * np.diag(np.sqrt(z) ** K)[np.newaxis, ...] \
        @ A

def fock_tmsv (r, d):
    '''
    Please note the (row, col, row, col) index ordering of the tensor.
    '''

    c = np.tanh(r)
    O = np.zeros(shape = (d, d, d, d))
    for i, j in np.ndindex(d, d):
        O[i, j, i, j] = c ** (i + j) * (1 - c ** 2)
    return O

def fock_circuit (r, z1, z2, m, d):
    R = fock_tmsv(r, d)
    M = fock_kraus_loss(z1, d)

    Om = np.einsum('kmn, npij, kqp -> mqij', M, R, M)[m, m]
    Ps = Om.trace()

    if Ps > 0:
        Om /= Ps

    M = fock_kraus_loss(z2, d)
    On = np.einsum('kmn, np, kqp -> mq', M, Om, M)

    return Ps, np.diag(On)

@pytest.mark.parametrize('z1', [ 0.25, 0.50, 0.75, 1.00 ])
@pytest.mark.parametrize('z2', [ 0.25, 0.50, 0.75, 1.00 ])
@pytest.mark.parametrize('r', [ 0.2, 0.5 ])
@pytest.mark.parametrize('m', [ 1, 2, 3, 4 ])
def test_evaluate_circuit (r, m, z1, z2):
    '''
    Uses full-scale Fock simulation to test the correctness of the improved
    evaluate_circuit_faster implementation.
    '''

    d = 20
    Ps, Pn = evaluate_circuit_pnrd_pnrd(r, z1, z2, m, d)
    Os, On = fock_circuit(r, z1, z2, m, d)

    assert np.abs(Ps - Os) < 1e-8
    assert np.abs(Pn - On).max() < 1e-8

