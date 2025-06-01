# 2025 Jan Provaznik (provaznik@optics.upol.cz)
# 
# This module implements the optical state preparation circuit. 
# Please refer to the documentation strings of the functions for details.

import numpy as np
import scipy.special as ss

def evaluate_circuit_fast (r, zeta1, zeta2, m, d):
    '''
    State preparation circuit based on a two mode squeezed vacuum state and
    a photon number resolving detector. The model accounts for the transmission
    loss in both the measured (heralding) mode and the marginal carrying the
    resulting state.

    The density matrix of the prepared state is always diagonal.
    The state is necessarily mixed for non-zero loss in either mode.

    This procedure aims to address the performance issues of the reference
    implementation by exploiting broadcasting and vectorization with respect to
    the first parameter (squeezing rate). While it could theoretically make
    sense to accept arrays for the other parameters, such as the measurement
    outcome, all the other parameters, except for the first one, are expected
    to be scalars.

    Parameters
    ----------
    r : float | np.ndarray
        Squeezing rate of the two mode squeezed vacuum state. 
        Either a scalar value or np.ndarray with (multiple) values.
    zeta1 : float
        Intensity transmittance of the loss channel acting on the heralding mode.
    zeta2 : float
        Intensity transmittance of the loss channel acting on the resulting mode.
    m : int
        Targeted Fock state (measurement outcome).
    d : int
        Number of coefficients to compute (dimension)

    Returns
    -------
    float | np.ndarray
        The probability of success.
    np.ndarray
        The diagonal of the resulting density matrix.
        Its values are generally ill-defined for zero probabilities of success.
    '''

    # Convert to np.ndarray instance. Converts scalars into (1, ) arrays.
    srarr = np.atleast_1d(r)

    # Substitutions used in the calculation. Note that alpha is np.ndarray now.
    alpha = (np.tanh(srarr) ** 2)
    beta1 = 1 - zeta1
    beta2 = 1 - zeta2

    # Probability of successful detection of (m) photons.
    Ps = (1 - alpha) * (alpha * zeta1) ** m \
       / (1 - alpha * (1 - zeta1)) ** (m + 1)

    # Compute the elements of the resulting diagonal density matrix.

    A = alpha[..., np.newaxis]
    X = beta1 * beta2 * A
    K = np.arange(0, np.minimum(m + 1, d))
    L = np.arange(m + 1, d)

    Pn = np.zeros(shape = (* srarr.shape, d))
    if K.size:
        Pn[..., K] = ss.hyp2f1(1 + m, 1 + m, 1 + m - K, X) * ss.binom(m, K) \
                   * (zeta2 ** K) * beta2 ** (m - K)
    if L.size:
        Pn[..., L] = ss.hyp2f1(1 + L, 1 + L, 1 + L - m, X) * ss.binom(L, m) \
                   * (zeta2 ** L) * beta1 ** (L - m) * A ** (L - m)
    Pn = Pn * (1 - beta1 * A) ** (1 + m)
    
    return np.squeeze(Ps), np.squeeze(Pn)

def evaluate_circuit_pnrd_pnrd (r, z1, z2, m, d):
    '''
    Implements the state preparation circuit with 
    (*) PNRD detector used for heralding, and
    (*) PNRD detector used for characterization of the prepared state.

    See evaluate_circuit or evaluate_circuit_fast for details.
    '''

    return evaluate_circuit_fast(r, z1, z2, m, d)

def evaluate_circuit_capd_pnrd (r, z1, z2, m, M, K, d):
    '''
    Implements the state preparation circuit with 
    (*) CAPD detector used for heralding, 
        where the cascade comprises M detectors, post-selects on m clicks,
        the computation uses expansion up to K elements, and
    (*) PNRD detector used for characterization of the prepared state.

    See evaluate_circuit or evaluate_circuit_fast for details.
    '''

    Os, On = 0.0, 0.0
    for k in np.arange(K):
        Wk = _detector_capd_weights(m, M, k)
        Ps, Pn = evaluate_circuit_pnrd_pnrd(r, z1, z2, k, d)
        Os += Wk * (Ps)
        On += Wk * (Pn * Ps[..., np.newaxis])
    return Os, On / Os[..., np.newaxis]

@np.vectorize(signature = '(), (), () -> ()')
def _detector_capd_weights (m, M, j):
    '''
    It is well known that photon number resolving detectors are out of this
    world. It is possible to construct their approximations with the currently
    available avalanche detectors. An infinite number of these can be arranged
    into a cascade that realizes the actual photon number resolving detector. 

    Unfortunately the amount of required detectors utterly exceeds the load
    bearing capacity of the standard optical table. This limits the number of
    detectors available in any experiment. A dire consequence is that a finite
    and excruciatingly low number of avalanche detectors is used to poorly
    approximate the elusive photon number resolving detector.

    Consider a cascade made of M avalanche detectors. Suppose that exactly m
    detectors in the cascade click. It does not matter which ones.

    With a bit of extremely nasty combinatorics the associated POVM element can
    be determined (10.1103/PhysRevLett.76.2464, equation 10) easily as

       T(m, M) := sum(j) w(m, M, j) |j><j|.

    This procedure produces the weight w(m, M, j).

    Parameters
    ----------
    m : int
        The number of clicks.
    M : int
        The number of avalanche detectors in the cascade.
    j : int
        The desired coefficient w(m, M, j).

    Returns
    -------
    float
        Value of the desired coefficient w(m, M, j).
    '''

    if j < m:
        return 0.0

    L = np.arange(m + 1)
    O = ss.binom(m, L) * (-1.0) ** L * (m - L) ** j
    return O.sum() * (ss.binom(M, m) * np.float64(M) ** (- j))

