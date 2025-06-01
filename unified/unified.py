# 2025 Jan Provaznik (provaznik@optics.upol.cz)
#
# Statistical sampling and processing used to get the 
# results presented in the paper.

DEF_EXPERIMENT_RATE = 100_000_000 
DEF_EXPERIMENT_RUNS = 1000

DEF_SAMPLE_R_NUM = 1000
DEF_SAMPLE_R_BEG = 0.00115
DEF_SAMPLE_R_END = 1.15

# Alternatively, for approximately 15dB squeezing.
# DEF_SAMPLE_R_BEG = 0.0015
# DEF_SAMPLE_R_END = 1.50

DEF_SAMPLE_Z_NUM = 51
# DEF_SAMPLE_Z_NUM = 1001
DEF_SAMPLE_Z_BEG = 0.5
DEF_SAMPLE_Z_END = 1.0

DEF_HERALD_CAPD_SPAN = 100
DEF_RESULT_DIMENSION = 20

DEF_DETECTOR_PNRD = [ 3, 4, 5 ]
DEF_DETECTOR_CAPD_CLICK = [ 3, 4, 5 ]
DEF_DETECTOR_CAPD_WIDTH = [ 10, 15, 20 ]

# Off we go.
#

import numpy as np
import itertools as it
import functools as ft
import mpi4py.futures

from circuit import evaluate_circuit_pnrd_pnrd
from circuit import evaluate_circuit_capd_pnrd
from stellar import threshold_curve
from certify import threshold_curve_certify
from helpers import zstd_pickle_dump, zstd_pickle_load
from helpers import Stopwatch, taskwrap, master_target_dispatcher

def cell_sampler (Ps, Pn):
    rng = np.random.default_rng()

    # (@) Sanitize Pn values.
    Pn = np.clip(Pn, 0.0, 1.0)

    # Cs ... count of successful heralding events within a single run
    Cs = np.int64(DEF_EXPERIMENT_RATE * Ps)
    # Cn ... simulated characterization events
    Cn = rng.multinomial(Cs, Pn, 
        size = (DEF_EXPERIMENT_RUNS, DEF_SAMPLE_R_NUM))
    Cn = np.swapaxes(Cn, 0, 1)

    # Fn ... simulated characterization frequencies
    Fn = np.divide(Cn, Cn.sum(axis = -1)[..., np.newaxis], 
        where = (Cs > 0)[..., np.newaxis, np.newaxis])
    return Cs, Fn

def cell_process (Rv, Ps, Cs, Fn, level):
    '''
    Rv ... a list of squeezing rates
    Ps ... theoretical probability of successful heralding event
           computed for each squeezing rate
    Cs ... counts of successful heralding events
           computed for each squeezing rate
    Fn ... theoretical P(n) of the prepared state
           computed for each squeezing rate
    '''

    # Xn, Yn ... certification (threshold curve) points
    Xn = Fn[..., level + 1:].sum(axis = -1)
    Yn = Fn[..., level]
    # a?, s? ... certification (threshold curve) points (statistics)
    aX, sX = Xn.mean(axis = 1), Xn.std(axis = 1)
    aY, sY = Yn.mean(axis = 1), Yn.std(axis = 1)

    # Ml ... certification (threshold curve, 3 sigma) based on (Lachman, 2019)
    Ml = threshold_curve_certify(threshold_curve(level), aX, sX, 3, aY, sY, 3)
    # Mc ... consider only those where Ps[res] > threshold
    Mc = Cs > 1000
    # Mx ... by the powers of these combined!
    Mx = np.logical_and(Ml, Mc)

    if not np.any(Mx):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    Ix = Ps[Mx].argmax()

    # (@) Returns
    #
    # - The maximal probability of success for a squeezing rate that still
    #   passes the certification
    # - The corresponding squeezing rate
    # - The corresponding averages and standard deviations for the ensemble

    return (
        Ps[Mx][Ix], 
        Rv[Mx][Ix],
        aX[Mx][Ix], sX[Mx][Ix], # 2, 3
        aY[Mx][Ix], sY[Mx][Ix]  # 4, 5
    )

# Individual simulation workflows wrapped into callable functions.
#

def task_worker_target_pnrd_pnrd (Rv, z1, z2, m):
    Ps, Pn = evaluate_circuit_pnrd_pnrd(Rv, z1, z2, m, 
        d = DEF_RESULT_DIMENSION)
    Cs, Fn = cell_sampler(Ps, Pn)
    return cell_process(Rv, Ps, Cs, Fn, m)

def task_worker_target_capd_pnrd (Rv, z1, z2, m, M):
    Ps, Pn = evaluate_circuit_capd_pnrd(Rv, z1, z2, m, M, 
        K = DEF_HERALD_CAPD_SPAN,
        d = DEF_RESULT_DIMENSION)
    Cs, Fn = cell_sampler(Ps, Pn)
    return cell_process(Rv, Ps, Cs, Fn, m)

# Dispatch simulation workflows and process the results.
#

def master_target_pnrd_pnrd (rspace, zspace, pool):
    master_target_dispatcher(zspace, pool,
        worker = taskwrap(task_worker_target_pnrd_pnrd, rspace),
        target_name = 'pnrd_pnrd',
        target_name_tail = '{:02}',
        target_tail_list = DEF_DETECTOR_PNRD)

def master_target_capd_pnrd (rspace, zspace, pool):
    master_target_dispatcher(zspace, pool,
        worker = taskwrap(task_worker_target_capd_pnrd, rspace),
        target_name = 'capd_pnrd',
        target_name_tail = '{:02}_{:02}',
        target_tail_list = it.product(
            DEF_DETECTOR_CAPD_CLICK, 
            DEF_DETECTOR_CAPD_WIDTH))

# Dispatcher. 
#
# Uses mpi4py.futures instead of concurrent.futures. Some versions of the
# latter library, in conjuction with some versions of numpy, resulted in
# deadlocks and performance issues.
#

def master ():
    rspace = np.linspace(DEF_SAMPLE_R_BEG, DEF_SAMPLE_R_END, DEF_SAMPLE_R_NUM)
    zspace = np.linspace(DEF_SAMPLE_Z_BEG, DEF_SAMPLE_Z_END, DEF_SAMPLE_Z_NUM)
    zstd_pickle_dump('result/rspace.pickle.zstd', rspace)
    zstd_pickle_dump('result/zspace.pickle.zstd', zspace)

    with mpi4py.futures.MPIPoolExecutor() as pool:
        master_target_pnrd_pnrd(rspace, zspace, pool)
        master_target_capd_pnrd(rspace, zspace, pool)

if (__name__ == '__main__'):
    master()

