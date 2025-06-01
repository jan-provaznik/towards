# 2025 Jan Provaznik (provaznik@optics.upol.cz)
#

import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as mpp
import matplotlib.patches as mpx

from helpers import zstd_pickle_load, transform

def make_curve (Zv, path):
    data = zstd_pickle_load(path)
    vals = data[:, :, 0]
    offt = 1 + np.max(np.where(np.isnan(np.nanmin(vals, axis = 1))), initial = -1)
    yvec = Zv[np.nanargmin(vals[offt:, :], axis = 1)]

    # Smoothen the jagged curves a little.
    # Alternatively, to keep them as rough as possible, 
    # set the s parameter as little as you desire.
    return make_splev(Zv[offt:], yvec, s = 4e-4)

def make_splev (x, y, s, k = 3):
    t = si.splrep(x, y, k = k, s = s)
    def curve (X):
        try:
            return si.splev(X, t, ext = 2)
        except ValueError:
            return np.nan
    return np.vectorize(curve)

@transform(list)
def load_dataset (base, pattern, target_list):
    Rv = zstd_pickle_load(f'{base}/rspace.pickle.zstd')
    Zv = zstd_pickle_load(f'{base}/zspace.pickle.zstd')

    for target in target_list:
        yield make_curve(Zv, f'{base}/{pattern.format(target)}')

def value_if_nan (data, value = 0.0):
    data[np.isnan(data)] = value
    return data

def make_plot (base, label):
    dataset_pnrd = load_dataset(base, 'pnrd_pnrd_{:02}.pickle.zstd', [ 3, 4, 5 ])
    dataset_capd_20 = load_dataset(base, 'capd_pnrd_{:02}_20.pickle.zstd', [ 3, 4, 5 ])
    dataset_capd_10 = load_dataset(base, 'capd_pnrd_{:02}_10.pickle.zstd', [ 3, 4, 5 ])

    color_list = [ '#000', '#f00', '#00f' ]

    fig = mpp.figure(figsize = (9, 4), layout = 'constrained')
    axs = fig.subplots(1, 1)

    xvec = np.linspace(0.6, 1.0, 201)

    for curve, color in zip(dataset_pnrd, color_list):
        axs.plot(1 - xvec, 1 - curve(xvec), color = color, linestyle = 'solid')
    for curve, color in zip(dataset_capd_20, color_list):
        axs.plot(1 - xvec, 1 - curve(xvec), color = color, linestyle = 'dashed')
    for curve, color in zip(dataset_capd_10, color_list):
        axs.plot(1 - xvec, value_if_nan(1 - curve(xvec)), color = color, linestyle = 'dotted')


    axs.axis([ 0, 0.40, 0, 0.40 ])
    axs.set_xlabel('$1 - \\zeta_{1}$\nHERALDING LOSS')
    axs.set_ylabel('$1 - \\zeta_{2}$\nCHARACTERIZATON LOSS')

    custom_lines = (
        [ mpp.Line2D([0], [0], color = '#bbb', linestyle = style, linewidth = 4) for style in [ (0, (2, 0)), (0, (2, 0.5)), (0, (1, 0.5)) ] ] +
        [ mpp.Line2D([0], [0], color = color, linewidth = 4) for color in color_list ]
    )
    custom_texts = (
        [ 'PNR', 'CAP (20)', 'CAP (10)' ] + 
        [ f'$\\vert{target}\\rangle$' for target in [ 3, 4, 5 ] ]
    )

    legs = axs.legend(custom_lines, custom_texts, ncols = 2, fancybox = False, frameon = False, handlelength = 2.5, handleheight = 2.0)
    for t in legs.get_texts():
        t.set_verticalalignment('bottom')

    fig.savefig(f'export/curves_unified_03_04_05_{label}.pdf', bbox_inches = 'tight')

make_plot('../results/unified.10dB.51', '10dB-51')
make_plot('../results/unified.10dB.1001', '10dB-1001')

