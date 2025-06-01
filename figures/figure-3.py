# 2025 Jan Provaznik (provaznik@optics.upol.cz)
#

import numpy as np
import matplotlib.pyplot as mpp
from helpers import zstd_pickle_load

def make_paper_plot (base_path, target_list, label = None):
    Rv = zstd_pickle_load(f'{base_path}/rspace.pickle.zstd')
    Zv = zstd_pickle_load(f'{base_path}/zspace.pickle.zstd')
    
    xgrid, ygrid = np.meshgrid((1 - Zv), (1 - Zv), indexing = 'ij')
    dmask = ~ np.logical_and(xgrid <= 0.40, ygrid <= 0.30)
    
    task_spec_list = [
        [
            (f'{base_path}/pnrd_pnrd_{targetm:02}.pickle.zstd',
                f'PNR ($m = {targetm}$)'),
            (f'{base_path}/capd_pnrd_{targetm:02}_20.pickle.zstd',
                f'CAP ($n = 20, m = {targetm}$)'),
            (f'{base_path}/capd_pnrd_{targetm:02}_15.pickle.zstd',
                f'CAP ($n = 15, m = {targetm}$)'),
            (f'{base_path}/capd_pnrd_{targetm:02}_10.pickle.zstd', 
                f'CAP ($n = 10, m = {targetm}$)'),
        ]
        for targetm in target_list
    ]
    task_spec_list_cols = 4
    task_spec_list_rows = len(task_spec_list)
    
    cmapP = 'turbo'
    vminP = -5.0
    vmaxP = -1.0
    
    fig = mpp.figure(layout = 'compressed', figsize = (4 * task_spec_list_cols + 3, 9))
    axs = fig.subplots(task_spec_list_rows, 1 + task_spec_list_cols, width_ratios = [ 1 ] * task_spec_list_cols + [0.1])

    for row, spec_list in enumerate(task_spec_list):
        for col, task_spec in enumerate(spec_list):
            file_path, plot_name = task_spec
            task_data = zstd_pickle_load(file_path)
        
            pdata = np.log10(task_data[:, :, 0])    
            pdata[dmask] = np.nan
            
            axs[row, col].pcolormesh(
                xgrid, ygrid, pdata,
                cmap = cmapP, vmin = vminP, vmax = vmaxP)
            axs[row, col].set_title(plot_name)
    
        cmP = mpp.cm.ScalarMappable(norm = mpp.Normalize(vmin = vminP, vmax = vmaxP), cmap = cmapP)
        fig.colorbar(cmP, cax = axs[row, -1], label = r'success rate $\log_{10} (P)$')
    
    for col in range(task_spec_list_cols):
        for row in range(task_spec_list_rows):
            axs[row, col].set_aspect('equal')
            axs[row, col].axis([ -0.015, 0.415, -0.015, 0.315 ])
            axs[row, col].set_xticks([ 0.00, 0.10, 0.20, 0.30, 0.40 ])
            axs[row, col].set_xticks([ 0.025, 0.050, 0.075, 0.125, 0.150, 0.175, 0.225, 0.250, 0.275, 0.325, 0.350, 0.375 ], minor = True)
            axs[row, col].set_yticks([ 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ])
            axs[row, col].set_yticks([ 0.025, 0.050, 0.075, 0.125, 0.150, 0.175, 0.225, 0.250, 0.275 ], minor = True)
            
    for col in range(task_spec_list_cols):
        for row in range(task_spec_list_rows - 1):
            axs[row, col].set_xticklabels([])
        axs[task_spec_list_rows - 1, col].set_xlabel('$1 - \\zeta_{1}$\nHERALDING LOSS')
    
    for row in range(task_spec_list_rows):
        axs[row, 0].set_ylabel('$1 - \\zeta_{2}$\nCHARACTERIZATON LOSS')
        for col in range(1, task_spec_list_cols):
            axs[row, col].set_yticklabels([])

    if label:
        fig.savefig(f'export/paper_unified_merged_{label}.pdf', bbox_inches = 'tight')
    else:
        fig.savefig(f'export/paper_unified_merged.pdf', bbox_inches = 'tight')

base_path = '../results/unified.10dB.51'
make_paper_plot(base_path, [ 4, 5 ], label = '10dB-51')
    
base_path = '../results/unified.15dB.51'
make_paper_plot(base_path, [ 4, 5 ], label = '15dB-51')
