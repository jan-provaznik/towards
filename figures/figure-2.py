# 2025 Jan Provaznik (provaznik@optics.upol.cz)
#

import numpy as np
import scipy.interpolate as si

import functools as ft
import matplotlib.pyplot as mpp

from helpers import Stopwatch, zstd_pickle_dump, zstd_pickle_load, sq2db
from stellar import threshold_curve
from circuit import evaluate_circuit_pnrd_pnrd

# Determine the illustratory state
Rv = zstd_pickle_load('../results/unified.10dB.1001/rspace.pickle.zstd')
Zv = zstd_pickle_load('../results/unified.10dB.1001/zspace.pickle.zstd')
Ds = zstd_pickle_load('../results/unified.10dB.1001/pnrd_pnrd_04.pickle.zstd')

# Consider stellar rank 4, retrieve the threshold function
rank = 4
tfun = threshold_curve(4)

# The example state, indices in the dataset
i1 = 600
i2 = 945

# ... information to be passed into the manuscript
print(f'Example (real) state')
print(f'... heralding loss        = {1 - Zv[i1]}')
print(f'... characterization loss = {1 - Zv[i2]}')

# Construct the example state
Ps, Rho = evaluate_circuit_pnrd_pnrd(Ds[i1, i2, 1], Zv[i1], Zv[i2], rank, 20)
# ... compute its derived quantities
rhoY = Rho[rank]
rhoX = 1 - Rho[:rank + 1].sum()

# Certification examples
examples_okay = [ (0.05, 0.70), (0.20, 0.70) ]
examples_fail = [ (0.20, 0.45), (0.05, 0.55) ]

# Plotting
fig = mpp.figure(figsize = (14, 4), layout = 'constrained')
axs = fig.subplots(1, 3)

# Plotting (a) the threshold curve

X0 = np.linspace(0, 1, 501)
Y0 = tfun(X0)

axs[0].plot(X0, Y0, color = '#6A6AD7')
axs[0].fill_between(X0, Y0, color = '#F5F5FB')

# Plotting (a) the certification examples

ps = 0.10
ms = 10

for x, y in examples_okay:
    px, py = x - ps / 2.0, y - ps / 2.0 
    axs[0].scatter([ x ], [ y ], ms, color = '#000099')
    axs[0].add_patch(mpp.Rectangle((px, py), ps, ps, fill = 0, linestyle = 'dashed', linewidth = 1, color = '#000'))

for x, y in examples_fail:
    px, py = x - ps / 2.0, y - ps / 2.0
    axs[0].scatter([ x ], [ y ], ms, color = '#f00')
    axs[0].add_patch(mpp.Rectangle((px, py), ps, ps, fill = 0, linestyle = 'dashed', linewidth = 1, color = '#f00'))

# Plotting (a) the example state marker

axs[0].scatter([ rhoX ], [ rhoY ], 200, marker = '+', color = '#000', zorder = 1000)
axs[0].set_aspect('equal')
axs[0].set_xlabel(f'computed quantity $x_{rank}$')
axs[0].set_ylabel(f'computed quantity $y_{rank}$')
axs[0].set_yticks(np.linspace(0, 0.8, 9))

# Plotting (b) the example state and actual error box 

pw = Ds[i1, i2][3] * 3.0
px = rhoX - pw
ph = Ds[i1, i2][5] * 3.0
py = rhoY - ph

X1 = np.linspace(rhoX - 2 * pw, rhoX + 2 * pw, 101)
Y1 = tfun(X1)

axs[1].plot(X1, Y1, color = '#6A6AD7')
axs[1].scatter([ rhoX ], [ rhoY ], 100, marker = '+', color = '#000')
axs[1].fill_between(X1, Y1, Y1.min(), color = '#F5F5FB')
axs[1].add_patch(mpp.Rectangle((px, py), 2 * pw, 2 * ph, fill = 0, linewidth = 1, linestyle = 'solid'))

axs[1].set_xticks([ rhoX ], labels = [ f'{rhoX:.3f}' ])
axs[1].set_yticks([ rhoY ], labels = [ f'{rhoY:.3f}' ], rotation = 90, ha = 'right', va = 'center')
axs[1].set_xlabel(f'computed quantity $x_{rank}$')

s = np.divide(np.diff(axs[0].get_ylim()), np.diff(axs[0].get_xlim()))[0]
axs[1].set_box_aspect(s)

# Plotting (c) the example state Pn distribution

n = 11
t = np.ceil(Rho[:n].max() * 10) / 10
u = np.ceil(Rho[:n].max() * 10) + 1

axs[2].bar(np.arange(n), Rho[:n], color = '#000099')
axs[2].set_xticks(np.arange(n))
axs[2].set_yticks(np.linspace(0, t, int(u)))
axs[2].set_xlabel(r'state component $n$')
axs[2].set_ylabel(r'component probability $p(n)$')
axs[2].set_box_aspect(s)

# Adjust plot margins (follows the margin logic)
axs[2].set_ylim(- axs[0].get_ymargin() * t, t)

# Add (abc) indicators
for i, t in enumerate('abc'):
    axs[i].text(0.965, 0.955, f'({t})', transform = axs[i].transAxes, ha = 'center', va = 'center', fontsize = 12)

# ... and that's all, folks.
fig.savefig('export/illustrate_process.pdf', bbox_inches = 'tight')

