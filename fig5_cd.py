import numpy as np
from lib.analytical import analytical_results
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
rc('text', usetex=True)
rc('font', size=11, family='sans-serif')
rc('legend', fontsize=10)
rc('text.latex', preamble=r'\usepackage{cmbright}')

# cmap is obtained from
# https://gitlab.com/harvard-machine-learning/double-descent/-/blob/master/intro_ocean_dynamics.ipynb
with open('./lib/colormap.txt', 'r') as f:
    a = f.read()
C = [list(map(int, val.split(' '))) for val in a.split('\n')[:-1]]
cm = mpl.colors.ListedColormap(np.array(C) / 255.0)


# n: number of training examples
n = 120
# d: number of total dimensions
d = 100
# p: number of fast learning dimensions
p = 80
# k: kappa -> the condition number of the modulation matrix, F
k = 40
# standard deviation of the noise added to the teacher output
noise = 0.2
# L2 regularization coefficient
list_log_lambda = np.linspace(0.1, 8, 100)

to_plot = []
for log_lambda in list_log_lambda:
    l2 = 10 ** (-log_lambda)
    ts, _, _, EG = analytical_results(0, n, d, p, k, noise, l2)
    to_plot += [EG]

to_plot = np.array(to_plot).T

fig = plt.figure(figsize=(13.3, 4))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1.3])
ax_1 = plt.subplot(gs[0])
ax_2 = plt.subplot(gs[1])
f1 = ax_1.pcolor(list_log_lambda, ts,
                 to_plot, cmap=cm, vmin=0.2, vmax=0.45, shading='auto')
fig.colorbar(f1, ax=ax_1, orientation='vertical', pad=0.02)

ticks = [0.1, 1, 2, 3, 4, 5, 6, 7, 8]
labels = [r'$10^{0}$', r'$10^{-1}$', r'$10^{-2}$',
          r'$10^{-3}$', r'$10^{-4}$', r'$10^{-5}$',
          r'$10^{-6}$', r'$10^{-7}$', r'$\approx 0.0$']
ax_1.set_xticks(ticks)
ax_1.set_xticklabels(labels)

ax_1.set_ylim([1, 1e4])
ax_1.set_yscale('log')

# three slices to plot against time
ax_2.plot(ts, to_plot[:, 10], label=r'Large $\lambda$', lw=2.2)
ax_2.plot(ts, to_plot[:, 30], label=r'Intermediate $\lambda$', lw=2.2)
ax_2.plot(ts, to_plot[:, -1], label=r'Small $\lambda$', lw=2.2)
ax_2.set_xlim([1, 1e4])
ax_2.set_ylim([0.15, 0.5])

ax_1.set_xlabel(r'Regularization strength $\lambda$')
ax_1.set_ylabel(r'Training time t')

ax_2.set_xlabel(r'Training time t')
ax_2.set_ylabel(r'MSE generalization loss', labelpad=20)

ax_2.set_xscale('log')
ax_2.legend()
ax_2.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.3)
plt.tight_layout()
plt.savefig('expected_results/fig5_cd.png')
# plt.show()
