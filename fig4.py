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

colors = [
    '#F3E955', '#EDB832', '#A6C859', '#67C59A',
    '#52B2CD', '#4B8FE8', '#4B62F4', '#4337CD'][::-1]

n_epochs = 1e7
# n: number of training examples
n = 110
# d: number of total dimensions
d = 100
# p: number of fast learning dimensions
p = 70
# standard deviation of the noise added to the teacher output
noise = 0.0
# L2 regularization coefficient
l2 = 0.0

all_Rs = []
all_Qs = []
all_EG = []
# k: kappa -> the condition number of the modulation matrix, F
for k in [1, 7, 12, 30, 100, 1000, 100000]:
    ts, Rs, Qs, EG = analytical_results(
        0, n, d, p, k, noise, l2, n_epochs=n_epochs)
    all_Rs += [Rs]
    all_Qs += [Qs]
    all_EG += [EG]

fig = plt.figure(figsize=(13.3, 4))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1])
ax_1 = plt.subplot(gs[0])
ax_2 = plt.subplot(gs[1])


Rs_list = np.linspace(0, 1.05, 100)
Qs_list = np.linspace(0, 1.3, 100)
E_Gs_heatmap = 1 - 2 * Rs_list[:, None] + Qs_list[None, :]
f1 = ax_1.pcolor(
    Rs_list, Qs_list, E_Gs_heatmap.T,
    cmap=cm, vmin=0.0, vmax=2.5, shading='auto')

for i, k in enumerate([1, 7, 12, 30, 100, 1000, 100000]):
    ax_1.plot(all_Rs[i], all_Qs[i], color=colors[i],
              lw='2', label=r'$\kappa=${}'.format(k))
    ax_1.scatter(all_Rs[i][-1:], all_Qs[i][-1:], color=colors[i], s=5)
    ax_2.plot(ts, all_EG[i], color=colors[i],
              lw='2', label=r'$\kappa=${}'.format(k))

ax_1.set_xlabel(r'$R$')
ax_1.set_ylabel(r'$Q$')
ax_2.set_xlabel(r'Training time $t$')
ax_2.set_ylabel(r'MSE generalization loss')

ax_2.set_xscale('log')
ax_2.set_xlim([1.0, 1e7])
ax_2.set_ylim([0.0, 0.5])

ax_1.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.1)
ax_2.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.4)
fig.colorbar(f1, ax=ax_1, orientation='vertical', pad=0.02)
ax_1.legend(loc="lower right")
ax_2.legend(loc="lower right")
plt.tight_layout()
plt.savefig('expected_results/fig4.png')
# plt.show()
