import sys
sys.path.append("..")
import numpy as np
from lib.analytical import analytical_results, analytical_results_general_case
from lib.empirical import gradient_descent_np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
mpl.use('tkagg')
np.random.seed(1234)

seeds = 100
n_epochs = 1e7
# n: number of training examples
n = 150
# d: number of total dimensions
d = 100
# p: number of fast learning dimensions
p = 50
# standard deviation of the noise added to the teacher output
noise = 0.0
# L2 regularization coefficient
l2 = 0.0


fig = plt.figure(figsize=(13.3, 4))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1])
ax_1 = plt.subplot(gs[0])
ax_2 = plt.subplot(gs[1])


for k, color in zip([30], ['tab:blue', 'tab:orange', 'tab:green']):

    # ts_a, _, _, EG_a_mean_std = gradient_descent_np(seeds, n, d, p, k, noise, l2)
    # EG_a, EG_a_std = EG_a_mean_std
    ts_b, _, _, EG_b = analytical_results_general_case(seeds, n, d, p, k, noise, l2)
    ts_c, _, _, EG_c = analytical_results(1, n, d, p, k, noise, l2)

    # ax_1.plot(ts_a, EG_a, lw=1.5, ls='--', color=color, alpha=0.8)
    # ax_1.errorbar(ts_a[::2], EG_a[::2], yerr=0.5*EG_a_std[::2], fmt='--o',
    #               capthick=2, color=color, alpha=0.8)
    # ax_2.plot(ts_a, EG_a, lw=1.5, ls='--', color=color, alpha=0.8)
    # ax_2.errorbar(ts_a[::2], EG_a[::2], yerr=0.5*EG_a_std[::2], fmt='--o',
    #               capthick=2, color=color, alpha=0.8)

    ax_2.plot(ts_b, EG_b, lw=2.2, color=color, alpha=0.8, ls='--')
    ax_2.plot(ts_c, EG_c, lw=2.2, color=color, alpha=0.8)

    # for legend only
    # ax_1.plot(np.NaN, np.NaN, c=color, label=r'$\kappa={}$'.format(k))
    # ax_2.plot(np.NaN, np.NaN, c=color, label=r'$\kappa={}$'.format(k))

# ax_1.plot(np.NaN, np.NaN, c='k', ls='--', label='Empirical gradient descent')
# ax_1.plot(np.NaN, np.NaN, c='k', label='Exact analytical (Eq. 9)')
ax_2.plot(np.NaN, np.NaN, c='k', ls='--', label='Empirical gradient descent')
ax_2.plot(np.NaN, np.NaN, c='k', label='Approximate analytical (Eq. 12-14)')

# Plotting
# ax_1.set_ylabel('MSE Generalization Error')
ax_2.set_ylabel('MSE Generalization Error')
# ax_1.set_xlabel('Training time')
ax_2.set_xlabel('Training time')
# ax_1.set_title('GD vs exact analytical results')
ax_2.set_title('GD vs approximate analytical results')
# ax_1.set_xlim([1, 1e7])
ax_2.set_xlim([1, 1e7])
# ax_1.set_ylim([0.0, 0.5])
ax_2.set_ylim([0.0, 0.5])
# ax_1.set_xscale('log')
ax_2.set_xscale('log')
# ax_1.legend()
ax_2.legend()
# ax_1.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.3)
ax_2.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.3)
plt.tight_layout()
# plt.savefig('emp_vs_analytic.png', dpi=200)
plt.show()
