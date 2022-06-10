import os
import numpy as np
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

path = './ResNet_experiments/'
# each experiment is repeated 5 times -> a total of 500 experiments
list_seeds = range(5)
# the first 8 experiments were unstable so we do not load them
list_log_lambda = np.linspace(0.1, 8, 100)[8:]
epochs = 1001

if os.path.exists(path + 'resnet-test-errors_mean.npy'):

    print('Reading from aggregated files...')
    train_errs = np.load(path + 'resnet-train-errors_mean.npy')
    test_errs = np.load(path + 'resnet-test-errors_mean.npy')
    train_errs_std = np.load(path + 'resnet-train-errors_std.npy')
    test_errs_std = np.load(path + 'resnet-test-errors_std.npy')

else:

    print('Reading from individual files...')
    train_errs = np.zeros((len(list_seeds), len(list_log_lambda), epochs))
    test_errs = np.zeros((len(list_seeds), len(list_log_lambda), epochs))

    for i, log_lambda in enumerate(list_log_lambda):
        for j, seed in enumerate(list_seeds):
            name = 'log_lambda_' + str(log_lambda) + '_seed_' + str(seed) + '.npy'
            try:
                results = np.load(path + '/individual_runs/' + name)
            except:
                print('failed: experiment is not available.')

            train_errs[j, i] = results[0]
            test_errs[j, i] = results[1]

    np.save(path + 'resnet-train-errors_std.npy', train_errs.std(0).T)
    np.save(path + 'resnet-test-errors_std.npy', test_errs.std(0).T)
    # average over seeds
    train_errs = train_errs.mean(0)
    test_errs = test_errs.mean(0)
    np.save(path + 'resnet-train-errors_mean.npy', train_errs.T)
    np.save(path + 'resnet-test-errors_mean.npy', test_errs.T)

to_plot = test_errs
# to_plot = train_errs

fig = plt.figure(figsize=(13.3, 4))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1.3])
ax_1 = plt.subplot(gs[0])
ax_2 = plt.subplot(gs[1])
epochs = to_plot.shape[0]
f1 = ax_1.pcolor(list_log_lambda, np.arange(1, epochs + 1),
                 to_plot, cmap=cm, vmin=0.18, vmax=0.5, shading='auto')
fig.colorbar(f1, ax=ax_1, orientation='vertical', pad=0.02)

ax_1.set_ylim([1, 1000])
ax_1.set_yscale('log')

ticks = [1, 2, 3, 4, 5, 6, 7, 8]
labels = [r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$',
          r'$10^{-4}$', r'$10^{-5}$', r'$10^{-6}$', r'$10^{-7}$', r'$\approx 0.0$']
ax_1.set_xticks(ticks)
ax_1.set_xticklabels(labels)

ax_2.plot(to_plot[:, 10], label=r'Large $\lambda$', lw=2.0)
ax_2.plot(to_plot[:, 17], label=r'Intermediate $\lambda$', lw=2.0)
ax_2.plot(to_plot[:, -1], label=r'Small $\lambda$', lw=2.0)
ax_2.set_xlim([1, 1000])
ax_2.set_ylim([0.18, 0.5])

ax_1.set_xlabel(r'Regularization strength $\lambda$')
ax_1.set_ylabel(r'Training time t')

ax_2.set_xlabel(r'Training time t')
ax_2.set_ylabel(r'0-1 generalization loss', labelpad=20)

ax_2.set_xscale('log')
ax_2.legend()
ax_2.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.3)
plt.tight_layout()
plt.savefig('expected_results/fig5_ab.png')
# plt.show()
