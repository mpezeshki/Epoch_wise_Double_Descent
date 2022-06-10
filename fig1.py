import numpy as np
from lib.analytical import analytical_results
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
np.random.seed(1234)

# n: number of training examples
n = 130
# d: number of total dimensions
d = 100
# p: number of fast learning dimensions
p = 70
# k: kappa -> the condition number of the modulation matrix, F
k = 100
# standard deviation of the noise added to the teacher output
noise = 0.0
# L2 regularization coefficient
l2 = 0.0

fig = plt.figure(figsize=(7, 7))
ax_1 = plt.subplot(2, 1, 1)
ax_2 = plt.subplot(2, 1, 2)

# Only the fast feature
ts, _, _, EG_fast_only = analytical_results(
    1, n, d, p, k, noise, l2, fast_features_only=True)
ax_1.plot(ts, EG_fast_only, color='#0076BA', lw=2.2, label='Fast feature')

# Only the slow feature
ts, _, _, EG_slow_only = analytical_results(
    1, n, d, p, k, noise, l2, slow_features_only=True)
ax_1.plot(ts, EG_slow_only, color='#FF9300', lw=2.2, label='Slow feature')

# Both features
ts, _, _, EG = analytical_results(1, n, d, p, k, noise, l2)
ax_2.plot(ts, EG, color='#353535', lw=2.2, label='Both features')

# Plotting
ax_1.set_ylabel('MSE Generalization Error')
ax_2.set_ylabel('MSE Generalization Error')
ax_2.set_xlabel('Training time')
ax_1.set_xlim([1, 1e5])
ax_2.set_xlim([1, 1e5])
ax_1.set_ylim([0.0, 0.5])
ax_2.set_ylim([0.0, 0.5])
ax_1.set_xscale('log')
ax_2.set_xscale('log')
ax_1.legend()
ax_2.legend()
ax_1.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.3)
ax_2.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.3)
plt.savefig('expected_results/fig1.png', dpi=200)
# plt.show()
