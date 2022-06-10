import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
from scipy.stats import ortho_group
from tqdm import tqdm


def get_modulation_matrix(d, p, k):
    U = ortho_group.rvs(d)
    VT = ortho_group.rvs(d)
    S = np.eye(d)
    S[:p, :p] *= 1
    S[p:, p:] *= 1 / k
    F = np.dot(U, np.dot(S, VT))
    return F


# Implements the teacher and generates the data
def get_data(seed, n, d, p, k, noise):
    np.random.seed(seed)
    Z = np.random.randn(n, d) / np.sqrt(d)
    Z_test = np.random.randn(1000, d) / np.sqrt(d)

    # teacher
    w = np.random.randn(d, 1)
    y = np.dot(Z, w)
    y = y + noise * np.random.randn(*y.shape)
    # test data is noiseless
    y_test = np.dot(Z_test, w)

    # the modulation matrix that controls students access to the data
    F = get_modulation_matrix(d, p, k)

    # X = F^T Z
    X = np.dot(Z, F)
    X_test = np.dot(Z_test, F)

    return X, y, X_test, y_test, F, w


def get_RQ(w_hat, F, w, d):
    # R: the alignment between the teacher and the student
    R = np.dot(np.dot(F, w_hat).T, w).item() / d
    # Q: the student's modulated norm
    Q = np.dot(np.dot(F, w_hat).T, np.dot(F, w_hat)).item() / d
    return R, Q

path = './ResNet_experiments/'
seeds = 100
# n: number of training examples
n = 120
# d: number of total dimensions
d = 100
# p: number of fast learning dimensions
p = 70
# standard deviation of the noise added to the teacher output
noise = 0.0
# L2 regularization coefficient
lambda_ = 0.0

fig = plt.figure(figsize=(13.4 *0.85, 4*0.85))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
ax_1 = plt.subplot(gs[0])
ax_2 = plt.subplot(gs[1])

epochs = 100000
k = 30

# regularization coeff
alpha_init = 0.6

EGs = np.zeros((seeds, epochs))
EGs_reg = np.zeros((seeds, epochs))
for seed in tqdm(range(seeds)):
    X, y, X_test, y_test, F, w = get_data(seed, n, d, p, k, noise)
    w_hat = np.zeros((d, 1))
    # regularized w_hat
    w_hat_reg = np.zeros((d, 1))

    XTy = np.dot(X.T, y)
    # eigendecomposition of the input covariance matrix
    XTX = np.dot(X.T, X)
    V, L, _ = np.linalg.svd(XTX)
    # optimal learning rates
    eta = 1.0 / (L[0] + lambda_)

    for i in range(epochs):
        EG = ((np.dot(X_test, w_hat) - y_test) ** 2).mean()
        EGs[seed, i] = EG
        w_hat = w_hat - eta * (np.dot(XTX, w_hat) - XTy)

        # regularization coeff scheduling
        alpha = alpha_init * np.log(1 + np.exp(-0.001 * i)) / np.log(2)
        EG_reg = ((np.dot(X_test, w_hat_reg) - y_test) ** 2).mean()
        EGs_reg[seed, i] = EG_reg
        w_hat_reg = w_hat_reg - eta * (np.dot((1 + alpha) * XTX, w_hat_reg) - XTy)

ax_2.set_title('ResNet-18 on Cifar-10')
ax_2.plot(np.load(path + 'wo_reg.npy'), label='w\\o regularization', lw=2.2)
ax_2.plot(np.load(path + 'w_reg.npy'), label='w\\ regularization', lw=2.2)
ax_2.set_xscale('log')
ax_2.set_ylabel('0-1 Generalization Error')
ax_2.set_xlabel('Training time')
ax_2.set_xlim([1, 1e3])
ax_2.set_ylim([0.12, 0.35])
ax_2.legend(loc=1)
ax_2.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.3)

ax_1.set_title('Linear teacher-student model')
ax_1.plot(0.5 * EGs.mean(0), label='w\\o regularization', lw=2.2)
ax_1.plot(0.5 * EGs_reg.mean(0), label='w\\ regularization', lw=2.2)
ax_1.set_xscale('log')
ax_1.set_ylabel('MSE Generalization Error')
ax_1.set_xlabel('Training time')
ax_1.set_xlim([1, 1e5])
ax_1.set_ylim([0.0, 0.5])
ax_1.legend(loc=1)
ax_1.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.3)

plt.tight_layout()
plt.savefig('expected_results/fig6.png')
# plt.show()
