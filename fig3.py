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
    # F = S
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

seeds = 100
# n: number of training examples
n = 150
# d: number of total dimensions
d = 100
# p: number of fast learning dimensions
p = 60
# standard deviation of the noise added to the teacher output
noise = 0.0
# L2 regularization coefficient
lambda_ = 0.0

fig = plt.figure(figsize=(13.4*0.85, 4*0.85))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
ax_1 = plt.subplot(gs[0])
ax_2 = plt.subplot(gs[1])

# k: kappa -> the condition number of the modulation matrix, F
for k, color in zip([1, 50, 100000], ['tab:blue', 'tab:orange', 'tab:green']):

    # empirical
    Rs_emp_sgd = np.zeros((seeds, 100))
    Qs_emp_sgd = np.zeros((seeds, 100))
    EGs_emp_sgd = np.zeros((seeds, 100))

    Rs_emp_ridge = np.zeros((seeds, 100))
    Qs_emp_ridge = np.zeros((seeds, 100))
    EGs_emp_ridge = np.zeros((seeds, 100))

    Rs_RMT = np.zeros((seeds, 100))
    Qs_RMT = np.zeros((seeds, 100))

    for seed in tqdm(range(seeds)):
        X, y, X_test, y_test, F, w = get_data(seed, n, d, p, k, noise)
        w_hat = np.zeros((d, 1))

        # eigendecomposition of the input covariance matrix
        XTX = np.dot(X.T, X)
        V, L, _ = np.linalg.svd(XTX)
        # optimal learning rates
        eta = 1.0 / (L[0] + lambda_)

        # getting w_bar using normal equations
        w_bar = np.dot(np.linalg.inv(XTX + lambda_ * np.eye(d)), np.dot(X.T, y))

        ts = []
        for i, j in enumerate(np.linspace(-3, 6, 100)):
            t = (10 ** j)
            ts += [t + 1]

            # Gradient Descent
            w_hat = np.dot(V, np.dot(
                np.eye(d) - np.diag(np.abs((1 - eta * lambda_) - eta * L) ** t),
                np.dot(V.T, w_bar)))

            R, Q = get_RQ(w_hat, F, w, d)
            EG = ((np.dot(X_test, w_hat) - y_test) ** 2).mean()

            Rs_emp_sgd[seed, i] = R
            Qs_emp_sgd[seed, i] = Q
            EGs_emp_sgd[seed, i] = EG

            # Ridge
            w_hat = np.dot(np.linalg.inv(XTX + 2.0 * (lambda_ + 1.0 / t) * np.eye(d)), np.dot(X.T, y))

            R, Q = get_RQ(w_hat, F, w, d)
            EG = ((np.dot(X_test, w_hat) - y_test) ** 2).mean()

            Rs_emp_ridge[seed, i] = R
            Qs_emp_ridge[seed, i] = Q
            EGs_emp_ridge[seed, i] = EG

            R_RMT = 0
            for l in L:
                R_RMT += (1 - np.abs((1 - eta * lambda_) - eta * l) ** t) * l / (l + lambda_)
            Rs_RMT[seed, i] = R_RMT / d

            Ft = np.dot(F, V)
            D = np.eye(d) - np.diag(np.abs((1 - eta * lambda_) - eta * L) ** t)
            J = np.diag(L / (L + lambda_))
            K = np.dot(D, J)

            A = np.dot(Ft, np.dot(K, np.linalg.inv(Ft)))
            B = np.dot(Ft, np.dot(K, np.diag(L**-0.5)))

            Qs_RMT[seed, i] = np.trace(np.dot(A.T, A)) / d + noise ** 2 * np.trace(np.dot(B.T, B)) / d

    # analatycal
    s1 = 0.5
    s2 = 0.5 / k
    ts = 10 ** np.linspace(-3, 6, 100)
    lambdas = 1 / ts + lambda_
    alpha_1 = n / p
    # here
    lambdas_1 = 0.5 * lambdas / ((p / d) * s1 ** 2)
    a1 = 1 + 2 * lambdas_1 / (1 - alpha_1 - lambdas_1 +
                              np.sqrt((1 - alpha_1 - lambdas_1) ** 2 +
                                      4 * lambdas_1))
    alpha_2 = n / (d - p)
    lambdas_2 = lambdas / (((d - p) / d) * s2 ** 2)
    a2 = 1 + 2 * lambdas_2 / (1 - alpha_2 - lambdas_2 +
                              np.sqrt((1 - alpha_2 - lambdas_2) ** 2 +
                                      4 * lambdas_2))

    R1 = (n / d) * 1 / a1
    R2 = (n / d) * 1 / a2

    b1 = alpha_1 / (a1 ** 2 - alpha_1)
    c1 = 1 + noise ** 2 - 2 * R2 - ((2 - a1) / a1) * (n / d)

    b2 = alpha_2 / (a2 ** 2 - alpha_2)
    c2 = 1 + noise ** 2 - 2 * R1 - ((2 - a2) / a2) * (n / d)

    Q1 = (b1 * b2 * c2 + b1 * c1) / (1 - b1 * b2)
    Q2 = (b1 * b2 * c1 + b2 * c2) / (1 - b1 * b2)

    R_an = R1 + R2
    Q_an = Q1 + Q2
    EG_an = 0.5 * (1 - 2 * R_an + Q_an)

    EGs_RMT = 0.5 * (1 - 2 * Rs_RMT + Qs_RMT)

    EGs_RMT_mean = EGs_RMT.mean(0)

    EGs_emp_sgd_mean = 0.5 * EGs_emp_sgd.mean(0)
    EGs_emp_ridge_mean = 0.5 * EGs_emp_ridge.mean(0)
    EGs_emp_sgd_std = 0.25 * EGs_emp_sgd.std(0)
    EGs_emp_ridge_std = 0.25 * EGs_emp_ridge.std(0)

    ax_1.plot(ts + 1, EGs_emp_sgd_mean, lw=2.2, color=color, alpha=0.8, ls='--')
    ax_1.fill_between(ts + 1, EGs_emp_sgd_mean - EGs_emp_sgd_std,
                      EGs_emp_sgd_mean + EGs_emp_sgd_std, color=color, alpha=0.2)
    ax_1.plot(ts + 1, EGs_RMT_mean, lw=2.2, color=color, alpha=0.8)

    ax_2.plot(ts + 1, EGs_emp_ridge_mean, lw=2.2, color=color, alpha=0.8, ls='--')
    ax_2.fill_between(ts + 1, EGs_emp_ridge_mean - EGs_emp_ridge_std,
                      EGs_emp_ridge_mean + EGs_emp_ridge_std, color=color, alpha=0.2)
    ax_2.plot(ts + 1, EG_an, lw=2.2, color=color, alpha=0.8)

    # for legend only
    ax_1.plot(np.NaN, np.NaN, c=color, label=r'$\kappa={}$'.format(k))
    ax_2.plot(np.NaN, np.NaN, c=color, label=r'$\kappa={}$'.format(k))

ax_1.plot(np.NaN, np.NaN, c='k', ls='--', label='Gradient descent')
ax_1.plot(np.NaN, np.NaN, c='k', label='Analytical Eqs. 9,10')
ax_2.plot(np.NaN, np.NaN, c='k', ls='--', label='Ridge regression')
ax_2.plot(np.NaN, np.NaN, c='k', label='Analytical Eqs. 14,16')

ax_1.set_ylabel('MSE Generalization Error')
ax_2.set_ylabel('MSE Generalization Error')
ax_1.set_xlabel(r'Training time $t$')
ax_2.set_xlabel(r'Inverse ridge coefficient $1/\lambda$')
ax_1.set_xlim([1, 1e6])
ax_2.set_xlim([1, 1e6])
ax_1.set_ylim([0.0, 0.5])
ax_2.set_ylim([0.0, 0.5])
ax_1.set_xscale('log')
ax_2.set_xscale('log')
ax_1.legend(loc=1)
ax_2.legend(loc=1)
ax_1.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.3)
ax_2.grid(color='k', linestyle='--', linewidth=0.5, alpha=0.3)
plt.tight_layout()
plt.savefig('expected_results/fig3.png')
# plt.show()
