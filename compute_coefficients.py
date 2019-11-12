import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import utils as utils

def kl_divergence(x, theta):
    return (np.log(theta / x) * theta
            + np.log((1 - theta) / (1 - x)) * (1 - theta))

def compute_coefficient(x, theta_i, theta_j, psi_i, psi_j):
    return psi_i * kl_divergence(x, theta_i) + psi_j * kl_divergence(x, theta_j)

def analytical_min(theta_i, theta_j, psi_i, psi_j):
    return (psi_i * theta_i + psi_j * theta_j) / (psi_i + psi_j)

# Note that this method distinguished top and not-top arms based on empirical
# means. Hence top_indeces will not necessarily equal S*, based on true means.
def compute_cross_coefficients(means, frequencies):
    top_indeces = utils.select_best_arms_from_theta(means, m)
    not_top_indeces = [i for i in range(len(means)) if i not in top_indeces]
    coefficients = {}

    # Those are the 'true cross-coefficients': between optimal and sub-optimal
    # sets.
    for j in top_indeces:
        psi_j = frequencies[j]
        theta_j = means[j]
        for i in not_top_indeces:
            psi_i = frequencies[i]
            theta_i = means[i]
            x_hat = analytical_min(theta_i, theta_j, psi_i, psi_j)
            coefficients[(j, i)] = compute_coefficient(
                x_hat, theta_i, theta_j, psi_i, psi_j)
    return coefficients

# TODO: Read true mean and m from log.
means_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
m = 4
df = pd.read_csv('logs/arms_2000.csv', sep='|', header=None)
empirical_means = []
frequencies = []

for cell in df[2]:
    empirical_means.append(np.fromstring(cell[1: -1], sep=' '))

for cell in df[4]:
    frequencies.append(np.fromstring(cell[1: -1], sep=' '))

tuples = zip(empirical_means, frequencies)

cross_df = pd.DataFrame()
df_ind = pd.DataFrame()
for (mean, frequency) in tuples:
    cross_coefficients_estimated = compute_cross_coefficients(mean, frequency)
    cross_coefficients_true = compute_cross_coefficients(means_true, frequency)

    df_ind = df_ind.append(pd.DataFrame({
        'arm_index': range(9),
        'means_estimated': mean,
        'frequencies': frequency
    }))

    cross_df = cross_df.append(pd.DataFrame({
        'cross_keys': list(cross_coefficients_estimated.keys()),
        'cross_coefficients_estimated': list(cross_coefficients_estimated.values()),
        'cross_coefficients_true': list(cross_coefficients_true.values())
    }))

df_ground = pd.DataFrame({'arm_index': range(9),
                          'means_true': means_true})

fig, ax = plt.subplots(3, 2)
sns.barplot(x='arm_index', y='frequencies', data=df_ind, ax=ax[0, 1])
sns.barplot(x='arm_index', y='means_true', data=df_ground, ax=ax[1, 0])
sns.boxplot(x='arm_index', y='means_estimated', data=df_ind, ax=ax[1, 1])
sns.boxplot(x='cross_keys', y='cross_coefficients_true', data=cross_df,
            ax=ax[2, 0])
sns.boxplot(x='cross_keys', y='cross_coefficients_estimated', data=cross_df,
            ax=ax[2, 1])

fig.suptitle("""Coefficients on the lhs are computed with the true means. \n
             Coefficients on the rhs are computed with the relative empirical
             means. (see 2nd row)\n The coefficients in the 3rd row correspond
             to Johannes' suggestion, calcuting all cross terms. \n The
             experiment has been repeated 148 times with 2000 steps.""")
plt.show()
