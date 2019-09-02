import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import utils as utils

# TODO: Read true mean and m from log.
means_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
m = 4
df = pd.read_csv('arms.csv', sep='|', header=None)
empirical_means = []
frequencies = []

for cell in df[2]:
    empirical_means.append(np.fromstring(cell[1: -1], sep=' '))

for cell in df[4]:
    frequencies.append(np.fromstring(cell[1: -1], sep=' '))

tuples = zip(empirical_means, frequencies)

def kl_divergence(x, theta):
    return np.log(theta / x) * theta + np.log((1 - theta) / (1 - x)) * (1 - theta)

def compute_coefficient(x, theta_i, theta_j, psi_i, psi_j):
    return psi_i * kl_divergence(x, theta_i) + psi_j * kl_divergence(x, theta_j)

def analytical_min(theta_i, theta_j, psi_i, psi_j):
    return (psi_i * theta_i + psi_j * theta_j) / (psi_i + psi_j)

def compute_cross_coefficients(means, frequencies):
    top_indeces = utils.select_best_arms_from_theta(means, m)
    not_top_indeces = [i for i in range(len(means)) if i not in top_indeces]
    coefficients = {}
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

def compute_coefficients(means, fequencies):
    top_indeces = utils.select_best_arms_from_theta(means, m)
    not_top_indeces = [i for i in range(len(means)) if i not in top_indeces]
    coefficients = np.zeros(len(means))
    for i in not_top_indeces:
        psi_i = frequencies[i]
        theta_i = means[i]
        min = 100000
        for j in top_indeces:
            psi_j = frequencies[j]
            theta_j = means[j]
            x_hat = analytical_min(theta_i, theta_j, psi_i, psi_j)
            coefficient = compute_coefficient(x_hat, theta_i, theta_j, psi_i, psi_j)

            if coefficient < min:
                min = coefficient

        coefficients[i] = min
    i = 2
    psi_i = frequencies[i]
    theta_i = means[i]
    for j in top_indeces:
        psi_j = frequencies[j]
        theta_j = means[j]
        x_hat = analytical_min(theta_i, theta_j, psi_i, psi_j)
        coefficients[j] = compute_coefficient(x_hat, theta_i, theta_j, psi_i, psi_j)
    return coefficients

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
sns.boxplot(x='cross_keys', y='cross_coefficients_true', data=cross_df, ax=ax[2, 0])
sns.boxplot(x='cross_keys', y='cross_coefficients_estimated', data=cross_df, ax=ax[2, 1])


fig.suptitle("Coefficients on the lhs are computed with the true means. \n Coefficients on the rhs are computed with the relative empirical means. (see 2nd row)\n The coefficients in 4th row corresponds to Johannes' suggestion, calcuting all cross terms. \n The experiment has been repeated 100 times with 1700 steps.")
plt.show()
