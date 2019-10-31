import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from scipy.optimize import least_squares
import utils as utils

# In this whole script 'p' stands for the measurement plan, a.k.a. allocation
# policy. This is a probability over the set of arms.

class ExecutionParameter:
    def __init__(self, theta, m, is_constrained):
        self.theta = theta
        self.m = m
        self.is_constrained = is_constrained
        self.optimal_arms = utils.select_best_arms_from_theta(theta, m)
        self.suboptimal_arms = utils.select_suboptimal_arms_from_theta(
            theta, m)


def analytical_min(psi_i, psi_j, theta_i, theta_j,):
    return (psi_i * theta_i + psi_j * theta_j) / (psi_i + psi_j)


def kl(p_i, p_j, theta_i, theta_j):
    x = analytical_min(p_i, p_j, theta_i, theta_j)
    return ((math.log(theta_i / x)) * theta_i
            + math.log((1 - theta_i) / (1 - x)) * (1 - theta_i))


def get_cross_constraint(i1, j1, i2, j2, p, theta):
    return (p[i1] * kl(p[i1], p[j1], theta[i1], theta[j1])
            + p[j1] * kl(p[j1], p[i1], theta[j1], theta[i1])
            - p[i2] * kl(p[i2], p[j2], theta[i2], theta[j2])
            - p[j2] * kl(p[j2], p[i2], theta[j2], theta[i2]))


def get_equations(p, execution_parameter):
    # Measurement plan is a probability and has to add up to 1.
    equations = [p.sum() - 1]
    # Constrained otimizaiton: Put 1/2 of effort on true best arms.
    if execution_parameter.is_constrained:
        equations += [p[:5].sum() - .5, p[5:].sum() - .5]
    # Add constraint for every tuple (i1, i2, j1, j2) from S*^c x S*^c x S* x
    # S*
    for i1 in execution_parameter.suboptimal_arms:
        for i2 in execution_parameter.suboptimal_arms:
            for j1 in execution_parameter.optimal_arms:
                for j2 in execution_parameter.optimal_arms:
                    if i1 == i2 and j1 == j2:
                        continue
                    equations += [
                        get_cross_constraint(
                            i1, j1, i2, j2, p, execution_parameter.theta)]
    return tuple(equations)


def compute_coefficients(p, execution_parameter):
    indices = []
    values = []
    theta = execution_parameter.theta
    for i1 in execution_parameter.suboptimal_arms:
        for j1 in execution_parameter.optimal_arms:
            v = (p[i1] * kl(p[i1], p[j1], theta[i1], theta[j1])
                 + p[j1] * kl(p[j1], p[i1], theta[j1], theta[i1]))
            indices += [f"({i1}, {j1})"]
            values += [v]
    return(indices, values)


def main():
    m1 = 4
    m2 = 4
    theta1 = np.array([i / 10 for i in range(1, 10)])
    theta2 = np.array([.4 + (i * .2) / 8 for i in range(0, 9)])
    run_parameters = [[0, theta1, m1], [1, theta2, m2]]

    # Produce plots per (theta, m) pair in a row.
    n_runs = len(run_parameters)
    col_titles = ['unconstrained', 'constrained', 'zoomed constrained']
    row_titles = ['theta_' + str(i) for i in range(1, n_runs + 1)]
    # row_titles = ["1", "2"]
    _, ax_allocation = plt.subplots(n_runs, len(col_titles), tight_layout=True)
    _, ax_coefficients = plt.subplots(n_runs, n_runs, tight_layout=True)

    for ax, title in zip(ax_allocation[0], col_titles):
        ax.set_title(title, size='large')
    for ax, title in zip(ax_allocation[:,0], row_titles):
        ax.set_ylabel(r'$\{}$'.format(title), rotation=0, size='large')
    for ax, title in zip(ax_coefficients[0], col_titles[:-1]):
        ax.set_title(title, size='large')
    for ax, title in zip(ax_coefficients[:,0], row_titles):
        ax.set_ylabel(r'$\{}$'.format(title), rotation=0, size='large')

    for index, theta, m in run_parameters:
        for is_constrained in [True, False]:
            execution_parameter = ExecutionParameter(theta, m, is_constrained)
            n_arms = len(execution_parameter.theta)
            # Use uniform allocation as guess.
            guess = [1 / n_arms] * n_arms
            def value_function(x): return get_equations(x, execution_parameter)
            result = least_squares(value_function, guess, loss='soft_l1',
                                   bounds=([np.nextafter(0, 1)] * n_arms,
                                           [1] * n_arms))
            
            print(result.x)
            print(result.cost)

            row_index = index
            col_index = 0 if is_constrained else 1
            ax_allocation[row_index, col_index].scatter(
                range(n_arms), result.x)
            indices, values = compute_coefficients(result.x, execution_parameter)
            ax_coefficients[row_index, col_index].scatter(
                x=indices, y=values)

            # Zoom into plot for unconstrained optimization.
            if not execution_parameter.is_constrained:
                print(result.x[execution_parameter.optimal_arms])
                top_value = result.x[execution_parameter.optimal_arms[-1]] * 1.2
                ax_allocation[row_index, col_index + 1].scatter(
                    range(n_arms), result.x)
                ax_allocation[row_index, col_index + 1].set_ylim(bottom=0, top=top_value)

    plt.show()


if __name__ == "__main__":
    # Enable Tex rendering for matplotlib.
    rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex = True)
    main()
