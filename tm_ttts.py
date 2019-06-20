import numpy as np
import utils

BETA = .5
N_STEPS = 250
N_ARMS = 5
N_MC_SAMPLES = 10000000
SEED = 14
M = 3
CONFIDENCE_LEVEL = .9


def select_arms_ttts(prior, m):
    theta = prior.draw()
    top_m_arms = utils.select_best_arms_from_theta(theta, m)
    new_top_m_arms = top_m_arms
    while utils.are_set_equal(new_top_m_arms, top_m_arms):
        new_theta = prior.draw()
        new_top_m_arms = utils.select_best_arms_from_theta(new_theta, m)
    edge_arms = np.setxor1d(top_m_arms, new_top_m_arms)
    return np.random.choice(edge_arms)


def run_tm_ttts(parameter=None):
    if parameter is None:
        true_theta = np.array([.1, .2, .3, .4, .5])
        parameter = utils.Parameter(N_STEPS, N_ARMS, N_MC_SAMPLES, M,
                                    CONFIDENCE_LEVEL, SEED, true_theta)
    sampler = lambda prior: select_arms_ttts(prior, parameter.m)
    utils.run_experiment(parameter, sampler)


if __name__ == "__main__":
    run_tm_ttts()
