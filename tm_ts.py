import numpy as np
import utils

BETA = .5
TITLE = 'tm_ts'
N_STEPS = 250
N_ARMS = 5
N_MC_SAMPLES = 10000000
SEED = 14
M = 3
CONFIDENCE_LEVEL = .9


def select_arms_ts(prior):
    theta = prior.draw()
    return utils.select_best_arm_from_theta(theta)


def select_arms_tsus(prior, m):
    theta = prior.draw()
    top_m_arms = utils.select_best_arms_from_theta(theta, m)
    return np.random.choice(top_m_arms)


def run_tm_ts(parameter=None):
    if parameter is None:
        true_theta = np.array([.1, .2, .3, .4, .5])
        parameter = utils.Parameter(N_STEPS, N_ARMS, N_MC_SAMPLES, M,
                                    CONFIDENCE_LEVEL, SEED, true_theta)
    utils.run_experiment(parameter, select_arms_ts)


def run_tm_tsus(parameter=None):
    if parameter is None:
        true_theta = np.array([.1, .2, .3, .4, .5])
        parameter = utils.Parameter(TITLE, N_STEPS, N_ARMS, N_MC_SAMPLES, M,
                                    CONFIDENCE_LEVEL, SEED, true_theta)
    sampler = lambda prior: select_arms_tsus(prior, parameter.m)
    utils.run_experiment(parameter, sampler)


def main():
    print("Running thompson sampling.")
    run_tm_ts()
    print("Running thompson sampling uniform sampling.")
    run_tm_tsus()


if __name__ == "__main__":
    main()
