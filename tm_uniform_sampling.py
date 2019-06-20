import numpy as np
import utils

N_STEPS = 250
N_ARMS = 5
N_MC_SAMPLES = 10000000
SEED = 14
M = 2
CONFIDENCE_LEVEL = .9


def select_arms_uniform(prior):
    return np.random.choice(range(prior.n_arms))


def run_tm_uniform(parameter=None):
    if parameter is None:
        true_theta = np.array([.1, .2, .3, .4, .5])
        parameter = utils.Parameter(N_STEPS, N_ARMS, N_MC_SAMPLES, M,
                                    CONFIDENCE_LEVEL, SEED, true_theta)
    utils.run_experiment(parameter, select_arms_uniform)


if __name__ == "__main__":
    run_tm_uniform()
