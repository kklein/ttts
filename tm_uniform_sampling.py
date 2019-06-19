import numpy as np
import utils

N_STEPS = 250
N_ARMS = 5
N_MC_SAMPLES = 10000000
SEED = 14
M = 2

def select_arms_uniform(prior):
    return np.random.choice(range(prior.n_arms))


def main():
    true_theta = np.array([.1, .2, .3, .4, .5])
    prior = utils.learn(SEED, N_ARMS, N_STEPS, select_arms_uniform,
                        true_theta)
    candidates = utils.get_topm_candidates(N_ARMS, M)
    true_best = utils.select_best_arms_from_theta(true_theta, M)
    utils.compute_confidence(prior, N_MC_SAMPLES, candidates,
                             utils.filter_samples_for_arms_vect, true_best)


if __name__ == "__main__":
    main()
