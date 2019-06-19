import numpy as np
import utils

BETA = .5
N_STEPS = 250
N_ARMS = 5
N_MC_SAMPLES = 10000000
SEED = 14
M = 3


def select_arms_ttts(prior, m):
    theta = prior.draw()
    top_m_arms = utils.select_best_arms_from_theta(theta, m)
    new_top_m_arms = top_m_arms
    while utils.are_set_equal(new_top_m_arms, top_m_arms):
        new_theta = prior.draw()
        new_top_m_arms = utils.select_best_arms_from_theta(new_theta, m)
    edge_arms = np.setxor1d(top_m_arms, new_top_m_arms)
    return np.random.choice(edge_arms)


def main():
    true_theta = np.array([.1, .2, .3, .4, .5])
    sampler = lambda prior: select_arms_ttts(prior, M)
    prior = utils.learn(SEED, N_ARMS, N_STEPS, sampler, true_theta)
    candidates = utils.get_topm_candidates(N_ARMS, M)
    true_best = utils.select_best_arms_from_theta(true_theta, M)
    utils.compute_confidence(prior, N_MC_SAMPLES, candidates,
                             utils.filter_samples_for_arms_vect, true_best)


if __name__ == "__main__":
    main()
