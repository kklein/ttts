import numpy as np
import utils

BETA = .5
N_STEPS = 250
N_ARMS = 5
N_MC_SAMPLES = 10000000
SEED = 14
M = 3


def select_arms_ts(prior):
    theta = prior.draw()
    return utils.select_best_arm_from_theta(theta)


def select_arms_tsus(prior, m):
    theta = prior.draw()
    top_m_arms = utils.select_best_arms_from_theta(theta, m)
    return np.random.choice(top_m_arms)


def run_tm_ts():
    true_theta = np.array([.1, .2, .3, .4, .5])
    prior = utils.learn(SEED, N_ARMS, N_STEPS, select_arms_ts, true_theta)
    candidates = utils.get_topm_candidates(N_ARMS, M)
    true_best = utils.select_best_arms_from_theta(true_theta, M)
    utils.compute_confidence(prior, N_MC_SAMPLES, candidates,
                             utils.filter_samples_for_arms_vect, true_best)


def run_tm_tsus():
    true_theta = np.array([.1, .2, .3, .4, .5])
    sampler = lambda prior: select_arms_tsus(prior, M)
    prior = utils.learn(SEED, N_ARMS, N_STEPS, sampler, true_theta)
    candidates = utils.get_topm_candidates(N_ARMS, M)
    true_best = utils.select_best_arms_from_theta(true_theta, M)
    utils.compute_confidence(prior, N_MC_SAMPLES, candidates,
                             utils.filter_samples_for_arms_vect, true_best)


def main():
    print("Running thompson sampling.")
    run_tm_ts()
    print("Running thompson sampling uniform sampling.")
    # run_tm_tsus()


if __name__ == "__main__":
    main()
