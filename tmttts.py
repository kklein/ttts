import itertools
import numpy as np
import utils

BETA = .5
N_STEPS = 750
N_ARMS = 5
N_MC_SAMPLES = 10000000
SEED = 12
M = 2


def select_best_arms_from_theta(theta):
    return theta.argsort()[-M:][::-1]


def are_set_equal(array_a, array_b):
    return np.sum(np.in1d(array_a, array_b)) == len(array_a)


def select_arms_ttts(prior):
    theta = prior.draw()
    top_m_arms = select_best_arms_from_theta(theta)

    new_top_m_arms = top_m_arms
    while are_set_equal(new_top_m_arms, top_m_arms):
        new_theta = prior.draw()
        new_top_m_arms = select_best_arms_from_theta(new_theta)

    edge_arms = np.setxor1d(top_m_arms, new_top_m_arms)

    return np.random.choice(edge_arms)


def filter_samples_m(samples, arm_combination):
    arm_filter = np.apply_along_axis(
        lambda x: are_set_equal(x, arm_combination), 1, samples.argsort(axis=1)[:, -M:])
    return samples[arm_filter, :]


def filter_samples_m_vect(samples, arm_combination):
    arm_filter = np.zeros(N_MC_SAMPLES)
    best = samples.argsort(axis=1)[:, -M:]
    for permutation in set(itertools.permutations(arm_combination)):
        aux = np.prod(best == np.array(permutation), axis=1)
        arm_filter += aux
    return samples[arm_filter > 0, :]


def main():
    true_theta = np.array([.1, .2, .3, .4, .5])
    prior = utils.learn(SEED, N_ARMS, N_STEPS, select_arms_ttts, true_theta)
    # All combinatorial possibilities to choose m fron {0, ..., N_ARMS-1}.
    candidates = list(itertools.combinations(range(N_ARMS), M))
    true_best = select_best_arms_from_theta(true_theta)
    utils.compute_confidence(prior, N_MC_SAMPLES, candidates,
                             filter_samples_m_vect, true_best)


if __name__ == "__main__":
    main()
