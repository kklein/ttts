import numpy as np
import utils

BETA = .5
N_STEPS = 500
N_ARMS = 5
N_MC_SAMPLES = 10000000
SEED = 12


def select_arm_from_theta(theta):
    return np.argmax(theta)


def select_arm_ttts(prior):
    theta = prior.draw()
    selected_arm = select_arm_from_theta(theta)

    # Resample in 1-BETA of the cases.
    if np.random.uniform(0, 1) > BETA:
        new_selected_arm = selected_arm
        while new_selected_arm == selected_arm:
            new_theta = prior.draw()
            new_selected_arm = select_arm_from_theta(new_theta)
        selected_arm = new_selected_arm

    return selected_arm


def filter_samples(samples, arm_index):
    arm_filter = np.max(samples, axis=1) == samples[:, arm_index]
    return samples[arm_filter, :]


def main():
    true_theta = np.array([.1, .2, .3, .4, .5])
    prior = utils.learn(SEED, N_ARMS, N_STEPS, select_arm_ttts, true_theta)

    candidates = range(N_ARMS)
    true_best = select_arm_from_theta(true_theta)
    utils.compute_confidence(prior, N_MC_SAMPLES, candidates,
                             filter_samples, true_best)


if __name__ == "__main__":
    main()
