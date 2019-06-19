import numpy as np
import utils

BETA = .5
N_STEPS = 500
N_ARMS = 5
N_MC_SAMPLES = 10000000
SEED = 12


def select_arm_ttts(prior):
    theta = prior.draw()
    selected_arm = utils.select_best_arm_from_theta(theta)

    # Resample in 1-BETA of the cases.
    if np.random.uniform(0, 1) > BETA:
        new_selected_arm = selected_arm
        while new_selected_arm == selected_arm:
            new_theta = prior.draw()
            new_selected_arm = utils.select_best_arm_from_theta(new_theta)
        selected_arm = new_selected_arm

    return selected_arm


def main():
    true_theta = np.array([.1, .2, .3, .4, .5])
    prior = utils.learn(SEED, N_ARMS, N_STEPS, select_arm_ttts, true_theta)
    candidates = range(N_ARMS)
    true_best = utils.select_best_arm_from_theta(true_theta)
    utils.compute_confidence(prior, N_MC_SAMPLES, candidates,
                             utils.filter_samples_for_arm, true_best)


if __name__ == "__main__":
    main()
