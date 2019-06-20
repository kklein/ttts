import numpy as np
import utils

BETA = .5
N_STEPS = 500
N_ARMS = 5
N_MC_SAMPLES = 10000000
CONFIDENCE_LEVEL = .9
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


def run_t1_ttts(parameter=None):
    if parameter is None:
        true_theta = np.array([.1, .2, .3, .4, .5])
        # Top 1 sampling implies m = 1.
        parameter = utils.Parameter(N_STEPS, N_ARMS, N_MC_SAMPLES, 1,
                                    CONFIDENCE_LEVEL, SEED, true_theta)
    utils.run_experiment(parameter, select_arm_ttts)

if __name__ == "__main__":
    run_t1_ttts()
