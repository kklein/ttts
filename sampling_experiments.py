import sys
import numpy as np
import utils

BETA = .5
N_STEPS = 2000
TITLE = 'title'
N_MC_SAMPLES = 100000
SEED = 14
M = 4

def select_arms_uniform(prior):
    return np.random.choice(range(prior.n_arms))


def run_tm_uniform(parameter):
    utils.run_experiment(parameter, select_arms_uniform)


def select_arms_ttts(prior, m):
    theta = prior.draw()
    top_m_arms = utils.select_best_arms_from_theta(theta, m)
    new_top_m_arms = top_m_arms
    while utils.are_set_equal(new_top_m_arms, top_m_arms):
        new_theta = prior.draw()
        new_top_m_arms = utils.select_best_arms_from_theta(new_theta, m)
    edge_arms = np.setxor1d(top_m_arms, new_top_m_arms)
    return np.random.choice(edge_arms)


def run_tm_ttts(parameter):
    sampler = lambda prior: select_arms_ttts(prior, parameter.m)
    utils.run_experiment(parameter, sampler)


def select_arms_ts(prior):
    theta = prior.draw()
    return utils.select_best_arm_from_theta(theta)


def select_arms_tsus(prior, m):
    theta = prior.draw()
    top_m_arms = utils.select_best_arms_from_theta(theta, m)
    return np.random.choice(top_m_arms)


def run_tm_ts(parameter):
    utils.run_experiment(parameter, select_arms_ts)


def run_tm_tsus(parameter):
    sampler = lambda prior: select_arms_tsus(prior, parameter.m)
    utils.run_experiment(parameter, sampler)


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


def run_t1_ttts(parameter):
    utils.run_experiment(parameter, select_arm_ttts)


if __name__ == "__main__":
    seed = int(sys.argv[1])
    true_theta = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9])
    n_arms = len(true_theta)
    n_steps = 1000
    confidence_level = 1
    control_interval = 100

    methods = [
        {'title': 'tm_ttts', 'run_method': run_tm_ttts},
        {'title': 'tm_tsus', 'run_method': run_tm_tsus},
        {'title': 'tm_uniform', 'run_method': run_tm_uniform},
        {'title': 'tm_ts', 'run_method': run_tm_ts}
    ]

    for method in methods:
        parameter = utils.Parameter(
            method['title'], n_steps, n_arms, N_MC_SAMPLES, M,
            confidence_level, control_interval, seed, true_theta)
        method['run_method'](parameter)
        print(f"Finished method {method['title']}")
