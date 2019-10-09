import numpy as np
import utils as utils

class State:
    def __init__(self, true_means, m):
        self.true_means = true_means
        self.m = m
        self.k = len(self.true_means)
        self.means = np.zeros(self.k)
        self.counts = np.zeros(self.k)
        self.betas = np.zeros(self.k)
        # 'lower bound': lowest of currently top m arms
        self.lb_index = 0
        # 'upper bound': highest of currently non-top m arms
        self.ub_index = 0

    def update_mean(self, index, reward):
        count = self.counts[index]
        mean = self.means[index]
        self.means[index] = reward / count + (count  - 1) * mean / count

    def sample_and_update(self, index):
        reward = np.random.binomial(1, self.true_means[index])
        self.counts[index] += 1
        self.update_mean(index, reward)

    def compute_parameters(self, t, delta):
        compute_beta = lambda count: np.sqrt(np.log(1.25 * self.k * t**4 / delta) / (2 * count))
        self.betas = [compute_beta(count) for count in self.counts]
        high_arms = utils.select_best_arms_from_theta(self.means, self.m)
        low_arms = [l for l in range(self.k) if l not in high_arms]
        # Minimizing f equals maximizing -f.
        theta = [-self.means[index] + self.betas[index] for index in high_arms]
        self.lb_index = high_arms[utils.select_best_arm_from_theta(theta)]
        theta = [self.means[index] + self.betas[index] for index in low_arms]
        self.ub_index = low_arms[utils.select_best_arm_from_theta(theta)]

    def compute_stopping_criterion(self):
        return (self.means[self.lb_index] + self.betas[self.lb_index]
                - self.means[self.ub_index] + self.betas[self.ub_index])

def main(delta, epsilon):
    true_means = [.9, .8, .7, .6, .5, .4, .3, .2, .1]
    state = State(true_means, 4)

    t = 1

    for arm_index in range(len(true_means)):
        state.sample_and_update(arm_index)

    state.compute_parameters(t, delta)

    criterion = 1

    while criterion >= epsilon:
        state.sample_and_update(state.lb_index)
        state.sample_and_update(state.ub_index)
        t += 1
        state.compute_parameters(t, delta)
        criterion = state.compute_stopping_criterion()
        print(f"Chose {state.lb_index} and {state.ub_index}")
        print(f"{t} rounds: {criterion}")

    n_samples = len(true_means) + 2 * t
    print(f"Took {n_samples}.")
    print(state.counts)

if __name__ == "__main__":
    delta = .05
    epsilon = 0.2
    main(delta, epsilon)
