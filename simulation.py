import numpy as np
import pymc3 as pm

BETA = .5
N_STEPS = 500
N_ARMS = 5
N_MC_SAMPLES = 1000000
SEED = 12

class Prior():
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.prior = pm.Beta.dist(alpha=np.ones(n_arms), beta=np.ones(n_arms))
        # Turn parameters, Tensor variables/constants, into numpy arrays as to
        # allow for incrementation via '+='.
        self.prior.alpha = self.prior.alpha.eval()
        self.prior.beta = self.prior.beta.eval()

    def logp(self, theta):
        theta = theta.reshape((1, self.n_arms))
        return self.prior.logp(theta).eval()

    def draw(self):
        return self.prior.random()

    def select_arm(self):
        theta = self.draw()
        return select_arm_from_theta(theta)

    def update(self, arm_index, reward):
        self.prior.alpha[arm_index] += reward
        self.prior.beta[arm_index] += 1 - reward

    @property
    def alpha(self):
        return self.prior.alpha

    @property
    def beta(self):
        return self.prior.beta


def select_arm_from_theta(theta):
    return np.argmax(theta)

def print_approximated_theta(prior):
    print(f"Theta: {prior.alpha / (prior.alpha + prior.beta)}")

def main():
    np.random.seed(SEED)
    # Define priors per arm.
    prior = Prior(N_ARMS)
    arm_selection_counts = np.zeros(N_ARMS)

    # Draw true theta.
    # true_theta = np.random.uniform(0, 1, N_ARMS)
    true_theta = np.array([.1, .2, .3, .4, .5])
    true_best_arm = select_arm_from_theta(true_theta)

    for step_index in range(N_STEPS):
        selected_arm = prior.select_arm()

        # Resample in 1-BETA of the cases.
        if np.random.uniform(0, 1) > BETA:
            new_selected_arm = selected_arm
            while new_selected_arm == selected_arm:
                new_selected_arm = prior.select_arm()
            selected_arm = new_selected_arm

        # Play option/arm.
        reward = np.random.binomial(1, true_theta[selected_arm])

        # Update priors.
        prior.update(selected_arm, reward)
        arm_selection_counts[selected_arm] += 1

    print(f"true theta: {true_theta}")
    print_approximated_theta(prior)
    print(f"Arm selections: {arm_selection_counts}")
    print(f"#successes: {np.sum(prior.alpha) - N_ARMS}")
    print(f"#failures: {np.sum(prior.beta) - N_ARMS}")
    assert np.sum(prior.alpha) + np.sum(prior.beta) - 2 * N_ARMS == N_STEPS

    max_confidence = -100000
    best_arm = -1

    # Compute confidence via Monte Carlo integration.
    samples = np.random.uniform(0, 1, (N_MC_SAMPLES, N_ARMS))
    for arm_index in range(N_ARMS):
        arm_filter = np.max(samples, axis=1) == samples[:, arm_index]
        filtered_samples = samples[arm_filter, :]
        print(f"{filtered_samples.shape[0]} samples for arm {arm_index}.")
        if filtered_samples.shape[0] == 0:
            continue
        probabilities = prior.prior.logp(filtered_samples).eval()
        # The volume of each parameter space THETA_I is 1^N_ARMS / N_ARMS.
        # The density of theta is the product of the densities of each theta_i.
        confidence = np.mean(np.prod(np.exp(probabilities), axis=1)) / N_ARMS

        print(f"Arm {arm_index}: confidence {confidence}")
        if confidence > max_confidence:
            best_arm = arm_index
            max_confidence = confidence

    print(f"True best arm: {true_best_arm}")
    print(f"Selected best arm: {best_arm}")
    print(f"Confidence: {max_confidence}")

    return prior

if __name__ == "__main__":
    prior = main()
