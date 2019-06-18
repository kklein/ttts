import numpy as np
import pymc3 as pm

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

    def update(self, arm_index, reward):
        self.prior.alpha[arm_index] += reward
        self.prior.beta[arm_index] += 1 - reward

    @property
    def alpha(self):
        return self.prior.alpha

    @property
    def beta(self):
        return self.prior.beta


def learn(seed, n_arms, n_steps, sampler, true_theta=None):
    np.random.seed(seed)
    # Define priors per arm.
    prior = Prior(n_arms)
    arm_selection_counts = np.zeros(n_arms)

    # Draw true theta.
    if true_theta is None:
        true_theta = np.random.uniform(0, 1, n_arms)

    true_theta = np.array([.1, .2, .3, .4, .5])

    for _ in range(n_steps):
        selected_arm = sampler(prior)
        # Play option/arm.
        reward = np.random.binomial(1, true_theta[selected_arm])
        # Update priors.
        prior.update(selected_arm, reward)
        arm_selection_counts[selected_arm] += 1

    print_sampling_results(prior, true_theta, arm_selection_counts)
    return prior


# Compute confidence via Monte Carlo integration.
def compute_confidence(prior, n_mc_samples, candidates, candidate_filter,
                       true_best):

    max_confidence = -100000
    best_candidate = -1

    samples = np.random.uniform(0, 1, (n_mc_samples, prior.n_arms))
    for candidate in candidates:
        filtered_samples = candidate_filter(samples, candidate)
        print(f"{filtered_samples.shape[0]} samples for candidate {candidate}.")
        if filtered_samples.shape[0] == 0:
            continue
        probabilities = prior.prior.logp(filtered_samples).eval()
        # The volume of each parameter space THETA_I is 1^N_ARMS / (N_ARMS
        # choose M).
        # The density of theta is the product of the densities of each
        # theta_i.
        confidence = np.mean(np.prod(np.exp(probabilities), axis=1)) / len(candidates)

        print(f"Arm combination {candidate}: confidence {confidence}")
        if confidence > max_confidence:
            best_candidate = candidate
            max_confidence = confidence
    print_confidence_results(best_candidate, max_confidence, true_best)


def print_sampling_results(prior, true_theta, arm_selection_counts):
    print(f"true theta: {true_theta}")
    print(f"Theta: {prior.alpha / (prior.alpha + prior.beta)}")
    print(f"Arm selections: {arm_selection_counts}")
    print(f"#successes: {np.sum(prior.alpha) - prior.n_arms}")
    print(f"#failures: {np.sum(prior.beta) - prior.n_arms}")


def print_confidence_results(best_candidate, max_confidence, true_best):
    print(f"True best candidate: {true_best}")
    print(f"Selected best candidate: {best_candidate}")
    print(f"Confidence: {max_confidence}")
