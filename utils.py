import csv
import itertools
import numpy as np
import pymc3 as pm


class Parameter():
    def __init__(self, title, n_steps, n_arms, n_mc_samples, m,
                 confidence_level, control_interval, seed, true_theta=None):
        self.title = title
        self.n_steps = n_steps
        self.n_arms = n_arms
        self.m = m
        self.n_mc_samples = n_mc_samples
        self.confidence_level = confidence_level
        self.control_interval = control_interval
        self.seed = seed
        # Draw true theta.
        if true_theta is None:
            true_theta = np.random.uniform(0, 1, n_arms)
        self.true_theta = true_theta


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


def learn(parameter, sampler, confidence_computer):
    np.random.seed(parameter.seed)
    # Define priors per arm.
    prior = Prior(parameter.n_arms)
    arm_selection_counts = np.zeros(parameter.n_arms)

    for step_index in range(parameter.n_steps):
        selected_arm = sampler(prior)
        # Play option/arm.
        reward = np.random.binomial(1, parameter.true_theta[selected_arm])
        # Update priors.
        prior.update(selected_arm, reward)
        arm_selection_counts[selected_arm] += 1

        if is_ready_to_stop(step_index, prior, parameter.confidence_level,
                            parameter.control_interval, confidence_computer):
            break

    print_sampling_results(prior, parameter.true_theta, arm_selection_counts)
    return prior, step_index


# Compute confidence via Monte Carlo integration.
def compute_confidence(prior, n_mc_samples, candidates, candidate_filter,
                       print_results, true_best):

    max_confidence = -100000
    best_candidate = -1

    samples = np.random.uniform(0, 1, (n_mc_samples, prior.n_arms))
    for candidate in candidates:
        filtered_samples = candidate_filter(samples, candidate)
        if filtered_samples.shape[0] == 0:
            continue
        probabilities = prior.prior.logp(filtered_samples).eval()
        # The volume of each parameter space THETA_I is 1^N_ARMS / (N_ARMS
        # choose M).
        # The density of theta is the product of the densities of each
        # theta_i.
        confidence = np.mean(np.prod(np.exp(probabilities), axis=1)) / len(candidates)
        if print_results:
            print(f"""{candidate}: #samples:
                  {filtered_samples.shape[0]}, confidence {confidence}""")
        if confidence > max_confidence:
            best_candidate = candidate
            max_confidence = confidence
    if print_results:
        print_confidence_results(best_candidate, max_confidence, true_best)
    return max_confidence, best_candidate


def run_experiment(parameter, sampler):
    if parameter.m == 1:
        candidate_filter = filter_samples_for_arm
        candidates = range(parameter.n_arms)
        true_best = select_best_arm_from_theta(parameter.true_theta)
    else:
        candidate_filter = filter_samples_for_arms_vect
        candidates = get_topm_candidates(parameter.n_arms, parameter.m)
        true_best = select_best_arms_from_theta(parameter.true_theta,
                                                parameter.m)
    confidence_computer = lambda prior: compute_confidence(
        prior, parameter.n_mc_samples, candidates,
        candidate_filter, False, true_best
    )
    prior, step_index = learn(parameter, sampler, confidence_computer)
    max_confidence, best_candidate = compute_confidence(
        prior, parameter.n_mc_samples, candidates, candidate_filter, True,
        true_best)
    theta = get_means_from_beta_distribution(prior)
    log_result(parameter, theta, true_best, best_candidate, max_confidence,
               step_index)


def is_ready_to_stop(step_index, prior, confidence_level, control_interval,
                     confidence_computer):
    if control_interval == -1:
        return False
    return (step_index > 0 and
            (step_index % control_interval == 0) and
            confidence_computer(prior)[0] >= confidence_level)

# All combinatorial possibilities to choose m fron {0, ..., N_ARMS-1}.
def get_topm_candidates(n_arms, m):
    return list(itertools.combinations(range(n_arms), m))


def select_best_arm_from_theta(theta):
    return np.argmax(theta)


def select_best_arms_from_theta(theta, m):
    return theta.argsort()[-m:][::-1]


def are_set_equal(array_a, array_b):
    return np.sum(np.in1d(array_a, array_b)) == len(array_a)


def get_means_from_beta_distribution(prior):
    return prior.alpha / (prior.alpha + prior.beta)


def filter_samples_for_arm(samples, arm_index):
    arm_filter = np.max(samples, axis=1) == samples[:, arm_index]
    return samples[arm_filter, :]


# Alternative implementation. Slower than vect for small M, should be faster
# for large M.
def filter_samples_for_arms(samples, arm_combination):
    m = len(arm_combination)
    arm_filter = np.apply_along_axis(
        lambda x: are_set_equal(x, arm_combination), 1, samples.argsort(axis=1)[:, -m:])
    return samples[arm_filter, :]

# Given an _unordered_ arm combination, say {0, 4, 7}, check all samples
# whether their top M arms (ordered) are equal to any of the ordered
# permutations of the combinations, e.g. [4, 0, 7], [7, 4, 0] etc.
def filter_samples_for_arms_vect(samples, arm_combination):
    n_samples = samples.shape[0]
    m = len(arm_combination)
    arm_filter = np.zeros(n_samples)
    best = samples.argsort(axis=1)[:, -m:]
    for permutation in set(itertools.permutations(arm_combination)):
        aux = np.prod(best == np.array(permutation), axis=1)
        arm_filter += aux
    return samples[arm_filter > 0, :]


def log_result(parameter, theta, true_best, best_candidate, confidence,
               step_index):
    with open('log.csv', mode='a+') as log_file:
        log_writer = csv.writer(log_file, delimiter='|', quotechar='',
                                quoting=csv.QUOTE_NONE)
        log_writer.writerow([
            parameter.title,
            parameter.true_theta,
            theta,
            parameter.m,
            true_best,
            best_candidate,
            parameter.confidence_level,
            confidence,
            parameter.n_steps,
            step_index + 1,
            parameter.seed,
            parameter.n_mc_samples
            ])


def print_sampling_results(prior, true_theta, arm_selection_counts):
    print(f"true theta: {true_theta}")
    print(f"Theta: {get_means_from_beta_distribution(prior)}")
    print(f"Arm selections: {arm_selection_counts}")
    print(f"#successes: {np.sum(prior.alpha) - prior.n_arms}")
    print(f"#failures: {np.sum(prior.beta) - prior.n_arms}")


def print_confidence_results(best_candidate, max_confidence, true_best):
    print(f"True best candidate: {true_best}")
    print(f"Selected best candidate: {best_candidate}")
    print(f"Confidence: {max_confidence}")
