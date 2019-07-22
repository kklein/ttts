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

    def draw(self, n_samples=None):
        if n_samples is None:
            return self.prior.random()
        return self.prior.random(size=n_samples)

    def update(self, arm_index, reward):
        self.prior.alpha[arm_index] += reward
        self.prior.beta[arm_index] += 1 - reward

    @property
    def alpha(self):
        return self.prior.alpha

    @property
    def beta(self):
        return self.prior.beta


def learn(parameter, sampler, confidence_computer, logger):
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

        if step_index % parameter.control_interval == 0:
            confidence, candidate = confidence_computer(prior)
            theta = get_means_from_beta_distribution(prior)
            logger(confidence, candidate, step_index, theta)

            if confidence > parameter.confidence_level:
                break

    arm_selections = arm_selection_counts / step_index
    # print_sampling_results(prior, parameter.true_theta, arm_selections)
    return prior, step_index, arm_selections


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


def compute_confidence_direct(prior, n_samples, candidates, candidate_filter,
                              print_results, true_best):

    max_confidence = -100000
    best_candidate = -1

    candidate_counts = np.zeros(len(candidates))

    samples = prior.draw(n_samples)

    candidate_index = 0
    for candidate in candidates:

        filtered_samples = candidate_filter(samples, candidate)
        candidate_counts[candidate_index] = filtered_samples.shape[0]
        candidate_index += 1

    assert n_samples == np.sum(candidate_counts)

    # Determine 'best' candidate.
    best_candidate_index = np.argmax(candidate_counts)
    best_candidate = candidates[best_candidate_index]
    max_confidence = candidate_counts[best_candidate_index] / n_samples

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
    confidence_computer = lambda prior: compute_confidence_direct(
        prior, parameter.n_mc_samples, candidates,
        candidate_filter, False, true_best
    )
    logger = lambda confidence, best_candidate, step_index, theta: log_result(
        parameter, theta, true_best, best_candidate, confidence, step_index
    )
    prior, step_index, arm_selections= learn(
        parameter, sampler, confidence_computer, logger)
    max_confidence, best_candidate = compute_confidence_direct(
        prior, parameter.n_mc_samples, candidates, candidate_filter, True,
        true_best)
    theta = get_means_from_beta_distribution(prior)
    log_result(parameter, theta, true_best, best_candidate, max_confidence,
               step_index)
    # log_arm_selections(parameter, arm_selections)


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


def log_arm_selections(parameter, arm_selections):
    with open('arm_selections.csv', mode='a+') as log_file:
        log_writer = csv.writer(log_file, delimiter='|', quotechar='',
                                quoting=csv.QUOTE_NONE, escapechar='\\')
        log_writer.writerow([
            parameter.title,
            parameter.true_theta,
            parameter.m,
            arm_selections
            ])


def log_result(parameter, theta, true_best, best_candidate, confidence,
               step_index):
    np.set_printoptions(linewidth=np.inf)
    with open('log_steps_2.csv', mode='a+') as log_file:
        log_writer = csv.writer(log_file, delimiter='|', quotechar='',
                                quoting=csv.QUOTE_NONE, escapechar='\\')
        log_writer.writerow([
            parameter.title,
            parameter.true_theta,
            theta,
            parameter.m,
            true_best,
            best_candidate,
            parameter.confidence_level,
            confidence,
            # parameter.n_steps,
            step_index + 1,
            parameter.seed,
            parameter.n_mc_samples
            ])


def print_sampling_results(prior, true_theta, arm_selections):
    print(f"true theta: {true_theta}")
    print(f"Theta: {get_means_from_beta_distribution(prior)}")
    print(f"Arm selections: {arm_selections}")
    print(f"#successes: {np.sum(prior.alpha) - prior.n_arms}")
    print(f"#failures: {np.sum(prior.beta) - prior.n_arms}")


def print_confidence_results(best_candidate, max_confidence, true_best):
    print(f"True best candidate: {true_best}")
    print(f"Selected best candidate: {best_candidate}")
    print(f"Confidence: {max_confidence}")
