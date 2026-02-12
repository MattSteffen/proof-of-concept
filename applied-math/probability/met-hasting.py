import numpy as np
from typing import Callable
import matplotlib.pyplot as plt


def metropolis_hastings(
    target_log_prob: Callable[[np.ndarray], float],
    proposal_fn: Callable[[np.ndarray], np.ndarray],
    initial_state: np.ndarray,
    n_samples: int,
    burn_in: int = 1000,
    thin: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    """
    Metropolis-Hastings MCMC sampler.

    Parameters
    ----------
    target_log_prob : callable
        Log probability of target distribution (can be unnormalized).
    proposal_fn : callable
        Function that takes current state and returns proposed state.
        Assumes symmetric proposal (random walk) - for asymmetric,
        you'd need to add proposal_log_ratio.
    initial_state : array
        Starting point for the chain.
    n_samples : int
        Number of samples to collect after burn-in.
    burn_in : int
        Number of initial samples to discard.
    thin : int
        Keep every thin-th sample (reduces autocorrelation).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    samples : array of shape (n_samples, dim)
        Samples from the target distribution.
    """
    if seed is not None:
        np.random.seed(seed)

    dim = len(initial_state)
    samples = np.zeros((n_samples, dim))

    current_state = np.array(initial_state, dtype=float)
    current_log_prob = target_log_prob(current_state)

    accepted = 0
    total_iterations = burn_in + n_samples * thin

    for i in range(total_iterations):
        # 1. Propose new state
        proposed_state = proposal_fn(current_state, .7)
        proposed_log_prob = target_log_prob(proposed_state)

        # 2. Compute acceptance probability (symmetric proposal)
        # log(α) = log(p(x')) - log(p(x))
        log_acceptance = proposed_log_prob - current_log_prob

        # 3. Accept or reject
        if np.log(np.random.random()) < log_acceptance:
            current_state = proposed_state
            current_log_prob = proposed_log_prob
            accepted += 1

        # 4. Store sample (after burn-in, with thinning)
        if i >= burn_in and (i - burn_in) % thin == 0:
            sample_idx = (i - burn_in) // thin
            samples[sample_idx] = current_state

    acceptance_rate = accepted / total_iterations
    print(f"Acceptance rate: {acceptance_rate:.3f}")

    return samples


# --- Example: Sampling from a 2D Gaussian mixture ---

def target_log_prob(x: np.ndarray) -> float:
    """Log probability of a 2D Gaussian mixture (unnormalized OK)."""
    # Two modes at (2, 2) and (-2, -2)
    mu1 = np.array([2.0, 2.0])
    mu2 = np.array([-2.0, -2.0])

    def log_gaussian(x, mu, sigma=1.0):
        return -0.5 * np.sum(((x - mu) / sigma) ** 2)

    # Mixture: log-sum-exp trick for numerical stability
    log_p1 = log_gaussian(x, mu1)
    log_p2 = log_gaussian(x, mu2)

    # log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
    max_log = max(log_p1, log_p2)
    return max_log + np.log(np.exp(log_p1 - max_log) + np.exp(log_p2 - max_log))


def proposal_fn(x: np.ndarray, step_size: float = 0.5) -> np.ndarray:
    """Random walk proposal: Gaussian perturbation."""
    return x + np.random.normal(0, step_size, size=x.shape)


if __name__ == "__main__":
    # Run sampler
    samples = metropolis_hastings(
        target_log_prob=target_log_prob,
        proposal_fn=proposal_fn,
        initial_state=np.array([0.0, 0.0]),
        n_samples=2000,
        burn_in=10000,
        seed=42,
    )

    # Basic diagnostics
    print(f"Sample mean: {samples.mean(axis=0)}")
    print(f"Sample std:  {samples.std(axis=0)}")

    # Quick visualization (requires matplotlib)
    try:

        plt.figure(figsize=(8, 6))
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=5)
        plt.title("MCMC Samples from Gaussian Mixture")
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.grid(True, alpha=0.3)
        plt.show()

        plt.plot(samples[:, 0], alpha=0.7)
        plt.xlabel("Iteration")
        plt.ylabel("x₁")
        plt.title("Trace Plot")
        plt.show()
    except ImportError:
        pass