"""
Bayesian Analysis Module
========================
RDMU Concept #4: Bayesian Analysis + Probability-Based Decision Making

Implements:
  - Bayesian belief updates for agent cooperation learning
  - Posterior probability computation
  - Predictive analytics for future cooperation likelihood
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple


class BayesianBeliefUpdater:
    """
    BAYESIAN INFERENCE ENGINE
    ─────────────────────────
    Models each agent's evolving belief about cooperation success.

    Uses Beta-Binomial conjugate model:
        Prior:      P(θ) ~ Beta(α₀, β₀)      (initial belief)
        Likelihood: X|θ  ~ Binomial(n, θ)    (observed outcomes)
        Posterior:  P(θ|X) ~ Beta(α₀+X, β₀+n-X)  (updated belief)

    The posterior mean E[θ|X] = (α₀+X)/(α₀+β₀+n) is used as cooperation probability.
    """

    def __init__(self, prior_alpha: float = 3.0, prior_beta: float = 3.0):
        """
        Args:
            prior_alpha: Prior successes (higher = more optimistic about cooperation)
            prior_beta:  Prior failures  (higher = more pessimistic)
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.alpha = prior_alpha
        self.beta = prior_beta
        self.observations: List[Tuple[bool, float]] = []  # (cooperated, outcome)

    @property
    def posterior_mean(self) -> float:
        """Current best estimate of cooperation success probability."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def posterior_std(self) -> float:
        """Uncertainty in the cooperation probability estimate."""
        a, b = self.alpha, self.beta
        return np.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1)))

    @property
    def credible_interval(self) -> Tuple[float, float]:
        """95% Bayesian credible interval for cooperation success probability."""
        lower = stats.beta.ppf(0.025, self.alpha, self.beta)
        upper = stats.beta.ppf(0.975, self.alpha, self.beta)
        return lower, upper

    def update(self, cooperated: bool, outcome: float,
               success_threshold: float = 0.5) -> float:
        """
        Update belief after observing an outcome.

        If agent cooperated and outcome > threshold → success → alpha += 1
        If agent cooperated and outcome ≤ threshold → failure → beta  += 1
        Non-cooperative rounds don't update the cooperation belief.
        """
        self.observations.append((cooperated, outcome))
        if cooperated:
            if outcome > success_threshold:
                self.alpha += 1.0
            else:
                self.beta += 1.0
        return self.posterior_mean

    def predict_next_round(self, n_future_rounds: int = 5) -> List[float]:
        """
        Predictive distribution: sample cooperation probabilities for future rounds.
        Uses the posterior Beta distribution (posterior predictive).
        """
        samples = np.random.beta(self.alpha, self.beta, size=n_future_rounds)
        return samples.tolist()

    def get_belief_summary(self) -> Dict:
        """Return full belief state summary for visualization."""
        ci_low, ci_high = self.credible_interval
        return {
            'posterior_mean': self.posterior_mean,
            'posterior_std': self.posterior_std,
            'ci_lower': ci_low,
            'ci_upper': ci_high,
            'alpha': self.alpha,
            'beta': self.beta,
            'n_observations': len(self.observations),
            'n_successes': int(self.alpha - self.prior_alpha),
            'n_failures': int(self.beta - self.prior_beta)
        }


def compute_cooperation_probability_evolution(
        history: List[Dict]) -> List[float]:
    """
    Track how cooperation probability evolves across rounds.
    Shows the learning trajectory for visualization.
    """
    probs = []
    alpha, beta = 3.0, 3.0  # Prior
    for record in history:
        if record.get('cooperated'):
            if record.get('outcome', 0) > 0.5:
                alpha += 1
            else:
                beta += 1
        probs.append(alpha / (alpha + beta))
    return probs


def bayesian_risk_assessment(severity_observations: List[float],
                              prior_mean: float = 0.5,
                              prior_strength: float = 5.0) -> Dict:
    """
    Bayesian estimation of disaster severity from noisy observations.
    Uses Normal-Normal conjugate model.

    Prior:     μ ~ N(μ₀, σ₀²)
    Likelihood: x ~ N(μ, σ²)
    Posterior:  μ|x ~ N(μₙ, σₙ²)
    """
    if not severity_observations:
        return {'posterior_mean': prior_mean, 'posterior_std': 0.2}

    n = len(severity_observations)
    obs_mean = np.mean(severity_observations)
    obs_var = np.var(severity_observations) + 0.01  # Avoid zero variance

    # Conjugate update
    sigma0_sq = 1.0 / prior_strength
    sigma_sq = obs_var / n

    posterior_var = 1.0 / (1.0 / sigma0_sq + n / sigma_sq)
    posterior_mean = posterior_var * (prior_mean / sigma0_sq + obs_mean / (sigma_sq / n))

    return {
        'posterior_mean': np.clip(posterior_mean, 0, 1),
        'posterior_std': np.sqrt(posterior_var),
        'n_observations': n,
        'likelihood_mean': obs_mean
    }
