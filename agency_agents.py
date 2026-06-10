"""
Agency Agents Module
====================
Implements Multi-Agent Systems (MAS) — RDMU Concept #1
Each agency is an autonomous agent with its own:
  - Utility function (what it values)
  - Strategy (cooperate vs compete)
  - Bayesian belief state (learns from experience)
  - Resource pool
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class AgentState:
    """Snapshot of an agent's state at a given simulation round."""
    round: int
    strategy: str          # 'cooperate' or 'compete'
    cooperation_prob: float
    resources_available: float
    resources_used: float
    payoff: float
    response_efficiency: float


class DisasterResponseAgent:
    """
    Base class for a disaster response agency agent.

    CONCEPT: Multi-Agent Systems (MAS)
    Each agent independently decides whether to cooperate or compete
    for shared disaster resources, updating beliefs via Bayesian inference.
    """

    def __init__(self, name: str, agent_type: str,
                 initial_resources: float, base_cooperation: float,
                 risk_tolerance: float):
        self.name = name
        self.agent_type = agent_type  # 'medical', 'rescue', 'logistics'
        self.initial_resources = initial_resources
        self.resources = initial_resources
        self.risk_tolerance = risk_tolerance  # 0=risk-averse, 1=risk-seeking

        # CONCEPT: Bayesian Analysis — prior belief about cooperation success
        self.cooperation_prob = base_cooperation  # P(cooperation succeeds)
        self.alpha = base_cooperation * 10        # Beta distribution alpha (successes)
        self.beta_param = (1 - base_cooperation) * 10  # Beta distribution beta (failures)

        # Strategy and history
        self.strategy = 'cooperate' if base_cooperation > 0.5 else 'compete'
        self.history: List[AgentState] = []
        self.total_payoff = 0.0
        self.rounds_cooperated = 0
        self.rounds_competed = 0

    def decide_strategy(self, disaster_severity: float,
                        resource_scarcity: float,
                        cooperation_incentive: float) -> str:
        """
        CONCEPT: Probability-Based Decision Making
        Agent chooses strategy by computing expected utility of each option.

        Expected Utility (Cooperate) = P(success|coop) * U(coop_payoff) * incentive
        Expected Utility (Compete)   = P(success|comp) * U(comp_payoff) * (1-incentive)
        """
        # Compute expected utility for cooperation
        # Higher severity → more incentive to pool resources
        eu_cooperate = (self.cooperation_prob
                        * (1.0 + cooperation_incentive)
                        * (1.0 + disaster_severity * 0.5)
                        * self._utility(0.7))  # moderate guaranteed payoff

        # Compute expected utility for competition
        # Higher scarcity → temptation to grab resources
        eu_compete = ((1.0 - resource_scarcity * 0.4)
                      * (1.0 - cooperation_incentive * 0.6)
                      * self._utility(1.0) * 0.8)  # high but risky

        # Add stochastic noise (bounded rationality)
        noise = np.random.normal(0, 0.05)
        eu_cooperate = max(0, eu_cooperate + noise)

        self.strategy = 'cooperate' if eu_cooperate >= eu_compete else 'compete'
        return self.strategy

    def _utility(self, payoff: float) -> float:
        """
        CONCEPT: Utility Theory
        Risk-averse agents (risk_tolerance < 0.5) use concave utility (sqrt)
        Risk-seeking agents (risk_tolerance > 0.5) use convex utility (square)
        """
        if self.risk_tolerance < 0.4:
            return np.sqrt(max(0, payoff))   # Concave — risk averse
        elif self.risk_tolerance > 0.6:
            return payoff ** 2               # Convex — risk seeking
        else:
            return payoff                     # Linear — risk neutral

    def update_beliefs(self, outcome: float, cooperated: bool):
        """
        CONCEPT: Bayesian Analysis
        Update P(cooperation succeeds) using Beta-Binomial conjugate update.

        If cooperation succeeded (outcome > threshold): alpha += 1
        If cooperation failed (outcome <= threshold):   beta += 1
        This is the Bayesian posterior update rule for binary outcomes.
        """
        threshold = 0.5
        if cooperated:
            if outcome > threshold:
                self.alpha += 1      # Success evidence
            else:
                self.beta_param += 1  # Failure evidence
            # Updated posterior mean: alpha / (alpha + beta)
            self.cooperation_prob = self.alpha / (self.alpha + self.beta_param)

    def receive_resources(self, allocated: float, disaster_severity: float) -> float:
        """
        Compute the payoff this agent receives given resource allocation.
        Payoff = (resources_used / demand) * efficiency_factor
        """
        demand = self.initial_resources * (1.0 + disaster_severity)
        utilization = min(allocated / max(demand, 1), 1.0)

        # Efficiency bonus for cooperative agents
        efficiency = utilization * (1.1 if self.strategy == 'cooperate' else 0.85)
        payoff = efficiency * allocated / self.initial_resources
        self.total_payoff += payoff
        return payoff

    def record_state(self, round_num: int, resources_used: float, payoff: float):
        """Save agent state for timeline visualization."""
        state = AgentState(
            round=round_num,
            strategy=self.strategy,
            cooperation_prob=self.cooperation_prob,
            resources_available=self.resources,
            resources_used=resources_used,
            payoff=payoff,
            response_efficiency=payoff / max(resources_used, 0.01)
        )
        self.history.append(state)
        if self.strategy == 'cooperate':
            self.rounds_cooperated += 1
        else:
            self.rounds_competed += 1


class MedicalAgency(DisasterResponseAgent):
    """Medical emergency response agency — highest priority in mass-casualty events."""
    def __init__(self, resources: float, cooperation: float):
        super().__init__(
            name="Medical Agency", agent_type="medical",
            initial_resources=resources,
            base_cooperation=cooperation,
            risk_tolerance=0.3  # Risk-averse: lives at stake
        )
        self.priority_weight = 1.4  # Medical has highest priority


class RescueAgency(DisasterResponseAgent):
    """Search and rescue teams — moderate risk tolerance, field-adaptive."""
    def __init__(self, resources: float, cooperation: float):
        super().__init__(
            name="Rescue Agency", agent_type="rescue",
            initial_resources=resources,
            base_cooperation=cooperation,
            risk_tolerance=0.6  # Moderately risk-seeking
        )
        self.priority_weight = 1.2


class LogisticsAgency(DisasterResponseAgent):
    """Supply chain and logistics coordination — resource efficiency focused."""
    def __init__(self, resources: float, cooperation: float):
        super().__init__(
            name="Logistics Agency", agent_type="logistics",
            initial_resources=resources,
            base_cooperation=cooperation,
            risk_tolerance=0.5  # Risk-neutral: cost optimization
        )
        self.priority_weight = 1.0
