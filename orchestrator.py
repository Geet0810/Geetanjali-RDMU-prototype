"""
Simulation Orchestrator
=======================
Coordinates all agents, models, and game-theoretic computations
across multiple rounds of the disaster response simulation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from agents.agency_agents import MedicalAgency, RescueAgency, LogisticsAgency
from models.game_theory import (compute_shapley_values, find_nash_equilibrium,
                                 compute_fairness_index, compute_social_welfare)
from models.monte_carlo import generate_mc_scenarios, compute_mc_statistics
from models.bayesian import BayesianBeliefUpdater, bayesian_risk_assessment


class DisasterSimulation:
    """
    Master simulation class that orchestrates:
    1. Multi-agent decision making (MAS)
    2. Game-theoretic resource allocation (Shapley + Nash)
    3. Monte Carlo uncertainty modeling
    4. Bayesian belief updates
    """

    def __init__(self, disaster_severity: float, total_resources: float,
                 cooperation_incentive: float, n_rounds: int = 15,
                 mc_trials: int = 500):

        self.disaster_severity = disaster_severity
        self.total_resources = total_resources
        self.cooperation_incentive = cooperation_incentive
        self.n_rounds = n_rounds
        self.mc_trials = mc_trials

        # Resource scarcity: 0=abundant, 1=scarce
        self.resource_scarcity = 1.0 - min(total_resources / 150.0, 1.0)

        # Initialize agents with proportional resource shares
        med_r = total_resources * 0.40
        res_r = total_resources * 0.35
        log_r = total_resources * 0.25

        # Initial cooperation probability driven by incentive parameter
        base_coop = 0.3 + cooperation_incentive * 0.5
        self.agents = [
            MedicalAgency(med_r, min(base_coop + 0.1, 0.99)),
            RescueAgency(res_r, base_coop),
            LogisticsAgency(log_r, max(base_coop - 0.05, 0.1))
        ]

        # Bayesian updaters — one per agent
        self.bayesian_updaters = {
            agent.name: BayesianBeliefUpdater(
                prior_alpha=base_coop * 8,
                prior_beta=(1 - base_coop) * 8
            )
            for agent in self.agents
        }

        # Results storage
        self.round_results: List[Dict] = []
        self.mc_scenarios = None
        self.aggregated_mc = None

    def run(self) -> Dict:
        """Run the full simulation: Monte Carlo + Multi-round agent simulation."""
        # Step 1: Generate Monte Carlo scenarios for uncertainty envelope
        self.mc_scenarios = generate_mc_scenarios(
            self.disaster_severity, self.total_resources,
            n_trials=self.mc_trials
        )

        # Step 2: Run multi-round agent simulation
        for round_num in range(1, self.n_rounds + 1):
            result = self._run_round(round_num)
            self.round_results.append(result)

        # Step 3: Compute MC statistics
        self.aggregated_mc = self._compute_mc_envelope()

        return self._compile_final_results()

    def _run_round(self, round_num: int) -> Dict:
        """Execute one simulation round."""
        # Severity evolves: can intensify or diminish over time
        round_severity = self.disaster_severity * (
            1.0 + 0.1 * np.sin(round_num * 0.5) +
            np.random.normal(0, 0.05)
        )
        round_severity = np.clip(round_severity, 0.01, 1.0)

        # Resources deplete over rounds
        round_resources = self.total_resources * max(0.3, 1.0 - round_num * 0.03)

        # Step A: Each agent decides strategy
        strategies = {}
        for agent in self.agents:
            strategy = agent.decide_strategy(
                round_severity, self.resource_scarcity, self.cooperation_incentive
            )
            strategies[agent.name] = strategy

        # Step B: Find Nash Equilibrium
        nash_strategies = find_nash_equilibrium(
            self.agents, self.cooperation_incentive, self.resource_scarcity
        )

        # Step C: Compute Shapley values (fair allocation)
        shapley_alloc = compute_shapley_values(
            self.agents, round_resources, round_severity
        )

        # Step D: Compute actual allocation (Shapley if cooperative, competitive if not)
        n_cooperating = sum(1 for s in strategies.values() if s == 'cooperate')
        cooperation_ratio = n_cooperating / len(self.agents)

        # Blend Shapley allocation with equal-split based on cooperation ratio
        equal_alloc = {a.name: round_resources / len(self.agents) for a in self.agents}
        final_alloc = {
            name: (cooperation_ratio * shapley_alloc.get(name, 0) +
                   (1 - cooperation_ratio) * equal_alloc[name])
            for name in shapley_alloc
        }

        # Step E: Compute payoffs and update beliefs
        payoffs = {}
        for agent in self.agents:
            alloc = final_alloc.get(agent.name, 0)
            payoff = agent.receive_resources(alloc, round_severity)
            payoffs[agent.name] = payoff

            # Bayesian update
            outcome_success = payoff / max(alloc / self.total_resources, 0.01)
            self.bayesian_updaters[agent.name].update(
                cooperated=(strategies[agent.name] == 'cooperate'),
                outcome=min(outcome_success, 1.0)
            )
            agent.update_beliefs(min(outcome_success, 1.0),
                                  cooperated=(strategies[agent.name] == 'cooperate'))
            agent.record_state(round_num, alloc, payoff)

        # Step F: Compute KPIs
        fairness = compute_fairness_index(final_alloc)
        social_welfare = compute_social_welfare(payoffs, strategies)

        # Response time: lower when more agents cooperate
        base_times = {'Medical Agency': 35, 'Rescue Agency': 45, 'Logistics Agency': 55}
        avg_response = np.mean([
            t * (0.7 if strategies[agent.name] == 'cooperate' else 1.3)
            for agent, t in zip(self.agents, base_times.values())
        ])

        # Resource utilization
        resource_utilization = sum(final_alloc.values()) / max(round_resources, 1)

        return {
            'round': round_num,
            'severity': round_severity,
            'strategies': strategies,
            'nash_strategies': nash_strategies,
            'shapley_allocation': shapley_alloc,
            'final_allocation': final_alloc,
            'payoffs': payoffs,
            'cooperation_ratio': cooperation_ratio,
            'fairness_index': fairness,
            'social_welfare': social_welfare,
            'avg_response_time': avg_response,
            'resource_utilization': resource_utilization,
            'bayesian_beliefs': {
                name: self.bayesian_updaters[name].posterior_mean
                for name in self.bayesian_updaters
            }
        }

    def _compute_mc_envelope(self) -> Dict:
        """Run quick MC aggregate for confidence interval visualization."""
        mc_results = []
        for scenario in self.mc_scenarios[:200]:  # Use 200 for speed
            # Simple outcome model per scenario
            coop_multiplier = 1.0 + self.cooperation_incentive * 0.4
            severity_effect = 1.0 - scenario.actual_severity * 0.3
            resource_effect = scenario.resource_availability

            outcome = (coop_multiplier * severity_effect *
                       resource_effect * self.total_resources / 100)
            mc_results.append({
                'outcome': outcome,
                'severity': scenario.actual_severity,
                'resources': scenario.resource_availability,
                'response_time': np.mean(list(scenario.agency_response_times.values()))
            })

        return compute_mc_statistics(mc_results)

    def _compile_final_results(self) -> Dict:
        """Compile all results into dashboard-ready format."""
        df = pd.DataFrame(self.round_results)

        # Timeline data
        timeline = []
        for r in self.round_results:
            for agent_name, strategy in r['strategies'].items():
                timeline.append({
                    'round': r['round'],
                    'agent': agent_name,
                    'strategy': strategy,
                    'payoff': r['payoffs'].get(agent_name, 0),
                    'allocation': r['final_allocation'].get(agent_name, 0),
                    'shapley': r['shapley_allocation'].get(agent_name, 0),
                    'bayesian_belief': r['bayesian_beliefs'].get(agent_name, 0)
                })

        # Final Shapley values from last round
        final_shapley = self.round_results[-1]['shapley_allocation'] if self.round_results else {}

        # Summary statistics
        avg_fairness = df['fairness_index'].mean()
        avg_cooperation = df['cooperation_ratio'].mean()
        avg_response = df['avg_response_time'].mean()
        avg_utilization = df['resource_utilization'].mean()
        total_welfare = df['social_welfare'].sum()

        # Bayesian final beliefs
        final_beliefs = {
            name: updater.get_belief_summary()
            for name, updater in self.bayesian_updaters.items()
        }

        return {
            'timeline_df': pd.DataFrame(timeline),
            'round_df': df,
            'final_shapley': final_shapley,
            'final_beliefs': final_beliefs,
            'mc_stats': self.aggregated_mc,
            'kpis': {
                'avg_fairness_index': avg_fairness,
                'avg_cooperation_rate': avg_cooperation,
                'avg_response_time_min': avg_response,
                'avg_resource_utilization': avg_utilization,
                'total_social_welfare': total_welfare
            },
            'agents': self.agents
        }
