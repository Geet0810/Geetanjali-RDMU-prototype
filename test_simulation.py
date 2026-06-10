"""
Test Suite — Disaster Response Coordination Simulator
RDMU Examination — Testing & Evaluation
"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agency_agents import MedicalAgency, RescueAgency, LogisticsAgency
from models.game_theory import (compute_shapley_values, find_nash_equilibrium,
                                 compute_fairness_index)
from models.monte_carlo import generate_mc_scenarios
from models.bayesian import BayesianBeliefUpdater
from simulation.orchestrator import DisasterSimulation


# ── Shapley Value Tests ──────────────────────────────────────────
class TestShapleyValues:
    def setup_method(self):
        self.agents = [
            MedicalAgency(40, 0.7),
            RescueAgency(35, 0.6),
            LogisticsAgency(25, 0.5)
        ]

    def test_shapley_sums_to_total_resources(self):
        total = 100.0
        shapley = compute_shapley_values(self.agents, total, 0.5)
        assert abs(sum(shapley.values()) - total) < 0.01, "Shapley values must sum to total resources (efficiency)"

    def test_all_agents_receive_positive_allocation(self):
        shapley = compute_shapley_values(self.agents, 100, 0.5)
        for agent, alloc in shapley.items():
            assert alloc > 0, f"{agent} should receive positive allocation"

    def test_medical_gets_most_resources(self):
        # Medical has highest priority_weight (1.4), should get most
        shapley = compute_shapley_values(self.agents, 100, 0.8)
        assert shapley['Medical Agency'] > shapley['Logistics Agency'], \
            "Medical should outrank Logistics in high-severity scenarios"

    def test_higher_severity_increases_medical_share(self):
        low_sev = compute_shapley_values(self.agents, 100, 0.1)
        high_sev = compute_shapley_values(self.agents, 100, 0.9)
        assert high_sev['Medical Agency'] >= low_sev['Medical Agency'], \
            "Medical share should increase with severity"


# ── Nash Equilibrium Tests ───────────────────────────────────────
class TestNashEquilibrium:
    def setup_method(self):
        self.agents = [
            MedicalAgency(40, 0.7), RescueAgency(35, 0.6), LogisticsAgency(25, 0.5)
        ]

    def test_high_incentive_leads_to_cooperation(self):
        nash = find_nash_equilibrium(self.agents, cooperation_incentive=0.9, resource_scarcity=0.2)
        coop_count = sum(1 for s in nash.values() if s == 'cooperate')
        assert coop_count >= 2, "High incentive + low scarcity should yield mostly cooperative Nash"

    def test_high_scarcity_may_yield_competition(self):
        nash = find_nash_equilibrium(self.agents, cooperation_incentive=0.1, resource_scarcity=0.95)
        comp_count = sum(1 for s in nash.values() if s == 'compete')
        assert comp_count >= 1, "High scarcity + low incentive should induce some competition"

    def test_nash_returns_valid_strategies(self):
        nash = find_nash_equilibrium(self.agents, 0.5, 0.5)
        for agent_name, strategy in nash.items():
            assert strategy in ['cooperate', 'compete'], f"Invalid strategy: {strategy}"


# ── Monte Carlo Tests ────────────────────────────────────────────
class TestMonteCarlo:
    def test_correct_number_of_scenarios(self):
        scenarios = generate_mc_scenarios(0.6, 100, n_trials=200)
        assert len(scenarios) == 200

    def test_severity_bounded(self):
        scenarios = generate_mc_scenarios(0.7, 100, n_trials=300)
        for s in scenarios:
            assert 0 <= s.actual_severity <= 1, "Severity must be in [0,1]"

    def test_resource_availability_positive(self):
        scenarios = generate_mc_scenarios(0.5, 100, n_trials=100)
        for s in scenarios:
            assert s.resource_availability > 0, "Resources must be positive"

    def test_seeded_reproducibility(self):
        s1 = generate_mc_scenarios(0.6, 100, n_trials=50, seed=99)
        s2 = generate_mc_scenarios(0.6, 100, n_trials=50, seed=99)
        assert s1[0].actual_severity == s2[0].actual_severity, "Same seed should yield same results"


# ── Bayesian Updater Tests ───────────────────────────────────────
class TestBayesianUpdater:
    def test_success_increases_cooperation_prob(self):
        updater = BayesianBeliefUpdater(3.0, 3.0)
        initial_prob = updater.posterior_mean
        for _ in range(10):
            updater.update(cooperated=True, outcome=0.8)
        assert updater.posterior_mean > initial_prob, "Successes should increase cooperation probability"

    def test_failure_decreases_cooperation_prob(self):
        updater = BayesianBeliefUpdater(3.0, 3.0)
        initial_prob = updater.posterior_mean
        for _ in range(10):
            updater.update(cooperated=True, outcome=0.2)
        assert updater.posterior_mean < initial_prob, "Failures should decrease cooperation probability"

    def test_credible_interval_valid(self):
        updater = BayesianBeliefUpdater(5.0, 3.0)
        ci_l, ci_h = updater.credible_interval
        assert 0 <= ci_l < ci_h <= 1, "Credible interval must be in [0,1]"

    def test_posterior_mean_bounded(self):
        updater = BayesianBeliefUpdater(2.0, 8.0)
        for _ in range(50):
            updater.update(True, np.random.random())
        assert 0 < updater.posterior_mean < 1, "Posterior mean must be in (0,1)"


# ── Integration Test ─────────────────────────────────────────────
class TestSimulationIntegration:
    def test_full_simulation_runs(self):
        sim = DisasterSimulation(0.6, 100, 0.6, n_rounds=5, mc_trials=100)
        results = sim.run()
        assert 'kpis' in results
        assert 'timeline_df' in results
        assert 'final_shapley' in results

    def test_fairness_index_in_range(self):
        sim = DisasterSimulation(0.5, 80, 0.7, n_rounds=5, mc_trials=50)
        results = sim.run()
        fi = results['kpis']['avg_fairness_index']
        assert 0 <= fi <= 1, f"Fairness index out of range: {fi}"

    def test_high_cooperation_improves_response_time(self):
        sim_low = DisasterSimulation(0.6, 100, 0.1, n_rounds=8, mc_trials=50)
        sim_high = DisasterSimulation(0.6, 100, 0.9, n_rounds=8, mc_trials=50)
        r_low = sim_low.run()
        r_high = sim_high.run()
        assert r_high['kpis']['avg_response_time_min'] < r_low['kpis']['avg_response_time_min'], \
            "High cooperation incentive should reduce average response time"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
