"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DISASTER RESPONSE COORDINATION SIMULATOR                                    ║
║  Risk and Decision Making Under Uncertainty (RDMU)                           ║
║  ─────────────────────────────────────────────────                           ║
║  Concepts Demonstrated:                                                       ║
║    1. Multi-Agent Systems (MAS)                                               ║
║    2. Game Theory (Shapley Values + Nash Equilibrium)                         ║
║    3. Monte Carlo Simulation                                                  ║
║    4. Bayesian Analysis + Probability-Based Decision Making                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.orchestrator import DisasterSimulation

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Disaster Response Simulator — RDMU",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white; padding: 20px 30px; border-radius: 12px;
        margin-bottom: 20px; text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 2em; }
    .main-header p  { margin: 5px 0 0; opacity: 0.8; }

    .kpi-card {
        background: #f8f9fa; border-radius: 10px; padding: 15px 20px;
        border-left: 5px solid #0f3460; margin: 5px 0;
    }
    .kpi-card h3 { margin: 0; font-size: 0.9em; color: #666; }
    .kpi-card h2 { margin: 5px 0 0; font-size: 1.8em; color: #1a1a2e; }

    .concept-badge {
        display: inline-block; background: #0f3460; color: white;
        border-radius: 20px; padding: 3px 12px; font-size: 0.78em;
        margin: 2px;
    }
    .agency-card {
        background: white; border-radius: 8px; padding: 10px 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🚨 Disaster Response Coordination Simulator</h1>
    <p>Risk & Decision Making Under Uncertainty (RDMU) — Exam Prototype</p>
    <p>
        <span class="concept-badge">Multi-Agent Systems</span>
        <span class="concept-badge">Game Theory</span>
        <span class="concept-badge">Monte Carlo Simulation</span>
        <span class="concept-badge">Bayesian Analysis</span>
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR — SIMULATION CONTROLS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Simulation Parameters")
    st.markdown("---")

    st.markdown("### 🌪️ Disaster Configuration")
    disaster_type = st.selectbox(
        "Disaster Type",
        ["Earthquake", "Flood", "Pandemic", "Industrial Accident", "Cyclone"],
        index=0
    )
    severity = st.slider(
        "Disaster Severity", 0.1, 1.0, 0.65, 0.05,
        help="0=Minor, 1=Catastrophic. Affects resource demand and cooperation incentives."
    )

    st.markdown("### 💰 Resource Configuration")
    total_resources = st.slider(
        "Total Available Resources (units)", 30, 200, 100, 10,
        help="Total resources available across all agencies (personnel, equipment, supplies)."
    )

    st.markdown("### 🤝 Cooperation Dynamics")
    cooperation_incentive = st.slider(
        "Cooperation Incentive", 0.0, 1.0, 0.6, 0.05,
        help="Government/policy cooperation bonuses. Higher = agencies prefer to work together."
    )

    st.markdown("### 🔬 Simulation Settings")
    n_rounds = st.slider("Simulation Rounds", 5, 25, 15, 1)
    mc_trials = st.selectbox("Monte Carlo Trials", [200, 500, 1000], index=1)

    st.markdown("---")
    run_button = st.button("▶ Run Simulation", type="primary", use_container_width=True)

    if st.button("🎲 Random Scenario", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.markdown("""
    **Concept Guide:**
    - 🤖 **MAS**: Each agency is an autonomous agent
    - 🎮 **Game Theory**: Shapley values + Nash equilibrium  
    - 🎲 **Monte Carlo**: Uncertainty in 500 scenarios
    - 📊 **Bayesian**: Agents learn from outcomes
    """)

# ─────────────────────────────────────────────
# RUN SIMULATION
# ─────────────────────────────────────────────
if run_button or 'results' not in st.session_state:
    if run_button or 'results' not in st.session_state:
        with st.spinner(f"🚨 Simulating {disaster_type} response... Running {mc_trials} Monte Carlo trials..."):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.015)
                progress.progress(i + 1)

            sim = DisasterSimulation(
                disaster_severity=severity,
                total_resources=total_resources,
                cooperation_incentive=cooperation_incentive,
                n_rounds=n_rounds,
                mc_trials=mc_trials
            )
            results = sim.run()
            st.session_state['results'] = results
            st.session_state['params'] = {
                'severity': severity, 'resources': total_resources,
                'incentive': cooperation_incentive, 'disaster': disaster_type
            }
            progress.empty()

results = st.session_state['results']
params = st.session_state.get('params', {})

# ─────────────────────────────────────────────
# KPI DASHBOARD ROW
# ─────────────────────────────────────────────
st.markdown("## 📊 Simulation Dashboard")

kpis = results['kpis']
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""<div class="kpi-card">
        <h3>🎯 Avg Fairness Index</h3>
        <h2>{kpis['avg_fairness_index']:.3f}</h2>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""<div class="kpi-card">
        <h3>🤝 Cooperation Rate</h3>
        <h2>{kpis['avg_cooperation_rate']:.1%}</h2>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""<div class="kpi-card">
        <h3>⏱️ Avg Response Time</h3>
        <h2>{kpis['avg_response_time_min']:.1f} min</h2>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""<div class="kpi-card">
        <h3>📦 Resource Utilization</h3>
        <h2>{kpis['avg_resource_utilization']:.1%}</h2>
    </div>""", unsafe_allow_html=True)

with col5:
    st.markdown(f"""<div class="kpi-card">
        <h3>🌐 Social Welfare</h3>
        <h2>{kpis['total_social_welfare']:.1f}</h2>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# TABS FOR VISUALIZATIONS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 Agent Strategies", "🎮 Shapley Values",
    "🎲 Monte Carlo", "📊 Bayesian Beliefs", "📋 Data Tables"
])

# ─────── TAB 1: STRATEGY EVOLUTION TIMELINE ───────
with tab1:
    st.markdown("### 🤖 Multi-Agent Strategy Evolution Timeline")
    st.markdown("""
    **RDMU Concept: Multi-Agent Systems + Game Theory**  
    Each agency independently chooses to cooperate or compete based on expected utility.
    The timeline shows how strategies evolve as agents learn and disaster conditions change.
    """)

    timeline_df = results['timeline_df']
    round_df = results['round_df']

    # Strategy evolution per agent
    fig_strategy = go.Figure()
    colors = {'Medical Agency': '#e74c3c', 'Rescue Agency': '#2980b9',
               'Logistics Agency': '#27ae60'}
    strategy_map = {'cooperate': 1, 'compete': 0}

    for agent_name in timeline_df['agent'].unique():
        agent_data = timeline_df[timeline_df['agent'] == agent_name]
        strat_vals = [strategy_map[s] for s in agent_data['strategy']]

        fig_strategy.add_trace(go.Scatter(
            x=agent_data['round'],
            y=[v + list(timeline_df['agent'].unique()).index(agent_name) * 0.05
               for v in strat_vals],
            mode='lines+markers',
            name=agent_name,
            line=dict(color=colors.get(agent_name, '#95a5a6'), width=3),
            marker=dict(size=10,
                        symbol=['circle' if s == 'cooperate' else 'x' for s in agent_data['strategy']]),
            hovertemplate=f"<b>{agent_name}</b><br>Round: %{{x}}<br>Strategy: %{{text}}<extra></extra>",
            text=agent_data['strategy']
        ))

    fig_strategy.update_layout(
        yaxis=dict(tickvals=[0, 1], ticktext=['⚔️ Compete', '🤝 Cooperate'],
                   range=[-0.2, 1.3]),
        xaxis_title="Simulation Round",
        title="Strategy Evolution: Cooperate vs Compete per Agency",
        height=350, legend=dict(orientation='h', y=-0.2),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_strategy, use_container_width=True)

    # Cooperation ratio + severity
    col_a, col_b = st.columns(2)

    with col_a:
        fig_coop = go.Figure()
        fig_coop.add_trace(go.Bar(
            x=round_df['round'], y=round_df['cooperation_ratio'],
            name='Cooperation Rate', marker_color='#3498db',
            opacity=0.8
        ))
        fig_coop.add_trace(go.Scatter(
            x=round_df['round'], y=round_df['severity'],
            name='Disaster Severity', line=dict(color='#e74c3c', width=2, dash='dash'),
            yaxis='y2'
        ))
        fig_coop.update_layout(
            title="Cooperation Rate vs Disaster Severity",
            yaxis=dict(title="Cooperation Rate", range=[0, 1.1]),
            yaxis2=dict(title="Severity", overlaying='y', side='right', range=[0, 1.1]),
            height=300, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', y=-0.3)
        )
        st.plotly_chart(fig_coop, use_container_width=True)

    with col_b:
        fig_payoff = px.area(
            timeline_df, x='round', y='payoff', color='agent',
            color_discrete_map=colors,
            title="Agent Payoffs Over Time"
        )
        fig_payoff.update_layout(
            height=300, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', y=-0.3)
        )
        st.plotly_chart(fig_payoff, use_container_width=True)

    # Nash Equilibrium analysis
    st.markdown("### ♟️ Nash Equilibrium Analysis")
    nash_data = round_df['nash_strategies'].tolist()
    if nash_data:
        last_nash = nash_data[-1]
        nash_cols = st.columns(len(last_nash))
        for idx, (agent, strategy) in enumerate(last_nash.items()):
            icon = "🤝" if strategy == 'cooperate' else "⚔️"
            with nash_cols[idx]:
                st.metric(f"{icon} {agent.split()[0]}", strategy.title(),
                          help=f"Nash equilibrium strategy for {agent}")

# ─────── TAB 2: SHAPLEY VALUES ───────
with tab2:
    st.markdown("### 🎮 Game Theory: Shapley Value Resource Allocation")
    st.markdown("""
    **RDMU Concept: Cooperative Game Theory**  
    Shapley values compute each agency's *fair* resource share based on their marginal contribution
    to every possible coalition. Formula: φᵢ = Σ [|S|!(|N|-|S|-1)!/|N|!] × [v(S∪{i}) - v(S)]
    """)

    final_shapley = results['final_shapley']

    col_s1, col_s2 = st.columns([1, 1])

    with col_s1:
        # Shapley pie chart
        fig_shapley = go.Figure(go.Pie(
            labels=list(final_shapley.keys()),
            values=[round(v, 2) for v in final_shapley.values()],
            hole=0.4,
            marker=dict(colors=['#e74c3c', '#2980b9', '#27ae60']),
            textinfo='label+percent+value',
            hovertemplate='%{label}<br>Allocation: %{value:.1f} units<br>Share: %{percent}<extra></extra>'
        ))
        fig_shapley.update_layout(
            title="Final Shapley Value Resource Allocation",
            height=380, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_shapley, use_container_width=True)

    with col_s2:
        # Allocation evolution over rounds
        alloc_records = []
        for r in results['round_df'].itertuples():
            for agent, alloc in r.shapley_allocation.items():
                alloc_records.append({'round': r.round, 'agent': agent, 'allocation': alloc})

        alloc_df = pd.DataFrame(alloc_records)
        fig_alloc = px.line(
            alloc_df, x='round', y='allocation', color='agent',
            color_discrete_map=colors,
            title="Shapley Allocation Evolution",
            markers=True
        )
        fig_alloc.update_layout(
            height=380, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', y=-0.25)
        )
        st.plotly_chart(fig_alloc, use_container_width=True)

    # Fairness index evolution
    fig_fairness = go.Figure()
    fig_fairness.add_trace(go.Scatter(
        x=round_df['round'], y=round_df['fairness_index'],
        fill='tozeroy', mode='lines+markers',
        line=dict(color='#9b59b6', width=2),
        name="Fairness Index"
    ))
    fig_fairness.add_hline(y=0.9, line_dash='dash', line_color='green',
                           annotation_text="High Fairness Threshold (0.9)")
    fig_fairness.add_hline(y=0.7, line_dash='dash', line_color='orange',
                           annotation_text="Minimum Acceptable (0.7)")
    fig_fairness.update_layout(
        title="Jain's Fairness Index (0=Unfair → 1=Perfectly Fair)",
        xaxis_title="Round", yaxis_title="Fairness Index",
        yaxis=dict(range=[0, 1.1]),
        height=280, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_fairness, use_container_width=True)

# ─────── TAB 3: MONTE CARLO ───────
with tab3:
    st.markdown("### 🎲 Monte Carlo Simulation — Uncertainty Envelope")
    st.markdown("""
    **RDMU Concept: Monte Carlo Simulation**  
    We run 500 stochastic trials sampling uncertain parameters (severity, resource availability,
    response times) from probability distributions. The resulting confidence intervals show
    the *range of plausible outcomes* rather than a single deterministic answer.
    """)

    mc_stats = results.get('mc_stats', {})

    if mc_stats:
        # Distribution of outcomes
        rng_vis = np.random.default_rng(42)

        # Simulate outcome distribution from stats
        outcome_mean = mc_stats.get('outcome', {}).get('mean', 1.0)
        outcome_std = mc_stats.get('outcome', {}).get('std', 0.2)
        sim_outcomes = rng_vis.normal(outcome_mean, outcome_std, 500)

        col_mc1, col_mc2 = st.columns(2)

        with col_mc1:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=sim_outcomes, nbinsx=30,
                marker_color='#3498db', opacity=0.75,
                name='Outcomes'
            ))
            fig_hist.add_vline(x=np.percentile(sim_outcomes, 5),
                                line_dash='dash', line_color='red',
                                annotation_text='5th pct (Worst Case)')
            fig_hist.add_vline(x=np.mean(sim_outcomes),
                                line_dash='solid', line_color='green',
                                annotation_text='Mean')
            fig_hist.add_vline(x=np.percentile(sim_outcomes, 95),
                                line_dash='dash', line_color='orange',
                                annotation_text='95th pct (Best Case)')
            fig_hist.update_layout(
                title="Monte Carlo Outcome Distribution (500 trials)",
                xaxis_title="Coordination Effectiveness",
                yaxis_title="Frequency",
                height=350, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_mc2:
            # Confidence interval box plot
            stat_labels = ['5th pct\n(Worst)', '25th pct', 'Median', '75th pct', '95th pct\n(Best)']
            stat_vals = [
                mc_stats.get('outcome', {}).get('p5', 0.5),
                mc_stats.get('outcome', {}).get('p25', 0.7),
                mc_stats.get('outcome', {}).get('median', 0.9),
                mc_stats.get('outcome', {}).get('p75', 1.1),
                mc_stats.get('outcome', {}).get('p95', 1.3),
            ]

            fig_ci = go.Figure()
            fig_ci.add_trace(go.Bar(
                x=['5th\n(Worst)', '25th', 'Median', '75th', '95th\n(Best)'],
                y=stat_vals,
                marker_color=['#e74c3c', '#e67e22', '#2ecc71', '#27ae60', '#1abc9c'],
                text=[f'{v:.2f}' for v in stat_vals], textposition='auto'
            ))
            fig_ci.update_layout(
                title="Monte Carlo Percentile Distribution",
                yaxis_title="Coordination Effectiveness Score",
                height=350, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_ci, use_container_width=True)

        # Response time distribution
        rt_mean = mc_stats.get('response_time', {}).get('mean', 45)
        rt_std = mc_stats.get('response_time', {}).get('std', 10)
        rt_samples = np.random.lognormal(np.log(rt_mean), 0.25, 500)

        fig_rt = px.violin(
            x=rt_samples, box=True, points='outliers',
            title="Response Time Distribution across 500 MC Scenarios (minutes)",
            labels={'x': 'Response Time (min)'}
        )
        fig_rt.update_layout(
            height=300, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_rt, use_container_width=True)

        # Summary stats table
        st.markdown("**Monte Carlo Summary Statistics:**")
        mc_summary = pd.DataFrame({
            'Metric': ['Coordination Effectiveness', 'Disaster Severity', 'Response Time (min)'],
            'Mean': [f"{mc_stats.get('outcome', {}).get('mean', 0):.3f}",
                     f"{mc_stats.get('severity', {}).get('mean', 0):.3f}",
                     f"{mc_stats.get('response_time', {}).get('mean', 0):.1f}"],
            'Std Dev': [f"{mc_stats.get('outcome', {}).get('std', 0):.3f}",
                        f"{mc_stats.get('severity', {}).get('std', 0):.3f}",
                        f"{mc_stats.get('response_time', {}).get('std', 0):.1f}"],
            '5th Pct': [f"{mc_stats.get('outcome', {}).get('p5', 0):.3f}",
                        f"{mc_stats.get('severity', {}).get('p5', 0):.3f}",
                        f"{mc_stats.get('response_time', {}).get('p5', 0):.1f}"],
            '95th Pct': [f"{mc_stats.get('outcome', {}).get('p95', 0):.3f}",
                         f"{mc_stats.get('severity', {}).get('p95', 0):.3f}",
                         f"{mc_stats.get('response_time', {}).get('p95', 0):.1f}"],
        })
        st.dataframe(mc_summary, use_container_width=True, hide_index=True)

# ─────── TAB 4: BAYESIAN BELIEFS ───────
with tab4:
    st.markdown("### 📊 Bayesian Belief Evolution")
    st.markdown("""
    **RDMU Concept: Bayesian Analysis**  
    Agents use Beta-Binomial conjugate updating to refine their belief about
    cooperation success probability. As more evidence accumulates, beliefs converge.  
    Prior: Beta(α₀, β₀) → Posterior: Beta(α₀+successes, β₀+failures)
    """)

    final_beliefs = results['final_beliefs']

    # Belief summary cards
    bel_cols = st.columns(3)
    for idx, (agent_name, belief) in enumerate(final_beliefs.items()):
        with bel_cols[idx]:
            icon = {'Medical Agency': '🏥', 'Rescue Agency': '🚒',
                    'Logistics Agency': '📦'}.get(agent_name, '🤖')
            ci_l, ci_h = belief['ci_lower'], belief['ci_upper']
            st.markdown(f"""
            <div class="agency-card">
                <h4>{icon} {agent_name}</h4>
                <b>P(cooperation succeeds) = {belief['posterior_mean']:.3f}</b><br>
                95% CI: [{ci_l:.3f}, {ci_h:.3f}]<br>
                Std Dev: ±{belief['posterior_std']:.3f}<br>
                Observations: {belief['n_observations']}<br>
                Successes: {belief['n_successes']} | Failures: {belief['n_failures']}
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Bayesian belief evolution lines
    bayesian_records = []
    for r in results['round_df'].itertuples():
        for agent, belief in r.bayesian_beliefs.items():
            bayesian_records.append({'round': r.round, 'agent': agent, 'belief': belief})

    bel_df = pd.DataFrame(bayesian_records)
    fig_bel = px.line(
        bel_df, x='round', y='belief', color='agent',
        color_discrete_map=colors,
        title="Bayesian Cooperation Belief Convergence Over Rounds",
        markers=True, labels={'belief': 'P(Cooperation Succeeds)', 'round': 'Simulation Round'}
    )
    fig_bel.add_hline(y=0.5, line_dash='dash', line_color='gray',
                       annotation_text='Decision Threshold (0.5)')
    fig_bel.update_layout(
        height=350, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation='h', y=-0.2)
    )
    st.plotly_chart(fig_bel, use_container_width=True)

    # Beta distribution visualization for final beliefs
    st.markdown("**Beta Posterior Distributions (Final Round):**")
    fig_beta = go.Figure()
    x_vals = np.linspace(0.01, 0.99, 200)
    beta_colors = ['#e74c3c', '#2980b9', '#27ae60']

    from scipy.stats import beta as beta_dist
    for idx, (agent_name, belief) in enumerate(final_beliefs.items()):
        a, b = belief['alpha'], belief['beta']
        y_vals = beta_dist.pdf(x_vals, a, b)
        fig_beta.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            fill='tozeroy', opacity=0.5,
            name=agent_name,
            line=dict(color=beta_colors[idx], width=2)
        ))

    fig_beta.update_layout(
        title="Beta Posterior: Uncertainty in Cooperation Success Probability",
        xaxis_title="P(Cooperation Succeeds)",
        yaxis_title="Probability Density",
        height=320, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', y=-0.25)
    )
    st.plotly_chart(fig_beta, use_container_width=True)

# ─────── TAB 5: DATA TABLES ───────
with tab5:
    st.markdown("### 📋 Simulation Data")

    st.markdown("**Round-by-Round Results:**")
    display_cols = ['round', 'severity', 'cooperation_ratio',
                    'fairness_index', 'avg_response_time', 'social_welfare']
    display_df = results['round_df'][display_cols].copy()
    display_df.columns = ['Round', 'Severity', 'Coop Rate', 'Fairness', 'Avg Response (min)', 'Social Welfare']
    display_df = display_df.round(3)
    st.dataframe(display_df, use_container_width=True, height=400)

    st.markdown("**Agent Timeline:**")
    tl_display = results['timeline_df'][['round', 'agent', 'strategy', 'allocation', 'payoff', 'bayesian_belief']].copy()
    tl_display.columns = ['Round', 'Agency', 'Strategy', 'Allocation', 'Payoff', 'Bayesian Belief']
    tl_display = tl_display.round(3)
    st.dataframe(tl_display, use_container_width=True, height=400)

    st.markdown("**Final Shapley Allocation:**")
    shapley_display = pd.DataFrame([
        {'Agency': k, 'Shapley Allocation (units)': round(v, 2),
         'Share (%)': f"{v/sum(results['final_shapley'].values())*100:.1f}%"}
        for k, v in results['final_shapley'].items()
    ])
    st.dataframe(shapley_display, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#666; font-size:0.85em; padding:10px">
    🎓 RDMU Examination Prototype · Concepts: Multi-Agent Systems · Game Theory (Shapley + Nash) · Monte Carlo Simulation · Bayesian Analysis<br>
    Built with Python + Streamlit · All simulations are stochastic — results vary with parameters
</div>
""", unsafe_allow_html=True)
