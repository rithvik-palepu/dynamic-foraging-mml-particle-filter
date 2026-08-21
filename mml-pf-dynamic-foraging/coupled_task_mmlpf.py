import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import aind_dynamic_foraging_database as db
from plot_foraging_session import plot_foraging_session

# ==========================================
# 1. MML Particle Filter (w/ Perseveration)
# ==========================================
def run_particle_filter_with_q_and_phi(choices, rewards, num_particles=1000):
    num_trials = len(choices)
    
    particles_Q = np.ones((num_particles, 2)) * 0.5
    particles_alpha = np.random.uniform(0.05, 0.25, num_particles)
    particles_beta = np.random.uniform(2.0, 5.0, num_particles)
    particles_phi = np.random.uniform(-1.0, 1.0, num_particles)
    weights = np.ones(num_particles) / num_particles
    
    pf_stats = {
        'p_right_mean': np.zeros(num_trials), 'p_right_std': np.zeros(num_trials),
        'alpha_mean': np.zeros(num_trials), 'alpha_std': np.zeros(num_trials),
        'beta_mean': np.zeros(num_trials), 'beta_std': np.zeros(num_trials),
        'phi_mean': np.zeros(num_trials), 'phi_std': np.zeros(num_trials),
        'q_left_mean': np.zeros(num_trials), 'q_right_mean': np.zeros(num_trials),
    }
    
    prev_c_R = 0
    prev_c_L = 0
    
    for t in range(num_trials):
        # Predict Step with Random Walk
        particles_alpha = np.clip(particles_alpha + np.random.normal(0, 0.02, num_particles), 0.01, 0.99)
        particles_beta = np.clip(particles_beta + np.random.normal(0, 0.2, num_particles), 0.1, 20.0)
        particles_phi = np.clip(particles_phi + np.random.normal(0, 0.1, num_particles), -5.0, 5.0)
        
        # Softmax Choice Probability with Spatial Perseveration
        log_odds = particles_beta * (particles_Q[:, 1] - particles_Q[:, 0]) + particles_phi * (prev_c_R - prev_c_L)
        prob_right = 1.0 / (1.0 + np.exp(-log_odds))
        
        # Store State
        pf_stats['p_right_mean'][t] = np.average(prob_right, weights=weights)
        pf_stats['p_right_std'][t] = np.sqrt(np.average((prob_right - pf_stats['p_right_mean'][t])**2, weights=weights))
        pf_stats['alpha_mean'][t] = np.average(particles_alpha, weights=weights)
        pf_stats['alpha_std'][t] = np.sqrt(np.average((particles_alpha - pf_stats['alpha_mean'][t])**2, weights=weights))
        pf_stats['beta_mean'][t] = np.average(particles_beta, weights=weights)
        pf_stats['beta_std'][t] = np.sqrt(np.average((particles_beta - pf_stats['beta_mean'][t])**2, weights=weights))
        pf_stats['phi_mean'][t] = np.average(particles_phi, weights=weights)
        pf_stats['phi_std'][t] = np.sqrt(np.average((particles_phi - pf_stats['phi_mean'][t])**2, weights=weights))
        pf_stats['q_left_mean'][t] = np.average(particles_Q[:, 0], weights=weights)
        pf_stats['q_right_mean'][t] = np.average(particles_Q[:, 1], weights=weights)
        
        actual_choice = int(choices[t])
        
        # Update Weights
        particle_likelihoods = prob_right if actual_choice == 1 else (1.0 - prob_right)
        weights *= particle_likelihoods
        
        w_sum = np.sum(weights)
        if w_sum < 1e-10:
            weights = np.ones(num_particles) / num_particles
        else:
            weights /= w_sum
            
        # ESS Resampling
        ess = 1.0 / np.sum(weights**2)
        if ess < (num_particles / 2.0):
            indices = np.random.choice(num_particles, size=num_particles, p=weights)
            particles_Q = particles_Q[indices]
            particles_alpha = particles_alpha[indices]
            particles_beta = particles_beta[indices]
            particles_phi = particles_phi[indices]
            weights = np.ones(num_particles) / num_particles
        
        # Q-Value Memory Update & Previous Choice Update
        actual_reward = rewards[t]
        particles_Q[:, actual_choice] += particles_alpha * (actual_reward - particles_Q[:, actual_choice])
        
        if actual_choice == 1:
            prev_c_R, prev_c_L = 1, 0
        else:
            prev_c_R, prev_c_L = 0, 1
        
    return pf_stats

# ==========================================
# 2. Synthetic Drifting Task Generator
# ==========================================
def generate_synthetic_drifting_task(total_trials=600):
    p_reward = np.zeros((2, total_trials))
    choices = np.zeros(total_trials)
    rewards = np.zeros(total_trials)
    
    true_q_history = np.zeros((2, total_trials))
    true_p_right = np.zeros(total_trials)
    
    # Drifting Ground Truth Parameters
    t_array = np.linspace(0, 1, total_trials)
    true_alpha = 0.2 + 0.1 * np.sin(2 * np.pi * t_array * 1.5)
    true_beta = 4.0 + 2.0 * np.cos(2 * np.pi * t_array * 1.0)
    true_phi = 1.0 + 1.5 * np.sin(2 * np.pi * t_array * 0.8) 
    
    agent_Q = np.array([0.5, 0.5])
    prev_c_R, prev_c_L = 0, 0
    
    t = 0
    state = 0 
    while t < total_trials:
        block_len = np.random.randint(40, 81)
        end_t = min(t + block_len, total_trials)
        
        if state == 0:
            p_reward[0, t:end_t] = 0.1 
            p_reward[1, t:end_t] = 0.8 
        else:
            p_reward[0, t:end_t] = 0.8 
            p_reward[1, t:end_t] = 0.1 
            
        for i in range(t, end_t):
            # Agent choice probability with perseveration
            log_odds = true_beta[i] * (agent_Q[1] - agent_Q[0]) + true_phi[i] * (prev_c_R - prev_c_L)
            prob_right = 1.0 / (1.0 + np.exp(-log_odds))
            true_p_right[i] = prob_right
            
            # Execute choice
            choice = 1 if np.random.rand() < prob_right else 0
            choices[i] = choice
            
            if choice == 1:
                prev_c_R, prev_c_L = 1, 0
            else:
                prev_c_R, prev_c_L = 0, 1
            
            # Environment reward
            reward = 1 if np.random.rand() < p_reward[choice, i] else 0
            rewards[i] = reward
            
            # Agent update
            agent_Q[choice] += true_alpha[i] * (reward - agent_Q[choice])
            true_q_history[:, i] = agent_Q
            
        t = end_t
        state = 1 - state 
        
    return choices, rewards, p_reward, true_alpha, true_beta, true_phi, true_q_history, true_p_right

# ==========================================
# 3. Plotting Logic (5 Panels)
# ==========================================
def generate_latent_plot(choices, rewards, p_reward, title, filename, 
                         true_alpha=None, true_beta=None, true_phi=None, true_q=None, true_p_right=None):
    print(f"Running MMLPF for {title}...")
    pf = run_particle_filter_with_q_and_phi(choices, rewards)
    trials = np.arange(len(choices))
    
    fig = plt.figure(figsize=(14, 14), dpi=300)
    gs = gridspec.GridSpec(5, 1, height_ratios=[1.5, 0.8, 0.6, 0.6, 0.6], hspace=0.35)
    
    ax0 = fig.add_subplot(gs[0])
    ax_q = fig.add_subplot(gs[1], sharex=ax0)
    ax_alpha = fig.add_subplot(gs[2], sharex=ax0)
    ax_beta = fig.add_subplot(gs[3], sharex=ax0)
    ax_phi = fig.add_subplot(gs[4], sharex=ax0)
    
    # 1. Behavioral Top Panel
    _, axes = plot_foraging_session(
        choice_history=choices, reward_history=rewards, p_reward=p_reward,
        plot_list=["choice"], ax=ax0
    )
    ax0 = axes[0]
    
    ax0.plot(trials, p_reward[1, :], color='y', linewidth=2.0, drawstyle='steps-post', label='Base rew. prob.')
    
    if true_p_right is not None:
        ax0.plot(trials, true_p_right, color='k', linestyle='--', linewidth=2.0, alpha=0.6, label='True Agent P(Right)')
    
    ax0.plot(trials, pf['p_right_mean'], color='#5b5cf0', linewidth=2.0, alpha=0.8, label='MMLPF P(Right)')
    
    ax0.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4 if true_p_right is not None else 3, fontsize=9)
    ax0.set_title(title, fontweight='bold', pad=55, fontsize=14)
    
    # 2. Q-Values
    if true_q is not None:
        ax_q.plot(trials, true_q[0, :], color='k', linestyle='--', linewidth=1.5, alpha=0.5, label='True Q(Left)')
        ax_q.plot(trials, true_q[1, :], color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='True Q(Right)')
        
    ax_q.plot(trials, pf['q_left_mean'], color='#d62728', linewidth=1.5, label='Filter Q(Left)')
    ax_q.plot(trials, pf['q_right_mean'], color='#1f77b4', linewidth=1.5, label='Filter Q(Right)')
    ax_q.set_ylabel("Action Value (Q)")
    
    legend_cols = 4 if true_q is not None else 2
    ax_q.legend(loc='upper right', fontsize=9, ncol=legend_cols)
    ax_q.spines['top'].set_visible(False)
    ax_q.spines['right'].set_visible(False)
    ax_q.tick_params(labelbottom=False)
    
    # 3. Learning Rate (Alpha)
    if true_alpha is not None:
        ax_alpha.plot(trials, true_alpha, color='k', linestyle='--', linewidth=1.5, label=r'Ground Truth $\alpha$')
        
    ax_alpha.plot(trials, pf['alpha_mean'], color='#2ca02c', linewidth=1.5, label=r'Filter $\alpha$')
    ax_alpha.fill_between(trials, pf['alpha_mean'] - pf['alpha_std'], pf['alpha_mean'] + pf['alpha_std'], color='#2ca02c', alpha=0.3)
    ax_alpha.set_ylabel(r"Learning Rate ($\alpha$)")
    ax_alpha.legend(loc='upper right', fontsize=9, ncol=2 if true_alpha is not None else 1)
    ax_alpha.spines['top'].set_visible(False)
    ax_alpha.spines['right'].set_visible(False)
    ax_alpha.tick_params(labelbottom=False)
    
    # 4. Inverse Temp (Beta)
    if true_beta is not None:
        ax_beta.plot(trials, true_beta, color='k', linestyle='--', linewidth=1.5, label=r'Ground Truth $\beta$')
        
    ax_beta.plot(trials, pf['beta_mean'], color='#9467bd', linewidth=1.5, label=r'Filter $\beta$')
    ax_beta.fill_between(trials, pf['beta_mean'] - pf['beta_std'], pf['beta_mean'] + pf['beta_std'], color='#9467bd', alpha=0.3)
    ax_beta.set_ylabel(r"Inverse Temp ($\beta$)")
    ax_beta.legend(loc='upper right', fontsize=9, ncol=2 if true_beta is not None else 1)
    ax_beta.spines['top'].set_visible(False)
    ax_beta.spines['right'].set_visible(False)
    ax_beta.tick_params(labelbottom=False)
    
    # 5. Perseveration (Phi)
    if true_phi is not None:
        ax_phi.plot(trials, true_phi, color='k', linestyle='--', linewidth=1.5, label=r'Ground Truth $\phi$')
        
    ax_phi.plot(trials, pf['phi_mean'], color='#d95f02', linewidth=1.5, label=r'Filter $\phi$')
    ax_phi.fill_between(trials, pf['phi_mean'] - pf['phi_std'], pf['phi_mean'] + pf['phi_std'], color='#d95f02', alpha=0.3)
    ax_phi.set_ylabel(r"Perseveration ($\phi$)")
    ax_phi.set_xlabel("Trial Number", fontsize=11)
    ax_phi.legend(loc='upper right', fontsize=9, ncol=2 if true_phi is not None else 1)
    ax_phi.spines['top'].set_visible(False)
    ax_phi.spines['right'].set_visible(False)
    ax_phi.set_xlim([0, len(choices)])
    
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

# ==========================================
# 4. Execution
# ==========================================
if __name__ == "__main__":
    # --- SYNTHETIC TASK ---
    syn_choices, syn_rewards, syn_p_reward, true_a, true_b, true_p, true_q, true_p_right = generate_synthetic_drifting_task(total_trials=600)
    generate_latent_plot(syn_choices, syn_rewards, syn_p_reward, 
                         "MMLPF Tracking: Drifting Synthetic Agent (w/ Perseveration)", 
                         "synthetic_drifting_mmlpf.png",
                         true_alpha=true_a, true_beta=true_b, true_phi=true_p, 
                         true_q=true_q, true_p_right=true_p_right)
    
    # --- EMPIRICAL TASK ---
    target_subject_id = '856493'
    target_date = '2026-07-09'
    
    print(f"\nQuerying DB for Empirical Subject {target_subject_id} on Date {target_date}...")
    
    try:
        # Attempt 1: String formatting
        all_sessions = db.select_sessions(where=f"subject_id = '{target_subject_id}'")
        if len(all_sessions) == 0:
            # Attempt 2: Integer formatting
            all_sessions = db.select_sessions(where=f"subject_id = {target_subject_id}")
    except Exception as e:
        print(f"WHERE clause failed: {e}. Attempting broad fetch...")
        all_sessions = db.select_sessions()
        all_sessions = all_sessions[all_sessions['subject_id'].astype(str) == target_subject_id]
        
    if len(all_sessions) > 0:
        # Convert to string and match date to avoid datetime64 dtype issues
        target_session = all_sessions[all_sessions['session_date'].astype(str).str.contains(target_date)]
        
        if len(target_session) > 0:
            target_session = target_session.iloc[0:1] # Safely grab the single row
            
            trials_df = db.fetch_trials(target_session, columns=["animal_response", "earned_reward", "reward_probabilityL", "reward_probabilityR"])
            
            # Guardrail: Check if trials actually downloaded
            if len(trials_df) > 0:
                valid_trials = trials_df[trials_df['animal_response'] != 2].copy() 
                
                emp_choices = valid_trials['animal_response'].astype(int).values
                emp_rewards = valid_trials['earned_reward'].astype(int).values
                
                if 'reward_probabilityL' in valid_trials.columns:
                    emp_p_left = valid_trials['reward_probabilityL'].values
                    emp_p_right = valid_trials['reward_probabilityR'].values
                    emp_p_reward = np.vstack([emp_p_left, emp_p_right])
                else:
                    print("Warning: True probabilities not found in DB pull. Base rew. prob. line will be flat.")
                    emp_p_reward = np.zeros((2, len(emp_choices)))
                    
                generate_latent_plot(emp_choices, emp_rewards, emp_p_reward,
                                     f"MMLPF Tracking: Empirical Subject {target_subject_id} (Uncoupled Expert w/ Perseveration)",
                                     f"empirical_{target_subject_id}_mmlpf_with_phi.png")
            else:
                print(f"Error: Session found, but fetch_trials returned 0 rows for {target_subject_id} on {target_date}.")
        else:
            print(f"Could not locate the specific session on {target_date} for Subject {target_subject_id}.")
            print(f"Available dates in your local DB: {all_sessions['session_date'].unique()}")
    else:
        print(f"No database entries found for Subject {target_subject_id}.")