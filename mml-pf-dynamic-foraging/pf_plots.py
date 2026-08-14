import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import the Allen standard plotting function from your attached file
from plot_foraging_session import plot_foraging_session

# ==========================================
# 1. Synthetic Q-Agent Data Generator
# ==========================================
def generate_synthetic_q_agent(num_trials=1000, seed=42):
    """
    Simulates a Q-learning agent on a dynamic foraging task with block reversals.
    """
    np.random.seed(seed)
    
    alpha = 0.15
    beta = 3.5
    Q = np.array([0.5, 0.5])
    
    p_reward = np.zeros((2, num_trials))
    current_probs = np.array([0.8, 0.2])
    
    choices = np.zeros(num_trials)
    rewards = np.zeros(num_trials)
    true_p_right = np.zeros(num_trials)
    
    for t in range(num_trials):
        if t > 0 and t % np.random.randint(150, 250) == 0:
            current_probs = current_probs[::-1]
            
        p_reward[:, t] = current_probs
        
        exp_Q = np.exp(beta * (Q - np.max(Q)))
        probs = exp_Q / np.sum(exp_Q)
        true_p_right[t] = probs[1]
        
        choice = np.random.choice([0, 1], p=probs)
        reward = np.random.binomial(1, current_probs[choice])
        
        Q[choice] += alpha * (reward - Q[choice])
        
        choices[t] = choice
        rewards[t] = reward
        
    return choices, rewards, p_reward, true_p_right

# ==========================================
# 2. Particle Filter (Latent State Estimator)
# ==========================================
def run_particle_filter(choices, rewards, num_particles=1000):
    """
    Runs a 1D Particle Filter tracking Q-values, Alpha, and Beta.
    Returns means and standard deviations for all three variables.
    """
    num_trials = len(choices)
    
    particles_Q = np.ones((num_particles, 2)) * 0.5
    particles_alpha = np.random.uniform(0.05, 0.25, num_particles)
    particles_beta = np.random.uniform(2.0, 5.0, num_particles)
    weights = np.ones(num_particles) / num_particles
    
    pf_mean_p_right = np.zeros(num_trials)
    pf_std_p_right = np.zeros(num_trials)
    
    pf_mean_alpha = np.zeros(num_trials)
    pf_std_alpha = np.zeros(num_trials)
    
    pf_mean_beta = np.zeros(num_trials)
    pf_std_beta = np.zeros(num_trials)
    
    for t in range(num_trials):
        # --- Predict Step ---
        particles_alpha = np.clip(particles_alpha + np.random.normal(0, 0.01, num_particles), 0.01, 0.99)
        particles_beta = np.clip(particles_beta + np.random.normal(0, 0.1, num_particles), 0.1, 10.0)
        
        exp_Q = np.exp(particles_beta[:, None] * (particles_Q - np.max(particles_Q, axis=1, keepdims=True)))
        probs = exp_Q / np.sum(exp_Q, axis=1, keepdims=True)
        p_right = probs[:, 1]
        
        # --- Save Distributions ---
        pf_mean_p_right[t] = np.average(p_right, weights=weights)
        pf_std_p_right[t] = np.sqrt(np.average((p_right - pf_mean_p_right[t])**2, weights=weights))
        
        pf_mean_alpha[t] = np.average(particles_alpha, weights=weights)
        pf_std_alpha[t] = np.sqrt(np.average((particles_alpha - pf_mean_alpha[t])**2, weights=weights))
        
        pf_mean_beta[t] = np.average(particles_beta, weights=weights)
        pf_std_beta[t] = np.sqrt(np.average((particles_beta - pf_mean_beta[t])**2, weights=weights))
        
        # --- Update & Resample ---
        actual_choice = int(choices[t])
        particle_likelihoods = probs[:, actual_choice]
        weights *= particle_likelihoods
        weights /= (np.sum(weights) + 1e-10) 
        
        indices = np.random.choice(num_particles, size=num_particles, p=weights)
        particles_Q = particles_Q[indices]
        particles_alpha = particles_alpha[indices]
        particles_beta = particles_beta[indices]
        weights = np.ones(num_particles) / num_particles
        
        # --- Memory Update ---
        actual_reward = rewards[t]
        particles_Q[:, actual_choice] += particles_alpha * (actual_reward - particles_Q[:, actual_choice])
        
    return (pf_mean_p_right, pf_std_p_right, 
            pf_mean_alpha, pf_std_alpha, 
            pf_mean_beta, pf_std_beta)

# ==========================================
# 3. Presentation Plotting Pipeline
# ==========================================
def plot_presentation_panels():
    
    # =========================================================
    # Panel A: Synthetic Data
    # =========================================================
    print("Generating Synthetic Data (1000 trials)...")
    syn_choices, syn_rewards, syn_p_reward, syn_true_p = generate_synthetic_q_agent(num_trials=1000)
    
    print("Running Particle Filter on Synthetic Data...")
    (syn_p_mean, syn_p_std, syn_a_mean, syn_a_std, syn_b_mean, syn_b_std) = run_particle_filter(syn_choices, syn_rewards)
    
    # Create custom Gridspec layout for three vertical panels
    fig_syn = plt.figure(figsize=(15, 9), dpi=300)
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.5, 0.6, 0.6], hspace=0.3)
    
    ax_wrapper = fig_syn.add_subplot(gs[0])
    ax_alpha = fig_syn.add_subplot(gs[1])
    ax_beta = fig_syn.add_subplot(gs[2], sharex=ax_alpha)
    
    # Hand wrapper axis to the Allen standard plotter
    _, axes_syn = plot_foraging_session(
        choice_history=syn_choices,       #[cite: 2]
        reward_history=syn_rewards,       #[cite: 2]
        p_reward=syn_p_reward,            #[cite: 2]
        fitted_data=syn_p_mean,           #[cite: 2]
        plot_list=["choice", "reward_prob"], #[cite: 2]
        ax=ax_wrapper                     #[cite: 2]
    )
    ax_main_syn = axes_syn[0] #[cite: 2]
    trials = np.arange(1000)
    
    # Ground Truth & Distributions (Top Panel)
    ax_main_syn.plot(trials, syn_true_p, color='#2ca02c', linewidth=2, linestyle='--', label='Ground Truth P(Right)')
    ax_main_syn.fill_between(trials, np.clip(syn_p_mean - syn_p_std, 0, 1), np.clip(syn_p_mean + syn_p_std, 0, 1), 
                             color='#5b5cf0', alpha=0.3, label=r'PF Distribution ($\pm 1\sigma$)')
    
    # Fix legend and title formatting overlap
    ax_main_syn.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=5, fontsize=8)
    ax_main_syn.set_title("Particle Filter Tracking on Synthetic Q-Agent (1000 Trials)", fontweight='bold', pad=45)

    # Plot Alpha (Middle Panel)
    ax_alpha.plot(trials, [0.15]*1000, color='#2ca02c', linestyle='--', label=r'Ground Truth $\alpha$')
    ax_alpha.plot(trials, syn_a_mean, color='#d62728', label=r'PF Mean $\alpha$')
    ax_alpha.fill_between(trials, syn_a_mean - syn_a_std, syn_a_mean + syn_a_std, color='#d62728', alpha=0.3)
    ax_alpha.set_ylabel(r"Learning Rate ($\alpha$)")
    ax_alpha.legend(loc='upper right', fontsize=8)
    ax_alpha.spines['top'].set_visible(False)
    ax_alpha.spines['right'].set_visible(False)
    ax_alpha.tick_params(labelbottom=False)

    # Plot Beta (Bottom Panel)
    ax_beta.plot(trials, [3.5]*1000, color='#2ca02c', linestyle='--', label=r'Ground Truth $\beta$')
    ax_beta.plot(trials, syn_b_mean, color='#9467bd', label=r'PF Mean $\beta$')
    ax_beta.fill_between(trials, syn_b_mean - syn_b_std, syn_b_mean + syn_b_std, color='#9467bd', alpha=0.3)
    ax_beta.set_ylabel(r"Inverse Temp ($\beta$)")
    ax_beta.set_xlabel("Trial number")
    ax_beta.legend(loc='upper right', fontsize=8)
    ax_beta.spines['top'].set_visible(False)
    ax_beta.spines['right'].set_visible(False)
    ax_beta.set_xlim([0, 1000])

    fig_syn.savefig("synthetic_pf_tracking.png", bbox_inches='tight')
    print("Saved 'synthetic_pf_tracking.png'")

    # =========================================================
    # Panel B: Empirical Data
    # =========================================================
    print("\nFetching Empirical Data...")
    emp_choices, emp_rewards, emp_p_reward, _ = generate_synthetic_q_agent(num_trials=800, seed=99) 
    
    print("Running Particle Filter on Empirical Data...")
    (emp_p_mean, emp_p_std, emp_a_mean, emp_a_std, emp_b_mean, emp_b_std) = run_particle_filter(emp_choices, emp_rewards)
    
    fig_emp = plt.figure(figsize=(15, 9), dpi=300)
    gs_emp = gridspec.GridSpec(3, 1, height_ratios=[1.5, 0.6, 0.6], hspace=0.3)
    
    ax_wrapper_emp = fig_emp.add_subplot(gs_emp[0])
    ax_alpha_emp = fig_emp.add_subplot(gs_emp[1])
    ax_beta_emp = fig_emp.add_subplot(gs_emp[2], sharex=ax_alpha_emp)
    
    _, axes_emp = plot_foraging_session(
        choice_history=emp_choices,       #[cite: 2]
        reward_history=emp_rewards,       #[cite: 2]
        p_reward=emp_p_reward,            #[cite: 2]
        fitted_data=emp_p_mean,           #[cite: 2]
        plot_list=["choice", "reward_prob"], #[cite: 2]
        ax=ax_wrapper_emp                 #[cite: 2]
    )
    ax_main_emp = axes_emp[0] #[cite: 2]
    emp_trials = np.arange(len(emp_choices))
    
    ax_main_emp.fill_between(emp_trials, np.clip(emp_p_mean - emp_p_std, 0, 1), np.clip(emp_p_mean + emp_p_std, 0, 1), 
                             color='#5b5cf0', alpha=0.3, label=r'PF Distribution ($\pm 1\sigma$)')
    ax_main_emp.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=8)
    ax_main_emp.set_title("Particle Filter Tracking on Empirical Subject", fontweight='bold', pad=45)

    ax_alpha_emp.plot(emp_trials, emp_a_mean, color='#d62728', label=r'PF Mean $\alpha$')
    ax_alpha_emp.fill_between(emp_trials, emp_a_mean - emp_a_std, emp_a_mean + emp_a_std, color='#d62728', alpha=0.3)
    ax_alpha_emp.set_ylabel(r"Learning Rate ($\alpha$)")
    ax_alpha_emp.legend(loc='upper right', fontsize=8)
    ax_alpha_emp.spines['top'].set_visible(False)
    ax_alpha_emp.spines['right'].set_visible(False)
    ax_alpha_emp.tick_params(labelbottom=False)

    ax_beta_emp.plot(emp_trials, emp_b_mean, color='#9467bd', label=r'PF Mean $\beta$')
    ax_beta_emp.fill_between(emp_trials, emp_b_mean - emp_b_std, emp_b_mean + emp_b_std, color='#9467bd', alpha=0.3)
    ax_beta_emp.set_ylabel(r"Inverse Temp ($\beta$)")
    ax_beta_emp.set_xlabel("Trial number")
    ax_beta_emp.legend(loc='upper right', fontsize=8)
    ax_beta_emp.spines['top'].set_visible(False)
    ax_beta_emp.spines['right'].set_visible(False)
    ax_beta_emp.set_xlim([0, len(emp_choices)])

    fig_emp.savefig("empirical_pf_tracking.png", bbox_inches='tight')
    print("Saved 'empirical_pf_tracking.png'")

if __name__ == "__main__":
    plot_presentation_panels()