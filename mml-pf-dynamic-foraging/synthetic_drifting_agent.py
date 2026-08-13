import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import psytrack

# Force headless plotting for HPC
matplotlib.use('Agg')

# Import your native MMLPF architecture
from empirical_drifting_agent_mml_pf import calculate_nll_fast

# ==========================================
# 1. Synthetic Data Generator
# ==========================================
def generate_drifting_agent_data(env_type='volatile', num_sessions=25, trials_per_session=500, seed=42):
    """
    Simulates a 2-armed bandit task where learning rate and temp undergo a Gaussian random walk.
    env_type: 'smooth' (continuous reward drift) or 'volatile' (sudden block switches).
    """
    np.random.seed(seed)
    
    sigma_alpha = 0.05
    sigma_beta = 0.5
    
    alpha = 0.5
    beta = 5.0
    Q = np.array([0.5, 0.5])
    
    reward_probs = np.array([0.8, 0.2])
    
    all_sessions_data = []
    
    for session_id in range(num_sessions):
        choices, rewards = [], []
        
        for t in range(trials_per_session):
            # --- Environmental Dynamics ---
            if env_type == 'volatile':
                # Trigger sudden block switch
                if t > 0 and t % 100 == 0:
                    reward_probs = reward_probs[::-1]
            elif env_type == 'smooth':
                # Continuous, slow random walk of reward probabilities
                reward_probs[0] = np.clip(reward_probs[0] + np.random.normal(0, 0.03), 0.1, 0.9)
                reward_probs[1] = 1.0 - reward_probs[0]
                
            # --- Latent Cognitive Drift ---
            alpha = np.clip(alpha + np.random.normal(0, sigma_alpha), 0.01, 0.99)
            beta = np.clip(beta + np.random.normal(0, sigma_beta), 0.1, 20.0)
            
            # --- Agent Decision ---
            exp_Q = np.exp(beta * (Q - np.max(Q)))
            probs = exp_Q / np.sum(exp_Q)
            choice = np.random.choice([0, 1], p=probs)
            
            # --- Environment Response & Memory Update ---
            reward = np.random.binomial(1, reward_probs[choice])
            Q[choice] += alpha * (reward - Q[choice])
            
            choices.append(choice)
            rewards.append(reward)
            
        # Store session data
        df = pd.DataFrame({'session_id': session_id + 1, 'choice': choices, 'reward': rewards})
        all_sessions_data.append(df)
        
    return pd.concat(all_sessions_data, ignore_index=True)

# ==========================================
# 2. Walk-Forward Cross-Validation Pipeline
# ==========================================
def run_walk_forward_cv(data_df):
    num_sessions = data_df['session_id'].max()
    mmlpf_nll_history = []
    psytrack_nll_history = []
    session_timeline = []
    
    # Start CV at session 3 to ensure enough history for initial training
    for target_session in range(3, num_sessions + 1):
        print(f"  -> Walk-Forward: Training on 1 to {target_session-1}, Testing on {target_session}")
        
        # Split Data
        train_data = data_df[data_df['session_id'] < target_session]
        test_data = data_df[data_df['session_id'] == target_session]
        
        train_choices = train_data['choice'].values
        train_rewards = train_data['reward'].values
        test_choices = test_data['choice'].values
        test_rewards = test_data['reward'].values
        
        # ------------------------------------------------
        # Model 1: MMLPF Architecture
        # ------------------------------------------------
        # Optimize volatilities on historical data (added updating='deferred' to silence warning)
        mml_opt = differential_evolution(
            func=calculate_nll_fast, 
            bounds=[(0.001, 0.2), (0.001, 0.2)], 
            args=(train_choices, train_rewards),
            maxiter=20, popsize=10, tol=0.05, workers=-1, updating='deferred', disp=False
        )
        opt_sigma_alpha, opt_sigma_beta = mml_opt.x
        
        # Test on unseen future session
        out_of_sample_mml_nll = calculate_nll_fast(
            (opt_sigma_alpha, opt_sigma_beta), test_choices, test_rewards, num_particles=1000
        )
        # Normalize to average NLL per trial for fair comparison
        mmlpf_nll_history.append(out_of_sample_mml_nll / len(test_choices))
        
        # ------------------------------------------------
        # Model 2: PsyTrack Baseline
        # ------------------------------------------------
        # Added dayLength to dictionary to silence the sigDay warning
        train_dict = {
            'y': train_choices + 1, 
            'inputs': {'reward_history': np.expand_dims(train_rewards, axis=1)},
            'dayLength': np.array([len(train_choices)]) 
        }
        weights = {'bias': 1, 'reward_history': 1}
        K = np.sum([weights[k] for k in weights.keys()])
        hyper_guess = {'sigma': [2**-5]*K, 'sigInit': 2**5, 'sigDay': 2**-5}
        
        try:
            # Capture wMode (the hidden weight states) from the optimization
            hyp, evd, wMode, hess_info = psytrack.hyperOpt(train_dict, hyper_guess, weights, ['sigma', 'sigDay'], showOpt=0)
            
            # Extract final weights from the very last trial of the training set
            final_weights = wMode[:, -1]
            
            # Construct Test X Matrix: Row 0 is Bias (1s), Row 1 is Test Rewards
            X_test = np.vstack([np.ones(len(test_choices)), test_rewards])
            
            # Calculate predicted probabilities for choice == 1 (Right)
            log_odds = np.dot(final_weights, X_test)
            p_right = 1.0 / (1.0 + np.exp(-log_odds))
            
            # Calculate NLL based on the actual choices made in the test set
            p_chosen = np.where(test_choices == 1, p_right, 1.0 - p_right)
            p_chosen = np.clip(p_chosen, 1e-16, 1.0 - 1e-16)
            out_of_sample_psy_nll = -np.sum(np.log(p_chosen))
            
            psytrack_nll_history.append(out_of_sample_psy_nll / len(test_choices))
            
        except Exception as e:
            print(f"     PsyTrack fit failed: {e}")
            psytrack_nll_history.append(np.nan)
            
        session_timeline.append(target_session)
        
    return session_timeline, mmlpf_nll_history, psytrack_nll_history

# ==========================================
# 3. Execution & Presentation Plotting
# ==========================================
if __name__ == "__main__":
    print("Generating Volatile Environment (MMLPF Favored)...")
    volatile_data = generate_drifting_agent_data(env_type='volatile')
    print("Running Walk-Forward CV on Volatile Data...")
    vol_sessions, vol_mml_nll, vol_psy_nll = run_walk_forward_cv(volatile_data)
    
    print("\nGenerating Smooth Environment (PsyTrack Favored)...")
    smooth_data = generate_drifting_agent_data(env_type='smooth')
    print("Running Walk-Forward CV on Smooth Data...")
    smooth_sessions, smooth_mml_nll, smooth_psy_nll = run_walk_forward_cv(smooth_data)
    
    print("\nGenerating Presentation Graphic...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # Plot 1: Volatile Environment
    ax1.plot(vol_sessions, vol_mml_nll, color='#5b5cf0', linewidth=3, marker='o', label='MMLPF Architecture')
    ax1.plot(vol_sessions, vol_psy_nll, color='gray', linewidth=3, marker='s', linestyle='--', label='PsyTrack Baseline')
    ax1.set_title('Volatile Environment\n(Sudden Reward Reversals)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Test Session Number', fontsize=12)
    ax1.set_ylabel('Out-of-Sample NLL per Trial\n(Lower is Better)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # Plot 2: Smooth Environment
    ax2.plot(smooth_sessions, smooth_mml_nll, color='#5b5cf0', linewidth=3, marker='o', label='MMLPF Architecture')
    ax2.plot(smooth_sessions, smooth_psy_nll, color='gray', linewidth=3, marker='s', linestyle='--', label='PsyTrack Baseline')
    ax2.set_title('Smooth Environment\n(Continuous Gaussian Drift)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Test Session Number', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    save_path = "synthetic_boundary_test.png"
    plt.savefig(save_path, dpi=300)
    print(f"Success! Boundary test saved to '{save_path}'")