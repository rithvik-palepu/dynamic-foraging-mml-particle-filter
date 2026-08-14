import numpy as np #[cite: 3]
import pandas as pd #[cite: 3]
import matplotlib #[cite: 3]
import matplotlib.pyplot as plt #[cite: 3]
from scipy.optimize import differential_evolution #[cite: 3]
import psytrack #[cite: 3]

# Force headless plotting for HPC
matplotlib.use('Agg') #[cite: 3]

# Import your native MMLPF architecture
from empirical_drifting_agent_mml_pf import calculate_nll_fast #[cite: 3]

# ==========================================
# 1. Synthetic Data Generator
# ==========================================
def generate_drifting_agent_data(env_type='volatile', num_sessions=25, trials_per_session=500, seed=42): #[cite: 3]
    """
    Simulates a 2-armed bandit task where learning rate and temp undergo a Gaussian random walk.
    env_type: 'smooth' (continuous reward drift) or 'volatile' (sudden block switches).
    """
    np.random.seed(seed) #[cite: 3]
    
    sigma_alpha = 0.05 #[cite: 3]
    sigma_beta = 0.5 #[cite: 3]
    
    alpha = 0.5 #[cite: 3]
    beta = 5.0 #[cite: 3]
    Q = np.array([0.5, 0.5]) #[cite: 3]
    
    reward_probs = np.array([0.8, 0.2]) #[cite: 3]
    
    all_sessions_data = [] #[cite: 3]
    
    for session_id in range(num_sessions): #[cite: 3]
        choices, rewards = [], [] #[cite: 3]
        
        for t in range(trials_per_session): #[cite: 3]
            # --- Environmental Dynamics ---
            if env_type == 'volatile': #[cite: 3]
                # Trigger sudden block switch
                if t > 0 and t % 100 == 0: #[cite: 3]
                    reward_probs = reward_probs[::-1] #[cite: 3]
            elif env_type == 'smooth': #[cite: 3]
                # Continuous, slow random walk of reward probabilities
                reward_probs[0] = np.clip(reward_probs[0] + np.random.normal(0, 0.03), 0.1, 0.9) #[cite: 3]
                reward_probs[1] = 1.0 - reward_probs[0] #[cite: 3]
                
            # --- Latent Cognitive Drift ---
            alpha = np.clip(alpha + np.random.normal(0, sigma_alpha), 0.01, 0.99) #[cite: 3]
            beta = np.clip(beta + np.random.normal(0, sigma_beta), 0.1, 20.0) #[cite: 3]
            
            # --- Agent Decision ---
            exp_Q = np.exp(beta * (Q - np.max(Q))) #[cite: 3]
            probs = exp_Q / np.sum(exp_Q) #[cite: 3]
            choice = np.random.choice([0, 1], p=probs) #[cite: 3]
            
            # --- Environment Response & Memory Update ---
            reward = np.random.binomial(1, reward_probs[choice]) #[cite: 3]
            Q[choice] += alpha * (reward - Q[choice]) #[cite: 3]
            
            choices.append(choice) #[cite: 3]
            rewards.append(reward) #[cite: 3]
            
        # Store session data
        df = pd.DataFrame({'session_id': session_id + 1, 'choice': choices, 'reward': rewards}) #[cite: 3]
        all_sessions_data.append(df) #[cite: 3]
        
    return pd.concat(all_sessions_data, ignore_index=True) #[cite: 3]

# ==========================================
# 2. Walk-Forward Cross-Validation Pipeline
# ==========================================
def run_walk_forward_cv(data_df): #[cite: 3]
    num_sessions = data_df['session_id'].max() #[cite: 3]
    mmlpf_nll_history = [] #[cite: 3]
    psytrack_nll_history = [] #[cite: 3]
    session_timeline = [] #[cite: 3]
    
    # Start CV at session 3 to ensure enough history for initial training
    for target_session in range(3, num_sessions + 1): #[cite: 3]
        print(f"  -> Walk-Forward: Training on 1 to {target_session-1}, Testing on {target_session}") #[cite: 3]
        
        # Split Data
        train_data = data_df[data_df['session_id'] < target_session] #[cite: 3]
        test_data = data_df[data_df['session_id'] == target_session] #[cite: 3]
        
        train_choices = train_data['choice'].values #[cite: 3]
        train_rewards = train_data['reward'].values #[cite: 3]
        test_choices = test_data['choice'].values #[cite: 3]
        test_rewards = test_data['reward'].values #[cite: 3]
        
        # ------------------------------------------------
        # Model 1: MMLPF Architecture
        # ------------------------------------------------
        # Optimize volatilities on historical data (added updating='deferred' to silence warning)
        mml_opt = differential_evolution(
            func=calculate_nll_fast, 
            bounds=[(0.001, 0.2), (0.001, 0.2)], 
            args=(train_choices, train_rewards),
            maxiter=20, popsize=10, tol=0.05, workers=-1, updating='deferred', disp=False
        ) #[cite: 3]
        opt_sigma_alpha, opt_sigma_beta = mml_opt.x #[cite: 3]
        
        # Test on unseen future session
        out_of_sample_mml_nll = calculate_nll_fast(
            (opt_sigma_alpha, opt_sigma_beta), test_choices, test_rewards, num_particles=1000
        ) #[cite: 3]
        # Normalize to average NLL per trial for fair comparison
        mmlpf_nll_history.append(out_of_sample_mml_nll / len(test_choices)) #[cite: 3]
        
        # ------------------------------------------------
        # Model 2: PsyTrack Baseline
        # ------------------------------------------------
        # Added dayLength to dictionary to silence the sigDay warning
        train_dict = {
            'y': train_choices + 1, 
            'inputs': {'reward_history': np.expand_dims(train_rewards, axis=1)},
            'dayLength': np.array([len(train_choices)]) 
        } #[cite: 3]
        weights = {'bias': 1, 'reward_history': 1} #[cite: 3]
        K = np.sum([weights[k] for k in weights.keys()]) #[cite: 3]
        hyper_guess = {'sigma': [2**-5]*K, 'sigInit': 2**5, 'sigDay': 2**-5} #[cite: 3]
        
        try:
            # Capture wMode (the hidden weight states) from the optimization
            hyp, evd, wMode, hess_info = psytrack.hyperOpt(train_dict, hyper_guess, weights, ['sigma', 'sigDay'], showOpt=0) #[cite: 3]
            
            # Extract final weights from the very last trial of the training set
            final_weights = wMode[:, -1] #[cite: 3]
            
            # ---> THE FIX: Create a shifted reward history array <---
            # Get the very last reward from the training set
            last_train_reward = train_rewards[-1]
            
            # Shift the test rewards right by 1, inserting the last train reward at index 0
            shifted_test_rewards = np.insert(test_rewards[:-1], 0, last_train_reward)
            
            # Construct Test X Matrix using the SHIFTED rewards
            X_test = np.vstack([np.ones(len(test_choices)), shifted_test_rewards])
            
            # Calculate predicted probabilities for choice == 1 (Right)
            log_odds = np.dot(final_weights, X_test) #[cite: 3]
            p_right = 1.0 / (1.0 + np.exp(-log_odds)) #[cite: 3]
            
            # Calculate NLL based on the actual choices made in the test set
            p_chosen = np.where(test_choices == 1, p_right, 1.0 - p_right) #[cite: 3]
            p_chosen = np.clip(p_chosen, 1e-16, 1.0 - 1e-16) #[cite: 3]
            out_of_sample_psy_nll = -np.sum(np.log(p_chosen)) #[cite: 3]
            
            psytrack_nll_history.append(out_of_sample_psy_nll / len(test_choices)) #[cite: 3]
            
        except Exception as e: #[cite: 3]
            print(f"     PsyTrack fit failed: {e}") #[cite: 3]
            psytrack_nll_history.append(np.nan) #[cite: 3]
            
        session_timeline.append(target_session) #[cite: 3]
        
    return session_timeline, mmlpf_nll_history, psytrack_nll_history #[cite: 3]

# ==========================================
# 3. Execution & Presentation Plotting
# ==========================================
if __name__ == "__main__":
    print("Generating Volatile Environment (MMLPF Favored)...") #[cite: 3]
    volatile_data = generate_drifting_agent_data(env_type='volatile') #[cite: 3]
    print("Running Walk-Forward CV on Volatile Data...") #[cite: 3]
    vol_sessions, vol_mml_nll, vol_psy_nll = run_walk_forward_cv(volatile_data) #[cite: 3]
    
    print("\nGenerating Smooth Environment (PsyTrack Favored)...") #[cite: 3]
    smooth_data = generate_drifting_agent_data(env_type='smooth') #[cite: 3]
    print("Running Walk-Forward CV on Smooth Data...") #[cite: 3]
    smooth_sessions, smooth_mml_nll, smooth_psy_nll = run_walk_forward_cv(smooth_data) #[cite: 3]
    
    print("\nGenerating Presentation Graphic...") #[cite: 3]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True) #[cite: 3]
    
    # Plot 1: Volatile Environment
    ax1.plot(vol_sessions, vol_mml_nll, color='#5b5cf0', linewidth=3, marker='o', label='MMLPF Architecture') #[cite: 3]
    ax1.plot(vol_sessions, vol_psy_nll, color='gray', linewidth=3, marker='s', linestyle='--', label='PsyTrack Baseline') #[cite: 3]
    ax1.set_title('Volatile Environment\n(Sudden Reward Reversals)', fontsize=14, fontweight='bold') #[cite: 3]
    ax1.set_xlabel('Test Session Number', fontsize=12) #[cite: 3]
    ax1.set_ylabel('Out-of-Sample NLL per Trial\n(Lower is Better)', fontsize=12) #[cite: 3]
    ax1.grid(True, linestyle='--', alpha=0.5) #[cite: 3]
    ax1.legend() #[cite: 3]

    # Plot 2: Smooth Environment
    ax2.plot(smooth_sessions, smooth_mml_nll, color='#5b5cf0', linewidth=3, marker='o', label='MMLPF Architecture') #[cite: 3]
    ax2.plot(smooth_sessions, smooth_psy_nll, color='gray', linewidth=3, marker='s', linestyle='--', label='PsyTrack Baseline') #[cite: 3]
    ax2.set_title('Smooth Environment\n(Continuous Gaussian Drift)', fontsize=14, fontweight='bold') #[cite: 3]
    ax2.set_xlabel('Test Session Number', fontsize=12) #[cite: 3]
    ax2.grid(True, linestyle='--', alpha=0.5) #[cite: 3]
    ax2.legend() #[cite: 3]
    
    plt.tight_layout() #[cite: 3]
    save_path = "synthetic_boundary_test.png" #[cite: 3]
    plt.savefig(save_path, dpi=300) #[cite: 3]
    print(f"Success! Boundary test saved to '{save_path}'") #[cite: 3]