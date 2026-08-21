import numpy as np
import pandas as pd
import psytrack
import aind_dynamic_foraging_database as db
from joblib import Parallel, delayed
import time

# ==========================================
# 1. MML Particle Filter (Adaptive)
# ==========================================
def run_particle_filter_with_q(choices, rewards, num_particles=1000):
    num_trials = len(choices)
    
    particles_Q = np.ones((num_particles, 2)) * 0.5
    particles_alpha = np.random.uniform(0.05, 0.25, num_particles)
    particles_beta = np.random.uniform(2.0, 5.0, num_particles)
    weights = np.ones(num_particles) / num_particles
    
    nll_history = np.zeros(num_trials)
    
    for t in range(num_trials):
        # Predict Step
        particles_alpha = np.clip(particles_alpha + np.random.normal(0, 0.01, num_particles), 0.01, 0.99)
        particles_beta = np.clip(particles_beta + np.random.normal(0, 0.1, num_particles), 0.1, 20.0) 
        
        exp_Q = np.exp(particles_beta[:, None] * (particles_Q - np.max(particles_Q, axis=1, keepdims=True)))
        probs = exp_Q / np.sum(exp_Q, axis=1, keepdims=True)
        
        p_right_mean = np.average(probs[:, 1], weights=weights)
        actual_choice = int(choices[t])
        
        # Log-Likelihood for Scoring (Cross-Entropy in Bits)
        predicted_prob = np.clip(p_right_mean if actual_choice == 1 else (1.0 - p_right_mean), 1e-16, 1.0)
        nll_history[t] = -np.log2(predicted_prob) 
        
        # Update Weights
        particle_likelihoods = probs[np.arange(num_particles), actual_choice]
        weights *= particle_likelihoods
        
        # Safe Normalization
        w_sum = np.sum(weights)
        if w_sum < 1e-10:
            weights = np.ones(num_particles) / num_particles
        else:
            weights /= w_sum
            
        # ESS-Triggered Resampling
        ess = 1.0 / np.sum(weights**2)
        if ess < (num_particles / 2.0):
            indices = np.random.choice(num_particles, size=num_particles, p=weights)
            particles_Q = particles_Q[indices]
            particles_alpha = particles_alpha[indices]
            particles_beta = particles_beta[indices]
            weights = np.ones(num_particles) / num_particles
        
        # Memory Update
        actual_reward = rewards[t]
        particles_Q[:, actual_choice] += particles_alpha * (actual_reward - particles_Q[:, actual_choice])
        
    return nll_history

# ==========================================
# 2. PsyTrack Feature Extraction
# ==========================================
def build_psytrack_features(choices, rewards, bridge_choice=0, bridge_reward=0):
    past_choice = np.insert(choices[:-1], 0, bridge_choice)
    past_reward = np.insert(rewards[:-1], 0, bridge_reward)
    
    rewarded_choice = np.zeros(len(choices))
    rewarded_choice[(past_choice == 1) & (past_reward == 1)] = 1
    rewarded_choice[(past_choice == 0) & (past_reward == 1)] = -1
    
    unrewarded_choice = np.zeros(len(choices))
    unrewarded_choice[(past_choice == 1) & (past_reward == 0)] = 1
    unrewarded_choice[(past_choice == 0) & (past_reward == 0)] = -1
    
    return rewarded_choice, unrewarded_choice

# ==========================================
# 3. Walk-Forward CV Logic per Mouse
# ==========================================
def process_mouse(subject_id, session_data_list):
    print(f"[{subject_id}] Starting 10-Session Evaluation...")
    results = []
    
    # We start at index 1 because index 0 is our first training session
    for i in range(1, len(session_data_list)):
        train_sessions = session_data_list[:i]
        test_session = session_data_list[i]
        
        train_choices = np.concatenate([s[0] for s in train_sessions])
        train_rewards = np.concatenate([s[1] for s in train_sessions])
        session_lengths = [len(s[0]) for s in train_sessions] 
        
        test_choices, test_rewards = test_session
        num_test_trials = len(test_choices)
        
        # 1. Evaluate MMLPF (Adaptive OOS)
        mml_nll_history = run_particle_filter_with_q(test_choices, test_rewards)
        mml_bits_per_trial = np.mean(mml_nll_history)
        
        # 2. Train PsyTrack Hyperparameters
        train_rew, train_unrew = build_psytrack_features(train_choices, train_rewards)
        train_dict = {
            'y': train_choices + 1, 
            'inputs': {'rewarded': train_rew.reshape(-1, 1), 'unrewarded': train_unrew.reshape(-1, 1)},
            'dayLength': np.array(session_lengths) 
        }
        weights_dict = {'bias': 1, 'rewarded': 1, 'unrewarded': 1}
        K = np.sum([weights_dict[k] for k in weights_dict.keys()])
        hyper_guess = {'sigma': [2**-5]*K, 'sigInit': 2**5, 'sigDay': 2**-5}
        
        try:
            train_hyp, _, _, _ = psytrack.hyperOpt(train_dict, hyper_guess, weights_dict, ['sigma', 'sigDay'], showOpt=0)
            
            # 3. Evaluate PsyTrack (Matched OOS Forward Filtering)
            test_rew, test_unrew = build_psytrack_features(test_choices, test_rewards, train_choices[-1], train_rewards[-1])
            test_dict = {
                'y': test_choices + 1, 
                'inputs': {'rewarded': test_rew.reshape(-1, 1), 'unrewarded': test_unrew.reshape(-1, 1)},
                'dayLength': np.array([num_test_trials]) 
            }
            
            # Extract weights using getMAP
            wMode_test = psytrack.getMAP.getMAP(test_dict, train_hyp, weights_dict)[0]
            
            # Shift weights to ensure one-step causal prediction
            w_shifted = np.hstack([wMode_test[:, 0:1], wMode_test[:, :-1]])
            
            X_test = np.vstack([np.ones(num_test_trials), test_rew, test_unrew])
            log_odds = np.sum(w_shifted * X_test, axis=0) 
            psy_p_right = 1.0 / (1.0 + np.exp(-log_odds))
            
            p_chosen = np.where(test_choices == 1, psy_p_right, 1.0 - psy_p_right)
            p_chosen = np.clip(p_chosen, 1e-16, 1.0)
            psy_bits_per_trial = np.mean(-np.log2(p_chosen))
            
        except Exception as e:
            print(f"[{subject_id}] PsyTrack Error on Session {i+1}: {e}")
            psy_bits_per_trial = np.nan
            
        results.append({
            "subject_id": subject_id,
            "test_session_number": i + 1,
            "cumulative_train_trials": len(train_choices),
            "test_trials": num_test_trials,
            "mmlpf_bits_per_trial": mml_bits_per_trial,
            "psytrack_bits_per_trial": psy_bits_per_trial
        })
        
    print(f"[{subject_id}] Complete.")
    return results

# ==========================================
# 4. Main Execution & DB Pull
# ==========================================
if __name__ == "__main__":
    start_time = time.time()
    print("Querying AIND Database for 10-Mouse Cohort...")
    cohort_query = "task LIKE '%Uncoupled%' AND foraging_eff > 0.65 AND finished_trials > 300"
    all_sessions = db.select_sessions(where=cohort_query)
    all_sessions = all_sessions.sort_values(by=['subject_id', 'session_date'])
    
    # Require at least 10 sessions
    session_counts = all_sessions['subject_id'].value_counts()
    valid_subjects = session_counts[session_counts >= 10].index.sort_values()[:10] # Cap at exactly 10 mice
    
    print(f"Identified {len(valid_subjects)} target mice. Fetching trial data...")
    
    mouse_jobs = []
    for subject_id in valid_subjects:
        # Cap at 10 sessions per mouse
        subject_sessions = all_sessions[all_sessions['subject_id'] == subject_id].head(10)
        trials_df = db.fetch_trials(subject_sessions, columns=["animal_response", "earned_reward"])
        valid_trials_df = trials_df[trials_df['animal_response'] != 2].copy()
        
        session_data_list = []
        for _, session_data in valid_trials_df.groupby(['session_date', 'session_id']):
            choices = session_data['animal_response'].astype(int).values
            rewards = session_data['earned_reward'].astype(int).values
            session_data_list.append((choices, rewards))
            
        mouse_jobs.append((subject_id, session_data_list))
    
    print("\nStarting parallel processing via Joblib...")
    # n_jobs=-1 automatically detects and utilizes all available CPU cores on your Mac
    all_results = Parallel(n_jobs=-1)(delayed(process_mouse)(sub, data) for sub, data in mouse_jobs)
    
    print("\nMerging results...")
    flat_results = [row for sublist in all_results for row in sublist]
    final_df = pd.DataFrame(flat_results)
    
    csv_filename = "local_cohort_comparable_validation.csv"
    final_df.to_csv(csv_filename, index=False)
    
    end_time = time.time()
    print(f"Success! Data saved to '{csv_filename}'. Total compute time: {(end_time - start_time) / 60:.2f} minutes.")