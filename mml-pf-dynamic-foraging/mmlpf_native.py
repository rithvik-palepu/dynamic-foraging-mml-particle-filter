"""
The MMLPF exactly as defined in empirical_drifting_agent_mml_pf.py, wrapped so
it can be scored against PsyTrack on equal terms.

Nothing about the model is changed here. The particle dynamics, the log-space
alpha/beta parameterization, the clip bounds, the uniform(0,1) Q initialization,
the every-trial resampling and the differential_evolution M-step are copied
from that file. What this module adds is only:

  * `mml_fit_volatility`  -- run the M-step on a set of TRAINING sessions
                             instead of on the session being scored, so the
                             volatilities are not fit on test data
  * `mmlpf_predict`       -- run the forward pass and return the one-step-ahead
                             predictive P(right) per trial

Both additions exist to make the comparison fair, not to change the model.

Two things worth knowing about the original file:

1. `calculate_nll_fast` already accumulates the correct quantity for a
   predictive comparison: `nll -= log(mean(p_choice))` is evaluated BEFORE the
   Q update and before resampling, so it is a genuine one-step-ahead
   (prequential) predictive log-likelihood. It is directly comparable to a
   PsyTrack forward filter. That is the score used here.

2. `execute_particle_smoother` is a fixed-lag SMOOTHER: the state at trial t is
   drawn from lineages that survived to trial t+15, so it uses future choices.
   That is the right tool for reading off trajectories, but it must never be
   used to score prediction -- it sees the answer. Its alpha/beta priors
   (log(0.3), log(15.0)) also differ from the NLL step's (log(0.05), log(4.0)),
   so the volatilities are optimized under one prior and the trajectories
   extracted under another.
"""

import numpy as np
from scipy.optimize import differential_evolution

__all__ = ["mmlpf_nll", "mml_fit_volatility", "mmlpf_predict"]


# ---------------------------------------------------------------------------
# forward pass -- particle dynamics copied verbatim from the source file
# ---------------------------------------------------------------------------
def _forward(choices, rewards, sigma_alpha, sigma_beta, num_particles,
             seed=42, prior_log_alpha=np.log(0.05), prior_log_beta=np.log(4.0),
             collect=False):
    """One forward pass. Returns (nll, p_right_one_step_ahead or None).

    `session_starts` handling is by the caller: run one session at a time, as
    the original script does via groupby.
    """
    np.random.seed(seed)
    num_trials = len(choices)

    particles = np.zeros((num_particles, 4))
    particles[:, 0] = np.random.uniform(0, 1, num_particles)
    particles[:, 1] = np.random.uniform(0, 1, num_particles)
    particles[:, 2] = np.random.normal(prior_log_alpha, 0.5, num_particles)
    particles[:, 3] = np.random.normal(prior_log_beta, 0.5, num_particles)

    nll = 0.0
    p_right = np.zeros(num_trials) if collect else None
    alpha_tr = np.zeros(num_trials) if collect else None
    beta_tr = np.zeros(num_trials) if collect else None

    for t in range(num_trials):
        particles[:, 2] += np.random.normal(0, sigma_alpha, num_particles)
        particles[:, 3] += np.random.normal(0, sigma_beta, num_particles)

        alpha_vals = np.clip(np.exp(particles[:, 2]), 1e-4, 0.99)
        beta_vals = np.clip(np.exp(particles[:, 3]), 0.01, 100.0)
        Q_vals = particles[:, :2]

        max_Q = np.max(Q_vals, axis=1, keepdims=True)
        exp_Q = np.exp(beta_vals[:, None] * (Q_vals - max_Q))
        probs = exp_Q / np.sum(exp_Q, axis=1, keepdims=True)

        # particles are equally weighted here (resampled every trial), so the
        # plain mean is the correct predictive probability
        if collect:
            p_right[t] = np.mean(probs[:, 1])
            alpha_tr[t] = np.mean(alpha_vals)
            beta_tr[t] = np.mean(beta_vals)

        p_choice = probs[np.arange(num_particles), choices[t]]
        nll -= np.log(np.mean(p_choice) + 1e-16)

        weights = p_choice / (np.sum(p_choice) + 1e-16)
        if np.sum(weights) < 1e-20:
            weights = np.ones(num_particles) / num_particles
        weights /= np.sum(weights)

        idx = np.random.choice(num_particles, size=num_particles, p=weights)
        particles = particles[idx].copy()

        alpha_vals_resampled = np.clip(np.exp(particles[:, 2]), 1e-4, 0.99)
        particles[:, choices[t]] += alpha_vals_resampled * (
            rewards[t] - particles[:, choices[t]])

    if collect:
        return nll, dict(p_right=p_right, alpha_mean=alpha_tr, beta_mean=beta_tr)
    return nll, None


def mmlpf_nll(hyperparams, sessions, num_particles=500, seed=42):
    """Total one-step-ahead NLL over a list of (choices, rewards) sessions.

    Per-session passes mean Q resets at each session boundary, matching the
    original script's per-session groupby.
    """
    sigma_alpha, sigma_beta = hyperparams
    total = 0.0
    for c, r in sessions:
        nll, _ = _forward(c, r, sigma_alpha, sigma_beta, num_particles, seed=seed)
        total += nll
    return total


def mml_fit_volatility(sessions, num_particles=500, maxiter=30, popsize=15,
                       tol=0.01, bounds=((0.001, 0.2), (0.001, 0.2)),
                       workers=-1, seed=42, disp=False):
    """M-step: differential_evolution over (sigma_alpha, sigma_beta).

    Same optimizer and bounds as the source file. Pass TRAINING sessions here;
    the returned volatilities are then applied to the held-out session, which
    is what makes the score out-of-sample.
    """
    res = differential_evolution(
        func=mmlpf_nll, bounds=list(bounds),
        args=(sessions, num_particles, seed),
        maxiter=maxiter, popsize=popsize, tol=tol,
        workers=workers, disp=disp, seed=seed)
    return float(res.x[0]), float(res.x[1]), res


def mmlpf_predict(choices, rewards, sigma_alpha, sigma_beta,
                  num_particles=1500, seed=0):
    """Forward pass on one session with fixed volatilities.

    Returns dict with `p_right` (one-step-ahead), `alpha_mean`, `beta_mean`,
    and `nll`. Uses the filter, not the smoother: the smoother would use future
    choices and cannot be scored as prediction.
    """
    nll, out = _forward(choices, rewards, sigma_alpha, sigma_beta,
                        num_particles, seed=seed, collect=True)
    out["nll"] = nll
    out["bits_per_trial"] = nll / len(choices) / np.log(2)
    return out
