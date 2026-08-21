"""
Matched one-step-ahead validation of a Q-learning particle filter (MMLPF)
against a PsyTrack-style dynamic GLM.

Both models are scored under identical rules:
  - at trial t, predict P(right) using only trials 0..t-1
  - then observe trial t and update

This removes the adaptation asymmetry in the original script, where the
particle filter filtered on the test session while PsyTrack was frozen at
`wMode[:, -1]` from the last training trial.

Other corrections relative to the original:
  - PsyTrack regressors split into rewarded and unrewarded choice history
    (the original ternary `reward_history` made unrewarded-left and
    unrewarded-right indistinguishable)
  - `dayLength` carries real per-session lengths, so sigDay has boundaries
  - particle weights normalized without an additive epsilon (which made them
    sum to <1 and crashed np.random.choice), with an explicit degenerate case
  - resampling triggered on ESS < N/2 instead of every trial
  - alpha/beta random walks in unconstrained space (logit / log), so particles
    cannot pile up against a hard clip
  - particle prior centered on a static maximum-likelihood fit to the TRAINING
    sessions, so the opening trials of the alpha/beta traces are not just
    relaxation away from an arbitrary uniform prior
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

__all__ = [
    "build_glm_regressors", "fit_static_qlearning", "run_particle_filter",
    "run_glm_filter", "bits_per_trial", "simulate_sessions",
]


# ----------------------------------------------------------------------------
# regressors
# ----------------------------------------------------------------------------
def build_glm_regressors(choices, rewards, session_ids, n_lags=2):
    """Rewarded- and unrewarded-choice history at lags 1..n_lags.

    Returns (rew, unrew) each shape (N, n_lags), coded +1 for a right choice,
    -1 for a left choice, 0 where the lag crosses a session boundary.
    Session boundaries are respected: trial 0 of a session has no history.
    """
    choices = np.asarray(choices, int)
    rewards = np.asarray(rewards, int)
    session_ids = np.asarray(session_ids)
    n = len(choices)

    signed = np.where(choices == 1, 1.0, -1.0)
    rew = np.zeros((n, n_lags))
    unrew = np.zeros((n, n_lags))

    for lag in range(1, n_lags + 1):
        idx = np.arange(lag, n)
        same_session = session_ids[idx] == session_ids[idx - lag]
        prev_signed = signed[idx - lag]
        prev_rewarded = rewards[idx - lag] == 1
        rew[idx, lag - 1] = np.where(same_session & prev_rewarded, prev_signed, 0.0)
        unrew[idx, lag - 1] = np.where(same_session & ~prev_rewarded, prev_signed, 0.0)
    return rew, unrew


def _design_matrix(rew, unrew):
    """Column order must match psytrack.helper.read_input, which iterates
    sorted(weights_dict) -> 'bias', 'rew', 'unrew'."""
    return np.column_stack([np.ones(len(rew)), rew, unrew])


# ----------------------------------------------------------------------------
# static Q-learning fit (used to centre the particle prior)
# ----------------------------------------------------------------------------
def _qlearning_nll(params, choices, rewards, session_starts):
    alpha = expit(params[0])
    beta = np.exp(params[1])
    Q = np.array([0.5, 0.5])
    nll = 0.0
    for t, (c, r) in enumerate(zip(choices, rewards)):
        if session_starts[t]:
            Q = np.array([0.5, 0.5])
        z = beta * (Q[1] - Q[0])
        p_right = expit(z)
        p = p_right if c == 1 else 1.0 - p_right
        nll -= np.log(max(p, 1e-12))
        Q[c] += alpha * (r - Q[c])
    return nll


def fit_static_qlearning(choices, rewards, session_ids):
    """Maximum-likelihood alpha, beta for a constant-parameter Q-learner."""
    session_ids = np.asarray(session_ids)
    session_starts = np.empty(len(choices), bool)
    session_starts[0] = True
    session_starts[1:] = session_ids[1:] != session_ids[:-1]

    best = None
    for a0, b0 in [(0.2, 3.0), (0.5, 6.0), (0.05, 1.5)]:
        res = minimize(_qlearning_nll, [logit(a0), np.log(b0)],
                       args=(np.asarray(choices, int), np.asarray(rewards, float),
                             session_starts),
                       method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-4})
        if best is None or res.fun < best.fun:
            best = res
    return float(expit(best.x[0])), float(np.exp(best.x[1])), float(best.fun)


# ----------------------------------------------------------------------------
# particle filter
# ----------------------------------------------------------------------------
def run_particle_filter(choices, rewards, session_ids,
                        alpha_prior=(0.2, 0.6), beta_prior=(3.0, 0.6),
                        sigma_alpha=0.05, sigma_beta=0.05,
                        n_particles=2000, ess_frac=0.5, seed=0):
    """One-step-ahead Q-learning particle filter.

    alpha_prior : (mean_alpha, sd_in_logit_space)
    beta_prior  : (mean_beta,  sd_in_log_space)
    sigma_alpha / sigma_beta : random-walk sd per trial in unconstrained space.

    Q resets to 0.5 at each session boundary; alpha/beta particles carry across.
    Returns a dict of per-trial one-step-ahead quantities.
    """
    rng = np.random.default_rng(seed)
    choices = np.asarray(choices, int)
    rewards = np.asarray(rewards, float)
    session_ids = np.asarray(session_ids)
    T = len(choices)
    N = n_particles

    a_un = logit(alpha_prior[0]) + rng.normal(0, alpha_prior[1], N)
    b_un = np.log(beta_prior[0]) + rng.normal(0, beta_prior[1], N)
    Q = np.full((N, 2), 0.5)
    logw = np.full(N, -np.log(N))

    out = {k: np.zeros(T) for k in
           ("p_right", "alpha_mean", "alpha_sd", "beta_mean", "beta_sd",
            "q_left", "q_right", "ess")}
    out["resampled"] = np.zeros(T, bool)

    for t in range(T):
        if t > 0 and session_ids[t] != session_ids[t - 1]:
            Q[:] = 0.5

        a_un = a_un + rng.normal(0, sigma_alpha, N)
        b_un = b_un + rng.normal(0, sigma_beta, N)
        alpha = expit(a_un)
        beta = np.exp(b_un)

        w = np.exp(logw - logw.max())
        w /= w.sum()

        p_right_i = expit(beta * (Q[:, 1] - Q[:, 0]))

        out["p_right"][t] = w @ p_right_i
        out["alpha_mean"][t] = w @ alpha
        out["alpha_sd"][t] = np.sqrt(w @ (alpha - out["alpha_mean"][t]) ** 2)
        out["beta_mean"][t] = w @ beta
        out["beta_sd"][t] = np.sqrt(w @ (beta - out["beta_mean"][t]) ** 2)
        out["q_left"][t] = w @ Q[:, 0]
        out["q_right"][t] = w @ Q[:, 1]

        c = choices[t]
        lik = p_right_i if c == 1 else 1.0 - p_right_i
        logw = logw + np.log(np.clip(lik, 1e-300, None))

        w = np.exp(logw - logw.max())
        s = w.sum()
        if not np.isfinite(s) or s <= 0:
            w = np.full(N, 1.0 / N)          # total degeneracy: reset, don't fudge
        else:
            w = w / s
        out["ess"][t] = 1.0 / np.sum(w ** 2)

        if out["ess"][t] < ess_frac * N:
            idx = rng.choice(N, size=N, p=w)
            a_un, b_un, Q = a_un[idx], b_un[idx], Q[idx].copy()
            alpha = expit(a_un)
            logw = np.full(N, -np.log(N))
            out["resampled"][t] = True
        else:
            logw = np.log(w)

        Q[:, c] += alpha * (rewards[t] - Q[:, c])

    return out


# ----------------------------------------------------------------------------
# dynamic-GLM forward filter (PsyTrack model, run causally)
# ----------------------------------------------------------------------------
def run_glm_filter(X, y, sigma, sig_init=4.0, sig_day=None, day_starts=None):
    """Causal one-step-ahead filter for a Bernoulli GLM with random-walk weights.

    Assumed-density (Laplace) filtering: the same generative model PsyTrack's
    hyperOpt fits, run forward in time instead of smoothed over the whole
    session. At trial t the prediction uses only trials 0..t-1.

    X : (T, K) design matrix, y : (T,) in {0,1}, sigma : (K,) random-walk sd.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    T, K = X.shape
    sigma = np.atleast_1d(np.asarray(sigma, float))
    if sigma.size == 1:
        sigma = np.full(K, sigma[0])

    w = np.zeros(K)
    P = np.eye(K) * sig_init ** 2
    p_hat = np.zeros(T)
    w_trace = np.zeros((T, K))

    for t in range(T):
        step = sigma ** 2
        if sig_day is not None and day_starts is not None and day_starts[t]:
            step = step + sig_day ** 2
        P = P + np.diag(step)

        x = X[t]
        p = expit(w @ x)
        p_hat[t] = p
        w_trace[t] = w

        Px = P @ x
        s = max(p * (1 - p), 1e-9)
        denom = 1.0 + s * (x @ Px)
        P = P - np.outer(Px, Px) * (s / denom)
        w = w + P @ (x * (y[t] - p))

    return p_hat, w_trace


# ----------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------
def bits_per_trial(p_right, choices):
    """Mean cross-entropy in bits. 1.0 = coin flip; lower is better."""
    p = np.clip(np.asarray(p_right, float), 1e-9, 1 - 1e-9)
    y = np.asarray(choices, int)
    return float(-np.mean(y * np.log2(p) + (1 - y) * np.log2(1 - p)))


# ----------------------------------------------------------------------------
# simulator (for parameter-recovery validation)
# ----------------------------------------------------------------------------
def simulate_sessions(n_sessions, trial_range=(500, 800), alpha=0.4, beta=5.0,
                      block=60, p_high=0.8, p_low=0.1, seed=0):
    """Uncoupled two-armed task. alpha/beta may be scalars or callables of
    (t, T) returning the value at trial t of a session of length T."""
    rng = np.random.default_rng(seed)
    a_fn = alpha if callable(alpha) else (lambda t, T: alpha)
    b_fn = beta if callable(beta) else (lambda t, T: beta)

    ch, rw, sid, a_true, b_true = [], [], [], [], []
    for s in range(n_sessions):
        T = int(rng.integers(*trial_range))
        Q = np.array([0.5, 0.5])
        p = np.array([p_high, p_low])
        for t in range(T):
            if t > 0 and t % block == 0:
                p = p[::-1].copy()
            at, bt = a_fn(t, T), b_fn(t, T)
            c = int(rng.random() < expit(bt * (Q[1] - Q[0])))
            r = int(rng.random() < p[c])
            Q[c] += at * (r - Q[c])
            ch.append(c); rw.append(r); sid.append(s)
            a_true.append(at); b_true.append(bt)
    return (np.array(ch), np.array(rw), np.array(sid),
            np.array(a_true), np.array(b_true))
