"""
Chronological walk-forward comparison of MMLPF vs PsyTrack by out-of-sample NLL.

Cohort: 10 mice x 10 sessions. Parallelized across mice.

Design (extends cross_validation.py's walk-forward loop to two models):

    for i in 1..9:
        train on sessions 0..i-1
        fit BOTH models' hyperparameters on those sessions only
        predict session i one trial at a time, accumulating NLL
        neither model ever sees session i during fitting

Both models are scored by the same quantity: the one-step-ahead (prequential)
predictive negative log-likelihood on the held-out session. At trial t each
model emits P(choice_t) using only trials 0..t-1 of the test session plus its
training fit, the log of that probability is accumulated, and only then does the
model observe trial t and update. This is what makes the two numbers comparable
-- a model scored any other way (smoothed, or refit on the test session) is not
in the same comparison.

    MMLPF   : calculate_nll_window() is copied verbatim from cross_validation.py.
              It already accumulates -log(mean(p_choice)) BEFORE the Q update and
              BEFORE resampling, so it is already a proper prequential NLL.
              Hyperparameters (sigma_alpha, sigma_beta) come from
              differential_evolution over the training sessions.

    PsyTrack: hyperOpt fits the random-walk variances on the training sessions,
              then the weights are filtered FORWARD through the test session
              (Laplace / assumed-density filter). PsyTrack's own hyperOpt is a
              smoother -- it sees the whole session at once -- so it cannot be
              used directly on test data without leaking the answer.

Outputs
-------
walk_forward_cv.csv     one row per (subject, test session) with both NLLs
walk_forward_cv.png     per-session curves and the paired difference

Usage
-----
    python mmlpf_vs_psytrack_cv.py            # 10 mice, 10 sessions, 8 workers
    python mmlpf_vs_psytrack_cv.py --workers 4
    python mmlpf_vs_psytrack_cv.py --summarize   # replot from an existing CSV
"""

import argparse
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.special import expit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import psytrack

# ---------------------------------------------------------------------------
# cohort definition
# ---------------------------------------------------------------------------
# NOTE: foraging_eff and finished_trials are per-SESSION columns, so this
# selects sessions, not animals. A mouse enters the cohort when it has enough
# qualifying sessions; its other sessions exist in the database and are simply
# not used. Describe it that way in methods.
#
# Threshold history: this was 0.8 for the walk-forward run and the first
# block-CV runs (713379 / 751766). Lowered to 0.65 to match
# single_subject_comparison.py and to widen the cohort -- high-efficiency
# sessions are where a value-learning model is most favoured, so restricting to
# them biases the comparison toward the MMLPF. Results produced before this
# change are NOT comparable to results produced after it.
COHORT_QUERY = ("task LIKE '%Uncoupled%' AND foraging_eff > 0.65 "
                "AND finished_trials > 300")
N_SUBJECTS = 10
N_SESSIONS = 10
MIN_TEST_TRIALS = 100

# MMLPF settings (from cross_validation.py)
MML_PARTICLES = 500
DE_BOUNDS = [(0.001, 0.2), (0.001, 0.2)]
DE_MAXITER = 30
DE_POPSIZE = 10

# PsyTrack settings
N_LAGS = 2                      # rewarded/unrewarded choice history lags
WITH_PERSEV = False             # also fit MMLPF + perseveration (--persev)
GLM_P0 = 0.25                   # forward-filter initial weight covariance


# ===========================================================================
# 1. MMLPF -- verbatim from cross_validation.py
# ===========================================================================
def calculate_nll_window(sigma_alpha, sigma_beta, choices, rewards,
                         num_particles=MML_PARTICLES):
    """Runs the particle filter over a complete session and returns total NLL."""
    np.random.seed(42)
    num_trials = len(choices)

    particles = np.zeros((num_particles, 4))
    particles[:, 0] = np.random.uniform(0, 1, num_particles)
    particles[:, 1] = np.random.uniform(0, 1, num_particles)

    # Matched empirical priors for highly trained cohort
    particles[:, 2] = np.random.normal(np.log(0.3), 0.5, num_particles)
    particles[:, 3] = np.random.normal(np.log(15.0), 0.5, num_particles)

    session_nll = 0.0

    for t in range(num_trials):
        particles[:, 2] += np.random.normal(0, sigma_alpha, num_particles)
        particles[:, 3] += np.random.normal(0, sigma_beta, num_particles)

        alpha_vals = np.clip(np.exp(particles[:, 2]), 1e-4, 0.99)
        beta_vals = np.clip(np.exp(particles[:, 3]), 0.01, 100.0)
        Q_vals = particles[:, :2]

        max_Q = np.max(Q_vals, axis=1, keepdims=True)
        exp_Q = np.exp(beta_vals[:, None] * (Q_vals - max_Q))
        probs = exp_Q / np.sum(exp_Q, axis=1, keepdims=True)

        p_choice = probs[np.arange(num_particles), choices[t]]

        session_nll -= np.log(np.mean(p_choice) + 1e-16)

        weights = p_choice / (np.sum(p_choice) + 1e-16)
        if np.sum(weights) < 1e-20:
            weights = np.ones(num_particles) / num_particles
        weights /= np.sum(weights)

        idx = np.random.choice(num_particles, size=num_particles, p=weights)
        particles = particles[idx].copy()

        alpha_vals_resampled = np.clip(np.exp(particles[:, 2]), 1e-4, 0.99)
        particles[:, choices[t]] += alpha_vals_resampled * (
            rewards[t] - particles[:, choices[t]])

    return session_nll


def calculate_nll_window_traces(sigma_alpha, sigma_beta, choices, rewards,
                                num_particles=MML_PARTICLES):
    """Same forward pass, additionally returning per-trial latents.

    Identical dynamics to calculate_nll_window -- this exists only so the
    per-trial predictive probabilities can be saved for the paired test.
    """
    np.random.seed(42)
    T = len(choices)
    particles = np.zeros((num_particles, 4))
    particles[:, 0] = np.random.uniform(0, 1, num_particles)
    particles[:, 1] = np.random.uniform(0, 1, num_particles)
    particles[:, 2] = np.random.normal(np.log(0.3), 0.5, num_particles)
    particles[:, 3] = np.random.normal(np.log(15.0), 0.5, num_particles)

    nll = 0.0
    out = {k: np.zeros(T) for k in ("p_right", "alpha", "beta", "rpe",
                                    "q_left", "q_right")}

    for t in range(T):
        particles[:, 2] += np.random.normal(0, sigma_alpha, num_particles)
        particles[:, 3] += np.random.normal(0, sigma_beta, num_particles)

        alpha_vals = np.clip(np.exp(particles[:, 2]), 1e-4, 0.99)
        beta_vals = np.clip(np.exp(particles[:, 3]), 0.01, 100.0)
        Q_vals = particles[:, :2]

        max_Q = np.max(Q_vals, axis=1, keepdims=True)
        exp_Q = np.exp(beta_vals[:, None] * (Q_vals - max_Q))
        probs = exp_Q / np.sum(exp_Q, axis=1, keepdims=True)

        out["p_right"][t] = np.mean(probs[:, 1])
        out["alpha"][t] = np.mean(alpha_vals)
        out["beta"][t] = np.mean(beta_vals)
        out["q_left"][t] = np.mean(Q_vals[:, 0])
        out["q_right"][t] = np.mean(Q_vals[:, 1])
        out["rpe"][t] = rewards[t] - np.mean(Q_vals[:, choices[t]])

        p_choice = probs[np.arange(num_particles), choices[t]]
        nll -= np.log(np.mean(p_choice) + 1e-16)

        weights = p_choice / (np.sum(p_choice) + 1e-16)
        if np.sum(weights) < 1e-20:
            weights = np.ones(num_particles) / num_particles
        weights /= np.sum(weights)

        idx = np.random.choice(num_particles, size=num_particles, p=weights)
        particles = particles[idx].copy()
        ar = np.clip(np.exp(particles[:, 2]), 1e-4, 0.99)
        particles[:, choices[t]] += ar * (rewards[t] - particles[:, choices[t]])

    out["nll"] = nll
    return out


def calculate_nll_window_persev(sigma_alpha, sigma_beta, sigma_phi,
                                choices, rewards, num_particles=MML_PARTICLES,
                                collect=False):
    """MMLPF plus a drifting perseveration weight phi:

        P(right) = sigmoid(beta*(Q_R - Q_L) + phi*prev)

    prev is +1 after a right choice, -1 after a left, 0 before the first trial.
    Particle state is 5-D: [Q_L, Q_R, log_alpha, log_beta, phi].

    Why this term exists: PsyTrack's regressors are signed choice history, so it
    models choice autocorrelation directly. The plain Q-learner has no such term
    -- its only route to a sticky choice sequence is a large beta, which then
    costs ~10 bits every time the animal departs from the greedy option. That is
    both why beta saturates and why the plain model loses. phi gives the
    autocorrelation somewhere to live that is not beta.
    """
    np.random.seed(42)
    T = len(choices)
    P = np.zeros((num_particles, 5))
    P[:, 0] = np.random.uniform(0, 1, num_particles)
    P[:, 1] = np.random.uniform(0, 1, num_particles)
    P[:, 2] = np.random.normal(np.log(0.3), 0.5, num_particles)
    P[:, 3] = np.random.normal(np.log(15.0), 0.5, num_particles)
    P[:, 4] = np.random.normal(0.0, 0.5, num_particles)

    nll = 0.0
    prev = 0.0
    if collect:
        out = {k: np.zeros(T) for k in
               ("p_right", "alpha", "beta", "phi", "rpe", "q_left", "q_right",
                # particle spread at trial t, taken at the same point as the
                # means: BEFORE the weight update, so it is the one-step-ahead
                # predictive spread rather than a post-hoc filtered one
                "p_right_sd", "alpha_sd", "beta_sd", "phi_sd")}

    for t in range(T):
        P[:, 2] += np.random.normal(0, sigma_alpha, num_particles)
        P[:, 3] += np.random.normal(0, sigma_beta, num_particles)
        P[:, 4] += np.random.normal(0, sigma_phi, num_particles)

        alpha_vals = np.clip(np.exp(P[:, 2]), 1e-4, 0.99)
        beta_vals = np.clip(np.exp(P[:, 3]), 0.01, 100.0)
        phi_vals = P[:, 4]
        Q = P[:, :2]

        z = beta_vals * (Q[:, 1] - Q[:, 0]) + phi_vals * prev
        pr_right = expit(np.clip(z, -50, 50))
        probs = np.column_stack([1.0 - pr_right, pr_right])

        if collect:
            out["p_right"][t] = np.mean(pr_right)
            out["alpha"][t] = np.mean(alpha_vals)
            out["beta"][t] = np.mean(beta_vals)
            out["phi"][t] = np.mean(phi_vals)
            out["q_left"][t] = np.mean(Q[:, 0])
            out["q_right"][t] = np.mean(Q[:, 1])
            out["rpe"][t] = rewards[t] - np.mean(Q[:, choices[t]])
            out["p_right_sd"][t] = np.std(pr_right)
            out["alpha_sd"][t] = np.std(alpha_vals)
            out["beta_sd"][t] = np.std(beta_vals)
            out["phi_sd"][t] = np.std(phi_vals)

        p_choice = probs[np.arange(num_particles), choices[t]]
        nll -= np.log(np.mean(p_choice) + 1e-16)

        weights = p_choice / (np.sum(p_choice) + 1e-16)
        if np.sum(weights) < 1e-20:
            weights = np.ones(num_particles) / num_particles
        weights /= np.sum(weights)

        idx = np.random.choice(num_particles, size=num_particles, p=weights)
        P = P[idx].copy()
        ar = np.clip(np.exp(P[:, 2]), 1e-4, 0.99)
        P[:, choices[t]] += ar * (rewards[t] - P[:, choices[t]])
        prev = 1.0 if choices[t] == 1 else -1.0

    if collect:
        out["nll"] = nll
        return out
    return nll


def objective_function_persev(hyperparams, train_sessions):
    sa, sb, sp = hyperparams
    return sum(calculate_nll_window_persev(sa, sb, sp, c, r)
               for c, r in train_sessions)


def fit_mmlpf_persev(train_sessions, de_workers=1):
    """M-step over (sigma_alpha, sigma_beta, sigma_phi).

    Bounds widened to 0.4: on real data the source file's 0.2 ceiling was
    binding on ~15% of fits, and a value sitting on its bound is not an estimate.
    """
    res = differential_evolution(
        func=objective_function_persev,
        bounds=[(0.001, 0.4), (0.001, 0.4), (0.001, 0.4)],
        args=(train_sessions,), maxiter=DE_MAXITER, popsize=DE_POPSIZE,
        workers=de_workers, disp=False, seed=42)
    return float(res.x[0]), float(res.x[1]), float(res.x[2])


def objective_function(hyperparams, train_sessions):
    """Joint NLL across all historical training sessions."""
    sigma_alpha, sigma_beta = hyperparams
    return sum(calculate_nll_window(sigma_alpha, sigma_beta, c, r)
               for c, r in train_sessions)


def fit_mmlpf(train_sessions, de_workers=1):
    """M-step. de_workers MUST stay 1 when the outer loop is parallel."""
    res = differential_evolution(
        func=objective_function, bounds=DE_BOUNDS, args=(train_sessions,),
        maxiter=DE_MAXITER, popsize=DE_POPSIZE, workers=de_workers, disp=False,
        seed=42)
    return float(res.x[0]), float(res.x[1])


# ===========================================================================
# 2. PsyTrack -- fit on train, filter forward through test
# ===========================================================================
def build_regressors(sessions):
    """Rewarded / unrewarded choice history at lags 1..N_LAGS.

    sessions: list of (choices, rewards). Lags never cross a session boundary.
    Returns X (n_total, 1 + 2*N_LAGS), y, and per-session lengths.
    Column order matches psytrack.read_input, which iterates sorted(weights):
    'bias', 'rew', 'unrew'.
    """
    rews, unrews, ys, lens = [], [], [], []
    for c, r in sessions:
        c = np.asarray(c, int)
        r = np.asarray(r, int)
        n = len(c)
        signed = np.where(c == 1, 1.0, -1.0)
        rew = np.zeros((n, N_LAGS))
        unrew = np.zeros((n, N_LAGS))
        for lag in range(1, N_LAGS + 1):
            idx = np.arange(lag, n)
            prev_signed = signed[idx - lag]
            prev_rewarded = r[idx - lag] == 1
            rew[idx, lag - 1] = np.where(prev_rewarded, prev_signed, 0.0)
            unrew[idx, lag - 1] = np.where(~prev_rewarded, prev_signed, 0.0)
        rews.append(rew)
        unrews.append(unrew)
        ys.append(c)
        lens.append(n)
    rew = np.vstack(rews)
    unrew = np.vstack(unrews)
    y = np.concatenate(ys)
    X = np.column_stack([np.ones(len(y)), rew, unrew])
    return X, y, rew, unrew, np.array(lens)


def fit_psytrack(train_sessions):
    """hyperOpt on the training sessions. Returns (sigma, final weights)."""
    X, y, rew, unrew, lens = build_regressors(train_sessions)
    K = 1 + 2 * N_LAGS
    dat = {"y": y + 1,
           "inputs": {"rew": rew, "unrew": unrew},
           "dayLength": lens}
    weights = {"bias": 1, "rew": N_LAGS, "unrew": N_LAGS}
    hyp, _evd, wMode, _hess = psytrack.hyperOpt(
        dat, {"sigma": [2 ** -5] * K, "sigInit": 2 ** 5,
              "sigDay": [2 ** -5] * K},
        weights, ["sigma", "sigDay"], showOpt=0)
    return hyp["sigma"], wMode[:, -1]


def psytrack_forward_nll(test_session, sigma, w0):
    """One-step-ahead NLL for PsyTrack's model on the held-out session.

    Laplace (assumed-density) filter: the same generative model hyperOpt fits --
    Bernoulli GLM with random-walk weights -- propagated causally. At trial t
    the prediction uses only trials 0..t-1.
    """
    c, r = test_session
    X, y, _rew, _unrew, _lens = build_regressors([(c, r)])
    T, K = X.shape
    w = np.asarray(w0, float).copy()
    P = np.eye(K) * GLM_P0
    s2 = np.atleast_1d(np.asarray(sigma, float)) ** 2
    if s2.size == 1:
        s2 = np.full(K, s2[0])

    nll = 0.0
    p_hat = np.zeros(T)
    for t in range(T):
        P = P + np.diag(s2)
        x = X[t]
        p = expit(w @ x)
        p_hat[t] = p
        pc = p if y[t] == 1 else 1.0 - p
        nll -= np.log(max(pc, 1e-16))

        Px = P @ x
        s = max(p * (1 - p), 1e-9)
        P = P - np.outer(Px, Px) * (s / (1.0 + s * (x @ Px)))
        w = w + P @ (x * (y[t] - p))
    return nll, p_hat


# ===========================================================================
# 3. paired significance test
# ===========================================================================
def block_bootstrap_diff(nll_a_pertrial, nll_b_pertrial, block=50, n_boot=2000,
                         seed=0):
    """95% interval on the mean per-trial NLL difference (a - b).

    Moving-block bootstrap: foraging choices are autocorrelated, so a
    trial-wise bootstrap treats correlated trials as independent and reports an
    interval that is far too narrow.
    """
    d = np.asarray(nll_a_pertrial) - np.asarray(nll_b_pertrial)
    nb = len(d) // block
    if nb < 4:
        return float(d.mean()), np.nan, np.nan
    means = d[:nb * block].reshape(nb, block).mean(1)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(means, nb).mean() for _ in range(n_boot)])
    return float(d.mean()), float(np.percentile(boot, 2.5)), \
        float(np.percentile(boot, 97.5))


# ===========================================================================
# 4. walk-forward for one subject
# ===========================================================================
RESULT_COLUMNS = [
    "subject_id", "test_session_number", "cumulative_train_trials",
    "test_trials", "opt_sigma_alpha", "opt_sigma_beta",
    "mmlpf_nll", "psytrack_nll", "base_nll",
    "mmlpf_nll_per_trial", "psytrack_nll_per_trial", "base_nll_per_trial",
    "mmlpf_bits", "psytrack_bits", "base_bits",
    "diff_bits", "ci_lo", "ci_hi", "separates",
    "mean_alpha", "mean_beta", "max_beta", "frac_beta_at_clip",
    "opt_sigma_phi", "mmlpf_persev_bits", "mean_phi", "max_beta_persev",
    "diff_persev_bits", "choice_autocorr",
    "seconds", "error",
]


def execute_chronological_walk_forward(session_data_list, subject_id="?",
                                       de_workers=1, verbose=True):
    """Train on sessions 0..i-1, predict session i, for i = 1..len-1."""
    rows = []
    train_sessions = []
    cumulative_train_trials = 0

    for i in range(1, len(session_data_list)):
        t0 = time.time()
        train_sessions.append(session_data_list[i - 1])
        cumulative_train_trials += len(session_data_list[i - 1][0])
        test_choices, test_rewards = session_data_list[i]
        n_test = len(test_choices)

        row = {c: np.nan for c in RESULT_COLUMNS}
        row.update(subject_id=subject_id, test_session_number=i + 1,
                   cumulative_train_trials=cumulative_train_trials,
                   test_trials=n_test, error="")

        if n_test < MIN_TEST_TRIALS:
            row["error"] = f"test session has {n_test} trials"
            rows.append(row)
            continue

        try:
            # --- MMLPF: fit volatilities on train, score test ---
            sa, sb = fit_mmlpf(train_sessions, de_workers=de_workers)
            mm = calculate_nll_window_traces(sa, sb, test_choices, test_rewards)
            p_mm = np.where(test_choices == 1, mm["p_right"], 1 - mm["p_right"])
            mm_pertrial = -np.log(np.clip(p_mm, 1e-16, None))

            # --- PsyTrack: fit on train, filter forward through test ---
            sigma, w_last = fit_psytrack(train_sessions)
            ps_nll, ps_p = psytrack_forward_nll(
                (test_choices, test_rewards), sigma, w_last)
            p_ps = np.where(test_choices == 1, ps_p, 1 - ps_p)
            ps_pertrial = -np.log(np.clip(p_ps, 1e-16, None))

            # --- baseline: training base rate ---
            base_p = np.mean(np.concatenate([c for c, _ in train_sessions]))
            base_p = np.clip(base_p, 1e-6, 1 - 1e-6)
            base_pertrial = -np.log(
                np.where(test_choices == 1, base_p, 1 - base_p))

            # --- optional: MMLPF + perseveration ---
            pv_bits = pv_phi = pv_bmax = pv_sp = np.nan
            pv_diff = np.nan
            if WITH_PERSEV:
                sa_p, sb_p, sp_p = fit_mmlpf_persev(train_sessions,
                                                    de_workers=de_workers)
                pv = calculate_nll_window_persev(
                    sa_p, sb_p, sp_p, test_choices, test_rewards, collect=True)
                p_pv = np.where(test_choices == 1, pv["p_right"], 1 - pv["p_right"])
                pv_pertrial = -np.log(np.clip(p_pv, 1e-16, None))
                pv_bits = float(pv_pertrial.mean() / np.log(2))
                pv_phi = float(pv["phi"].mean())
                pv_bmax = float(pv["beta"].max())
                pv_sp = sp_p
                pv_diff = float((ps_pertrial - pv_pertrial).mean() / np.log(2))

            # lag-1 signed-choice autocorrelation: how much of this session is
            # pure choice stickiness, which PsyTrack models and a plain
            # Q-learner does not
            sgn = np.where(test_choices == 1, 1.0, -1.0)
            autoc = float(np.corrcoef(sgn[:-1], sgn[1:])[0, 1]) if len(sgn) > 2 \
                else np.nan

            diff, lo, hi = block_bootstrap_diff(ps_pertrial, mm_pertrial)
            ln2 = np.log(2)

            row.update(
                opt_sigma_alpha=sa, opt_sigma_beta=sb,
                mmlpf_nll=float(mm_pertrial.sum()),
                psytrack_nll=float(ps_nll),
                base_nll=float(base_pertrial.sum()),
                mmlpf_nll_per_trial=float(mm_pertrial.mean()),
                psytrack_nll_per_trial=float(ps_pertrial.mean()),
                base_nll_per_trial=float(base_pertrial.mean()),
                mmlpf_bits=float(mm_pertrial.mean() / ln2),
                psytrack_bits=float(ps_pertrial.mean() / ln2),
                base_bits=float(base_pertrial.mean() / ln2),
                diff_bits=diff / ln2,
                ci_lo=lo / ln2, ci_hi=hi / ln2,
                separates=bool(lo > 0 or hi < 0) if np.isfinite(lo) else False,
                mean_alpha=float(mm["alpha"].mean()),
                mean_beta=float(mm["beta"].mean()),
                max_beta=float(mm["beta"].max()),
                frac_beta_at_clip=float(np.mean(mm["beta"] > 95.0)),
                opt_sigma_phi=pv_sp, mmlpf_persev_bits=pv_bits,
                mean_phi=pv_phi, max_beta_persev=pv_bmax,
                diff_persev_bits=pv_diff, choice_autocorr=autoc,
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc(file=sys.stderr)

        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        if verbose and not row["error"]:
            print(f"  [{subject_id}] train {len(train_sessions)} sess "
                  f"({cumulative_train_trials} tr) -> session {i+1} "
                  f"({n_test} tr): MMLPF {row['mmlpf_bits']:.3f}  "
                  f"PsyTrack {row['psytrack_bits']:.3f}  "
                  f"diff {row['diff_bits']:+.3f} bits/trial "
                  f"({row['seconds']}s)", flush=True)
    return rows


# ===========================================================================
# 5. data loading
# ===========================================================================
def load_cohort():
    """Return {subject_id: [(choices, rewards), ...]} for N_SUBJECTS mice."""
    import aind_dynamic_foraging_database as db

    sessions = db.select_sessions(where=COHORT_QUERY).sort_values(
        by=["subject_id", "session_date"])
    counts = sessions["subject_id"].value_counts()
    eligible = counts[counts >= N_SESSIONS].index.sort_values()
    if len(eligible) < N_SUBJECTS:
        print(f"WARNING: only {len(eligible)} subjects have >= {N_SESSIONS} "
              f"sessions; requested {N_SUBJECTS}")
    subjects = list(eligible[:N_SUBJECTS])

    picked = (sessions[sessions["subject_id"].isin(subjects)]
              .groupby("subject_id", group_keys=False).head(N_SESSIONS))
    trials = db.fetch_trials(picked, columns=["animal_response", "earned_reward"])
    valid = trials[trials["animal_response"] != 2].copy()

    out = {}
    for sid, sdata in valid.groupby("subject_id"):
        seqs = []
        for _key, g in sdata.groupby(["session_date", "session_id"], sort=True):
            seqs.append((g["animal_response"].astype(int).values,
                         g["earned_reward"].astype(int).values))
        out[sid] = seqs[:N_SESSIONS]
    return out


# ===========================================================================
# 6. parallel driver
# ===========================================================================
def _worker(args):
    """Module-level for picklability. One subject per worker."""
    sid, sessions, with_persev = args
    global WITH_PERSEV
    WITH_PERSEV = with_persev
    # Each worker is single-threaded. differential_evolution(workers=-1) inside
    # a pool worker would fork a nested pool and oversubscribe every core, which
    # runs SLOWER than serial -- de_workers stays 1 here.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    return execute_chronological_walk_forward(sessions, subject_id=sid,
                                              de_workers=1, verbose=True)


def run_all(out_csv="walk_forward_cv.csv", workers=8):
    cohort = load_cohort()
    if not cohort:
        raise RuntimeError(
            "no subjects returned by load_cohort(): check COHORT_QUERY and "
            f"that at least {N_SUBJECTS} subjects have >= {N_SESSIONS} sessions")
    print(f"{len(cohort)} subjects x "
          f"{min(len(v) for v in cohort.values())}-"
          f"{max(len(v) for v in cohort.values())} sessions, {workers} worker(s)")

    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        missing = [c for c in RESULT_COLUMNS if c not in prev.columns]
        if missing:
            # a CSV written by an earlier version of this script: resuming into
            # it would silently mix rows from two different model sets
            stamp = time.strftime("%Y%m%d_%H%M%S")
            backup = out_csv.replace(".csv", f"_old_{stamp}.csv")
            os.rename(out_csv, backup)
            print(f"{out_csv} was written by an older version "
                  f"(missing {len(missing)} columns, e.g. {missing[0]}).")
            print(f"moved it to {backup} and starting a fresh run.")
        else:
            done = set(prev["subject_id"].astype(str))
            print(f"resuming: {len(done)} subjects already in {out_csv}")
    todo = [(sid, s, WITH_PERSEV) for sid, s in cohort.items()
            if str(sid) not in done]
    if not todo:
        return pd.read_csv(out_csv)

    t0 = time.time()
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as pool:
            for rows in pool.imap_unordered(_worker, todo):
                pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(
                    out_csv, mode="a", header=not os.path.exists(out_csv),
                    index=False)
    else:
        for job in todo:
            rows = _worker(job)
            pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(
                out_csv, mode="a", header=not os.path.exists(out_csv),
                index=False)
    print(f"\nfinished in {(time.time() - t0) / 60:.1f} min")
    return pd.read_csv(out_csv)


# ===========================================================================
# 7. summary and figure
# ===========================================================================
def summarize(df):
    ok = df[(df["error"].isna()) | (df["error"] == "")].copy()
    n_sub = ok["subject_id"].nunique()
    print(f"\n{'='*68}\nOUT-OF-SAMPLE NLL  ({len(ok)} held-out sessions, "
          f"{n_sub} mice)\n{'='*68}")
    print(f"{'':22s} {'nats/trial':>11s} {'bits/trial':>11s}")
    for col, lab in [("base_nll_per_trial", "base rate"),
                     ("psytrack_nll_per_trial", "PsyTrack"),
                     ("mmlpf_nll_per_trial", "MMLPF")]:
        bits = col.replace("_nll_per_trial", "_bits").replace("base_bits",
                                                              "base_bits")
        print(f"  {lab:20s} {ok[col].mean():11.4f} {ok[bits].mean():11.4f}")

    if "mmlpf_persev_bits" in ok and ok["mmlpf_persev_bits"].notna().any():
        pv = ok["mmlpf_persev_bits"].dropna()
        print(f"  {'MMLPF + persev':20s} {pv.mean()*np.log(2):11.4f} {pv.mean():11.4f}")

    d = ok["diff_bits"]                      # PsyTrack - MMLPF; >0 = MMLPF better
    wins = int((d > 0).sum())
    print(f"\nMMLPF better on {wins}/{len(ok)} held-out sessions")
    print(f"mean difference {d.mean():+.4f} bits/trial "
          f"(PsyTrack minus MMLPF; positive favours MMLPF)")
    print(f"median {d.median():+.4f}   range [{d.min():+.4f}, {d.max():+.4f}]")

    # subject-level test: average within subject first, so the unit of
    # replication is the mouse and not the session
    per_subj = ok.groupby("subject_id")["diff_bits"].mean()
    if len(per_subj) >= 3:
        from scipy import stats
        t, p = stats.ttest_1samp(per_subj, 0.0)
        try:
            _w, wp = stats.wilcoxon(per_subj)
        except ValueError:
            wp = np.nan
        print(f"\nper-mouse mean gap: {per_subj.mean():+.4f} bits/trial "
              f"(n = {len(per_subj)} mice)")
        print(f"  paired t-test  t = {t:.2f}, p = {p:.4g}")
        print(f"  Wilcoxon       p = {wp:.4g}")

    if "diff_persev_bits" in ok and ok["diff_persev_bits"].notna().any():
        dp = ok["diff_persev_bits"].dropna()
        print(f"\nwith perseveration: MMLPF better on {int((dp>0).sum())}/{len(dp)} "
              f"sessions, mean gap {dp.mean():+.4f} bits/trial")
        print(f"  mean phi {ok['mean_phi'].mean():+.2f}; "
              f"beta max {ok['max_beta'].max():.1f} -> "
              f"{ok['max_beta_persev'].max():.1f} with the phi term")
    if "choice_autocorr" in ok and ok["choice_autocorr"].notna().any():
        from scipy import stats as _st
        rho, pv_ = _st.spearmanr(ok["choice_autocorr"].abs(), ok["diff_bits"])
        print(f"\nlag-1 choice autocorrelation: mean "
              f"{ok['choice_autocorr'].mean():+.2f}; "
              f"correlation with the MMLPF gap rho = {rho:+.2f} (p = {pv_:.3g})")
        print("  diff_bits is PsyTrack minus MMLPF, so NEGATIVE rho means the "
              "MMLPF\n  falls further behind on the stickiest sessions -- the "
              "signature of a\n  missing choice-history term")

    print(f"\nfitted volatilities: sigma_alpha {ok['opt_sigma_alpha'].mean():.4f}"
          f"  sigma_beta {ok['opt_sigma_beta'].mean():.4f}")
    for col in ("opt_sigma_alpha", "opt_sigma_beta"):
        edge = int(((ok[col] < 0.0015) | (ok[col] > 0.1985)).sum())
        if edge:
            print(f"  WARNING: {edge}/{len(ok)} fits have {col} on a "
                  f"differential_evolution bound -- widen DE_BOUNDS")
    print(f"beta: mean {ok['mean_beta'].mean():.1f}, max {ok['max_beta'].max():.1f}"
          f" (clip 100)")
    clip = ok["frac_beta_at_clip"].mean()
    if clip > 0.01:
        print(f"  WARNING: beta sits above 95 on {clip:.1%} of trials on "
              f"average -- it is saturating, so beta trajectories are not "
              f"interpretable and the MMLPF's NLL is a lower bound")
    return ok


def plot_results(df, fname="walk_forward_cv.png"):
    """Three panels. If the CSV contains the perseveration model, it is drawn
    alongside the plain MMLPF -- omitting it understates the value model."""
    ok = df[(df["error"].isna()) | (df["error"] == "")].copy()
    C_MM, C_PV, C_PS, GREY = "#9aa0d4", "#17a398", "#c8511b", "#8a8a8a"
    has_pv = ("mmlpf_persev_bits" in ok
              and ok["mmlpf_persev_bits"].notna().any())

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # --- a: mean out-of-sample NLL vs amount of training data ---
    grp = ok.groupby("test_session_number")
    series = [("psytrack_bits", C_PS, "PsyTrack"),
              ("mmlpf_bits", C_MM, "MMLPF")]
    if has_pv:
        series.append(("mmlpf_persev_bits", C_PV, "MMLPF + persev."))
    for col, colour, lab in series:
        m = grp[col].mean()
        se = grp[col].sem()
        ax1.plot(m.index, m.values, color=colour, lw=2.2, label=lab)
        ax1.fill_between(m.index, m - se, m + se, color=colour, alpha=0.18, lw=0)
    ax1.axhline(ok["base_bits"].mean(), color=GREY, ls=":", lw=1.0)
    ax1.text(0.02, ok["base_bits"].mean(), "base rate", fontsize=6.5,
             color=GREY, va="bottom", transform=ax1.get_yaxis_transform())
    ax1.set_xlabel("Test session (chronological)")
    ax1.set_ylabel("Out-of-sample NLL (bits/trial)")
    ax1.set_title("Held-out NLL, mean $\\pm$ SEM across mice", loc="left")
    ax1.set_xticks(sorted(ok["test_session_number"].unique()))
    ax1.legend(frameon=False, fontsize=7, loc="upper right")
    ax1.text(0.02, 0.03, "lower = better", transform=ax1.transAxes,
             fontsize=6.5, color=GREY)
    ax1.margins(y=0.12)

    # --- b: paired per-session scatter against PsyTrack ---
    cols = ["psytrack_bits", "mmlpf_bits"] + (["mmlpf_persev_bits"] if has_pv else [])
    allv = ok[cols].values.ravel()
    allv = allv[np.isfinite(allv)]
    lim = [allv.min() - 0.04, allv.max() + 0.04]
    ax2.plot(lim, lim, color="k", lw=1.0, ls="--", zorder=1)
    ax2.scatter(ok["psytrack_bits"], ok["mmlpf_bits"], s=15, c=C_MM,
                alpha=0.7, lw=0, label="MMLPF", zorder=2)
    if has_pv:
        ax2.scatter(ok["psytrack_bits"], ok["mmlpf_persev_bits"], s=15, c=C_PV,
                    alpha=0.8, lw=0, label="MMLPF + persev.", zorder=3)
    ax2.set_xlim(lim); ax2.set_ylim(lim)
    ax2.set_xlabel("PsyTrack (bits/trial)")
    ax2.set_ylabel("MMLPF variant (bits/trial)")
    n_mm = int((ok["diff_bits"] > 0).sum())
    ttl = f"MMLPF better on {n_mm}/{len(ok)} sessions"
    if has_pv:
        n_pv = int((ok["diff_persev_bits"] > 0).sum())
        ttl += f"; with persev. {n_pv}/{int(ok['diff_persev_bits'].notna().sum())}"
    ax2.set_title(ttl, loc="left", fontsize=8.5)
    ax2.legend(frameon=False, fontsize=7, loc="upper left")
    ax2.text(0.97, 0.06, "below line =\nMMLPF better", transform=ax2.transAxes,
             fontsize=6.5, ha="right", color=GREY)

    # --- c: per-mouse mean gap, both variants ---
    per = ok.groupby("subject_id")["diff_bits"].mean().sort_values()
    y = np.arange(len(per))
    h = 0.38 if has_pv else 0.7
    ax3.barh(y + (h / 2 if has_pv else 0), per.values, height=h, color=C_MM,
             label="MMLPF")
    if has_pv:
        perp = ok.groupby("subject_id")["diff_persev_bits"].mean().reindex(per.index)
        ax3.barh(y - h / 2, perp.values, height=h, color=C_PV,
                 label="MMLPF + persev.")
    ax3.axvline(0, color="k", lw=1.0)
    ax3.set_yticks(y)
    ax3.set_yticklabels(per.index.astype(str), fontsize=6.5)
    ax3.set_xlabel("PsyTrack minus MMLPF (bits/trial)")
    ax3.set_ylabel("Mouse")
    ax3.set_title("Per-mouse mean difference", loc="left")
    ax3.legend(frameon=False, fontsize=7, loc="upper left")
    ax3.text(0.97, 0.02, "right = MMLPF better", transform=ax3.transAxes,
             fontsize=6.5, ha="right", color=GREY)
    ax3.margins(y=0.06)

    for ax, L in zip((ax1, ax2, ax3), "abc"):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(-0.15, 1.02, L, transform=ax.transAxes, fontsize=12,
                fontweight="bold", va="bottom")

    fig.tight_layout(w_pad=2.6)
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"saved {fname}")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="walk_forward_cv.csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--persev", action="store_true",
                    help="also fit MMLPF + perseveration (closes most of the "
                         "gap to PsyTrack and stops beta saturating)")
    ap.add_argument("--summarize", action="store_true",
                    help="skip fitting; summarize and replot an existing CSV")
    args = ap.parse_args()

    WITH_PERSEV = args.persev
    if args.summarize:
        df = pd.read_csv(args.csv)
    else:
        df = run_all(args.csv, workers=args.workers)
    summarize(df)
    plot_results(df)
