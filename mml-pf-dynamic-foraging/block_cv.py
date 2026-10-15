"""
Interleaved-block cross-validation, per Pillow's prescription, applied
symmetrically to PsyTrack and to the MMLPF with perseveration.

WHY THIS EXISTS
---------------
Our walk-forward design trains on sessions 1..i-1 and tests on session i. That
answers "can the model predict the next session", and it is matched -- both
models get a causal forward filter. But it handicaps the MMLPF in a way that is
an artefact of the design rather than of the model: across the train/test
boundary PsyTrack carries 10 numbers (5 random-walk variances + 5 fitted
weights) while the MMLPF carries 2 (sigma_alpha, sigma_beta) and re-draws its
alpha/beta/phi particles from a fixed prior at the start of every test session.
That is why the held-out NLL was flat across sessions: the MMLPF could not use
the extra training data.

Pillow's scheme removes that asymmetry. Held-out trials are BLOCKS INSIDE the
session, so both models are warm at every test point:

  - hold out blocks of consecutive trials (he suggests 10-20 in a row, and
    blocks rather than scattered singletons so train and test are closer to
    independent -- in a bandit every trial's regressors depend on its
    predecessors, so scattered test trials are each surrounded by training
    trials and the sets are badly entangled);
  - fit hyperparameters on the remaining trials only -- "you just wouldn't
    include the likelihoods from those trials";
  - keep ALL data as regressors, held-out trials included. The animal's choice
    and reward on a held-out trial are observed; they belong in x_t and in the
    Q update. Only the likelihood term is withheld.

WHAT EACH MODEL GETS
--------------------
Identical treatment, which is the whole point:

  PsyTrack : Laplace forward filter over the session. On a training trial the
             weights are updated; on a held-out trial they are propagated but
             not updated. Predictive P(right) is read off the propagated
             weights. Optional RTS backward pass (--smooth) implements Pillow's
             point 1 -- posterior mode given the training data rather than
             linear interpolation -- and is two-sided.
  MMLPF+phi: the same filter you already run. On a training trial particles are
             reweighted and resampled; on a held-out trial they are propagated
             but not reweighted. The Q update runs on EVERY trial, because the
             choice and reward are observed either way. Predictive P(right) is
             the particle mean before any reweighting.

Both models' hyperparameters are fit by the same optimiser on the same
objective: the summed predictive negative log-likelihood over training trials
only. Using psytrack's own hyperOpt here would not work -- it has no way to
exclude the held-out trials from its evidence, which is exactly the leak the
scheme is designed to avoid.

WHAT THIS DOES NOT CLAIM
------------------------
Two-sided inference at held-out blocks is straightforward for the GLM (the RTS
pass) and awkward for the particle filter: Q is a deterministic function of
alpha and the observed history, so the smoothing transition density is
degenerate and standard backward-simulation smoothing does not apply cleanly.
The default therefore runs BOTH models one-sided, which is symmetric and
answers the prediction question. `--smooth` scores the GLM two-sided as well,
so you can measure how much Pillow's point 1 is worth; if it is small, the
one-sided comparison is the honest one to report.

Usage
-----
    python block_cv.py --demo
    python block_cv.py --subjects 713379 751766
    python block_cv.py --subjects 713379 --block 20 --test-frac 0.2 --smooth
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.special import expit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mmlpf_vs_psytrack_cv as cv

N_LAGS = cv.N_LAGS
M_PARTICLES = 300           # particles during the hyperparameter search
E_PARTICLES = 1500          # particles for the scored pass
DE_MAXITER = 20
DE_POPSIZE = 12
# Parameterisation version. Bumped whenever a change alters what the states
# MEAN, so that results fit under different versions cannot be silently pooled
# or resumed into one another. v2: alpha moved from log to logit space and the
# beta clip moved from the value to the state (see mmlpf_masked).
PARAM_VERSION = 2

# alpha lives in (0, 1) and is tracked in LOGIT space, so no clip is required:
# expit maps the whole real line into the open interval. It was previously
# tracked in log space and clipped at 0.99, which allowed a particle's state to
# random-walk far above the clip while displaying as 0.99 -- an absorbing shelf
# where many distinct states collapsed to one value. On data simulated with a
# CONSTANT alpha of 0.35, that put 32% of particle mass above alpha = 0.95 at
# sigma_alpha = 0.22 and 44% at 0.40. The apparent bimodality was the
# parameterisation, not the animal.
ALPHA_PRIOR_LOGIT = (-0.847, 0.65)   # logit(0.30); sd chosen to span ~0.14-0.53

# beta is positive and unbounded above, so log space is the right choice for
# it, but the same shelf existed: exp(P) was clipped while P itself kept
# walking. The state is now clipped instead, which makes the boundary
# one-step-reversible.
LOG_BETA_LO, LOG_BETA_HI = np.log(0.01), np.log(100.0)

def _resample_idx(weights, num, scheme):
    """Indices of surviving particles.

    "multinomial" draws num independent samples -- the highest-variance choice
    available, and the original behaviour. "systematic" places num equally
    spaced pointers after a single uniform jitter, which has strictly lower
    resampling variance for the same particle count and is the standard
    default in the particle-filter literature. Lower resampling variance means
    a larger effective sample size for the same N, which is what the volatility
    fit is short of.
    """
    if scheme == "systematic":
        u = np.random.uniform(0.0, 1.0 / num)
        pos = u + np.arange(num) / num
        cum = np.cumsum(weights)
        cum[-1] = 1.0
        return np.searchsorted(cum, pos)
    if scheme != "multinomial":
        raise ValueError(f"resample must be 'systematic' or 'multinomial', "
                         f"got {scheme!r}")
    return np.random.choice(num, size=num, p=weights)


DE_BOUNDS = [(0.001, 0.4)] * 3
# Fourth entry is sigma_bias, and it is DELIBERATELY NARROWER than the others.
#
# With the wide (0.001, 0.4) range the volatility fit fits sigma_bias ~ 0.16 even when
# the generative bias is exactly constant, and ~0.19-0.30 on real sessions --
# an order of magnitude above the truth in the cases where truth is known. The
# consequence is measured, not hypothetical: the wandering bias state absorbs
# 30-45% of the perseveration weight (median phi 1.35 -> 0.76 on constant-bias
# simulations) and costs ~0.02-0.04 bits/trial of held-out accuracy. A free
# walk here does not let the data decide how much the bias drifts; binary
# choices carry too little information per trial to pin it down, so the walk
# wanders by default.
#
# 0.05 keeps a genuinely slow within-session drift reachable -- PsyTrack's own
# bias weight moves by ~2 units across a real session, which a hard 0 could not
# represent -- while making the runaway values unreachable. Lower bound is 0.0
# so the volatility fit can choose a fully static bias.
#
# Results from before this change used (0.001, 0.4) and are NOT comparable;
# any 6-state cohort numbers predating it need refitting.
DE_BOUNDS_BIAS = [(0.001, 0.4)] * 3 + [(0.0, 0.05)]
# Fifth entry is sigma_engage, and it is deliberately MUCH WIDER than the
# others. That looks inconsistent with the tight sigma_bias bound; it is not,
# and the reason matters.
#
# A bias state has no observation channel of its own -- it is inferred only from
# how skewed the choices look -- so a wide walk there wanders and steals
# variance from phi. Engagement does have its own channel: the no-response
# trials are direct observations of theta, so the data pins it down and the
# bound only has to be wide enough to let it move.
#
# And it must move fast. Observed ignore runs are 27-32 trials with abrupt
# onsets, so theta has to travel several logits within ~10 trials. Measured on
# simulated data with known ignore blocks, the separation between theta inside
# and outside those blocks is +0.001 at sigma 0.05, +0.44 at 0.30 and +0.68 at
# 1.5, with training NLL still improving to 1.5 and flat by 2.0. A 0.05 cap --
# the value that is right for the bias -- leaves theta frozen at its prior and
# the state does nothing.
DE_BOUNDS_ENGAGE = [(0.001, 0.4)] * 3 + [(0.0, 0.05), (0.01, 2.0)]
GLM_BOUNDS = [(1e-4, 1.0)]  # one shared sigma for the GLM random walk
BURN_IN = 50                # leading trials always assigned to training


# ---------------------------------------------------------------------------
# block mask
# ---------------------------------------------------------------------------
def make_block_mask(n_trials, block=20, test_frac=0.2, burn_in=BURN_IN, seed=0):
    """True = training trial, False = held out.

    Blocks are consecutive and non-overlapping. The first `burn_in` trials are
    always training: every model needs some history before its predictions mean
    anything, and scoring a cold model tells you about the prior, not the model.
    """
    rng = np.random.default_rng(seed)
    mask = np.ones(n_trials, dtype=bool)
    usable = np.arange(burn_in, n_trials - block + 1, block)
    if len(usable) == 0:
        raise ValueError(
            f"session of {n_trials} trials cannot hold a {block}-trial block "
            f"after a {burn_in}-trial burn-in")
    n_blocks = max(1, int(round(test_frac * n_trials / block)))
    n_blocks = min(n_blocks, len(usable))
    starts = rng.choice(usable, size=n_blocks, replace=False)
    for s in starts:
        mask[s:s + block] = False
    return mask


# ---------------------------------------------------------------------------
# PsyTrack model under a training mask
# ---------------------------------------------------------------------------
# Moved here from matched_validation.py, which was removed. Every caller in
# the pipeline imported this one function and nothing else from that module,
# and it belongs beside the GLM it feeds.
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


def glm_masked(X, y, sigma, train_mask, w0=None, P0=0.25, smooth=False,
               return_weights=False):
    """Laplace filter that updates only on training trials.

    Returns (p_pred, p_smooth). `p_pred` is the one-sided predictive P(right) at
    every trial -- at a held-out trial this is the weight propagated from the
    last training trial, never updated by the held-out outcome. `p_smooth` is
    the two-sided RTS version, or None when smooth=False.

    With return_weights=True, returns (p_pred, p_smooth, w_pred, P_pred): the
    predictive weight trajectory and its covariance at every trial. These are
    PsyTrack's latents -- the direct counterpart of the particle filter's
    alpha/beta/phi -- and w_pred is the state actually used to predict trial t,
    read BEFORE any update, so it is comparable to the particle mean.
    """
    T, K = X.shape
    s2 = np.atleast_1d(np.asarray(sigma, float)) ** 2
    if s2.size == 1:
        s2 = np.full(K, s2[0])
    Qd = np.diag(s2)

    w = np.zeros(K) if w0 is None else np.asarray(w0, float).copy()
    P = np.eye(K) * P0

    w_pred = np.zeros((T, K)); P_pred = np.zeros((T, K, K))
    w_filt = np.zeros((T, K)); P_filt = np.zeros((T, K, K))
    p_pred = np.zeros(T)

    for t in range(T):
        P = P + Qd
        w_pred[t] = w
        P_pred[t] = P
        p_pred[t] = expit(np.clip(w @ X[t], -50, 50))

        if train_mask[t]:
            x = X[t]
            p = p_pred[t]
            Px = P @ x
            s = max(p * (1 - p), 1e-9)
            P = P - np.outer(Px, Px) * (s / (1.0 + s * (x @ Px)))
            w = w + P @ (x * (y[t] - p))
        w_filt[t] = w
        P_filt[t] = P

    if not smooth:
        if return_weights:
            return p_pred, None, w_pred, P_pred
        return p_pred, None

    # RTS backward pass. Random walk => predicted mean at t+1 is the filtered
    # mean at t, so the smoother gain is P_filt[t] @ inv(P_pred[t+1]).
    w_s = w_filt.copy()
    for t in range(T - 2, -1, -1):
        J = P_filt[t] @ np.linalg.inv(P_pred[t + 1])
        w_s[t] = w_filt[t] + J @ (w_s[t + 1] - w_filt[t])
    p_smooth = expit(np.clip(np.einsum("tk,tk->t", w_s, X), -50, 50))
    if return_weights:
        return p_pred, p_smooth, w_pred, P_pred
    return p_pred, p_smooth


def _glm_train_nll(log_sigma, X, y, train_mask):
    p, _ = glm_masked(X, y, np.exp(log_sigma[0]), train_mask)
    pc = np.where(y == 1, p, 1.0 - p)[train_mask]
    return -np.sum(np.log(np.clip(pc, 1e-12, 1.0)))


def fit_glm_sigma(X, y, train_mask, maxiter=DE_MAXITER, popsize=DE_POPSIZE,
                  workers=1):
    """Fit the GLM random-walk sigma on TRAINING trials only."""
    res = differential_evolution(
        _glm_train_nll, bounds=[(np.log(1e-4), np.log(1.0))],
        args=(X, y, train_mask), maxiter=maxiter, popsize=popsize,
        workers=workers, tol=0.01, seed=0, polish=False)
    return float(np.exp(res.x[0]))


# ---------------------------------------------------------------------------
# MMLPF + perseveration under a training mask
# ---------------------------------------------------------------------------
def mmlpf_masked(sigma_alpha, sigma_beta, sigma_phi, choices, rewards,
                 train_mask, num_particles=M_PARTICLES, seed=42, collect=False,
                 sigma_bias=None, hist_bins=None, sigma_engage=None,
                 ignored=None, resample="multinomial",
                 resample_threshold=1.0):
    """The perseveration filter, reweighting only on training trials.

    Identical to calculate_nll_window_persev except for the mask. The Q update
    runs on every trial: the animal's choice and reward are observed on a
    held-out trial too, and withholding them would be withholding data rather
    than withholding a likelihood term.

    OPTIONAL DRIFTING BIAS (sigma_bias)
    -----------------------------------
    sigma_bias=None (default) is the five-state filter:

        z = beta_t * (Q_R - Q_L) + phi_t * c_{t-1}

    Passing a float adds a SIXTH state b_t, the analogue of PsyTrack's bias
    regressor:

        z = beta_t * (Q_R - Q_L) + phi_t * c_{t-1} + b_t
        b_t = b_{t-1} + N(0, sigma_bias)

    b_t is a constant offset -- identical on every trial -- whereas
    phi * c_{t-1} FLIPS SIGN with the animal's last choice. They are different
    regressors and cannot substitute: phi can absorb at most (2p-1)^2 of a
    constant bias, where p = P(preferred side), so 36% for an 80/20 animal and
    4% for a 60/40 one.

    BACKWARD COMPATIBILITY is load-bearing. With sigma_bias=None no extra
    random draws are made, so the RNG stream is identical to the original
    implementation and every previously reported number reproduces exactly. Do
    not "simplify" this by always drawing the sixth state.
    """
    np.random.seed(seed)
    T = len(choices)
    with_bias = sigma_bias is not None
    # Engagement needs BOTH a volatility and an observed ignore channel: without
    # `ignored` there are no no-response events to identify theta from, and the
    # state would be a free parameter with no data behind it.
    with_eng = sigma_engage is not None and ignored is not None
    if sigma_engage is not None and ignored is None:
        raise ValueError("sigma_engage requires `ignored`: the engagement state "
                         "is identified from observed no-response trials, not "
                         "inferred from choices alone")
    if ignored is None:
        ignored = np.zeros(T, dtype=bool)
    ignored = np.asarray(ignored, dtype=bool)

    # Callers mark ignored trials with an out-of-range choice (-1) so that any
    # code path forgetting the mask fails loudly. Numpy would instead index the
    # LAST column silently, so build an explicit safe index: the value is never
    # used on an ignored trial (its likelihood is 1 - theta and Q does not
    # move), but it must not be a negative index either.
    choices = np.asarray(choices)
    bad = ~ignored & ~np.isin(choices, (0, 1))
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} answered trials have a choice outside (0, 1); "
            f"first at index {int(np.argmax(bad))} with value "
            f"{choices[np.argmax(bad)]}")
    safe_choices = np.where(ignored, 0, choices).astype(int)

    # Adaptive resampling. Resampling on EVERY trial discards particle
    # diversity even when the weights are near-uniform: multinomial draws with
    # uniform weights still leave only N(1-(1-1/N)^N) ~ 0.63N distinct
    # particles. Resampling only when the effective sample size falls below a
    # threshold keeps that diversity. With resample_threshold = 1.0 the filter
    # resamples every trial and every code path below is byte-identical to the
    # original, so old results reproduce exactly.
    adaptive = resample_threshold < 1.0
    w = np.full(num_particles, 1.0 / num_particles) if adaptive else None
    if adaptive:
        def wmean(x):
            return float(np.sum(w * x))

        def wstd(x):
            m = float(np.sum(w * x))
            return float(np.sqrt(max(np.sum(w * (x - m) ** 2), 0.0)))

        def whist(x, e):
            # scaled to sum to num_particles so the units match the
            # unweighted path
            return np.histogram(x, bins=e, weights=w * num_particles)[0]
    else:
        wmean, wstd = np.mean, np.std

        def whist(x, e):
            return np.histogram(x, bins=e)[0]

    # column layout, built so that DISABLED states draw no random numbers at
    # all -- that is what keeps the RNG stream, and therefore every earlier
    # result, bit-identical when a state is switched off
    c_bias = 5 if with_bias else None
    c_eng = (5 + int(with_bias)) if with_eng else None
    P = np.zeros((num_particles, 5 + int(with_bias) + int(with_eng)))
    P[:, 0] = np.random.uniform(0, 1, num_particles)
    P[:, 1] = np.random.uniform(0, 1, num_particles)
    P[:, 2] = np.random.normal(*ALPHA_PRIOR_LOGIT, num_particles)
    P[:, 3] = np.random.normal(np.log(15.0), 0.5, num_particles)
    P[:, 4] = np.random.normal(0.0, 0.5, num_particles)
    if with_bias:
        # centred on zero with phi's prior width: PsyTrack's bias weight also
        # starts at 0 rather than at the observed choice rate
        P[:, c_bias] = np.random.normal(0.0, 0.5, num_particles)
    if with_eng:
        # logit space, centred on theta = 0.88 because most trials in this task
        # are answered. Weakly informative on purpose: theta is identified by
        # the observed ignores, unlike alpha/beta/phi which are inferred only
        # from choice consistency.
        P[:, c_eng] = np.random.normal(2.0, 1.0, num_particles)

    nll_train = 0.0
    prev = 0.0
    p_right = np.zeros(T)
    out = ({k: np.zeros(T) for k in
            ("alpha", "beta", "phi", "rpe", "q_left", "q_right",
             "alpha_sd", "beta_sd", "phi_sd", "p_right_sd",
             "bias", "bias_sd", "engage", "engage_sd")}
           if collect else None)

    # Optional per-trial particle HISTOGRAMS. The posterior over a latent is not
    # generally unimodal -- resampling can split the cloud -- so mean +- sd can
    # describe a distribution that does not exist. hist_bins returns the full
    # marginal at every trial so multimodality and collapse are visible.
    # beta is binned in log10 because it spans 0.01 to 100.
    do_hist = collect and hist_bins
    if do_hist:
        nb = int(hist_bins)
        edges = {"alpha": np.linspace(0.0, 1.0, nb + 1),
                 "beta": np.linspace(-2.0, 2.0, nb + 1),      # log10 scale
                 "phi": np.linspace(-6.0, 6.0, nb + 1),
                 "bias": np.linspace(-6.0, 6.0, nb + 1)}
        for k, e in edges.items():
            out[f"{k}_hist"] = np.zeros((T, nb))
            out[f"{k}_edges"] = e

    for t in range(T):
        P[:, 2] += np.random.normal(0, sigma_alpha, num_particles)
        P[:, 3] += np.random.normal(0, sigma_beta, num_particles)
        P[:, 4] += np.random.normal(0, sigma_phi, num_particles)
        if with_bias:
            P[:, c_bias] += np.random.normal(0, sigma_bias, num_particles)
        if with_eng:
            P[:, c_eng] += np.random.normal(0, sigma_engage, num_particles)

        # clip the STATE, not the value: a state left free to wander past the
        # boundary takes many steps to return, which is what created the
        # pile-up at the clip
        P[:, 3] = np.clip(P[:, 3], LOG_BETA_LO, LOG_BETA_HI)
        alpha_vals = expit(P[:, 2])          # in (0, 1) by construction
        beta_vals = np.exp(P[:, 3])          # in [0.01, 100] by construction
        phi_vals = P[:, 4]
        # clipped so a runaway bias cannot single-handedly saturate the logit;
        # generous bound -- |b| = 9.2 already gives P(right) = 0.9999
        bias_vals = np.clip(P[:, c_bias], -10.0, 10.0) if with_bias else 0.0
        # P(the animal answers this trial at all)
        theta_vals = (expit(np.clip(P[:, c_eng], -10.0, 10.0))
                      if with_eng else None)
        Q = P[:, :2]

        z = beta_vals * (Q[:, 1] - Q[:, 0]) + phi_vals * prev + bias_vals
        pr_right = expit(np.clip(z, -50, 50))
        probs = np.column_stack([1.0 - pr_right, pr_right])
        p_right[t] = wmean(pr_right)

        if collect:
            out["alpha"][t] = wmean(alpha_vals)
            out["beta"][t] = wmean(beta_vals)
            out["phi"][t] = wmean(phi_vals)
            out["alpha_sd"][t] = wstd(alpha_vals)
            out["beta_sd"][t] = wstd(beta_vals)
            out["phi_sd"][t] = wstd(phi_vals)
            out["p_right_sd"][t] = wstd(pr_right)
            out["q_left"][t] = wmean(Q[:, 0])
            out["q_right"][t] = wmean(Q[:, 1])
            out["rpe"][t] = (np.nan if ignored[t] else
                             rewards[t] - wmean(Q[:, safe_choices[t]]))
            if with_bias:
                out["bias"][t] = wmean(bias_vals)
                out["bias_sd"][t] = wstd(bias_vals)
            if with_eng:
                out["engage"][t] = wmean(theta_vals)
                out["engage_sd"][t] = wstd(theta_vals)
            if do_hist:
                out["alpha_hist"][t] = whist(alpha_vals, edges["alpha"])
                out["beta_hist"][t] = whist(
                    np.log10(np.clip(beta_vals, 1e-2, 1e2)), edges["beta"])
                out["phi_hist"][t] = whist(phi_vals, edges["phi"])
                if with_bias:
                    out["bias_hist"][t] = whist(bias_vals, edges["bias"])

        p_choice = probs[np.arange(num_particles), safe_choices[t]]

        # p_obs is the likelihood of what was actually OBSERVED on this trial.
        # Without engagement that is the choice. With engagement the observation
        # is two-stage: either the animal did not answer (probability 1 - theta)
        # or it answered and chose (theta * p_choice). p_right stays the
        # CONDITIONAL choice probability either way, so scoring against a model
        # without an engagement state compares like with like.
        if with_eng:
            p_obs = (1.0 - theta_vals) if ignored[t] else theta_vals * p_choice
        else:
            p_obs = p_choice

        if train_mask[t]:
            if adaptive:
                # the incremental log evidence uses the weights carried in from
                # earlier trials: log p(y_t | y_1:t-1) = log sum_i w_i p_obs_i.
                # It reduces to log mean(p_obs) whenever w is uniform, which is
                # every trial in the resample-always path.
                nll_train -= np.log(float(np.sum(w * p_obs)) + 1e-16)
                w = w * p_obs
                tot = float(np.sum(w))
                w = (np.full(num_particles, 1.0 / num_particles)
                     if tot < 1e-20 else w / tot)
                if 1.0 / np.sum(w ** 2) < resample_threshold * num_particles:
                    P = P[_resample_idx(w, num_particles, resample)].copy()
                    w = np.full(num_particles, 1.0 / num_particles)
            else:
                nll_train -= np.log(np.mean(p_obs) + 1e-16)
                weights = p_obs / (np.sum(p_obs) + 1e-16)
                if np.sum(weights) < 1e-20:
                    weights = np.ones(num_particles) / num_particles
                weights /= np.sum(weights)
                P = P[_resample_idx(weights, num_particles, resample)].copy()

        # observed either way -- the choice happened, only its likelihood term
        # is withheld. On an IGNORED trial there was no choice and no outcome,
        # so Q does not move and `prev` carries the last actual choice forward:
        # perseveration is about the last thing the animal DID.
        if not ignored[t]:
            ar = expit(P[:, 2])
            ci = safe_choices[t]
            P[:, ci] += ar * (rewards[t] - P[:, ci])
            prev = 1.0 if ci == 1 else -1.0

    if collect:
        out["p_right"] = p_right
        out["nll_train"] = nll_train
        return out
    return nll_train, p_right


def _mmlpf_train_nll(hyper, choices, rewards, train_mask,
                     resample="systematic", resample_threshold=0.5):
    """Objective for the 3-parameter (no bias) volatility fit.

    The resampling scheme is an explicit argument rather than a module-level
    setting because differential_evolution(workers>1) pickles this function and
    its args; a module global would not survive a spawn-based worker.
    """
    sa, sb, sp = hyper
    nll, _ = mmlpf_masked(sa, sb, sp, choices, rewards, train_mask,
                          resample=resample,
                          resample_threshold=resample_threshold)
    return nll


def _mmlpf_bias_train_nll(hyper, choices, rewards, train_mask,
                          resample="systematic", resample_threshold=0.5):
    """Objective for the 4-parameter volatility fit, drifting bias enabled.

    Separate function rather than a branch inside _mmlpf_train_nll so the
    3-parameter path is untouched and stays picklable for workers>1.
    """
    sa, sb, sp, sbias = hyper
    nll, _ = mmlpf_masked(sa, sb, sp, choices, rewards, train_mask,
                          sigma_bias=sbias, resample=resample,
                          resample_threshold=resample_threshold)
    return nll


def fit_pooled(sessions, maxiter=DE_MAXITER, popsize=DE_POPSIZE,
               num_particles=M_PARTICLES, seed=0, with_bias=True,
               with_engage=False, bounds=None,
               resample="systematic", resample_threshold=0.5):
    """ONE volatility vector for all sessions of a subject.

    WHY POOL
    --------
    The per-session volatility estimate is barely identified. On six sessions
    simulated from an identical generative process with a TRUE sigma_alpha of
    exactly zero, the per-session estimate ranged over [0.001, 0.400] -- the
    whole search interval -- with a median of 0.13. The likelihood surface in
    sigma is nearly flat and what the optimiser finds is dominated by
    session-level noise, not by the parameter.

    Pooling replaces S noisy draws with one estimate from the joint likelihood.
    It is also FREE: the per-session scheme runs S separate searches of D
    evaluations each, the pooled scheme runs one search of D evaluations that
    each touch S sessions. Same S*D filter runs either way.

    NOT LEAKAGE
    -----------
    Every session's held-out blocks are excluded from its own training
    likelihood, so no held-out trial contributes to the estimate. A session is
    scored with a sigma informed by that subject's OTHER sessions, which is
    pooling across independent replicates -- the same thing a hierarchical
    model does -- not information flowing backwards from the test set.

    Parameters
    ----------
    sessions : list of dicts with keys "choices", "rewards", "train_mask", and
        optionally "ignored".
    bounds : search box. Defaults to the module bounds for the requested model
        size, which are deliberately tight for sigma_bias and sigma_engage --
        with a likelihood surface this flat a bound is a weak prior doing real
        work, not a thumb on the scale.

    Returns
    -------
    (sigmas, result) -- the fitted tuple and the scipy OptimizeResult.
    """
    if not sessions:
        raise ValueError("fit_pooled needs at least one session")
    if bounds is None:
        bounds = (DE_BOUNDS_ENGAGE if with_engage
                  else DE_BOUNDS_BIAS if with_bias else DE_BOUNDS)

    def obj(v):
        total = 0.0
        for s in sessions:
            total += mmlpf_masked(
                v[0], v[1], v[2], s["choices"], s["rewards"], s["train_mask"],
                num_particles=num_particles, seed=seed,
                sigma_bias=(v[3] if with_bias or with_engage else None),
                sigma_engage=(v[4] if with_engage else None),
                ignored=s.get("ignored"),
                resample=resample, resample_threshold=resample_threshold)[0]
        return total

    res = differential_evolution(obj, bounds=bounds, maxiter=maxiter,
                                 popsize=popsize, tol=0.01, seed=0,
                                 polish=False)
    return tuple(float(x) for x in res.x), res


def fit_mmlpf_masked(choices, rewards, train_mask, maxiter=DE_MAXITER,
                     popsize=DE_POPSIZE, workers=1, with_bias=False,
                     resample="systematic", resample_threshold=0.5):
    """Maximum marginal likelihood fit of the volatilities, training trials only.

    with_bias=False -> returns (sigma_alpha, sigma_beta, sigma_phi)
    with_bias=True  -> returns (sigma_alpha, sigma_beta, sigma_phi, sigma_bias)

    The 4-parameter search costs roughly 4/3 as many objective evaluations for
    the same maxiter/popsize, and each evaluation is marginally slower, so
    budget about 1.5x the runtime of the 3-parameter fit.
    """
    if with_bias:
        res = differential_evolution(
            _mmlpf_bias_train_nll, bounds=DE_BOUNDS_BIAS,
            args=(choices, rewards, train_mask, resample,
                  resample_threshold), maxiter=maxiter,
            popsize=popsize, workers=workers, tol=0.01, seed=0, polish=False)
    else:
        res = differential_evolution(
            _mmlpf_train_nll, bounds=DE_BOUNDS,
            args=(choices, rewards, train_mask, resample,
                  resample_threshold), maxiter=maxiter,
            popsize=popsize, workers=workers, tol=0.01, seed=0, polish=False)
    return tuple(float(v) for v in res.x)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def bits(p_right, y, mask):
    """Mean bits/trial over the masked subset."""
    pc = np.where(y == 1, p_right, 1.0 - p_right)[mask]
    return float(-np.mean(np.log2(np.clip(pc, 1e-12, 1.0))))


def score_session(choices, rewards, block=20, test_frac=0.2, seed=0,
                  smooth=False, workers=1, verbose=True, sigmas=None,
                  fit_only=False, resample="systematic",
                  resample_threshold=0.5):
    """Run the block-CV comparison on one session of valid trials.

    sigmas : (sigma_alpha, sigma_beta, sigma_phi) to USE rather than fit. This
        is what enables two-pass subject-level scoring: pass 1 fits a sigma per
        session, the caller takes the median across that subject's sessions,
        and pass 2 re-scores every session with the shared value. The
        per-session estimate is dominated by session noise -- on sessions
        simulated with a TRUE sigma of exactly zero the per-session fit ranged
        over essentially the whole search interval, while the median across
        four sessions had about a third of the across-subject spread.

    fit_only : return just the fitted sigmas, skipping the GLM and the scored
        pass. Pass 1 needs nothing else, and a scored pass at E_PARTICLES is
        wasted when its sigma is about to be replaced.
    """
    T = len(choices)
    train_mask = make_block_mask(T, block=block, test_frac=test_frac, seed=seed)
    test_mask = ~train_mask
    if verbose:
        print(f"  {T} valid trials; {int(test_mask.sum())} held out "
              f"({test_mask.mean():.0%}) in {int(test_mask.sum() // block)} "
              f"blocks of {block}")

    if fit_only:
        return fit_mmlpf_masked(choices, rewards, train_mask, workers=workers,
                                resample=resample,
                                resample_threshold=resample_threshold)

    sess = np.zeros(T, dtype=int)
    rew, unrew = build_glm_regressors(choices, rewards, sess, n_lags=N_LAGS)
    X = np.column_stack([np.ones(T), rew, unrew])

    sig = fit_glm_sigma(X, choices, train_mask, workers=workers)
    p_glm, p_glm_s = glm_masked(X, choices, sig, train_mask, smooth=smooth)

    if sigmas is None:
        sa, sb, sp = fit_mmlpf_masked(
            choices, rewards, train_mask, workers=workers, resample=resample,
            resample_threshold=resample_threshold)
    else:
        sa, sb, sp = (float(v) for v in sigmas)
    pf = mmlpf_masked(sa, sb, sp, choices, rewards, train_mask,
                      num_particles=E_PARTICLES, collect=True,
                      resample=resample, resample_threshold=resample_threshold)

    base = np.full(T, choices[train_mask].mean())

    def _concentration(p):
        """Share of total held-out loss coming from the worst 5% of trials.

        Log loss is unbounded on confident errors: one trial at p=1e-4 costs
        ~13 bits, which over 120 held-out trials moves the mean by ~0.11
        bits/trial on its own. A model whose loss is concentrated this way has
        a mean that is a statement about a handful of trials, not about the
        session -- report the median across block placements instead.
        """
        pc = np.where(choices == 1, p, 1.0 - p)[test_mask]
        L = -np.log2(np.clip(pc, 1e-12, 1.0))
        n5 = max(1, int(0.05 * len(L)))
        return float(np.sort(L)[::-1][:n5].sum() / L.sum()), float(L.max())

    mm_conc, mm_worst = _concentration(pf["p_right"])
    glm_conc, glm_worst = _concentration(p_glm)

    row = dict(
        n_trials=T, n_test=int(test_mask.sum()), block=block,
        base_bits=bits(base, choices, test_mask),
        glm_bits=bits(p_glm, choices, test_mask),
        mmlpf_bits=bits(pf["p_right"], choices, test_mask),
        glm_sigma=sig, sigma_alpha=sa, sigma_beta=sb, sigma_phi=sp,
        mean_beta=float(pf["beta"].mean()), max_beta=float(pf["beta"].max()),
        mean_phi=float(pf["phi"].mean()),
        mmlpf_loss_conc=mm_conc, mmlpf_worst_bits=mm_worst,
        glm_loss_conc=glm_conc, glm_worst_bits=glm_worst,
    )
    row["diff_bits"] = row["glm_bits"] - row["mmlpf_bits"]
    if smooth and p_glm_s is not None:
        row["glm_bits_smooth"] = bits(p_glm_s, choices, test_mask)
        row["smooth_gain"] = row["glm_bits"] - row["glm_bits_smooth"]
    return row, dict(train_mask=train_mask, p_glm=p_glm, p_glm_smooth=p_glm_s,
                     pf=pf)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_session(subject_id, session_number=2, max_sessions=None):
    """Load one session's valid trials, using the cohort's session ordering."""
    import aind_dynamic_foraging_database as db

    max_sessions = max_sessions or cv.N_SESSIONS
    sessions = db.select_sessions(where=cv.COHORT_QUERY).sort_values(
        by=["subject_id", "session_date"])
    sessions = sessions[sessions["subject_id"].astype(str) == str(subject_id)]
    if len(sessions) < session_number:
        raise ValueError(f"subject {subject_id}: only {len(sessions)} sessions "
                         f"pass COHORT_QUERY")
    picked = sessions.head(max_sessions)
    trials = db.fetch_trials(picked, columns=["animal_response", "earned_reward"])
    groups = list(trials.groupby(["session_date", "session_id"], sort=True))
    g = groups[session_number - 1][1]
    v = g[g["animal_response"] != 2]
    return (v["animal_response"].astype(int).values,
            v["earned_reward"].astype(int).values, len(g))


def simulate_demo(T=600, alpha=0.35, beta=6.0, phi=2.0, lapse=0.06,
                  block=60, seed=0):
    """Q-learner with perseveration and lapses. For pipeline testing only --
    data generated by one of the models under test cannot adjudicate between
    them."""
    rng = np.random.default_rng(seed)
    Q = np.array([0.5, 0.5]); prev = 0.0; p = np.array([0.8, 0.1])
    ch, rw = [], []
    for t in range(T):
        if t and t % block == 0:
            p = p[::-1].copy()
        pr = (1 - lapse) * expit(beta * (Q[1] - Q[0]) + phi * prev) + lapse / 2
        c = int(rng.random() < pr)
        r = int(rng.random() < p[c])
        Q[c] += alpha * (r - Q[c])
        prev = 1.0 if c == 1 else -1.0
        ch.append(c); rw.append(r)
    return np.array(ch), np.array(rw)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="*", default=["713379", "751766"])
    ap.add_argument("--session", type=int, default=2,
                    help="which qualifying session to score")
    ap.add_argument("--block", type=int, default=20,
                    help="held-out block length in trials (Pillow: 10-20)")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seeds", type=int, default=5,
                    help="repeat with different block placements")
    ap.add_argument("--smooth", action="store_true",
                    help="also score the GLM two-sided (RTS), to measure what "
                         "Pillow's point 1 is worth")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--csv", default="block_cv_results.csv")
    ap.add_argument("--demo", action="store_true",
                    help="run on simulated data, no database needed")
    args = ap.parse_args()

    rows = []
    if args.demo:
        jobs = [("demo", simulate_demo(seed=1))]
    else:
        jobs = []
        for sid in args.subjects:
            c, r, n_raw = load_session(sid, session_number=args.session)
            print(f"subject {sid}: session {args.session}, {n_raw} trials, "
                  f"{len(c)} valid")
            jobs.append((sid, (c, r)))

    for sid, (c, r) in jobs:
        print(f"\n{'=' * 66}\nsubject {sid}\n{'=' * 66}")
        for seed in range(args.seeds):
            row, _ = score_session(c, r, block=args.block,
                                   test_frac=args.test_frac, seed=seed,
                                   smooth=args.smooth, workers=args.workers,
                                   verbose=(seed == 0))
            row.update(subject_id=sid, seed=seed)
            rows.append(row)
            extra = (f"  GLM(smooth) {row['glm_bits_smooth']:.4f}"
                     if "glm_bits_smooth" in row else "")
            print(f"  seed {seed}: GLM {row['glm_bits']:.4f}  "
                  f"MMLPF {row['mmlpf_bits']:.4f}  "
                  f"diff {row['diff_bits']:+.4f}{extra}")

    df = pd.DataFrame(rows)
    df.to_csv(args.csv, index=False)

    print(f"\n{'=' * 66}\nBLOCK-CV SUMMARY (held-out blocks, both models masked "
          f"identically)\n{'=' * 66}")
    g = df.groupby("subject_id").agg(
        glm_med=("glm_bits", "median"), mmlpf_med=("mmlpf_bits", "median"),
        diff_med=("diff_bits", "median"), diff_mean=("diff_bits", "mean"),
        diff_sd=("diff_bits", "std"), base=("base_bits", "mean"),
        max_beta=("max_beta", "max"), conc=("mmlpf_loss_conc", "mean"))
    print(g.round(4).to_string())
    print("\nmedian is the headline: block placement is arbitrary, and a single "
          "\nplacement where the filter goes confidently wrong inside one block "
          "\nmoves the mean by more than the effect being measured.")

    # per-subject stability check
    for sid, sub in df.groupby("subject_id"):
        gsd = sub["glm_bits"].std()
        msd = sub["mmlpf_bits"].std()
        ratio = (msd / gsd) ** 2 if gsd > 0 else np.nan
        bad = sub[sub["mmlpf_bits"] > sub["base_bits"]]
        if ratio > 4 or len(bad):
            print(f"\n  WARNING {sid}: MMLPF variance across block placements is "
                  f"{ratio:.0f}x the GLM's")
            if len(bad):
                print(f"    {len(bad)}/{len(sub)} placements score WORSE than the "
                      f"base rate -- a model predicting at chance would beat it. "
                      f"\n    That is a filter failure on those placements, not a "
                      f"model comparison.")
            print(f"    loss concentration {sub['mmlpf_loss_conc'].mean():.2f} "
                  f"(share of loss from worst 5% of held-out trials); "
                  f"worst single trial {sub['mmlpf_worst_bits'].max():.1f} bits")
            print(f"    -> raise --seeds to 20+ and report the median, or "
                  f"investigate those placements before quoting a mean")
    if "smooth_gain" in df:
        print(f"\nGLM two-sided (RTS) minus one-sided: "
              f"{df['smooth_gain'].mean():+.4f} bits/trial. "
              f"If this is small, Pillow's point 1 does not change the verdict "
              f"and the symmetric one-sided comparison is the one to report.")
    print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
