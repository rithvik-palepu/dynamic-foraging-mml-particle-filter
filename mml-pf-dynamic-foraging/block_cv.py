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
from matched_validation import build_glm_regressors

N_LAGS = cv.N_LAGS
M_PARTICLES = 300           # particles during the hyperparameter search
E_PARTICLES = 1500          # particles for the scored pass
DE_MAXITER = 20
DE_POPSIZE = 12
DE_BOUNDS = [(0.001, 0.4)] * 3
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
                 train_mask, num_particles=M_PARTICLES, seed=42, collect=False):
    """The perseveration filter, reweighting only on training trials.

    Identical to calculate_nll_window_persev except for the mask. The Q update
    runs on every trial: the animal's choice and reward are observed on a
    held-out trial too, and withholding them would be withholding data rather
    than withholding a likelihood term.
    """
    np.random.seed(seed)
    T = len(choices)
    P = np.zeros((num_particles, 5))
    P[:, 0] = np.random.uniform(0, 1, num_particles)
    P[:, 1] = np.random.uniform(0, 1, num_particles)
    P[:, 2] = np.random.normal(np.log(0.3), 0.5, num_particles)
    P[:, 3] = np.random.normal(np.log(15.0), 0.5, num_particles)
    P[:, 4] = np.random.normal(0.0, 0.5, num_particles)

    nll_train = 0.0
    prev = 0.0
    p_right = np.zeros(T)
    out = ({k: np.zeros(T) for k in
            ("alpha", "beta", "phi", "rpe", "q_left", "q_right",
             "alpha_sd", "beta_sd", "phi_sd", "p_right_sd")}
           if collect else None)

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
        p_right[t] = np.mean(pr_right)

        if collect:
            out["alpha"][t] = np.mean(alpha_vals)
            out["beta"][t] = np.mean(beta_vals)
            out["phi"][t] = np.mean(phi_vals)
            out["alpha_sd"][t] = np.std(alpha_vals)
            out["beta_sd"][t] = np.std(beta_vals)
            out["phi_sd"][t] = np.std(phi_vals)
            out["p_right_sd"][t] = np.std(pr_right)
            out["q_left"][t] = np.mean(Q[:, 0])
            out["q_right"][t] = np.mean(Q[:, 1])
            out["rpe"][t] = rewards[t] - np.mean(Q[:, choices[t]])

        p_choice = probs[np.arange(num_particles), choices[t]]

        if train_mask[t]:
            nll_train -= np.log(np.mean(p_choice) + 1e-16)
            weights = p_choice / (np.sum(p_choice) + 1e-16)
            if np.sum(weights) < 1e-20:
                weights = np.ones(num_particles) / num_particles
            weights /= np.sum(weights)
            idx = np.random.choice(num_particles, size=num_particles, p=weights)
            P = P[idx].copy()

        # observed either way -- the choice happened, only its likelihood term
        # is withheld
        ar = np.clip(np.exp(P[:, 2]), 1e-4, 0.99)
        P[:, choices[t]] += ar * (rewards[t] - P[:, choices[t]])
        prev = 1.0 if choices[t] == 1 else -1.0

    if collect:
        out["p_right"] = p_right
        out["nll_train"] = nll_train
        return out
    return nll_train, p_right


def _mmlpf_train_nll(hyper, choices, rewards, train_mask):
    sa, sb, sp = hyper
    nll, _ = mmlpf_masked(sa, sb, sp, choices, rewards, train_mask)
    return nll


def fit_mmlpf_masked(choices, rewards, train_mask, maxiter=DE_MAXITER,
                     popsize=DE_POPSIZE, workers=1):
    """M-step over (sigma_alpha, sigma_beta, sigma_phi) on training trials."""
    res = differential_evolution(
        _mmlpf_train_nll, bounds=DE_BOUNDS,
        args=(choices, rewards, train_mask), maxiter=maxiter, popsize=popsize,
        workers=workers, tol=0.01, seed=0, polish=False)
    return tuple(float(v) for v in res.x)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def bits(p_right, y, mask):
    """Mean bits/trial over the masked subset."""
    pc = np.where(y == 1, p_right, 1.0 - p_right)[mask]
    return float(-np.mean(np.log2(np.clip(pc, 1e-12, 1.0))))


def score_session(choices, rewards, block=20, test_frac=0.2, seed=0,
                  smooth=False, workers=1, verbose=True):
    """Run the block-CV comparison on one session of valid trials."""
    T = len(choices)
    train_mask = make_block_mask(T, block=block, test_frac=test_frac, seed=seed)
    test_mask = ~train_mask
    if verbose:
        print(f"  {T} valid trials; {int(test_mask.sum())} held out "
              f"({test_mask.mean():.0%}) in {int(test_mask.sum() // block)} "
              f"blocks of {block}")

    sess = np.zeros(T, dtype=int)
    rew, unrew = build_glm_regressors(choices, rewards, sess, n_lags=N_LAGS)
    X = np.column_stack([np.ones(T), rew, unrew])

    sig = fit_glm_sigma(X, choices, train_mask, workers=workers)
    p_glm, p_glm_s = glm_masked(X, choices, sig, train_mask, smooth=smooth)

    sa, sb, sp = fit_mmlpf_masked(choices, rewards, train_mask, workers=workers)
    pf = mmlpf_masked(sa, sb, sp, choices, rewards, train_mask,
                      num_particles=E_PARTICLES, collect=True)

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
