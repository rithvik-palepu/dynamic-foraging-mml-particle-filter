"""
Targeted out-of-sample validation: MMLPF (w/ Perseveration) vs dynamic GLM (PsyTrack)
Train: Session 1 | Test: Session 2
Target Subjects: 713379, 751766

Same structure as the original two_subject_comparison.py. Four things changed,
each because the original produced numbers that disagreed with the cohort run
(walk_forward_cv.csv) for the same subject and session:

1. SESSION SELECTION. The original queried `subject_id == X` with no other
   filter, so its "Session 1/2" were the first two sessions the animal ever ran
   -- not the first two that pass COHORT_QUERY. For 713379 the cohort's test
   session 2 has 607 trials and 14 ignored; the original plot showed ~540 trials
   and 200+ ignored, i.e. a different session entirely. This module applies
   COHORT_QUERY and the same MAX_SESSIONS cap the cohort used.

2. THE FILTER. The original ran a hand-written particle filter with volatilities
   hardcoded at 0.02/0.2/0.1 and beta capped at 20. That is not the MMLPF: the
   "MML" is maximum marginal likelihood over the volatilities, fit by
   differential evolution. This module calls the same
   `fit_mmlpf_persev` / `calculate_nll_window_persev` the cohort run used, so
   the bits/trial printed here reproduce the CSV.

3. M-STEP ON TRAINING ONLY. Volatilities are fit on session 1 and then held
   fixed while scoring session 2. Fitting them on the test session would be
   scoring in-sample.

4. LATENT DISTRIBUTIONS. All four latents (alpha, beta, phi, and P(right)) are
   drawn as posterior mean +/- 1 SD across particles, taken before the weight
   update so the band is the one-step-ahead predictive spread.

Usage
-----
    python two_subject_comparison.py
    python two_subject_comparison.py --subjects 713379 751766
    python two_subject_comparison.py --subjects 713379 --train-sessions 1
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import psytrack

# Make sibling modules importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matched_validation as mv
from matched_validation import build_glm_regressors, bits_per_trial
from scipy.special import expit
from plot_foraging_session import plot_foraging_session

# the cohort run's own fitting and scoring -- imported, not reimplemented, so
# the numbers here cannot drift from walk_forward_cv.csv
import mmlpf_vs_psytrack_cv as cv

N_LAGS = cv.N_LAGS
MML_E_PARTICLES = 1500      # particles for the scored forward pass
MAX_SESSIONS = cv.N_SESSIONS

# default pair: one session where MMLPF+phi edges PsyTrack, one where it loses,
# both with few ignored trials (713379 s2: 14/607; 751766 s8: 0/630)
DEFAULT_SUBJECTS = ["713379", "751766"]


# ==========================================
# 1. MMLPF w/ Perseveration (the real one)
# ==========================================
def run_mmlpf_persev(train_sessions, test_choices, test_rewards,
                     num_particles=MML_E_PARTICLES, de_workers=1):
    """M-step on the training sessions, then a scored forward pass on test.

    Returns the same dict shape the original's `run_particle_filter_with_q_and_phi`
    produced (means and SDs per trial), plus the fitted volatilities.
    """
    sa, sb, sp = cv.fit_mmlpf_persev(train_sessions, de_workers=de_workers)

    out = cv.calculate_nll_window_persev(
        sa, sb, sp, test_choices, test_rewards,
        num_particles=num_particles, collect=True)

    return {
        "p_right_mean": out["p_right"], "p_right_std": out["p_right_sd"],
        "alpha_mean": out["alpha"], "alpha_std": out["alpha_sd"],
        "beta_mean": out["beta"], "beta_std": out["beta_sd"],
        "phi_mean": out["phi"], "phi_std": out["phi_sd"],
        "q_left_mean": out["q_left"], "q_right_mean": out["q_right"],
        "rpe": out["rpe"], "nll": out["nll"],
        "sigma_alpha": sa, "sigma_beta": sb, "sigma_phi": sp,
    }


# ==========================================
# 2. PsyTrack Wrapper & Data Loading
# ==========================================
def glm_filter(X, y, sigma, w0, P0=0.25):
    """Causal one-step-ahead Laplace filter for the PsyTrack generative model.

    PsyTrack's own hyperOpt is a SMOOTHER -- it uses the whole session including
    future choices. Scoring prediction with it would leak the answer, so the
    weights are filtered forward here instead.
    """
    T, K = X.shape
    w = np.asarray(w0, float).copy()
    P = np.eye(K) * P0
    s2 = np.atleast_1d(np.asarray(sigma, float)) ** 2
    if s2.size == 1:
        s2 = np.full(K, s2[0])
    p_hat = np.zeros(T)
    w_trace = np.zeros((T, K))
    for t in range(T):
        P = P + np.diag(s2)
        x = X[t]
        p = expit(w @ x)
        p_hat[t] = p
        w_trace[t] = w
        Px = P @ x
        s = max(p * (1 - p), 1e-9)
        P = P - np.outer(Px, Px) * (s / (1.0 + s * (x @ Px)))
        w = w + P @ (x * (y[t] - p))
    return p_hat, w_trace


def load_train_test(subject_id, n_train=1):
    """Load the first `n_train` qualifying sessions as train, the next as test.

    Session ordering matches the cohort run exactly: COHORT_QUERY, sorted by
    (subject_id, session_date), capped at MAX_SESSIONS, then grouped by
    (session_date, session_id). Without COHORT_QUERY the same subject yields a
    different session 1/2 and every number downstream disagrees with the CSV.
    """
    import aind_dynamic_foraging_database as db

    sessions = db.select_sessions(where=cv.COHORT_QUERY).sort_values(
        by=["subject_id", "session_date"])
    sessions = sessions[sessions["subject_id"].astype(str) == str(subject_id)]
    if len(sessions) < n_train + 1:
        raise ValueError(
            f"subject {subject_id}: {len(sessions)} sessions pass COHORT_QUERY, "
            f"need >= {n_train + 1}")
    picked = sessions.head(MAX_SESSIONS)

    cols = ["animal_response", "earned_reward",
            "reward_probabilityL", "reward_probabilityR"]
    try:
        trials = db.fetch_trials(picked, columns=cols)
    except Exception:
        trials = db.fetch_trials(picked, columns=cols[:2])

    groups = list(trials.groupby(["session_date", "session_id"], sort=True))
    if len(groups) < n_train + 1:
        raise ValueError(
            f"subject {subject_id}: only {len(groups)} sessions after fetch")

    blocks = []
    for i, (_key, g) in enumerate(groups[:n_train + 1]):
        v = g[g["animal_response"] != 2]
        blocks.append((v["animal_response"].astype(int).values,
                       v["earned_reward"].astype(int).values,
                       np.full(len(v), i)))

    train = tuple(np.concatenate([b[i] for b in blocks[:n_train]])
                  for i in range(3))
    test = blocks[n_train]
    raw_test = groups[n_train][1].reset_index(drop=True)

    n_ign = int((raw_test["animal_response"] == 2).sum())
    print(f"  session {n_train + 1} (test): {len(raw_test)} trials, "
          f"{n_ign} ignored ({n_ign / len(raw_test):.1%}), "
          f"{len(test[0])} valid")
    return train, test, raw_test


def run(train, test, de_workers=1):
    """One-step-ahead comparison of the GLM and the MMLPF on the test session."""
    tr_c, tr_r, tr_s = train
    te_c, te_r, _te_s = test

    # --- regressors over train+test, session-boundary aware ---
    all_c = np.concatenate([tr_c, te_c])
    all_r = np.concatenate([tr_r, te_r])
    all_s = np.concatenate([tr_s, np.full(len(te_c), tr_s.max() + 1)])
    rew, unrew = build_glm_regressors(all_c, all_r, all_s, n_lags=N_LAGS)
    X = np.column_stack([np.ones(len(all_c)), rew, unrew])
    n_tr = len(tr_c)

    # --- GLM hyperparameters on TRAIN only ---
    day_lengths = np.array([np.sum(tr_s == s) for s in np.unique(tr_s)])
    K = 1 + 2 * N_LAGS
    train_dict = {"y": tr_c + 1,
                  "inputs": {"rew": rew[:n_tr], "unrew": unrew[:n_tr]},
                  "dayLength": day_lengths}
    wdict = {"bias": 1, "rew": N_LAGS, "unrew": N_LAGS}
    hyp, _, wMode, _ = psytrack.hyperOpt(
        train_dict,
        {"sigma": [2 ** -5] * K, "sigInit": 2 ** 5, "sigDay": [2 ** -5] * K},
        wdict, ["sigma", "sigDay"], showOpt=0)

    # --- one-step-ahead predictions on TEST ---
    p_glm, w_trace = glm_filter(X[n_tr:], te_c, hyp["sigma"], wMode[:, -1])

    # --- MMLPF w/ perseveration: M-step on train, forward pass on test ---
    train_sessions = [(tr_c[tr_s == s], tr_r[tr_s == s])
                      for s in np.unique(tr_s)]
    pf = run_mmlpf_persev(train_sessions, te_c, te_r, de_workers=de_workers)

    scores = {
        "base rate": bits_per_trial(np.full(len(te_c), tr_c.mean()), te_c),
        "GLM one-step": bits_per_trial(p_glm, te_c),
        "MMLPF one-step": bits_per_trial(pf["p_right_mean"], te_c),
    }
    return dict(pf=pf, p_glm=p_glm, w_trace=w_trace, scores=scores,
                te_c=te_c, te_r=te_r)


# ==========================================
# 3. Plotting Logic
# ==========================================
def generate_comparison_plot(res, raw_test, subject_id, n_train=1, fname=None):
    """Behaviour panel plus every latent with its particle distribution."""
    p_glm = res["p_glm"]
    pf = res["pf"]

    resp = raw_test["animal_response"].values
    choice_full = np.where(resp == 2, np.nan, resp).astype(float)
    reward_full = raw_test["earned_reward"].astype(int).values
    keep = ~np.isnan(choice_full)

    def scatter_back(v):
        """Place a valid-trial trace back on the rig's original numbering."""
        out = np.full(len(choice_full), np.nan)
        out[keep] = v
        return out

    if {"reward_probabilityL", "reward_probabilityR"}.issubset(raw_test.columns):
        p_reward = np.vstack([raw_test["reward_probabilityL"].values,
                              raw_test["reward_probabilityR"].values])
    else:
        p_reward = np.zeros((2, len(choice_full)))

    fig = plt.figure(figsize=(14, 14), dpi=150)
    gs = gridspec.GridSpec(5, 1, height_ratios=[1.5, 0.6, 0.6, 0.6, 0.6],
                           hspace=0.32)
    ax0 = fig.add_subplot(gs[0])
    ax_alpha = fig.add_subplot(gs[1], sharex=ax0)
    ax_beta = fig.add_subplot(gs[2], sharex=ax0)
    ax_phi = fig.add_subplot(gs[3], sharex=ax0)
    ax_rpe = fig.add_subplot(gs[4], sharex=ax0)

    # --- 1. behaviour + both models' one-step-ahead P(right) ---
    _, axes = plot_foraging_session(
        choice_history=choice_full, reward_history=reward_full,
        p_reward=p_reward, plot_list=["choice"], ax=ax0)
    ax0 = axes[0]

    trials = np.arange(len(choice_full))
    pr = scatter_back(pf["p_right_mean"])
    pr_sd = scatter_back(pf["p_right_std"])
    valid_mask = ~np.isnan(pr)

    ax0.plot(trials, scatter_back(p_glm), color="#c8511b", linewidth=1.4,
             alpha=0.85, label="PsyTrack P(Right)")
    ax0.fill_between(trials, pr - pr_sd, pr + pr_sd, where=valid_mask,
                     color="#17a398", alpha=0.25, lw=0)
    ax0.plot(trials, pr, color="#17a398", linewidth=1.4, alpha=0.9,
             label="MMLPF P(Right)")
    ax0.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4,
               fontsize=9)

    s = res["scores"]
    title_str = (
        f"Subject {subject_id} | Train: Session 1"
        f"{'' if n_train == 1 else f'-{n_train}'}, Test: Session {n_train + 1}\n"
        f"GLM: {s['GLM one-step']:.3f} bits/trial | "
        f"MMLPF (w/ $\\phi$): {s['MMLPF one-step']:.3f} bits/trial | "
        f"base rate: {s['base rate']:.3f}\n"
        f"M-step on training: $\\sigma_\\alpha$={pf['sigma_alpha']:.3f}  "
        f"$\\sigma_\\beta$={pf['sigma_beta']:.3f}  "
        f"$\\sigma_\\varphi$={pf['sigma_phi']:.3f}")
    ax0.set_title(title_str, fontweight="bold", pad=44, fontsize=11)

    # --- 2-4. latents, mean +/- 1 SD across particles ---
    panels = [
        (ax_alpha, "alpha", r"Learning Rate ($\alpha$)", "#2ca02c"),
        (ax_beta, "beta", r"Inverse Temp ($\beta$)", "#9467bd"),
        (ax_phi, "phi", r"Perseveration ($\varphi$)", "#d95f02"),
    ]
    for ax, key, ylab, colour in panels:
        m = scatter_back(pf[f"{key}_mean"])
        sd = scatter_back(pf[f"{key}_std"])
        vm = ~np.isnan(m)
        ax.plot(trials, m, color=colour, linewidth=1.4,
                label=f"Filter {ylab.split('(')[1].rstrip(')')}")
        ax.fill_between(trials, m - sd, m + sd, where=vm, color=colour,
                        alpha=0.3, lw=0)
        ax.set_ylabel(ylab)
        ax.legend(loc="upper right", fontsize=9, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelbottom=False)

    # phi is signed: above 0 = repeat the last choice, below = alternate
    ax_phi.axhline(0, color="#999999", lw=0.8, ls=":", zorder=0)

    # --- 5. RPE (posterior mean; a point estimate, not a distribution) ---
    ax_rpe.plot(trials, scatter_back(pf["rpe"]), color="#1f77b4", linewidth=1.0,
                label="RPE (posterior mean)")
    ax_rpe.axhline(0, color="#999999", lw=0.8, ls=":", zorder=0)
    ax_rpe.set_ylabel("RPE")
    ax_rpe.set_xlabel("Trial Number", fontsize=11)
    ax_rpe.legend(loc="upper right", fontsize=9, frameon=False)
    ax_rpe.spines["top"].set_visible(False)
    ax_rpe.spines["right"].set_visible(False)
    ax_rpe.set_xlim([0, len(choice_full)])

    # beta saturation makes the alpha/beta traces uninterpretable -- say so on
    # the figure rather than letting the reader assume they are meaningful
    frac_clip = float(np.mean(pf["beta_mean"] > 95.0))
    if frac_clip > 0.01:
        ax_beta.text(0.01, 0.04,
                     f"beta > 95 on {frac_clip:.1%} of trials (clip = 100): "
                     f"saturating, trace not interpretable",
                     transform=ax_beta.transAxes, fontsize=7.5, color="#c8511b")

    fname = fname or f"comparison_{subject_id}_sess1_sess{n_train + 1}_with_phi.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {fname}")
    return fname


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="*", default=DEFAULT_SUBJECTS)
    ap.add_argument("--train-sessions", type=int, default=1,
                    help="number of leading sessions to train on; the next "
                         "session is the test session")
    ap.add_argument("--de-workers", type=int, default=1,
                    help="parallel workers for the differential_evolution "
                         "M-step")
    args = ap.parse_args()

    rows = []
    for sid in args.subjects:
        print(f"\n{'=' * 60}")
        print(f"Processing Subject {sid}...")
        print(f"{'=' * 60}")
        try:
            train, test, raw_test = load_train_test(
                sid, n_train=args.train_sessions)
            print(f"Loaded train: {len(train[0])} valid trials over "
                  f"{args.train_sessions} session(s)")

            t0 = time.time()
            res = run(train, test, de_workers=args.de_workers)
            print(f"\nFinished in {time.time() - t0:.1f} s. "
                  f"Scores on session {args.train_sessions + 1}:")
            for k, v in res["scores"].items():
                print(f"  {k:22s} {v:.3f} bits/trial")
            gap = res["scores"]["GLM one-step"] - res["scores"]["MMLPF one-step"]
            print(f"  {'gap (GLM - MMLPF)':22s} {gap:+.3f}  "
                  f"({'MMLPF' if gap > 0 else 'PsyTrack'} better)")

            generate_comparison_plot(res, raw_test, sid,
                                     n_train=args.train_sessions)
            rows.append(dict(subject_id=sid, **res["scores"], gap=gap,
                             sigma_alpha=res["pf"]["sigma_alpha"],
                             sigma_beta=res["pf"]["sigma_beta"],
                             sigma_phi=res["pf"]["sigma_phi"]))
        except Exception as e:
            print(f"Failed processing {sid}: {e}")
            import traceback
            traceback.print_exc()

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv("two_subject_comparison.csv", index=False)
        print(f"\n{'=' * 60}")
        print(df.round(4).to_string(index=False))
        print("\nwrote two_subject_comparison.csv")


if __name__ == "__main__":
    main()
