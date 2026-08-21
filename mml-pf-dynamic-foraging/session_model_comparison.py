"""
Single-session comparison of MMLPF + perseveration against PsyTrack, drawn on
the lab's standard session format (plot_foraging_session).

Panels, top to bottom:
    1. choice / reward raster + smoothed choice   (plot_foraging_session)
    2. programmed reward probabilities            (plot_foraging_session)
    3. one-step-ahead P(right) from both models, over the choice trace
    4. learning rate alpha            (MMLPF only -- PsyTrack has no alpha)
    5. inverse temperature beta       (MMLPF only)
    6. perseveration weight phi       (MMLPF only)
    7. reward prediction error        (MMLPF only)

Both models are fit on the training sessions and scored one-step-ahead on the
held-out session, exactly as in mmlpf_vs_psytrack_cv.py -- this script reuses
that module's fitting and scoring functions rather than reimplementing them, so
the numbers in the title match what the cohort run reports.

Trial alignment: models are fit on VALID trials only (animal_response != 2), but
the figure uses the rig's original trial numbering with ignored trials left as
gaps. Model traces are scattered back onto the full timeline so that trial N on
the x-axis is trial N in the session, not trial N of the valid subset.

PsyTrack has no learning rate, no inverse temperature and no reward prediction
error -- it is a regression on choice/reward history. Those panels are therefore
MMLPF-only by construction, which is the point of the comparison: the two models
are matched on predictive accuracy but only one yields latent RL quantities.

Usage
-----
    python session_model_comparison.py --subject 689515
    python session_model_comparison.py --subject 689515 --n-sessions 10
    python session_model_comparison.py --demo          # simulated, no database
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mmlpf_vs_psytrack_cv as cv
from plot_foraging_session import plot_foraging_session

N_SESSIONS = 10          # last session is held out, earlier ones train
C_MM = "#17a398"         # MMLPF + perseveration
C_PS = "#c8511b"         # PsyTrack
C_A = "#2ca02c"          # alpha
C_B = "#9467bd"          # beta
C_PHI = "#d62728"        # phi
C_RPE = "#1f77b4"        # RPE
GREY = "#8a8a8a"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_subject(subject_id, n_sessions=N_SESSIONS):
    """Return (train_sessions, test_valid, raw_test_df).

    train_sessions : list of (choices, rewards) on valid trials
    test_valid     : (choices, rewards) on valid trials of the held-out session
    raw_test_df    : the held-out session's rows INCLUDING ignored trials, so
                     the figure can use the original trial numbering
    """
    import aind_dynamic_foraging_database as db

    sessions = db.select_sessions(where=cv.COHORT_QUERY)
    sessions = sessions[sessions["subject_id"].astype(str) == str(subject_id)]
    sessions = sessions.sort_values("session_date").head(n_sessions)
    if len(sessions) < 2:
        raise ValueError(f"{subject_id}: found {len(sessions)} sessions, need >= 2")

    cols = ["animal_response", "earned_reward"]
    for extra in ("reward_probabilityL", "reward_probabilityR"):
        cols.append(extra)
    trials = db.fetch_trials(sessions, columns=cols)

    groups = list(trials.groupby(["session_date", "session_id"], sort=True))
    train_sessions = []
    for _key, g in groups[:-1]:
        v = g[g["animal_response"] != 2]
        train_sessions.append((v["animal_response"].astype(int).values,
                               v["earned_reward"].astype(int).values))
    raw_test = groups[-1][1].reset_index(drop=True)
    v = raw_test[raw_test["animal_response"] != 2]
    test_valid = (v["animal_response"].astype(int).values,
                  v["earned_reward"].astype(int).values)
    return train_sessions, test_valid, raw_test


def simulate_demo(n_sessions=N_SESSIONS, alpha=0.35, beta=6.0, persev=2.0,
                  lapse=0.06, block=60, seed=0, n_trials=450, p_ignore=0.05):
    """Simulated stand-in with the same shape as load_subject's return.

    Used only by --demo, so the plotting path can be exercised without the
    database. A simulator is not evidence about mice; it exercises the code.
    """
    from scipy.special import expit
    rng = np.random.default_rng(seed)

    def one(sd):
        r = np.random.default_rng(sd)
        Q = np.array([0.5, 0.5])
        prev = 0.0
        p = np.array([0.8, 0.1])
        ch, rw, pL, pR = [], [], [], []
        for t in range(n_trials):
            if t and t % block == 0:
                p = p[::-1].copy()
            pr = (1 - lapse) * expit(beta * (Q[1] - Q[0]) + persev * prev) + lapse / 2
            c = int(r.random() < pr)
            rew = int(r.random() < p[c])
            Q[c] += alpha * (rew - Q[c])
            prev = 1.0 if c == 1 else -1.0
            ch.append(c); rw.append(rew); pL.append(p[0]); pR.append(p[1])
        return (np.array(ch), np.array(rw), np.array(pL), np.array(pR))

    sess = [one(seed + i) for i in range(n_sessions)]
    train_sessions = [(c, r) for c, r, _, _ in sess[:-1]]

    c, r, pL, pR = sess[-1]
    ignored = rng.random(n_trials) < p_ignore
    raw_test = pd.DataFrame({
        "animal_response": np.where(ignored, 2, c),
        "earned_reward": np.where(ignored, 0, r),
        "reward_probabilityL": pL,
        "reward_probabilityR": pR,
    })
    test_valid = (c[~ignored], r[~ignored])
    return train_sessions, test_valid, raw_test


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------
def fit_and_score(train_sessions, test_valid):
    """Fit both models on train, score one-step-ahead on the held-out session."""
    test_c, test_r = test_valid

    sa, sb, sp = cv.fit_mmlpf_persev(train_sessions, de_workers=1)
    mm = cv.calculate_nll_window_persev(sa, sb, sp, test_c, test_r, collect=True)
    p_mm = np.where(test_c == 1, mm["p_right"], 1 - mm["p_right"])
    mm_bits = float(-np.mean(np.log2(np.clip(p_mm, 1e-16, None))))

    sigma, w_last = cv.fit_psytrack(train_sessions)
    _ps_nll, ps_p = cv.psytrack_forward_nll(test_valid, sigma, w_last)
    p_ps = np.where(test_c == 1, ps_p, 1 - ps_p)
    ps_bits = float(-np.mean(np.log2(np.clip(p_ps, 1e-16, None))))

    base_p = float(np.clip(np.mean(np.concatenate(
        [c for c, _ in train_sessions])), 1e-6, 1 - 1e-6))
    base_bits = float(-np.mean(np.log2(
        np.where(test_c == 1, base_p, 1 - base_p))))

    return dict(mm=mm, ps_p=ps_p, sigma_alpha=sa, sigma_beta=sb, sigma_phi=sp,
                mm_bits=mm_bits, ps_bits=ps_bits, base_bits=base_bits)


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def make_figure(raw_test, res, fname="session_model_comparison.png",
                subject_id="", smooth_factor=5):
    """Session-format plot with model traces and MMLPF latents underneath."""
    resp = raw_test["animal_response"].values
    choice_full = np.where(resp == 2, np.nan, resp).astype(float)
    reward_full = raw_test["earned_reward"].astype(int).values
    valid = ~np.isnan(choice_full)
    n_trials = len(choice_full)

    if {"reward_probabilityL", "reward_probabilityR"}.issubset(raw_test.columns):
        p_reward = np.vstack([raw_test["reward_probabilityL"].values,
                              raw_test["reward_probabilityR"].values])
    else:
        p_reward = np.full((2, n_trials), np.nan)

    def scatter_back(v):
        """Place a valid-trial trace back onto the original trial numbering."""
        out = np.full(n_trials, np.nan)
        out[valid] = v
        return out

    mm = res["mm"]
    traces = [
        ("P(right), one step ahead",
         [("MMLPF + persev.", scatter_back(mm["p_right"]), C_MM),
          ("PsyTrack", scatter_back(res["ps_p"]), C_PS)],
         (0, 1)),
        (r"Learning rate $\alpha$",
         [("MMLPF", scatter_back(mm["alpha"]), C_A)], None),
        (r"Inverse temp. $\beta$",
         [("MMLPF", scatter_back(mm["beta"]), C_B)], None),
        (r"Perseveration $\varphi$",
         [("MMLPF", scatter_back(mm["phi"]), C_PHI)], None),
        ("RPE",
         [("MMLPF", scatter_back(mm["rpe"]), C_RPE)], None),
    ]

    n_extra = len(traces)
    fig = plt.figure(figsize=(15, 3.2 + 1.5 * n_extra), dpi=200)
    gs = gridspec.GridSpec(1 + n_extra, 1, figure=fig,
                           height_ratios=[3.0] + [1.0] * n_extra, hspace=0.32)

    # plot_foraging_session consumes the axis it is given: it builds a 2-row
    # subgridspec from that slot (choice raster + reward schedule) and removes
    # the placeholder.
    ax_slot = fig.add_subplot(gs[0, 0])
    _fig, (ax_choice, ax_sched) = plot_foraging_session(
        choice_history=choice_full, reward_history=reward_full,
        p_reward=p_reward, ax=ax_slot, smooth_factor=smooth_factor)

    # plot_foraging_session labels the reward-schedule axis as the bottom of a
    # standalone figure. Here it sits mid-stack, so hand the x-axis to the
    # lowest trace panel instead of leaving a label and ticks in the middle.
    ax_sched.set_xlabel("")
    ax_sched.tick_params(labelbottom=False)

    n_ign = int(np.sum(resp == 2))
    ax_choice.text(
        0.0, 1.52,
        f"subject {subject_id}   held-out session, {n_trials} trials "
        f"({n_ign} ignored)\n"
        f"one-step-ahead NLL: MMLPF + persev. {res['mm_bits']:.3f}   "
        f"PsyTrack {res['ps_bits']:.3f}   base rate {res['base_bits']:.3f} "
        f"bits/trial (lower = better)\n"
        f"M-step on {res['n_train_sessions']} training sessions: "
        f"$\\sigma_\\alpha$={res['sigma_alpha']:.3f}  "
        f"$\\sigma_\\beta$={res['sigma_beta']:.3f}  "
        f"$\\sigma_\\varphi$={res['sigma_phi']:.3f}",
        transform=ax_choice.transAxes, fontsize=8, va="bottom")

    axes = [ax_choice, ax_sched]
    for i, (ylab, series, ylim) in enumerate(traces):
        ax = fig.add_subplot(gs[1 + i, 0], sharex=ax_choice)
        for lab, y, colour in series:
            ax.plot(np.arange(1, n_trials + 1), y, color=colour, lw=1.0,
                    label=lab)
        ax.set_ylabel(ylab, fontsize=8)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if len(series) > 1:
            ax.legend(fontsize=6.5, frameon=False, ncol=len(series),
                      loc="upper right")
        if ylab == "RPE":
            ax.axhline(0, color=GREY, lw=0.6, ls=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
        if i < n_extra - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Trial number", fontsize=8)
        axes.append(ax)

    # beta saturation is a fitting pathology, not a finding -- flag it in place
    b = mm["beta"]
    if np.mean(b > 95.0) > 0.01:
        ax_b = axes[2 + 2]
        ax_b.text(0.995, 0.92,
                  f"saturating: {np.mean(b > 95.0):.1%} of trials above 95 "
                  f"(clip 100) -- not interpretable",
                  transform=ax_b.transAxes, fontsize=6.5, ha="right",
                  va="top", color=C_PS)

    fig.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"saved {fname}")
    return fig, axes


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", default=None,
                    help="subject_id; omit with --demo")
    ap.add_argument("--n-sessions", type=int, default=N_SESSIONS,
                    help="sessions to use; the last is held out")
    ap.add_argument("--demo", action="store_true",
                    help="run on simulated data (no database access)")
    ap.add_argument("--out", default="session_model_comparison.png")
    args = ap.parse_args()

    if args.demo:
        train_sessions, test_valid, raw_test = simulate_demo(args.n_sessions)
        sid = "simulated"
    else:
        if args.subject is None:
            ap.error("--subject is required unless --demo is passed")
        train_sessions, test_valid, raw_test = load_subject(
            args.subject, args.n_sessions)
        sid = args.subject

    print(f"{len(train_sessions)} training sessions "
          f"({sum(len(c) for c, _ in train_sessions)} valid trials), "
          f"held-out session {len(test_valid[0])} valid trials")

    res = fit_and_score(train_sessions, test_valid)
    res["n_train_sessions"] = len(train_sessions)
    print(f"MMLPF + persev. {res['mm_bits']:.4f} bits/trial")
    print(f"PsyTrack        {res['ps_bits']:.4f} bits/trial")
    print(f"base rate       {res['base_bits']:.4f} bits/trial")
    print(f"gap (PsyTrack - MMLPF) {res['ps_bits'] - res['mm_bits']:+.4f} "
          f"bits/trial; positive favours MMLPF")

    make_figure(raw_test, res, fname=args.out, subject_id=sid)


if __name__ == "__main__":
    main()
