"""
Both models' latents and choice probabilities over one session, on the lab's
standard session format (plot_foraging_session).

Panels, top to bottom:
    1. choice / reward raster + smoothed choice      (plot_foraging_session)
    2. programmed reward probabilities               (plot_foraging_session)
    3. one-step-ahead P(right), BOTH models, with held-out blocks shaded
    4. PsyTrack weights: bias, rewarded-choice, unrewarded-choice  (+/- 1 SD)
    5. MMLPF learning rate alpha                                   (+/- 1 SD)
    6. MMLPF inverse temperature beta                              (+/- 1 SD)
    7. MMLPF perseveration phi                                     (+/- 1 SD)
    8. MMLPF reward prediction error

WHY BOTH MODELS GET LATENT PANELS
---------------------------------
PsyTrack is not a black box against which only the MMLPF has internals. Its
weights ARE its latents: the bias weight is a drifting side preference, and the
rewarded/unrewarded choice-history weights are how strongly the animal repeats
a choice that paid versus one that did not. Panel 4 is the honest counterpart
to panels 5-7, and it makes the actual asymmetry visible: PsyTrack has no
learning rate, no inverse temperature, and no reward prediction error, because
it has no value representation. That absence is the argument for the MMLPF, and
it reads better when the reader can see PsyTrack's own latents next to it
rather than an empty half of the figure.

Both models are run under the SAME block-CV mask, so the shaded spans in
panels 3-8 are trials neither model was allowed to learn from. Inside a shaded
span, watch the weights go flat (frozen) while the particle latents keep
diffusing -- that is the difference in how the two models extrapolate.

Everything is drawn on the rig's ORIGINAL trial numbering: models are fit on
valid trials only, and traces are scattered back so ignored trials appear as
gaps rather than silently compressing the axis.

Usage
-----
    python latent_comparison.py --subject 713379 --session 2
    python latent_comparison.py --demo
    python latent_comparison.py --subject 647286 --session 3 --block 20 --seed 0
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import block_cv as bc
import mmlpf_vs_psytrack_cv as cv
from matched_validation import build_glm_regressors
from plot_foraging_session import plot_foraging_session

C_GLM = "#c8511b"
C_MM = "#17a398"
C_A, C_B, C_P, C_R = "#2ca02c", "#9467bd", "#d95f02", "#1f77b4"
GREY = "#8a8a8a"
SHADE = "#cfcfcf"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_session(subject_id, session_number=2):
    """One session, using the cohort's session ordering. Returns valid-trial
    arrays plus the RAW frame so ignored trials can be drawn as gaps."""
    import aind_dynamic_foraging_database as db

    sessions = db.select_sessions(where=cv.COHORT_QUERY).sort_values(
        by=["subject_id", "session_date"])
    sessions = sessions[sessions["subject_id"].astype(str) == str(subject_id)]
    if len(sessions) < session_number:
        raise ValueError(
            f"subject {subject_id}: {len(sessions)} sessions pass "
            f"COHORT_QUERY, cannot reach session {session_number}")
    # Fetch at least enough sessions to REACH the requested one. cv.N_SESSIONS
    # is 10 (the walk-forward default) while the block-CV cohort used 20, so
    # head(cv.N_SESSIONS) silently truncated and any request above 10 died with
    # an opaque "list index out of range" from the groups[] lookup below.
    n_fetch = max(cv.N_SESSIONS, session_number)
    picked = sessions.head(n_fetch)

    cols = ["animal_response", "earned_reward",
            "reward_probabilityL", "reward_probabilityR"]
    try:
        trials = db.fetch_trials(picked, columns=cols)
    except Exception:
        trials = db.fetch_trials(picked, columns=cols[:2])

    groups = list(trials.groupby(["session_date", "session_id"], sort=True))
    if session_number > len(groups):
        raise ValueError(
            f"subject {subject_id}: asked for session {session_number} but only "
            f"{len(groups)} sessions came back from fetch_trials (requested "
            f"{n_fetch}). The session numbering must match "
            f"hpc_block_cv.prepare's, which orders by session_date and takes "
            f"the first {getattr(cv, 'N_SESSIONS', '?')}+ per subject.")
    raw = groups[session_number - 1][1].reset_index(drop=True)
    v = raw[raw["animal_response"] != 2]
    n_ign = int((raw["animal_response"] == 2).sum())
    print(f"subject {subject_id} session {session_number}: {len(raw)} trials, "
          f"{n_ign} ignored ({n_ign / len(raw):.1%}), {len(v)} valid")
    return (v["animal_response"].astype(int).values,
            v["earned_reward"].astype(int).values, raw)


def simulate_demo(T=600, seed=1):
    c, r = bc.simulate_demo(T=T, seed=seed)
    rng = np.random.default_rng(seed + 100)
    # ~4% ignored, and a block-switching reward schedule, to exercise the
    # gap handling and the p_reward panel
    ign = rng.random(T) < 0.04
    resp = np.where(ign, 2, c)
    pl, pr = np.zeros(T), np.zeros(T)
    p = np.array([0.8, 0.1])
    for t in range(T):
        if t and t % 60 == 0:
            p = p[::-1].copy()
        pl[t], pr[t] = p[0], p[1]
    raw = pd.DataFrame(dict(animal_response=resp, earned_reward=r,
                            reward_probabilityL=pl, reward_probabilityR=pr))
    v = raw[raw["animal_response"] != 2]
    return (v["animal_response"].astype(int).values,
            v["earned_reward"].astype(int).values, raw)


# ---------------------------------------------------------------------------
# fit both models under one mask
# ---------------------------------------------------------------------------
def fit_both(choices, rewards, block=20, test_frac=0.2, seed=0,
             n_particles=bc.E_PARTICLES, workers=1):
    T = len(choices)
    train_mask = bc.make_block_mask(T, block=block, test_frac=test_frac,
                                   seed=seed)
    rew, unrew = build_glm_regressors(choices, rewards, np.zeros(T, dtype=int),
                                      n_lags=cv.N_LAGS)
    X = np.column_stack([np.ones(T), rew, unrew])

    sig = bc.fit_glm_sigma(X, choices, train_mask, workers=workers)
    p_glm, _, w_pred, P_pred = bc.glm_masked(
        X, choices, sig, train_mask, return_weights=True)

    sa, sb, sp = bc.fit_mmlpf_masked(choices, rewards, train_mask,
                                     workers=workers)
    pf = bc.mmlpf_masked(sa, sb, sp, choices, rewards, train_mask,
                         num_particles=n_particles, collect=True)

    test_mask = ~train_mask
    base = np.full(T, choices[train_mask].mean())
    scores = dict(base=bc.bits(base, choices, test_mask),
                  glm=bc.bits(p_glm, choices, test_mask),
                  mmlpf=bc.bits(pf["p_right"], choices, test_mask))
    # marginal SD of each weight, for the +/- 1 SD band
    w_sd = np.sqrt(np.einsum("tkk->tk", P_pred))
    return dict(train_mask=train_mask, p_glm=p_glm, w=w_pred, w_sd=w_sd,
                pf=pf, scores=scores, glm_sigma=sig,
                sigma_alpha=sa, sigma_beta=sb, sigma_phi=sp, X=X)


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def make_figure(raw, res, subject_id="", session_number=None,
                fname="latent_comparison.png", smooth_factor=5,
                block_len=20):
    resp = raw["animal_response"].values
    choice_full = np.where(resp == 2, np.nan, resp).astype(float)
    reward_full = raw["earned_reward"].astype(int).values
    keep = ~np.isnan(choice_full)
    n_full = len(choice_full)
    trials = np.arange(n_full)

    def sb(v):
        """Scatter a valid-trial trace back onto the original trial numbering."""
        out = np.full(n_full, np.nan)
        out[keep] = np.asarray(v, float)
        return out

    if {"reward_probabilityL", "reward_probabilityR"}.issubset(raw.columns):
        p_reward = np.vstack([raw["reward_probabilityL"].values,
                              raw["reward_probabilityR"].values])
    else:
        p_reward = np.zeros((2, n_full))

    # held-out spans, mapped onto the original numbering.
    # An ignored trial sitting inside a held-out block is NaN after scattering,
    # which would split one block into two shaded slivers. A block is a
    # contiguous run of held-out VALID trials, so treat an interior NaN as
    # continuation and only close the span at the next training trial.
    held = sb((~res["train_mask"]).astype(float))
    spans = []
    in_span = False
    for t in range(n_full):
        if np.isnan(held[t]):
            continue                      # ignored trial: neither opens nor closes
        if held[t] == 1.0 and not in_span:
            start, in_span = t, True
        elif held[t] == 0.0 and in_span:
            spans.append((start, t)); in_span = False
    if in_span:
        spans.append((start, n_full))
    n_blocks = len(spans)
    # blocks are drawn without replacement from non-overlapping starts, so two
    # can land adjacent and merge into one contiguous shaded span. Report both
    # numbers rather than letting the merged count masquerade as the design.
    n_drawn = int((~res["train_mask"]).sum()) // block_len

    pf = res["pf"]
    w, w_sd = res["w"], res["w_sd"]

    fig = plt.figure(figsize=(14, 17), dpi=150)
    gs = gridspec.GridSpec(7, 1, height_ratios=[1.5, .62, .62, .62, .62, .62, .62],
                           hspace=0.30)
    ax0 = fig.add_subplot(gs[0])
    ax_p = fig.add_subplot(gs[1], sharex=ax0)
    ax_w = fig.add_subplot(gs[2], sharex=ax0)
    ax_a = fig.add_subplot(gs[3], sharex=ax0)
    ax_b = fig.add_subplot(gs[4], sharex=ax0)
    ax_f = fig.add_subplot(gs[5], sharex=ax0)
    ax_r = fig.add_subplot(gs[6], sharex=ax0)

    # --- 1-2: behaviour, from the lab's own function ---
    _, axes = plot_foraging_session(
        choice_history=choice_full, reward_history=reward_full,
        p_reward=p_reward, ax=ax0, smooth_factor=smooth_factor)
    ax_choice = axes[0]
    if len(axes) > 1:      # hand the x-axis to the bottom panel of the stack
        axes[1].set_xlabel("")
        axes[1].tick_params(labelbottom=False)

    s = res["scores"]
    ttl = (f"Subject {subject_id}"
           + (f"  session {session_number}" if session_number else "")
           + f"   held-out NLL: PsyTrack {s['glm']:.3f}  "
             f"MMLPF+$\\varphi$ {s['mmlpf']:.3f}  base {s['base']:.3f} bits/trial\n"
             f"block-CV mask: {int((~res['train_mask']).sum())} of "
             f"{len(res['train_mask'])} valid trials held out in "
             f"{n_drawn} blocks of {block_len}"
             + (f" ({n_blocks} shaded spans; adjacent blocks merge)"
                if n_blocks != n_drawn else "")
             + "   |   "
             f"$\\sigma_\\alpha$={res['sigma_alpha']:.3f}  "
             f"$\\sigma_\\beta$={res['sigma_beta']:.3f}  "
             f"$\\sigma_\\varphi$={res['sigma_phi']:.3f}  "
             f"$\\sigma_{{GLM}}$={res['glm_sigma']:.4f}")
    # plot_foraging_session anchors its own legend above the axes, so the title
    # cannot use set_title's pad without landing on it -- place it in figure
    # coordinates above everything instead
    fig.text(0.5, 0.925, ttl, ha="center", va="bottom", fontweight="bold",
             fontsize=10.5)

    def shade(ax, label=False):
        for k, (a, b) in enumerate(spans):
            ax.axvspan(a, b, color=SHADE, alpha=0.55, lw=0, zorder=0,
                       label=("held out (neither model learns here)"
                              if (label and k == 0) else None))

    # --- 3: both models' one-step-ahead P(right) ---
    shade(ax_p, label=True)
    pr = sb(pf["p_right"]); pr_sd = sb(pf["p_right_sd"])
    vm = ~np.isnan(pr)
    ax_p.plot(trials, sb(res["p_glm"]), color=C_GLM, lw=1.1, label="PsyTrack")
    ax_p.fill_between(trials, pr - pr_sd, pr + pr_sd, where=vm, color=C_MM,
                      alpha=0.25, lw=0)
    ax_p.plot(trials, pr, color=C_MM, lw=1.1, label="MMLPF + $\\varphi$")
    ax_p.scatter(trials[keep][np.asarray(choice_full[keep]) == 1],
                 np.full(int((choice_full[keep] == 1).sum()), 1.06),
                 s=1.5, c="k", marker="|")
    ax_p.scatter(trials[keep][np.asarray(choice_full[keep]) == 0],
                 np.full(int((choice_full[keep] == 0).sum()), -0.06),
                 s=1.5, c="k", marker="|")
    ax_p.set_ylim(-0.12, 1.12)
    ax_p.set_ylabel("P(right)\none step ahead")
    ax_p.legend(frameon=False, fontsize=7, loc="center left",
                bbox_to_anchor=(1.005, 0.5))

    # --- 4: PsyTrack weights (its latents) ---
    shade(ax_w)
    wl = [("bias", 0, "#333333")]
    k = 1
    for lag in range(cv.N_LAGS):
        wl.append((f"rewarded, lag {lag+1}", k, C_GLM)); k += 1
    for lag in range(cv.N_LAGS):
        wl.append((f"unrewarded, lag {lag+1}", k, "#e8a76a")); k += 1
    for lab, j, col in wl:
        if j >= w.shape[1]:
            continue
        m, sd = sb(w[:, j]), sb(w_sd[:, j])
        ls = "-" if "lag 1" in lab or lab == "bias" else "--"
        ax_w.plot(trials, m, color=col, lw=1.0, ls=ls, label=lab)
        if lab == "bias" or "lag 1" in lab:
            ax_w.fill_between(trials, m - sd, m + sd, where=~np.isnan(m),
                              color=col, alpha=0.18, lw=0)
    ax_w.axhline(0, color=GREY, lw=0.8, ls=":", zorder=1)
    ax_w.set_ylabel("PsyTrack\nweights")
    ax_w.legend(frameon=False, fontsize=6, loc="center left",
                bbox_to_anchor=(1.005, 0.5))

    # --- 5-7: MMLPF latents ---
    for ax, key, lab, col in [(ax_a, "alpha", r"MMLPF" "\n" r"$\alpha$", C_A),
                              (ax_b, "beta", r"MMLPF" "\n" r"$\beta$", C_B),
                              (ax_f, "phi", r"MMLPF" "\n" r"$\varphi$", C_P)]:
        shade(ax)
        m, sd = sb(pf[key]), sb(pf[f"{key}_sd"])
        ax.plot(trials, m, color=col, lw=1.1)
        ax.fill_between(trials, m - sd, m + sd, where=~np.isnan(m), color=col,
                        alpha=0.28, lw=0)
        ax.set_ylabel(lab)
    ax_f.axhline(0, color=GREY, lw=0.8, ls=":", zorder=1)

    frac_clip = float(np.mean(pf["beta"] > 95.0))
    if frac_clip > 0.01:
        ax_b.text(0.005, 0.06,
                  f"$\\beta$ > 95 on {frac_clip:.1%} of trials (clip 100): "
                  f"saturating, trace not interpretable",
                  transform=ax_b.transAxes, fontsize=7, color=C_GLM)

    # --- 8: RPE ---
    shade(ax_r)
    ax_r.plot(trials, sb(pf["rpe"]), color=C_R, lw=0.9)
    ax_r.axhline(0, color=GREY, lw=0.8, ls=":", zorder=1)
    ax_r.set_ylabel("MMLPF\nRPE")
    ax_r.set_xlabel("Trial number", fontsize=11)
    ax_r.set_xlim(0, n_full)

    for ax in (ax_p, ax_w, ax_a, ax_b, ax_f):
        ax.tick_params(labelbottom=False)
    for ax in (ax_p, ax_w, ax_a, ax_b, ax_f, ax_r):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"saved {fname}")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", default=None)
    ap.add_argument("--session", type=int, default=2)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0,
                    help="block placement; different seeds shade different "
                         "trials")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--demo", action="store_true",
                    help="simulated session, no database needed")
    args = ap.parse_args()

    if args.demo:
        c, r, raw = simulate_demo()
        sid, sess = "simulated", None
    else:
        if not args.subject:
            ap.error("--subject is required unless --demo is given")
        c, r, raw = load_session(args.subject, args.session)
        sid, sess = args.subject, args.session

    res = fit_both(c, r, block=args.block, test_frac=args.test_frac,
                   seed=args.seed, workers=args.workers)
    for k, v in res["scores"].items():
        print(f"  {k:8s} {v:.4f} bits/trial")
    gap = res["scores"]["glm"] - res["scores"]["mmlpf"]
    print(f"  {'gap':8s} {gap:+.4f}  "
          f"({'MMLPF' if gap > 0 else 'PsyTrack'} better)")

    out = args.out or (f"latent_comparison_{sid}"
                       + (f"_s{sess}" if sess else "") + ".png")
    make_figure(raw, res, subject_id=sid, session_number=sess, fname=out,
                block_len=args.block)


if __name__ == "__main__":
    main()
