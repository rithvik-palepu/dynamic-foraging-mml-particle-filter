"""
MMLPF with a drifting bias vs PsyTrack: latents, weights, and particle clouds.

Produces one multi-panel figure per condition, plus a summary bar chart. Every
figure shows, on the same trial axis:

    1. choice / reward raster with both models' one-step-ahead P(right)
    2. PsyTrack's filtered weights (bias + rewarded/unrewarded history)
    3-6. the MMLPF's FULL PARTICLE DISTRIBUTION for alpha, beta, phi, bias
    7. the MMLPF's reward prediction error

Panels 3-6 are the point of this script. A particle filter's posterior over a
latent is not generally unimodal -- resampling can split or collapse the cloud
-- so a mean +- sd ribbon can describe a distribution that never existed. These
panels draw the marginal histogram at every trial as a density image with the
posterior mean over it, which makes multimodality, collapse, and clipping
visible instead of hidden.

CONDITIONS
----------
    synthetic-biased    Q-agent with a constant side bias in the logit. This is
                        the case the sixth state was added for.
    synthetic-plain     the same agent with bias = 0. Control: shows what the
                        extra state costs when there is nothing to capture.
    real                a mouse from the AIND dynamic-foraging database.

READ THE SYNTHETIC PANELS AS SYNTHETIC. Both synthetic conditions are generated
from the MMLPF's OWN generative model, so recovery there is a CEILING -- it
shows the filter can identify its own parameters, not that the model describes
mice. Only the `real` condition is evidence about behaviour. Figure titles carry
this label; do not crop it off.

SCORING
-------
Interleaved-block CV, identical masks for both models: hyperparameters are fit
on training trials only, held-out blocks are scored one-step-ahead, and neither
model updates its inferred state inside a held-out block. Bits/trial in the
titles are held-out. Lower is better.

USAGE
-----
    # both synthetic conditions -- no database, no cluster
    python bias_showcase.py --conditions synthetic-biased synthetic-plain

    # add a real mouse (queries the public S3 database directly)
    python bias_showcase.py --conditions real --subject 651554 --session 1

    # everything, as sent to collaborators
    python bias_showcase.py --conditions synthetic-biased synthetic-plain real \
        --subject 651554 --session 1

The database is a PUBLIC S3 bucket read by DuckDB -- no AWS credentials needed:
    pip install aind-dynamic-foraging-database

RUNTIME
-------
Each condition runs two volatility fits (3-parameter and 4-parameter differential
evolution). Budget 2-5 minutes per condition at default settings; pass
--fast to cut the search for a quick look at the layout.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import block_cv as bc                                     # noqa: E402
from block_cv import build_glm_regressors        # noqa: E402

N_LAGS = 2
E_PARTICLES = 2000
HIST_BINS = 48
BLOCK = 20
TEST_FRAC = 0.2

C_GLM = "#c8511b"
C_MM = "#17a398"
C_TRUE = "#222222"
GREY = "#8a8a8a"
CMAP = "magma_r"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def simulate_agent(T=600, alpha=0.35, beta=6.0, phi=1.5, bias=0.0, lapse=0.05,
                   block=60, seed=0):
    """Q-learning agent with perseveration and an optional constant side bias.

    Returns choices, rewards, and the programmed reward probabilities so the
    raster panel can show the block structure. The generative model is the
    MMLPF's own, which is what makes recovery here a ceiling.
    """
    rng = np.random.default_rng(seed)
    Q = np.array([0.5, 0.5])
    prev = 0.0
    p = np.array([0.8, 0.1])
    ch, rw, pL, pR = [], [], [], []
    for t in range(T):
        if t and t % block == 0:
            p = p[::-1].copy()
        z = beta * (Q[1] - Q[0]) + phi * prev + bias
        pr = (1 - lapse) * expit(z) + lapse / 2
        c = int(rng.random() < pr)
        r = int(rng.random() < p[c])
        Q[c] += alpha * (r - Q[c])
        prev = 1.0 if c == 1 else -1.0
        ch.append(c)
        rw.append(r)
        pL.append(p[0])
        pR.append(p[1])
    return (np.array(ch), np.array(rw),
            np.array(pL), np.array(pR))


def load_real_session(subject_id, session_number=1, query=None,
                      min_trials=200):
    """One real session from the public AIND parquet database.

    Session numbering follows hpc_block_cv.prepare: sessions passing the cohort
    query, ordered by date, 1-indexed. Ignored trials (animal_response == 2) are
    dropped for fitting, matching every other analysis in this project.
    """
    import aind_dynamic_foraging_database as db
    import mmlpf_vs_psytrack_cv as cv

    q = query or cv.COHORT_QUERY
    sel = db.select_sessions(q, subjects=[str(subject_id)])
    if not len(sel):
        raise ValueError(
            f"subject {subject_id} has no session passing the cohort query:\n"
            f"  {q}\n"
            f"Pick another subject, or pass --query to relax it.")
    sel = sel.sort_values("session_date")
    if session_number > len(sel):
        raise ValueError(
            f"subject {subject_id} has {len(sel)} qualifying sessions; "
            f"cannot reach session {session_number}")

    cols = ["animal_response", "earned_reward",
            "reward_probabilityL", "reward_probabilityR"]
    try:
        trials = db.fetch_trials(sel, columns=cols)
    except Exception:
        trials = db.fetch_trials(sel, columns=cols[:2])

    groups = list(trials.groupby(["session_date", "session_id"], sort=True))
    raw = groups[session_number - 1][1].reset_index(drop=True)
    valid = raw[raw["animal_response"] != 2]
    if len(valid) < min_trials:
        raise ValueError(
            f"session {session_number} has only {len(valid)} valid trials "
            f"(< {min_trials})")
    pL = (valid["reward_probabilityL"].values
          if "reward_probabilityL" in valid else None)
    pR = (valid["reward_probabilityR"].values
          if "reward_probabilityR" in valid else None)
    n_ign = int((raw["animal_response"] == 2).sum())
    print(f"  subject {subject_id} session {session_number}: {len(raw)} trials, "
          f"{n_ign} ignored ({n_ign / len(raw):.1%}), {len(valid)} valid")
    return (valid["animal_response"].astype(int).values,
            valid["earned_reward"].astype(int).values, pL, pR)


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------
def bits(p, y, mask):
    p = np.clip(np.asarray(p)[mask], 1e-9, 1 - 1e-9)
    y = np.asarray(y)[mask]
    return float(-np.mean(y * np.log2(p) + (1 - y) * np.log2(1 - p)))


def fit_condition(choices, rewards, seed=0, fast=False):
    """Fit PsyTrack and the MMLPF (with drifting bias) on one session.

    Both models see the same interleaved-block mask, fit hyperparameters on
    training trials only, and are scored one-step-ahead on the held-out blocks.
    The 5-state MMLPF is fit too, so the figure can report what the sixth state
    bought on this session.
    """
    T = len(choices)
    train_mask = bc.make_block_mask(T, block=BLOCK, test_frac=TEST_FRAC,
                                    seed=seed)
    test_mask = ~train_mask
    maxiter = 5 if fast else bc.DE_MAXITER
    popsize = 4 if fast else bc.DE_POPSIZE
    nparts = 600 if fast else E_PARTICLES

    # --- PsyTrack ---
    rew, unrew = build_glm_regressors(choices, rewards,
                                      np.zeros(T, dtype=int), n_lags=N_LAGS)
    X = np.column_stack([np.ones(T), rew, unrew])
    glm_sigma = bc.fit_glm_sigma(X, choices, train_mask)
    glm_p, _, w_pred, P_pred = bc.glm_masked(
        X, choices, glm_sigma, train_mask, return_weights=True)

    # --- MMLPF, 5 state (reference) and 6 state (with drifting bias) ---
    s5 = bc.fit_mmlpf_masked(choices, rewards, train_mask,
                             maxiter=maxiter, popsize=popsize)
    _, p5 = bc.mmlpf_masked(*s5, choices, rewards, train_mask,
                            num_particles=nparts, seed=seed)
    s6 = bc.fit_mmlpf_masked(choices, rewards, train_mask, with_bias=True,
                             maxiter=maxiter, popsize=popsize)
    o6 = bc.mmlpf_masked(*s6[:3], choices, rewards, train_mask,
                         num_particles=nparts, seed=seed, sigma_bias=s6[3],
                         collect=True, hist_bins=HIST_BINS)

    base_p = np.full(T, float(choices[train_mask].mean()))
    return dict(
        train_mask=train_mask, test_mask=test_mask,
        glm_p=glm_p, w_pred=w_pred, P_pred=P_pred, glm_sigma=glm_sigma,
        mm=o6, sigmas5=s5, sigmas6=s6,
        base_bits=bits(base_p, choices, test_mask),
        glm_bits=bits(glm_p, choices, test_mask),
        mmlpf5_bits=bits(p5, choices, test_mask),
        mmlpf6_bits=bits(o6["p_right"], choices, test_mask),
        n_particles=nparts,
    )


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def _heldout_spans(train_mask):
    """[(start, stop_inclusive), ...] runs of held-out trials."""
    held = ~train_mask
    spans, t, T = [], 0, len(held)
    while t < T:
        if held[t]:
            j = t
            while j < T and held[j]:
                j += 1
            spans.append((t, j - 1))
            t = j
        else:
            t += 1
    return spans


def _shade_heldout(ax, train_mask, ribbon=False):
    """Mark trials where neither model learns.

    ribbon=True draws an opaque strip along the top of the axes instead of a
    background span. The density panels are drawn with pcolormesh, which paints
    over any background axvspan -- so on those panels a background shade is
    invisible and the held-out blocks would silently vanish from the figure.
    """
    for lo, hi in _heldout_spans(train_mask):
        if ribbon:
            ax.axvspan(lo, hi + 1, ymin=0.955, ymax=1.0, color="0.25",
                       lw=0, zorder=5)
        else:
            ax.axvspan(lo, hi, color="0.85", lw=0, zorder=0)


def _density_panel(ax, hist, edges, mean_trace, label, truth=None,
                   log_beta=False, clip_at=None):
    """One latent's full particle marginal over trials, mean drawn on top."""
    H = hist / np.maximum(hist.sum(axis=1, keepdims=True), 1)
    T, nb = H.shape
    ax.pcolormesh(np.arange(T + 1), edges, H.T, cmap=CMAP, shading="flat",
                  rasterized=True)
    y = np.log10(np.clip(mean_trace, 1e-2, 1e2)) if log_beta else mean_trace
    ax.plot(np.arange(T), y, color=C_MM, lw=1.1, label="posterior mean")
    if truth is not None:
        ty = np.log10(truth) if log_beta else truth
        ax.axhline(ty, color=C_TRUE, ls="--", lw=1.1, label="truth")
    if clip_at is not None:
        ax.axhline(clip_at, color=C_GLM, ls=":", lw=1.0)
        ax.text(0.995, clip_at, " clip", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=5.6, color=C_GLM)
    ax.set_ylabel(label)
    ax.set_ylim(edges[0], edges[-1])


def make_figure(choices, rewards, res, title, fname, pL=None, pR=None,
                truth=None, smooth=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    truth = truth or {}
    T = len(choices)
    t = np.arange(T)
    mm = res["mm"]
    tm = res["train_mask"]

    fig, axes = plt.subplots(7, 1, figsize=(11.5, 15.0), sharex=True,
                             gridspec_kw={"height_ratios":
                                          [1.5, 1.3, 1.0, 1.0, 1.0, 1.0, 0.9]})
    ax_r, ax_w, ax_a, ax_b, ax_p, ax_bi, ax_e = axes

    # --- 1. raster + both models' predictions ---
    k = np.ones(smooth) / smooth
    ax_r.plot(t, np.convolve(choices, k, mode="same"), color="0.25", lw=1.0,
              label=f"choice (smoothed, {smooth})")
    if pL is not None and pR is not None:
        with np.errstate(invalid="ignore"):
            frac = np.asarray(pR, float) / np.maximum(
                np.asarray(pL, float) + np.asarray(pR, float), 1e-9)
        ax_r.plot(t, frac, color="#b8a400", lw=1.0, label="programmed P(right)")
    ax_r.plot(t, res["glm_p"], color=C_GLM, lw=1.0, alpha=0.9, label="PsyTrack")
    ax_r.plot(t, mm["p_right"], color=C_MM, lw=1.0, alpha=0.9,
              label=r"MMLPF + bias")
    # Choice raster: RIGHT choices above the axis, LEFT choices below, so the
    # vertical position matches the P(right) axis. Tall tick = rewarded, short
    # tick = unrewarded. An earlier version drew a single unlabelled row of
    # rewarded trials, which reads as "right choices with gaps" and hides the
    # left choices entirely -- both sides must be visible to see a side bias.
    ch = np.asarray(choices).astype(int)
    rw = np.asarray(rewards).astype(int)
    for side, y0, sgn in ((1, 1.05, +1), (0, -0.05, -1)):
        for got, h, col in ((1, 0.14, "0.15"), (0, 0.07, "0.62")):
            idx = np.where((ch == side) & (rw == got))[0]
            ax_r.vlines(idx, y0, y0 + sgn * h, color=col, lw=0.45)
    ax_r.text(-0.008, 1.12, "right", transform=ax_r.get_yaxis_transform(),
              ha="right", va="center", fontsize=6.0, color="0.15")
    ax_r.text(-0.008, -0.12, "left", transform=ax_r.get_yaxis_transform(),
              ha="right", va="center", fontsize=6.0, color="0.15")
    ax_r.text(0.998, 0.995, "tall = rewarded, short = unrewarded",
              transform=ax_r.transAxes, ha="right", va="top", fontsize=5.8,
              color="0.35")
    # headroom above the right-choice raster so the legend sits on empty space
    # instead of over the P(right) traces
    ax_r.set_ylim(-0.22, 1.52)
    ax_r.set_yticks([0, 0.5, 1.0])
    ax_r.set_ylabel("P(right)")
    ax_r.legend(frameon=False, fontsize=6.2, ncol=4, loc="upper left")
    ax_r.set_title(title, loc="left", fontsize=9)

    # --- 2. PsyTrack weights ---
    names = ["bias"] + [f"rewarded, lag {i+1}" for i in range(N_LAGS)] \
                     + [f"unrewarded, lag {i+1}" for i in range(N_LAGS)]
    shades = ["#222222", "#c8511b", "#e8834d", "#7b9ec9", "#a8c2dc"]
    W = np.asarray(res["w_pred"])
    P = np.asarray(res["P_pred"])
    for j, (nm, col) in enumerate(zip(names, shades)):
        ax_w.plot(t, W[:, j], color=col, lw=1.0, label=nm)
        if P.ndim == 3:
            sd = np.sqrt(np.clip(P[:, j, j], 0, None))
        else:
            sd = np.sqrt(np.clip(P[:, j], 0, None))
        ax_w.fill_between(t, W[:, j] - sd, W[:, j] + sd, color=col, alpha=0.16,
                          lw=0)
    ax_w.axhline(0, color="k", lw=0.7, ls=":")
    ax_w.set_ylabel("PsyTrack\nweights")
    ax_w.legend(frameon=False, fontsize=6.0, ncol=5, loc="upper left")

    # --- 3-6. particle distributions ---
    _density_panel(ax_a, mm["alpha_hist"], mm["alpha_edges"], mm["alpha"],
                   r"MMLPF $\alpha$", truth.get("alpha"))
    _density_panel(ax_b, mm["beta_hist"], mm["beta_edges"], mm["beta"],
                   r"MMLPF $\log_{10}\beta$", truth.get("beta"),
                   log_beta=True, clip_at=2.0)
    _density_panel(ax_p, mm["phi_hist"], mm["phi_edges"], mm["phi"],
                   r"MMLPF $\varphi$", truth.get("phi"))
    _density_panel(ax_bi, mm["bias_hist"], mm["bias_edges"], mm["bias"],
                   r"MMLPF bias $b$", truth.get("bias"))
    for ax in (ax_a, ax_b, ax_p, ax_bi):
        ax.legend(frameon=False, fontsize=6.0, loc="upper right")

    # --- 7. RPE ---
    ax_e.plot(t, mm["rpe"], color="#3b6ea5", lw=0.7)
    ax_e.axhline(0, color="k", lw=0.7, ls=":")
    ax_e.set_ylabel("MMLPF\nRPE")
    ax_e.set_xlabel("Trial")

    dens = {id(a) for a in (ax_a, ax_b, ax_p, ax_bi)}
    for ax in axes:
        _shade_heldout(ax, tm, ribbon=id(ax) in dens)
        ax.set_xlim(0, T)
    ax_a.text(0.005, 0.985, "dark strip = held out (neither model learns)",
              transform=ax_a.transAxes, va="top", fontsize=5.8, color="0.15")

    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    print(f"  saved {fname}")
    return fig


def summary_figure(rows, fname="bias_showcase_summary.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(1.9 + 2.1 * len(d), 4.2))
    x = np.arange(len(d))
    w = 0.26
    ax.bar(x - w, d["glm_bits"], w, color=C_GLM, edgecolor="white", lw=0.7,
           label="PsyTrack")
    ax.bar(x, d["mmlpf5_bits"], w, color="#9a9a9a", edgecolor="white", lw=0.7,
           label="MMLPF (no bias)")
    ax.bar(x + w, d["mmlpf6_bits"], w, color=C_MM, edgecolor="white", lw=0.7,
           label="MMLPF + drifting bias")
    for i, r in enumerate(d.itertuples()):
        for dx, v in ((-w, r.glm_bits), (0, r.mmlpf5_bits), (w, r.mmlpf6_bits)):
            ax.text(i + dx, v + 0.006, f"{v:.3f}", ha="center", fontsize=6.2)
        ax.axhline(r.base_bits, xmin=(i + 0.08) / len(d),
                   xmax=(i + 0.92) / len(d), color=GREY, ls="--", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(d["condition"], fontsize=7.5)
    ax.set_ylabel("Held-out bits/trial (lower is better)")
    ax.set_title("One session per condition; dashed line = base rate.\n"
                 "Synthetic conditions are generated from the MMLPF's own "
                 "model and are a ceiling, not evidence about mice.",
                 loc="left", fontsize=8)
    ax.legend(frameon=False, fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"saved {fname}")
    return fig


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", nargs="+",
                    default=["synthetic-biased", "synthetic-plain"],
                    choices=["synthetic-biased", "synthetic-plain", "real"])
    ap.add_argument("--subject", default="651554",
                    help="subject id for the `real` condition")
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--query", default=None,
                    help="override the cohort query for the real session")
    ap.add_argument("--trials", type=int, default=600,
                    help="length of the synthetic sessions")
    ap.add_argument("--bias", type=float, default=2.0,
                    help="true side bias for synthetic-biased")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", action="store_true",
                    help="short hyperparameter search: for checking the "
                         "layout, not for numbers worth reporting")
    ap.add_argument("--out-prefix", default="bias_showcase")
    args = ap.parse_args()

    rows = []
    for cond in args.conditions:
        print(f"\n=== {cond} ===", flush=True)
        truth = {}
        pL = pR = None
        if cond == "synthetic-biased":
            truth = dict(alpha=0.35, beta=6.0, phi=1.5, bias=args.bias)
            c, r, pL, pR = simulate_agent(T=args.trials, bias=args.bias,
                                          seed=args.seed)
            label = (f"SYNTHETIC side-biased Q-agent  (true bias "
                     f"{args.bias:+.1f})")
        elif cond == "synthetic-plain":
            truth = dict(alpha=0.35, beta=6.0, phi=1.5, bias=0.0)
            c, r, pL, pR = simulate_agent(T=args.trials, bias=0.0,
                                          seed=args.seed)
            label = "SYNTHETIC plain Q-agent  (true bias 0)"
        else:
            c, r, pL, pR = load_real_session(args.subject, args.session,
                                             query=args.query)
            label = f"REAL mouse {args.subject}, session {args.session}"

        res = fit_condition(c, r, seed=args.seed, fast=args.fast)
        gap = res["glm_bits"] - res["mmlpf6_bits"]
        # quantify "biased": overall side preference, and how strongly the
        # subject tracks which side the schedule currently favours
        p_right = float(c.mean())
        if pL is not None and pR is not None:
            br = np.asarray(pR, float) > np.asarray(pL, float)
            follows = float(c[br].mean() - c[~br].mean()) if br.any() and (~br).any() else np.nan
        else:
            follows = np.nan
        bias_line = (f"P(right) = {p_right:.3f}"
                     + (f",  schedule-following = {follows:+.3f}"
                        if np.isfinite(follows) else ""))
        title = (
            f"{label}\n"
            f"held-out bits/trial:  base {res['base_bits']:.3f}   "
            f"PsyTrack {res['glm_bits']:.3f}   "
            f"MMLPF {res['mmlpf5_bits']:.3f}   "
            f"MMLPF+bias {res['mmlpf6_bits']:.3f}   "
            f"(PsyTrack - MMLPF+bias = {gap:+.3f})\n"
            r"fitted $\sigma$: " +
            f"a {res['sigmas6'][0]:.3f}  b {res['sigmas6'][1]:.3f}  "
            f"phi {res['sigmas6'][2]:.3f}  bias {res['sigmas6'][3]:.3f}   |   "
            f"{res['n_particles']} particles, {int((~res['train_mask']).sum())}"
            f" of {len(c)} trials held out\n"
            f"side bias: {bias_line}")
        fname = f"{args.out_prefix}_{cond}.png"
        make_figure(c, r, res, title, fname, pL=pL, pR=pR, truth=truth)
        print(f"  base {res['base_bits']:.4f}  PsyTrack {res['glm_bits']:.4f}  "
              f"MMLPF {res['mmlpf5_bits']:.4f}  "
              f"MMLPF+bias {res['mmlpf6_bits']:.4f}")
        # a volatility pinned near its search bound means the volatility fit wanted to
        # go further: the fitted value is a floor, and the latent's trajectory
        # is not interpretable at that setting
        names = ("sigma_alpha", "sigma_beta", "sigma_phi", "sigma_bias")
        hi = bc.DE_BOUNDS_BIAS[0][1]
        at_bound = [f"{n}={v:.3f}" for n, v in zip(names, res["sigmas6"])
                    if v > 0.95 * hi]
        if at_bound:
            print(f"  ** {', '.join(at_bound)} sits within 5% of the search "
                  f"bound {hi}: treat that trajectory as uninterpretable and "
                  f"widen DE_BOUNDS_BIAS before reporting it **")
        if args.fast:
            print("  ** --fast: short hyperparameter search. Layout only; "
                  "these bits/trial are not reportable numbers **")
        rows.append(dict(condition=cond.replace("-", "\n"),
                         base_bits=res["base_bits"],
                         glm_bits=res["glm_bits"],
                         mmlpf5_bits=res["mmlpf5_bits"],
                         mmlpf6_bits=res["mmlpf6_bits"],
                         sigma_bias=res["sigmas6"][3],
                         p_right=p_right, schedule_following=follows))

    if len(rows) > 1:
        summary_figure(rows, fname=f"{args.out_prefix}_summary.png")
    pd.DataFrame(rows).to_csv(f"{args.out_prefix}_scores.csv", index=False)
    print(f"\nwrote {args.out_prefix}_scores.csv")


if __name__ == "__main__":
    main()
