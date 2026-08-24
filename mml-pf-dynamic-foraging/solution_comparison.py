"""
Do the two models find the same solution?

Correlates the MMLPF's best-fit parameters against PsyTrack's fitted weights
across every session, then draws example fittings on the lab's standard session
format (plot_foraging_session, via latent_comparison.make_figure).

WHAT IS COMPARED, AND WHY
-------------------------
The two models are not reparameterisations of each other, so there is no
one-to-one map. What CAN be compared are quantities that play the same role in
each model's decision variable:

    PsyTrack:  z_t = w_bias + sum_k w_rew[k]*rew_hist[k] + sum_k w_unrew[k]*unrew_hist[k]
    MMLPF:     z_t = beta_t*(Q_R - Q_L) + phi_t*c_{t-1}

Aligned pairings, each with the reason it should hold:

  value term        beta_t*(Q_R-Q_L)   vs  the reward-history part of PsyTrack's z
                    Both are "how strongly does accumulated reward evidence push
                    the choice". Compared TRIAL BY TRIAL within a session.

  perseveration     phi_t*c_{t-1}      vs  PsyTrack's bias weight w_bias
                    PsyTrack has no explicit choice-history regressor in this
                    design (only rewarded/unrewarded outcome history), so any
                    choice repetition it captures has to load on the bias, which
                    drifts. Compared trial by trial.

  total drive       |z| under each model
                    Sanity check: if both models are describing the same animal,
                    their decision variables should track each other even where
                    the decomposition differs.

  reward weight     sum_k |w_rew[k]|   vs  session median beta
                    Both are overall reward sensitivity. Compared ACROSS
                    SESSIONS (one number each).

  kernel decay      w_rew[1]/w_rew[2]  vs  session median alpha
                    A fast learner discounts older rewards more steeply, so the
                    lag-1/lag-2 ratio should rise with alpha. Across sessions.

INTERPRETING A LOW CORRELATION
------------------------------
A weak trial-level correlation is not automatically a discrepancy: the MMLPF's
latents are only identified as session-level summaries (its trial-by-trial
alpha/beta/phi series track known ground truth at |r| < 0.1 in simulation), so
the value and perseveration TERMS -- which are driven by Q and by the observed
last choice -- are the meaningful trial-level quantities, not the parameters
themselves. The script therefore reports the terms trial-wise and the parameters
across sessions, and never correlates a raw parameter series against a weight
series.

Every trial-level correlation is computed on TRAINING trials only, so held-out
blocks (where both models are frozen and their traces are propagated rather
than fitted) cannot manufacture agreement.

USAGE
-----
    # all 1000 sessions, correlations + summary figure
    python solution_comparison.py --cache-dir ./cache --cv-csv cohort_block_cv_hpc.csv \\
        --workers 8

    # the eight example fittings named on the command line below
    python solution_comparison.py --examples --cache-dir ./cache \\
        --cv-csv cohort_block_cv_hpc.csv

    # re-analyse without refitting
    python solution_comparison.py --summarize
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import block_cv as bc
import mmlpf_vs_psytrack_cv as cv
import latent_comparison as lc
from matched_validation import build_glm_regressors

E_PARTICLES = 1500
BLOCK = 20
TEST_FRAC = 0.2
MIN_TRIALS = 200

# X = [ones, rew_1..rew_K, unrew_1..unrew_K]; K = cv.N_LAGS
K = cv.N_LAGS
I_BIAS = 0
I_REW = slice(1, 1 + K)
I_UNREW = slice(1 + K, 1 + 2 * K)

RESULT_COLUMNS = [
    "subject_id", "session_number", "n_trials",
    # trial-level, training trials only
    "r_value_term", "r_persev_term", "r_total_drive",
    # session-level scalars
    "beta_med", "alpha_med", "phi_med",
    "sum_abs_w_rew", "sum_abs_w_unrew", "w_bias_med", "w_bias_sd",
    "kernel_ratio", "glm_sigma",
    "sigma_alpha", "sigma_beta", "sigma_phi",
    "base_bits", "glm_bits", "mmlpf_bits", "gap_bits",
    "seconds", "error",
]


def decompose(res, choices):
    """Split each model's decision variable into comparable pieces.

    Returns dict of per-trial arrays. `value_glm` deliberately EXCLUDES the bias
    column: the bias is what absorbs choice repetition in this design, so it
    belongs with the perseveration comparison, not the value one.
    """
    X, w = res["X"], res["w"]           # w is (T, K_total) filtered weights
    pf = res["pf"]
    T = len(choices)

    value_glm = (X[:, I_REW] * w[:, I_REW]).sum(1) + \
                (X[:, I_UNREW] * w[:, I_UNREW]).sum(1)
    persev_glm = w[:, I_BIAS] * X[:, I_BIAS]          # X[:,0] is 1 by construction

    q_diff = pf["q_right"] - pf["q_left"]
    value_mm = pf["beta"] * q_diff
    prev_c = np.concatenate([[0.0], np.where(choices[:-1] == 1, 1.0, -1.0)])
    persev_mm = pf["phi"] * prev_c

    return dict(value_glm=value_glm, persev_glm=persev_glm,
                drive_glm=value_glm + persev_glm,
                value_mm=value_mm, persev_mm=persev_mm,
                drive_mm=value_mm + persev_mm)


def _safe_r(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _one_session(args):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    import time
    sid, sess_no, ch, rw, sig_cached, block, test_frac, n_part, seed = args
    row = {c: np.nan for c in RESULT_COLUMNS}
    row.update(subject_id=sid, session_number=sess_no, n_trials=len(ch), error="")
    t0 = time.time()
    try:
        T = len(ch)
        train_mask = bc.make_block_mask(T, block=block, test_frac=test_frac,
                                        seed=seed)
        rew_h, unrew_h = build_glm_regressors(ch, rw, np.zeros(T, dtype=int),
                                              n_lags=K)
        X = np.column_stack([np.ones(T), rew_h, unrew_h])
        gsig = bc.fit_glm_sigma(X, ch, train_mask, workers=1)
        p_glm, _, w_pred, P_pred = bc.glm_masked(X, ch, gsig, train_mask,
                                                 return_weights=True)
        if sig_cached is None:
            sa, sb, sp = bc.fit_mmlpf_masked(ch, rw, train_mask, workers=1)
        else:
            sa, sb, sp = sig_cached
        pf = bc.mmlpf_masked(sa, sb, sp, ch, rw, train_mask,
                             num_particles=n_part, collect=True)

        res = dict(X=X, w=w_pred, pf=pf)
        dec = decompose(res, ch)

        # trial-level correlations on TRAINING trials only
        tm = train_mask
        row["r_value_term"] = _safe_r(dec["value_glm"][tm], dec["value_mm"][tm])
        row["r_persev_term"] = _safe_r(dec["persev_glm"][tm], dec["persev_mm"][tm])
        row["r_total_drive"] = _safe_r(dec["drive_glm"][tm], dec["drive_mm"][tm])

        # session-level scalars
        w_rew = w_pred[:, I_REW]
        row["beta_med"] = float(np.median(pf["beta"]))
        row["alpha_med"] = float(np.median(pf["alpha"]))
        row["phi_med"] = float(np.median(pf["phi"]))
        row["sum_abs_w_rew"] = float(np.median(np.abs(w_rew).sum(1)))
        row["sum_abs_w_unrew"] = float(
            np.median(np.abs(w_pred[:, I_UNREW]).sum(1)))
        row["w_bias_med"] = float(np.median(w_pred[:, I_BIAS]))
        row["w_bias_sd"] = float(np.std(w_pred[:, I_BIAS]))
        if K >= 2:
            l1 = np.median(w_rew[:, 0]); l2 = np.median(w_rew[:, 1])
            row["kernel_ratio"] = float(l1 / l2) if abs(l2) > 1e-6 else np.nan
        row["glm_sigma"] = float(gsig) if np.isscalar(gsig) else float(np.mean(gsig))
        row.update(sigma_alpha=sa, sigma_beta=sb, sigma_phi=sp)

        test_mask = ~train_mask
        base = np.full(T, ch[train_mask].mean())
        row["base_bits"] = bc.bits(base, ch, test_mask)
        row["glm_bits"] = bc.bits(p_glm, ch, test_mask)
        row["mmlpf_bits"] = bc.bits(pf["p_right"], ch, test_mask)
        row["gap_bits"] = row["glm_bits"] - row["mmlpf_bits"]
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    row["seconds"] = round(time.time() - t0, 1)
    return row


def load_sigmas(cv_csv):
    d = pd.read_csv(cv_csv)
    d = d[(d["error"].isna()) | (d["error"] == "")]
    d["subject_id"] = d["subject_id"].astype(str)
    g = (d.groupby(["subject_id", "session_number"])
         [["sigma_alpha", "sigma_beta", "sigma_phi"]].median())
    return {k: tuple(v) for k, v in g.iterrows()}


def run(cache_dir, cv_csv=None, workers=1, n_particles=E_PARTICLES,
        block=BLOCK, test_frac=TEST_FRAC, seed=0, limit=None,
        out_csv="solution_comparison.csv"):
    import hpc_block_cv as h

    mf_path = os.path.join(cache_dir, "manifest.csv")
    if not os.path.exists(mf_path):
        raise FileNotFoundError(
            f"{mf_path} not found -- run `hpc_block_cv.py prepare` first.")
    manifest = pd.read_csv(mf_path)
    manifest["subject_id"] = manifest["subject_id"].astype(str)
    subjects = list(manifest["subject_id"])[:limit] if limit \
        else list(manifest["subject_id"])

    sigmas = load_sigmas(cv_csv) if cv_csv else {}
    if cv_csv:
        print(f"reusing MMLPF volatilities for {len(sigmas)} sessions from "
              f"{cv_csv}; PsyTrack's sigma is refit per session (it is cheap)",
              flush=True)
    else:
        print("no --cv-csv: refitting MMLPF volatilities per session (slow)",
              flush=True)

    jobs = []
    for sid in subjects:
        for (sess_no, ch, rw, _nr, _ni) in h.load_cached_subject(cache_dir, sid):
            if len(ch) < MIN_TRIALS:
                continue
            jobs.append((sid, sess_no, ch, rw, sigmas.get((sid, sess_no)),
                         block, test_frac, n_particles, seed))
    print(f"{len(subjects)} subjects, {len(jobs)} sessions", flush=True)

    rows = []
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for n, row in enumerate(pool.imap_unordered(_one_session, jobs), 1):
                rows.append(row)
                if n % 50 == 0 or n == len(jobs):
                    print(f"  [{n}/{len(jobs)}]", flush=True)
    else:
        for n, job in enumerate(jobs, 1):
            rows.append(_one_session(job))
            if n % 50 == 0 or n == len(jobs):
                print(f"  [{n}/{len(jobs)}]", flush=True)

    df = pd.DataFrame(rows, columns=RESULT_COLUMNS).sort_values(
        ["subject_id", "session_number"])
    df.to_csv(out_csv, index=False)
    n_err = int((df["error"].notna() & (df["error"] != "")).sum())
    print(f"wrote {out_csv} ({len(df)} sessions, {n_err} errors)")
    return df


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def _r_ci(r, n):
    if not np.isfinite(r) or n < 5:
        return (np.nan, np.nan)
    z, se = np.arctanh(np.clip(r, -0.999, 0.999)), 1 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


CROSS_SESSION_PAIRS = [
    ("sum_abs_w_rew", "beta_med", "+",
     "reward-weight magnitude vs beta"),
    ("kernel_ratio", "alpha_med", "+",
     "reward-kernel lag1/lag2 vs alpha"),
    ("w_bias_sd", "phi_med", "+",
     "bias-weight drift vs phi"),
]


def analyse(df):
    ok = df[(df["error"].isna()) | (df["error"] == "")].copy()
    ok["subject_id"] = ok["subject_id"].astype(str)
    print(f"\n{'='*78}")
    print(f"SOLUTION COMPARISON  ({ok['subject_id'].nunique()} subjects, "
          f"{len(ok)} sessions)")
    print("="*78)

    print("\n--- trial-level agreement within a session (training trials only) ---")
    print(f"{'quantity':>34} {'median r':>10} {'IQR':>20} {'>0.5':>7}")
    for c, lab in [("r_value_term", "value term: beta*(Q_R-Q_L) vs w_rew.x"),
                   ("r_persev_term", "perseveration: phi*c_prev vs w_bias"),
                   ("r_total_drive", "total decision variable")]:
        v = ok[c].dropna()
        if not len(v):
            continue
        print(f"{lab:>34} {v.median():10.3f} "
              f"{f'[{v.quantile(.25):+.2f}, {v.quantile(.75):+.2f}]':>20} "
              f"{(v > 0.5).mean():7.0%}")

    print("\n--- across-session agreement (one number per session) ---")
    print(f"{'pairing':>34} {'expect':>7} {'Spearman':>9} {'95% CI':>18} {'p':>9}")
    rows = []
    for a, b, sign, lab in CROSS_SESSION_PAIRS:
        s = ok[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < 10:
            continue
        r_, p_ = stats.spearmanr(s[a], s[b])
        lo, hi = _r_ci(r_, len(s))
        supports = (r_ > 0) if sign == "+" else (r_ < 0)
        flag = ""
        if not (lo <= 0 <= hi):
            flag = " *" if supports else " * WRONG SIGN"
        print(f"{lab:>34} {sign:>7} {r_:9.3f} "
              f"{f'[{lo:+.2f}, {hi:+.2f}]':>18} {p_:9.2g}{flag}")
        rows.append(dict(pair=lab, a=a, b=b, expected=sign, rho=r_,
                         ci_lo=lo, ci_hi=hi, p=p_, n=len(s),
                         supports=bool(supports and not (lo <= 0 <= hi))))

    print("\n* = CI excludes zero. A pairing significant in the WRONG direction")
    print("  means the two models disagree about that quantity, which is a")
    print("  finding, not noise.")

    # does agreement predict the performance gap?
    print("\n--- does trial-level agreement track the performance gap? ---")
    for c in ("r_value_term", "r_persev_term", "r_total_drive"):
        s = ok[[c, "gap_bits"]].dropna()
        if len(s) < 10:
            continue
        r_, p_ = stats.spearmanr(s[c], s["gap_bits"])
        print(f"  {c:>16} vs gap: rho = {r_:+.2f}  p = {p_:.2g}")
    print("  (positive = the models agree most where the MMLPF does best)")
    return pd.DataFrame(rows), ok


def plot_summary(res, ok, fname="solution_comparison.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_A, C_B, C_C, GREY = "#17a398", "#c8511b", "#9467bd", "#8a8a8a"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    axA, axB, axC = axes

    # A: distributions of the three trial-level correlations
    data, labs, cols = [], [], []
    for c, lab, col in [("r_value_term", "value\nterm", C_A),
                        ("r_persev_term", "persev.\nterm", C_B),
                        ("r_total_drive", "total\ndrive", C_C)]:
        v = ok[c].dropna().values
        if len(v):
            data.append(v); labs.append(lab); cols.append(col)
    if data:
        parts = axA.violinplot(data, showmedians=True, widths=0.75)
        for pc, col in zip(parts["bodies"], cols):
            pc.set_facecolor(col); pc.set_alpha(0.6); pc.set_edgecolor("white")
        for k in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[k].set_color("#333333"); parts[k].set_linewidth(0.9)
    axA.axhline(0, color="k", lw=1.0)
    axA.set_xticks(range(1, len(labs) + 1)); axA.set_xticklabels(labs)
    axA.set_ylabel("within-session correlation")
    axA.set_ylim(-1.05, 1.05)
    axA.set_title("Trial-level agreement between the\ntwo decision variables",
                  loc="left")
    axA.text(0.98, 0.03, f"n = {len(ok)} sessions", transform=axA.transAxes,
             ha="right", fontsize=6.5, color=GREY)

    # B: across-session pairings with CIs
    if len(res):
        y = np.arange(len(res))[::-1]
        bcols = [C_A if s else C_B for s in res["supports"]]
        axB.barh(y, res["rho"].values, color=bcols, height=0.55,
                 edgecolor="white", lw=0.8)
        axB.errorbar(res["rho"].values, y,
                     xerr=[res["rho"] - res["ci_lo"], res["ci_hi"] - res["rho"]],
                     fmt="none", ecolor="#333333", elinewidth=1.0, capsize=3)
        axB.axvline(0, color="k", lw=1.1)
        axB.set_yticks(y)
        axB.set_yticklabels([p.replace(" vs ", "\nvs ") for p in res["pair"]],
                            fontsize=6.5)
        axB.set_xlabel("Spearman rho across sessions")
        axB.set_title("Do the models agree about\nsession-level parameters?",
                      loc="left")
        axB.text(0.98, 0.02, "CI crossing 0 = no evidence",
                 transform=axB.transAxes, ha="right", fontsize=6.2, color=GREY)

    # C: agreement vs gap
    v = ok[["r_total_drive", "gap_bits"]].dropna()
    if len(v):
        axC.scatter(v["r_total_drive"], v["gap_bits"], s=6, color=C_C,
                    alpha=0.35, lw=0)
        rho, p_ = stats.spearmanr(v["r_total_drive"], v["gap_bits"])
        lo_, hi_ = _r_ci(rho, len(v))
        axC.axhline(0, color="k", lw=1.0)
        axC.set_xlabel("agreement of the total decision variable")
        axC.set_ylabel("PsyTrack minus MMLPF (bits/trial)")
        # title states the measured direction rather than asserting one; a
        # hardcoded narrative here would contradict the data whenever the
        # correlation comes out the other way
        if lo_ <= 0 <= hi_:
            head = "Agreement does not predict\nthe performance gap"
        elif rho > 0:
            head = "The gap closes where the\nmodels agree"
        else:
            head = "The gap WIDENS where the\nmodels agree"
        axC.set_title(rf"{head}  ($\rho = {rho:+.2f}$)", loc="left")
        axC.text(0.03, 0.03, "above 0 = MMLPF better", transform=axC.transAxes,
                 fontsize=6.3, color=GREY)

    for ax, L in zip(axes, "abc"):
        ax.text(-0.15, 1.02, L, transform=ax.transAxes, fontsize=12,
                fontweight="bold", va="bottom")
    fig.tight_layout(w_pad=2.5)
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"saved {fname}")
    return fig


# ---------------------------------------------------------------------------
# example fittings, on the lab's session format
# ---------------------------------------------------------------------------
EXAMPLE_CASES = [
    ("psytrack_win_large", "gap_bits", "min", "largest PsyTrack win"),
    ("mmlpf_win_large", "gap_bits", "max", "largest MMLPF win"),
    ("sigma_alpha_low", "sigma_alpha", "min", r"lowest $\sigma_\alpha$"),
    ("sigma_alpha_high", "sigma_alpha", "max", r"highest $\sigma_\alpha$"),
    ("sigma_beta_low", "sigma_beta", "min", r"lowest $\sigma_\beta$"),
    ("sigma_beta_high", "sigma_beta", "max", r"highest $\sigma_\beta$"),
    ("sigma_phi_low", "sigma_phi", "min", r"lowest $\sigma_\varphi$"),
    ("sigma_phi_high", "sigma_phi", "max", r"highest $\sigma_\varphi$"),
]


def pick_examples(df, source_csv=None):
    """Choose one session per case. Reads the solution-comparison CSV, or the
    cohort CV CSV if that is what you have."""
    ok = df[(df["error"].isna()) | (df["error"] == "")].copy()
    ok["subject_id"] = ok["subject_id"].astype(str)
    if "gap_bits" not in ok.columns and "diff_bits" in ok.columns:
        # cohort CV CSV: aggregate placements to sessions first
        ok = (ok.groupby(["subject_id", "session_number"])
              .agg(gap_bits=("diff_bits", "median"),
                   sigma_alpha=("sigma_alpha", "median"),
                   sigma_beta=("sigma_beta", "median"),
                   sigma_phi=("sigma_phi", "median")).reset_index())
    picks = []
    used = set()
    for name, col, how, label in EXAMPLE_CASES:
        if col not in ok.columns:
            continue
        pool = ok[~ok.set_index(["subject_id", "session_number"]).index.isin(used)]
        if not len(pool):
            break
        r = (pool.nsmallest(1, col) if how == "min"
             else pool.nlargest(1, col)).iloc[0]
        used.add((r["subject_id"], int(r["session_number"])))
        picks.append(dict(case=name, label=label, criterion=col,
                          subject_id=str(r["subject_id"]),
                          session_number=int(r["session_number"]),
                          **{c: r[c] for c in
                             ("gap_bits", "sigma_alpha", "sigma_beta", "sigma_phi")
                             if c in r.index}))
    return pd.DataFrame(picks)


def draw_examples(picks, cache_dir=None, cv_csv=None, out_dir="examples",
                  block=BLOCK, test_frac=TEST_FRAC, seed=0,
                  n_particles=E_PARTICLES, workers=1):
    """One session figure per case, via latent_comparison.make_figure so the
    layout is the lab's standard plot_foraging_session stack."""
    os.makedirs(out_dir, exist_ok=True)
    sigmas = load_sigmas(cv_csv) if cv_csv else {}
    made = []
    for _, p in picks.iterrows():
        sid, sess = str(p["subject_id"]), int(p["session_number"])
        try:
            ch, rw, raw = lc.load_session(sid, sess)
            res = lc.fit_both(ch, rw, block=block, test_frac=test_frac,
                              seed=seed, n_particles=n_particles,
                              workers=workers)
            fname = os.path.join(out_dir, f"{p['case']}_{sid}_s{sess}.png")
            lc.make_figure(raw, res, subject_id=sid, session_number=sess,
                           fname=fname, block_len=block)
            made.append(fname)
            print(f"  {p['case']:>18}: {sid} s{sess} -> {fname}   "
                  f"gap {res['scores']['glm'] - res['scores']['mmlpf']:+.4f}",
                  flush=True)
        except Exception as e:
            print(f"  {p['case']:>18}: {sid} s{sess} FAILED "
                  f"{type(e).__name__}: {e}", flush=True)
    return made


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="./cache")
    ap.add_argument("--cv-csv", default=None,
                    help="cohort CV CSV; reuses its fitted MMLPF volatilities")
    ap.add_argument("--csv", default="solution_comparison.csv")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--particles", type=int, default=E_PARTICLES)
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--test-frac", type=float, default=TEST_FRAC)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--summarize", action="store_true",
                    help="skip fitting; analyse an existing --csv")
    ap.add_argument("--examples", action="store_true",
                    help="draw the eight example session figures")
    ap.add_argument("--examples-from", default=None,
                    help="pick examples from this CSV instead of --csv "
                         "(e.g. the cohort CV CSV)")
    ap.add_argument("--out-dir", default="examples")
    args = ap.parse_args()

    if args.summarize or args.examples:
        src = args.examples_from or args.csv
        if not os.path.exists(src):
            ap.error(f"{src} not found; run without --summarize/--examples "
                     f"first, or pass --examples-from")
        df = pd.read_csv(src)
    else:
        df = run(args.cache_dir, cv_csv=args.cv_csv, workers=args.workers,
                 n_particles=args.particles, block=args.block,
                 test_frac=args.test_frac, seed=args.seed, limit=args.limit,
                 out_csv=args.csv)

    if args.examples:
        picks = pick_examples(df)
        picks.to_csv("example_sessions_picked.csv", index=False)
        print(f"\nchose {len(picks)} example sessions "
              f"(wrote example_sessions_picked.csv):")
        print(picks[["case", "subject_id", "session_number"]].to_string(index=False))
        print()
        draw_examples(picks, cache_dir=args.cache_dir, cv_csv=args.cv_csv,
                      out_dir=args.out_dir, block=args.block,
                      test_frac=args.test_frac, seed=args.seed,
                      n_particles=args.particles)
        return

    res, ok = analyse(df)
    if len(res) or len(ok):
        plot_summary(res, ok)


if __name__ == "__main__":
    main()
