"""
Construct validity for the MMLPF's latent parameters.

THE QUESTION
------------
Parameter recovery on self-generated data shows the filter can identify its own
parameters. It does not show that the latents measure what their names claim.
This script tests that directly: does a latent estimated from one set of
sessions predict a MODEL-FREE behavioural signature measured on DIFFERENT
sessions from the same animal?

Model-free means computed from choices and rewards by descriptive regression
only -- no value model, no learning rate, nothing the MMLPF fitted. So the two
sides of each correlation are methodologically independent, and the split by
session makes them statistically independent too.

THE THREE PAIRINGS
------------------
    alpha  <->  reward-history integration timescale tau
                Fit a logistic regression of choice on the last K signed
                rewards; fit an exponential to the resulting coefficient
                profile. A fast learner weights recent rewards heavily and
                old ones not at all, i.e. short tau. Predicted sign: NEGATIVE
                (higher alpha, shorter tau). Reported as -tau so the expected
                correlation is positive and comparable to the others.

    beta   <->  NOT TESTED. No descriptive index was found that tracks true
                beta well enough to validate against -- see the comment on
                PAIRS below for the seven that were tried and rejected. beta is
                weakly identified from choice data, so its construct validity
                cannot be established this way.

    phi    <->  lag-1 choice-history coefficient c_1, and lose-stay rate
                Both measure choice repetition with reward held out. c_1 comes
                from the same regression (so it is adjusted for reward history);
                lose-stay is the raw repeat rate after unrewarded trials.
                Predicted: POSITIVE.

WHY THE SPLIT MATTERS
---------------------
Estimating the latent and the model-free index on the SAME trials shares noise
between them: a session that happens to look sticky inflates both phi and c_1
regardless of the animal's true tendency. The script reports both, but only the
cross-session correlation is a construct-validity result. Note also that with
~50 subjects the two are usually not distinguishable from each other -- do not
read a mechanism into their difference without a power calculation.

INPUTS
------
Reads the .npz cache written by `hpc_block_cv.py prepare` (choices, rewards,
per session), and optionally the per-placement CSV from `combine` to reuse the
already-fitted volatilities instead of refitting them. Reusing them is strongly
preferred: it turns an M-step per session into a single forward pass, which is
roughly a hundredfold cheaper and guarantees the latents match the ones behind
your cross-validation numbers.

    python construct_validity.py --cache-dir ./cache \\
        --cv-csv cohort_block_cv_hpc.csv --workers 8

Writes construct_validity.csv (per subject, per half) and
construct_validity.png.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.special import expit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import block_cv as bc

N_LAGS = 8            # lags in the descriptive regression
RIDGE = 1.0           # ridge penalty; the design is collinear at high lag
E_PARTICLES = 1500
MIN_TRIALS = 200


# ---------------------------------------------------------------------------
# model-free side: descriptive logistic regression on choice/reward history
# ---------------------------------------------------------------------------
def _logistic_ridge(X, y, ridge=RIDGE, iters=60, tol=1e-8):
    """IRLS logistic regression with an L2 penalty (intercept unpenalised).

    Written out rather than pulled from sklearn so the script has no dependency
    beyond scipy, and so the penalty structure is explicit.
    """
    n, k = X.shape
    w = np.zeros(k)
    P = np.eye(k) * ridge
    P[0, 0] = 0.0                      # column 0 is the intercept
    for _ in range(iters):
        eta = np.clip(X @ w, -30, 30)
        mu = expit(eta)
        s = np.clip(mu * (1 - mu), 1e-9, None)
        # Newton step on penalised log-likelihood
        H = X.T @ (X * s[:, None]) + P
        g = X.T @ (y - mu) - P @ w
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        w = w + step
        if np.max(np.abs(step)) < tol:
            break
    return w


def history_kernel(choices, rewards, n_lags=N_LAGS):
    """Descriptive regression of choice on signed reward and choice history.

    Returns dict with the reward kernel b_1..b_K, the choice kernel c_1..c_K,
    and summaries. No value model is involved -- this is a GLM on observables.
    """
    c = np.asarray(choices).astype(int)
    r = np.asarray(rewards).astype(int)
    T = len(c)
    if T <= n_lags + 20:
        return None
    ch_signed = np.where(c == 1, 1.0, -1.0)
    rw_signed = ch_signed * r                      # +1 rewarded R, -1 rewarded L, 0 none

    rows, ys = [], []
    for t in range(n_lags, T):
        feat = [1.0]
        feat += [rw_signed[t - k] for k in range(1, n_lags + 1)]
        feat += [ch_signed[t - k] for k in range(1, n_lags + 1)]
        rows.append(feat)
        ys.append(c[t])
    X = np.asarray(rows)
    y = np.asarray(ys, float)
    w = _logistic_ridge(X, y)

    b = w[1:n_lags + 1]                            # reward kernel
    cc = w[n_lags + 1:]                            # choice kernel

    # integration timescale: exponential fit to the reward kernel.
    # tau is in trials; a fast learner has a short tau.
    lags = np.arange(1, n_lags + 1, dtype=float)
    tau = np.nan
    if b[0] > 1e-6 and np.all(np.isfinite(b)):
        try:
            popt, _ = curve_fit(lambda x, a0, tt: a0 * np.exp(-x / tt),
                                lags, b, p0=[max(b[0], 1e-3), 2.0],
                                bounds=([0, 0.2], [np.inf, 40.0]), maxfev=5000)
            tau = float(popt[1])
        except (RuntimeError, ValueError):
            tau = np.nan

    # raw, regression-free repetition measures
    rep = c[1:] == c[:-1]
    lose = r[:-1] == 0
    win = r[:-1] == 1
    lose_stay = float(rep[lose].mean()) if lose.sum() else np.nan
    win_stay = float(rep[win].mean()) if win.sum() else np.nan

    return dict(
        tau=tau,
        neg_tau=-tau if np.isfinite(tau) else np.nan,
        rew_sensitivity=float(np.sum(np.abs(b))),
        persev_c1=float(cc[0]),
        lose_stay=lose_stay,
        win_stay=win_stay,
        repeat_rate=float(rep.mean()),
        b_kernel=b, c_kernel=cc)


# ---------------------------------------------------------------------------
# model side: one forward pass per session at the already-fitted volatilities
# ---------------------------------------------------------------------------
def session_latents(choices, rewards, sa, sb, sp, n_particles=E_PARTICLES,
                    seed=0):
    """Median latent level over the session, from a single unmasked pass.

    No mask: this is not a prediction task, it is parameter estimation, so the
    filter sees every trial. Volatilities come from the cross-validation fit so
    the latents are the same quantity those numbers were computed with.
    """
    o = bc.mmlpf_masked(sa, sb, sp, np.asarray(choices).astype(int),
                        np.asarray(rewards).astype(int),
                        np.ones(len(choices), bool),
                        num_particles=n_particles, seed=seed, collect=True)
    return dict(alpha=float(np.median(o['alpha'])),
                beta=float(np.median(o['beta'])),
                phi=float(np.median(o['phi'])),
                max_beta=float(np.max(o['beta'])))


# ---------------------------------------------------------------------------
def load_sigmas(cv_csv):
    """Per-(subject, session) median fitted volatilities from the CV run."""
    d = pd.read_csv(cv_csv)
    d = d[(d['error'].isna()) | (d['error'] == '')]
    d['subject_id'] = d['subject_id'].astype(str)
    g = (d.groupby(['subject_id', 'session_number'])
         [['sigma_alpha', 'sigma_beta', 'sigma_phi']].median())
    return {k: tuple(v) for k, v in g.iterrows()}


def _one_session(args):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    sid, sess_no, ch, rw, sig, n_lags, n_part = args
    row = dict(subject_id=sid, session_number=sess_no, n_trials=len(ch),
               error="")
    try:
        if sig is None:
            sa, sb, sp = bc.fit_mmlpf_masked(ch, rw, np.ones(len(ch), bool),
                                             workers=1)
        else:
            sa, sb, sp = sig
        row.update(sigma_alpha=sa, sigma_beta=sb, sigma_phi=sp)
        row.update(session_latents(ch, rw, sa, sb, sp, n_particles=n_part))
        mf = history_kernel(ch, rw, n_lags=n_lags)
        if mf is None:
            row['error'] = 'too few trials for the history regression'
        else:
            row.update({k: v for k, v in mf.items()
                        if k not in ('b_kernel', 'c_kernel')})
    except Exception as e:
        row['error'] = f"{type(e).__name__}: {e}"
    return row


def run(cache_dir, cv_csv=None, workers=1, n_lags=N_LAGS,
        n_particles=E_PARTICLES, limit=None, out_csv="construct_validity.csv"):
    import hpc_block_cv as h

    mf_path = os.path.join(cache_dir, "manifest.csv")
    if not os.path.exists(mf_path):
        raise FileNotFoundError(
            f"{mf_path} not found. This script reads the cache written by "
            f"`hpc_block_cv.py prepare`.")
    manifest = pd.read_csv(mf_path)
    manifest['subject_id'] = manifest['subject_id'].astype(str)
    subjects = list(manifest['subject_id'])
    if limit:
        subjects = subjects[:limit]

    sigmas = load_sigmas(cv_csv) if cv_csv else {}
    if cv_csv:
        print(f"reusing fitted volatilities for {len(sigmas)} sessions from "
              f"{cv_csv} (one forward pass each, no M-step)", flush=True)
    else:
        print("no --cv-csv given: refitting volatilities per session, which is "
              "~100x slower and may not reproduce the CV fit exactly", flush=True)

    jobs = []
    for sid in subjects:
        for (sess_no, ch, rw, _nr, _ni) in h.load_cached_subject(cache_dir, sid):
            if len(ch) < MIN_TRIALS:
                continue
            jobs.append((sid, sess_no, ch, rw,
                         sigmas.get((sid, sess_no)), n_lags, n_particles))
    print(f"{len(subjects)} subjects, {len(jobs)} sessions", flush=True)

    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            rows = []
            for n, row in enumerate(pool.imap_unordered(_one_session, jobs), 1):
                rows.append(row)
                if n % 25 == 0 or n == len(jobs):
                    print(f"  [{n}/{len(jobs)}]", flush=True)
    else:
        rows = []
        for n, job in enumerate(jobs, 1):
            rows.append(_one_session(job))
            if n % 25 == 0 or n == len(jobs):
                print(f"  [{n}/{len(jobs)}]", flush=True)

    df = pd.DataFrame(rows).sort_values(['subject_id', 'session_number'])
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(df)} sessions)")
    return df


# ---------------------------------------------------------------------------
# analysis: cross-session correlations
# ---------------------------------------------------------------------------
# Only pairings whose model-free side was VERIFIED to track the true parameter
# on simulated agents with known values (60 agents, alpha/beta/phi drawn
# independently). Verified correlations with ground truth:
#
#     true phi   vs persev_c1         r = +0.87
#     true phi   vs lose_stay         r = +0.83
#     true alpha vs -tau              r = +0.39
#
# BETA IS DELIBERATELY ABSENT. Seven candidate descriptive indices were tested
# against known beta -- reward-kernel amplitude b_1, sum|b_k|, win-stay,
# lose-switch, accuracy, win-stay-minus-lose-stay, and raw repeat rate. The
# best partial correlation controlling for alpha and phi was 0.35, and several
# had the wrong sign. So beta has no behavioural signature strong enough to
# validate against, which is a finding rather than a gap in this script: beta
# is weakly identified from choice data, consistent with its poor recovery and
# its saturation in the cohort fits. Do not substitute an unvalidated index
# here to fill the row -- a pairing that does not track ground truth in
# simulation cannot support a claim on real data.
PAIRS = [
    ('phi', 'persev_c1', '+',
     r'$\varphi$ vs lag-1 choice weight'),
    ('phi', 'lose_stay', '+',
     r'$\varphi$ vs lose-stay rate'),
    ('alpha', 'neg_tau', '+',
     r'$\alpha$ vs $-\tau$ (reward integration)'),
]


def _r_ci(r, n):
    if not np.isfinite(r) or n < 5:
        return (np.nan, np.nan)
    z, se = np.arctanh(np.clip(r, -0.999, 0.999)), 1 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def analyse(df):
    ok = df[(df['error'].isna()) | (df['error'] == '')].copy()
    ok['subject_id'] = ok['subject_id'].astype(str)
    ok['half'] = ok['session_number'] % 2

    # per subject, per half: median over that half's sessions
    cols = sorted({c for c, m, _s, _l in PAIRS} | {m for _c, m, _s, _l in PAIRS})
    per = ok.groupby(['subject_id', 'half'])[cols].median().reset_index()
    wide = {}
    for c in cols:
        w = per.pivot_table(index='subject_id', columns='half', values=c)
        if {0, 1} <= set(w.columns):
            wide[c] = w.dropna()

    print(f"\n{'='*78}")
    print(f"CONSTRUCT VALIDITY  ({ok['subject_id'].nunique()} subjects, "
          f"{len(ok)} sessions)")
    print('='*78)
    print("latent from ODD sessions vs model-free index from EVEN sessions "
          "(and vice versa),\naveraged. Same-session shown for contrast only -- "
          "it shares noise between\nthe two measures and is not a validity "
          "result.")
    print(f"\n{'pairing':>34} {'expect':>7} {'cross-session r':>17} "
          f"{'95% CI':>18} {'same':>7}")

    out = []
    for lat, mfree, sign, label in PAIRS:
        if lat not in wide or mfree not in wide:
            continue
        idx = wide[lat].index.intersection(wide[mfree].index)
        if len(idx) < 5:
            continue
        L, M = wide[lat].loc[idx], wide[mfree].loc[idx]
        # cross: latent(half 0) vs model-free(half 1), and the reverse
        r01 = stats.pearsonr(L[0].values, M[1].values)
        r10 = stats.pearsonr(L[1].values, M[0].values)
        r_cross = float(np.mean([r01.statistic, r10.statistic]))
        # Fisher-average the two, then a single CI at the subject n
        lo, hi = _r_ci(r_cross, len(idx))
        r_same = float(np.mean([
            stats.pearsonr(L[0].values, M[0].values).statistic,
            stats.pearsonr(L[1].values, M[1].values).statistic]))
        supports = (r_cross > 0) if sign == '+' else (r_cross < 0)
        flag = ""
        if not (lo <= 0 <= hi):
            flag = " *" if supports else " * WRONG SIGN"
        print(f"{label:>34} {sign:>7} {r_cross:17.3f} "
              f"{f'[{lo:+.2f}, {hi:+.2f}]':>18} {r_same:7.3f}{flag}")
        out.append(dict(latent=lat, model_free=mfree, label=label,
                        expected=sign, r_cross=r_cross, ci_lo=lo, ci_hi=hi,
                        r_same=r_same, n=len(idx),
                        r01=r01.statistic, r10=r10.statistic,
                        supports=bool(supports and not (lo <= 0 <= hi))))

    res = pd.DataFrame(out)
    print("\n* = CI excludes zero. A correlation whose CI spans zero is not "
          "evidence\n  either way at this n.")
    if len(res):
        n_sup = int(res['supports'].sum())
        print(f"\n{n_sup}/{len(res)} pairings show a non-zero correlation in "
              f"the predicted direction.")
        wrong = res[(~res['supports']) & ((res['ci_lo'] > 0) | (res['ci_hi'] < 0))]
        if len(wrong):
            print(f"WARNING: {len(wrong)} pairing(s) are significant in the "
                  f"WRONG direction:\n  "
                  + ", ".join(wrong['label']) +
                  "\n  a latent that anti-correlates with its own behavioural "
                  "signature is\n  evidence against the interpretation, not "
                  "noise.")
    return res, wide


def plot(res, wide, df, fname="construct_validity.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_OK, C_NO, GREY = '#17a398', '#c8511b', '#8a8a8a'
    n_pan = min(len(res), 3)
    fig, axes = plt.subplots(1, n_pan + 1,
                             figsize=(3.6 * (n_pan + 1), 3.9))
    axes = np.atleast_1d(axes)

    for ax, (_, row) in zip(axes[:n_pan], res.iterrows()):
        lat, mfree = row['latent'], row['model_free']
        idx = wide[lat].index.intersection(wide[mfree].index)
        L, M = wide[lat].loc[idx], wide[mfree].loc[idx]
        x = np.concatenate([L[0].values, L[1].values])
        y = np.concatenate([M[1].values, M[0].values])
        col = C_OK if row['supports'] else C_NO
        ax.scatter(x, y, s=26, color=col, edgecolor='k', lw=0.3, alpha=0.85)
        if len(x) > 2:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 40)
            ax.plot(xs, m * xs + b, ls='--', lw=1.2, color='#333333')
        ax.set_xlabel(f'inferred {lat} (one half)')
        ax.set_ylabel(f'{mfree} (other half)')
        ax.set_title(row['label'], loc='left')
        ax.text(0.03, 0.97,
                f"r = {row['r_cross']:+.2f}\n"
                f"95% CI [{row['ci_lo']:+.2f}, {row['ci_hi']:+.2f}]\n"
                f"n = {int(row['n'])} mice",
                transform=ax.transAxes, va='top', fontsize=6.8)
        ax.margins(0.06)

    axS = axes[-1]
    y = np.arange(len(res))[::-1]
    cols = [C_OK if s else C_NO for s in res['supports']]
    axS.barh(y, res['r_cross'].values, color=cols, height=0.6,
             edgecolor='white', lw=0.8)
    axS.errorbar(res['r_cross'].values, y,
                 xerr=[res['r_cross'] - res['ci_lo'],
                       res['ci_hi'] - res['r_cross']],
                 fmt='none', ecolor='#333333', elinewidth=1.0, capsize=3)
    axS.axvline(0, color='k', lw=1.1)
    axS.set_yticks(y)
    axS.set_yticklabels([f"{r['latent']} / {r['model_free']}"
                         for _, r in res.iterrows()], fontsize=6.8)
    axS.set_xlabel('cross-session correlation')
    axS.set_title('All pairings, with 95% CIs', loc='left')
    axS.text(0.98, 0.02, 'CI crossing 0 = no evidence',
             transform=axS.transAxes, ha='right', fontsize=6.3, color=GREY)

    for ax, L in zip(axes, 'abcd'):
        ax.text(-0.16, 1.02, L, transform=ax.transAxes, fontsize=12,
                fontweight='bold', va='bottom')
    fig.tight_layout(w_pad=2.2)
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"saved {fname}")
    return fig


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="./cache")
    ap.add_argument("--cv-csv", default=None,
                    help="per-placement CSV from `combine`; reuses its fitted "
                         "volatilities instead of refitting (strongly "
                         "recommended)")
    ap.add_argument("--csv", default="construct_validity.csv")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--lags", type=int, default=N_LAGS)
    ap.add_argument("--particles", type=int, default=E_PARTICLES)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--summarize", action="store_true",
                    help="skip fitting; analyse an existing --csv")
    args = ap.parse_args()

    if args.summarize:
        df = pd.read_csv(args.csv)
    else:
        df = run(args.cache_dir, cv_csv=args.cv_csv, workers=args.workers,
                 n_lags=args.lags, n_particles=args.particles,
                 limit=args.limit, out_csv=args.csv)
    res, wide = analyse(df)
    if len(res):
        plot(res, wide, df)


if __name__ == "__main__":
    main()
