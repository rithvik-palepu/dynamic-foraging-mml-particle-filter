"""
Should the side bias drift, be static, or be absent?

Three parameterisations of the same sixth state, fit and scored identically:

    absent    no bias state at all                       (the 5-state MMLPF)
    static    bias present, volatility FIXED at 0        (one constant/session)
    drifting  bias present, volatility FITTED by volatility fit  (random walk)

`static` is the comparison that was missing. Earlier work in this project
compared drifting against absent and found drifting costs ~0.02 bits/trial on
data with no true bias -- but that does not distinguish "a bias term is wrong"
from "letting the bias WANDER is wrong". Decomposing it showed the static offset
was free (+0.004) and the drift carried the whole cost (-0.023), which is what
this script tests directly across generative regimes and real sessions.

REGIMES
-------
    const-bias   true bias constant at +2. Static is correctly specified.
    drift-bias   true bias random-walks (sigma 0.02). Drifting is correctly
                 specified; static is misspecified but may still win if the
                 drift is slow relative to the session.
    real         sessions from the AIND database. Truth unknown; this is the
                 case that decides the modelling choice.

The drift-bias regime is the honest test of the recommendation. If static wins
even when the truth genuinely drifts, the conclusion is about identifiability
(a free walk cannot be estimated from binary choices) rather than about mice.

USAGE
-----
    python bias_parameterization.py --regimes const-bias drift-bias
    python bias_parameterization.py --regimes real --subjects 651554 651555
    python bias_parameterization.py --summarize --csv bias_param.csv
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import differential_evolution
from scipy.special import expit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import block_cv as bc                                     # noqa: E402

BLOCK = 20
TEST_FRAC = 0.2
M_PARTICLES = 200
E_PARTICLES = 1500
SIGMA_BOUND = (0.001, 0.4)
# Search range for sigma_bias ALONE, separate from the other three. A free walk
# with the wide bound fits sigma_bias ~ 0.16 even when the true bias is exactly
# constant, because the volatility fit cannot switch the state off -- so the wide bound
# does not let the data decide, it guarantees wandering. Tightening this to
# 0.05 keeps a genuinely slow drift reachable while making the runaway values
# unreachable. Override per run with --bias-bound; 0.0 means static.
BIAS_BOUND = (0.0, 0.05)
MODELS = ("absent", "static", "drifting")


def simulate(T=500, alpha=0.35, beta=6.0, phi=1.5, bias0=2.0, sigma_bias=0.0,
             lapse=0.05, block=60, seed=0):
    """Q-agent with perseveration and a side bias that may drift."""
    rng = np.random.default_rng(seed)
    bias = bias0 + np.cumsum(rng.normal(0, sigma_bias, T))
    Q = np.array([0.5, 0.5])
    prev = 0.0
    p = np.array([0.8, 0.1])
    ch, rw = [], []
    for t in range(T):
        if t and t % block == 0:
            p = p[::-1].copy()
        z = beta * (Q[1] - Q[0]) + phi * prev + bias[t]
        pr = (1 - lapse) * expit(np.clip(z, -50, 50)) + lapse / 2
        c = int(rng.random() < pr)
        r = int(rng.random() < p[c])
        Q[c] += alpha * (r - Q[c])
        prev = 1.0 if c == 1 else -1.0
        ch.append(c)
        rw.append(r)
    return np.array(ch), np.array(rw), bias


def bits(p, y, mask):
    p = np.clip(np.asarray(p)[mask], 1e-9, 1 - 1e-9)
    y = np.asarray(y)[mask]
    return float(-np.mean(y * np.log2(p) + (1 - y) * np.log2(1 - p)))


def fit_one(choices, rewards, train_mask, model, seed=0, maxiter=8, popsize=5,
            bias_bound=None):
    """Fit one parameterisation. Returns (sigmas dict, collected output).

    absent   -> sigma_bias=None, so mmlpf_masked allocates 5 states
    static   -> sigma_bias=0.0, so the state exists but cannot walk
    drifting -> sigma_bias is a fourth search dimension
    """
    n_dim = 4 if model == "drifting" else 3
    sb_fixed = {"absent": None, "static": 0.0}.get(model)

    def obj(vec):
        sb = float(vec[3]) if model == "drifting" else sb_fixed
        nll, _ = bc.mmlpf_masked(vec[0], vec[1], vec[2], choices, rewards,
                                 train_mask, num_particles=M_PARTICLES,
                                 seed=seed, sigma_bias=sb)
        return nll

    # the first three dimensions keep the wide range; sigma_bias gets its own
    bounds = [SIGMA_BOUND] * 3
    if model == "drifting":
        bounds = bounds + [bias_bound or BIAS_BOUND]
    res = differential_evolution(
        obj, bounds=bounds, maxiter=maxiter, popsize=popsize,
        tol=0.01, seed=0, polish=False)
    sb = float(res.x[3]) if model == "drifting" else sb_fixed
    out = bc.mmlpf_masked(res.x[0], res.x[1], res.x[2], choices, rewards,
                          train_mask, num_particles=E_PARTICLES, seed=seed,
                          sigma_bias=sb, collect=True)
    sig = dict(alpha=float(res.x[0]), beta=float(res.x[1]),
               phi=float(res.x[2]), bias=sb)
    return sig, out


def score_session(choices, rewards, seed=0, truth_bias=None, maxiter=8,
                  popsize=5, bias_bound=None):
    T = len(choices)
    tm = bc.make_block_mask(T, block=BLOCK, test_frac=TEST_FRAC, seed=seed)
    base = bits(np.full(T, float(choices[tm].mean())), choices, ~tm)
    row = dict(n_trials=T, p_right=float(choices.mean()), base_bits=base,
               n_test=int((~tm).sum()))
    for model in MODELS:
        sig, out = fit_one(choices, rewards, tm, model, seed=seed,
                           maxiter=maxiter, popsize=popsize,
                           bias_bound=bias_bound)
        row[f"{model}_bits"] = bits(out["p_right"], choices, ~tm)
        row[f"{model}_phi"] = float(np.median(out["phi"]))
        row[f"{model}_beta"] = float(np.median(out["beta"]))
        row[f"{model}_clip"] = float(np.mean(out["beta"] > 95))
        row[f"{model}_sig_bias"] = (np.nan if sig["bias"] is None
                                    else sig["bias"])
        if model != "absent":
            row[f"{model}_bias_level"] = float(np.median(out["bias"]))
            if truth_bias is not None:
                row[f"{model}_bias_err"] = float(
                    np.median(out["bias"]) - np.median(truth_bias))
    return row


def run(regimes, subjects=("651554",), sessions=(1, 2), seeds=2, trials=500,
        out_csv="bias_param.csv", maxiter=8, popsize=5, bias_bound=None):
    rows = []
    for regime in regimes:
        if regime == "real":
            import bias_showcase as bs
            for sid in subjects:
                for sess in sessions:
                    try:
                        c, r, _pL, _pR = bs.load_real_session(sid, sess)
                    except Exception as e:
                        print(f"  {sid} s{sess}: {type(e).__name__}: "
                              f"{str(e)[:80]}", flush=True)
                        continue
                    for s in range(seeds):
                        t0 = time.time()
                        row = score_session(c, r, seed=s, maxiter=maxiter,
                                            popsize=popsize,
                                            bias_bound=bias_bound)
                        row.update(regime=regime, label=f"{sid}_s{sess}",
                                   seed=s, seconds=round(time.time() - t0, 1),
                                   bias_bound_hi=(bias_bound or BIAS_BOUND)[1])
                        rows.append(row)
                        print(f"  {row['label']} seed{s}: absent "
                              f"{row['absent_bits']:.3f}  static "
                              f"{row['static_bits']:.3f}  drifting "
                              f"{row['drifting_bits']:.3f}", flush=True)
        else:
            sb_true = 0.0 if regime == "const-bias" else 0.02
            for s in range(seeds):
                c, r, btrue = simulate(T=trials, sigma_bias=sb_true,
                                       seed=200 + s)
                t0 = time.time()
                row = score_session(c, r, seed=s, truth_bias=btrue,
                                    maxiter=maxiter, popsize=popsize,
                                    bias_bound=bias_bound)
                row.update(regime=regime, label=regime, seed=s,
                           bias_bound_hi=(bias_bound or BIAS_BOUND)[1],
                           true_sigma_bias=sb_true,
                           true_bias_range=float(btrue.max() - btrue.min()),
                           seconds=round(time.time() - t0, 1))
                rows.append(row)
                print(f"  {regime} seed{s}: absent {row['absent_bits']:.3f}  "
                      f"static {row['static_bits']:.3f}  drifting "
                      f"{row['drifting_bits']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    return df


def analyse(df):
    print("\n" + "=" * 72)
    print("HELD-OUT BITS/TRIAL BY PARAMETERISATION (lower is better)")
    print("=" * 72)
    print(f"{'regime':<14} {'base':>7} {'absent':>8} {'static':>8} "
          f"{'drifting':>9} {'static-drift':>13}")
    for reg, g in df.groupby("regime"):
        st, dr = g["static_bits"].mean(), g["drifting_bits"].mean()
        print(f"{reg:<14} {g['base_bits'].mean():7.3f} "
              f"{g['absent_bits'].mean():8.3f} {st:8.3f} {dr:9.3f} "
              f"{st - dr:+13.4f}")
    print("\nnegative static-drift = STATIC predicts better")

    print("\n--- fitted sigma_bias when the walk is allowed ---")
    for reg, g in df.groupby("regime"):
        extra = ""
        if "true_sigma_bias" in g and g["true_sigma_bias"].notna().any():
            extra = f"   (true {g['true_sigma_bias'].mean():.3f})"
        print(f"  {reg:<14} {g['drifting_sig_bias'].mean():.3f}{extra}")

    print("\n--- does the bias state swallow perseveration? "
          "(median phi) ---")
    print(f"{'regime':<14} {'absent':>8} {'static':>8} {'drifting':>9}")
    for reg, g in df.groupby("regime"):
        print(f"{reg:<14} {g['absent_phi'].mean():8.3f} "
              f"{g['static_phi'].mean():8.3f} {g['drifting_phi'].mean():9.3f}")

    print("\n--- beta saturation (fraction of trials at the clip) ---")
    print(f"{'regime':<14} {'absent':>8} {'static':>8} {'drifting':>9}")
    for reg, g in df.groupby("regime"):
        print(f"{reg:<14} {g['absent_clip'].mean():8.3f} "
              f"{g['static_clip'].mean():8.3f} "
              f"{g['drifting_clip'].mean():9.3f}")

    n = len(df)
    if n >= 5:
        w = stats.wilcoxon(df["static_bits"], df["drifting_bits"])
        print(f"\nstatic vs drifting across all {n} fits: "
              f"static better on {(df['static_bits'] < df['drifting_bits']).mean():.0%}, "
              f"Wilcoxon p = {w.pvalue:.3g}")
    else:
        print(f"\nonly {n} fits: too few for a signed-rank test; "
              f"static better on "
              f"{(df['static_bits'] < df['drifting_bits']).mean():.0%} of them")
    return df


def plot(df, fname="bias_parameterization.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = {"absent": "#9a9a9a", "static": "#17a398", "drifting": "#c8511b"}
    regs = list(df["regime"].unique())
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.0, 4.4))

    x = np.arange(len(regs))
    w = 0.26
    for i, m in enumerate(MODELS):
        vals = [df[df["regime"] == r][f"{m}_bits"].mean() for r in regs]
        axA.bar(x + (i - 1) * w, vals, w, color=C[m], edgecolor="white",
                lw=0.7, label=m)
        for xi, v in zip(x + (i - 1) * w, vals):
            axA.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=6.0)
    for xi, r in zip(x, regs):
        b = df[df["regime"] == r]["base_bits"].mean()
        axA.plot([xi - 1.6 * w, xi + 1.6 * w], [b, b], color="0.4", ls="--",
                 lw=1.0)
    axA.set_xticks(x)
    axA.set_xticklabels(regs, fontsize=7.5)
    axA.set_ylabel("Held-out bits/trial (lower is better)")
    # Report the MEASURED pattern. A hardcoded "static beats drifting"
    # contradicts the data whenever a regime comes out the other way -- and
    # const-bias did exactly that on the first run.
    wins = {r: (df[df["regime"] == r]["static_bits"].mean()
                < df[df["regime"] == r]["drifting_bits"].mean())
            for r in regs}
    if all(wins.values()):
        head = "Static predicts better in every regime tested."
    elif not any(wins.values()):
        head = "Drifting predicts better in every regime tested."
    else:
        won = [r for r in regs if wins[r]]
        head = ("Direction is regime-dependent: static wins in "
                + ", ".join(won) + "; drifting wins elsewhere.")
    axA.set_title(head + "\nDashed = base rate; a bar above it is worse "
                  "than ignoring the data.", loc="left", fontsize=8)
    axA.legend(frameon=False, fontsize=7, ncol=3)

    # per-fit paired difference
    d = df["static_bits"] - df["drifting_bits"]
    axB.axhline(0, color="k", lw=1.0)
    for i, (reg, g) in enumerate(df.groupby("regime")):
        dd = g["static_bits"] - g["drifting_bits"]
        axB.scatter(np.full(len(dd), i) + np.random.uniform(-.06, .06, len(dd)),
                    dd, s=42, color=C["static"], edgecolor="white", lw=0.6,
                    zorder=3)
    axB.set_xticks(range(len(regs)))
    axB.set_xticklabels(regs, fontsize=7.5)
    axB.set_ylabel("static minus drifting (bits/trial)")
    try:
        p_w = stats.wilcoxon(df["static_bits"], df["drifting_bits"]).pvalue
        ptxt = f", Wilcoxon p = {p_w:.2g}"
    except Exception:
        ptxt = ""
    axB.set_title(f"Below zero = static wins. Static better on "
                  f"{(d < 0).mean():.0%} of {len(d)} fits{ptxt}.\n"
                  f"Each point is one (session, mask-seed) fit.",
                  loc="left", fontsize=8)
    for ax, L in zip((axA, axB), "ab"):
        ax.text(-0.1, 1.06, L, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"saved {fname}")
    return fig


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regimes", nargs="+",
                    default=["const-bias", "drift-bias", "real"],
                    choices=["const-bias", "drift-bias", "real"])
    ap.add_argument("--subjects", nargs="+", default=["651554"])
    ap.add_argument("--sessions", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--maxiter", type=int, default=8)
    ap.add_argument("--popsize", type=int, default=5)
    ap.add_argument("--bias-bound", type=float, default=BIAS_BOUND[1],
                    help="upper limit on sigma_bias for the drifting model "
                         "(default %(default)s). 0.0 forces it static; 0.4 "
                         "reproduces the original unbounded run.")
    ap.add_argument("--csv", default="bias_param.csv")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    df = (pd.read_csv(args.csv) if args.summarize
          else run(args.regimes, subjects=args.subjects,
                   sessions=tuple(args.sessions), seeds=args.seeds,
                   trials=args.trials, out_csv=args.csv,
                   maxiter=args.maxiter, popsize=args.popsize,
                   bias_bound=(0.0, args.bias_bound)))
    analyse(df)
    plot(df)


if __name__ == "__main__":
    main()
