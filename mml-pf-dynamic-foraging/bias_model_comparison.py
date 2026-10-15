"""
Does a drifting bias term close the MMLPF's predictive gap?

Fits THREE models on the same interleaved-block masks and scores them on the
same held-out trials:

    PsyTrack        GLM with an explicit bias regressor  (the benchmark)
    MMLPF-5         beta, phi                            (the current model)
    MMLPF-6         beta, phi, drifting bias b_t         (the proposed model)

WHY
---
The cohort run showed the MMLPF's deficit tracks how side-biased the animal is
(subject-clustered rho = +0.25 between the held-out cost of a constant-rate
predictor and the gap). The mechanism is a missing regressor: PsyTrack's
decision variable carries an explicit bias weight, the MMLPF's does not.

    MMLPF-5   z = beta*(Q_R - Q_L) + phi*c_{t-1}
    MMLPF-6   z = beta*(Q_R - Q_L) + phi*c_{t-1} + b_t
    PsyTrack  z = w_bias + sum_k w_k x_k

phi cannot stand in for b: phi*c_{t-1} flips sign with the last choice, b does
not. Analytically phi absorbs at most (2p-1)^2 of a constant bias, where p is
P(preferred side) -- 36% for an 80/20 animal, 4% for a 60/40 one.

WHAT THIS SCRIPT DECIDES
------------------------
Three questions, in order of importance:

    1. Does MMLPF-6 predict better than MMLPF-5 out of sample, and does the
       improvement concentrate on side-biased sessions as predicted?
    2. Does adding b_t reduce beta saturation? If beta was standing in for
       bias, the fraction of trials at the clip of 100 should drop.
    3. Does phi change? Any shift is the contamination being removed, so the
       previously reported phi values (and the construct-validity result that
       rests on them) need restating.

An extra parameter that only improves TRAINING fit proves nothing -- the
comparison is held-out bits/trial throughout, on trials neither model updated
on.

The three models are scored on IDENTICAL masks with identical particle counts
and seeds, so differences are attributable to the model, not the protocol.

USAGE
-----
    # cohort, reading the .npz cache written by hpc_block_cv.prepare
    python bias_model_comparison.py --cache-dir ./cache --workers 8

    # one subject, for a quick look
    python bias_model_comparison.py --cache-dir ./cache --limit 1

    # re-analyse an existing CSV without refitting
    python bias_model_comparison.py --summarize --csv bias_comparison.csv

RUNTIME
-------
The 4-parameter volatility fit costs about 1.5x the 3-parameter one, and this script
runs BOTH plus the GLM, so budget roughly 3.5x a plain cohort run. On 50
subjects x 20 sessions x 10 seeds that is substantial -- start with --limit to
size it before committing to the full grid.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

# so `python bias_model_comparison.py` works from any cwd, not only from the
# directory holding block_cv.py / matched_validation.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import block_cv as bc                                    # noqa: E402
from block_cv import build_glm_regressors       # noqa: E402

N_LAGS = 2
M_PARTICLES = bc.M_PARTICLES
E_PARTICLES = bc.E_PARTICLES
BLOCK = 20
TEST_FRAC = 0.2
SEEDS = 10
MIN_TRIALS = 200

RESULT_COLUMNS = [
    "subject_id", "session_number", "seed", "n_trials", "n_test",
    "base_bits", "glm_bits", "mmlpf5_bits", "mmlpf6_bits",
    "diff5_glm", "diff6_glm", "diff6_5",
    "glm_sigma",
    "sigma_alpha5", "sigma_beta5", "sigma_phi5",
    "sigma_alpha6", "sigma_beta6", "sigma_phi6", "sigma_bias6",
    "sigma_bias_bound_hi", "p_right", "side_skew",
    "mean_beta5", "max_beta5", "frac_beta_clip5",
    "mean_beta6", "max_beta6", "frac_beta_clip6",
    "mean_phi5", "mean_phi6", "mean_bias6", "sd_bias6",
    "seconds", "error",
]


def bits(p, y, mask):
    """Held-out bits/trial. Lower is better."""
    p = np.clip(np.asarray(p)[mask], 1e-9, 1 - 1e-9)
    y = np.asarray(y)[mask]
    return float(-np.mean(y * np.log2(p) + (1 - y) * np.log2(1 - p)))


def score_session(choices, rewards, seed):
    """Fit and score all three models on one block-CV placement.

    Returns a dict of scalars. Everything is computed on the SAME mask, so the
    only difference between models is the model.
    """
    T = len(choices)
    train_mask = bc.make_block_mask(T, block=BLOCK, test_frac=TEST_FRAC,
                                   seed=seed)
    test_mask = ~train_mask

    # --- base rate: constant predictor fit on training trials ---
    base_p = np.full(T, float(choices[train_mask].mean()))

    # --- PsyTrack (GLM with explicit bias regressor) ---
    rew, unrew = build_glm_regressors(choices, rewards,
                                      np.zeros(T, dtype=int), n_lags=N_LAGS)
    X = np.column_stack([np.ones(T), rew, unrew])
    glm_sigma = bc.fit_glm_sigma(X, choices, train_mask)
    # glm_masked returns (p_pred, p_smooth). p_pred is the ONE-SIDED predictive
    # P(right) -- the only one usable for scoring, since p_smooth uses future
    # choices. p_smooth is None unless smooth=True.
    glm_p, _ = bc.glm_masked(X, choices, glm_sigma, train_mask)

    # --- MMLPF-5: the current model ---
    s5 = bc.fit_mmlpf_masked(choices, rewards, train_mask)
    o5 = bc.mmlpf_masked(*s5, choices, rewards, train_mask,
                         num_particles=E_PARTICLES, seed=seed, collect=True)

    # --- MMLPF-6: with the drifting bias ---
    s6 = bc.fit_mmlpf_masked(choices, rewards, train_mask, with_bias=True)
    o6 = bc.mmlpf_masked(*s6[:3], choices, rewards, train_mask,
                         num_particles=E_PARTICLES, seed=seed,
                         sigma_bias=s6[3], collect=True)

    b_base = bits(base_p, choices, test_mask)
    b_glm = bits(glm_p, choices, test_mask)
    b_5 = bits(o5["p_right"], choices, test_mask)
    b_6 = bits(o6["p_right"], choices, test_mask)

    return dict(
        n_trials=T, n_test=int(test_mask.sum()),
        base_bits=b_base, glm_bits=b_glm, mmlpf5_bits=b_5, mmlpf6_bits=b_6,
        # DIRECT side-bias measures. base_bits was used for this earlier and is
        # the wrong instrument: it is the held-out cost of a constant predictor,
        # so a 70/30 animal sits near 0.88 -- indistinguishable from a balanced
        # one -- and only extreme skews push it low. side_skew separates the
        # skew itself from how stable it is.
        p_right=float(choices.mean()),
        side_skew=float(abs(choices.mean() - 0.5)),
        # positive = the MMLPF variant is better than the comparator
        diff5_glm=b_glm - b_5, diff6_glm=b_glm - b_6, diff6_5=b_5 - b_6,
        glm_sigma=glm_sigma,
        sigma_alpha5=s5[0], sigma_beta5=s5[1], sigma_phi5=s5[2],
        sigma_alpha6=s6[0], sigma_beta6=s6[1], sigma_phi6=s6[2],
        sigma_bias6=s6[3],
        # stamp the search range that produced this row: sigma_bias fits
        # depend strongly on it, so rows from different bounds must never be
        # pooled or resumed into one another
        sigma_bias_bound_hi=float(bc.DE_BOUNDS_BIAS[3][1]),
        mean_beta5=float(np.median(o5["beta"])),
        max_beta5=float(np.max(o5["beta"])),
        frac_beta_clip5=float(np.mean(o5["beta"] > 95.0)),
        mean_beta6=float(np.median(o6["beta"])),
        max_beta6=float(np.max(o6["beta"])),
        frac_beta_clip6=float(np.mean(o6["beta"] > 95.0)),
        mean_phi5=float(np.median(o5["phi"])),
        mean_phi6=float(np.median(o6["phi"])),
        mean_bias6=float(np.median(o6["bias"])),
        sd_bias6=float(np.std(o6["bias"])),
    )


def write_demo_cache(cache_dir="./cache_demo", n_subjects=4, n_sessions=3,
                     n_trials=460, seed=0):
    """Write a cache of SIMULATED subjects, in hpc_block_cv.prepare's format.

    For checking the pipeline runs and for seeing the bias term work on data
    whose bias is KNOWN -- not for any claim about real mice. Subjects differ in
    their true side bias (0 to 3 in the logit), which is the quantity the sixth
    state is meant to capture, so the reported gain should grow across subjects.

    The generative model here IS the MMLPF's own model, so these numbers are a
    ceiling on recovery, not evidence about behaviour.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(cache_dir, exist_ok=True)
    truth = []
    for si in range(n_subjects):
        # spread the true bias across subjects so the gain-vs-bias relationship
        # is visible even at this tiny n
        bias = 3.0 * si / max(n_subjects - 1, 1)
        alpha = float(rng.uniform(0.2, 0.5))
        beta = float(rng.uniform(4.0, 8.0))
        phi = float(rng.uniform(0.5, 2.0))
        all_ch, all_rw = [], []
        for j in range(n_sessions):
            Q = np.array([0.5, 0.5])
            prev = 0.0
            p = np.array([0.8, 0.1])
            ch, rw = [], []
            g = np.random.default_rng(1000 * si + j)
            for t in range(n_trials):
                if t and t % 60 == 0:
                    p = p[::-1].copy()
                z = beta * (Q[1] - Q[0]) + phi * prev + bias
                pr = 0.95 / (1.0 + np.exp(-np.clip(z, -50, 50))) + 0.025
                c = int(g.random() < pr)
                r = int(g.random() < p[c])
                Q[c] += alpha * (r - Q[c])
                prev = 1.0 if c == 1 else -1.0
                ch.append(c)
                rw.append(r)
            all_ch.append(np.array(ch))
            all_rw.append(np.array(rw))
        sid = f"SIM{si:03d}"
        # SAME layout as hpc_block_cv.prepare: concatenated arrays + offsets,
        # so load_cached reads demo and real caches through one code path
        lens = [len(a) for a in all_ch]
        np.savez_compressed(
            os.path.join(cache_dir, f"{sid}.npz"),
            session_numbers=np.arange(1, n_sessions + 1),
            session_dates=np.array([f"2026-01-{j+1:02d}" for j in range(n_sessions)]),
            session_ids=np.array([f"{sid}_{j+1}" for j in range(n_sessions)]),
            n_raw=np.array(lens), n_ignored=np.zeros(n_sessions, dtype=int),
            choices=np.concatenate(all_ch), rewards=np.concatenate(all_rw),
            offsets=np.concatenate([[0], np.cumsum(lens)]))
        truth.append(dict(subject_id=sid, true_bias=bias, true_alpha=alpha,
                          true_beta=beta, true_phi=phi))
    tdf = pd.DataFrame(truth)
    tdf.to_csv(os.path.join(cache_dir, "ground_truth.csv"), index=False)
    print(f"wrote {n_subjects} simulated subjects x {n_sessions} sessions "
          f"x {n_trials} trials to {cache_dir}")
    print(f"true side bias per subject: "
          f"{', '.join(f'''{r.subject_id} {r.true_bias:.1f}''' for r in tdf.itertuples())}")
    return cache_dir


def load_cached(cache_dir, max_sessions=None):
    """Per-subject session arrays written by hpc_block_cv.prepare.

    max_sessions caps how many sessions per subject are used. The cap is applied
    AFTER the cache is built, so subject eligibility (which needs the full
    session count) is unaffected -- the cohort is the same animals, just fewer
    sessions each. That makes a laptop-sized run comparable in kind to the full
    one, where reducing --sessions in `prepare` would change WHICH animals
    qualify.

    Session NUMBERS come from the cache, not from enumeration: prepare skips
    sessions below its own min-trials floor, so the j-th cached session is not
    necessarily session j. Using the stored number keeps rows joinable to
    cohort_block_cv_hpc.csv.
    """
    import hpc_block_cv as hpc

    out = {}
    for f in sorted(glob.glob(os.path.join(cache_dir, "*.npz"))):
        sid = os.path.splitext(os.path.basename(f))[0]
        # delegate to the canonical reader rather than re-deriving the layout
        # (concatenated choices/rewards + offsets, not one array per session)
        sessions = [(sess_no, c, r)
                    for sess_no, c, r, _n_raw, _n_ign
                    in hpc.load_cached_subject(cache_dir, sid)
                    if len(c) >= MIN_TRIALS]
        if max_sessions:
            sessions = sessions[:max_sessions]
        if sessions:
            out[sid] = sessions
    return out


def _worker(args):
    """Module-level for picklability. One (subject, session, seed) placement."""
    import time
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    sid, sess_no, seed, choices, rewards = args
    t0 = time.time()
    row = dict(subject_id=sid, session_number=sess_no, seed=seed,
               error="", seconds=0.0)
    try:
        row.update(score_session(choices, rewards, seed))
    except Exception as e:                                   # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {e}"
    row["seconds"] = round(time.time() - t0, 1)
    return {k: row.get(k, np.nan) for k in RESULT_COLUMNS}


def run(cache_dir, out_csv="bias_comparison.csv", workers=1, limit=None,
        seeds=SEEDS, max_sessions=None):
    cohort = load_cached(cache_dir, max_sessions=max_sessions)
    if not cohort:
        raise RuntimeError(
            f"no .npz files with >= {MIN_TRIALS}-trial sessions in {cache_dir}. "
            f"Run hpc_block_cv.prepare first, or check the path.")
    sids = sorted(cohort)[:limit] if limit else sorted(cohort)

    # schema guard: never resume into a CSV written by an older version
    if os.path.exists(out_csv):
        have = set(pd.read_csv(out_csv, nrows=1).columns)
        missing = [c for c in RESULT_COLUMNS if c not in have]
        if missing:
            import datetime
            bak = out_csv.replace(
                ".csv", f"_old_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv")
            os.rename(out_csv, bak)
            print(f"stale schema (missing {len(missing)} cols) -> renamed to {bak}")

    # A bound change silently invalidates resume: the (subject, session, seed)
    # key is identical, so old rows would be kept and new ones appended, giving
    # one CSV with two different models in it. Detect and start fresh instead.
    cur_bound = float(bc.DE_BOUNDS_BIAS[3][1])
    if os.path.exists(out_csv):
        d0 = pd.read_csv(out_csv)
        if "sigma_bias_bound_hi" in d0.columns and len(d0):
            olds = sorted(set(d0["sigma_bias_bound_hi"].dropna().round(6)))
            if olds and any(abs(o - cur_bound) > 1e-9 for o in olds):
                import datetime
                bak = out_csv.replace(
                    ".csv", f"_bound{olds[0]:g}_"
                            f"{datetime.datetime.now():%Y%m%d_%H%M%S}.csv")
                os.rename(out_csv, bak)
                print(f"** sigma_bias bound changed: CSV holds {olds}, "
                      f"DE_BOUNDS_BIAS now caps at {cur_bound:g}. Renamed the "
                      f"old file to {bak} and starting fresh -- rows fit under "
                      f"different bounds are different models and must not be "
                      f"pooled. **", flush=True)

    done = set()
    if os.path.exists(out_csv):
        d0 = pd.read_csv(out_csv)
        done = set(zip(d0["subject_id"].astype(str),
                       d0["session_number"].astype(int),
                       d0["seed"].astype(int)))

    tasks = []
    for sid in sids:
        for sess_no, c, r in cohort[sid]:
            for s in range(seeds):
                if (sid, sess_no, s) not in done:
                    tasks.append((sid, sess_no, s, c, r))
    print(f"{len(sids)} subjects | {len(tasks)} placements to fit "
          f"({len(done)} already done)", flush=True)
    if not tasks:
        return pd.read_csv(out_csv)

    def _write(row):
        pd.DataFrame([row], columns=RESULT_COLUMNS).to_csv(
            out_csv, mode="a", header=not os.path.exists(out_csv), index=False)

    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for i, row in enumerate(pool.imap_unordered(_worker, tasks), 1):
                _write(row)
                if i % 25 == 0 or i == len(tasks):
                    print(f"  [{i}/{len(tasks)}]", flush=True)
    else:
        for i, t in enumerate(tasks, 1):
            row = _worker(t)
            _write(row)
            if i % 10 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] {row['subject_id']} "
                      f"s{row['session_number']} "
                      f"{row['error'] or f'''diff6_5 {row['diff6_5']:+.4f}'''}",
                      flush=True)

    return pd.read_csv(out_csv)


def _cluster_ci(df, col, n_boot=2000, seed=0):
    """Bootstrap CI on a median, resampling SUBJECTS (sessions nest in them)."""
    groups = [g for _, g in df.groupby("subject_id")]
    if len(groups) < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        s = pd.concat([groups[i] for i in pick], ignore_index=True)
        boots.append(s[col].median())
    return (float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)))


def analyse(df):
    """Collapse placement -> session -> subject, then test across subjects."""
    ok = df[(df["error"].isna()) | (df["error"] == "")].copy()
    ok["subject_id"] = ok["subject_id"].astype(str)
    if not len(ok):
        # every placement failed -- surface the reason rather than returning
        # empty frames and letting plot() draw blank axes
        errs = df["error"].value_counts()
        raise RuntimeError(
            f"all {len(df)} placements failed. Most common error:\n  "
            + "\n  ".join(f"{n}x  {e}" for e, n in errs.head(3).items()))
    frac_err = 1.0 - len(ok) / len(df)
    if frac_err > 0.10:
        print(f"\n** WARNING: {frac_err:.0%} of placements failed -- the "
              f"medians below are computed on a biased subset **")
        for e, n in df.loc[df["error"].fillna("") != "", "error"] \
                      .value_counts().head(3).items():
            print(f"   {n}x  {e}")

    n_err = int(len(df) - len(ok))
    print(f"\n{'='*70}\nBIAS TERM: DOES IT CLOSE THE GAP?\n{'='*70}")
    print(f"{len(ok)} usable placements, {n_err} errors, "
          f"{ok['subject_id'].nunique()} subjects")

    sess = ok.groupby(["subject_id", "session_number"]).median(
        numeric_only=True).reset_index()
    subj = sess.groupby("subject_id").median(numeric_only=True).reset_index()

    print("\n--- held-out bits/trial (subject medians; lower is better) ---")
    for lab, col in [("base rate", "base_bits"), ("PsyTrack", "glm_bits"),
                     ("MMLPF-5 (current)", "mmlpf5_bits"),
                     ("MMLPF-6 (with bias)", "mmlpf6_bits")]:
        print(f"  {lab:22s} {subj[col].median():.4f}")

    print("\n--- paired gaps (positive favours the MMLPF variant) ---")
    for lab, col in [("MMLPF-5 minus PsyTrack", "diff5_glm"),
                     ("MMLPF-6 minus PsyTrack", "diff6_glm"),
                     ("MMLPF-6 minus MMLPF-5", "diff6_5")]:
        med = subj[col].median()
        lo, hi = _cluster_ci(sess, col)
        w = stats.wilcoxon(subj[col])[1] if len(subj) > 5 else np.nan
        ci = (f"[{lo:+.4f}, {hi:+.4f}]" if np.isfinite(lo)
              else f"[n/a: needs >=5 subjects, have {subj['subject_id'].nunique()}]")
        pstr = f"p={w:.2g}" if np.isfinite(w) else "p=n/a"
        print(f"  {lab:26s} {med:+.4f}  95% CI {ci}  "
              f"{pstr}  better on {(subj[col] > 0).mean():.0%} of subjects")

    print("\n--- does the bias term help most where predicted? ---")
    # side_skew = |P(right) - 0.5|, so HIGH means strongly side-biased. Note the
    # direction is opposite to base_bits, which earlier versions binned on: a
    # quartile labelled "most side-biased" is the TOP skew quartile, not the
    # bottom one.
    sess["q"] = pd.qcut(sess["side_skew"], 4,
                        labels=["Q1 least biased", "Q2", "Q3",
                                "Q4 most biased"])
    print(f"{'bin':>22} {'skew':>7} {'MMLPF6-MMLPF5':>15} "
          f"{'MMLPF6-PsyTrack':>17}")
    for lab, g in sess.groupby("q", observed=True):
        print(f"{str(lab):>22} {g['side_skew'].median():7.3f} "
              f"{g['diff6_5'].median():+15.4f} {g['diff6_glm'].median():+17.4f}")
    rho, p_ = stats.spearmanr(sess["side_skew"], sess["diff6_5"])
    print(f"  rho(side_skew, gain from bias) = {rho:+.3f}  p = {p_:.2g}")
    print("  POSITIVE rho is the prediction: more side-biased -> larger gain")
    rho_b, p_b = stats.spearmanr(sess["base_bits"], sess["diff6_5"])
    print(f"  base_bits for comparison with earlier runs: {rho_b:+.3f}"
          f"  p = {p_b:.2g}  (its sign convention is inverted: low = biased)")

    print("\n--- beta saturation: was beta standing in for bias? ---")
    print(f"  frac of trials with beta > 95, MMLPF-5: "
          f"{sess['frac_beta_clip5'].median():.1%}")
    print(f"  frac of trials with beta > 95, MMLPF-6: "
          f"{sess['frac_beta_clip6'].median():.1%}")
    print(f"  sessions with any beta > 95, MMLPF-5: "
          f"{(sess['max_beta5'] > 95).mean():.1%}")
    print(f"  sessions with any beta > 95, MMLPF-6: "
          f"{(sess['max_beta6'] > 95).mean():.1%}")

    print("\n--- phi: how much was contamination? ---")
    print(f"  median phi, MMLPF-5: {sess['mean_phi5'].median():+.3f}")
    print(f"  median phi, MMLPF-6: {sess['mean_phi6'].median():+.3f}")
    print(f"  shift: {sess['mean_phi6'].median() - sess['mean_phi5'].median():+.3f}")
    print("  A shift means previously reported phi values -- and the")
    print("  construct-validity correlations resting on them -- need restating.")

    print("\n--- is the sigma_bias bound binding? ---")
    # Threshold must track DE_BOUNDS_BIAS, not a literal. It was hardcoded at
    # 0.38 for the old (0.001, 0.4) range; after the bound was tightened to
    # 0.05 that test could never fire, so a binding bound would have gone
    # unreported.
    _bias_hi = float(bc.DE_BOUNDS_BIAS[3][1])
    hi_frac = float((sess["sigma_bias6"] > 0.95 * _bias_hi).mean())
    print(f"  sessions with sigma_bias within 5% of the bound "
          f"({_bias_hi:g}): {hi_frac:.1%}")
    if hi_frac > 0.05:
        print("  ** BOUND IS BINDING: widen DE_BOUNDS_BIAS and refit **")
    print(f"  median fitted sigma_bias: {sess['sigma_bias6'].median():.3f}")

    return sess, subj


def plot(sess, subj, fname="bias_comparison.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(14, 4.0))
    C5, C6, CG, GREY = "#c8511b", "#17a398", "#8a8a8a", "#555555"

    # A: the three models' absolute performance
    labs = ["base\nrate", "PsyTrack", "MMLPF-5\n(current)", "MMLPF-6\n(+bias)"]
    cols = ["base_bits", "glm_bits", "mmlpf5_bits", "mmlpf6_bits"]
    meds = [subj[c].median() for c in cols]
    axA.bar(range(4), meds, color=[GREY, CG, C5, C6],
            edgecolor="white", lw=0.8, width=0.62)
    axA.set_xticks(range(4))
    axA.set_xticklabels(labs, fontsize=6.8)
    axA.set_ylabel("Held-out bits/trial (lower is better)")
    axA.set_title("Absolute predictive performance", loc="left")
    for i, m in enumerate(meds):
        axA.text(i, m + 0.008, f"{m:.3f}", ha="center", fontsize=7)

    # B: per-subject gain from the bias term
    g = subj["diff6_5"].values
    axB.hist(g, bins=20, color=C6, edgecolor="white", lw=0.6)
    axB.axvline(0, color="k", lw=1.2)
    axB.axvline(np.median(g), color=C5, ls="--", lw=1.3)
    axB.set_xlabel("MMLPF-6 minus MMLPF-5 (bits/trial)")
    axB.set_ylabel("Subjects")
    axB.set_title(f"Gain from the bias term\n"
                  f"(better on {(g > 0).mean():.0%} of subjects)", loc="left")
    axB.text(0.98, 0.95, f"median {np.median(g):+.4f}",
             transform=axB.transAxes, ha="right", va="top", fontsize=7)

    # C: gain vs side-lock -- the mechanistic prediction
    axC.scatter(sess["side_skew"], sess["diff6_5"], s=6, color=C6,
                alpha=0.3, lw=0)
    axC.axhline(0, color="k", lw=1.1)
    if len(sess) > 3:
        z = np.polyfit(sess["side_skew"], sess["diff6_5"], 1)
        xs = np.linspace(sess["side_skew"].min(), sess["side_skew"].max(), 50)
        axC.plot(xs, np.polyval(z, xs), color="#222222", ls="--", lw=1.4)
        rho, p_ = stats.spearmanr(sess["side_skew"], sess["diff6_5"])
        # Report the MEASURED direction. The prediction is rho > 0 (more
        # side-biased -> larger gain); hardcoding that as the title would
        # contradict the data whenever the sign comes out the other way.
        if p_ < 0.05 and rho > 0:
            head = "Bias term pays most on\nside-biased sessions"
        elif p_ < 0.05 and rho < 0:
            head = "Bias term pays LEAST on\nside-biased sessions"
        else:
            head = "Side bias does not predict\nthe gain from the bias term"
        axC.set_title(f"{head}\n" + rf"$\rho = {rho:+.2f}$, p = {p_:.2g}",
                      loc="left", fontsize=8)
    axC.set_xlabel(r"Side bias  $|P(\mathrm{right}) - 0.5|$" "\n"
                   "high = strongly one-sided animal")
    axC.set_ylabel("Gain from the bias term (bits/trial)")

    for ax, L in zip((axA, axB, axC), "abc"):
        ax.text(-0.14, 1.06, L, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")
    fig.tight_layout(w_pad=2.5)
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"\nsaved {fname}")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="./cache",
                    help="directory of .npz files from hpc_block_cv.prepare")
    ap.add_argument("--csv", default="bias_comparison.csv")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel placements; if it hangs silently, use 1")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N subjects only (use this to size the run)")
    ap.add_argument("--seeds", type=int, default=SEEDS,
                    help="block placements per session")
    ap.add_argument("--summarize", action="store_true",
                    help="re-analyse an existing CSV without refitting")
    ap.add_argument("--demo", action="store_true",
                    help="simulate a small cohort with KNOWN side bias and run "
                         "on that; needs no database and no cluster cache. "
                         "Checks the pipeline and shows the bias term working "
                         "on data whose truth is known -- says nothing about "
                         "real mice.")
    ap.add_argument("--demo-subjects", type=int, default=4)
    ap.add_argument("--demo-sessions", type=int, default=3)
    ap.add_argument("--max-sessions", type=int, default=None,
                    help="use only the first N sessions per subject. Applied "
                         "after the cache is built, so the cohort is the same "
                         "animals with fewer sessions each -- use this to size "
                         "a laptop run, NOT prepare --sessions, which changes "
                         "which animals qualify.")
    args = ap.parse_args()

    if args.demo:
        cache = write_demo_cache(n_subjects=args.demo_subjects,
                                 n_sessions=args.demo_sessions)
        df = run(cache, out_csv=args.csv, workers=args.workers,
                 limit=None, seeds=args.seeds)
    elif args.summarize:
        df = pd.read_csv(args.csv)
    else:
        df = run(args.cache_dir, out_csv=args.csv, workers=args.workers,
                 limit=args.limit, seeds=args.seeds,
                 max_sessions=args.max_sessions)

    sess, subj = analyse(df)
    if len(sess):
        plot(sess, subj)
        if args.demo:
            _demo_truth_check(sess)


def _demo_truth_check(sess):
    """In demo mode the true bias is known, so check recovery against it."""
    gt_path = os.path.join("./cache_demo", "ground_truth.csv")
    if not os.path.exists(gt_path):
        return
    gt = pd.read_csv(gt_path)
    gt["subject_id"] = gt["subject_id"].astype(str)
    s = sess.groupby("subject_id").median(numeric_only=True).reset_index()
    m = s.merge(gt, on="subject_id", how="inner")
    if len(m) < 2:
        return
    print("\n--- demo only: recovery against the KNOWN simulated bias ---")
    print(f"{'subject':>9} {'true bias':>10} {'recovered':>10} "
          f"{'sigma_bias':>11} {'gain6_5':>9}")
    for r in m.itertuples():
        print(f"{r.subject_id:>9} {r.true_bias:10.2f} {r.mean_bias6:+10.2f} "
              f"{r.sigma_bias6:11.3f} {r.diff6_5:+9.4f}")
    if len(m) >= 3:
        rho = stats.spearmanr(m["true_bias"], m["mean_bias6"])[0]
        rg = stats.spearmanr(m["true_bias"], m["diff6_5"])[0]
        print(f"  rho(true bias, recovered bias) = {rho:+.2f}  "
              f"(should be strongly positive)")
        print(f"  rho(true bias, gain from the term) = {rg:+.2f}  "
              f"(should be positive: more bias -> more to gain)")
    print("  Simulated from the MMLPF's own generative model, so this is a")
    print("  CEILING on recovery -- it is not evidence about real mice.")


if __name__ == "__main__":
    main()
