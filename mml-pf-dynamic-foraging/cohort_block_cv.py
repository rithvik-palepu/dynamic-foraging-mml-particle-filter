"""
Cohort-wide interleaved-block cross-validation: MMLPF (+ perseveration) vs
PsyTrack, every qualifying subject, no selection on outcome.

WHY THIS EXISTS
---------------
The two-subject block-CV run used 713379 and 751766. Both were chosen from the
walk-forward results: 713379 had the smallest deficit in the cohort and 751766
was picked as a clean-session example of the MMLPF losing. That is selection on
the outcome variable. Whatever the block CV then says about those two is
conditioned on how they scored under the previous design, so it cannot be
quoted as a cohort result -- the favourable subject was favourable by
construction.

This runs the identical procedure over every subject that passes COHORT_QUERY,
with no reference to any previous score. It also reports where the two
hand-picked mice fall within the resulting distribution, which is the honest
way to show how much the selected pair overstated the case.

DESIGN
------
Unit of replication is the SUBJECT, not the placement and not the session.
Placements within a session differ only by an arbitrary random mask, and
sessions within a subject are repeated measures on the same animal; treating
either as independent would inflate n by an order of magnitude and produce a
p-value that describes the block-placement RNG rather than the cohort.

    placement -> per-(subject, session) MEDIAN across placements
              -> per-subject MEDIAN across sessions
              -> across-subject paired test, n = number of subjects

Median rather than mean at both inner levels: on some animals the particle
filter destabilises on particular block placements and scores worse than the
base rate, and a single such placement moves a mean by more than the effect
being measured.

Usage
-----
    python cohort_block_cv.py --sessions 5 --seeds 10 --workers 8
    python cohort_block_cv.py --summarize          # replot/re-summarise the CSV

Runtime is dominated by the MMLPF M-step (~30 s per placement). Subjects x
sessions x seeds placements; 10 x 5 x 10 = 500 is roughly half an hour on 8
workers. The CSV is appended per placement and completed work is skipped on
restart, so the run is safe to interrupt.
"""

import argparse
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import block_cv as bc
import mmlpf_vs_psytrack_cv as cv

RESULT_COLUMNS = [
    "subject_id", "session_number", "seed", "n_trials", "n_test", "block",
    "base_bits", "glm_bits", "mmlpf_bits", "diff_bits",
    "glm_sigma", "sigma_alpha", "sigma_beta", "sigma_phi",
    "mean_beta", "max_beta", "mean_phi",
    "mmlpf_loss_conc", "mmlpf_worst_bits", "glm_loss_conc", "glm_worst_bits",
    "seconds", "error",
]

# the pair used in the two-subject run, recorded so the summary can locate them
# inside the unselected distribution
PRESELECTED = ["713379", "751766"]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_cohort(min_sessions, max_sessions, min_trials):
    """Every subject passing COHORT_QUERY with enough sessions. No outcome filter."""
    import aind_dynamic_foraging_database as db

    sessions = db.select_sessions(where=cv.COHORT_QUERY).sort_values(
        by=["subject_id", "session_date"])
    counts = sessions["subject_id"].value_counts()
    subjects = sorted(str(s) for s in counts[counts >= min_sessions].index)

    cohort = {}
    for sid in subjects:
        subj = sessions[sessions["subject_id"].astype(str) == sid].head(max_sessions)
        trials = db.fetch_trials(subj, columns=["animal_response", "earned_reward"])
        blocks = []
        for _key, g in trials.groupby(["session_date", "session_id"], sort=True):
            v = g[g["animal_response"] != 2]
            if len(v) >= min_trials:
                blocks.append((v["animal_response"].astype(int).values,
                               v["earned_reward"].astype(int).values))
        if len(blocks) >= 1:
            cohort[sid] = blocks
    return cohort


# ---------------------------------------------------------------------------
# one placement
# ---------------------------------------------------------------------------
def _worker(job):
    """Module-level so multiprocessing can pickle it."""
    # each worker is single-threaded: the parallelism is across placements, and
    # letting BLAS also spawn threads per worker oversubscribes the cores
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"

    sid, sess_i, seed, choices, rewards, block, test_frac = job
    row = {c: np.nan for c in RESULT_COLUMNS}
    row.update(subject_id=sid, session_number=sess_i, seed=seed, error="")
    t0 = time.time()
    try:
        res, _ = bc.score_session(choices, rewards, block=block,
                                  test_frac=test_frac, seed=seed,
                                  smooth=False, workers=1, verbose=False)
        row.update({k: v for k, v in res.items() if k in RESULT_COLUMNS})
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    row["seconds"] = round(time.time() - t0, 1)
    return row


def run_cohort(out_csv, sessions_per_subject=5, seeds=10, block=20,
               test_frac=0.2, workers=1, min_sessions=5, min_trials=200,
               limit=None):
    cohort = load_cohort(min_sessions, sessions_per_subject, min_trials)
    if not cohort:
        raise RuntimeError(
            "no subjects returned: check COHORT_QUERY and that subjects have "
            f">= {min_sessions} sessions of >= {min_trials} valid trials")
    if limit:
        cohort = {k: cohort[k] for k in sorted(cohort)[:limit]}

    n_sess = sum(len(v) for v in cohort.values())
    print(f"{len(cohort)} subjects, {n_sess} sessions, {seeds} placements each "
          f"= {n_sess * seeds} fits")

    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        missing = [c for c in RESULT_COLUMNS if c not in prev.columns]
        if missing:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            backup = out_csv.replace(".csv", f"_old_{stamp}.csv")
            os.rename(out_csv, backup)
            print(f"existing CSV was written by an older version (missing "
                  f"{missing}); moved to {backup} and starting fresh")
        else:
            done = {(str(r.subject_id), int(r.session_number), int(r.seed))
                    for r in prev.itertuples()}
            print(f"resuming: {len(done)} placements already in {out_csv}")

    jobs = []
    for sid, blocks in sorted(cohort.items()):
        for i, (c, r) in enumerate(blocks, start=1):
            for s in range(seeds):
                if (sid, i, s) not in done:
                    jobs.append((sid, i, s, c, r, block, test_frac))
    if not jobs:
        print("nothing to do")
        return pd.read_csv(out_csv)
    print(f"{len(jobs)} placements to fit")

    def _append(row):
        pd.DataFrame([row], columns=RESULT_COLUMNS).to_csv(
            out_csv, mode="a", header=not os.path.exists(out_csv), index=False)

    t_start = time.time()
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for n, row in enumerate(pool.imap_unordered(_worker, jobs), 1):
                _append(row)
                el = time.time() - t_start
                print(f"[{n}/{len(jobs)}] {row['subject_id']} s{row['session_number']} "
                      f"seed {row['seed']}: "
                      f"{row['error'] or f'''diff {row['diff_bits']:+.4f}'''}  "
                      f"({row['seconds']}s, eta {el / n * (len(jobs) - n) / 60:.0f} min)",
                      flush=True)
    else:
        for n, job in enumerate(jobs, 1):
            row = _worker(job)
            _append(row)
            el = time.time() - t_start
            print(f"[{n}/{len(jobs)}] {row['subject_id']} s{row['session_number']} "
                  f"seed {row['seed']}: "
                  f"{row['error'] or f'''diff {row['diff_bits']:+.4f}'''}  "
                  f"({row['seconds']}s, eta {el / n * (len(jobs) - n) / 60:.0f} min)",
                  flush=True)

    return pd.read_csv(out_csv)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def summarize(df):
    from scipy import stats

    ok = df[(df["error"].isna()) | (df["error"] == "")].copy()
    ok["subject_id"] = ok["subject_id"].astype(str)
    n_err = len(df) - len(ok)

    print(f"\n{'=' * 74}")
    print(f"COHORT BLOCK-CV  ({ok['subject_id'].nunique()} subjects, "
          f"{len(ok)} placements, {n_err} failed)")
    print("=" * 74)

    # placement -> session -> subject
    per_sess = (ok.groupby(["subject_id", "session_number"])
                .agg(diff=("diff_bits", "median"),
                     glm=("glm_bits", "median"),
                     mmlpf=("mmlpf_bits", "median"),
                     base=("base_bits", "median"),
                     mm_sd=("mmlpf_bits", "std"),
                     glm_sd=("glm_bits", "std"),
                     n_blown=("mmlpf_bits", lambda s: np.nan))
                .reset_index())
    blown = (ok.assign(b=ok["mmlpf_bits"] > ok["base_bits"])
             .groupby(["subject_id", "session_number"])["b"].sum().values)
    per_sess["n_blown"] = blown

    per_subj = (per_sess.groupby("subject_id")
                .agg(diff=("diff", "median"), glm=("glm", "median"),
                     mmlpf=("mmlpf", "median"), base=("base", "median"),
                     n_sessions=("diff", "size"), blown=("n_blown", "sum"))
                .sort_values("diff", ascending=False))
    var_ratio = (ok.groupby("subject_id")
                 .apply(lambda s: (s["mmlpf_bits"].std() / s["glm_bits"].std()) ** 2,
                        include_groups=False))
    per_subj["var_ratio"] = var_ratio.reindex(per_subj.index).values

    print("\nper-subject (median over sessions of median over placements):")
    print(per_subj.round(4).to_string())

    dd = per_subj["diff"].values
    n = len(dd)
    wins = int((dd > 0).sum())
    print(f"\n{'-' * 74}")
    print(f"MMLPF better on {wins}/{n} subjects")
    print(f"median gap {np.median(dd):+.4f} bits/trial   "
          f"mean {dd.mean():+.4f}  sd {dd.std(ddof=1):.4f}")
    print(f"IQR [{np.percentile(dd, 25):+.4f}, {np.percentile(dd, 75):+.4f}]")
    if n >= 3:
        t = stats.ttest_1samp(dd, 0.0)
        w = stats.wilcoxon(dd)
        print(f"paired t across subjects  t = {t.statistic:.2f}, p = {t.pvalue:.4g}")
        print(f"Wilcoxon signed-rank      p = {w.pvalue:.4g}")
        # effect size in interpretable units, per the bits/trial convention
        med_glm = per_subj["glm"].median()
        med_base = per_subj["base"].median()
        med_mm = per_subj["mmlpf"].median()
        print(f"\nbase {med_base:.3f}  PsyTrack {med_glm:.3f}  "
              f"MMLPF {med_mm:.3f} bits/trial (cohort medians)")
        if med_glm < med_base:
            frac = (med_base - med_mm) / (med_base - med_glm)
            # >100% means the MMLPF explains MORE than PsyTrack; the "fraction
            # of PsyTrack's structure" framing only reads sensibly below 1
            if frac <= 1.0:
                print(f"MMLPF captures {frac:.1%} of the structure PsyTrack "
                      f"does")
            else:
                print(f"MMLPF explains MORE than PsyTrack "
                      f"({frac:.1%} of its structure) -- report the gap "
                      f"directly, not as a fraction")
        # median gap is GLM minus MMLPF, so a POSITIVE gap favours the MMLPF
        gap = float(np.median(dd))
        lr = 2 ** abs(gap)
        ahead = "MMLPF" if gap > 0 else "PsyTrack"
        print(f"per-trial likelihood ratio 2^|gap| = {lr:.3f}  "
              f"({ahead} assigns that factor more probability to the observed "
              f"choice)")

    # stability: how common is the 751766-style failure?
    unstable = per_subj[(per_subj["var_ratio"] > 4) | (per_subj["blown"] > 0)]
    print(f"\n{'-' * 74}")
    print(f"filter stability: {len(unstable)}/{n} subjects show MMLPF variance "
          f">4x the GLM's\n  or at least one placement scoring worse than the "
          f"base rate")
    if len(unstable):
        print(unstable[["diff", "var_ratio", "blown", "n_sessions"]].round(3).to_string())
        print("  on these animals the MEDIAN is the only defensible summary; a "
              "mean is set by\n  the failed placements rather than by the model")

    # where did the hand-picked pair land?
    present = [s for s in PRESELECTED if s in per_subj.index]
    if present:
        print(f"\n{'-' * 74}")
        print("the two hand-picked subjects, located in the unselected distribution:")
        for sid in present:
            v = per_subj.loc[sid, "diff"]
            pct = float((dd < v).mean())
            print(f"  {sid}: diff {v:+.4f}  -> {pct:.0%} percentile of the cohort")
        print("  a subject chosen because it scored well under the previous "
              "design will sit\n  high here by construction; the cohort median "
              "is the number to quote")
    return per_subj


def plot_cohort(per_subj, ok, fname="cohort_block_cv.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_GLM, C_MM, GREY = "#c8511b", "#17a398", "#8a8a8a"
    s = per_subj.sort_values("diff")
    n = len(s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    cols = [GREY if b > 0 else (C_MM if v > 0 else C_GLM)
            for v, b in zip(s["diff"], s["blown"])]
    ax1.barh(np.arange(n), s["diff"].values, color=cols, height=0.75)
    ax1.axvline(0, color="k", lw=1.0)
    ax1.axvline(s["diff"].median(), color=C_MM, ls="--", lw=1.2)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(s.index, fontsize=6.5)
    ax1.set_xlabel("GLM minus MMLPF (bits/trial)")
    ax1.set_ylabel("Subject")
    ax1.set_title(f"MMLPF better on {int((s['diff'] > 0).sum())}/{n} subjects",
                  loc="left")
    ax1.text(0.985, 0.02, "right = MMLPF better", transform=ax1.transAxes,
             ha="right", fontsize=6.5, color=GREY)
    if (s["blown"] > 0).any():
        ax1.text(0.02, 0.98, "grey = filter destabilised\non >=1 placement",
                 transform=ax1.transAxes, va="top", fontsize=6.5, color=GREY)

    x = np.arange(n)
    ax2.plot(x, s["base"].values, "o-", color=GREY, ms=3, lw=1.0,
             label="base rate")
    ax2.plot(x, s["glm"].values, "o-", color=C_GLM, ms=3, lw=1.2, label="PsyTrack")
    ax2.plot(x, s["mmlpf"].values, "o-", color=C_MM, ms=3, lw=1.2,
             label="MMLPF + persev.")
    ax2.set_xticks(x)
    ax2.set_xticklabels(s.index, rotation=90, fontsize=6.5)
    ax2.set_ylabel("Held-out NLL (bits/trial)")
    ax2.set_xlabel("Subject")
    ax2.set_title("Both models against the base rate", loc="left")
    ax2.legend(frameon=False, fontsize=7)
    ax2.text(0.015, 0.03, "lower = better", transform=ax2.transAxes,
             fontsize=6.5, color=GREY)

    for ax, L in zip((ax1, ax2), "ab"):
        ax.text(-0.12, 1.02, L, transform=ax.transAxes, fontsize=12,
                fontweight="bold", va="bottom")
    fig.tight_layout(w_pad=2.4)
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"\nsaved {fname}")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="cohort_block_cv.csv")
    ap.add_argument("--sessions", type=int, default=5,
                    help="sessions per subject")
    ap.add_argument("--seeds", type=int, default=10,
                    help="block placements per session")
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--min-sessions", type=int, default=5)
    ap.add_argument("--min-trials", type=int, default=200)
    ap.add_argument("--limit", type=int, default=10,
                    help="use the first N qualifying subjects (alphabetical by "
                         "subject_id, so the selection is independent of how "
                         "any subject scored). Pass 0 for the whole cohort")
    ap.add_argument("--summarize", action="store_true",
                    help="skip fitting; summarise and replot an existing CSV")
    args = ap.parse_args()

    if args.summarize:
        df = pd.read_csv(args.csv)
    else:
        df = run_cohort(args.csv, sessions_per_subject=args.sessions,
                        seeds=args.seeds, block=args.block,
                        test_frac=args.test_frac, workers=args.workers,
                        min_sessions=args.min_sessions,
                        min_trials=args.min_trials,
                        limit=(args.limit or None))

    ok = df[(df["error"].isna()) | (df["error"] == "")].copy()
    ok["subject_id"] = ok["subject_id"].astype(str)
    per_subj = summarize(df)
    plot_cohort(per_subj, ok)


if __name__ == "__main__":
    main()
