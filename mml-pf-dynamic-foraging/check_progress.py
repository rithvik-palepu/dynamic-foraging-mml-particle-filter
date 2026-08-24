"""
Progress and health check for a running hpc_block_cv job array.

Reads only the results directory and the cache manifest -- safe to run while the
array is still going, and never touches the database.

    python check_progress.py                        # progress + early results
    python check_progress.py --errors               # also show failure messages
    python check_progress.py --cache-dir ./cache --out-dir ./results

Reports, per subject: placements finished out of expected, and the running
median gap. A subject with 0 rows either has not started (still queued) or its
task died -- check the .out log for that array index.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


def check(cache_dir="./cache", out_dir="./results", seeds=10, show_errors=False):
    mf_path = os.path.join(cache_dir, "manifest.csv")
    if not os.path.exists(mf_path):
        raise FileNotFoundError(f"{mf_path} not found -- run the prepare stage")
    mf = pd.read_csv(mf_path)
    mf["subject_id"] = mf["subject_id"].astype(str)
    expected = dict(zip(mf["subject_id"], mf["n_sessions"] * seeds))
    total_expected = int(sum(expected.values()))

    files = sorted(f for f in glob.glob(os.path.join(out_dir, "*.csv"))
                   if "_old_" not in os.path.basename(f))
    if not files:
        print(f"no result CSVs in {out_dir} yet.")
        print(f"expecting {len(mf)} files, {total_expected} placements total.")
        print("if the array has been running a while, check a .out log -- the "
              "tasks may be failing at import or activation.")
        return None

    frames = []
    for f in files:
        try:
            d = pd.read_csv(f)
            if len(d):
                frames.append(d)
        except pd.errors.EmptyDataError:
            pass          # a task that has written only the header so far
    if not frames:
        print(f"{len(files)} CSVs exist but all are empty -- tasks started but "
              f"no placement has finished yet.")
        return None
    df = pd.concat(frames, ignore_index=True)
    df["subject_id"] = df["subject_id"].astype(str)

    ok = df[(df["error"].isna()) | (df["error"] == "")]
    n_err = len(df) - len(ok)
    done = len(df)
    print(f"{'='*70}")
    print(f"PROGRESS  {done}/{total_expected} placements "
          f"({done/total_expected:.1%})   {len(files)}/{len(mf)} subjects started")
    if n_err:
        print(f"  {n_err} placements recorded an error")
    if len(ok):
        med_s = ok["seconds"].median()
        remaining = total_expected - done
        print(f"  median {med_s:.0f}s per placement; {remaining} remaining")
        print(f"  (wall time depends on how many array tasks are running "
              f"concurrently)")
    print(f"{'='*70}")

    per = (df.groupby("subject_id")
           .agg(rows=("seed", "size"),
                med_diff=("diff_bits", "median"),
                med_sec=("seconds", "median"))
           .reindex(mf["subject_id"]))
    per["expected"] = mf["subject_id"].map(expected).values
    per["rows"] = per["rows"].fillna(0).astype(int)
    per["pct"] = (per["rows"] / per["expected"] * 100).round(0)
    print("\nper subject (blank med_diff = nothing finished yet):")
    print(per[["rows", "expected", "pct", "med_diff", "med_sec"]]
          .round(4).to_string())

    not_started = per[per["rows"] == 0]
    if len(not_started):
        idx = {s: i for i, s in enumerate(mf["subject_id"])}
        print(f"\n{len(not_started)} subjects with no rows yet -- array indices: "
              f"{sorted(idx[s] for s in not_started.index)}")
        print("  queued tasks look identical to dead ones here; confirm with "
              "squeue/sacct.")

    if len(ok) >= 5:
        dd = ok.groupby("subject_id")["diff_bits"].median()
        complete = per[per["rows"] >= per["expected"]].index
        dd_c = dd.reindex(complete).dropna()
        print(f"\n--- provisional, {len(dd)} subjects with any data "
              f"({len(dd_c)} complete) ---")
        print(f"MMLPF better on {int((dd > 0).sum())}/{len(dd)} subjects so far")
        print(f"median gap {dd.median():+.4f} bits/trial")
        blown = int((ok["mmlpf_bits"] > ok["base_bits"]).sum())
        print(f"placements where MMLPF scored worse than the base rate: "
              f"{blown}/{len(ok)}")
        print("These are PROVISIONAL. Partial subjects are biased by which "
              "placements\nhappened to finish first -- use `combine` once the "
              "array is done.")

    if show_errors and n_err:
        print(f"\n--- errors ---")
        e = df[(df["error"].notna()) & (df["error"] != "")]
        for msg, grp in e.groupby("error"):
            print(f"  {len(grp)}x  {msg[:160]}")
            print(f"        subjects: {sorted(grp['subject_id'].unique())[:6]}")
    return per


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="./cache")
    ap.add_argument("--out-dir", default="./results")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--errors", action="store_true")
    args = ap.parse_args()
    check(args.cache_dir, args.out_dir, seeds=args.seeds,
          show_errors=args.errors)


if __name__ == "__main__":
    main()
