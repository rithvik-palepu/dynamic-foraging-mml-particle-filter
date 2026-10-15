"""
Does modelling the ignored trials help?

Compares two MMLPF variants, fit and scored on identical block-CV masks:

    MMLPF-6   alpha, beta, phi, drifting bias            (current best)
    MMLPF-7   the same, plus a drifting engagement state theta

and PsyTrack as the external reference.

WHY AN ENGAGEMENT STATE
-----------------------
Every analysis in this project so far DROPS trials where the animal did not
respond (animal_response == 2). In the sessions we looked at those are 8-24% of
trials, and they arrive in runs of up to 32 consecutive trials where chance
predicts 2-4 -- disengagement is sustained, not scattered. A model with no way
to represent it has to explain those stretches through the states it does have,
which is the suspected reason sigma_beta pins against its search bound on
high-ignore sessions.

MMLPF-7 adds theta_t = P(the animal answers trial t). The observation becomes
two-stage:

    ignored trial:   P = 1 - theta_t
    choice c:        P = theta_t * p_choice(c)

theta is the only latent in this model with its own observation channel. alpha,
beta and phi are inferred solely from how consistent the choices look, which is
why they came out weakly identified; theta is inferred from observed events.
That is also why its volatility gets a wide search bound while the bias state
gets a narrow one -- see DE_BOUNDS_ENGAGE in block_cv.py.

SCORING -- THE POINT TO GET RIGHT
---------------------------------
MMLPF-7 assigns probability to a strictly larger outcome space than MMLPF-6 or
PsyTrack (three outcomes per trial rather than two). Comparing total likelihoods
would therefore compare different quantities and MMLPF-7 would look worse purely
from carrying an extra channel.

So all three models are scored on the SAME thing: the two-alternative
conditional choice likelihood, evaluated on held-out trials where the animal
DID respond. For MMLPF-7 that is p_choice, not theta * p_choice. The engagement
channel is reported separately as a diagnostic (`engage_bits`), never folded
into the comparison.

TRIAL ALIGNMENT
---------------
MMLPF-7 needs the ignored trials in place, so sessions are loaded RAW and the
ignore mask is carried alongside. MMLPF-6 and PsyTrack cannot use them, so they
receive the same session with the ignored trials removed -- which is exactly
current practice. Held-out blocks are drawn on the raw trial axis and then
projected onto the compressed axis, so both models are tested on the same
animal behaviour even though their trial vectors differ in length.

USAGE
-----
    # recovery on synthetic agents with known states, including known
    # engagement blocks (no database needed)
    python engagement_comparison.py recovery --seeds 6

    # 5 random subjects from the public database
    python engagement_comparison.py real --subjects 5 --seeds 3

    python engagement_comparison.py real --summarize      # re-analyse a CSV
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

import block_cv as bc                                    # noqa: E402
from block_cv import build_glm_regressors       # noqa: E402

M_PARTICLES = 200        # inside the volatility fit
E_PARTICLES = 1500       # for the scored forward pass
BLOCK = 20
TEST_FRAC = 0.2
N_LAGS = 2
MIN_TRIALS = 200

RESULT_COLUMNS = [
    "label", "subject_id", "session_number", "seed",
    "n_raw", "n_valid", "ignored_frac", "ignore_run_mean", "ignore_run_max",
    "base_bits", "glm_bits", "mmlpf6_bits", "mmlpf7_bits",
    "diff7_6", "diff6_glm", "diff7_glm",
    "engage_bits", "engage_base_bits",
    "sa6", "sb6", "sp6", "sbias6",
    "sa7", "sb7", "sp7", "sbias7", "sengage7",
    "mean_beta6", "max_beta6", "mean_beta7", "max_beta7",
    "mean_phi6", "mean_phi7", "mean_theta7", "min_theta7",
    "seconds", "error",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def bits(p, y, mask):
    p = np.clip(np.asarray(p)[mask], 1e-9, 1 - 1e-9)
    y = np.asarray(y)[mask]
    return float(-np.mean(y * np.log2(p) + (1 - y) * np.log2(1 - p)))


def run_lengths(mask):
    """Lengths of consecutive-True runs; [0] when there are none."""
    out, n = [], 0
    for v in mask:
        if v:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return np.array(out) if out else np.array([0])


def project_mask(train_mask_raw, ignored):
    """Raw-axis train mask -> compressed axis (ignored trials removed).

    Held-out BLOCKS are defined once on the raw trial axis so that both models
    withhold the same stretch of the animal's behaviour. Dropping the ignored
    trials shortens the vector the 6-state model sees, so the mask has to be
    projected rather than regenerated -- regenerating it would hold out
    different trials for each model and the comparison would be unmatched.
    """
    return np.asarray(train_mask_raw)[~np.asarray(ignored, dtype=bool)]


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------
def fit6(choices, rewards, train_mask, maxiter, popsize, seed):
    def obj(v):
        return bc.mmlpf_masked(v[0], v[1], v[2], choices, rewards, train_mask,
                               num_particles=M_PARTICLES, seed=seed,
                               sigma_bias=v[3])[0]
    res = differential_evolution(obj, bounds=bc.DE_BOUNDS_BIAS,
                                 maxiter=maxiter, popsize=popsize,
                                 tol=0.01, seed=0, polish=False)
    sig = tuple(float(v) for v in res.x)
    out = bc.mmlpf_masked(*sig[:3], choices, rewards, train_mask,
                          num_particles=E_PARTICLES, seed=seed,
                          sigma_bias=sig[3], collect=True)
    return sig, out


def fit7(choices, rewards, ignored, train_mask, maxiter, popsize, seed):
    def obj(v):
        return bc.mmlpf_masked(v[0], v[1], v[2], choices, rewards, train_mask,
                               num_particles=M_PARTICLES, seed=seed,
                               sigma_bias=v[3], sigma_engage=v[4],
                               ignored=ignored)[0]
    res = differential_evolution(obj, bounds=bc.DE_BOUNDS_ENGAGE,
                                 maxiter=maxiter, popsize=popsize,
                                 tol=0.01, seed=0, polish=False)
    sig = tuple(float(v) for v in res.x)
    out = bc.mmlpf_masked(*sig[:3], choices, rewards, train_mask,
                          num_particles=E_PARTICLES, seed=seed,
                          sigma_bias=sig[3], sigma_engage=sig[4],
                          ignored=ignored, collect=True)
    return sig, out


def score_session(choices_raw, rewards_raw, ignored, seed,
                  maxiter=7, popsize=4, label="", subject_id="",
                  session_number=0):
    """Fit and score both variants plus PsyTrack on one block-CV placement."""
    t0 = time.time()
    ignored = np.asarray(ignored, dtype=bool)
    T_raw = len(choices_raw)
    keep = ~ignored
    c_v = np.asarray(choices_raw)[keep].astype(int)
    r_v = np.asarray(rewards_raw)[keep].astype(int)

    row = {k: np.nan for k in RESULT_COLUMNS}
    rl = run_lengths(ignored)
    row.update(label=label, subject_id=subject_id,
               session_number=session_number, seed=seed,
               n_raw=T_raw, n_valid=int(keep.sum()),
               ignored_frac=float(ignored.mean()),
               ignore_run_mean=float(rl.mean()), ignore_run_max=int(rl.max()),
               error="")

    if keep.sum() < MIN_TRIALS:
        row["error"] = f"only {int(keep.sum())} answered trials"
        row["seconds"] = round(time.time() - t0, 1)
        return row

    # one set of held-out blocks on the RAW axis, projected for the models that
    # never see the ignored trials
    tm_raw = bc.make_block_mask(T_raw, block=BLOCK, test_frac=TEST_FRAC,
                                seed=seed)
    tm_v = project_mask(tm_raw, ignored)
    test_v = ~tm_v
    if test_v.sum() < 20:
        row["error"] = f"only {int(test_v.sum())} answered held-out trials"
        row["seconds"] = round(time.time() - t0, 1)
        return row

    # --- PsyTrack, on answered trials (its only option) ---
    rew, unrew = build_glm_regressors(c_v, r_v, np.zeros(len(c_v), dtype=int),
                                      n_lags=N_LAGS)
    X = np.column_stack([np.ones(len(c_v)), rew, unrew])
    glm_sigma = bc.fit_glm_sigma(X, c_v, tm_v)
    glm_p, _ = bc.glm_masked(X, c_v, glm_sigma, tm_v)

    # --- MMLPF-6: ignored trials dropped, i.e. current practice ---
    s6, o6 = fit6(c_v, r_v, tm_v, maxiter, popsize, seed)

    # --- MMLPF-7: ignored trials modelled ---
    s7, o7 = fit7(np.asarray(choices_raw).astype(int),
                  np.asarray(rewards_raw).astype(int),
                  ignored, tm_raw, maxiter, popsize, seed)

    # MMLPF-7's p_right lives on the RAW axis; compress to the answered axis so
    # every model is scored on the same trials
    p7_v = np.asarray(o7["p_right"])[keep]

    base_p = np.full(len(c_v), float(c_v[tm_v].mean()))
    row.update(
        base_bits=bits(base_p, c_v, test_v),
        glm_bits=bits(glm_p, c_v, test_v),
        mmlpf6_bits=bits(o6["p_right"], c_v, test_v),
        mmlpf7_bits=bits(p7_v, c_v, test_v),
        sa6=s6[0], sb6=s6[1], sp6=s6[2], sbias6=s6[3],
        sa7=s7[0], sb7=s7[1], sp7=s7[2], sbias7=s7[3], sengage7=s7[4],
        mean_beta6=float(np.median(o6["beta"])),
        max_beta6=float(np.max(o6["beta"])),
        mean_beta7=float(np.median(o7["beta"])),
        max_beta7=float(np.max(o7["beta"])),
        mean_phi6=float(np.median(o6["phi"])),
        mean_phi7=float(np.median(o7["phi"])),
        mean_theta7=float(np.mean(o7["engage"])),
        min_theta7=float(np.min(o7["engage"])),
    )
    row["diff7_6"] = row["mmlpf6_bits"] - row["mmlpf7_bits"]
    row["diff6_glm"] = row["glm_bits"] - row["mmlpf6_bits"]
    row["diff7_glm"] = row["glm_bits"] - row["mmlpf7_bits"]

    # engagement channel, reported but NOT part of the comparison: how well
    # theta predicts answered-vs-ignored on held-out raw trials, against the
    # constant-rate predictor fit on training trials
    test_raw = ~tm_raw
    answered = (~ignored).astype(int)
    row["engage_bits"] = bits(o7["engage"], answered, test_raw)
    row["engage_base_bits"] = bits(
        np.full(T_raw, float(answered[tm_raw].mean())), answered, test_raw)

    row["seconds"] = round(time.time() - t0, 1)
    return row


# --------------------------------------------------------------------------
# synthetic recovery
# --------------------------------------------------------------------------
def simulate(T=600, alpha=0.35, beta=6.0, phi=1.5, bias=1.5,
             engage_blocks=((120, 160), (330, 362), (470, 495)),
             theta_engaged=0.97, theta_disengaged=0.15,
             block=60, lapse=0.0, seed=0):
    """Q-learner with a constant side bias and SUSTAINED disengagement blocks.

    All four decision states are constant, so any apparent drift the filter
    reports is spurious. Engagement is a two-level square wave, matching the
    run-length structure measured in the real sessions (long runs, abrupt
    onsets) rather than independent coin-flip lapses.

    Returns choices, rewards, ignored, and the true theta trajectory. Choices on
    ignored trials are filled with -1 so that any code path which forgets to
    respect the ignore mask fails loudly instead of silently treating them as
    left choices.
    """
    rng = np.random.default_rng(seed)
    Q = np.array([0.5, 0.5])
    prev = 0.0
    p = np.array([0.8, 0.1])
    theta_true = np.full(T, theta_engaged)
    for lo, hi in engage_blocks:
        theta_true[lo:hi] = theta_disengaged

    ch, rw, ign = [], [], []
    for t in range(T):
        if t and t % block == 0:
            p = p[::-1].copy()
        if rng.random() > theta_true[t]:
            ch.append(-1)
            rw.append(0)
            ign.append(True)
            continue
        z = beta * (Q[1] - Q[0]) + phi * prev + bias
        pr = (1 - lapse) * expit(np.clip(z, -50, 50)) + lapse / 2
        c = int(rng.random() < pr)
        r = int(rng.random() < p[c])
        Q[c] += alpha * (r - Q[c])
        prev = 1.0 if c == 1 else -1.0
        ch.append(c)
        rw.append(r)
        ign.append(False)
    return (np.array(ch), np.array(rw), np.array(ign, dtype=bool),
            theta_true)


TRUTH = dict(alpha=0.35, beta=6.0, phi=1.5, bias=1.5)


def recovery(seeds=6, trials=600, maxiter=7, popsize=4,
             out_csv="engagement_recovery.csv"):
    """Do the two models recover alpha, beta, phi, bias -- and theta?"""
    rows = []
    for s in range(seeds):
        c_raw, r_raw, ign, th_true = simulate(T=trials, seed=500 + s)
        keep = ~ign
        c_v, r_v = c_raw[keep].astype(int), r_raw[keep].astype(int)
        tm_raw = bc.make_block_mask(len(c_raw), block=BLOCK,
                                    test_frac=TEST_FRAC, seed=s)
        tm_v = project_mask(tm_raw, ign)

        s6, o6 = fit6(c_v, r_v, tm_v, maxiter, popsize, s)
        s7, o7 = fit7(c_raw.astype(int), r_raw.astype(int), ign, tm_raw,
                      maxiter, popsize, s)

        rec = dict(seed=s, ignored_frac=float(ign.mean()))
        for nm, o, tag in ((6, o6, "6"), (7, o7, "7")):
            rec[f"alpha{tag}"] = float(np.median(o["alpha"]))
            rec[f"beta{tag}"] = float(np.median(o["beta"]))
            rec[f"phi{tag}"] = float(np.median(o["phi"]))
            rec[f"bias{tag}"] = float(np.median(o["bias"]))
        # theta recovery: correlation of the estimated trajectory with truth,
        # and separation between disengaged and engaged stretches
        th_hat = np.asarray(o7["engage"])
        rec["theta_r"] = float(np.corrcoef(th_hat, th_true)[0, 1])
        rec["theta_in"] = float(th_hat[th_true < 0.5].mean())
        rec["theta_out"] = float(th_hat[th_true > 0.5].mean())
        rec["sengage7"] = s7[4]
        rec["sb6"] = s6[1]
        rec["sb7"] = s7[1]
        rows.append(rec)
        print(f"  seed {s}: beta6 {rec['beta6']:6.2f}  beta7 {rec['beta7']:6.2f}"
              f"  (true 6.0)   theta r {rec['theta_r']:+.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    report_recovery(df)
    return df


def report_recovery(df):
    print("\n" + "=" * 70)
    print("PARAMETER RECOVERY ON SYNTHETIC AGENTS WITH KNOWN STATES")
    print("=" * 70)
    print(f"{len(df)} seeds, {df['ignored_frac'].mean():.1%} of trials ignored "
          f"(sustained blocks)\n")
    print(f"{'state':>8} {'truth':>8} {'MMLPF-6':>10} {'MMLPF-7':>10} "
          f"{'|err| 6':>9} {'|err| 7':>9}")
    for nm, tv in TRUTH.items():
        m6, m7 = df[f"{nm}6"].median(), df[f"{nm}7"].median()
        print(f"{nm:>8} {tv:8.2f} {m6:10.2f} {m7:10.2f} "
              f"{abs(m6 - tv):9.2f} {abs(m7 - tv):9.2f}")
    print(f"\n{'theta':>8} {'--':>8} {'not modelled':>10} "
          f"{df['theta_r'].median():10.2f}   <- correlation with true theta")
    print(f"  theta in disengaged blocks {df['theta_in'].median():.3f} "
          f"(truth 0.15)   engaged {df['theta_out'].median():.3f} (truth 0.97)")
    print(f"  fitted sigma_engage: median {df['sengage7'].median():.3f} "
          f"(bound {bc.DE_BOUNDS_ENGAGE[4][1]})")
    print(f"\nsigma_beta: MMLPF-6 {df['sb6'].median():.3f}   "
          f"MMLPF-7 {df['sb7'].median():.3f}   "
          f"(bound {bc.DE_BOUNDS_BIAS[1][1]})")
    print("  If the 6-state model is absorbing disengagement through beta, its")
    print("  sigma_beta should be the larger of the two.")


# --------------------------------------------------------------------------
# real subjects
# --------------------------------------------------------------------------
def load_raw_sessions(subject_id, n_sessions=3, query=None):
    """RAW sessions for one subject, ignored trials KEPT.

    Every other loader in this project drops animal_response == 2; this one
    must not, because the ignore events are the engagement model's data.
    """
    import aind_dynamic_foraging_database as db
    import mmlpf_vs_psytrack_cv as cv

    q = query or cv.COHORT_QUERY
    sel = db.select_sessions(q, subjects=[str(subject_id)])
    if not len(sel):
        return []
    sel = sel.sort_values("session_date")
    trials = db.fetch_trials(sel, columns=["animal_response", "earned_reward"])
    out = []
    for i, (_, g) in enumerate(
            trials.groupby(["session_date", "session_id"], sort=True), 1):
        if len(out) >= n_sessions:
            break
        g = g.reset_index(drop=True)
        resp = g["animal_response"].values
        ign = (resp == 2)
        if (~ign).sum() < MIN_TRIALS:
            continue
        ch = np.where(ign, -1, resp).astype(int)
        rw = g["earned_reward"].values.astype(int)
        out.append((i, ch, rw, ign))
    return out


def pick_subjects(n=5, seed=0, query=None, min_sessions=3):
    """n subjects drawn at RANDOM from those passing the cohort query.

    Random rather than alphabetical because the user asked for random. Either is
    defensible -- both are independent of any model score, which is the property
    that matters -- but the seed is recorded so the draw is reproducible.
    """
    import aind_dynamic_foraging_database as db
    import mmlpf_vs_psytrack_cv as cv

    q = query or cv.COHORT_QUERY
    sel = db.select_sessions(q)
    counts = sel["subject_id"].astype(str).value_counts()
    eligible = sorted(counts[counts >= min_sessions].index)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(eligible), size=min(n, len(eligible)), replace=False)
    chosen = [eligible[i] for i in sorted(pick)]
    print(f"{len(eligible)} subjects have >= {min_sessions} qualifying "
          f"sessions; drew {len(chosen)} at random (seed {seed}):")
    print("  " + ", ".join(chosen))
    return chosen


def run_real(n_subjects=5, sessions=3, seeds=3, pick_seed=0, maxiter=7,
             popsize=4, out_csv="engagement_real.csv", query=None,
             subject_ids=None):
    """A random draw is the default; subject_ids overrides it.

    Naming subjects explicitly is selection on a covariate, not on an outcome,
    so it is legitimate for a targeted mechanism test -- e.g. restricting to
    high-ignore sessions, which is where an engagement state can do anything at
    all. It is NOT legitimate for a headline comparison, because picking
    subjects after seeing their scores would be selection on the outcome.
    """
    if subject_ids:
        subjects = [str(s) for s in subject_ids]
        print(f"using {len(subjects)} named subjects (selection on a "
              f"covariate, not on a score): {', '.join(subjects)}")
    else:
        subjects = pick_subjects(n_subjects, seed=pick_seed, query=query,
                                 min_sessions=sessions)
    done = set()
    if os.path.exists(out_csv):
        d0 = pd.read_csv(out_csv)
        done = set(zip(d0["subject_id"].astype(str),
                       d0["session_number"].astype(int),
                       d0["seed"].astype(int)))
        print(f"resuming: {len(done)} placements already in {out_csv}")

    for sid in subjects:
        for sess_no, ch, rw, ign in load_raw_sessions(sid, sessions, query):
            for s in range(seeds):
                if (str(sid), int(sess_no), int(s)) in done:
                    continue
                row = score_session(ch, rw, ign, s, maxiter=maxiter,
                                    popsize=popsize,
                                    label=f"{sid}_s{sess_no}",
                                    subject_id=str(sid),
                                    session_number=sess_no)
                pd.DataFrame([row], columns=RESULT_COLUMNS).to_csv(
                    out_csv, mode="a", header=not os.path.exists(out_csv),
                    index=False)
                msg = row["error"] or (
                    f"PsyTrack {row['glm_bits']:.3f}  "
                    f"MMLPF6 {row['mmlpf6_bits']:.3f}  "
                    f"MMLPF7 {row['mmlpf7_bits']:.3f}  "
                    f"gain {row['diff7_6']:+.3f}")
                print(f"  {sid} s{sess_no} seed{s}: {msg} "
                      f"({row['ignored_frac']:.0%} ignored, {row['seconds']}s)",
                      flush=True)
    return pd.read_csv(out_csv)


def _cluster_ci(frame, col, group="subject_id", B=4000, seed=0):
    """Subject-clustered bootstrap CI on a median.

    Sessions and placements are nested in subjects, so resampling rows would
    report an interval several times too narrow.
    """
    rng = np.random.default_rng(seed)
    groups = [g for _, g in frame.groupby(group)]
    if len(groups) < 3:
        return (np.nan, np.nan)
    boots = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        vals = pd.concat([groups[i] for i in pick])[col].dropna()
        if len(vals):
            boots.append(np.median(vals))
    if not boots:
        return (np.nan, np.nan)
    return (float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)))


def report_real(df):
    ok = df[(df["error"].isna()) | (df["error"] == "")].copy()
    if not len(ok):
        raise RuntimeError("every placement failed:\n"
                           + df["error"].value_counts().to_string())
    ok["subject_id"] = ok["subject_id"].astype(str)

    sess = ok.groupby(["subject_id", "session_number"]).median(
        numeric_only=True).reset_index()
    subj = sess.groupby("subject_id").median(numeric_only=True).reset_index()

    print("\n" + "=" * 70)
    print("DOES MODELLING THE IGNORED TRIALS HELP?")
    print("=" * 70)
    print(f"{len(ok)} placements, {ok['subject_id'].nunique()} subjects, "
          f"{int((df['error'].fillna('') != '').sum())} failures")
    print(f"ignored trials: median {ok['ignored_frac'].median():.1%}, "
          f"longest run {int(ok['ignore_run_max'].max())}")

    print("\n--- held-out bits/trial on ANSWERED trials (subject medians) ---")
    for lab, col in [("base rate", "base_bits"), ("PsyTrack", "glm_bits"),
                     ("MMLPF-6 (bias)", "mmlpf6_bits"),
                     ("MMLPF-7 (+engagement)", "mmlpf7_bits")]:
        print(f"  {lab:24s} {subj[col].median():.4f}")

    print("\n--- paired gaps (positive favours the second-named model) ---")
    for lab, col in [("MMLPF-7 over MMLPF-6", "diff7_6"),
                     ("MMLPF-6 over PsyTrack", "diff6_glm"),
                     ("MMLPF-7 over PsyTrack", "diff7_glm")]:
        med = subj[col].median()
        lo, hi = _cluster_ci(sess, col)
        ci = (f"[{lo:+.4f}, {hi:+.4f}]" if np.isfinite(lo)
              else f"[n/a: needs >=3 subjects, have "
                   f"{sess['subject_id'].nunique()}]")
        w = (stats.wilcoxon(subj[col])[1] if len(subj) > 5 else np.nan)
        pstr = f"p={w:.2g}" if np.isfinite(w) else "p=n/a (n<6)"
        print(f"  {lab:24s} {med:+.4f}  95% CI {ci}  {pstr}  "
              f"better on {(subj[col] > 0).mean():.0%} of subjects")

    print("\n--- the engagement channel (diagnostic, not in the comparison) ---")
    print(f"  theta predicts answered-vs-ignored at "
          f"{sess['engage_bits'].median():.4f} bits/trial")
    print(f"  constant-rate predictor on the same trials: "
          f"{sess['engage_base_bits'].median():.4f}")
    gain = sess["engage_base_bits"].median() - sess["engage_bits"].median()
    print(f"  gain {gain:+.4f} -- positive means theta tracks disengagement "
          f"beyond its base rate")
    print(f"  mean theta {sess['mean_theta7'].median():.3f}, "
          f"minimum reached {sess['min_theta7'].median():.3f}")
    print(f"  fitted sigma_engage: median {sess['sengage7'].median():.3f} "
          f"(bound {bc.DE_BOUNDS_ENGAGE[4][1]})")

    print("\n--- was beta absorbing disengagement? ---")
    print(f"  sigma_beta  MMLPF-6 {sess['sb6'].median():.3f}   "
          f"MMLPF-7 {sess['sb7'].median():.3f}")
    hi_b = bc.DE_BOUNDS_BIAS[1][1]
    print(f"  sessions with sigma_beta within 5% of the {hi_b} bound: "
          f"MMLPF-6 {(sess['sb6'] > 0.95 * hi_b).mean():.1%}   "
          f"MMLPF-7 {(sess['sb7'] > 0.95 * hi_b).mean():.1%}")
    print(f"  median beta  MMLPF-6 {sess['mean_beta6'].median():.1f}   "
          f"MMLPF-7 {sess['mean_beta7'].median():.1f}")
    print(f"  median phi   MMLPF-6 {sess['mean_phi6'].median():+.3f}   "
          f"MMLPF-7 {sess['mean_phi7'].median():+.3f}")

    print("\n--- does the gain track how much the animal disengages? ---")
    if len(sess) > 3:
        rho, p_ = stats.spearmanr(sess["ignored_frac"], sess["diff7_6"])
        lo, hi = _cluster_ci_rho(sess, "ignored_frac", "diff7_6")
        print(f"  rho(ignored fraction, gain from engagement) = {rho:+.3f}  "
              f"naive p = {p_:.2g}")
        if np.isfinite(lo):
            print(f"  subject-clustered 95% CI [{lo:+.3f}, {hi:+.3f}]")
        print("  POSITIVE is the prediction: more disengagement -> larger gain")
    return sess, subj


def _cluster_ci_rho(frame, xcol, ycol, group="subject_id", B=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [g for _, g in frame.groupby(group)]
    if len(groups) < 3:
        return (np.nan, np.nan)
    boots = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        sub = pd.concat([groups[i] for i in pick])
        if sub[xcol].nunique() > 2:
            boots.append(stats.spearmanr(sub[xcol], sub[ycol])[0])
    if not boots:
        return (np.nan, np.nan)
    return (float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)))


# --------------------------------------------------------------------------
def plot_recovery(df, fname="engagement_recovery.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from figure_style import apply_figure_style, panel_letter, META_GREY
        apply_figure_style()
    except Exception:
        def panel_letter(ax, s, **k):
            ax.text(-0.14, 1.04, s, transform=ax.transAxes, fontweight="bold")
        META_GREY = "#8a8a8a"

    C6, C7, CT = "#c8511b", "#17a398", "#222222"
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.6, 4.0))

    names = list(TRUTH)
    xs = np.arange(len(names))
    # Ratio to truth, so four states with different units share one axis, on a
    # LOG scale so that 2x too high and 2x too low are the same distance from
    # the 1.0 reference. Every seed is drawn: the medians alone would hide a
    # seed-to-seed spread that is larger than the difference between models.
    rng = np.random.default_rng(0)
    n_flip = 0
    for k, (tag, col) in enumerate((("6", C6), ("7", C7))):
        off = -0.17 if tag == "6" else 0.17
        for j, n in enumerate(names):
            v = (df[f"{n}{tag}"] / TRUTH[n]).values
            xj = np.full(len(v), j + off) + rng.uniform(-0.05, 0.05, len(v))
            # A ratio can be NEGATIVE (recovered value has the opposite sign to
            # truth). |ratio| would render that as a mild underestimate, which
            # is the wrong reading -- a sign flip is a qualitative failure. Draw
            # those hollow.
            flip = v < 0
            n_flip += int(flip.sum())
            axA.scatter(xj[~flip], np.clip(v[~flip], 1e-2, 1e2),
                        s=16, color=col, alpha=0.6, lw=0, zorder=3)
            axA.scatter(xj[flip], np.clip(np.abs(v[flip]), 1e-2, 1e2),
                        s=22, facecolors="none", edgecolors=col, lw=1.0,
                        zorder=3)
            # median over correctly-signed seeds only; a median that mixes
            # sign-flipped magnitudes is not a location estimate of anything
            keep = v[~flip]
            med = np.median(keep) if len(keep) else np.nan
            if np.isfinite(med):
                axA.plot([j + off - 0.12, j + off + 0.12], [med, med],
                         color=col, lw=2.2, zorder=4,
                         label=(("MMLPF-6 (ignored trials dropped)"
                                 if tag == "6"
                                 else "MMLPF-7 (ignored trials modelled)")
                                if j == 0 else None))
    if n_flip:
        axA.scatter([], [], s=22, facecolors="none", edgecolors=CT, lw=1.0,
                    label=f"wrong sign ({n_flip} fits)")
    axA.axhline(1.0, color=CT, ls="--", lw=1.2, zorder=1)
    axA.set_yscale("log")
    axA.set_yticks([0.1, 0.3, 1, 3, 10])
    axA.set_yticklabels(["0.1", "0.3", "1", "3", "10"])
    axA.set_xticks(xs)
    axA.set_xticklabels([r"$\alpha$", r"$\beta$", r"$\varphi$", "bias"])
    axA.set_ylabel("Recovered / true value\n(1 = perfect, log scale)")
    axA.set_title("Neither variant recovers the decision states;\n"
                  "seed spread exceeds the gap between them", loc="left")
    axA.legend(frameon=False, fontsize=6.8, loc="upper left")
    axA.set_xlim(-0.5, len(names) - 0.5)
    panel_letter(axA, "a")

    axB.axhline(0.97, color=CT, ls="--", lw=1.0)
    axB.axhline(0.15, color=CT, ls="--", lw=1.0)
    axB.text(0.02, 0.97, "true engaged", fontsize=6.3, color=CT, va="bottom")
    axB.text(0.02, 0.15, "true disengaged", fontsize=6.3, color=CT, va="bottom")
    axB.scatter(np.full(len(df), 0.3) + np.random.uniform(-.05, .05, len(df)),
                df["theta_out"], s=22, color=C7, zorder=3)
    axB.scatter(np.full(len(df), 0.7) + np.random.uniform(-.05, .05, len(df)),
                df["theta_in"], s=22, color=C7, zorder=3)
    axB.set_xlim(0, 1)
    axB.set_xticks([0.3, 0.7])
    axB.set_xticklabels(["engaged\nstretches", "disengaged\nstretches"])
    axB.set_ylim(0, 1.05)
    axB.set_ylabel(r"Estimated $\theta$")
    axB.set_title(r"$\theta$ separates the two regimes", loc="left")
    panel_letter(axB, "b")

    fig.text(0.5, -0.03,
             f"{len(df)} simulated sessions; all four decision states constant "
             f"by construction, engagement a two-level square wave.",
             ha="center", va="top", fontsize=6.4, color=META_GREY)
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"saved {fname}")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["recovery", "real"])
    ap.add_argument("--seeds", type=int, default=3,
                    help="block placements per session (real) or simulated "
                         "sessions (recovery)")
    ap.add_argument("--subjects", type=int, default=5)
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--pick-seed", type=int, default=0,
                    help="seed for the random subject draw")
    ap.add_argument("--trials", type=int, default=600)
    ap.add_argument("--maxiter", type=int, default=7)
    ap.add_argument("--popsize", type=int, default=4)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--query", default=None,
                    help="override COHORT_QUERY")
    ap.add_argument("--subject-ids", nargs="+", default=None,
                    help="fit these subjects instead of a random draw. Use for "
                         "a targeted test (e.g. high-ignore animals); not for "
                         "a headline comparison.")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    if args.mode == "recovery":
        csv = args.csv or "engagement_recovery.csv"
        df = (pd.read_csv(csv) if args.summarize
              else recovery(seeds=args.seeds, trials=args.trials,
                            maxiter=args.maxiter, popsize=args.popsize,
                            out_csv=csv))
        if args.summarize:
            report_recovery(df)
        plot_recovery(df)
    else:
        csv = args.csv or "engagement_real.csv"
        df = (pd.read_csv(csv) if args.summarize
              else run_real(n_subjects=args.subjects, sessions=args.sessions,
                            seeds=args.seeds, pick_seed=args.pick_seed,
                            maxiter=args.maxiter, popsize=args.popsize,
                            out_csv=csv, query=args.query,
                            subject_ids=args.subject_ids))
        report_real(df)


if __name__ == "__main__":
    main()
