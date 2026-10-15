# MML-PF vs PsyTrack on dynamic foraging

Out-of-sample comparison of a maximum-marginal-likelihood particle filter
against PsyTrack, on held-out blocks of trials within a session.

## What the comparison does

Both models are scored **one step ahead on identical held-out trials**, with
hyperparameters fit on training trials only. Held-out blocks are 20
consecutive trials scattered within the session (per Pillow's prescription),
and all data stay available as regressors — only the held-out likelihood terms
are withheld. No smoother is used for scoring, because a smoother sees future
choices.

Anything inferred from the likelihood freezes on a held-out trial (PsyTrack's
weights; the filter's α, β, φ, bias particles). Anything computed directly from
observable data keeps updating (PsyTrack's regressors; the filter's Q values).

The **subject is the unit of replication**: placement-level results collapse by
median to session, then to subject, then are tested across subjects. Report
effect size and win fraction ahead of p-values — at cohort size a negligible
gap still yields a small p-value.

## Pipeline

```bash
# 1. cache trial arrays (needs the database; run where it is reachable)
python hpc_block_cv.py prepare --cache-dir ./cache --subjects 50 --sessions 20

# 2. one array task per subject (no database access needed)
python hpc_block_cv.py task --cache-dir ./cache --out-dir ./results --task-id $ID

# 3. combine
python hpc_block_cv.py combine --out-dir ./results --csv data/cohort.csv
```

Stage 2 is **two-pass**. Pass 1 fits a volatility per (session, placement);
pass 2 takes the median across that subject's placements and re-scores
everything at that shared value. The per-session estimate is dominated by
session noise — on sessions simulated with a true σ of exactly zero it ranged
over essentially the whole search interval. Both values are recorded:
`sigma_*` is the shared value used for scoring, `sigma_*_session` is that
placement's own fit.

Because the shared value depends on all of a subject's placements, **the resume
unit is the subject**: a complete CSV is skipped, a partial one is moved aside
and refit. Resuming placement-by-placement would leave rows scored under
different volatilities in one file.

## Things that will bite you

**Results are not comparable across parameterisation versions.** `block_cv.py`
carries `PARAM_VERSION`. Version 2 moved α to logit space (no clip) and moved
the β clip from the value to the state. Fitted volatilities are in different
units across that change. Everything in `data/` predates it.

**The cohort is not reproducible from the query alone.** Running the identical
`COHORT_QUERY` with the identical eligibility rule months apart returned a
disjoint set of animals — the upstream database contents changed. Freeze an
explicit subject and session list alongside any result you intend to publish,
and treat the query as provenance rather than as the definition.

**Volatilities are barely identified per session.** Report the across-placement
spread, which `task` prints, rather than a point estimate. A σ whose range
covers the search interval is unidentified, not estimated.

**β is reportable, α is not.** After the version-2 reparameterisation β sits
well inside its range with tight posterior bands. α remains spread across
(0, 1) at the volatility the fit selects, even on constant-α simulated data.

**Simulating from the model under test is a ceiling, not a validation.** Every
`--demo` and recovery mode generates data from these same equations.

## Layout

| | |
|---|---|
| `block_cv.py` | filter, GLM, masking, scoring, `build_glm_regressors`. The core. |
| `hpc_block_cv.py` | three-stage cluster harness |
| `mmlpf_vs_psytrack_cv.py` | cohort query, PsyTrack fitting, shared constants |
| `bias_model_comparison.py`, `bias_parameterization.py`, `engagement_comparison.py` | model-extension comparisons |
| `latent_comparison.py`, `bias_showcase.py` | session figures |
| `data/` | curated results and figures (tracked) |
| `cache*/`, `out*/`, root `*.png` | generated, gitignored |

`empirical_drifting_agent_mml_pf.py` is the original model definition. Nothing
imports it any more, but the particle dynamics, parameterisations and clip
bounds in `block_cv.py` were copied from it verbatim — it is the provenance for
the whole comparison. The remaining unimported scripts are earlier exploratory
work kept for reference.

All results live in `data/`. Regenerating a figure from a kept CSV needs the
new path, e.g.:

```bash
python bias_model_comparison.py --summarize --csv data/bias_comparison.csv
python bias_parameterization.py --summarize --csv data/bias_param_tight.csv
```
