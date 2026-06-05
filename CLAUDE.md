# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LymphXPress classifies DLBCL molecular subtypes (MCD, A53, N1, BN2, EZB, ST2, Other) from gene
expression. The repo has three layers:

- **`lymphxpress/`** — the Python package: classifier core + CLI (`pip install -e .`).
- **`data/`** — bundled training CSVs and the N1 gene-weight signature, loaded at runtime via
  `Path(__file__).parent.parent / "data"` (i.e. the package resolves `../data` relative to itself, so
  the `data/` dir must sit beside `lymphxpress/`).
- **`web/`** — a FastAPI + Celery + React app that wraps the package as an async job service.

## Commands

### Package / CLI

```bash
pip install -e .                          # editable install (Python 3.10–3.12)
lymphxpress data/X_ffm.csv --out out.csv  # run the CLI on a CSV (samples=rows, genes=cols)
lymphxpress <csv> -m 100 -s rnaseq_dge -r 0 --no-scaler --ray
```

There is **no test suite, linter config, or formatter config** in this repo. Validate changes by running
the CLI or the Python API against the CSVs in `data/`.

### Web app

```bash
cd web && bash dev.sh        # local dev: Redis + FastAPI(:8000) + Celery + Vite(:5173)
cd web && docker compose up --build   # full stack behind nginx at http://localhost
```

- API docs at `http://localhost:8000/docs`. The frontend dev server proxies `/api` → `:8000`.
- Frontend build: `cd web/frontend && npm run build` (`tsc && vite build`).
- `dev.sh` sets `JOBS_DIR=/tmp/lymphxpress_jobs` and `PYTHONPATH` to the repo root so the worker imports
  the local `lymphxpress` package.

## Architecture

### Classifier core (`lymphxpress/clf.py`)

The prediction pipeline is a **two-stage ensemble**, orchestrated by `LymphxpressClassifier`:

1. **`ConsensusClassifier`** — the reusable engine. Trains `n_models` XGBoost models, each on a
   stratified sample subset and a random feature subset (`n_features_per_model="auto"` sizes subsets for
   ~95% expected feature coverage across the ensemble; see `compute_required_features_per_model`).
   Probabilities are aggregated in **logit space** (`_aggregate_binary_logits` /
   `_aggregate_multiclass_logits`); hard predictions are **majority vote**, ties broken by aggregated
   probability. Supports thread-parallel (`n_jobs`) or Ray (`use_ray`) training.

2. **`LymphxpressClassifier`** — composes two `ConsensusClassifier`s:
   - an **N1 binary gate** (trained with the N1 gene signature forced in via `keep_features`), and
   - a **multi-class classifier** trained on the non-N1 samples only.

   `predict()` assigns N1 when the N1 gate's vote majority fires, else the multi-class plurality winner.
   `predict_proba()` blends them: if the N1 gate fires, `p[N1]=p_n1` and other classes are scaled by
   `(1 - p_n1)`; otherwise N1 is zeroed and multi-class probs pass through.

**Training data + scalers.** Four reference cohorts (`cornell_pseudobulk`, `vancouver`, `cornell_bulk`,
`ffm`) are read from `data/X_*.csv` / `y_*.csv` and reduced to their common gene set, then intersected
again with the test data's genes. Two named scalers exist: `rnaseq_dge` (fit on vancouver+cornell_bulk)
and `rnapanel` (= the ffm scaler). The caller picks one via the `scaler=` argument to `add_test_data`.

**Caching is performance-critical and subtle.** `LymphxpressClassifier._CACHE` is a *process-level* cache
keyed `"default"`, valid **only** when no class filtering is applied (`keep_train_classes` /
`drop_train_classes` both unset) and not in custom-training mode. It stores the fully-scaled
full-feature training matrices + fitted scalers; per-job feature intersection is then served by slicing
(`_subset_scaler` rebuilds a `StandardScaler` over a column subset rather than refitting). The Celery
worker calls `LymphxpressClassifier._prewarm()` at process init so the first job pays no setup cost. When
editing the setup paths, keep the cacheable / class-filtering / custom-training branches in
`_setup_training_data` consistent.

**Required call order** (enforced with `RuntimeError`s): `add_test_data` → `fit` → `predict` /
`predict_proba`. `LymphxpressClassifier` is single-use: `add_test_data` may be called only once per
instance.

### Web backend (`web/backend/app/`)

FastAPI (`main.py`) + Celery (`tasks.py`, `celery_app.py`) + SQLite (`database.py`, `models.py`).

- **Flow:** `POST /api/jobs` saves the uploaded CSV under `JOBS_DIR/<job_id>/`, creates a `Job` row, and
  dispatches the Celery task with `task_id=job_id`. The worker runs the classifier, deletes the uploaded
  CSV immediately after load, writes `results.json` + `results.csv`, and updates the `Job` row with live
  progress via `on_n1_progress`/`on_multi_progress` callbacks (throttled to ~1/sec).
- **Progress is polled, not streamed:** the frontend polls `GET /api/jobs/{id}` every 1s
  (`streamJobProgress` in `api.ts`). Despite the nginx/vite SSE-buffering comments, there is no SSE
  endpoint.
- **Auth:** session-token based (`X-Access-Token` header). `require_session` guards all job endpoints.
  Email verification is gated by `SMTP_ENABLED`; when false, accounts auto-verify and the verification
  link is printed to stdout.
- **SQLite specifics:** WAL mode, one DB file at `JOBS_DIR/jobs.db`. Schema changes are handled by
  hand-rolled `ALTER TABLE` in `database._migrate()` (no Alembic) — add new columns there. The
  `Registration` model is dead/legacy, kept only to avoid FK conflicts.
- Jobs and their files older than `RESULTS_RETENTION_DAYS` are purged on startup and on job creation.

### Web frontend (`web/frontend/src/`)

React + TypeScript + Vite + Tailwind, React Router. All server calls go through `lib/api.ts` (token
stored in `localStorage`; a 401 dispatches an `auth:unauthorized` window event). Pages: `Home`,
`Classify` (upload), `JobProgress` (polling), `Results` (Plotly viz), `History`, `VerifyEmail`.

## Conventions

- The canonical class order is `["MCD", "A53", "N1", "BN2", "EZB", "ST2", "Other"]`
  (`LYMPHXPRESS_CLASSES` in `tasks.py`); align result columns to it.
- Scaler names are the lowercase strings `rnaseq_dge` / `rnapanel`; `none`/`--no-scaler` disables
  scaling. The web layer additionally accepts the literal `"none"`.
- Determinism: each model's `random_state` is `(seed or 0) + model_idx`; subsets are pre-sampled before
  training so ordering is path-independent across the sequential/threaded/Ray code paths. Preserve this
  when touching `fit`.
