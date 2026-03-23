---
name: model-eval-playbook
description: Run model evaluations, diagnose MLflow runs, report stats, and compare models using the models-eval-playbook repo. Use when the user asks to evaluate an LLM, run the playbook, check eval results, diagnose MLflow run errors, compare models, or generate a report from experiment data. Triggers on mentions of running evals, playbook, MLflow run IDs, experiment IDs, model comparison, or eval suites (enterprise, hallucination, model, vision).
---

# Model Eval Playbook

Run LLM evaluations across 4 suites, diagnose failures via MLflow, and compare results.

## Prerequisites

Before any operation, ensure Databricks auth is active:

```bash
# Extract profile from .env MLFLOW_TRACKING_URI (e.g. "databricks://adb-249..." → profile "adb-249...")
/opt/homebrew/bin/databricks auth token --profile <PROFILE> 2>&1
```

If expired, re-authenticate:
```bash
/opt/homebrew/bin/databricks auth login --profile <PROFILE> --host <DATABRICKS_HOST>
```

Read `.env` to discover `MLFLOW_TRACKING_URI`, `DATABRICKS_HOST`, and `MLFLOW_EXPERIMENT_ID`.

## Task 1: Run Evaluations

### Determine provider flags

Read `.env` to determine the correct `--provider` and `--judge-provider`:

| `MODEL_API_BASE` contains | `--provider` |
|---|---|
| `/anthropic` | `anthropic` |
| `/openai/v1` or openrouter | `openai` |
| `localhost:11434` | `openai` (ollama) |

Same logic for `JUDGE_API_BASE` → `--judge-provider`.

### Execute

```bash
set -a && source .env && set +a
uv run playbook.py \
  --model "<model-name>" \
  --judge "<judge-model>" \
  --provider <provider> \
  --judge-provider <judge-provider> \
  [--eval <suites>] \
  [--limit N] \
  [--max-concurrent N]
```

- Omit `--eval` to run default suites (enterprise, hallucination, model)
- Use `--eval all` to run all suites including vision
- Use `--eval enterprise,vision` to run specific suites (comma-separated)
- Use `--limit N` for debugging (limits test cases per suite)
- Use `--max-concurrent N` to control concurrent predict\_fn calls (default: 10)

### Run structure

```
Parent: playbook-{model}-{epoch}
├── Child: playbook-{model}-enterprise-{epoch}
├── Child: playbook-{model}-hallucination-{epoch}
├── Child: playbook-{model}-model-{epoch}
└── Child: playbook-{model}-vision-{epoch}
```

## Task 2: Report Stats

```bash
set -a && source .env && set +a
uv run report.py --experiment <EXPERIMENT_ID>
```

Uses `MLFLOW_EXPERIMENT_ID` from `.env`. Shows comparison table across all models with latest run per model. Add `--output results.csv` for CSV export.

## Task 3: Compare Models

Run `report.py` as above — it automatically shows all models side-by-side. To compare specific suites, filter the output table rows by suite prefix (e.g. `enterprise/`, `vision/`).

For a deeper comparison of two specific runs, query both run IDs and diff their metrics. See `references/mlflow-diagnosis.md` for API patterns.

## Task 4: Diagnose MLflow Run Errors

When a user provides a run ID with errors, follow this sequence:

1. **Get run info** — determine run name, status, parent/child structure
2. **List child runs** — find which suite(s) had errors
3. **Get traces** — identify ERROR-state traces and extract error counts
4. **Get spans** — extract the actual exception message from span events

Full API patterns and code snippets are in `references/mlflow-diagnosis.md`. Read that file when diagnosing.

### Common error patterns

| Error message | Root cause | Fix |
|---|---|---|
| `image was specified using image/png but appears to be image/jpeg` | S3 images with wrong Content-Type/extension | Fixed: `_detect_image_type` uses magic bytes |
| `scorer requires the following fields: outputs` | predict_fn returned no output (trace ERROR state) | Check span events for the upstream API error |
| `OPENAI_API_KEY not set` | `.env` not sourced before running | Source .env: `set -a && source .env && set +a` |
| `--with-vision` not recognized | Stale run scripts using old CLI flag | Use `--eval all` to include vision |
