# How to run the playbook

## Prerequisite

1. `uv` installed on machine
2. Databricks workspace with MLflow tracking enabled (or local MLflow server)
3. Databricks CLI installed (if using Databricks tracking)

## Setup

### 1. Environment Configuration

Copy `.env.example` to `.env` and configure your API endpoints:

```bash
cp .env.example .env
```

Edit `.env` with your actual values. The playbook supports **separate configurations** for the model being tested and the judge model:

```sh
# MLflow Tracking - Databricks Unity Catalog
MLFLOW_TRACKING_URI=databricks://<DATABRICKS_AUTH_PROFILE>
MLFLOW_REGISTRY_URI=databricks-uc
DATABRICKS_HOST=<DATABRICKS_URL>
MLFLOW_EXPERIMENT_ID=<EXPERIMENT_ID>

# Model Configuration (the model being evaluated)
MODEL_API_BASE=https://openrouter.ai/api/v1
MODEL_API_KEY=<YOUR_MODEL_API_KEY>
MODEL_API_VERSION=preview

# Judge Configuration (the model scoring responses)
JUDGE_API_BASE=https://openrouter.ai/api/v1
JUDGE_API_KEY=<YOUR_JUDGE_API_KEY>
JUDGE_API_VERSION=preview
```

**Supported Providers:**
- **OpenRouter**: `https://openrouter.ai/api/v1`
- **Azure OpenAI/Foundry**: `https://<INSTANCE_NAME>.openai.azure.com/openai/v1`
- **Ollama** (local): `http://localhost:11434/v1/`
- **Alibaba Cloud**: `https://[ServiceCode].[RegionID].aliyuncs.com/v1`
- Others (Anthropic, Bedrock, Cohere, Together AI) may require additional dependencies

### 2. Databricks Authentication

If using Databricks MLflow tracking, authenticate before running evaluations:

```bash
databricks auth login --profile <DATABRICKS_AUTH_PROFILE> --host <DATABRICKS_URL>
```

> **Note**: Replace `<DATABRICKS_AUTH_PROFILE>` and `<DATABRICKS_URL>` with values from your `.env` file.

## Running Evaluations

### Run All Suites (End-to-End)

When no `--eval` flag is provided, all four suites run in canonical order:
**enterprise → hallucination → model → vision**.

```bash
uv run playbook.py --model "openai/gpt-4o" --judge "openai/gpt-4o"
```

### Run Specific Suites

Use one or more `--eval` (or `-e`) flags to select which suites to run.
Suites **always execute in canonical order** regardless of the order you
specify them on the command line.

```bash
# Run only the enterprise suite
uv run playbook.py --model "openai/gpt-4o" --judge "openai/gpt-4o" \
    --eval enterprise

# Run enterprise and model suites (executed as enterprise → model)
uv run playbook.py --model "openai/gpt-4o" --judge "openai/gpt-4o" \
    -e enterprise -e model

# Run everything except vision
uv run playbook.py --model "openai/gpt-4o" --judge "openai/gpt-4o" \
    -e enterprise -e hallucination -e model
```

### Test with Limit (for debugging)

```bash
uv run playbook.py --model "google/gemini-3-flash-preview" --judge "gpt-5.2" --limit 1
```

### Available Suites

| Suite | Data File | Description |
|---|---|---|
| `enterprise` | `data/enterprise.json` | Brand safety, regulatory compliance, privacy, prompt injection, and other enterprise policy checks |
| `hallucination` | `data/hallucination.json` | Groundedness and factual accuracy evaluation |
| `model` | `data/data.json` | General model quality — correctness, clarity, BLEU, ROUGE, cosine similarity |
| `vision` | `data/vision.json` | Multimodal / vision evaluation (optional — skipped if data file is absent) |

### Canonical Execution Order

Suites always run in this fixed order, regardless of how they are specified on the command line:

```
enterprise → hallucination → model → vision
```

If a suite's data file is missing, that suite is skipped with a warning.
If any suite fails, the run aborts immediately.

### Command Options

| Option | Short | Required | Description |
|---|---|---|---|
| `--model` | | ✅ | Model name to evaluate (e.g. `openai/gpt-4o`) |
| `--judge` | | ✅ | Judge model name for scoring |
| `--provider` | | | Provider for the model being tested (default: `openai`) |
| `--judge-provider` | | | Provider for the judge model (default: same as `--provider`) |
| `--eval` | `-e` | | Suite(s) to run. Repeatable. If omitted, all suites run. |
| `--limit` | | | Limit number of test cases per dataset (useful for debugging) |
| `--data-dir` | | | Path to test data directory (default: `./data`) |

## MLflow Run Structure

```
Parent Run: playbook-{model}-{epoch}
├── Child: playbook-{model}-enterprise-{epoch}
├── Child: playbook-{model}-hallucination-{epoch}
├── Child: playbook-{model}-model-{epoch}
└── Child: playbook-{model}-vision-{epoch}
```

Each child run contains the evaluation metrics and traces for its suite.
When running a subset of suites, only the selected child runs are created.
Average token usage metrics are logged to every child run and propagated to the parent.

## Viewing Results

Results are automatically logged to your configured MLflow tracking server.

### Databricks Users

View results in your Databricks workspace:
```
https://<DATABRICKS_HOST>/ml/experiments/<EXPERIMENT_ID>
```

### Local MLflow Users

Start the MLflow UI:
```bash
mlflow ui
```

Then navigate to http://localhost:5000

### Generating Reports

Compare results across models from an MLflow experiment:

```bash
# Print comparison table
uv run report.py --experiment <EXPERIMENT_ID>

# Export to CSV
uv run report.py --experiment <EXPERIMENT_ID> --output results.csv
```

## Configuration Details

### Separate Model and Judge Endpoints

The playbook supports different API endpoints for the tested model and judge model. This enables scenarios like:

- Testing a local Ollama model with GPT-5 as judge
- Testing an Azure deployment with OpenRouter judge  
- Using different API keys for tested vs judge models

**Environment Variable Fallback Chain:**

1. **Model being tested**: `MODEL_API_*` → `OPENAI_API_*`
2. **Judge model**: `JUDGE_API_*` → `MODEL_API_*` → `OPENAI_API_*`

This maintains backward compatibility with existing configurations.

> ⚠️ **Never commit `.env` to version control.** Ensure it is listed in `.gitignore`.

## Evaluation Specification

See [`enterprise.md`](enterprise.md) for the full enterprise evaluation
specification, including all 45 adversarial test prompts and expected model
behaviours across 9 policy categories.