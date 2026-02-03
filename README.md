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

### Basic Usage

```bash
uv run playbook.py --model <MODEL_NAME> --judge <JUDGE_MODEL_NAME>
```

### Examples

**Test with limit (for debugging):**
```bash
uv run playbook.py --model "google/gemini-3-flash-preview" --judge "gpt-5.2" --limit 1
```

**Full evaluation:**
```bash
uv run playbook.py --model "google/gemini-3-flash-preview" --judge "gpt-5.2"
```

**With vision support:**
```bash
uv run playbook.py --model "gpt-4o" --judge "gpt-5.2" --with-vision
```

### Command Options

- `--model`: Model name to evaluate (required)
- `--judge`: Judge model name for scoring (required)
- `--provider`: Provider for the model being tested (default: `openai`)
- `--judge-provider`: Provider for the judge model (default: same as `--provider`)
- `--limit`: Limit number of test cases per dataset (useful for debugging)
- `--with-vision`: Include vision evaluation tests
- `--data-dir`: Path to test data directory (default: `./data`)

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