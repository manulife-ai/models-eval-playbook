# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click",
#     "databricks-agents",
#     "databricks-sdk",
#     "langchain",
#     "langchain-anthropic",
#     "langchain-openai",
#     "mlflow",
#     "nltk",
#     "numpy",
#     "pandas",
#     "scikit-learn",
# ]
# ///
import datetime
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import click
import mlflow
import numpy as np
import pandas as pd
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from mlflow.genai.scorers import (Correctness, ExpectationsGuidelines,
                                  Safety, scorer)
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

# =============================================================================
# Model Configuration
# =============================================================================


@dataclass
class ModelConfig:
    """Configuration for a model endpoint (either tested model or judge model).

    Attributes:
        provider: The model provider (e.g., "openai", "azure", "ollama")
        api_base: The API base URL
        api_key: The API key for authentication
        api_version: The API version (required for Azure OpenAI)
    """

    provider: str
    api_base: str | None = None
    api_key: str | None = None
    api_version: str | None = None

    def to_env_vars(self) -> dict[str, str]:
        """Return environment variables needed for this config."""
        env = {}
        if self.provider == "anthropic":
            if self.api_base:
                env["ANTHROPIC_BASE_URL"] = self.api_base
            if self.api_key:
                env["ANTHROPIC_API_KEY"] = self.api_key
        else:
            if self.api_base:
                env["OPENAI_API_BASE"] = self.api_base
            if self.api_key:
                env["OPENAI_API_KEY"] = self.api_key
            if self.api_version:
                env["OPENAI_API_VERSION"] = self.api_version
        return env

    def apply_to_env(self) -> None:
        """Apply this config to environment variables."""
        for key, value in self.to_env_vars().items():
            os.environ[key] = value


def load_model_config(provider: str = "openai") -> ModelConfig:
    """Load configuration for the model being tested.

    Fallback chain: MODEL_API_* -> OPENAI_API_*
    """
    return ModelConfig(
        provider=provider,
        api_base=os.getenv("MODEL_API_BASE", os.getenv("OPENAI_API_BASE")),
        api_key=os.getenv("MODEL_API_KEY", os.getenv("OPENAI_API_KEY")),
        api_version=os.getenv("MODEL_API_VERSION", os.getenv("OPENAI_API_VERSION")),
    )


def load_judge_config(provider: str | None = None) -> ModelConfig:
    """Load configuration for the judge model.

    Fallback chain: JUDGE_API_* -> MODEL_API_* -> OPENAI_API_*
    If provider is None, falls back to MODEL provider from env or "openai".
    """
    if provider is None:
        provider = os.getenv("JUDGE_PROVIDER", os.getenv("MODEL_PROVIDER", "openai"))

    return ModelConfig(
        provider=provider,
        api_base=os.getenv(
            "JUDGE_API_BASE",
            os.getenv("MODEL_API_BASE", os.getenv("OPENAI_API_BASE")),
        ),
        api_key=os.getenv(
            "JUDGE_API_KEY",
            os.getenv("MODEL_API_KEY", os.getenv("OPENAI_API_KEY")),
        ),
        api_version=os.getenv(
            "JUDGE_API_VERSION",
            os.getenv("MODEL_API_VERSION", os.getenv("OPENAI_API_VERSION")),
        ),
    )


def extract_content(outputs) -> str:
    """Extract string content from the outputs either pass on the message if its already a string, if its a dictionary try and get the content field."""
    if isinstance(outputs, str):
        return outputs
    if isinstance(outputs, dict):
        if "content" in outputs:
            return outputs["content"]
        if (
            "messages" in outputs
            and isinstance(outputs["messages"], list)
            and len(outputs["messages"]) > 0
        ):
            return extract_content(outputs=outputs["messages"][-1])
    if isinstance(outputs, BaseMessage):
        return outputs.content
    if isinstance(outputs, list):
        return extract_content(outputs=outputs[-1])
    return str(outputs)


def flatten_json(obj, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten nested JSON to dot-notation paths for field-level comparison.

    Examples:
        {"a": {"b": 1}} -> {"a.b": 1}
        {"items": [{"x": 1}, {"x": 2}]} -> {"items[0].x": 1, "items[1].x": 2}
    """
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(flatten_json(v, new_key, sep))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}[{i}]"
            items.update(flatten_json(v, new_key, sep))
    else:
        items[parent_key] = obj
    return items


def preflight_check(
    model: str,
    judge: str,
    model_config: ModelConfig,
    judge_config: ModelConfig,
) -> None:
    """Verify both model and judge are reachable before starting evaluation.

    Sends a trivial completion request to each endpoint. Raises SystemExit
    on failure so the user gets immediate feedback instead of silent scorer
    errors buried in MLflow assessments.
    """
    from openai import APIStatusError, OpenAI

    # Errors that prove the model exists (it responded, just not how we wanted)
    REACHABLE_INDICATORS = [
        "content_policy", "ContentPolicyViolation",
        "content management policy", "maximum context length",
        "rate limit", "Rate limit", "429", "quota",
        "max_tokens", "max_completion_tokens", "model output limit",
    ]

    checks = [
        ("Model", model, model_config),
        ("Judge", judge, judge_config),
    ]
    for label, name, config in checks:
        click.echo(f"  Checking {label} ({name})...", nl=False)
        try:
            client = OpenAI(
                api_key=config.api_key or "dummy",
                base_url=config.api_base,
            )
            client.chat.completions.create(
                model=name,
                messages=[{"role": "user", "content": "Say OK"}],
                max_completion_tokens=1,
            )
            click.echo(" OK")
        except APIStatusError as e:
            err_text = str(e)
            if any(indicator in err_text for indicator in REACHABLE_INDICATORS):
                click.echo(" OK (reachable)")
            else:
                click.echo(" FAILED")
                click.echo(f"\n❌ {label} '{name}' is not reachable: {e.message}")
                raise SystemExit(1)
        except Exception as e:
            click.echo(" FAILED")
            click.echo(f"\n❌ {label} '{name}' is not reachable: {e}")
            raise SystemExit(1)


def to_langchain_model_name(model: str, config: ModelConfig) -> str:
    """Format model name for LangChain.

    Returns:
        Formatted model name for LangChain (provider:model)
    """
    return f"{config.provider}:{model}"


def to_litellm_model_name(model: str, config: ModelConfig) -> str:
    """Format model name for LiteLLM and apply config to environment.

    Args:
        model: The model name/identifier
        config: ModelConfig with provider and API credentials

    Returns:
        Formatted model name for LiteLLM (provider:/model)
    """
    config.apply_to_env()
    return f"{config.provider}:/{model}"



def avg_token_usage(traces) -> dict[str, float]:
    usg = {}
    n = 0
    for trace in traces:
        if total_usage := trace.info.token_usage:
            n += 1
            for key, value in total_usage.items():
                usg[key] = usg.get(key, 0) + value
    if n == 0:
        return {}
    return {f"avg_{key}": value / n for key, value in usg.items()}


@scorer(
    name="bleu",
    description="Checks how many words from the reference are in the output",
)
def bleu(outputs, expectations) -> float:
    if "expected_response" in expectations:
        expected = expectations.get("expected_response", "")
        actual = extract_content(outputs)
        # BLEU scores (different n-gram lengths)
        smoothing = SmoothingFunction().method4
        bleu_1 = sentence_bleu(
            [expected.split()],
            actual.split(),
            weights=(1.0, 0, 0, 0),
            smoothing_function=smoothing,
        )
        bleu_2 = sentence_bleu(
            [expected.split()],
            actual.split(),
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=smoothing,
        )
        bleu_3 = sentence_bleu(
            [expected.split()],
            actual.split(),
            weights=(0.33, 0.33, 0.33, 0),
            smoothing_function=smoothing,
        )
        mean_bleu = np.mean([bleu_1, bleu_2, bleu_3])
        return mean_bleu


@scorer(
    name="rouge",
    description="Checks the overlap of n-grams between the reference and the output",
)
def rouge(outputs, expectations) -> float:
    if "expected_response" in expectations:
        expected = expectations.get("expected_response", "")
        predicted = extract_content(outputs)
        expected_tokens = expected.split()
        predicted_tokens = predicted.split()
        overlap = set(expected_tokens) & set(predicted_tokens)
        if len(expected_tokens) == 0:
            return 0.0
        rouge_score = len(overlap) / len(expected_tokens)
        return rouge_score


@scorer(
    name="cosine_similarity",
    description="Computes cosine similarity between expected and actual responses",
)
def cosine_similarity(outputs, expectations):
    if "expected_response" in expectations:
        expected = expectations.get("expected_response", "")
        predicted = extract_content(outputs)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        tfidf_matrix = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(
            [predicted, expected]
        )
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity


@scorer(
    name="field_accuracy",
    description="Calculates percentage of correctly extracted fields from JSON output",
)
def field_accuracy(outputs, expectations) -> float | None:
    """Calculate field-level accuracy for JSON extraction tasks.

    Returns the percentage of fields that were correctly extracted,
    comparing flattened JSON structures field by field.

    Returns:
        float: Percentage accuracy (0-100), or None if no expected_json provided.
    """
    if "expected_json" not in expectations:
        return None

    try:
        expected = json.loads(expectations["expected_json"])
    except (json.JSONDecodeError, TypeError):
        return None

    # Extract and parse actual output
    actual_str = extract_content(outputs)

    # Handle markdown code blocks (```json ... ```)
    if "```json" in actual_str:
        actual_str = actual_str.split("```json")[1].split("```")[0].strip()
    elif "```" in actual_str:
        actual_str = actual_str.split("```")[1].split("```")[0].strip()

    try:
        actual = json.loads(actual_str)
    except (json.JSONDecodeError, TypeError):
        return 0.0  # Invalid JSON output = 0% accuracy

    # Flatten both JSON structures for field-by-field comparison
    expected_flat = flatten_json(expected)
    actual_flat = flatten_json(actual)

    if not expected_flat:
        return 100.0  # No fields to compare = perfect match

    # Count correctly extracted fields
    correct_fields = sum(
        1
        for key, expected_value in expected_flat.items()
        if key in actual_flat and actual_flat[key] == expected_value
    )

    total_fields = len(expected_flat)
    return (correct_fields / total_fields) * 100



# =============================================================================
# Suite Runners
# =============================================================================

# Canonical execution order – suites always run in this sequence.
SUITE_ORDER: list[str] = ["enterprise", "hallucination", "model", "vision"]
DEFAULT_SUITES: list[str] = ["enterprise", "hallucination", "model"]


def _log_token_usage(run_id: str) -> None:
    """Search traces for a run and log avg token usage to the child run."""
    traces = mlflow.search_traces(
        filter_string=f"run_id = '{run_id}'", return_type="list"
    )
    token_metrics = avg_token_usage(traces)
    if token_metrics:
        mlflow.log_metrics(token_metrics)


def run_enterprise(
    *,
    model: str,
    judge_model_name: str,
    predict_fn: Callable,
    data_dir: Path,
    parent_run_id: str,
    epoch: int,
    limit: int | None = None,
) -> None:
    """Enterprise-policy evaluation (brand safety, regulatory, privacy, …)."""
    data_path = data_dir / "enterprise.json"
    if not data_path.exists():
        click.echo(f"⚠️  Skipping enterprise: {data_path} not found")
        return

    data = pd.read_json(data_path, orient="records")
    if limit:
        data = data.head(limit)

    click.echo(f"\n{'='*60}")
    click.echo(f"  Enterprise Evaluation  ({len(data)} cases)")
    click.echo(f"{'='*60}")

    scorers = [
        ExpectationsGuidelines(
            model=judge_model_name,
            name="follows_instructions",
        ),
        Safety(
            model=judge_model_name,
        ),
    ]

    with mlflow.start_run(
        run_name=f"playbook-{model}-enterprise-{epoch}",
        description=f"Enterprise evaluation for model {model}",
        parent_run_id=parent_run_id,
        nested=True,
        tags={"suite": "enterprise"},
    ) as suite_run:
        click.echo(f"Created enterprise run with ID: {suite_run.info.run_id}")
        mlflow.autolog(disable=True)
        results = mlflow.genai.evaluate(
            data=data,
            predict_fn=predict_fn,
            scorers=scorers,
        )
        mlflow.log_metrics(results.metrics)
        _log_token_usage(suite_run.info.run_id)


def run_hallucination(
    *,
    model: str,
    judge_model_name: str,
    predict_fn: Callable,
    data_dir: Path,
    parent_run_id: str,
    epoch: int,
    limit: int | None = None,
) -> None:
    """Hallucination / groundedness evaluation."""
    data_path = data_dir / "hallucination.json"
    if not data_path.exists():
        click.echo(f"⚠️  Skipping hallucination: {data_path} not found")
        return

    data = pd.read_json(data_path, orient="records")
    if limit:
        data = data.head(limit)

    click.echo(f"\n{'='*60}")
    click.echo(f"  Hallucination Evaluation  ({len(data)} cases)")
    click.echo(f"{'='*60}")

    scorers = [
        ExpectationsGuidelines(
            model=judge_model_name,
            name="follows_instructions",
        ),
    ]

    with mlflow.start_run(
        run_name=f"playbook-{model}-hallucination-{epoch}",
        description=f"Hallucination evaluation for model {model}",
        parent_run_id=parent_run_id,
        nested=True,
        tags={"suite": "hallucination"},
    ) as suite_run:
        click.echo(f"Created hallucination run with ID: {suite_run.info.run_id}")
        mlflow.autolog(disable=True)
        results = mlflow.genai.evaluate(
            data=data,
            predict_fn=predict_fn,
            scorers=scorers,
        )
        mlflow.log_metrics(results.metrics)
        _log_token_usage(suite_run.info.run_id)


def run_model(
    *,
    model: str,
    judge_model_name: str,
    predict_fn: Callable,
    data_dir: Path,
    parent_run_id: str,
    epoch: int,
    limit: int | None = None,
) -> None:
    """Model-quality evaluation (correctness, clarity, BLEU, ROUGE, …)."""
    data_path = data_dir / "data.json"
    if not data_path.exists():
        click.echo(f"⚠️  Skipping model: {data_path} not found")
        return

    data = pd.read_json(data_path, orient="records")
    if limit:
        data = data.head(limit)

    click.echo(f"\n{'='*60}")
    click.echo(f"  Model Evaluation  ({len(data)} cases)")
    click.echo(f"{'='*60}")

    scorers = [
        ExpectationsGuidelines(
            model=judge_model_name,
            name="follows_instructions",
        ),
        Safety(
            model=judge_model_name,
        ),
        Correctness(
            model=judge_model_name,
        ),
        bleu,
        rouge,
        cosine_similarity,
    ]

    with mlflow.start_run(
        run_name=f"playbook-{model}-model-{epoch}",
        description=f"Model quality evaluation for model {model}",
        parent_run_id=parent_run_id,
        nested=True,
        tags={"suite": "model"},
    ) as suite_run:
        click.echo(f"Created model run with ID: {suite_run.info.run_id}")
        mlflow.autolog(disable=True)
        results = mlflow.genai.evaluate(
            data=data,
            predict_fn=predict_fn,
            scorers=scorers,
        )
        mlflow.log_metrics(results.metrics)
        _log_token_usage(suite_run.info.run_id)


def run_vision(
    *,
    model: str,
    judge_model_name: str,
    predict_fn: Callable,
    data_dir: Path,
    parent_run_id: str,
    epoch: int,
    limit: int | None = None,
) -> None:
    """Vision / multimodal evaluation (optional – skipped if no data file)."""
    data_path = data_dir / "vision.json"
    if not data_path.exists():
        click.echo(f"⚠️  Skipping vision: {data_path} not found")
        return

    data = pd.read_json(data_path, orient="records")
    if limit:
        data = data.head(limit)

    click.echo(f"\n{'='*60}")
    click.echo(f"  Vision Evaluation  ({len(data)} cases)")
    click.echo(f"{'='*60}")

    scorers = [
        Safety(
            model=judge_model_name,
        ),
        Correctness(
            model=judge_model_name,
        ),
        ExpectationsGuidelines(
            model=judge_model_name,
            name="follows_instructions",
        ),
        field_accuracy,
    ]

    with mlflow.start_run(
        run_name=f"playbook-{model}-vision-{epoch}",
        description=f"Vision evaluation for model {model}",
        parent_run_id=parent_run_id,
        nested=True,
        tags={"suite": "vision"},
    ) as suite_run:
        click.echo(f"Created vision run with ID: {suite_run.info.run_id}")
        mlflow.autolog(disable=True)
        results = mlflow.genai.evaluate(
            data=data,
            predict_fn=predict_fn,
            scorers=scorers,
        )
        mlflow.log_metrics(results.metrics)
        _log_token_usage(suite_run.info.run_id)


# Dispatch map for suite runners
SUITE_RUNNERS: dict[str, Callable] = {
    "enterprise": run_enterprise,
    "hallucination": run_hallucination,
    "model": run_model,
    "vision": run_vision,
}


# =============================================================================
# CLI
# =============================================================================


def parse_eval(ctx, param, value: str | None) -> list[str]:
    """Parse --eval value: comma-separated suite names or 'all'."""
    if value is None:
        return list(DEFAULT_SUITES)
    names = [n.strip().lower() for n in value.split(",") if n.strip()]
    if not names:
        raise click.BadParameter("No suite names provided.")
    if "all" in names:
        return list(SUITE_ORDER)
    invalid = [n for n in names if n not in SUITE_ORDER]
    if invalid:
        raise click.BadParameter(
            f"Unknown suite(s): {', '.join(invalid)}. "
            f"Valid: {', '.join(SUITE_ORDER)} or 'all'"
        )
    return [s for s in SUITE_ORDER if s in names]


@click.command
@click.option("--model", help="Model to evaluate", required=True)
@click.option("--judge", help="Judge to use for evaluation", required=True)
@click.option("--provider", help="Provider of the model being tested", default="openai")
@click.option(
    "--judge-provider",
    help="Provider of the judge model (defaults to --provider if not set)",
    default=None,
)
@click.option(
    "--eval", "-e",
    "evals",
    default=None,
    callback=parse_eval,
    expose_value=True,
    is_eager=False,
    help="Comma-separated suites or 'all'. Default: enterprise,hallucination,model",
)
@click.option(
    "--data-dir",
    default="./data",
    help="Path to the data directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Limit number of test cases per dataset (useful for debugging)",
)
@click.option(
    "--max-concurrent",
    "-c",
    default=10,
    type=int,
    help="Max concurrent predict_fn calls across all suites (default: 10)",
)
def evaluate(
    model: str,
    judge: str,
    data_dir: str,
    provider: str = "openai",
    judge_provider: str | None = None,
    evals: list[str] = None,
    limit: int | None = None,
    max_concurrent: int = 10,
):
    # Build configurations for model and judge
    model_config = load_model_config(provider=provider)
    judge_config = load_judge_config(
        provider=judge_provider if judge_provider else provider
    )

    # evals is already parsed/validated by the parse_eval callback
    to_run = evals

    if not to_run:
        click.echo("No valid evaluation suites selected.")
        raise SystemExit(1)

    click.echo(
        f"Starting evaluation for model: {model} (provider: {model_config.provider}, api_base: {model_config.api_base})"
    )
    click.echo(
        f"Using judge: {judge} (provider: {judge_config.provider}, api_base: {judge_config.api_base})"
    )
    click.echo(f"Suites: {', '.join(to_run)}")
    if limit:
        click.echo(f"Limiting each dataset to {limit} entries for debugging")

    click.echo("\nPre-flight connectivity check:")
    preflight_check(model, judge, model_config, judge_config)
    click.echo()

    epoch_time = int(time.time())
    data_path = Path(data_dir)

    with mlflow.start_run(run_name=f"playbook-{model}-{epoch_time}") as run:
        mlflow.log_params(
            {
                "model": model,
                "judge": judge,
                "model_provider": model_config.provider,
                "judge_provider": judge_config.provider,
                "model_api_base": model_config.api_base or "default",
                "judge_api_base": judge_config.api_base or "default",
                "run_date": datetime.datetime.now().isoformat(),
                "suites": ",".join(to_run),
            }
        )
        print(f"Created parent run with ID: {run.info.run_id}")

        # Log data artifacts for the selected suites
        suite_data_files = {
            "enterprise": "enterprise.json",
            "hallucination": "hallucination.json",
            "model": "data.json",
            "vision": "vision.json",
        }
        for suite_name in to_run:
            artifact = data_path / suite_data_files[suite_name]
            if artifact.exists():
                mlflow.log_artifact(str(artifact), artifact_path="data")

        # Create agent with credentials baked into the model instance
        # (no env-var dependency → safe for concurrent predict_fn calls)
        model_kwargs = {}
        if model_config.api_key:
            model_kwargs["api_key"] = model_config.api_key
        if model_config.api_base:
            model_kwargs["base_url"] = model_config.api_base
        chat_model = init_chat_model(
            to_langchain_model_name(model, model_config),
            **model_kwargs,
        )
        agent = create_agent(model=chat_model)

        # Get judge model name (applies judge config to env)
        judge_model_name = to_litellm_model_name(judge, judge_config)

        def _detect_image_type(data: bytes) -> str:
            """Detect actual image MIME type from magic bytes."""
            if data[:3] == b"\xff\xd8\xff":
                return "image/jpeg"
            if data[:8] == b"\x89PNG\r\n\x1a\n":
                return "image/png"
            if data[:4] == b"GIF8":
                return "image/gif"
            if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
                return "image/webp"
            return ""

        def _url_to_data_uri(url: str) -> str:
            """Download an image URL and return a base64 data URI."""
            import base64
            import mimetypes
            import urllib.request
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    data = resp.read()
                    # Detect actual format from bytes; fall back to header/extension
                    content_type = _detect_image_type(data)
                    if not content_type:
                        content_type = resp.headers.get("Content-Type", "")
                    if not content_type:
                        content_type, _ = mimetypes.guess_type(url)
                        content_type = content_type or "image/png"
                    b64 = base64.b64encode(data).decode("utf-8")
                    return f"data:{content_type};base64,{b64}"
            except Exception as e:
                click.echo(f"⚠️  Could not convert URL to data URI: {url} ({e})")
                return url  # fall back to original URL

        def _sanitize_messages(messages):
            """Strip unsupported fields and convert image URLs to base64 data URIs."""
            sanitized = []
            for msg in messages:
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, list):
                        new_content = []
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "image_url":
                                # Remove unsupported 'detail' field
                                part = {k: v for k, v in part.items() if k != "detail"}
                                # Convert URL to base64 data URI for models that can't fetch URLs
                                img = part.get("image_url", {})
                                url = img.get("url", "")
                                if url.startswith("http"):
                                    part = {**part, "image_url": {**img, "url": _url_to_data_uri(url)}}
                            new_content.append(part)
                        msg = {**msg, "content": new_content}
                sanitized.append(msg)
            return sanitized

        concurrency_sem = threading.Semaphore(max_concurrent)

        def predict_fn(messages):
            with concurrency_sem:
                try:
                    result = agent.invoke({"messages": _sanitize_messages(messages)})
                    return result if result and str(result).strip() else "[Model returned empty response]"
                except Exception as e:
                    return f"[Model error: {e}]"

        # Run selected suites sequentially.
        # MLflow's evaluate() parallelizes predict_fn calls internally;
        # the semaphore above controls concurrency. Inter-suite parallelism
        # is not used because MLflow's trace manager singleton causes
        # contention/data loss when multiple evaluate() calls overlap.
        for suite_name in to_run:
            click.echo(f"\n▶ Running suite: {suite_name}")
            SUITE_RUNNERS[suite_name](
                model=model,
                judge_model_name=judge_model_name,
                predict_fn=predict_fn,
                data_dir=data_path,
                parent_run_id=run.info.run_id,
                epoch=epoch_time,
                limit=limit,
            )

        # Mark run as complete so report.py can identify fully successful runs
        mlflow.set_tag("playbook.status", "complete")
        click.echo(f"\n✅ All requested suites complete. Epoch: {epoch_time}")


if __name__ == "__main__":
    evaluate()
