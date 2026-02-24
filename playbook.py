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
#     "py-readability-metrics",
#     "rouge",
#     "scikit-learn",
#     "psutil",
# ]
# ///
import datetime
import json
import os
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

import click
import mlflow
import numpy as np
import pandas as pd
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage
from mlflow.genai.scorers import (
    Correctness,
    ExpectationsGuidelines,
    Guidelines,
    Safety,
    Scorer,
    scorer,
)
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from contextlib import contextmanager

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


def to_langchain_model_name(model: str, config: ModelConfig) -> str:
    """Format model name for LangChain and apply config to environment.

    Args:
        model: The model name/identifier
        config: ModelConfig with provider and API credentials

    Returns:
        Formatted model name for LangChain (provider:model)
    """
    config.apply_to_env()
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


@contextmanager
def temporary_env(config: ModelConfig):
    """Temporarily apply a config's environment variables, then restore previous values.

    This ensures that the model agent uses model credentials even if judge credentials
    were applied to environment variables afterwards.
    """
    # Save current env vars
    saved_env = {}
    env_vars = config.to_env_vars()

    for key in env_vars.keys():
        saved_env[key] = os.environ.get(key)

    # Apply new config
    config.apply_to_env()

    try:
        yield
    finally:
        # Restore previous values
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def avg_token_usage(traces) -> dict[str, float]:
    usg = {}
    for trace in traces:
        if total_usage := trace.info.token_usage:
            for key, value in total_usage.items():
                usg[key] = usg.get(key, 0) + value
    return {f"avg_{key}": value / len(traces) for key, value in usg.items()}


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


def run_evaluations(
    model: str,
    judge: str,
    data: pd.DataFrame,
    predict_fn: Callable,
    scorers: list[Scorer] = [],
    model_config: ModelConfig | None = None,
    judge_config: ModelConfig | None = None,
) -> pd.DataFrame:
    """Run evaluations with separate model and judge configurations.

    Args:
        model: Name of the model being tested
        judge: Name of the judge model
        data: DataFrame with test cases
        predict_fn: Function to get predictions from the model
        scorers: Additional scorers to use
        model_config: Configuration for the tested model (loads from env if None)
        judge_config: Configuration for the judge model (loads from env if None)
    """
    # Load configs from environment if not provided
    if model_config is None:
        model_config = load_model_config()
    if judge_config is None:
        judge_config = load_judge_config()

    epoch_time = int(time.time())
    with mlflow.start_run(
        run_name=f"playbook-eval-{model}-{epoch_time}",
        description=f"Evaluation run for model {model} with judge {judge}",
    ) as run:
        mlflow.log_params(
            {
                "model": model,
                "judge": judge,
                "model_provider": model_config.provider,
                "judge_provider": judge_config.provider,
                "model_api_base": model_config.api_base or "default",
                "judge_api_base": judge_config.api_base or "default",
                "run_date": datetime.datetime.now().isoformat(),
                "num_cases": len(data),
            }
        )
        with TemporaryDirectory(delete=True) as temp_dir:
            temp_file = f"{temp_dir}/evaluation.json"
            data.to_json(temp_file, orient="records")
            mlflow.log_artifact(temp_file, artifact_path="data")

        # Get judge model name (applies judge config to env)
        judge_model_name = to_litellm_model_name(judge, judge_config)

        # Define the evaluation functions or scorers here
        safety = Safety(
            model=judge_model_name,
        )
        correctness = Correctness(
            model=judge_model_name,
        )
        clarity = Guidelines(
            model=judge_model_name,
            name="clarity",
            guidelines=["The response must be clear, coherent, and concise"],
        )
        english = Guidelines(
            model=judge_model_name,
            name="follows_user_language_preference",
            guidelines=[
                "The response must be in the exact same language as the user's original input",
            ],
        )

        @scorer(
            name="follows_guidelines",
            description="Checks if the response follows provided guidelines",
        )
        def follows_guidelines(inputs, outputs, expectations) -> bool:
            if "guidelines" in expectations:
                guidelines = ExpectationsGuidelines(
                    model=judge_model_name,
                    name="follows_instructions",
                )
                return guidelines(
                    inputs=inputs, outputs=outputs, expectations=expectations
                )
            return True

        # Run Evaluations
        results = mlflow.genai.evaluate(
            data=data,
            predict_fn=predict_fn,
            scorers=[
                # Enterprise level scorers
                safety,
                clarity,
                english,
                # Average token usage will be logged separately (see below)
                # Model level scorers
                correctness,
                follows_guidelines,
                # Pattern level scorers
                bleu,
                rouge,
                cosine_similarity,
                # Usecase level scorers
                *scorers,
            ],
        )
        run_id = run.info.run_id
        traces = mlflow.search_traces(
            filter_string=f"run_id = '{run_id}'", return_type="list"
        )
        mlflow.log_metrics(avg_token_usage(traces))
        print(
            f"Logged evaluation run {run.info.run_id} for model {model} using judge {judge}."
        )
        print(f"{len(traces)} traces are available for this run.")
        return results


# =============================================================================
# Suite Runners
# =============================================================================

# Canonical execution order – suites always run in this sequence.
SUITE_ORDER: list[str] = ["enterprise", "hallucination", "model", "vision"]


def _log_token_usage(run_id: str, parent_run_id: str) -> None:
    """Search traces for a run and log avg token usage to both child and parent."""
    traces = mlflow.search_traces(
        filter_string=f"run_id = '{run_id}'", return_type="list"
    )
    token_metrics = avg_token_usage(traces)
    mlflow.log_metrics(token_metrics)
    mlflow.log_metrics(token_metrics, run_id=parent_run_id)


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
    ) as suite_run:
        click.echo(f"Created enterprise run with ID: {suite_run.info.run_id}")
        mlflow.autolog(disable=True)
        results = mlflow.genai.evaluate(
            data=data,
            predict_fn=predict_fn,
            scorers=scorers,
        )
        _log_token_usage(suite_run.info.run_id, parent_run_id)
        mlflow.log_metrics(results.metrics, run_id=parent_run_id)


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
    ) as suite_run:
        click.echo(f"Created hallucination run with ID: {suite_run.info.run_id}")
        results = mlflow.genai.evaluate(
            data=data,
            predict_fn=predict_fn,
            scorers=scorers,
        )
        _log_token_usage(suite_run.info.run_id, parent_run_id)
        mlflow.log_metrics(results.metrics, run_id=parent_run_id)


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
    ) as suite_run:
        click.echo(f"Created model run with ID: {suite_run.info.run_id}")
        results = mlflow.genai.evaluate(
            data=data,
            predict_fn=predict_fn,
            scorers=scorers,
        )
        _log_token_usage(suite_run.info.run_id, parent_run_id)
        mlflow.log_metrics(results.metrics, run_id=parent_run_id)


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
    ) as suite_run:
        click.echo(f"Created vision run with ID: {suite_run.info.run_id}")
        results = mlflow.genai.evaluate(
            data=data,
            predict_fn=predict_fn,
            scorers=scorers,
        )
        _log_token_usage(suite_run.info.run_id, parent_run_id)
        mlflow.log_metrics(results.metrics, run_id=parent_run_id)


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
    multiple=True,
    type=click.Choice(SUITE_ORDER, case_sensitive=False),
    help="Evaluation suite(s) to run. Repeatable. If omitted, all suites run.",
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
def evaluate(
    model: str,
    judge: str,
    data_dir: str,
    provider: str = "openai",
    judge_provider: str | None = None,
    evals: tuple[str, ...] = (),
    limit: int | None = None,
):
    # Build configurations for model and judge
    model_config = load_model_config(provider=provider)
    judge_config = load_judge_config(
        provider=judge_provider if judge_provider else provider
    )

    # Resolve which suites to run, always in canonical order
    requested = [e.lower() for e in evals] if evals else SUITE_ORDER
    to_run = [s for s in SUITE_ORDER if s in requested]

    if not to_run:
        click.echo("No valid evaluation suites selected.")
        raise SystemExit(1)

    print(
        f"Starting evaluation for model: {model} (provider: {model_config.provider}, api_base: {model_config.api_base})"
    )
    print(
        f"Using judge: {judge} (provider: {judge_config.provider}, api_base: {judge_config.api_base})"
    )
    print(f"Suites: {', '.join(to_run)}")
    if limit:
        print(f"Limiting each dataset to {limit} entries for debugging")

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

        # Create agent (config will be applied during each invocation via predict_fn)
        agent = create_agent(model=to_langchain_model_name(model, model_config))

        # Get judge model name (applies judge config to env)
        judge_model_name = to_litellm_model_name(judge, judge_config)

        predict_lock = threading.Lock()
        def predict_fn(messages):
            with predict_lock:
                # Use model config during agent invocation to avoid using judge credentials
                with temporary_env(model_config):
                    return agent.invoke({"messages": messages})

        # Run selected suites in canonical order
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

        click.echo(f"\n✅ All requested suites complete. Epoch: {epoch_time}")


if __name__ == "__main__":
    evaluate()
