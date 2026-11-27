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
from mlflow.entities import Feedback
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



class PredictionStats:
    """Thread-safe collector for per-prediction metrics within a suite."""

    def __init__(self):
        self._lock = threading.Lock()
        self.latencies: list[float] = []
        self.error_count: int = 0
        self.total_calls: int = 0
        self.response_lengths: list[int] = []
        self.input_tokens: list[int] = []
        self.output_tokens: list[int] = []
        self.total_tokens: list[int] = []

    def record(self, latency_s: float, response: str, is_error: bool,
               usage: dict | None = None):
        with self._lock:
            self.total_calls += 1
            self.latencies.append(latency_s)
            if is_error:
                self.error_count += 1
            else:
                self.response_lengths.append(len(response))
            if usage:
                self.input_tokens.append(usage.get("input_tokens", 0))
                self.output_tokens.append(usage.get("output_tokens", 0))
                self.total_tokens.append(usage.get("total_tokens", 0))

    def to_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if self.total_calls > 0:
            metrics["error_count"] = self.error_count
            metrics["error_rate"] = self.error_count / self.total_calls
            metrics["total_predictions"] = self.total_calls
        if self.latencies:
            arr = np.array(self.latencies)
            metrics["latency_avg_s"] = float(np.mean(arr))
            metrics["latency_p50_s"] = float(np.median(arr))
            metrics["latency_p95_s"] = float(np.percentile(arr, 95))
        if self.response_lengths:
            metrics["response_length_avg"] = float(np.mean(self.response_lengths))
        return metrics

    def token_metrics(self) -> dict[str, float]:
        if not self.input_tokens:
            return {}
        inp = np.array(self.input_tokens)
        out = np.array(self.output_tokens)
        tot = np.array(self.total_tokens)
        return {
            "avg_input_tokens": float(np.mean(inp)),
            "avg_output_tokens": float(np.mean(out)),
            "avg_total_tokens": float(np.mean(tot)),
            "p50_total_tokens": float(np.median(tot)),
            "p95_total_tokens": float(np.percentile(tot, 95)),
            "sum_total_tokens": float(np.sum(tot)),
        }


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
# mubench logging (opt-in via --mubench)
# =============================================================================
#
# With --mubench, per-case results are collected while the playbook runs as
# usual, then written once more as a mubench Run (mubench.log helper) into the
# MUBENCH_MLFLOW_EXPERIMENT experiment. The playbook's own MLflow logging is
# untouched. `mubench` is only imported when the flag is set.

MUBENCH_HARNESS = "models-eval-playbook"
# suite -> (mubench suite id, primary scorer name as logged by mlflow.genai)
MUBENCH_SUITES: dict[str, tuple[str, str]] = {
    "enterprise": ("playbook-enterprise", "follows_instructions"),
    "hallucination": ("playbook-hallucination", "follows_instructions"),
    "model": ("playbook-model", "correctness"),
    "vision": ("playbook-vision", "correctness"),
}
# The playbook has no prompt template: each case carries its own system + user
# messages in `inputs`, which are hashed as part of the Items.
MUBENCH_PROMPT = "inputs.messages sent as-is (per-case system + user messages)"


def _messages_key(messages) -> str:
    """Stable key for a case's input messages (inputs are unique per dataset)."""
    return json.dumps(messages, sort_keys=True, ensure_ascii=False, default=str)


def mubench_item_ids(data: pd.DataFrame) -> list[str]:
    """Item ids from `tags.prompt_id` / `tags.id`; repeats get a `-2`, `-3` suffix."""
    ids, seen = [], {}
    for n, tags in enumerate(data.get("tags", [None] * len(data))):
        tags = tags if isinstance(tags, dict) else {}
        base = str(tags.get("prompt_id") or tags.get("id") or n)
        seen[base] = seen.get(base, 0) + 1
        ids.append(base if seen[base] == 1 else f"{base}-{seen[base]}")
    return ids


def mubench_items(data: pd.DataFrame) -> list[dict]:
    """Items (id + inputs + expectations) as hashed into the Suite Version."""
    return [
        {"item_id": item_id, "inputs": row["inputs"], "expectations": row["expectations"]}
        for item_id, (_, row) in zip(mubench_item_ids(data), data.iterrows())
    ]


def mubench_score(value) -> float | None:
    """Judge feedback value -> 0..1 (yes/pass/true -> 1, no/fail/false -> 0)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        v = float(value)
        if np.isnan(v) or v < 0:
            return None
        return v if v <= 1 else min(v / 100.0, 1.0)  # 0-100 scales
    text = str(getattr(value, "value", value)).strip().lower()
    if text in ("yes", "pass", "true", "correct"):
        return 1.0
    if text in ("no", "fail", "false", "incorrect"):
        return 0.0
    try:
        return mubench_score(float(text))
    except ValueError:
        return None


def mubench_usage(result) -> dict:
    """Model-under-test usage from a LangChain agent result, summed over its AI messages.

    Uses LangChain's `usage_metadata` (input/output plus input_token_details
    cache_read/cache_creation/audio and output_token_details reasoning/audio).
    `cost_usd` only when the provider put a per-call cost in the response
    (e.g. OpenRouter `usage.cost`); otherwise omitted.
    """
    msgs = result.get("messages", []) if isinstance(result, dict) else [result]
    totals = dict.fromkeys(
        ("input_tokens", "output_tokens", "cached_input_tokens", "cache_write_tokens",
         "audio_input_tokens", "reasoning_tokens", "audio_output_tokens"), 0)
    cost, found = None, False
    for msg in msgs:
        usage = getattr(msg, "usage_metadata", None)
        if not isinstance(usage, dict):
            continue
        found = True
        inp = usage.get("input_token_details") or {}
        out = usage.get("output_token_details") or {}
        totals["input_tokens"] += int(usage.get("input_tokens") or 0)
        totals["output_tokens"] += int(usage.get("output_tokens") or 0)
        totals["cached_input_tokens"] += int(inp.get("cache_read") or 0)
        totals["cache_write_tokens"] += int(inp.get("cache_creation") or 0)
        totals["audio_input_tokens"] += int(inp.get("audio") or 0)
        totals["reasoning_tokens"] += int(out.get("reasoning") or 0)
        totals["audio_output_tokens"] += int(out.get("audio") or 0)
        meta = getattr(msg, "response_metadata", None) or {}
        raw = meta.get("token_usage") or meta.get("usage") or {}
        c = raw.get("cost") if isinstance(raw, dict) else None
        if isinstance(c, (int, float)) and not isinstance(c, bool):
            cost = (cost or 0.0) + float(c)
    if not found:
        return {}
    if cost is not None:
        totals["cost_usd"] = cost
    return totals


class MubenchCollector:
    """Per-case model-call records (keyed by input messages) and per-suite results."""

    def __init__(self, judge: str):
        self.judge = judge
        self._lock = threading.Lock()
        self.calls: dict[str, dict] = {}
        self.suites: list[dict] = []

    def record_call(self, messages, *, latency_s: float, is_error: bool, result=None):
        rec = {"latency_s": latency_s, "is_error": is_error,
               "usage": {} if is_error else mubench_usage(result),
               "trace_id": None}
        try:
            rec["trace_id"] = mlflow.get_active_trace_id()
        except Exception:  # noqa: BLE001 - trace id is optional
            pass
        with self._lock:
            self.calls[_messages_key(messages)] = rec

    def collect_suite(self, suite_name: str, data: pd.DataFrame, run_id: str,
                      metrics: dict) -> None:
        """Call after evaluate(): pair cases with model calls and primary-scorer values."""
        suite_id, primary = MUBENCH_SUITES[suite_name]
        scores: dict[str, object] = {}
        traces = mlflow.search_traces(
            filter_string=f"run_id = '{run_id}'", return_type="list"
        )
        for trace in traces:
            for a in trace.info.assessments or []:
                if (isinstance(a, Feedback) and a.name == primary
                        and getattr(a, "valid", True) is not False):
                    scores[trace.info.trace_id] = a.value
        items = mubench_items(data)
        rows, judge_errors = [], 0
        for item in items:
            call = self.calls.get(_messages_key(item["inputs"]["messages"]))
            if call is None:
                raise RuntimeError(f"mubench: no model call recorded for item {item['item_id']}")
            usage = call["usage"]
            row = {"item_id": item["item_id"], "latency_s": call["latency_s"],
                   "input_tokens": 0, "output_tokens": 0, **usage}
            score = mubench_score(scores.get(call["trace_id"]))
            if call["is_error"]:
                row["status"] = "error"
            elif score is None:
                # Judge failed / gave no usable value: counted as error (scored 0).
                judge_errors += 1
                row["status"] = "error"
            else:
                row["status"] = "ok"
                row["score"] = score
            if not usage and not call["is_error"]:
                click.echo(f"⚠️  mubench: no token usage for item {item['item_id']}")
            rows.append(row)
        if judge_errors:
            click.echo(f"⚠️  mubench: {judge_errors} {suite_id} case(s) had no "
                       f"'{primary}' judge value; logged as status=error")
        info = {k: float(v) for k, v in metrics.items()
                if k.endswith("/mean") and k != f"{primary}/mean"
                and isinstance(v, (int, float)) and not np.isnan(v)}
        if judge_errors:
            info["judge_error/count"] = float(judge_errors)
        self.suites.append({
            "suite_id": suite_id,
            "primary": primary,
            "version": suite_version_for(items, primary, self.judge),
            "rows": rows,
            "info": info,
        })


def suite_version_for(items: list[dict], primary: str, judge: str) -> str:
    from mubench import log as mb

    return mb.suite_version(items, MUBENCH_PROMPT, primary, judge)


def _git_commit() -> str:
    import subprocess

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        sha = ""
    return sha or "unknown"


def log_mubench_run(collector: MubenchCollector, *, model: str, judge: str,
                    model_config: ModelConfig) -> str | None:
    """Write the collected suites as one mubench Run; returns its MLflow run id."""
    from mubench import log as mb

    if not collector.suites:
        click.echo("⚠️  mubench: no suites collected; nothing logged")
        return None
    gateway_url = (model_config.api_base or "").rstrip("/")
    if gateway_url.endswith("/v1"):
        gateway_url = gateway_url[: -len("/v1")]
    with mb.run(
        harness=MUBENCH_HARNESS,
        model=model,
        access_route="openrouter" if model.startswith("or/") else "foundry",
        judge=judge,
        git_commit=_git_commit(),
        gateway_url=gateway_url or None,
        gateway_key=model_config.api_key,
    ) as mrun:
        for suite in collector.suites:
            with mb.suite(suite["suite_id"], version=suite["version"],
                          primary_metric=suite["primary"]) as s:
                for row in suite["rows"]:
                    s.log_item(**row)
                for name, value in suite["info"].items():
                    s.log_info(name, value)
            click.echo(f"  mubench suite {suite['suite_id']} "
                       f"version={suite['version']} items={len(suite['rows'])}")
    run_id = mrun.info.run_id
    click.echo(f"mubench run ID: {run_id}  (publish: mubench publish {run_id})")
    return run_id


# =============================================================================
# Suite Runners
# =============================================================================

# Canonical execution order – suites always run in this sequence.
SUITE_ORDER: list[str] = ["enterprise", "hallucination", "model", "vision"]
DEFAULT_SUITES: list[str] = ["enterprise", "hallucination", "model"]


def _log_suite_metrics(run_id: str, stats: PredictionStats) -> None:
    """Compute and log comprehensive suite metrics from traces and prediction stats."""
    traces = mlflow.search_traces(
        filter_string=f"run_id = '{run_id}'", return_type="list"
    )

    # Try trace-level token usage first (populated by MLflow autolog integrations)
    trace_input, trace_output, trace_total = [], [], []
    total_cost = 0.0
    has_cost = False

    for trace in traces:
        if usage := trace.info.token_usage:
            trace_input.append(usage.get("input_tokens", 0))
            trace_output.append(usage.get("output_tokens", 0))
            trace_total.append(usage.get("total_tokens", 0))
        if cost := getattr(trace.info, "cost", None):
            if isinstance(cost, dict) and "total_cost" in cost:
                total_cost += cost["total_cost"]
                has_cost = True

    metrics: dict[str, float] = {}

    # Prefer trace-level tokens; fall back to PredictionStats (from response metadata)
    if trace_input:
        inp = np.array(trace_input)
        out = np.array(trace_output)
        tot = np.array(trace_total)
        metrics["avg_input_tokens"] = float(np.mean(inp))
        metrics["avg_output_tokens"] = float(np.mean(out))
        metrics["avg_total_tokens"] = float(np.mean(tot))
        metrics["p50_total_tokens"] = float(np.median(tot))
        metrics["p95_total_tokens"] = float(np.percentile(tot, 95))
        metrics["sum_total_tokens"] = float(np.sum(tot))
    else:
        metrics.update(stats.token_metrics())

    if has_cost:
        metrics["total_cost_usd"] = total_cost
        if trace_total:
            metrics["avg_cost_per_call_usd"] = total_cost / len(trace_total)

    metrics.update(stats.to_metrics())

    if metrics:
        mlflow.log_metrics(metrics)


def run_enterprise(
    *,
    model: str,
    judge_model_name: str,
    predict_fn: Callable,
    prediction_stats: PredictionStats,
    data_dir: Path,
    parent_run_id: str,
    epoch: int,
    limit: int | None = None,
    mubench: MubenchCollector | None = None,
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
        t0 = time.perf_counter()
        results = mlflow.genai.evaluate(
            data=data,
            predict_fn=predict_fn,
            scorers=scorers,
        )
        wall_clock = time.perf_counter() - t0
        mlflow.log_metrics(results.metrics)
        mlflow.log_metric("suite_wall_clock_s", wall_clock)
        if prediction_stats.total_calls > 0:
            mlflow.log_metric("throughput_calls_per_s", prediction_stats.total_calls / wall_clock)
        _log_suite_metrics(suite_run.info.run_id, prediction_stats)
        if mubench:
            mubench.collect_suite("enterprise", data, suite_run.info.run_id, results.metrics)


def run_hallucination(
    *,
    model: str,
    judge_model_name: str,
    predict_fn: Callable,
    prediction_stats: PredictionStats,
    data_dir: Path,
    parent_run_id: str,
    epoch: int,
    limit: int | None = None,
    mubench: MubenchCollector | None = None,
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
        t0 = time.perf_counter()
        results = mlflow.genai.evaluate(
            data=data,
            predict_fn=predict_fn,
            scorers=scorers,
        )
        wall_clock = time.perf_counter() - t0
        mlflow.log_metrics(results.metrics)
        mlflow.log_metric("suite_wall_clock_s", wall_clock)
        if prediction_stats.total_calls > 0:
            mlflow.log_metric("throughput_calls_per_s", prediction_stats.total_calls / wall_clock)
        _log_suite_metrics(suite_run.info.run_id, prediction_stats)
        if mubench:
            mubench.collect_suite("hallucination", data, suite_run.info.run_id, results.metrics)


def run_model(
    *,
    model: str,
    judge_model_name: str,
    predict_fn: Callable,
    prediction_stats: PredictionStats,
    data_dir: Path,
    parent_run_id: str,
    epoch: int,
    limit: int | None = None,
    mubench: MubenchCollector | None = None,
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
        t0 = time.perf_counter()
        results = mlflow.genai.evaluate(
            data=data,
            predict_fn=predict_fn,
            scorers=scorers,
        )
        wall_clock = time.perf_counter() - t0
        mlflow.log_metrics(results.metrics)
        mlflow.log_metric("suite_wall_clock_s", wall_clock)
        if prediction_stats.total_calls > 0:
            mlflow.log_metric("throughput_calls_per_s", prediction_stats.total_calls / wall_clock)
        _log_suite_metrics(suite_run.info.run_id, prediction_stats)
        if mubench:
            mubench.collect_suite("model", data, suite_run.info.run_id, results.metrics)


def run_vision(
    *,
    model: str,
    judge_model_name: str,
    predict_fn: Callable,
    prediction_stats: PredictionStats,
    data_dir: Path,
    parent_run_id: str,
    epoch: int,
    limit: int | None = None,
    mubench: MubenchCollector | None = None,
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
        t0 = time.perf_counter()
        results = mlflow.genai.evaluate(
            data=data,
            predict_fn=predict_fn,
            scorers=scorers,
        )
        wall_clock = time.perf_counter() - t0
        mlflow.log_metrics(results.metrics)
        mlflow.log_metric("suite_wall_clock_s", wall_clock)
        if prediction_stats.total_calls > 0:
            mlflow.log_metric("throughput_calls_per_s", prediction_stats.total_calls / wall_clock)
        _log_suite_metrics(suite_run.info.run_id, prediction_stats)
        if mubench:
            mubench.collect_suite("vision", data, suite_run.info.run_id, results.metrics)


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
@click.option(
    "--mubench",
    "use_mubench",
    is_flag=True,
    default=False,
    help="Also log a mubench Run (needs the mubench package; see README)",
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
    use_mubench: bool = False,
):
    # Build configurations for model and judge
    model_config = load_model_config(provider=provider)
    judge_config = load_judge_config(
        provider=judge_provider if judge_provider else provider
    )

    collector: MubenchCollector | None = None
    if use_mubench:
        try:
            import mubench.log  # noqa: F401
        except ImportError:
            click.echo(
                "❌ --mubench needs the mubench package, e.g.\n"
                "   uv run --with 'mubench @ git+https://github.com/mfc-innersource/"
                "mubench@feature/mubench-service#subdirectory=tools/mubench' playbook.py ..."
            )
            raise SystemExit(1)
        collector = MubenchCollector(judge=judge)

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
        _current_stats: list[PredictionStats | None] = [None]

        def _extract_usage(result) -> dict | None:
            # LangGraph agent returns dict with "messages"; unwrap to last message
            msg = result
            if isinstance(result, dict) and "messages" in result:
                msgs = result["messages"]
                if isinstance(msgs, list) and msgs:
                    msg = msgs[-1]
            elif isinstance(result, list) and result:
                msg = result[-1]
            usage = getattr(msg, "usage_metadata", None)
            if usage and isinstance(usage, dict):
                return usage
            resp = getattr(msg, "response_metadata", None)
            if resp and isinstance(resp, dict) and "usage" in resp:
                u = resp["usage"]
                return {
                    "input_tokens": u.get("prompt_tokens", 0),
                    "output_tokens": u.get("completion_tokens", 0),
                    "total_tokens": u.get("total_tokens", 0),
                }
            return None

        def predict_fn(messages):
            with concurrency_sem:
                t0 = time.perf_counter()
                try:
                    result = agent.invoke({"messages": _sanitize_messages(messages)})
                    elapsed = time.perf_counter() - t0
                    usage = _extract_usage(result)
                    response_str = str(result) if result else ""
                    if collector:
                        collector.record_call(messages, latency_s=elapsed, is_error=False,
                                              result=result)
                    if not response_str.strip():
                        if _current_stats[0]:
                            _current_stats[0].record(elapsed, "", is_error=False, usage=usage)
                        return "[Model returned empty response]"
                    if _current_stats[0]:
                        _current_stats[0].record(elapsed, response_str, is_error=False, usage=usage)
                    return result
                except Exception as e:
                    elapsed = time.perf_counter() - t0
                    if _current_stats[0]:
                        _current_stats[0].record(elapsed, "", is_error=True)
                    if collector:
                        collector.record_call(messages, latency_s=elapsed, is_error=True)
                    return f"[Model error: {e}]"

        # Run selected suites sequentially.
        # MLflow's evaluate() parallelizes predict_fn calls internally;
        # the semaphore above controls concurrency. Inter-suite parallelism
        # is not used because MLflow's trace manager singleton causes
        # contention/data loss when multiple evaluate() calls overlap.
        for suite_name in to_run:
            click.echo(f"\n▶ Running suite: {suite_name}")
            _current_stats[0] = PredictionStats()
            SUITE_RUNNERS[suite_name](
                model=model,
                judge_model_name=judge_model_name,
                predict_fn=predict_fn,
                prediction_stats=_current_stats[0],
                data_dir=data_path,
                parent_run_id=run.info.run_id,
                epoch=epoch_time,
                limit=limit,
                mubench=collector,
            )

        # Mark run as complete so report.py can identify fully successful runs
        mlflow.set_tag("playbook.status", "complete")
        click.echo(f"\n✅ All requested suites complete. Epoch: {epoch_time}")

    # Logged after the playbook's run has closed: mubench.log.run() switches the
    # active experiment and starts its own top-level run.
    if collector:
        click.echo("\nLogging mubench Run:")
        log_mubench_run(collector, model=model, judge=judge, model_config=model_config)


if __name__ == "__main__":
    evaluate()
