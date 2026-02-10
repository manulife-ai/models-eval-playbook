# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click",
#     "databricks-sdk",
#     "mlflow",
#     "pandas",
#     "tabulate",
# ]
# ///
import re
from collections import defaultdict

import click
import mlflow
import pandas as pd
from tabulate import tabulate

# Categories and the metrics we expect from each
CATEGORY_METRICS = {
    "enterprise": ["safety/mean", "follows_instructions/mean"],
    "hallucination": ["follows_instructions/mean"],
    "vision": [
        "safety/mean",
        "correctness/mean",
        "follows_instructions/mean",
        "field_accuracy/mean",
    ],
    "model": [
        "safety/mean",
        "correctness/mean",
        "follows_instructions/mean",
        "bleu/mean",
        "rouge/mean",
        "cosine_similarity/mean",
        "avg_input_tokens",
        "avg_output_tokens",
        "avg_total_tokens",
    ],
}

KNOWN_CATEGORIES = set(CATEGORY_METRICS.keys())


def parse_experiment_id(value: str) -> str:
    """Extract experiment ID from a raw ID string or a Databricks URL."""
    match = re.search(r"/experiments/(\d+)", value)
    if match:
        return match.group(1)
    if value.isdigit():
        return value
    raise click.BadParameter(
        f"Cannot parse experiment ID from: {value}. "
        "Provide a numeric ID or a Databricks experiment URL."
    )


def extract_category(run_name: str) -> str | None:
    """Extract the category from a child run name like 'playbook-{model}-{category}-{epoch}'.

    The category is the second-to-last segment when split by '-'.
    """
    for category in KNOWN_CATEGORIES:
        if f"-{category}-" in run_name:
            return category
    return None


@click.command()
@click.option(
    "--experiment",
    required=True,
    help="Experiment ID (numeric) or full Databricks experiment URL.",
)
@click.option(
    "--run-id",
    default=None,
    help="Optional parent run ID to report on a single run instead of all runs.",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Optional path to write CSV output (e.g. report.csv).",
)
def report(experiment: str, run_id: str | None, output: str | None):
    """Generate a tabular comparison of evaluation metrics across models."""

    experiment_id = parse_experiment_id(experiment)
    if run_id:
        click.echo(f"Querying experiment {experiment_id}, parent run {run_id} ...")
    else:
        click.echo(f"Querying experiment {experiment_id} ...")

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        output_format="pandas",
    )

    if runs_df.empty:
        click.echo("No runs found for this experiment.")
        raise SystemExit(1)

    # Identify parent vs child runs via the mlflow.parentRunId tag
    parent_tag = "tags.mlflow.parentRunId"
    if parent_tag not in runs_df.columns:
        click.echo("No child runs found — nothing to report.")
        raise SystemExit(1)

    parent_runs = runs_df[runs_df[parent_tag].isna()]
    child_runs = runs_df[runs_df[parent_tag].notna()]

    # Filter to a single parent run if --run-id is provided
    if run_id:
        parent_runs = parent_runs[parent_runs["run_id"] == run_id]
        if parent_runs.empty:
            click.echo(f"Parent run {run_id} not found in experiment.")
            raise SystemExit(1)
        child_runs = child_runs[child_runs[parent_tag] == run_id]
        if child_runs.empty:
            click.echo(f"No child runs found for parent run {run_id}.")
            raise SystemExit(1)

    if parent_runs.empty:
        click.echo("No parent runs found.")
        raise SystemExit(1)

    # When reporting all runs, keep only the latest parent run per model
    # that has child runs with actual metric data, to avoid picking an
    # incomplete/in-progress run over a fully completed earlier one.
    if not run_id:
        metric_cols = [c for c in child_runs.columns if c.startswith("metrics.")]
        children_with_metrics = child_runs[
            child_runs[metric_cols].notna().any(axis=1)
        ]
        parent_ids_with_metrics = set(children_with_metrics[parent_tag].unique())

        parent_runs = parent_runs.sort_values("start_time", ascending=False)

        seen_models: dict[str, str] = {}  # model_name -> run_id (latest with data)
        for _, row in parent_runs.iterrows():
            model_name = row.get("params.model", None)
            if not model_name:
                name = row.get("tags.mlflow.runName", "")
                parts = name.split("-")
                model_name = "-".join(parts[1:-1]) if len(parts) >= 3 else name

            if model_name not in seen_models and row["run_id"] in parent_ids_with_metrics:
                seen_models[model_name] = row["run_id"]

        latest_ids = set(seen_models.values())
        parent_runs = parent_runs[parent_runs["run_id"].isin(latest_ids)]
        child_runs = child_runs[child_runs[parent_tag].isin(latest_ids)]

        click.echo(f"Using latest run for each model: {', '.join(seen_models.keys())}")

    # Build a lookup: parent_run_id -> model name
    parent_model = {}
    for _, row in parent_runs.iterrows():
        rid = row["run_id"]
        model_name = row.get("params.model", None)
        if model_name:
            parent_model[rid] = model_name
        else:
            # Fallback: try to parse model from run name (playbook-{model}-{epoch})
            name = row.get("tags.mlflow.runName", "")
            parts = name.split("-")
            if len(parts) >= 3:
                parent_model[rid] = "-".join(parts[1:-1])
            else:
                parent_model[rid] = name

    # Collect metrics: { "category/metric": { model: value } }
    table_data: dict[str, dict[str, float | None]] = defaultdict(dict)
    all_models: set[str] = set()

    for _, row in child_runs.iterrows():
        parent_id = row[parent_tag]
        model_name = parent_model.get(parent_id, "unknown")
        all_models.add(model_name)

        run_name = row.get("tags.mlflow.runName", "")
        category = extract_category(run_name)
        if category is None:
            continue

        expected = CATEGORY_METRICS.get(category, [])
        for metric_key in expected:
            col = f"metrics.{metric_key}"
            if col in row.index and pd.notna(row[col]):
                label = f"{category}/{metric_key}"
                table_data[label][model_name] = row[col]

    if not table_data:
        click.echo("No metrics found in child runs.")
        raise SystemExit(1)

    # Sort models alphabetically and metrics by category order
    sorted_models = sorted(all_models)

    # Build ordered metric list following CATEGORY_METRICS declaration order
    ordered_metrics = []
    for category, metrics in CATEGORY_METRICS.items():
        for m in metrics:
            label = f"{category}/{m}"
            if label in table_data:
                ordered_metrics.append(label)

    # Build table rows
    rows = []
    for metric in ordered_metrics:
        row = [metric]
        for model in sorted_models:
            val = table_data[metric].get(model)
            if val is None:
                row.append("-")
            else:
                row.append(f"{val:.4f}")
        rows.append(row)

    headers = ["Metric"] + sorted_models

    click.echo()
    click.echo(tabulate(rows, headers=headers, tablefmt="grid"))
    click.echo()

    if output:
        df_out = pd.DataFrame(rows, columns=headers)
        df_out.to_csv(output, index=False)
        click.echo(f"Report written to {output}")


if __name__ == "__main__":
    report()
