# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click",
#     "databricks-agents",
#     "databricks-sdk",
#     "langchain",
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
import time
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


def to_langchain_model_name(model: str, provider: str):
    return f"{provider}:{model}"


def to_litellm_model_name(model: str, provider: str):
    return f"{provider}:/{model}"


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


def run_evaluations(
    model: str,
    judge: str,
    data: pd.DataFrame,
    predict_fn: Callable,
    scorers: list[Scorer] = [],
    provider: str = "openai",
) -> pd.DataFrame:
    epoch_time = int(time.time())
    with mlflow.start_run(
        run_name=f"playbook-eval-{model}-{epoch_time}",
        description=f"Evaluation run for model {model} with judge {judge}",
    ) as run:
        mlflow.log_params(
            {
                "model": model,
                "judge": judge,
                "provider": provider,
                "run_date": datetime.datetime.now().isoformat(),
                "num_cases": len(data),
            }
        )
        with TemporaryDirectory(delete=True) as temp_dir:
            temp_file = f"{temp_dir}/evaluation.json"
            data.to_json(temp_file, orient="records")
            mlflow.log_artifact(temp_file, artifact_path="data")
        # Define the evaluation functions or scorers here
        safety = Safety(
            model=to_litellm_model_name(judge, provider),
        )
        correctness = Correctness(
            model=to_litellm_model_name(judge, provider),
        )
        clarity = Guidelines(
            model=to_litellm_model_name(judge, provider),
            name="clarity",
            guidelines=["The response must be clear, coherent, and concise"],
        )
        english = Guidelines(
            model=to_litellm_model_name(judge, provider),
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
                    model=to_litellm_model_name(judge, provider),
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


@click.command
@click.option("--model", help="Model to evaluate", required=True)
@click.option("--judge", help="Judge to use for evaluation", required=True)
@click.option("--provider", help="Provider of the model", default="openai")
@click.option(
    "--with-vision",
    default=False,
    help="Whether to include vision evaluations",
    is_flag=True,
)
@click.option(
    "--data-dir",
    default="./data",
    help="Path to the data directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
def evaluate(
    model: str,
    judge: str,
    data_dir: str,
    provider: str = "openai",
    with_vision: bool = False,
):
    print(
        f"Starting evaluation for model: {model}, judge: {judge}, provider: {provider}, with_vision: {with_vision}, data_dir: {data_dir}"
    )
    epoch_time = int(time.time())
    with mlflow.start_run(run_name=f"playbook-{model}-{epoch_time}") as run:
        mlflow.log_params(
            {
                "model": model,
                "judge": judge,
                "provider": provider,
                "run_date": datetime.datetime.now().isoformat(),
            }
        )
        print(f"Created parent run with ID: {run.info.run_id}")
        enterpise_data = pd.read_json(
            Path(data_dir) / "enterprise.json", orient="records"
        )
        hallucination_data = pd.read_json(
            Path(data_dir) / "hallucination.json", orient="records"
        )
        vision_data = pd.read_json(Path(data_dir) / "vision.json", orient="records"
        )
        model_data = pd.read_json(Path(data_dir) / "data.json", orient="records")
        mlflow.log_artifact(Path(data_dir) / "enterprise.json", artifact_path="data")
        mlflow.log_artifact(Path(data_dir) / "hallucination.json", artifact_path="data")
        mlflow.log_artifact(Path(data_dir) / "vision.json", artifact_path="data")
        mlflow.log_artifact(Path(data_dir) / "data.json", artifact_path="data")
        agent = create_agent(model=to_langchain_model_name(model, provider))

        def predict_fn(messages):
            return agent.invoke({"messages": messages})

        with mlflow.start_run(
            run_name=f"playbook-{model}-enterprise-{epoch_time}",
            description=f"Comprehensive enterprise evaluation run for model {model} with judge {judge}",
            parent_run_id=run.info.run_id,
            nested=True,
        ) as enterprise_run:
            print(f"Created enterprise run with ID: {enterprise_run.info.run_id}")
            enterprise_scorers = [
                ExpectationsGuidelines(
                    model=to_litellm_model_name(judge, provider),
                    name="follows_instructions",
                ),
                Safety(
                    model=to_litellm_model_name(judge, provider),
                ),
            ]
            results = mlflow.genai.evaluate(
                data=enterpise_data,
                predict_fn=predict_fn,
                scorers=enterprise_scorers,
            )
            mlflow.log_metrics(results.metrics, run_id=run.info.run_id)

        # Hallucination evaluations
        with mlflow.start_run(
            run_name=f"playbook-{model}-hallucination-{epoch_time}",
            description=f"Comprehensive hallucination evaluation run for model {model} with judge {judge}",
            parent_run_id=run.info.run_id,
            nested=True,
        ) as hallucination_run:
            print(f"Created hallucination run with ID: {hallucination_run.info.run_id}")
            hallucination_scorers = [
                ExpectationsGuidelines(
                    model=to_litellm_model_name(judge, provider),
                    name="follows_instructions",
                ),
            ]
            results = mlflow.genai.evaluate(
                data=hallucination_data,
                predict_fn=predict_fn,
                scorers=hallucination_scorers,
            )
            mlflow.log_metrics(results.metrics, run_id=run.info.run_id)

        # Vision evaluations
        if with_vision:
            with mlflow.start_run(
                run_name=f"playbook-{model}-vision-{epoch_time}",
                description=f"Comprehensive vision evaluation run for model {model} with judge {judge}",
                parent_run_id=run.info.run_id,
                nested=True,
            ) as vision_run:
                print(f"Created vision run with ID: {vision_run.info.run_id}")
                vision_scorers = [
                    Safety(
                        model=to_litellm_model_name(judge, provider),
                    ),
                    Correctness(
                        model=to_litellm_model_name(judge, provider),
                    ),
                    ExpectationsGuidelines(
                        model=to_litellm_model_name(judge, provider),
                        name="follows_instructions",
                    ),
                ]
                results = mlflow.genai.evaluate(
                    data=vision_data,
                    predict_fn=predict_fn,
                    scorers=vision_scorers,
                )
                mlflow.log_metrics(results.metrics, run_id=run.info.run_id)
        # Model evaluations
        with mlflow.start_run(
            run_name=f"playbook-{model}-model-{epoch_time}",
            description=f"Comprehensive model evaluation run for model {model} with judge {judge}",
            parent_run_id=run.info.run_id,
            nested=True,
        ) as model_run:
            print(f"Created model run with ID: {model_run.info.run_id}")
            model_scorers = [
                ExpectationsGuidelines(
                    model=to_litellm_model_name(judge, provider),
                    name="follows_instructions",
                ),
                Safety(
                    model=to_litellm_model_name(judge, provider),
                ),
                Correctness(
                    model=to_litellm_model_name(judge, provider),
                ),
                bleu,
                rouge,
                cosine_similarity,
            ]
            results = mlflow.genai.evaluate(
                data=model_data,
                predict_fn=lambda messages: agent.invoke({"messages": messages}),
                scorers=model_scorers,
            )
            traces = mlflow.search_traces(
                filter_string=f"run_id = '{model_run.info.run_id}'", return_type="list"
            )
            mlflow.log_metrics(avg_token_usage(traces))
            mlflow.log_metrics(results.metrics, run_id=run.info.run_id)


if __name__ == "__main__":
    evaluate()
