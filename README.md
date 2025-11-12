# How to run the playbook

## Prerequisite

1. uv installed on machine
2. mlflow server available (used for tracking runs, not required but helpful. Can also run a local mlflow server)

## Setup

Verify the provider environment variables are available.

```sh
# [Optional] If you are using MLFlow UI to view the results.
MLFLOW_TRACKING_URI="" # Use value http://localhost:5000/ for local mlflow
MLFLOW_EXPERIMENT_ID="" # [OPTIONAL] defaults to the Default experiment on local mlflow server

# [Optional] If you are using Databricks UC as a tracking server Include the following environments and ensure you are logged in to the profile.
# databricks auth login --profile <DATBARICKS_AUTH_PROFILE> --host <DATABRICKS_URL>
MLFLOW_TRACKING_URI=databricks://<DATBARICKS_AUTH_PROFILE>
MLFLOW_REGISTRY_URI=databricks-uc
DATABRICKS_HOST=<DATABRICKS_URL>
MLFLOW_EXPERIMENT_ID=<EXPERIMENT_ID>

# [REQUIRED] Evaluators must provide one of the following:
# Add the following environment variables if you are using OpenAI API Compliant Providers. Eg:
# 1. Azure OpenAI/Foundry: Example https://<INSTANCE_NAME>.openai.azure.com/openai/v1 (Note that Azure OpenAI and Foundry instance provides provide openai api compliant API at the following API Base <URL>/openai/v1)
# 2. OpenRouter: https://openrouter.ai/api/v1
# 3. Ollama: http://localhost:11434/v1/
# 4. Alicloud: https://[ServiceCode].[RegionID].aliyuncs.com/v1
# 5. Anthropic (** will require some additional dependencies... see the Playbook owners for support.)
# 6. Amazon Bedrock (** will require some additional dependencies... see the Playbook owners for support.)
# 7. Cohere (** will require some additional dependencies... see the Playbook owners for support.)
# 8. Together AI (** will require some additional dependencies... see the Playbook owners for support.)
# 9. Any other providers such as Google Gemini, xAI, Mistral, and more. (** will require some additional dependencies... see the Playbook owners for support.)

OPENAI_API_VERSION=preview
OPENAI_API_BASE=<YOUR_ENDPOINT_HERE> 
OPENAI_API_KEY=<YOUR_API_KEY>
```

## Running Evaluations
Assuming the above prerequisites are met and the setup is done:

```sh
uv run --script playbook.py --model <MODEL_NAME> --judge <MODEL_NAME> # Provider can also be customized if you anything other than OpenAI --provider <provider>. Additional Library Requirements may apply. See developer for full setup details.
```

Example for running an eval on gpt-4o and gpt-5 as judge
```sh
uv run --script playbook.py --model gpt-4o --judge gpt-5
```