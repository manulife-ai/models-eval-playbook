# MLflow Diagnosis API Patterns

All requests require a Bearer token from Databricks CLI:

```bash
TOKEN=$(/opt/homebrew/bin/databricks auth token --profile <PROFILE> | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
HOST="<DATABRICKS_HOST>"  # from .env
```

## 1. Get run info

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  "${HOST}/api/2.0/mlflow/runs/get?run_id=<RUN_ID>" | python3 -m json.tool
```

Key fields: `run.info.run_name`, `run.info.status`, `run.data.tags` (look for `mlflow.parentRunId`).

## 2. List child runs of a parent

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  "${HOST}/api/2.0/mlflow/runs/search" \
  -d '{"experiment_ids":["<EXP_ID>"],"filter":"tags.mlflow.parentRunId = '\''<PARENT_RUN_ID>'\''","max_results":10}' \
  -H "Content-Type: application/json"
```

## 3. Get traces for a run (find errors)

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  "${HOST}/api/2.0/mlflow/traces?experiment_ids=<EXP_ID>&filter=run_id%20%3D%20%27<RUN_ID>%27&max_results=50" \
  -H "Content-Type: application/json"
```

Parse traces to find ERROR state:
```python
for t in traces:
    tags = {tag['key']: tag['value'] for tag in t.get('tags', [])}
    test_id = tags.get('id', '?')
    status = t.get('status', '?')  # OK or ERROR
    # Check assessment tags for error_message
```

## 4. Get span events (actual exception)

Use the ajax API to get spans with full exception details:

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  "${HOST}/ajax-api/2.0/mlflow/get-trace-artifact?path=spans&request_id=<TRACE_ID>"
```

Parse the exception from span events:
```python
spans = data.get('spans', data)
for span in spans:
    for event in span.get('events', []):
        if event.get('name') == 'exception':
            attrs = event['attributes']
            print(attrs.get('exception.type'))
            print(attrs.get('exception.message'))
            print(attrs.get('exception.stacktrace'))
```

## 5. Count errors by type across all traces

```python
import re
from collections import Counter

errors = re.findall(r'"error_message":"([^"]+)"', raw_response_text)
for msg, count in Counter(errors).most_common():
    print(f'[{count}x] {msg[:120]}')
```

## 6. Compare two runs

Fetch metrics for both runs and diff:
```bash
for RUN_ID in <RUN_A> <RUN_B>; do
  curl -s -H "Authorization: Bearer $TOKEN" \
    "${HOST}/api/2.0/mlflow/runs/get?run_id=${RUN_ID}" | \
    python3 -c "import sys,json; [print(f\"{m['key']}: {m['value']}\") for m in json.load(sys.stdin)['run']['data']['metrics']]"
done
```
