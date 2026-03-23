# Repo Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up dead files, fix the `--eval` CLI to accept comma-separated suites with a sensible default, and remove unused dependencies.

**Architecture:** All changes are in-place edits to existing files plus one file deletion. No new modules or structural changes. `playbook.py` stays as a single file.

**Tech Stack:** Python 3.12+, Click (CLI), uv (runner)

**Spec:** `docs/superpowers/specs/2026-03-23-repo-cleanup-design.md`

---

### Task 1: Delete dead file and clean .gitignore

**Files:**
- Delete: `mistral-eval-findings.md`
- Modify: `.gitignore:75`

- [ ] **Step 1: Delete `mistral-eval-findings.md`**

```bash
rm mistral-eval-findings.md
```

- [ ] **Step 2: Remove stale .gitignore entry**

In `.gitignore`, remove line 75 (`mistral-eval-findings.md`). Keep line 76 (`run_eval_suite.sh`).

After edit, `.gitignore` lines 74-76 should be:
```
.env.bak
run_eval_suite.sh
```

- [ ] **Step 3: Commit**

```bash
git add mistral-eval-findings.md .gitignore
git commit -m "chore: delete mistral-eval-findings.md, clean .gitignore"
```

---

### Task 2: Remove unused deps from playbook.py

**Files:**
- Modify: `playbook.py:1-18` (inline script deps)

- [ ] **Step 1: Remove 3 unused dependencies**

In `playbook.py` lines 1-18, remove these 3 lines from the inline `dependencies` list:
- Line 14: `#     "py-readability-metrics",`
- Line 15: `#     "rouge",`
- Line 17: `#     "psutil",`

After edit, lines 1-16 should be:
```python
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
```

- [ ] **Step 2: Verify playbook.py still parses**

```bash
python -c "import ast; ast.parse(open('playbook.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add playbook.py
git commit -m "chore: remove unused deps (psutil, py-readability-metrics, rouge)"
```

---

### Task 3: Fix --eval CLI option

**Files:**
- Modify: `playbook.py:474` (SUITE_ORDER constant area)
- Modify: `playbook.py:718-724` (--eval click option)
- Modify: `playbook.py:743` (evals parameter type)
- Modify: `playbook.py:752-754` (suite resolution logic)

- [ ] **Step 1: Add DEFAULT_SUITES constant**

After the `SUITE_ORDER` line (line 474), add:

```python
DEFAULT_SUITES: list[str] = ["enterprise", "hallucination", "model"]
```

- [ ] **Step 2: Add parse_eval callback**

Add this function above the `@click.command` decorator (before line 709):

```python
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
```

- [ ] **Step 3: Replace the --eval click option**

Replace lines 718-724:
```python
@click.option(
    "--eval", "-e",
    "evals",
    multiple=True,
    type=click.Choice(SUITE_ORDER, case_sensitive=False),
    help="Evaluation suite(s) to run. Repeatable. If omitted, all suites run.",
)
```

With:
```python
@click.option(
    "--eval", "-e",
    "evals",
    default=None,
    callback=parse_eval,
    expose_value=True,
    is_eager=False,
    help="Comma-separated suites or 'all'. Default: enterprise,hallucination,model",
)
```

- [ ] **Step 4: Update evaluate function signature**

Change line 743 from:
```python
    evals: tuple[str, ...] = (),
```
To:
```python
    evals: list[str] = None,
```

- [ ] **Step 5: Update suite resolution logic**

Replace lines 752-754:
```python
    # Resolve which suites to run, always in canonical order
    requested = [e.lower() for e in evals] if evals else SUITE_ORDER
    to_run = [s for s in SUITE_ORDER if s in requested]
```

With:
```python
    # evals is already parsed/validated by the parse_eval callback
    to_run = evals
```

Note: keep the existing `if not to_run` guard at lines 756-758 as defensive validation.

- [ ] **Step 6: Verify playbook.py parses**

```bash
python -c "import ast; ast.parse(open('playbook.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 7: Verify --help output**

```bash
uv run playbook.py --help
```

Expected: `--eval` shows in help with description "Comma-separated suites or 'all'..."

- [ ] **Step 8: Commit**

```bash
git add playbook.py
git commit -m "fix: --eval accepts comma-separated suites, defaults to no vision"
```

---

### Task 4: Update run_eval_suite.sh

**Files:**
- Modify: `run_eval_suite.sh:71`

- [ ] **Step 1: Replace --with-vision with --eval all**

Change line 71 from:
```bash
    --with-vision \
```
To:
```bash
    --eval all \
```

Note: `run_eval_suite.sh` is in `.gitignore` (local-only batch runner). No commit needed — this is a local edit only.

---

### Task 5: Update SKILL.md

**Files:**
- Modify: `.skills/model-eval-playbook/SKILL.md:49-54,101`

- [ ] **Step 1: Update eval usage in Task 1 section**

Replace lines 49-54:
```markdown
  [--eval <suite>]... \
  [--limit N]
```

- Omit `--eval` to run all suites (enterprise -> hallucination -> model -> vision)
- Use `-e enterprise -e vision` to run specific suites
```

With:
```markdown
  [--eval <suites>] \
  [--limit N]
```

- Omit `--eval` to run default suites (enterprise, hallucination, model)
- Use `--eval all` to run all suites including vision
- Use `--eval enterprise,vision` to run specific suites (comma-separated)
```

- [ ] **Step 2: Update troubleshooting table**

Replace line 101:
```markdown
| `--with-vision` not recognized | Stale run scripts using old CLI flag | Use `--eval vision` instead |
```

With:
```markdown
| `--with-vision` not recognized | Stale run scripts using old CLI flag | Use `--eval all` to include vision |
```

- [ ] **Step 3: Commit**

```bash
git add .skills/model-eval-playbook/SKILL.md
git commit -m "docs: update SKILL.md for new --eval semantics"
```
