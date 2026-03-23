# Repo Cleanup: Maintainability + Correctness

**Date:** 2026-03-23
**Scope:** Delete dead files, fix `--eval` CLI, update batch runner, update docs

## Context

`playbook.py` is the eval orchestrator for benchmarking LLMs across 4 suites (enterprise, hallucination, model, vision). The `--eval` CLI works but is verbose, and `run_eval_suite.sh` references a nonexistent `--with-vision` flag. Most disposable shell scripts have already been deleted; only untracked remnants remain.

## Changes

### 1. Delete Files

Only 1 file actually needs deletion (the rest were already removed):

| File | Status | Action |
|------|--------|--------|
| `mistral-eval-findings.md` | Exists (untracked, in .gitignore) | Delete |

### 2. Fix `--eval` CLI in `playbook.py`

**Current behavior:**
- `--eval` / `-e` accepts `multiple=True` with `click.Choice(SUITE_ORDER)`
- Must repeat flag: `--eval enterprise --eval model --eval vision`
- If omitted, runs ALL suites (including vision)

**New behavior:**
- Single `--eval` option accepting comma-separated values or `all`
- Default (omitted) = `enterprise,hallucination,model` (no vision)
- `--eval all` = all 4 suites including vision
- `--eval enterprise,vision` = just those two
- Validation: reject unknown suite names with clear error message

**Edge cases:**
- `--eval vision` alone = valid, runs only vision
- `--eval enterprise,enterprise` = deduplicated naturally by canonical-order filter
- `--eval all,vision` = treat `all` as expanding to full set, ignore redundant names
- `--eval ""` = validation error

**Examples:**
```bash
# Default: 3 suites, no vision
uv run playbook.py --model claude-opus-4-6 --judge gpt-5.1

# All suites including vision
uv run playbook.py --model claude-opus-4-6 --judge gpt-5.1 --eval all

# Specific suites
uv run playbook.py --model claude-opus-4-6 --judge gpt-5.1 --eval enterprise,vision
```

**Implementation:** Replace the `click.option` for `--eval` with a `callback=` that splits on commas and validates against `SUITE_ORDER`. Add an `all` keyword. Change the default from `()` (meaning all) to `None` (meaning default set without vision).

### 3. Update `run_eval_suite.sh`

- Replace `--with-vision` with `--eval all`
- Keep everything else (logging, timing, batch structure)
- Note: this file is currently in `.gitignore` — leave as-is (local batch runner)

### 4. Clean up `playbook.py` deps

Remove 3 unused dependencies from inline script header:
- `psutil` — imported but never used
- `py-readability-metrics` — not referenced anywhere
- `rouge` — the `rouge` scorer in playbook.py is hand-rolled (string splitting), not using this pip package

Also remove any corresponding unused `import` statements.

### 5. Update SKILL.md

Update `.skills/model-eval-playbook/SKILL.md` to reflect new `--eval` semantics:
- Line 53: change "Omit `--eval` to run all suites" → "Omit `--eval` to run default suites (enterprise, hallucination, model). Use `--eval all` for all including vision."
- Line 54: change `-e enterprise -e vision` → `--eval enterprise,vision`
- Line 101: update `--with-vision` troubleshooting row → reference `--eval all`

### 6. Clean up `.gitignore`

- Remove `mistral-eval-findings.md` entry (file being deleted)
- Keep `run_eval_suite.sh` entry (still a local-only batch runner)

## Out of Scope

- Parallelizing suite execution
- Adding tests
- Splitting `playbook.py` into modules
- Modifying `report.py`
- Modifying data files or MLflow structure
- Modifying scorer logic

## Risks

- **Low:** Changing `--eval` default from "all suites" to "no vision" could surprise users who relied on the old default. Mitigated by explicit `--eval all` and updating the batch runner.
