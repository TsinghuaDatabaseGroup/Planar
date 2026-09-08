---
name: env-setup-skill
description: Record or verify the runtime environment a downstream non-interactive SQPE agent must use. Use this as runner Phase 0 before generating or running a per-system pipeline skill. It captures the activation command, existing env/config files and actual field names, cost/token tracking fields, import probes, and optional setup notes in `<system_root>/.agent/ENV.md`. It does not standardize `.env` variable names, create native aliases, rewrite model config, or install dependencies unless the user explicitly asks for repair.
---

# SQPE Environment + Model Setup

This skill is runner **Phase 0**. It records enough environment information for later non-interactive agents to use the correct system runtime and preserve cost tracking. It does not force every system into one `.env` schema.

Pipeline composition, operator-level usage, and benchmark task execution belong to the per-system pipeline-skill produced separately by `new-system-operator-skill-creator`.

## Why a separate skill

Every SQPE differs in environment layout. The stable handoff is not a universal env schema; it is a short record of what environment to activate, which config files exist, and what native setup pattern the target system expects.

## Outputs (the contract)

When this skill finishes, the system root contains:

1. **Activation command** — the exact command the user uses, or `current shell: <python executable>` if the current shell is already the intended environment.
2. **System identity** — source root, system name, and Python package/import root.
3. **Env/config files** — actual paths and actual field names already used by the system. Secrets are redacted; aliases are not invented.
4. **Cost/token/concurrency tracking** — actual fields or native API attributes used for model id, model provider/adaptor format, model-call kwargs such as thinking controls, context/output token caps, per-token cost, model-call concurrency, rate limits, and cost accounting. Missing values are recorded as caveats.
5. **Verification probe** — the import/version probe command and result, including the resolved Python executable path.
6. **Setup notes** — optional. Include a setup helper path only if one already exists or the system truly needs one to expose its native API cleanly.
7. **`<system_root>/.agent/ENV.md`** — the concise handoff file for Phase 1 and Phase 2.

If this record is missing or wrong, downstream agents may run in the wrong Python environment or use the wrong system configuration.

## Workflow

### Step 0: Scope

When the runner provides `--system-env-note` and `--env-path`, treat those arguments as the user's confirmed Phase 0 inputs. Do not ask again unless the values are missing, unreadable, or contradictory.

Confirm with the user:

- Path to the system source.
- Short system name.
- Python package/import root.
- Env/config file path(s) the system already uses.
- Environment activation note, such as `conda activate <name>`.

Runner-supported non-interactive Phase 0:

This mode should read the chosen env/config file, write `<system_root>/.agent/ENV.md`, and print warnings for missing model, cost/token, or concurrency fields. The user can edit `.env` and rerun Phase 0 until the warnings are resolved.

### Fast path: existing environment

If the user already has a working environment, **do not create or reinstall anything**. Treat environment setup as a verification-and-recording step.

Use this fast path when either:

- The user provides an activation command, or
- The current shell already imports the engine successfully, or
- The system root already has an env marker such as `.venv/`, `uv.lock`, `poetry.lock`, `environment.yml`, or a documented active conda/venv setup and the import probe passes.

Fast-path behavior:

1. Record the activation command exactly as provided. If the current shell is already active and the user wants to keep it, record `current shell: <python executable>` plus the command the user uses externally, if known.
2. Run only the import probe and any minimal version/config probe needed to confirm the engine is usable.
3. Skip dependency installation entirely unless the import probe fails and the user explicitly approves repair.
4. If env/config files exist, preserve their current field names and values. Record missing or uncertain fields; do not normalize names just for the runner.
5. Record cost/token/concurrency tracking fields exactly as they exist, including cost source if it comes from a library such as LiteLLM rather than `.env`.
6. Record the resolved Python executable from the probe (for example `/path/to/env/bin/python`) and prefer it as the downstream non-interactive launch command with `-u`.
7. If no explicit model-call concurrency or rate-limit setting exists, ask the user whether they want to add stable runtime fields to the chosen env/config file. Do not silently rely on a library default without recording it.
8. If a setup helper already exists, inspect and smoke-test it. Patch only if the user asks or if the helper is clearly broken.
9. Always update `<system_root>/.agent/ENV.md` so downstream pipeline skills know which existing environment and helper were verified.

### Step 1: Python environment

> "Do you already have a Python environment for this system? If yes, share the activation command (e.g., `conda activate <name>`, `source .venv/bin/activate`). If no, I'll find the install path the system documents and walk you through creating one."

- **Existing env:** use the fast path above. Record the activation command. Activate it only if needed. Verify the engine imports (`python -c "import <pkg>; print(getattr(<pkg>, '__version__', 'ok'))"`). If the current shell already imports the engine, do not create a new env. If import fails, explain the failure and ask before installing or repairing.
- **No env:** read `pyproject.toml` / `requirements.txt` / `environment.yml` / `setup.py` / the engine's README install section. Present the canonical install path back to the user (conda env, `uv venv`, plain `python -m venv` + `pip install -e .`, ...). **Run installation only after explicit user confirmation.** Verify with the import probe above.

Capture the activation command verbatim — Step 5 records it.

### Step 2: Env/config files

Inspect the target system's actual env/config files and record their current shape.

- Preserve existing field names.
- Redact secret values.
- Record which relevant fields are present, missing, or uncertain.
- Record whether the system's native model constructor expects a provider-qualified model id, deployment name, or other adaptation that differs from the raw model field in the env/config file. Do not rename the env field just to satisfy that API; document the transformation in the native setup snippet.
- Record model-call adapter/generation kwargs that affect correctness or latency, such as `extra_body`, nested `chat_template_kwargs`, `enable_thinking`, `reasoning`, `temperature`, or provider-specific flags. If present, the native setup snippet must show how to parse and pass them to the system model/operator object without flattening nested provider payloads.
- Record cost/token/concurrency related fields when present: model id, input/output cost, max context tokens, max output tokens, model-call concurrency, batch size, RPM/TPM limits, and the system's native cost accounting source.
- If concurrency/rate-limit fields are absent, explicitly ask the user whether to keep the system default or add stable fields to the chosen env/config file. Use system-native field names when they exist; otherwise use clear project-local names such as `<SYSTEM>_MAX_BATCH_SIZE`, `<SYSTEM>_RATE_LIMIT_RPM`, and `<SYSTEM>_TPM_LIMIT`, and document how the setup snippet consumes them.
- Do not add `OPENAI_*`, `LLM_*`, LiteLLM, or runner aliases just for uniformity.
- If the runner needs a different field name to launch OpenClaw, adapt the runner rather than rewriting the system `.env`.
- Do not overwrite endpoint, key, model, cost, token, or concurrency values silently.

Only edit env/config files when the user explicitly asks for a change.

### Step 2.5: Cost/token tracking

Cost/token tracking must be recorded because Planar results include top-level `cost_usd` and `total_tokens` fields.

Record, using the system's actual names and mechanisms:

- Main model field and value source.
- Required model-id/provider transformation between the env/config value and the system's native model constructor, if any.
- Required model-call kwargs or provider flags, especially thinking/reasoning controls. For OpenAI-compatible reasoning endpoints, disabling thinking may require a nested provider payload such as `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`; record and preserve that exact nesting so downstream scripts pass `extra_body=...` into the system's native model constructor/operator call rather than expanding it to `enable_thinking=False`.
- Optional embedding/retrieval model field and value source.
- Input and output cost fields, or the native pricing source if the system uses a library-managed pricing table.
- Context and output token cap fields. Record output-token caps as **hard ceilings only**, not recommended defaults for every model call; downstream pipeline skills must choose smaller per-operation limits such as 32/64/128 for filters/classifiers and 128/256/512 for compact extraction.
- Where task scripts should read accumulated cost after a run.

Do not invent missing values. If a value is absent, record it under caveats so the pipeline skill can either use the system's native cost accounting or report a documented limitation.

### Step 2.6: Concurrency and rate-limit policy

Model-call concurrency is a first-class runtime setting because it affects endpoint stability, elapsed time, and cost attribution.

Record, using the system's actual names and mechanisms:

- Effective max model-call concurrency or batch size.
- RPM/request rate limit, if any.
- TPM/token rate limit, if any.
- The source of the value: env/config field, setup helper argument, library default, or user decision.
- Whether downstream task scripts must pass the value explicitly when constructing the system's model/operator object.

If no explicit field exists, ask the user:

> "No explicit concurrency/rate-limit field was found. Do you want to add stable runtime fields to `<env/config path>` for model-call concurrency or rate limits? If yes, what field names and values should be used?"

In non-interactive runner mode, print this as a warning and record it as a caveat in ENV.md rather than blocking. The user can add the fields to `.env` and rerun Phase 0.

Only edit env/config files after explicit user confirmation. Prefer system-native names such as `max_batch_size`, `rate_limit`, `tpm_limit`, `num_workers`, `concurrency`, or documented equivalents. If the system has no native env names, it is acceptable to add project-local fields to the user-chosen `.env`, usually with an upper-case system prefix such as `<SYSTEM>_MAX_BATCH_SIZE`; this prefix is not fixed to any one system. The ENV.md setup snippet must show exactly how those fields are read and passed into the system.

Do not silently leave the downstream agent guessing. If the user chooses not to add fields, record the effective library default and mark it as an intentional default or caveat.

### Step 3: Optional setup notes

A setup helper is optional. Create or patch one only when the target system has no clean native way for downstream task scripts to configure itself.

If a helper is needed:

- Keep it outside package source unless the system already expects helpers there.
- Load the system's existing env/config files.
- Use the system's actual field names.
- Configure the system through its native API.
- Preserve any required model-id/provider adaptation in code rather than changing the user's env field names. For example, if the system delegates to LiteLLM and requires `provider/model`, record a setup snippet that derives that constructor value from the existing raw model field.
- Preserve any required thinking/reasoning-disable kwargs in code. For example, if the env/config contains `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`, parse the value as JSON when valid or with `ast.literal_eval` when it uses Python literals, then pass it as `extra_body=parsed_extra_body` into the native model constructor or operator call. Do not flatten nested provider payloads into `enable_thinking=False`.
- Do not hardcode endpoint, key, model, or cost values.

If no helper is needed, record the native setup pattern instead.

### Step 4: Verify

Run the smallest useful probe from the intended environment. At minimum:

```bash
python -c "import <package>; print(getattr(<package>, '__version__', 'ok'))"
```

Confirm:

- Import succeeded.
- The command used the intended Python executable/environment.
- Any recorded native setup helper or setup snippet imports successfully.

Capture `sys.executable` from the successful probe. For downstream OpenClaw/agent execution, prefer a direct unbuffered interpreter command:

```bash
<resolved-python-executable> -u <pipeline.py>
```

For conda environments, this direct path is usually more observable and reproducible than `conda run`, because `conda run` may capture stdout/stderr until the child exits. If a direct executable path cannot be resolved or is not portable for the target system, record this fallback instead:

```bash
conda run --no-capture-output -n <env> python -u <pipeline.py>
```

Avoid endpoint calls unless the user explicitly wants a live model test.

### Step 5: Record outputs

Write `<system_root>/.agent/ENV.md` with these sections:

- **Activation:** the activation command. One line.
- **System:** root path and Python package/import root.
- **Env/config files:** paths, fields present, missing or uncertain fields. Redact secrets.
- **Cost/token tracking:** model, token caps, per-token costs or native pricing source, how to read accumulated cost, and an explicit note that max-output-token fields are ceilings while task pipelines should use operation-specific lower caps.
- **Model provider/adaptor notes:** whether the env model value is passed through unchanged or transformed before constructing the system model object.
- **Model-call kwargs / thinking controls:** provider-specific kwargs from env/config that downstream scripts must pass to avoid wrong output channels or slow reasoning output.
- **Concurrency/rate limits:** effective max concurrency or batch size, RPM/TPM limits, source fields/defaults, and whether task scripts must pass these values explicitly.
- **Native setup:** helper path or native configuration notes, if any, plus the non-interactive command downstream agents should use to run task scripts. Prefer the resolved Python executable from the verification probe, e.g. `<resolved-python-executable> -u <pipeline.py>`. For conda environments where only the env name is known, record `conda run --no-capture-output -n <env> python -u <pipeline.py>` as the fallback OpenClaw/agent exec form instead of `conda activate <env> && python ...`.
- **Verification:** import/setup probe command and result.
- **Caveats:** unresolved environment issues.

This file is the handoff to the per-system pipeline-skill.

## What this skill does NOT do

- It does not write per-task pipelines, choose operators, or interpret benchmark queries — that's the per-system pipeline-skill (produced by `new-system-operator-skill-creator`).
- It does not bake benchmark-specific data paths into `.env` — datasets are passed at runtime by the benchmark, not configured globally.
- It does not silently choose model-call concurrency. It records an explicit setting, a user-approved env/config field, or an intentional default.
- It does not force canonical env variable names onto the target system.
- It does not paper over missing user input. If the environment cannot be verified, stop and surface the gap.

## Anti-patterns to avoid

- Creating aliases just because a runner uses different names.
- Rewriting env/config files without explicit user approval.
- Installing or repairing dependencies without explicit user approval.
- Putting query-specific or dataset-specific knobs into env/config files.
- Leaving library default concurrency unstated in ENV.md.

## Quality bar

Before declaring done, all of these must hold:

1. Activation command works in a fresh shell; the engine imports successfully. ENV.md records the resolved Python executable from the probe and a non-interactive unbuffered task launch form such as `<resolved-python-executable> -u <pipeline.py>`. If a direct executable cannot be recorded, ENV.md records a streaming fallback such as `conda run --no-capture-output -n <env> python -u <pipeline.py>`.
2. Env/config file paths and field names are recorded as they actually exist.
3. Any required model-id/provider transformation is recorded in the native setup notes without renaming the user's env fields.
4. Any required model-call kwargs or thinking/reasoning controls are recorded and shown in the native setup notes.
5. Cost/token tracking fields or native cost accounting source are recorded, with caveats for missing values.
6. Model-call concurrency/rate-limit policy is recorded, including user-approved env/config fields or the intentional default.
7. Optional helper/setup notes match the target system's native API.
8. `<system_root>/.agent/ENV.md` exists and is enough for a non-interactive agent to activate or identify the intended environment.
