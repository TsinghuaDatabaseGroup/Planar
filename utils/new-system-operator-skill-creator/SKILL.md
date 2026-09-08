---
name: new-system-operator-skill-creator
description: Read one new operator-native UDA system (SQPE) and condense its **usage** into a single self-contained skill that lets a downstream agent run Planar tasks on that system end-to-end — translate `query` (or a reference operator DAG) into engine code, run on the dataset directory, and save a compact result with `task_id`, `answer`, `elapsed_seconds`, `cost_usd`, and `total_tokens`. Use this whenever the user adds a new operator-native engine to the testbed, points at a system source directory and asks how to use it on the benchmark, says "make/extract/build a skill for system X", "onboard <system>", "I added a new engine, can we wire it in", or any time the goal is to compress one new SQPE end-to-end into reusable guidance for the testbed's agent loop. The produced skill covers operators, prompt patterns, pipeline composition, execution, and benchmark binding; it does **not** cover environment installation or model endpoint wiring — those are recorded first by `env-setup-skill` in `<system_root>/.agent/ENV.md`.
---

# New SQPE System → Skill Extractor

Your job is to read one operator-native UDA system (SQPE) and emit a single skill that teaches a downstream agent how to **use** that system on Planar. Usage means: which operators the engine ships, how to compose them into pipelines, how to write prompts the engine actually accepts, and how to take a benchmark task (`query` + `answer_schema` + optional `reference_semantic_plan`) all the way to a compact five-field result.

A "SQPE" here means a system that ships explicit relational and semantic operators (filter / map / extract / aggregate / join / rank / dedup style) and requires a human or agent to compose those operators into a plan. The benchmark wraps each SQPE in an agent loop, and that loop consults the skill you produce. If the skill is wrong, every downstream task pays the price.

## Why this skill exists

Operator-native engines look superficially similar (filter / map / extract / aggregate / join / rank / dedup) but diverge sharply at the layer the downstream agent actually touches:

- **Operator surface** — which ops the engine ships, what they're called, what their signatures look like, which Planar Operator Catalog ops they cover directly versus by composition.
- **Container model** — does the engine pass DataFrames between operators, a custom `Dataset`, a YAML-bound table, a stream of records?
- **Prompt conventions** — how fields get interpolated, how output shape is enforced (JSON-mode flag, `out_schema` arg, structured-output decorator), how few-shot examples are passed, and where content/record field placeholders should appear relative to predicate criteria.
- **Pipeline launch and observability** — `pipeline.run()`, a CLI, a builder pattern? Where do intermediate outputs land? How is total cost surfaced after a run? Cross-check cost/token tracking against `<system_root>/.agent/ENV.md`.
- **Runtime setup fidelity** — follow `<system_root>/.agent/ENV.md` exactly for native setup, including any model-id/provider adaptation recorded there (for example, systems backed by LiteLLM may require `provider/model` even when the env file stores only the raw model name).
- **Non-interactive environment launch** — if ENV.md records an interactive activation command such as `conda activate <env>`, produced skills must also state the command form that works in OpenClaw/agent non-interactive exec calls. Prefer the resolved Python executable recorded by ENV.md, e.g. `<resolved-python-executable> -u <pipeline.py>`. If ENV.md cannot provide a direct executable, use a streaming conda fallback such as `conda run --no-capture-output -n <env> python -u <pipeline.py>`. Do not teach agents to spend attempts on `conda init` or `conda activate` failures.
- **Process lifecycle** — benchmark agents must run pipeline scripts synchronously in the foreground, wait for completion, and clean up the full process group on real errors/timeouts. A long-running healthy operator call is not a failure by itself; produced skills must not teach `&`, `nohup`, detached tmux/screen runs, polling patterns that let pipeline subprocesses outlive the agent, shell `timeout` wrappers, or shorter exec-tool timeout values for the main pipeline. The runner owns the wall-clock cutoff.
- **Dataset cache usage** — the runner provides a cache/manifest handoff through `SQPE_DATASET_CACHE_JSONL`, `SQPE_DATASET_CACHE_MANIFEST`, and `SQPE_DATA_DIR`; produced skills must inspect the manifest, reuse the appropriate loader, and avoid re-extracting every task.
- **Metadata-table datasets** — when the manifest reports `cache_mode=csv_metadata_text_files`, produced skills must teach agents to treat source CSV files as metadata tables and apply exact metadata filters/sorts/groups there first, then load referenced text only for reduced semantic candidates. Do not teach agents to concatenate an unbounded full-text JSONL cache into memory just to filter metadata columns.
- **Mixed-source datasets** — when the manifest reports `cache_mode=mixed_sources`, the cache JSONL mixes several collections and every record carries `collection` and `cache_record_type`. Produced skills must teach agents to read the manifest `collections` list first and filter cache rows by `collection` before any scan; `csv_inline_text_row` and `json_record` rows already carry their text inline (no file loading), and manifest `metadata_only_tables` are auxiliary relational CSVs with no cache rows — exact joins/predicates on them must read the source CSVs directly.
- **Text/candidate planning** — produced skills must teach how to push exact deterministic query predicates before expensive operators when they are independent, and how to avoid unexamined heuristic reductions that may drop true positives.
- **Long-input planning** — produced skills must teach that output-token caps do not trim input. For row-wise semantic operators over very long records, require a separate budgeted evidence column with a conservative safety margin, not a near-context-limit payload. For 128k-context models, teach roughly 60k-80k input tokens as the default row-wise evidence budget, using the engine's native token counters/encode/decode helpers when available. Require local evidence retrieval/section/window selection before any token trimming; blind prefix slicing is only a last-resort fallback.
- **Output-token budgeting** — max-output env fields in ENV.md are hard ceilings only. Produced skills must teach task-specific LM/token caps: small caps for classification/filtering, medium caps for extraction/JSON, and larger caps only for real summaries/reports.

A downstream agent should not re-discover any of this for every task. Compress the system once, well, and every benchmark task collapses to "translate query into a sequence of these operators, run, save schema-conforming output".

## What the produced skill must teach (the benchmark binding)

The benchmark contract is fixed across systems and lives in `references/benchmark_context.md`: the per-task input shape (`query`, `answer_schema`, optional `reference_semantic_plan`), the five-field output JSON, the Planar Operator Catalog (the vocabulary `reference_semantic_plan` is composed from), the `answer_schema` output-shape rules, and the per-task lifecycle. Read that file before drafting; reuse it (don't paraphrase) in the produced skill's benchmark-description section.

What is **not** fixed and must be system-specific: the operator surface, the container model, the prompt conventions, the launch pattern, and how `cost_usd` plus input, output, and total token counters are read out at the end. These are what your produced skill is for.

## Workflow

Think like an analyst, not a transcriber. **Read → ask → draft → smoke-test → polish.** Don't try to write the whole skill in one shot.

### Phase 0: Scope

Confirm the system path, a short hyphenated name (used as the skill folder and frontmatter `name`), and where the produced skill should live (default: `<system_root>/.agent/skills/<system-name>/SKILL.md`).

Then check that env-setup-skill has been run for this system. The signal: `<system_root>/.agent/ENV.md` exists and records the activation command, env/config files, native setup notes, and cost/token tracking source. If it doesn't exist, stop and tell the user to run env-setup-skill first.

### Phase 1: Verify ENV handoff with a model-call smoke test

Before analyzing or drafting the produced skill, prove that `<system_root>/.agent/ENV.md` is actually usable.

1. Read ENV.md and follow its activation/native setup notes exactly. Do not invent env aliases, hardcode endpoint/key/model values, or bypass the target system.
2. Run a minimal import/setup probe in the recorded environment.
3. Run one tiny **operator-native model-call smoke test** through the target system itself. Use an in-memory one-row input and the cheapest semantic operator/API that calls the configured model. Do not solve a benchmark task and do not use direct OpenAI/Anthropic/LiteLLM/HTTP calls.
4. Pass every ENV.md-recorded model-call kwarg into the system's native model/operator object, especially thinking/reasoning controls. Preserve provider-specific nesting exactly: for OpenAI-compatible endpoints, `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` must be parsed and passed as `extra_body=...`, not flattened to `enable_thinking=False`. These flags affect both latency and whether the answer appears in the output channel that the system parser expects.
5. Confirm the smoke test returns a sensible non-empty value and that the system's cost/token accounting path recorded in ENV.md is readable. If native `cost_usd` is zero while input/output/total token counters are non-zero and ENV.md records input/output per-token prices, treat this as cost wiring that needs a deterministic fallback helper: prefer the native total cost, otherwise compute `input_tokens * input_cost + output_tokens * output_cost` from the ENV.md cost fields. If the system uses a native pricing registry or cost map, prefer a system-local registration/mapping repair when feasible so the native cost surface works; still include the fallback helper for robustness. If the system's native attributes are named prompt/completion tokens, normalize them internally for cost calculation, but persist only their sum as the top-level `total_tokens` field. If only `total_tokens` is exposed, document whether the fallback uses the input rate as a conservative lower bound or whether the system cannot compute USD precisely.
6. If setup, provider/model formatting, thinking/reasoning controls, concurrency, or cost tracking is wrong, patch ENV.md first, then rerun the smoke test. Do not proceed to skill generation on a broken ENV handoff.

Keep the smoke-test artifact small and local, for example under the run scratch/state area or a temporary file. The produced pipeline skill must later mention any ENV caveat or provider/model adaptation discovered here, but it must not restate secrets or endpoint values.

### Phase 2: Analyze the system

Touch every layer below in order. Reading source beats trusting READMEs.

**Top-down framing.** 
- `README.md`, `docs/`, `examples/`, `tutorials/`, `notebooks/`. What does the engine claim to be? What's its idiomatic API style (Pandas accessor / declarative YAML / DAG builder / SQL-over-LLM)? How do real users in `examples/` actually call it? Lift the engine's own framing rather than inventing one.

**Operator surface.** 
Find where operators are defined — common locations: `sem_ops/`, `operations/`, `operators/`, `query/operators/`, `core/ops/`. For each operator, capture the entrypoint (function / class / accessor / decorator), required vs optional args with types, the container shape it expects and returns, and — crucially — the prompt template and postprocessor the operator builds internally: what placeholders it exposes, what exact output labels it expects from the LLM, what fallback/default it applies when parsing fails, and whether it strips code fences or parses JSON automatically. For filter/classifier operators, never assume YES/NO is accepted; verify whether the engine expects `True`/`False`, `yes`/`no`, JSON booleans, labels, or another exact format.

**Data ingestion.** 
- How does data enter? Pandas DataFrame? A registered `Dataset` factory? A YAML `datasets:` block? Raw JSON path? 
- Does the engine natively handle the file types the benchmark may supply (text, HTML, PDF, tabular, JSON, image), or do you pre-extract first? 
- What schema constraints does it impose (required column names, reserved field names, encoding, max document size, default chunking)?

**Pipeline composition and execution.** 
- How is a pipeline built and launched (CLI, `Pipeline.run()`, `Dataset(...).run()`, a builder pattern)? 
- Where do intermediate outputs / cache live? How is total cost / token usage exposed (an attribute on the pipeline object, a callback, an external log)?
- If native cost can stay zero despite non-zero token usage, identify the exact input/output/total token attributes, or the system's native equivalent names, plus the ENV.md input/output cost fields needed for a deterministic fallback helper.

If any of the above is unclear, **read source**. Code doesn't lie; READMEs sometimes do.

### Phase 3: Optional — map the Planar Operator Catalog to engine operators

If the engine exposes operators that align reasonably 1:1 with the catalog's relational + semantic ops, build an explicit mapping table — it makes the downstream agent's `reference_semantic_plan` translation cheap. Three outcomes per row: **Direct** (1:1 method exists), **Compose** (chain 2+ ops; document the chain), **Gap** (no clean way; document the workaround in plain Python or note the gap in the produced skill).

Skip this phase if the engine's abstractions are too far from the catalog (e.g., one general-purpose `LLMOp`, or pure declarative SQL-over-LLM with no operator surface). In that case, teach the engine's natural primitives in detail and explain how they satisfy the catalog's intent — that's a fair substitute, and the produced skill should say why an explicit mapping wasn't useful.

### Phase 4: Draft the produced skill

Open `references/target_skill_template.md` and use it as a **flexible scaffold**, not a rigid mold. Different engines call for different shapes: a YAML-driven engine warrants a YAML-heavy walkthrough; a Pandas-accessor engine warrants Python snippets; a declarative SQL-over-LLM engine warrants schema-first examples. The template marks the pieces every produced skill must surface, but leaves the substructure and ordering inside each piece for you to design.

The pieces (described in the template, summarized here so you have the shape in mind):

- **Workflow overview at the top.** A clear per-task loop the downstream agent executes for each benchmark task. The shape is **read → write → run → save → iterate-if-failed**; the script structure that delivers it is yours to design. The write phase must use `answer_schema` to design the final output shape, not defer schema matching to a separate post-hoc phase.
- **System description.** Two or three sentences about what the engine is and when an agent should reach for it, lifted from the engine's own framing.
- **Data input handling.** How the engine ingests the dataset directory the benchmark hands over at runtime. Teach the runner cache/manifest handoff first: inspect `SQPE_DATASET_CACHE_MANIFEST`, load or stream `SQPE_DATASET_CACHE_JSONL` for file-per-document caches, choose a source-table loader when the manifest says `csv_metadata_text_files`, and filter cache rows by `collection` (per the manifest `collections` list) when it says `mixed_sources`. For metadata-table datasets, teach a scalable source-CSV-first pattern for metadata predicates and only then load referenced text for semantic operators. For mixed-source datasets, teach that `csv_inline_text_row` and `json_record` rows already carry their text and that `metadata_only_tables` are queried from their source CSVs. Include a raw-directory loader only as a fallback for unavailable cache rows or unsupported extraction cases, plus any schema quirks worth flagging (chunking defaults, required column names, encoding).
- **Operator guide.** Two parts. First, a top-level **LLM Operation Prompts** subsection capturing prompt patterns that apply across all LLM-evaluated operators on this engine (field interpolation syntax, field placeholder placement, structured-output enforcement, output-shape directive language, few-shot conventions, input-size handling, common cross-operator failure modes with their remedies, one canonical 3–5 line prompt template). Input-size handling must distinguish output `max_tokens` from input context budgeting, and should teach local evidence retrieval/sectioning for long row-wise inputs before final token-budget trimming; arbitrary small hard slices and near-context-limit payloads are both unsafe. Second, a per-operator detail block covering every operator the engine ships — definition, minimal production signature, runnable production example using the engine's own docs domain (not a benchmark-specific one), and operator-specific prompt notes for LLM-evaluated ops (placeholders unique to that op, output-shape directives that work specifically for it, quirks). Normal examples should omit optional diagnostic, tracing, or introspection parameters.
- **Planar description.** Paste the canonical block from `references/benchmark_context.md` and adapt only the system-specific glue (e.g., "this engine wants records with a `text` column; rename the loader's `contents` accordingly"). Do not edit the catalog, the exact five output keys, or the answer-schema rules.
- **Pipeline workflow.** The heart of the skill. A worked end-to-end per-task script — reads `query` + `answer_schema` + optional `reference_semantic_plan`, uses the cache/manifest handoff with the correct loader for `cache_mode`, builds the pipeline so the final `answer` already matches the `answer_schema` shape, runs on the required dataset scope, and saves the runner-supplied result file with exactly `task_id`, `answer`, `elapsed_seconds`, `cost_usd`, and `total_tokens`. Plus a cookbook of answer-shape patterns by `answer_schema.type` (scalar, named_entity, table, dictionary, label) scoped to how this engine surfaces results.
- **Environment handoff notes.** A short note near the workflow/setup example saying ENV.md was smoke-tested through the system's own model-calling path, and listing any non-secret caveats the downstream agent must preserve (provider/model adaptation, thinking/reasoning-disable kwargs such as `extra_body` with nested `chat_template_kwargs`, explicit max batch size, cost fallback, max-output fields as ceilings rather than per-call defaults, and the direct unbuffered non-interactive launch command to use instead of brittle shell activation). If fallback cost is needed, the produced skill's worked example must include a small helper that first reads native accumulated cost and token counters, then uses ENV.md per-token prices with native token counters only when native cost is zero and token usage is non-zero. The helper must return `cost_usd` and the aggregate `total_tokens`; input/output counters may be used internally but must not be persisted.
- **Execution lifecycle.** The produced skill must explicitly tell the downstream agent to run the pipeline script in the foreground and wait for the process to exit. It must not set a per-command exec/tool timeout for the main pipeline, must not wrap the command in shell `timeout`, and must not kill and retry a still-running healthy pipeline just because progress is slow. If a tool schema requires an explicit timeout, the skill should tell the downstream agent to use the runner-provided value and never a shorter guessed value. If the engine reports a timeout/error or the runner timeout is reached, the skill must instruct cleanup of all child processes before retrying.
- **Troubleshooting and quick reference** (recommended). Five to ten common failure modes with quick fixes, plus a handful of one-line CLI / API recipes the agent will reuse.

The downstream agent runs on the full dataset directory by default and iterates only when something fails. Don't prescribe a per-task small-slice smoke-test as part of the produced skill's flow — running 10–30 records before every task is expensive when there are many tasks, and the operator-level patterns in the produced skill already give the agent enough confidence to run full input. Per-task smoke-testing belongs to your Phase 5 below, not to the produced skill's normal flow.

Length target for the produced skill: 350–650 lines. Shorter than the engine's full docs (the agent should leave faster, not slower) but long enough to cover every operator with at least one runnable example.

### Phase 5: Smoke test the produced skill

Before declaring done, **use the produced skill yourself** end-to-end on one query you pick from the Planar inputs. Activate the env per ENV.md, follow the native setup notes recorded there (import a helper only if ENV.md names one), read one task's `query` + `answer_schema` + (if present) `reference_semantic_plan`, write the per-task pipeline using only what's in the produced skill plus ENV.md — no fishing in the source tree. Run it. Open the saved JSON and confirm it contains exactly `task_id`, `answer`, `elapsed_seconds`, `cost_usd`, and `total_tokens`; confirm that `answer` matches the task's `answer_schema`, and that `cost_usd` is non-zero or the ENV.md-documented deterministic token-price fallback is applied correctly.

Whatever trips you up here is a gap in the skill. Patch the skill, not your local notes — future agents won't have your context. Once the smoke test passes, copy the working code into the produced skill's "Worked example" block so future agents have a concrete reference.

## What the produced skill must NOT contain

- **Environment install commands**, `.env` layout, model endpoint values, API keys, or cost-per-token values. Those live in ENV.md. The produced skill may tell the downstream agent to follow ENV.md native setup notes, but it must not restate secrets or deployment values.
- **Testbed-internal details.** Track structure, evaluators, metric definitions, plan-agent designs — none of that belongs here.
- **Agent-system framing.** This skill family is exclusively about operator-native engines that need explicit pipeline composition.
- **Generic LLM advice.** Spend the skill's budget on system-specific quirks. Generic prompt-engineering essays add weight without helping the agent.
- **Baked-in benchmark dataset paths.** The benchmark hands the dataset directory at runtime per task; the produced skill teaches the agent how to point the loader at whatever directory arrives, not at any specific path.

## Frontmatter for the produced skill

```yaml
---
name: <system-name>
description: Build and run Planar tasks with <system-name>. Use whenever an agent needs to translate a benchmark query (NL or reference operator DAG) into runnable <system-name> code, execute it on the supplied dataset directory, and emit the compact result with `task_id`, `answer`, `elapsed_seconds`, `cost_usd`, and `total_tokens`. Trigger on mentions of <system-name>'s operators (<3-5 real operator names>) or any request to write/run a benchmark pipeline on this engine — even when the user doesn't explicitly say the system name.
---
```

Make the description pushy — bias toward triggering, since the testbed loop relies on the skill firing whenever the agent is about to write a pipeline for this engine. Plug in 3–5 real operator names so keyword matches catch. Don't surface the existence of `env-setup-skill` or any prerequisite-checking framing in the description; the produced skill speaks to a downstream agent that just wants to know what the skill does, not how the testbed is wired internally.

## Quality bar (checklist before declaring done)

A produced skill is good if all of these hold:

1. A fresh agent given only the produced skill + the Planar per-task input + ENV.md (env activation, actual env/config files, native setup notes, cost/token tracking source) + runner cache env vars (`SQPE_DATASET_CACHE_JSONL`, `SQPE_DATASET_CACHE_MANIFEST`, `SQPE_DATA_DIR`) can solve at least one query end-to-end and emit a JSON containing exactly `task_id`, `answer`, `elapsed_seconds`, `cost_usd`, and `total_tokens`, conforming to the task's `answer_schema`.
2. Section 3 contains both (a) a top-level "LLM Operation Prompts" subsection capturing general prompt patterns for the engine, including field placeholder placement, and (b) per-operator detail blocks where every LLM-evaluated operator has an SDK example plus operator-specific prompt notes. Every prompt example that uses content/record placeholders such as `{text}` or `{document_text}` writes task, criteria, and output directive first, then places the placeholder at the end after a field label such as `Document text:`. Normal examples omit optional diagnostic, tracing, or introspection parameters.
3. The Planar-description section reuses the canonical block from `benchmark_context.md` (adapted glue is fine, paraphrasing the contract is not).
4. ENV.md passed an operator-native model-call smoke test before drafting; if it initially failed, ENV.md was patched and the successful smoke test is summarized in non-secret terms in the produced skill.
5. The worked-example pipeline runs as-is on the smoke-tested query, reports aggregate text-model usage in the top-level `total_tokens` field, and reports a non-zero `cost_usd` or applies the ENV.md-documented deterministic token-price fallback correctly: native total cost first, then normalized input/output token counters multiplied by ENV.md input/output cost fields only when native cost is zero and token usage is non-zero.
6. Pipeline execution guidance is process-safe: no detached/background long-running scripts, no returning before the pipeline exits, no shell `timeout` wrappers or shorter exec-tool timeout values on the main pipeline command, and explicit cleanup on real errors/timeouts.
7. Text/candidate guidance is explicit: use the runner cache/manifest handoff, load JSONL directly only for file-per-document caches, prefer source CSV metadata tables for exact predicates when `cache_mode=csv_metadata_text_files`, filter cache rows by `collection` and use the inline `text` of `csv_inline_text_row`/`json_record` rows when `cache_mode=mixed_sources`, read `metadata_only_tables` from their source CSVs, and push exact deterministic query predicates before expensive semantic operators when independent. For mention/exclusion predicates over explicit terms, phrases, entities, or acronyms, the produced skill teaches high-recall local keyword/regex filtering before semantic operators. Prompt examples place evidence field placeholders after the predicate/criteria block with a field label such as `Document text:`. Long row-wise inputs are handled through a separate budgeted evidence column with a conservative safety margin, using local evidence retrieval plus section/window selection before final token trimming; for 128k-context models, default to roughly 60k-80k row-wise evidence tokens rather than near-context-limit payloads. `LLM_MODEL_MAX_OUTPUT_TOKENS` or equivalent env fields are described only as hard ceilings.
8. No env install commands, `.env` layout, model endpoint values, cost-per-token values, or "you must run env-setup-skill first" framing appears in the produced skill body. If ENV.md names a setup helper, the worked example may import it as a black box; otherwise it follows ENV.md's native setup notes without restating secrets or deployment values, including recorded model-id/provider transformations.
9. The skill is shorter and more specific than the engine's full docs.

## Reference files

Two files in `references/` carry reusable text. Read both before drafting.

- `references/benchmark_context.md` — Planar's input/output contract and Operator Catalog. Reuse verbatim in the produced skill's benchmark-description section.
- `references/target_skill_template.md` — a flexible scaffold for the produced skill. It marks required pieces but leaves substructure to you.

## Communication tips

Tell the user where you are in the workflow (Phase 1 ENV smoke-testing, Phase 2 reading source, Phase 3 mapping, Phase 5 smoke-testing the produced skill). Ask before installing anything or running multi-minute jobs. Surface uncertainty explicitly — if the engine's cost API is unclear after Phase 1/2, say so in the produced skill rather than papering over it.

When you finish, summarize: where the produced skill lives, whether ENV.md passed the operator-native model-call smoke test, which catalog ops are mapped (if you did Phase 3), which benchmark query you smoke-tested with, and any remaining caveats. The user uses that summary to decide whether to onboard the engine for real.
