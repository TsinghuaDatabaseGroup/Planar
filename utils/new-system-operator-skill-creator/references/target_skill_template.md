# Target Skill Template

This is a **flexible scaffold** for the produced skill, not a rigid mold. Different engines call for different shapes — a YAML-driven engine warrants a YAML-heavy walkthrough; a Pandas-accessor engine warrants Python snippets; a declarative SQL-over-LLM engine warrants schema-first examples. The pieces below are the ones every produced skill must surface; the substructure inside each piece is yours to design based on what the engine actually looks like.

## How to use this template

1. Read every section below first; understand the shape before filling.
2. The placeholders intentionally describe **what each section achieves**, not exactly **how**. Build subsections, tables, and code blocks that match the engine's idioms.
3. A piece marked **required** must appear in some form. A piece marked **optional** appears only when it adds value for this engine.
4. Drop helper code (data loader, output saver, anything 50+ lines) into a sibling file under `<system_root>/...` and reference it from the skill — don't inline long helpers in the skill body.
5. The Pipeline Workflow worked example must execute end-to-end on one real benchmark query. If you can't run it, the skill isn't ready.
6. The produced skill speaks to a downstream agent that already has a working environment record in ENV.md and whose native setup path has been smoke-tested through the system's own model-calling API. Write as if environment setup is already handled — no "prerequisites" section and no install guidance. The worked example follows the ENV.md native setup notes; import a helper only if ENV.md names one. Do not restate endpoint values, API keys, or cost-per-token values.

---

## TEMPLATE STARTS HERE — copy everything below this line

````markdown
---
name: <system-name>
description: [FILL IN per the meta-skill's frontmatter guidance: pushy description that mentions 3-5 real operator names.]
---

# <System-Name> Pipeline Development for Planar

[FILL IN: 2–3 sentence intro lifted from the engine's own framing. What it is, when an agent reaches for it.]

## Workflow Overview: Per-Task Loop

For each Planar task the agent receives `query`, `answer_schema`, and (optionally) `reference_semantic_plan`. The loop is **read → write → run → save → iterate-if-failed**, executed on the full dataset. No per-task small-slice smoke-test by default — the operator-level patterns in section 3 give enough confidence to run full input.

### Phase 1: Read the task
Parse `query`, `answer_schema`, and optionally `reference_semantic_plan`. Decide which operators are needed: if `reference_semantic_plan` is present, translate it directly using the catalog mapping below; if not, plan from `query` alone, using the Planar Operator Catalog in §4 as a thinking aid.

### Phase 2: Write the pipeline
Compose the engine's operators per the plan from Phase 1. Reference and learning the prompt patterns and per-operator examples from section 3. Thinking about whether it's possible to push exact deterministic query predicates over existing fields before expensive semantic operators when those predicates do not depend on the semantic judgment. Use `answer_schema` while designing the final operator/output shape, not only after the engine returns.

### Phase 3: Run on the full dataset directory
Execute the pipeline against the complete cache built from whatever directory the benchmark hands you at runtime. Inspect raw files only as a fallback when a cache row reports extraction failure or a task needs a source-format-specific check.

Run the pipeline script synchronously in the foreground and wait for it to exit. Do not use `&`, `nohup`, `disown`, tmux/screen, daemon mode, shell `timeout`, exec-tool `timeout` arguments, or any detached subprocess pattern for the main pipeline. A slow healthy operator call is not a failure by itself; full-dataset runs may be long, so only stop the process when the engine reports an error/timeout or the runner timeout is reached.

### Phase 4: Save
Shape the engine's raw output to match `answer_schema` (see the answer-shape guidance in section 5). Save the runner-supplied result file with exactly `task_id`, `answer`, `elapsed_seconds`, `cost_usd`, and `total_tokens`. `answer` must already follow `answer_schema`.

### Phase 5: Iterate only if the run failed
If the engine errored, reported a timeout, the JSON does not have exactly the five required keys, `answer` does not match the schema, or `cost_usd` is zero despite non-zero `total_tokens`, clean up any child processes, adjust the pipeline (see Section 6 troubleshooting), and rerun. Do not kill and retry a still-running healthy pipeline just because progress is slow, add a shorter per-command timeout, or fall back to per-task slice smoke-testing.

---

## Section 1: System Description **[required]**

[FILL IN — short, oriented at the downstream agent. Cover (in whatever order fits):

- One-sentence framing of the engine, lifted from the README.
- The engine's API style (Pandas accessor / declarative YAML / DAG builder / SQL-over-LLM / ...).
- Strengths the agent can exploit on benchmark tasks.
- Limitations relevant to the benchmark (e.g., "no native semantic dedup; emulate via similarity join + threshold").

Keep it short. The agent doesn't need a tutorial; it needs orientation.]

## Section 2: Data Input **[required]**

The benchmark hands the agent a dataset directory at runtime, and the runner builds a reusable cache/manifest handoff before task execution. The original directory may contain mixed file types: text (`.txt`), HTML (`.htm`/`.html`), PDF (`.pdf`), tabular (`.csv`/`.tsv`/`.parquet`), JSON/JSONL, sometimes images. The agent should inspect the manifest first, then choose the cache or source-table loader that matches `cache_mode` (`file_per_document`, `csv_metadata_text_files`, or `mixed_sources`).

[FILL IN — cover what the engine actually needs:

- The engine's native container(s) for input data and how records are addressed inside it.
- A runnable manifest-aware loader pattern. For `file_per_document`, read `SQPE_DATASET_CACHE_JSONL` and produce the engine's native container. Every cache row includes `document_id`, `collection`, `cache_record_type`, `filename`, `relative_path`, `extension`, `source_path`, `text`, `text_chars`, `text_words`, and `extraction`; `SQPE_DATASET_CACHE_MANIFEST` records the source fingerprint, `cache_mode`, and a `collections` array describing each source group. The result must be ready for the engine's first operator with no repeated extraction.
- A scalable metadata-table pattern for manifests whose `cache_mode` is `csv_metadata_text_files`: the source CSV files under `SQPE_DATA_DIR` are metadata tables with a column that references text files. For exact predicates over metadata columns, read and filter/sort/group the source CSV table first, then load referenced text only for reduced candidates that need semantic operators. Do not teach agents to materialize the complete full-text JSONL cache into one DataFrame when metadata columns can narrow the task. If JSONL scanning is unavoidable, stream it in chunks, immediately discard non-candidates and unused full-text columns, and process semantic candidates in bounded batches.
- A mixed-source pattern for manifests whose `cache_mode` is `mixed_sources`: the cache JSONL mixes several collections, and the manifest `collections` array lists each source group (CSV text-path tables, inline-text CSV tables, JSONL/JSON record collections, loose document files) with its schema and row count. Teach agents to pick the collection(s) the task needs and filter cache rows by `collection` while streaming — never scan the whole mixed cache wholesale. `csv_inline_text_row` rows carry document text inline from the source CSV column named by the collection's `inline_text_column`; `json_record` rows keep the original nested object under `record`, with flattened scalar fields top-level and under `metadata`. Manifest `metadata_only_tables` are auxiliary relational CSVs that produce no cache rows — teach exact joins/predicates on them straight from the source CSVs under `SQPE_DATA_DIR`.
- A fallback raw-directory loader only for unavailable cache rows, extraction failures, or source-format-specific checks. Do not make raw extraction the normal per-task path.
- Text/candidate handling suitable for this engine. Teach the agent how to push exact deterministic query predicates such as word-count thresholds before expensive semantic operators if they are independent of the semantic judgment. Candidate filters or shortened snippets are strategy choices, not defaults; use them only when the query or plan explains why they preserve the needed evidence.
- Long-input handling for row-wise semantic operators. Make clear that output-token caps do not trim input. When full records exceed or approach the context window, create a separate evidence/input column and size it by a conservative input-token budget with a large provider/prompt margin; for 128k-context models, default to roughly 60k-80k row-wise evidence tokens, e.g. `min(max_ctx_len - max_tokens - max(32768, int(max_ctx_len * 0.25)), 80000)`. If the engine exposes native token counters/encode/decode helpers, use those helpers rather than direct provider-client imports. Prefer evidence retrieval over blind hard truncation: exact filters first, then relevant sections/headings and query/evidence windows; token trim only the already selected evidence, and use prefix-only slicing only as a last resort.
- Schema constraints worth flagging up front: required column names, reserved field names, encoding assumptions, default chunking knobs and how to override them.

Don't pre-bind to a specific dataset path; teach the agent to point its loader at whatever directory arrives at runtime.]

## Section 3: Operator Guide **[required]**

This section lets the agent compose pipelines without consulting the engine's docs.

### LLM Operation Prompts — General Patterns **[required]**

Cross-operator prompt patterns for **this engine**. The downstream agent often hasn't written prompts for it before; don't assume it knows the conventions. Cover at minimum:

[FILL IN, scoped to this engine:

- **Field interpolation syntax** — how the engine substitutes record fields into the prompt (`{column_name}` / `{{ input.field }}` / `${field}` / Jinja / explicit `format()` / something else). One short example.
- **Field placeholder placement** — for semantic operators, teach the agent to write the task, predicate/criteria, and output directive first. For Pandas-accessor / lotus-style engines, construct examples as concise sentence strings with a field label at the end, such as `Document text: {text}` or `Title: {title}`. Use this placement in all prompt examples.
- **Structured-output enforcement** — how JSON / typed output is requested (`response_format`, engine-level `out_schema` arg, Pydantic annotation, JSON-mode flag, prompt-side instruction). Show the actual API.
- **Output-shape directive language** — the phrasing the engine's prompt parser expects when telling the LLM what to return. Adapt to this engine's conventions.
- **Few-shot examples** — structured `examples=` argument vs inlining in the prompt body. If structured, show how to populate it.
- **Long-input handling** — chunking policy and how to point a prompt at a chunked or budgeted evidence column; default chunk size and how to override. For row-wise semantic operators over long documents, state a conservative safe input-token budget and require local evidence retrieval plus section/window selection before final token trimming.
- **Output-token budgeting** — ENV.md's max-output field is only a hard ceiling. Show how to create per-step model/operator instances with small caps for classification/filtering (typically 32/64/128), medium caps for extraction/JSON (typically 128/256/512), and larger caps only for summaries or reports.
- **Production defaults** — examples should use the minimal production call shape. Omit optional diagnostic, tracing, or introspection parameters from normal operator examples.
- **Cross-operator failure modes and remedies** — at least: LLM returns prose instead of JSON, LLM repeats the input, output keys drift from `out_schema`, batch returns empty, long input overflows context. With a fix for each.
- **A canonical prompt template** for this engine (3–5 lines) the agent can copy and modify.]

**The canonical template should keep content/record field placeholders after the task definition**:

```text
{task description}. {decision/output criteria}. {field label}: {column_name}
```

Use this placement in every per-operator prompt example that references content/record fields.

### Mapping to Planar Operator Catalog **[optional]**

[FILL IN — include this only if the engine has explicit operator surfaces that align reasonably with the catalog (the catalog is in §4 of `benchmark_context.md`). If included, cover the catalog ops that matter for this engine: which engine op satisfies each, by `Direct` 1:1 method, by `Compose` (document the chain), or by `Gap` workaround (drop into Python or note the limitation).

If the engine's abstractions are too far from the catalog (one general-purpose `LLMOp`, pure declarative SQL-over-LLM, ...), drop this subsection and explain in section 1 or in per-operator detail below why an explicit mapping wasn't useful.]

### Per-operator detail **[required]**

For every operator the engine ships (relational + semantic), include enough to write a call from scratch. At minimum, per operator:

[FILL IN per operator:

- One-line definition.
- Minimal production SDK signature. Omit optional diagnostic, tracing, or introspection parameters from the main signature unless the engine cannot be used correctly without them.
- A runnable example using the engine's own example domain (whatever appears in its docs / README — *not* tied to any benchmark dataset; the goal is to teach the operator's mechanics).
- For LLM-evaluated operators only: operator-specific prompt notes — placeholders unique to this op, output-shape directives that work specifically for it, quirks. Don't repeat the general patterns from "LLM Operation Prompts" above.
- Quirks worth flagging — does the operator drop columns, mutate in place, require a specific column name, fail on long records without chunking?

Group, table, or list these however reads best for the engine. A YAML engine might inline operator examples in YAML blocks; a Pandas accessor engine might use code-block-per-op. Pick what teaches fastest.]

### Common operator combinations **[optional]**

[FILL IN if there are 2–3 patterns the agent will reuse repeatedly (e.g., "filter then aggregate", "extract then sort, take top-k", "cross-record join via similarity, then thresholded dedupe"). Show actual API calls.]

## Section 4: Planar Description **[required]**

[FILL IN by pasting from `references/benchmark_context.md`:

- §1 Per-task input the downstream agent receives
- §2 Output JSON the downstream agent must save
- §3 Planar Operator Catalog
- §4 `answer_schema` shapes + output-shape rules
- §5 `dictionary` answer_schema sub-formats
- §7 Per-task lifecycle

Adapt only the system-specific glue (e.g., "this engine wants records with a `text` column; rename the loader's `contents` accordingly"). Do not edit the catalog, the exact five output keys, or the answer-schema rules.]

## Section 5: Pipeline Workflow **[required]**

This section turns the abstract per-task loop into engine-specific code.

### Environment handoff notes **[required]**

Summarize the non-secret result of the meta-skill's ENV smoke test:

- The ENV.md native setup path was verified by an import/setup probe and one tiny operator-native model call.
- Preserve any provider/model adaptation, thinking/reasoning-disable kwargs such as `chat_template_kwargs`, explicit concurrency/batch-size setting, or cost fallback recorded in ENV.md.
- If native cost can be zero while token counters are non-zero and ENV.md records input/output per-token prices, include the deterministic fallback helper in the worked example: first read native accumulated cost and token counters; only when cost is zero, compute `input_tokens * input_cost + output_tokens * output_cost` from ENV.md. If the system uses a native pricing registry or cost map, prefer a system-local registration/mapping repair when feasible so the native cost surface works; still include the fallback helper for robustness. If the system's native attributes are named prompt/completion tokens, normalize them internally for cost calculation. The helper must return `cost_usd` and aggregate `total_tokens`; input/output counters must not be persisted.
- Treat max-output-token env fields as ceilings only; choose the smallest per-step cap that fits the operation.
- If ENV.md records an interactive activation command such as `conda activate <env>`, also state the non-interactive launch command for agent exec calls. Prefer the resolved Python executable recorded by ENV.md, e.g. `<resolved-python-executable> -u <pipeline.py>`. If ENV.md cannot provide a direct executable, use a streaming conda fallback such as `conda run --no-capture-output -n <env> python -u <pipeline.py>`. Do not instruct the downstream agent to fix this by running `conda init`.
- If the first model-calling operator fails during a task with the same setup/provider/cost issue, stop and repair ENV.md rather than working around it with direct model/API calls.

Do not include endpoint values, API keys, or cost-per-token values here.

### Worked example **[required]**

One fully working per-task script that an agent can copy and adapt. After your Phase 5 smoke test in the meta-skill, paste the actual code that ran successfully. The script must:

- Read `query`, `answer_schema`, and (if present) `reference_semantic_plan` from whatever input source the benchmark exposes per task.
- Obtain a configured engine handle by following ENV.md's native setup notes. Import a helper only if ENV.md names one; otherwise use the system's native setup pattern without restating secrets or deployment values. Preserve any ENV.md-recorded model-id/provider transformation exactly.
- Pass ENV.md-recorded model-call kwargs into the system's native model/operator object, including thinking/reasoning controls such as `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` when present.
- Use the system's own operators/API for all model calls. Do not replace a broken ENV handoff by importing OpenAI/Anthropic/LiteLLM directly or calling model HTTP endpoints.
- Use the runner cache/manifest handoff and reuse it across attempts. For `file_per_document`, load `SQPE_DATASET_CACHE_JSONL` into the engine's native container. Do not rebuild full-text extraction inside each task script.
- Inspect `SQPE_DATASET_CACHE_MANIFEST` before loading. If `cache_mode` is `csv_metadata_text_files`, prefer source CSV metadata tables for exact filtering/sorting/grouping and read full text only after candidate reduction; never concatenate unbounded full-text cache chunks merely to filter metadata.
- If `cache_mode` is `mixed_sources`, filter cache rows by `collection` (per the manifest `collections` list) before any semantic operator; `csv_inline_text_row` and `json_record` rows need no extra file loading, and `metadata_only_tables` must be read from their source CSVs for exact joins/predicates.
- Push exact deterministic query predicates before expensive semantic operators when independent. For semantic filtering/classification, use the field/span that contains the needed evidence; use full text only when it comfortably fits with a large safety margin. For very long records, build a budgeted evidence column with relevant sections/headings and query/evidence windows; for 128k-context models, use roughly 60k-80k input tokens by default instead of near-context-limit payloads. Do not use filename-only, keyword-only, tiny snippets, or arbitrary first-N slices as a blind shortcut for count/ranking queries.
- Use operation-specific output-token caps: classification/filtering should usually use 32/64/128 tokens, compact extraction/JSON 128/256/512, and larger caps only for true summaries/reports. Never use ENV.md's max-output ceiling as the default for every call.
- Build and execute the pipeline.
- Produce the final `answer` in the shape required by `answer_schema`.
- Save the runner-supplied result file with exactly `task_id`, `answer`, `elapsed_seconds`, `cost_usd`, and `total_tokens`.
- Compute `cost_usd` and `total_tokens` through the engine's native cost/token surface first. If ENV.md documents a token-price fallback, call that helper before saving rather than writing 0.0 when token usage is non-zero.
- Run synchronously in the foreground. Do not use shell `timeout` or an exec-tool `timeout` argument for the main pipeline command; the runner owns the wall-clock cutoff. If subprocesses are required, keep them managed and clean up the full process group on real error/timeout before returning.
- Use the ENV.md-documented non-interactive launch command. For conda environments, prefer the resolved Python executable form, e.g. `<resolved-python-executable> -u <pipeline.py>`, inside agent exec calls. If only the conda env name is available, use `conda run --no-capture-output -n <env> python -u <pipeline.py>`. Avoid `conda activate <env> && python ...` unless the shell has already been initialized for activation.

[FILL IN: the actual worked-example code. Choose the script structure that fits the engine — a single file, a small CLI, a notebook export, whatever the engine's users would write naturally. Annotate sparingly: comments only where the agent needs intent the code doesn't convey.]

### Answer-shape guidance

[FILL IN — show how to design the final pipeline output for each `answer_schema.type` the engine is likely to produce, plus any lightweight normalization needed before JSON writing. Group by what's actually different on this engine: if the engine returns DataFrames, cover the DataFrame-to-schema output patterns; if it returns dicts, cover those. Cover at minimum the types that appear in the benchmark: `scalar`, `named_entity`, `ordered_list`, `table`, `dictionary` (flat / nested / mixed), `label`. Two to three lines per type.]

### Iteration tips

[FILL IN: 3–5 bullets on what to do when a run fails or produces something off — these power Phase 5 of the per-task loop. Examples: "If the LLM returned prose instead of JSON, switch to the engine's structured-output flag." "If `cost_usd` is zero, the engine's cost-tracking surface isn't wired through — inspect it." "If long inputs overflow context or run slowly, first push exact query predicates when available, then use engine-native chunking or batching; for row-wise calls, build a conservative evidence column using local retrieval, relevant sections/headings, and query/evidence windows before final token trimming."]

Include one process-lifecycle tip: if the pipeline is still running without an engine error, keep waiting rather than killing it as a failed attempt; do not add shell `timeout` or a shorter exec-tool timeout to the main pipeline command; if it exits with an error or timeout, terminate any remaining child processes before retrying.

## Section 6: Troubleshooting & Quick Reference **[optional but recommended]**

[FILL IN — five to ten common failure modes with quick fixes, plus a handful of one-line CLI / API recipes the agent will reuse. Examples to model on:

### `answer` is empty
- Verify the LLM returned a non-empty string at the relevant pipeline stage.
- Strip code fences before JSON-parsing.
- Lower `temperature`, then choose a task-appropriate per-call max output cap. Do not jump straight to ENV.md's global max-output ceiling.

### Full run is much slower than expected
- Confirm the script uses the runner cache/manifest handoff rather than re-extracting the dataset.
- If the manifest says `cache_mode == "csv_metadata_text_files"`, confirm source CSV metadata is filtered before referenced text files are read, and no unbounded full-text JSONL chunks are concatenated just to filter metadata.
- If the manifest says `cache_mode == "mixed_sources"`, confirm cache rows are filtered by `collection` instead of scanned wholesale, and `metadata_only_tables` joins read the source CSVs.
- Confirm exact deterministic query constraints over existing columns are applied before expensive semantic operators when independent. For mention/exclusion predicates over explicit terms, phrases, entities, or acronyms, use high-recall local keyword/regex filtering before semantic operators.
- Confirm semantic filters/classifiers use the text field/span that contains the needed evidence; full text is appropriate when the evidence location is unknown, but narrower spans are fine when they remain faithful to the query.
- For long row-wise inputs, confirm the script builds a budgeted evidence column with a conservative safety margin and uses local evidence retrieval plus strategic section/window selection before final token trimming, rather than arbitrary prefix slices or near-context-limit payloads.
- Confirm classification/filtering model calls use small `max_tokens` values such as 32/64/128.

### `cost_usd` is 0.0
- The engine's cost map isn't wired through — token counts are accruing but USD per token isn't applied. Inspect the engine's cost-tracking surface (LiteLLM `model_cost`, native cost callback, ...) and confirm it has values for the current model.
- If ENV.md records per-token prices and the engine exposes token counters, add/use the deterministic fallback helper: native total cost first, then token counters times ENV.md input/output prices only when native cost is zero.
- Cache hits don't accrue cost — clear the engine's LLM cache to confirm the pipeline actually called the model.

### `reference_semantic_plan` operator has no engine analog
- Consult section 3's catalog mapping table (if present) for the `Compose` or `Gap` workaround.

### Quick reference

```bash
# Run one task
[FILL IN with the non-interactive command, e.g. <resolved-python-executable> -u <pipeline.py> or conda run --no-capture-output -n <env> python -u <pipeline.py>]

# Clear LLM cache
[FILL IN]

# Print engine version
python -c "import <pkg>; print(<pkg>.__version__)"
```
]

## Documentation pointers

- `<system_root>/README.md` — engine overview.
- `<system_root>/docs/` — full operator reference (advanced parameters not covered here).
- The benchmark's `answer_schema` and the Planar Operator Catalog as documented in section 4 of this skill.
````

## TEMPLATE ENDS HERE

---

## Final checks before declaring the produced skill done

- No `[FILL IN]` placeholders remain.
- Section 3 has both: a top-level "LLM Operation Prompts" subsection covering general prompt patterns, including field placeholder placement, AND a per-operator block for every operator the engine ships, with operator-specific prompt notes for every LLM-evaluated one.
- Section 4 is a faithful paste from `benchmark_context.md` — only system-glue is adapted.
- The worked example in Section 5 runs as-is on the benchmark query you smoke-tested with, with no per-task small-slice step.
- Section 5 includes environment handoff notes that mention the successful operator-native ENV/model-call smoke test and any non-secret caveats to preserve, including provider/model adaptation and thinking/reasoning-disable kwargs if present.
- Sections 2 and 5 load or stream `SQPE_DATASET_CACHE_JSONL` by default for file-per-document caches, inspect `SQPE_DATASET_CACHE_MANIFEST`, prefer source CSV metadata tables when `cache_mode` is `csv_metadata_text_files`, filter cache rows by `collection` and read `metadata_only_tables` from source CSVs when `cache_mode` is `mixed_sources`, explain exact predicate pushdown and semantic input selection, and avoid repeated full-dataset extraction or unbounded full-text DataFrames inside task scripts.
- The worked example uses per-step output-token caps and treats ENV.md's max-output-token field only as a hard ceiling. If it handles long row-wise inputs, it builds a separate budgeted evidence column with a conservative safety margin and documents the evidence retrieval plus section/window selection strategy.
- The worked example does not background or detach the main pipeline, and it cannot return while task subprocesses are still running.
- The smoke-test output contains exactly `task_id`, `answer`, `elapsed_seconds`, `cost_usd`, and `total_tokens`; `cost_usd` is non-zero or the produced skill's worked example applies the ENV.md-documented deterministic token-price fallback when native cost is zero and token usage is non-zero.
- The produced skill body contains no env install commands, model endpoint values, API keys, or cost-per-token values. Avoid prerequisite framing. If the worked example needs setup, refer to ENV.md as the external Phase 0 handoff and use its native setup notes without copying secrets or deployment-specific values.
- Frontmatter description mentions real operator names so keyword triggers catch.
