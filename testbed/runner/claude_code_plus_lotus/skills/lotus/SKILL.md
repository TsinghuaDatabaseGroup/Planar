---
name: lotus
description: Build and run Planar tasks with lotus. Use whenever an agent needs to translate a benchmark query (NL or reference operator DAG) into runnable lotus code, execute it on the dataset directory the benchmark hands over, and emit the required result JSON with exactly `task_id`, `answer`, `elapsed_seconds`, `cost_usd`, and `total_tokens`. Trigger on mentions of lotus's operators (sem_filter, sem_map, sem_extract, sem_agg, sem_topk, sem_join) or any request to write/run a benchmark pipeline on this engine, even when the user does not explicitly say the system name.
---

# Lotus Pipeline Development for Planar

Lotus is a framework for LLM-powered data processing that provides an intuitive Pandas-like API with semantic operators. It extends relational operators to unstructured data processing with AI, allowing you to easily capture data-intensive AI programs from simple RAG to document extraction, classification, and complex synthesis. Lotus is ideal for benchmark tasks requiring semantic filtering, extraction, aggregation, ranking, and joining over text corpora.

## Workflow Overview: Per-Task Loop

For each benchmark task the agent receives `query`, `answer_schema`, and (optionally) `reference_semantic_plan`. The loop is **read → write → run → save → iterate-if-failed**, executed on the full dataset. No per-task small-slice smoke-test by default — the operator-level patterns in section 3 give enough confidence to run full input.

### Phase 1: Read the task
Parse `query`, `answer_schema`, and optionally `reference_semantic_plan`. Decide which operators are needed: if `reference_semantic_plan` is present, translate it directly using the catalog mapping below; if not, plan from `query` alone, using the Planar Operator Catalog in §4 as a thinking aid.

### Phase 2: Write the pipeline
Compose lotus's Pandas accessor operators per the plan from Phase 1. Reference the prompt patterns and per-operator examples from section 3. Think about whether it's possible to push exact deterministic query predicates over existing fields before expensive semantic operators when those predicates do not depend on the semantic judgment. Use `answer_schema` while designing the final operator/output shape, not only after the engine returns.

### Phase 3: Run on the full dataset directory
Execute the pipeline against the complete cache built from whatever directory the benchmark hands you at runtime. Inspect raw files only as a fallback when a cache row reports extraction failure or a task needs a source-format-specific check.

Run the pipeline script synchronously in the foreground and wait for it to exit. Do not use `&`, `nohup`, `disown`, tmux/screen, daemon mode, shell `timeout`, exec-tool `timeout` arguments, or any detached subprocess pattern for the main pipeline. A slow healthy operator call is not a failure by itself; full-dataset runs may be long, so only stop the process when the engine reports an error/timeout or the runner timeout is reached.

### Phase 4: Save
Shape the engine's raw output to match `answer_schema` (see the answer-shape guidance in section 5). Save the JSON file at the exact output path supplied by the runner with exactly `task_id`, `answer`, `elapsed_seconds`, `cost_usd`, and `total_tokens`. `answer` must already follow `answer_schema`.

### Phase 5: Iterate only if the run failed
If the engine errored, reported a timeout, the JSON is missing keys, `answer` does not match the schema, or `cost_usd` is zero despite non-zero token usage, clean up any child processes, adjust the pipeline (see Section 6 troubleshooting), and rerun. Do not kill and retry a still-running healthy pipeline just because progress is slow, add a shorter per-command timeout, or fall back to per-task slice smoke-testing.

---

## Section 1: System Description

Lotus provides semantic operators as Pandas DataFrame accessors, allowing you to chain operations naturally: `df.sem_filter(...).sem_map(...).sem_extract(...)`. The engine handles prompt construction, batching, caching, and cost tracking automatically. Lotus supports multimodal inputs (text + images), few-shot learning, chain-of-thought reasoning, and cascade optimization for large datasets.

**Strengths:**
- Familiar Pandas API with semantic extensions
- Automatic prompt templating and output parsing
- Built-in cost tracking via `lm.stats.physical_usage`
- Native support for filter, map, extract, aggregate, join, topk, dedup, search, cluster operations

**Limitations:**
- Semantic search, clustering, and deduplication require separate retrieval model (RM) and vector store (VS) configuration beyond the base LM
- Aggregation returns a special output object (`._output[0]`) rather than a DataFrame column
- Some operators mutate the DataFrame in place; others return new DataFrames

---

## Section 2: Data Input

The benchmark hands the agent a dataset directory at runtime, and the runner builds a reusable cache/manifest handoff before task execution. The original directory may contain mixed file types: text (`.txt`), HTML (`.htm`/`.html`), PDF (`.pdf`), tabular (`.csv`/`.tsv`/`.parquet`), JSON/JSONL, sometimes images. Inspect the manifest first, then choose the loader that matches `cache_mode` (`file_per_document`, `csv_metadata_text_files`, or `mixed_sources`).

### Loading from the runner text cache

Lotus ingests data as Pandas DataFrames. The runner provides `SQPE_DATASET_CACHE_JSONL` with pre-extracted text and `SQPE_DATASET_CACHE_MANIFEST` with the original source structure. Inspect the manifest before choosing a loader:

```python
import json
import os
import pandas as pd

manifest_path = os.environ.get("SQPE_DATASET_CACHE_MANIFEST")
cache_path = os.environ.get("SQPE_DATASET_CACHE_JSONL")
manifest = {}
if manifest_path and os.path.exists(manifest_path):
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

cache_mode = manifest.get("cache_mode", "file_per_document")
if cache_mode == "file_per_document" and cache_path and os.path.exists(cache_path):
    df = pd.read_json(cache_path, lines=True)  # acceptable for small/moderate corpora
elif cache_mode == "mixed_sources" and cache_path and os.path.exists(cache_path):
    # The cache JSONL mixes several collections; every row carries `collection`
    # and `cache_record_type`. Pick the collection(s) the task needs from
    # manifest["collections"] and filter while streaming — never load the whole mix.
    wanted = {c["collection"] for c in manifest.get("collections", [])
              if c["kind"] == "csv_inline_text_table"}  # example choice; pick per task
    pieces = [chunk[chunk["collection"].isin(wanted)]
              for chunk in pd.read_json(cache_path, lines=True, chunksize=2000)]
    pieces = [p for p in pieces if not p.empty]
    df = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
else:
    # For csv_metadata_text_files, prefer the source CSV metadata tables first.
    df = pd.DataFrame()
```

Every cache row includes `document_id`, `collection`, `cache_record_type`, `filename`, `relative_path`, `extension`, `source_path`, `text`, `text_chars`, `text_words`, and `extraction`. Join rows (`cache_record_type == "csv_metadata_text_file"`) add `metadata_source`, `metadata`, `text_path`, `text_relative_path`, and top-level copies of metadata columns. Inline rows (`csv_inline_text_row`) carry the document text inline from the source CSV column named by the collection's `inline_text_column`. JSON records (`json_record`) keep the original nested object under `record`, with flattened scalar fields top-level and under `metadata`, and `text` as a deterministic `key: value` rendering. Use `text` as the semantic content column only for rows that actually need semantic operators.

### Metadata-table datasets

When the manifest reports `cache_mode == "csv_metadata_text_files"`, the original CSV files are metadata tables that point to text files. For exact predicates over metadata columns, read the source CSV first, filter/sort/group there, and only then load text for the reduced candidates. This is required for large SEC-style datasets; the full text cache can be many GB, and concatenating JSONL chunks that contain `text` can OOM the tmux session.

```python
import json
import os
from pathlib import Path
import pandas as pd

data_dir = Path(os.environ.get("SQPE_DATA_DIR", manifest.get("data_dir", "")))
table = manifest["metadata_tables"][0]
csv_path = data_dir / table["relative_path"]
text_col = table["text_path_column"]

meta = pd.read_csv(csv_path)
# Example exact predicates. Convert string-like numeric fields explicitly.
if "word_count" in meta.columns:
    meta["word_count_int"] = pd.to_numeric(meta["word_count"], errors="coerce")
if "year" in meta.columns:
    meta["year_int"] = pd.to_numeric(meta["year"], errors="coerce")

mask = pd.Series(True, index=meta.index)
if "year_int" in meta.columns:
    mask &= meta["year_int"] == 2024
if "word_count_int" in meta.columns:
    mask &= meta["word_count_int"] > 40000
candidates = meta[mask].copy()

def resolve_text_path(value: str) -> Path:
    raw = Path(str(value))
    if raw.is_absolute():
        return raw
    direct = data_dir / raw
    if direct.exists():
        return direct
    return csv_path.parent / raw

def load_candidate_text(row: pd.Series) -> str:
    path = resolve_text_path(row[text_col])
    return path.read_text(encoding="utf-8", errors="replace")

# Load text only after metadata filters have reduced the set.
needed = candidates.copy()
needed["text"] = needed.apply(load_candidate_text, axis=1)
```

If the task genuinely needs every referenced text file, process metadata rows in bounded batches and run Lotus on each batch before discarding it. Do not build one DataFrame that contains thousands of full SEC 10-K texts.

For JSONL cache scans that are unavoidable, stream and drop text early:

```python
pieces = []
for chunk in pd.read_json(cache_path, lines=True, chunksize=1000):
    chunk["year_int"] = pd.to_numeric(chunk["year"], errors="coerce")
    filtered = chunk[chunk["year_int"] == 2024].copy()
    filtered = filtered.drop(columns=["text"], errors="ignore")  # reload text later if needed
    pieces.append(filtered)
df_meta = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
```

### Mixed-source datasets

When the manifest reports `cache_mode == "mixed_sources"`, the dataset combines several source groups (for example: an inline-text CSV, a JSONL record file, and loose report files). The manifest `collections` array lists each group with its `collection` tag, `kind`, schema, and row count. Never scan the whole mixed cache wholesale — pick the collection(s) the task needs and filter while streaming:

```python
manifest = json.load(open(os.environ["SQPE_DATASET_CACHE_MANIFEST"], encoding="utf-8"))
for c in manifest.get("collections", []):
    print(c["collection"], c["kind"], c.get("row_count"))

wanted = {"jsonl:recalls.jsonl"}  # chosen from the printout above per the task
pieces = []
for chunk in pd.read_json(os.environ["SQPE_DATASET_CACHE_JSONL"], lines=True, chunksize=2000):
    part = chunk[chunk["collection"].isin(wanted)]
    if not part.empty:
        pieces.append(part)
df = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
```

Row types inside a mixed cache:

- `csv_inline_text_row` — one source-CSV row; document text is already inline in `text` (from the collection's `inline_text_column`), other CSV columns are top-level and under `metadata`. No file loading needed.
- `json_record` — one JSONL/JSON record; the original nested object is under `record`, flattened scalar fields are top-level (e.g. `vehicle.make`) and under `metadata`, and `text` is a deterministic `key: value` rendering. No file loading needed.
- `file_document` — one loose source file per row, same shape as `file_per_document` rows.
- `csv_metadata_text_file` — a metadata-CSV row joined with its referenced text file (appears in a mixed cache when the dataset also has a text-path table); same shape as `csv_metadata_text_files` rows.

Exact predicates over a collection's structured fields work directly on the top-level columns — push them before any semantic operator, e.g. `df[df["vehicle.make"] == "KIA"]` for a JSONL collection or `df[df["crash_flag"] == "Y"]` for an inline-text CSV collection.

### Metadata-only relational tables

The manifest's `metadata_only_tables` lists auxiliary relational CSVs that produce no cache rows (for example per-incident attribute tables keyed by a shared id column). They can appear alongside any `cache_mode`. Read them straight from the source CSVs for exact joins/predicates, then join the reduced ids against cache records:

```python
data_dir = Path(os.environ["SQPE_DATA_DIR"])
aux = {t["relative_path"]: pd.read_csv(data_dir / t["relative_path"])
       for t in manifest.get("metadata_only_tables", [])}
# Example: reduce ids via an exact predicate on an auxiliary table, then join to cached docs.
events = aux.get("events.csv")
if events is not None:
    wanted_ids = set(events.loc[events["event_type"] == "anomaly", "incident_id"])
    df = df[df["incident_id"].isin(wanted_ids)]
```

### Fallback: Raw directory loader

Only use this when cache rows are unavailable or extraction failed:

```python
import os
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import pandas as pd

def load_text_corpus(dataset_dir: str) -> pd.DataFrame:
    """Load mixed PDF/HTML/TXT files into a DataFrame."""
    records = []
    for fname in sorted(os.listdir(dataset_dir)):
        path = os.path.join(dataset_dir, fname)
        if not os.path.isfile(path):
            continue
        lower = fname.lower()
        if lower.endswith(".pdf"):
            try:
                doc = fitz.open(path)
                text = "\n".join(p.get_text() for p in doc)
                doc.close()
            except Exception:
                text = f"[PDF: {fname}]"
        elif lower.endswith((".htm", ".html")):
            with open(path, encoding="utf-8", errors="replace") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
            text = soup.get_text(separator="\n", strip=True)
        else:
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
        records.append({"document_id": fname, "text": text})
    return pd.DataFrame(records)
```

### Text/candidate handling

Push exact deterministic query predicates before expensive semantic operators when they are independent of the semantic judgment:

```python
# Good: filter by word count before semantic filter
df_candidates = df[df["text"].str.split().str.len() > 100]
df_result = df_candidates.sem_filter("{text} discusses machine learning")

# Avoid: blind keyword-only or filename-only shortcuts for count queries
# Use full text when evidence location is unknown
```

For mention/exclusion predicates over explicit terms, phrases, entities, or acronyms, run a high-recall local keyword/regex pass before any semantic operator. Use `sem_filter` only on reduced or ambiguous candidates; do not send thousands of full long documents to an LLM when local text matching can safely narrow the input.

### Schema constraints

- Lotus expects column names in curly braces: `{column_name}`
- For joins, use `:left` and `:right` suffixes: `{Course Name:left}`, `{Skill:right}`
- No reserved field names, but `text` is conventional for document content
- Long documents: do not rely on automatic overflow behavior. Build a separate evidence column such as `text_for_semantics` before `sem_filter`, `sem_extract`, or `sem_map`. Keep the original `text` column intact for audit/debugging.


---

## Section 3: Operator Guide

### LLM Operation Prompts — General Patterns

**Field interpolation syntax:**
Lotus uses `{column_name}` curly-brace placeholders that are automatically populated with the value of that DataFrame column for each row. For `sem_join`, use `:left`/`:right` suffixes:

```python
user_instruction = "{text} is about renewable energy"  # simple
user_instruction = "Taking {Course Name:left} helps learn {Skill:right}"  # join
```

**Field placeholder placement — critical rule:**
Write the task/predicate/criteria first, then append the field label and column reference at the end of the instruction string:

```python
# Good placement: task first, field last
"This document covers recent AI advances. Document text: {text}"
"Rate the sentiment as positive/negative/neutral. Review: {review}"

# Avoid: putting the field first
"{text} is about something"  # acceptable but less explicit
```

**Structured-output enforcement:**
- `sem_extract`: automatically uses `response_format={"type": "json_object"}` unless CoT is enabled. Pass `output_cols={"field_name": "description"}`.
- `sem_filter`: expects `True` or `False` in the LLM output. The postprocessor parses "True"/"False" strings case-insensitively. Do NOT expect "yes"/"no" or JSON booleans.
- `sem_map`: returns free text per row; the new column has suffix `_map` by default. For JSON-shaped output, use `sem_extract` instead.
- `sem_agg`: returns a single text answer; shape it in the instruction.

**Output-shape directive language:**
For filter: phrase as a claim that can be True or False:
```python
"{text} mentions a specific date"          # ✓ claim form
"Does {text} mention a date? Answer yes/no"  # ✗ question form
```
For map: phrase as a transformation instruction:
```python
"Summarize the main point in one sentence. Document: {text}"
"Classify the category as: finance / technology / health. Title: {title}"
```
For extract: describe each output column:
```python
output_cols = {
    "author": "name of the primary author",
    "year": "publication year as a 4-digit string",
    "main_topic": "main research topic in 5 words or fewer",
}
```
For agg: phrase as an aggregation task:
```python
"Summarize all {text} into a 3-sentence overview"
"Count how many {title} entries mention quantum computing"
```
For topk: phrase as a ranking criterion (the op selects the best K):
```python
"Which {title} is most relevant to climate change policy?"
```

**Few-shot examples:**
Pass as `examples` (DataFrame with same columns + `"Answer"` column) or via `examples_multimodal_data` / `examples_answers` for the functional API:

```python
import pandas as pd
examples = pd.DataFrame({
    "text": ["The cat sat on the mat.", "Revenue increased by 15%."],
    "Answer": ["False", "True"]
})
df.sem_filter("{text} contains a financial metric", examples=examples)
```

**Long-input handling:**
For long per-row documents, prepare a bounded evidence column before calling row-wise semantic operators. `max_tokens` controls output length only; it does not safely trim input evidence. Do not aim near the advertised context limit for row-wise SEC-style documents; provider tokenization and Lotus prompt overhead can push apparently valid inputs over the limit. For 128k-context models, use a conservative evidence budget of about 60k-80k input tokens:

```python
prompt_margin_tokens = max(32768, int(max_ctx_len * 0.25))
target_input_tokens = min(max_ctx_len - max_tokens - prompt_margin_tokens, 80000)
```

Use this budget for the field inserted into prompts (`{text_for_semantics}`), not for the full cached `text`. Build that field by evidence retrieval/selection first; token trimming is only the final guardrail:

- Push exact deterministic predicates first (year, word count, existing metadata, numeric thresholds).
- Retrieve and preserve query-relevant sections when headings or known SEC-style labels exist (business, employees, risk factors, management discussion, going concern, AI/ML, etc.).
- When the relevant section is unknown, assemble a budgeted evidence field from filing/header context plus windows around query terms, synonyms, metadata-derived names, and important candidate sections; include tail context only when it may contain the needed evidence.
- Use Lotus's native token helpers on the configured `LM` (`lm.count_tokens`, `lm.encode_text`, `lm.decode_tokens`) for token-budget checks. Do not import LiteLLM directly just to count or trim tokens.
- Do not send raw full documents and do not blindly keep the first N tokens. Apply token-level trimming with `lm.encode_text(... )[:target_input_tokens]` and `lm.decode_tokens(...)` only after the retrieved evidence has been ordered by relevance.
- For corpus-level summaries or aggregation, use native chunking instead of row-wise hard truncation.

For very long aggregation inputs, pass `long_context_strategy` to `sem_agg` to enable chunked hierarchical aggregation:

```python
from lotus.types import LongContextStrategy
df.sem_agg("Summarize {text}", long_context_strategy=LongContextStrategy.CHUNK)
```

**Output-token budgeting:**
`LLM_MODEL_MAX_OUTPUT_TOKENS=32768` is the configured hard ceiling only. Use task-appropriate caps:
- **Filter/classify**: `max_tokens=32` to `128` — only "True"/"False" or label strings needed
- **Extraction/JSON**: `max_tokens=128` to `512`
- **Map/transformation**: `max_tokens=64` to `256`
- **Aggregation/report**: `max_tokens=512` to `2048`

Create per-step LM objects when different steps need different caps:

```python
from pipeline_helpers import get_lm
lm_filter = get_lm(max_tokens=64, max_batch_size=max_batch_size)
lm_extract = get_lm(max_tokens=512, max_batch_size=max_batch_size)
lotus.settings.configure(lm=lm_filter)   # used by sem_filter
# switch later:
lotus.settings.configure(lm=lm_extract)  # used by sem_extract
```

**Cross-operator failure modes and remedies:**

| Failure | Cause | Remedy |
|---|---|---|
| `sem_filter` returns all True or all False | Prompt not phrased as a True/False claim | Rewrite as a declarative claim; add `default=False` |
| `sem_extract` returns empty dicts `{}` | JSON parse failed; CoT mode enabled | Remove CoT strategy for extract; simplify output_cols descriptions |
| `sem_map` returns verbatim input text | Instruction ambiguous | Make instruction a clear transformation task |
| `sem_agg` returns empty string | All docs too long or partition_ids all different | Set `long_context_strategy=LongContextStrategy.CHUNK` |
| Context overflow / truncated output | Per-row evidence is too large for `max_ctx_len` | Build a budgeted `text_for_semantics` column using local evidence retrieval, relevant section/window selection, and a conservative 60k-80k token cap for 128k-context models; use chunked agg for aggregation |
| `sem_topk` returns wrong items | Ranking instruction ambiguous | Phrase as an explicit preference criterion |
| Cost `total_cost=0` despite non-zero tokens | Pricing registration missed | Load the runner environment before first LM instantiation and use the configured bare model name `Qwen3.5-397B-A17B` |

**Canonical prompt template (3–5 lines):**
```text
{task/predicate description}. {output format or decision criteria}. {field label}: {column_name}
```
Example:
```text
"This passage discusses recent advancements in renewable energy. Passage: {text}"
"Extract the main conclusion in one sentence. Document: {text}"
"Classify sentiment as positive/negative/neutral. Answer in one word. Review: {review}"
```

---

### Mapping to Planar Operator Catalog

| Planar Catalog Op | Lotus Engine Op | Coverage |
|---|---|---|
| `SCAN_DOCS` | `file_per_document`: load JSONL cache with `pd.read_json(lines=True)`; `csv_metadata_text_files`: read source CSV metadata first, then load referenced text for reduced candidates; `mixed_sources`: stream the JSONL cache and filter rows by `collection` | Direct |
| `SCAN_TABLE` | `pd.read_csv` / `pd.read_parquet` | Direct (raw Pandas) |
| `SCAN_MEDIA` | Not needed for text benchmark | Gap |
| `PROJECT` | `df[["col_a", "col_b"]]` | Direct (Pandas) |
| `FILTER` | `df[df["col"] == val]` | Direct (Pandas) |
| `JOIN` | `df1.merge(df2, ...)` | Direct (Pandas) |
| `GROUP_BY` | `df.groupby(...)` | Direct (Pandas) |
| `ORDER_BY` | `df.sort_values(...)` | Direct (Pandas) |
| `LIMIT` | `df.head(k)` | Direct (Pandas) |
| `UNION` | `pd.concat([df1, df2])` | Direct (Pandas) |
| `DEDUP` | `df.drop_duplicates(...)` | Direct (Pandas) |
| `SEM_EXTRACT` | `df.sem_extract(input_cols, output_cols)` | Direct |
| `SEM_FILTER` | `df.sem_filter(user_instruction)` | Direct |
| `SEM_AGGREGATE` | `df.sem_agg(user_instruction)` | Direct |
| `SEM_CLASSIFY` | `df.sem_map(user_instruction)` with the complete target-label set in the instruction | Compose |
| `SEM_MAP` | `df.sem_map(user_instruction)` | Direct |
| `SEM_JOIN` | `df1.sem_join(df2, join_instruction)` | Direct |
| `SEM_RANK` | `df.sem_topk(user_instruction, K=k)` | Direct |
| `SEM_DEDUP` | `df.sem_dedup(col_name, threshold)` | Direct (requires RM + VS) |

---

### Per-Operator Detail

#### `sem_filter` — Keep rows satisfying a natural-language predicate

**Definition:** For each row, asks the LLM whether the row satisfies the claim. Returns a filtered DataFrame with only rows where the answer is `True`.

**Minimal production signature:**
```python
result_df = df.sem_filter(user_instruction, strategy=None)
```

**Parameters:**
- `user_instruction` (str): claim phrased as True/False statement; `{column_name}` placeholders interpolated per row
- `strategy` (str|None): `None` (default), `"cot"` (chain-of-thought), `"zs_cot"` (zero-shot CoT)
- `examples` (pd.DataFrame|None): DataFrame with same columns plus `"Answer"` column (contains `"True"` or `"False"`)
- `default` (bool): value used when LLM output cannot be parsed; defaults to `True`
- `return_explanations` (bool): add `_filter_explanation` column with LLM reasoning

**Runnable example:**
```python
import pandas as pd, lotus
from pipeline_helpers import setup

lm, _ = setup(max_tokens=64)
df = pd.DataFrame({"title": ["Deep Learning Review", "Recipe for pasta", "Climate Change Report"]})
filtered = df.sem_filter("{title} is about a scientific or technical topic")
print(filtered)
# Returns rows 0 and 2
```

**Prompt notes:**
- LLM must return `True` or `False` — the postprocessor parses case-insensitively
- Use `default=False` if you prefer false-negative errors over false-positive
- For large corpora, push keyword or length filters first to reduce LLM calls
- `max_tokens=32` to `64` is sufficient; "True" or "False" is 1–2 tokens

---

#### `sem_map` — Per-row semantic transformation

**Definition:** Applies a natural-language instruction to each row independently, producing a new text column (default suffix `_map`).

**Minimal production signature:**
```python
result_df = df.sem_map(user_instruction, suffix="_map")
```

**Parameters:**
- `user_instruction` (str): transformation instruction with `{column_name}` placeholders
- `suffix` (str): output column name suffix; defaults to `"_map"`
- `system_prompt` (str|None): optional system prompt
- `strategy` (str|None): `None`, `"cot"`, or `"zs_cot"`
- `examples` (pd.DataFrame|None): DataFrame with same columns plus `"Answer"` column
- `return_explanations` (bool): add explanation column (requires CoT strategy)

**Runnable example:**
```python
df = pd.DataFrame({"review": ["Excellent build quality.", "Battery dies quickly."]})
df = df.sem_map("Classify the sentiment as positive/negative/neutral. Answer in one word. Review: {review}")
print(df[["review", "_map"]])
# _map column: ["positive", "negative"]
```

**Prompt notes:**
- Output is free text; for structured JSON output use `sem_extract` instead
- If you need a fixed label, append the label set to the instruction: `"... Choose exactly one: finance / tech / health"`
- `max_tokens=64` to `256` depending on output verbosity

---

#### `sem_extract` — Structured field extraction per row

**Definition:** Extracts structured fields from each row using the LLM, returning a new DataFrame with one new column per extracted field.

**Minimal production signature:**
```python
result_df = df.sem_extract(input_cols, output_cols)
```

**Parameters:**
- `input_cols` (list[str]): columns to include as context
- `output_cols` (dict[str, str|None]): `{"output_col": "description or None"}` — description guides the LLM on what to extract
- `extract_quotes` (bool): also extract supporting quotes; defaults to `False`
- `strategy` (str|None): `None` (uses JSON mode), `"cot"` (disables JSON mode)

**Runnable example:**
```python
df = pd.DataFrame({"bio": ["Alice Smith, 34, works as a data scientist in NYC."]})
result = df.sem_extract(
    input_cols=["bio"],
    output_cols={
        "name": "full name of the person",
        "age": "age as integer string",
        "city": "city of residence",
    }
)
print(result[["name", "age", "city"]])
# name: "Alice Smith", age: "34", city: "NYC"
```

**Prompt notes:**
- Uses `response_format={"type": "json_object"}` by default — JSON mode is reliable
- When `strategy="cot"`, JSON mode is disabled; use this only if reasoning steps are needed
- All output values are strings regardless of the field type — cast with `int(row["age"])` if needed
- Missing fields return `None` in the column
- `max_tokens=256` to `512` for multi-field extraction

---

#### `sem_agg` — Aggregate all rows into a single answer

**Definition:** Hierarchically aggregates documents in the DataFrame to produce a single synthesized answer.

**Minimal production signature:**
```python
agg_result = df.sem_agg(user_instruction)
answer_text = agg_result._output[0]
```

**Parameters:**
- `user_instruction` (str): aggregation instruction with `{column_name}` placeholders
- `long_context_strategy` (LongContextStrategy|None): `LongContextStrategy.CHUNK` for very large corpora; `LongContextStrategy.TRUNCATE` (default)
- `partition_by` (str|None): column name to group by; each group gets its own aggregation

**Runnable example:**
```python
from lotus.types import LongContextStrategy
df = pd.DataFrame({"text": ["Paris is in France.", "Berlin is in Germany.", "Rome is in Italy."]})
result = df.sem_agg("Summarize all {text} in one sentence")
print(result._output[0])
# "Paris, Berlin, and Rome are capital cities in France, Germany, and Italy, respectively."
```

**Group-by aggregation:**
```python
df = pd.DataFrame({
    "category": ["finance", "tech", "finance", "tech"],
    "headline": ["Rate hike announced", "New chip released", "Market dips", "AI research grows"]
})
result = df.sem_agg("Summarize the {headline} entries", partition_by="category")
# result is a DataFrame with one row per category; access via result._output
```

**Prompt notes:**
- `sem_agg` takes column names in the instruction using `{column_name}` syntax
- Output is stored in `._output` (list of strings) — `._output[0]` for the first partition
- For multi-partition, iterate over `result._output`
- `max_tokens=512` to `2048` — reports need space

---

#### `sem_topk` — Return top-K rows by ranking criterion

**Definition:** Ranks all rows against a natural-language criterion and returns the K best rows, using pairwise comparison.

**Minimal production signature:**
```python
sorted_df, stats = df.sem_topk(user_instruction, K=k, return_stats=True)
```

**Parameters:**
- `user_instruction` (str): ranking criterion with `{column_name}` placeholders
- `K` (int): number of top results to return
- `method` (str): `"quick"` (default), `"heap"`, or `"naive"`
- `return_stats` (bool): return comparison stats; defaults to `False`
- `strategy` (str|None): `None`, `"cot"`, or `"zs_cot"`
- `group_by` (str|None): perform ranking within groups

**Runnable example:**
```python
df = pd.DataFrame({"abstract": [
    "Transformer architectures enable long-range dependencies.",
    "Convolutional nets excel at image classification.",
    "Recurrent nets model sequential dependencies.",
]})
top_df, stats = df.sem_topk("Which {abstract} is most relevant to natural language processing?", K=2, return_stats=True)
print(top_df)
```

**Prompt notes:**
- The pairwise comparison prompt expects `"Document 1"` or `"Document 2"` in the LLM response
- `max_tokens=32` to `64` for pairwise comparisons
- For ZS_COT, the LLM must end with `"Answer: Document 1 or Document 2"`

---

#### `sem_join` — Semantic join across two DataFrames

**Definition:** For each pair of rows across two DataFrames, asks the LLM if the join condition holds. Returns matching pairs as a merged DataFrame.

**Minimal production signature:**
```python
result_df = df1.sem_join(df2, join_instruction)
```

**Parameters:**
- `df2` (pd.DataFrame): right-side DataFrame
- `join_instruction` (str): join condition using `{col:left}` and `{col:right}` notation
- `how` (str): `"inner"` (default), `"left"`, `"right"`, `"cross"`
- `strategy` (str|None): `None`, `"cot"`, or `"zs_cot"`
- `examples` (pd.DataFrame|None): few-shot examples

**Runnable example:**
```python
df1 = pd.DataFrame({"title": ["Operating Systems", "Algorithms", "Cooking Basics"]})
df2 = pd.DataFrame({"skill": ["programming", "culinary arts"]})
result = df1.sem_join(df2, "Taking {title:left} develops skills in {skill:right}")
print(result)
# Rows: (Operating Systems, programming), (Algorithms, programming), (Cooking Basics, culinary arts)
```

**Prompt notes:**
- Internally uses `sem_filter` with a combined doc from both sides
- For large cross-products, use `sem_sim_join` with a retrieval model to pre-filter candidates
- `max_tokens=32` to `64` — returns True/False per pair

---

#### `sem_dedup` — Semantic deduplication (requires RM + VS)

**Definition:** Removes semantically duplicate rows using embedding similarity above a threshold.

**Minimal production signature:**
```python
dedup_df = df.sem_dedup(col_name, threshold)
```

**Requires:** `lotus.settings.configure(lm=..., rm=..., vs=...)` with retrieval model and vector store.

**Parameters:**
- `col_name` (str): column to compare for duplicates
- `threshold` (float): similarity threshold (0.0–1.0); rows above this threshold are considered duplicates

**Runnable example:**
```python
from lotus.models import SentenceTransformersRM
from lotus.vector_store import FaissVS

rm = SentenceTransformersRM(model="Qwen3-Embedding-8B")
vs = FaissVS()
lotus.settings.configure(lm=lm, rm=rm, vs=vs)

df = pd.DataFrame({"text": ["AI and machine learning", "Machine learning and AI", "Cooking recipes"]})
df.sem_index("text", "text_idx")
df.load_sem_index("text", "text_idx")
dedup_df = df.sem_dedup("text", threshold=0.9)
print(dedup_df)  # removes near-duplicate rows
```

---

#### `sem_search` — Semantic search (requires RM + VS)

**Definition:** Returns the K most semantically similar rows to a query string.

**Minimal production signature:**
```python
result_df = df.sem_search(col_name, query, K=k)
```

**Requires:** `rm` and `vs` configured via `lotus.settings.configure`.

**Runnable example:**
```python
result_df = df.sem_search("text", "recent developments in transformer models", K=5)
```

---

#### `sem_topk` with `group_by` — Ranked aggregation within groups

```python
# Top 2 entries per category
top_df, _ = df.sem_topk(
    "Which {title} is most important for the category?",
    K=2,
    group_by="category",
    return_stats=True,
)
```

---

### Common Operator Combinations

**Filter → Map → Save (classification pipeline):**
```python
# Keep relevant docs, then classify each
df_relevant = df.sem_filter("This document discusses AI safety. Document: {text}")
df_labeled = df_relevant.sem_map(
    "Classify the primary concern as: alignment / robustness / privacy / other. "
    "Answer in one word. Document: {text}"
)
```

**Filter → Extract → Aggregate (structured synthesis):**
```python
# Find matching docs, extract fields, synthesize
df_hits = df.sem_filter("{text} contains a product release announcement")
df_fields = df_hits.sem_extract(
    input_cols=["text"],
    output_cols={"product_name": "name of the product", "release_date": "release date string"}
)
summary = df_fields.sem_agg("List all {product_name} and their {release_date} in a table")
result = summary._output[0]
```

**Filter → TopK (ranked retrieval):**
```python
# Find candidates, then rank top 5
df_candidates = df.sem_filter("{text} is relevant to climate policy")
top5_df, _ = df_candidates.sem_topk(
    "Which {text} provides the most actionable policy recommendations?",
    K=5,
    return_stats=True,
)
```

---

## Section 4: Planar Description

### 1. Per-task input the downstream agent receives

For every task the benchmark hands the agent only three things:

- `query` — the natural-language question. The pipeline must answer this.
- `answer_schema` — declares the shape `answer` must take. See §4.
- `reference_semantic_plan` — a reference operator DAG (may be present as a hint; see §3 for the Planar Operator Catalog the DAG is composed from). The agent may follow it, adapt it, or generate a different plan from `query` alone.

A `task_id` is also passed (or derivable from the input filename) so the agent can name its output file.

### 2. Output JSON the downstream agent must save

The downstream agent must save this object at the exact output path supplied by the runner:

```json
{
  "task_id": "<TASK_ID>",
  "answer": "<conforms to answer_schema>",
  "elapsed_seconds": 0.0,
  "cost_usd": 0.0,
  "total_tokens": 0
}
```

- `task_id` — must equal the input `task_id`.
- `answer` — must conform to `answer_schema` (see §4).
- `elapsed_seconds` — wall-clock pipeline time, rounded to 2 decimals.
- `cost_usd` — total LLM cost (USD) for this task. Read it from `lm.stats.physical_usage.total_cost`. A zero `cost_usd` means the engine's per-token cost map isn't applying — fix the wiring; never paper over it with a fake value.
- `total_tokens` — aggregate physical LLM tokens used by the target-system pipeline. Sum Lotus input and output token counters; do not persist a nested usage breakdown.

### 3. Planar Operator Catalog

Every `reference_semantic_plan` is composed of operators drawn from the catalog below. See the mapping table in Section 3 for how each catalog op maps to a lotus operator.

**Scan:**
- `SCAN_DOCS(dataset, doc_ids | selector, columns)` — load text-bearing documents.
- `SCAN_TABLE(dataset, table_name, columns)` — load a structured table.
- `SCAN_MEDIA(dataset, asset_ids | selector, modality)` — load image / audio / etc.

**Relational:**
`PROJECT(cols)`, `FILTER(predicate_expr)`, `JOIN(join_type, on)`, `GROUP_BY(keys, aggs)`, `ORDER_BY(keys)`, `LIMIT(k)`, `UNION(distinct)`, `DEDUP(keys)`.

**Semantic (LLM-evaluated):**
- `SEM_EXTRACT(instruction, out_schema, input_bindings)` — pull structured fields from each record (N→N).
- `SEM_FILTER(instruction, input_bindings)` — keep records satisfying a natural-language predicate.
- `SEM_AGGREGATE(instruction, out_schema, group_by?, input_bindings)` — summarize a group with an LLM.
- `SEM_CLASSIFY(instruction, target_labels, input_bindings)` — assign each record exactly one label from a fixed target-label set.
- `SEM_MAP(instruction, out_schema, input_bindings)` — per-record semantic transform.
- `SEM_JOIN(instruction, input_bindings_left, input_bindings_right)` — semantic join across two relations.
- `SEM_RANK(instruction, k, group_by?, input_bindings)` — return top-k per natural-language ranking criterion.
- `SEM_DEDUP(instruction, input_bindings)` — semantic deduplication.

### 4. `answer_schema` shapes

`answer_schema.type` indicates the shape `answer` must take:

| `answer_schema.type` | shape of `answer` |
|---|---|
| `scalar` | a single number (int or float) |
| `named_entity` | a single string |
| `table` | array of objects (or array of strings, treated as single-column) |
| `ordered_list` | array of objects/strings; order matters |
| `dictionary` | a single JSON object — three sub-formats; see §5 |
| `label` | a single label string drawn from `answer_schema.labels` |

**Coercion rules:**
- Strip code fences (` ```json ... ``` `) from LLM output before JSON-parsing.
- Lowercase / strip whitespace on `label` and `named_entity` answers.
- For `table`: ensure each row has the columns declared in `answer_schema.columns`; missing columns become `null`.
- For `dictionary`: respect the `keys` order in flat-format `answer_schema`; for nested-format, populate every required field in `value_schema`.
- For `scalar` / `named_entity`: never wrap in a list, never embed in a dict.

### 5. `dictionary` answer_schema sub-formats

**Format A — Flat dictionary** (keys + values are simple types):
```json
{"type": "dictionary", "keys": ["field_a", "field_b"], "value_types": ["string", "integer"]}
```
→ `answer = {"field_a": "...", "field_b": 42}`

**Format B — Nested dictionary** (keys are data-driven, each value is a sub-object):
```json
{"type": "dictionary", "key_description": "company name (string)", "value_schema": {"count": "integer", "avg_score": "float"}}
```
→ `answer = {"Acme Corp": {"count": 5, "avg_score": 4.2}, ...}`

**Format C — Single-key map** (one key → list of values):
```json
{"type": "dictionary", "key": "entity_name", "value_type": "list[string]"}
```
→ `answer = {"entity_name": ["val1", "val2"]}`

### 7. Per-task lifecycle

1. **Read** `query`, `answer_schema`, and `reference_semantic_plan` (if present).
2. **Plan** — write engine code. Use `reference_semantic_plan` as a hint or follow it; or generate a plan from `query` alone.
3. **Run** — execute the pipeline against the dataset directory the benchmark passed.
4. **Coerce** — shape the engine's raw output to match `answer_schema`.
5. **Save** — write the supplied output path with exactly `task_id`, `answer`, `elapsed_seconds`, `cost_usd`, and `total_tokens`, including non-zero `cost_usd` when token usage is non-zero.
6. **Inspect** — use console progress when useful, but do not persist operator-level traces, intermediate rows, or diagnostic records beside the final result.


---

## Section 5: Pipeline Workflow

### Environment handoff notes

The ENV.md native setup path was verified by an import/setup probe and one tiny operator-native model call (`sem_filter` on a 1-row DataFrame). The smoke test returned a non-empty filtered DataFrame with non-zero cost tracking (`$4.563e-05` for 93 input + 4 output tokens).

**Critical setup requirements:**

1. **Model name:** Use the bare model name `Qwen3.5-397B-A17B`. Do not add a provider prefix. Endpoint routing and credentials come from the runner environment.

2. **Thinking-disable configuration:** Use `pipeline_helpers.setup`/`get_lm`; the helper passes `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` in the form expected by the model endpoint.

3. **Explicit `max_batch_size`:** Pass the requested concurrency to `pipeline_helpers.setup`. The runner exposes it through `LOTUS_LLM_MAX_CONCURRENCY`, with a default of 10.

4. **Per-step output-token caps:** `LLM_MODEL_MAX_OUTPUT_TOKENS=32768` is the configured hard ceiling only. Use task-appropriate caps:
   - Filter/classify: `max_tokens=32` to `128`
   - Extract/JSON: `max_tokens=128` to `512`
   - Map/transform: `max_tokens=64` to `256`
   - Aggregate/report: `max_tokens=512` to `2048`

5. **Cost tracking:** `pipeline_helpers` registers the model card used by the testbed. After the pipeline runs, read `lm.stats.physical_usage.total_cost` for USD cost. Token counters are `lm.stats.physical_usage.prompt_tokens` (input), `lm.stats.physical_usage.completion_tokens` (output), and `lm.stats.physical_usage.total_tokens`.

6. **Non-interactive launch:** Use the Python executable supplied by the runner environment:
   ```bash
   python -u <pipeline.py>
   ```
   Do not use `conda activate lotus && python ...` in agent exec calls unless the shell has already been initialized for activation.

---

### Worked Example

Below is a complete per-task script that loads the runner text cache, builds a lotus pipeline, and saves the required output JSON. This example solves a hypothetical task: "Count how many documents discuss renewable energy."

```python
#!/usr/bin/env python3
"""
Lotus pipeline for a Planar task.
Usage: python pipeline.py <task_id> <query> <answer_schema_json> [reference_semantic_plan]
"""
import json
import os
import sys
import time
from pathlib import Path

# 1. The runner supplies environment/config values.
import pandas as pd
import lotus
from pipeline_helpers import setup

# 2. Parse task inputs
task_id = sys.argv[1]
query = sys.argv[2]
answer_schema = json.loads(sys.argv[3])
reference_semantic_plan = sys.argv[4] if len(sys.argv) > 4 else None

# 3. Configure Lotus through the testbed helper. It uses the bare
# Qwen3.5-397B-A17B model name and the endpoint supplied by the runner.
max_batch_size = int(os.environ.get("LOTUS_LLM_MAX_CONCURRENCY", 10))
lm, _ = setup(max_tokens=512, max_batch_size=max_batch_size)

# 7. Load dataset from runner cache/manifest handoff.
# file_per_document can be loaded directly. csv_metadata_text_files must use
# source CSV metadata first and load referenced text in bounded batches.
# mixed_sources must filter cache rows by `collection` while streaming.
def load_manifest() -> dict:
    manifest_path = os.environ.get("SQPE_DATASET_CACHE_MANIFEST")
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def resolve_text_path(data_dir: Path, csv_path: Path, value: str) -> Path:
    raw = Path(str(value))
    if raw.is_absolute():
        return raw
    direct = data_dir / raw
    if direct.exists():
        return direct
    return csv_path.parent / raw


def iter_dataset_batches(batch_size: int = 100):
    manifest = load_manifest()
    cache_mode = manifest.get("cache_mode", "file_per_document")
    if cache_mode == "csv_metadata_text_files":
        data_dir = Path(os.environ.get("SQPE_DATA_DIR", manifest.get("data_dir", "")))
        for table in manifest.get("metadata_tables", []):
            csv_path = data_dir / table["relative_path"]
            text_col = table["text_path_column"]
            for meta in pd.read_csv(csv_path, chunksize=batch_size):
                batch = meta.copy()
                ref_col = "_text_path_ref" if text_col == "text" else text_col
                if ref_col != text_col:
                    batch[ref_col] = batch[text_col]

                def read_text(row: pd.Series) -> str:
                    path = resolve_text_path(data_dir, csv_path, row[ref_col])
                    return path.read_text(encoding="utf-8", errors="replace")

                batch["text"] = batch.apply(read_text, axis=1)
                yield batch
    elif cache_mode == "mixed_sources":
        cache_path = os.environ.get("SQPE_DATASET_CACHE_JSONL")
        if not (cache_path and os.path.exists(cache_path)):
            raise FileNotFoundError("SQPE_DATASET_CACHE_JSONL not found")
        # Choose the collection(s) this task needs from manifest["collections"];
        # this example keeps only the loose document files:
        wanted = {
            c["collection"]
            for c in manifest.get("collections", [])
            if c.get("kind") == "document_files"
        }
        for chunk in pd.read_json(cache_path, lines=True, chunksize=max(batch_size, 500)):
            part = chunk[chunk["collection"].isin(wanted)]
            if not part.empty:
                yield part
    else:
        cache_path = os.environ.get("SQPE_DATASET_CACHE_JSONL")
        if not (cache_path and os.path.exists(cache_path)):
            raise FileNotFoundError("SQPE_DATASET_CACHE_JSONL not found")
        yield pd.read_json(cache_path, lines=True)

# 8. Build and run pipeline
start_time = time.time()

# Step 1: Filter for documents discussing renewable energy
lotus.settings.configure(lm=lm)
filtered_count = 0
scanned_count = 0
filtered_report_texts = []
for df_batch in iter_dataset_batches():
    scanned_count += len(df_batch)
    df_filtered = df_batch.sem_filter("This document discusses renewable energy. Document: {text}")
    filtered_count += len(df_filtered)
    if answer_schema["type"] == "report" and len(filtered_report_texts) < 200:
        remaining = 200 - len(filtered_report_texts)
        filtered_report_texts.extend(df_filtered["text"].head(remaining).tolist())
print(f"Scanned {scanned_count} documents; filtered to {filtered_count}")

# Step 2: Count (for scalar answer_schema)
if answer_schema["type"] == "scalar":
    answer = filtered_count
elif answer_schema["type"] == "report":
    # Aggregate filtered docs into a report
    lotus.settings.configure(lm=lm)
    if filtered_report_texts:
        report_df = pd.DataFrame({"text": filtered_report_texts})
        agg_result = report_df.sem_agg("Count how many documents discuss renewable energy and summarize the count")
        answer = agg_result._output[0]
    else:
        answer = "0 documents discuss renewable energy."
else:
    # For other schema types, adapt accordingly
    answer = filtered_count

elapsed_seconds = time.time() - start_time

# 9. Compute aggregate cost and token usage
# Lotus tracks the filter and aggregate calls on the configured LM.
total_cost = lm.stats.physical_usage.total_cost
input_tokens = lm.stats.physical_usage.prompt_tokens
output_tokens = lm.stats.physical_usage.completion_tokens
total_tokens = input_tokens + output_tokens

# 10. Save output
output = {
    "task_id": task_id,
    "answer": answer,
    "elapsed_seconds": round(elapsed_seconds, 2),
    "cost_usd": round(total_cost, 6),
    "total_tokens": total_tokens,
}

output_path = os.environ["PLANAR_OUTPUT_PATH"]
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Saved output to {output_path}")
print(f"Answer: {answer}")
print(f"Cost: ${total_cost:.6f}")
print(f"Tokens: {input_tokens} input, {output_tokens} output, {total_tokens} total")
```

**Notes on the worked example:**
- The runner supplies credentials and endpoint configuration; the script uses the bare `Qwen3.5-397B-A17B` model name and does not embed service addresses.
- It loads `SQPE_DATASET_CACHE_JSONL` directly only for `file_per_document` caches. For `csv_metadata_text_files`, it reads source CSV metadata and loads referenced text in bounded batches to avoid full-text OOM. For `mixed_sources`, it streams the cache and keeps only the `collection`(s) the task needs.
- It configures Lotus through the testbed `pipeline_helpers.setup` entry point and uses a task-appropriate output cap.
- It reads aggregate cost and tokens from the configured LM's physical-usage counters.
- It sums Lotus input and output counters into the persisted `total_tokens` value without saving an operator-level usage breakdown.
- It runs synchronously in the foreground; no `&`, `nohup`, or detached subprocess patterns.

---

### Answer-shape guidance

Lotus operators return DataFrames or special output objects. Shape them to match `answer_schema.type`:

| `answer_schema.type` | Lotus output → `answer` coercion |
|---|---|
| `scalar` | `len(df)` for counts; `df["col"].sum()` for sums; `int(df.iloc[0]["col"])` for single-value extraction |
| `named_entity` | `df.iloc[0]["col"].strip().lower()` — single string from first row |
| `table` | `df[["col1", "col2"]].to_dict(orient="records")` — array of objects |
| `ordered_list` | `df["col"].tolist()` — array of strings/values; preserve order |
| `dictionary` (flat) | `df.iloc[0].to_dict()` for single-row; or build manually: `{k: df.iloc[0][k] for k in answer_schema["keys"]}` |
| `dictionary` (nested) | Build from grouped aggregation: `{row["key"]: {"field": row["field"]} for _, row in df.iterrows()}` |
| `label` | `df.iloc[0]["_map"].strip().lower()` — single label from `sem_map` output |
| `report` | `agg_result._output[0]` — text from `sem_agg` |

**Common coercions:**
- Strip code fences: `answer = answer.replace("```json", "").replace("```", "").strip()`
- Ensure table columns: `for row in answer: row.setdefault("missing_col", None)`
- Lowercase labels: `answer = answer.strip().lower()`

---

### Iteration tips

**If `answer` is empty:**
- Verify the LLM returned a non-empty string at the relevant pipeline stage. Print `df` after each operator to inspect intermediate results.
- For `sem_extract`, check that `output_cols` descriptions are clear. If JSON parse fails, the output dict will be `{}`.
- For `sem_agg`, check that `partition_ids` are correct (all same partition for single aggregation).
- Lower `temperature` (default is 0.0 in lotus), then choose a task-appropriate per-call `max_tokens` cap. Do not jump straight to ENV.md's global max-output ceiling.

**If `cost_usd` is zero despite non-zero tokens:**
- Verify the runner environment was loaded before the first `LM()` instantiation.
- Verify `pipeline_helpers` uses the bare model name `Qwen3.5-397B-A17B` and has registered its pricing card.
- Check that `lotus.pricing.register_custom_model_pricing()` ran (it's automatic on import, but call it explicitly if unsure).
- Inspect `lm.stats.physical_usage.total_cost` directly; if it's zero, the pricing entry is missing from LiteLLM's cost map.

**If full run is much slower than expected:**
- Confirm the script uses the runner cache/manifest handoff rather than re-extracting the dataset.
- If the manifest says `cache_mode == "csv_metadata_text_files"`, confirm the script filters source CSV metadata before reading referenced text files. Do not concatenate full-text JSONL chunks just to filter metadata.
- If the manifest says `cache_mode == "mixed_sources"`, confirm the script filters cache rows by `collection` instead of scanning the whole mixed cache, and reads `metadata_only_tables` from their source CSVs for exact joins.
- Confirm exact deterministic query constraints over existing columns are applied before expensive semantic operators when independent. Example: `df[df["text"].str.len() > 500].sem_filter(...)` instead of `df.sem_filter(...)` on all rows.
- For very large corpora, use `sem_search` with a retrieval model to pre-filter candidates before `sem_filter` or `sem_topk`.

**If long inputs overflow context or run slowly:**
- First push exact query predicates when available (word count, keyword presence, date range).
- For row-wise operators, build a separate `text_for_semantics` column sized by an input-token budget, not by the output `max_tokens`.
- Prefer evidence retrieval and selection over hard prefix slicing: keep relevant sections/headings and windows around query terms, synonyms, and metadata-derived names, plus document/header context needed to identify the entity. Use token trimming only after the retrieved evidence has been ordered by relevance; use a simple first-N slice only when the needed evidence is known to occur near the beginning or no better strategy is available.
- For aggregation, use lotus's native chunking: `sem_agg(..., long_context_strategy=LongContextStrategy.CHUNK)`.
- Do not use filename-only, keyword-only, or tiny snippets as a blind shortcut for count/ranking queries.

**If the pipeline is still running without an engine error:**
- Keep waiting rather than killing it as a failed attempt. Full-dataset runs may be long.
- Do not add shell `timeout` or a shorter exec-tool timeout to the main pipeline command; the runner owns the wall-clock cutoff.
- If it exits with an error or timeout, terminate any remaining child processes before retrying.

**If `sem_filter` returns all True or all False:**
- Rewrite the instruction as a declarative claim that can be True or False, not a question.
- Add `default=False` to prefer false-negative errors over false-positive.
- Check that the field placeholder is at the end: `"Claim about topic. Document: {text}"`.

**If `sem_extract` returns empty dicts `{}`:**
- Remove CoT strategy for extract; JSON mode is more reliable.
- Simplify `output_cols` descriptions; avoid multi-sentence descriptions.
- Check that `input_cols` contains the column with the needed evidence.

**If `sem_map` returns verbatim input text:**
- Make the instruction a clear transformation task, not a question.
- Append the expected output format: `"... Answer in one word."` or `"... Choose exactly one: label1 / label2 / label3"`.

---

## Section 6: Troubleshooting & Quick Reference

### `answer` is empty
- Verify the LLM returned a non-empty string at the relevant pipeline stage. Print intermediate DataFrames.
- Strip code fences before JSON-parsing: `answer.replace("```json", "").replace("```", "").strip()`.
- Lower `temperature` (default 0.0), then choose a task-appropriate per-call `max_tokens` cap. Do not jump straight to ENV.md's global max-output ceiling.

### `cost_usd` is zero despite non-zero tokens
- Verify the runner environment was loaded before the first `LM()` instantiation.
- Verify `pipeline_helpers` uses the bare model name `Qwen3.5-397B-A17B` and has registered its pricing card.
- Call `lotus.pricing.register_custom_model_pricing()` explicitly if unsure.
- Inspect `lm.stats.physical_usage.total_cost` directly.

### Full run is much slower than expected
- Confirm the script uses the runner cache/manifest handoff rather than re-extracting the dataset.
- If the manifest says `cache_mode == "csv_metadata_text_files"`, filter source CSV metadata first and load referenced text only for reduced candidates.
- If the manifest says `cache_mode == "mixed_sources"`, filter cache rows by `collection`; `csv_inline_text_row` and `json_record` rows already carry their text inline.
- Avoid `pd.concat` over unbounded chunks that still contain full document `text`.
- Confirm exact deterministic query constraints over existing columns are applied before expensive semantic operators when independent.
- For very large corpora, use `sem_search` with a retrieval model to pre-filter candidates.
- If per-row texts are very long, construct `text_for_semantics` with a conservative input-token budget and local evidence retrieval/section/window selection before row-wise semantic operators. For 128k-context models, keep row-wise evidence around 60k-80k tokens instead of aiming near the context limit.

### `sem_filter` returns all True or all False
- Rewrite the instruction as a declarative claim, not a question.
- Add `default=False` to prefer false-negative errors over false-positive.
- Check that the field placeholder is at the end: `"Claim. Field: {column}"`.

### `sem_extract` returns empty dicts `{}`
- Remove CoT strategy for extract; JSON mode is more reliable.
- Simplify `output_cols` descriptions; avoid multi-sentence descriptions.
- Check that `input_cols` contains the column with the needed evidence.

### `sem_map` returns verbatim input text
- Make the instruction a clear transformation task, not a question.
- Append the expected output format: `"... Answer in one word."`.

### `sem_agg` returns empty string
- Check that `partition_ids` are correct (all same partition for single aggregation).
- Set `long_context_strategy=LongContextStrategy.CHUNK` for very large corpora.

### Context overflow / truncated output
- Do not overwrite the cached `text` column. Create `text_for_semantics` or another evidence column for semantic operators.
- Size the evidence column by input-token budget.
- Prefer retrieval over hard truncation: exact filters first, then relevant sections/headings and keyword/evidence windows; token trim only the already selected evidence, and use prefix-only slicing only as a last resort.
- Use chunked aggregation: `sem_agg(..., long_context_strategy=LongContextStrategy.CHUNK)`.

### `sem_topk` returns wrong items
- Phrase the ranking instruction as an explicit preference criterion.
- Use `strategy="zs_cot"` to see the LLM's reasoning in `return_explanations=True`.

### `sem_join` returns too many or too few pairs
- Refine the join instruction to be more specific.
- For large cross-products, use `sem_sim_join` with a retrieval model to pre-filter candidates.

### `sem_dedup` / `sem_search` / `sem_cluster_by` fail with "retrieval model must be configured"
- These operators require `rm` (retrieval model) and `vs` (vector store):
  ```python
  from lotus.models import SentenceTransformersRM
  from lotus.vector_store import FaissVS
  rm = SentenceTransformersRM(model="Qwen3-Embedding-8B")
  vs = FaissVS()
  lotus.settings.configure(lm=lm, rm=rm, vs=vs)
  ```
- Before using these operators, call `df.sem_index(col_name, index_name)` once, then `df.load_sem_index(col_name, index_name)` in subsequent runs.

### Pipeline is still running without an engine error
- Keep waiting rather than killing it as a failed attempt. Full-dataset runs may be long.
- Do not add shell `timeout` or a shorter exec-tool timeout to the main pipeline command.
- If it exits with an error or timeout, terminate any remaining child processes before retrying.

---

### Quick Reference

**Load file-per-document runner text cache:**
```python
df = pd.read_json(os.environ["SQPE_DATASET_CACHE_JSONL"], lines=True)
```

**Load metadata-table datasets safely:**
```python
manifest = json.load(open(os.environ["SQPE_DATASET_CACHE_MANIFEST"], encoding="utf-8"))
table = manifest["metadata_tables"][0]
data_dir = Path(os.environ["SQPE_DATA_DIR"])
meta = pd.read_csv(data_dir / table["relative_path"])
# Apply exact metadata filters here, then read only referenced texts from table["text_path_column"].
```

**Load one collection from a mixed-sources cache:**
```python
manifest = json.load(open(os.environ["SQPE_DATASET_CACHE_MANIFEST"], encoding="utf-8"))
wanted = {c["collection"] for c in manifest["collections"] if c["kind"] == "json_records"}
pieces = [chunk[chunk["collection"].isin(wanted)]
          for chunk in pd.read_json(os.environ["SQPE_DATASET_CACHE_JSONL"], lines=True, chunksize=2000)]
pieces = [p for p in pieces if not p.empty]
df = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
```

**Configure lotus with LM:**
```python
from pipeline_helpers import setup
lm, _ = setup(max_tokens=64, max_batch_size=10)
```

**Read cost and tokens after pipeline:**
```python
cost_usd = lm.stats.physical_usage.total_cost
input_tokens = lm.stats.physical_usage.prompt_tokens
output_tokens = lm.stats.physical_usage.completion_tokens
total_tokens = lm.stats.physical_usage.total_tokens
```

**Common operator patterns:**
```python
# Filter
df_filtered = df.sem_filter("{text} discusses topic X")

# Map
df_labeled = df.sem_map("Classify as label1/label2/label3. Answer in one word. Text: {text}")

# Extract
df_fields = df.sem_extract(["text"], {"field1": "description", "field2": None})

# Aggregate
agg_result = df.sem_agg("Summarize all {text}")
answer = agg_result._output[0]

# TopK
top_df, stats = df.sem_topk("Which {text} is most relevant to X?", K=5, return_stats=True)

# Join
result_df = df1.sem_join(df2, "{col:left} relates to {col:right}")
```

**DataFrame to answer coercion:**
```python
# Scalar
answer = len(df)

# Named entity
answer = df.iloc[0]["col"].strip().lower()

# Table
answer = df[["col1", "col2"]].to_dict(orient="records")

# Ordered list
answer = df["col"].tolist()

# Dictionary (flat)
answer = {k: df.iloc[0][k] for k in answer_schema["keys"]}

# Report
answer = agg_result._output[0]
```

**Non-interactive launch:**
```bash
python -u pipeline.py <task_id> <query> <answer_schema_json>
```

---

**End of lotus pipeline skill.**
