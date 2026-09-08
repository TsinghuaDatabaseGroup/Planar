# Planar Input/Output Contract

This file carries Planar's **fixed contract** — the input shape, output shape, the Planar Operator Catalog, and answer-schema rules every produced SQPE skill must surface to its downstream agent. Reuse these blocks in section 4 ("Planar Description") of the produced skill verbatim where possible.

What is **not** in this file: anything that varies by deployment (model endpoint, model identifiers, environment activation, dataset directory paths, API keys, per-token costs, concurrency knobs). Those are deployment plumbing the user configures separately; by the time any per-task script in the produced skill runs, the engine is already configured and reachable.

If you find yourself paraphrasing or trimming these blocks for brevity, stop. The downstream agent's outputs flow into a fixed evaluator; mismatches in schema shape silently break scoring.

---

## 1. Per-task input the downstream agent receives

For every task the benchmark hands the agent only three things:

- `query` — the natural-language question. The pipeline must answer this.
- `answer_schema` — declares the shape `answer` must take. See §4.
- `reference_semantic_plan` — a reference operator DAG (may be present as a hint; see §3 for the Planar Operator Catalog the DAG is composed from). The agent may follow it, adapt it, or generate a different plan from `query` alone.

A `task_id` is also passed (or derivable from the input filename) so the agent can name its output file.

The benchmark does **not** hand the agent the full queries file, the gold answer, the evaluation metric, or per-task metadata beyond the three fields above. Anything else the produced skill doesn't supply, the agent doesn't have.

## 2. Output JSON the downstream agent must save

The downstream agent must save this object to the runner-supplied result path:

```json
{
  "task_id": "<TASK_ID>",
  "answer": <conforms to answer_schema>,
  "elapsed_seconds": <float>,
  "cost_usd": <float>,
  "total_tokens": <integer>
}
```

- `task_id` — must equal the input `task_id`.
- `answer` — must conform to `answer_schema` (see §4).
- `elapsed_seconds` — wall-clock pipeline time, rounded to 2 decimals.
- `cost_usd` — total LLM cost (USD) for this task. Read it from the engine's cost attribute / callback after the pipeline finishes. A zero `cost_usd` means the engine's per-token cost map isn't applying — fix the wiring; never paper over it with a fake value.
- `total_tokens` — total physical text-model tokens reported by the target system, including cached tokens when the native counter includes them.

The produced skill must include a `save_output()` helper (or equivalent) that enforces these required keys. Empty / null `answer` should raise — silent failures are worse than crashes.

## 3. Planar Operator Catalog

Every `reference_semantic_plan` is composed of operators drawn from the catalog below. The produced skill should teach the agent how to satisfy each (directly, by composition, or via a documented gap workaround).

### Scan
- `SCAN_DOCS(dataset, doc_ids | selector, columns)` — load text-bearing documents.
- `SCAN_TABLE(dataset, table_name, columns)` — load a structured table.

### Relational
`PROJECT(cols)`, `FILTER(predicate_expr)`, `JOIN(join_type, on)`, `GROUP_BY(keys, aggs)`, `ORDER_BY(keys)`, `LIMIT(k)`, `DEDUP(keys)`.

### Semantic (LLM-evaluated)

- `SEM_EXTRACT(instruction, out_schema, input_bindings)` — pull structured fields from each record (N→N).
- `SEM_FILTER(instruction, input_bindings)` — keep records satisfying a natural-language predicate.
- `SEM_CLASSIFY(instruction, target_labels, input_bindings)` — assign each record to one label from a fixed set.
- `SEM_AGGREGATE(instruction, out_schema, group_by?, input_bindings)` — summarize a group with an LLM.
- `SEM_JOIN(instruction, input_bindings_left, input_bindings_right)` — semantic join across two relations.

The produced skill should include this list (or a tight paraphrase) as part of section 4.

## 4. `answer_schema` shapes

`answer_schema.type` indicates the shape `answer` must take. The produced skill must teach the agent to coerce the engine's raw output into the matching shape before saving.

| `answer_schema.type` | shape of `answer`                                                      |
|----------------------|------------------------------------------------------------------------|
| `scalar`             | a single number (int or float)                                          |
| `named_entity`       | a single string                                                         |
| `table`              | array of objects (or array of strings, treated as single-column)        |
| `ordered_list`       | array of objects/strings; order matters                                 |
| `dictionary`         | a single JSON object — three sub-formats; see §5                        |
| `label`              | a single label string drawn from `answer_schema.labels`                 |

Coercion rules the produced skill should enforce:

- Strip code fences (```json ... ```) from LLM output before JSON-parsing.
- Lowercase / strip whitespace on `label` and `named_entity` answers.
- For `table`: ensure each row has the columns declared in `answer_schema.columns`; missing columns become `null`.
- For `dictionary`: respect the `keys` order in flat-format `answer_schema`; for nested-format, populate every required field in `value_schema`.
- For `scalar` / `named_entity`: never wrap in a list, never embed in a dict.

## 5. `dictionary` answer_schema sub-formats

Three shapes. The produced skill must teach the agent to detect which one and shape the output accordingly.

**Format A — Flat dictionary** (keys + values are simple types):

```json
{"type": "dictionary",
 "keys": ["field_a", "field_b", "field_c"],
 "value_types": ["string", "integer", "list[string]"]}
```

**Format B — Nested dictionary** (keys are data-driven, each value is a sub-object with the same schema):

```json
{"type": "dictionary",
 "key_description": "<description of the key, e.g., 'company name (string)'>",
 "value_schema": {"count": "integer — ...", "avg_score": "float — ..."}}
```

**Format C — Mixed dictionary** (some top-level values are themselves dicts):

```json
{"type": "dictionary",
 "keys": ["summary_count", "groups"],
 "value_types": ["integer", "dictionary"],
 "items": {"groups": {"key_description": "...", "value_schema": {...}}}}
```

## 6. Data file types the benchmark may supply

The benchmark exposes datasets as directories of files. Common types the produced skill should be ready for:

- Plain text (`.txt`, `.md`).
- HTML (`.htm`, `.html`).
- PDF (`.pdf`).
- Tabular (`.csv`, `.tsv`, `.parquet`).
- JSON / JSONL.

The dataset directory itself is whatever the benchmark hands the agent at runtime — the produced skill should teach the agent to point its loader at that directory rather than baking a path in. A reference loader for mixed text-bearing files (PDF + HTM + TXT → records of `{doc_id, contents}`) is one useful primitive to include in section 2 of the produced skill:

```python
import os, fitz
from bs4 import BeautifulSoup
import pandas as pd


def load_text_corpus(dataset_dir: str) -> pd.DataFrame:
    """Walk a directory of mixed PDF / HTM / TXT files into a DataFrame
    of {doc_id, contents}. Adapt the output container to whatever the
    target engine ingests natively."""
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
        records.append({"doc_id": fname, "contents": text})
    return pd.DataFrame(records)
```

Engines that don't accept DataFrames natively need a conversion step (e.g., write the DataFrame to a JSON file and point a `datasets:` block at it; wrap it in the engine's own `Dataset.from_records` factory; etc.). The produced skill must show that conversion in section 2.

## 7. Per-task lifecycle (what the downstream agent does)

The produced skill teaches the agent to follow this loop **per task**:

1. **Read** `query`, `answer_schema`, and `reference_semantic_plan` (if present).
2. **Plan** — write engine code. Use `reference_semantic_plan` as a hint or follow it; or generate a plan from `query` alone.
3. **Run** — execute the pipeline against the dataset directory the benchmark passed.
4. **Coerce** — shape the engine's raw output to match `answer_schema`.
5. **Save** — write the runner-supplied result file with exactly `task_id`, `answer`, `elapsed_seconds`, `cost_usd`, and `total_tokens`.
6. **Inspect** — print one or two key statistics (number of intermediate rows, sample of `answer`) so a human can spot obvious failures.

If any step fails (engine threw, answer is empty, cost is zero), the produced skill should have a "Troubleshooting" subsection listing the most likely causes for *that engine*.
