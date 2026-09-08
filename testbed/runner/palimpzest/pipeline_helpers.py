"""Shared Palimpzest helpers for the normalized testbed pipelines."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[2]
BASELINE_DIR = REPO_ROOT / "baseline" / "palimpzest"
SRC_DIR = BASELINE_DIR / "src"
DATA_ROOT = Path(
    os.getenv("PZ_DATA_ROOT", str(BASELINE_DIR / "data"))
).resolve()

# Runtime credentials live with the self-contained testbed runners.
load_dotenv(BASE_DIR.parent / ".env")

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ.setdefault("PZ_OFFLINE_MODEL_METADATA", "true")
sys.path.insert(0, str(SRC_DIR))

import litellm  # noqa: E402

LLM_MAX_CONCURRENCY = int(os.getenv("PZ_LLM_MAX_CONCURRENCY", "10"))
if LLM_MAX_CONCURRENCY < 1:
    raise ValueError("PZ_LLM_MAX_CONCURRENCY must be at least 1")

_LLM_REQUEST_SEMAPHORE = threading.BoundedSemaphore(LLM_MAX_CONCURRENCY)
_RAW_LITELLM_COMPLETION = litellm.completion


@wraps(_RAW_LITELLM_COMPLETION)
def _limited_litellm_completion(*args, **kwargs):
    """Limit only text-generation API requests, not Palimpzest workers."""
    with _LLM_REQUEST_SEMAPHORE:
        return _RAW_LITELLM_COMPLETION(*args, **kwargs)


litellm.completion = _limited_litellm_completion

import palimpzest as pz  # noqa: E402
from palimpzest.core.elements.records import DataRecordCollection  # noqa: E402
from palimpzest.core.lib.schemas import create_schema_from_df  # noqa: E402
from palimpzest.core.models import ExecutionStats  # noqa: E402
from palimpzest.query.optimizer.cost_model import (  # noqa: E402
    SampleBasedCostModel,
)
from palimpzest.query.processor.query_processor_factory import (  # noqa: E402
    QueryProcessorFactory,
)

EXTRA_BODY = json.loads(os.getenv("PZ_EXTRA_BODY", "{}") or "{}")
API_BASE = os.getenv("LLM_API_BASE", os.getenv("PZ_API_BASE", "")).strip()

if API_BASE:
    os.environ["OPENAI_API_BASE"] = API_BASE
    os.environ["OPENAI_BASE_URL"] = API_BASE
if os.getenv("VLLM_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["VLLM_API_KEY"]


def _configured_api_base() -> str:
    if not API_BASE:
        raise RuntimeError(
            "LLM_API_BASE is required; create testbed/runner/.env from "
            "testbed/runner/.env.example"
        )
    return API_BASE


TEXT_MODEL_ID = "Qwen3.5-397B-A17B"
TEXT_MODEL_MAX_INPUT_TOKENS = int(
    os.getenv("MODEL_MAX_INPUT_TOKENS", "128000")
)
TEXT_MODEL_MAX_OUTPUT_TOKENS = int(
    os.getenv("MODEL_MAX_OUTPUT_TOKENS", "32768")
)
TEXT_MODEL_INPUT_COST = float(
    os.getenv("MODEL_USD_PER_INPUT_TOKEN", "0.00000039")
)
TEXT_MODEL_OUTPUT_COST = float(
    os.getenv("MODEL_USD_PER_OUTPUT_TOKEN", "0.00000234")
)
TEXT_MODEL_QUALITY = float(os.getenv("MODEL_MMLU_PRO_SCORE", "80"))
TEXT_MODEL_SECONDS_PER_OUTPUT_TOKEN = float(
    os.getenv("MODEL_SECONDS_PER_OUTPUT_TOKEN", "0.01")
)

DYNAMIC_OPTIMIZATION = os.getenv(
    "PZ_DYNAMIC_OPTIMIZATION",
    "0",
).strip().lower() in {"1", "true", "yes", "on"}
SENTINEL_K = int(os.getenv("PZ_SENTINEL_K", "6"))
SENTINEL_J = int(os.getenv("PZ_SENTINEL_J", "4"))
SENTINEL_SAMPLE_BUDGET = int(
    os.getenv("PZ_SENTINEL_SAMPLE_BUDGET", "100")
)
SENTINEL_SEED = int(os.getenv("PZ_SENTINEL_SEED", "42"))

EMBED_MODEL_ID = "Qwen3-Embedding-8B"
EMBED_MODEL_MAX_INPUT_TOKENS = int(
    os.getenv("EMBEDDING_MAX_INPUT_TOKENS", "32768")
)
EMBED_MODEL_INPUT_COST = float(
    os.getenv("EMBEDDING_USD_PER_INPUT_TOKEN", "0.00000001")
)
FORCE_RAG = os.getenv("PZ_FORCE_RAG", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# SemanticAggregate has no native context-reduction implementation. Keep each
# aggregate request below the model context window while reserving space for
# Palimpzest's prompt template, the completion, and tokenizer variance.
SEM_AGG_CONTEXT_TOKENS = int(
    os.getenv("PZ_SEM_AGG_CONTEXT_TOKENS", str(TEXT_MODEL_MAX_INPUT_TOKENS))
)
SEM_AGG_OUTPUT_TOKEN_RESERVE = int(
    os.getenv("PZ_SEM_AGG_OUTPUT_TOKEN_RESERVE", "4096")
)
SEM_AGG_PROMPT_TOKEN_RESERVE = int(
    os.getenv("PZ_SEM_AGG_PROMPT_TOKEN_RESERVE", "8192")
)
SEM_AGG_TOKEN_SAFETY_FACTOR = float(
    os.getenv("PZ_SEM_AGG_TOKEN_SAFETY_FACTOR", "1.25")
)
if SEM_AGG_CONTEXT_TOKENS < 1:
    raise ValueError("PZ_SEM_AGG_CONTEXT_TOKENS must be at least 1")
if SEM_AGG_OUTPUT_TOKEN_RESERVE < 1:
    raise ValueError("PZ_SEM_AGG_OUTPUT_TOKEN_RESERVE must be at least 1")
if SEM_AGG_PROMPT_TOKEN_RESERVE < 0:
    raise ValueError("PZ_SEM_AGG_PROMPT_TOKEN_RESERVE cannot be negative")
if SEM_AGG_TOKEN_SAFETY_FACTOR < 1.0:
    raise ValueError("PZ_SEM_AGG_TOKEN_SAFETY_FACTOR must be at least 1.0")
if (
    SEM_AGG_OUTPUT_TOKEN_RESERVE + SEM_AGG_PROMPT_TOKEN_RESERVE
    >= SEM_AGG_CONTEXT_TOKENS
):
    raise ValueError("semantic aggregate token reserves exceed context window")


def _register_models() -> None:
    entries = {
        TEXT_MODEL_ID: {
            "max_tokens": TEXT_MODEL_MAX_INPUT_TOKENS,
            "max_input_tokens": TEXT_MODEL_MAX_INPUT_TOKENS,
            "max_output_tokens": TEXT_MODEL_MAX_OUTPUT_TOKENS,
            "input_cost_per_token": TEXT_MODEL_INPUT_COST,
            "output_cost_per_token": TEXT_MODEL_OUTPUT_COST,
            "litellm_provider": "openai",
        }
    }
    if EMBED_MODEL_ID:
        entries[EMBED_MODEL_ID] = {
            "max_tokens": EMBED_MODEL_MAX_INPUT_TOKENS,
            "max_input_tokens": EMBED_MODEL_MAX_INPUT_TOKENS,
            "max_output_tokens": 0,
            "input_cost_per_token": EMBED_MODEL_INPUT_COST,
            "output_cost_per_token": 0.0,
            "litellm_provider": "openai",
            "mode": "embedding",
        }
    litellm.register_model(entries)


_register_models()


def _text_model(max_tokens: int) -> pz.Model:
    model = pz.Model(
        TEXT_MODEL_ID,
        api_base=_configured_api_base(),
        max_tokens=min(max_tokens, TEXT_MODEL_MAX_OUTPUT_TOKENS),
        extra_body=EXTRA_BODY,
    )
    model.model_specs.update(
        {
            "usd_per_input_token": TEXT_MODEL_INPUT_COST,
            "usd_per_output_token": TEXT_MODEL_OUTPUT_COST,
            "max_input_tokens": TEXT_MODEL_MAX_INPUT_TOKENS,
            "max_output_tokens": TEXT_MODEL_MAX_OUTPUT_TOKENS,
            "MMLU_Pro_score": TEXT_MODEL_QUALITY,
            "seconds_per_output_token": TEXT_MODEL_SECONDS_PER_OUTPUT_TOKEN,
            "is_text_model": True,
            "is_embedding_model": False,
        }
    )
    return model


def _embedding_model() -> pz.Model | None:
    if not EMBED_MODEL_ID:
        return None
    model = pz.Model(EMBED_MODEL_ID, api_base=_configured_api_base())
    model.model_specs.update(
        {
            "usd_per_input_token": EMBED_MODEL_INPUT_COST,
            "usd_per_output_token": 0.0,
            "max_input_tokens": EMBED_MODEL_MAX_INPUT_TOKENS,
            "max_output_tokens": 0,
            "is_text_model": False,
            "is_embedding_model": True,
            # Palimpzest permits only one vLLM model, while embedding operators
            # call this model through LiteLLM's embedding endpoint directly.
            "is_vllm_model": False,
        }
    )
    return model


def get_config(
    max_tokens: int = 4096,
    *,
    all_optimizations: bool = False,
    include_small_model: bool = False,
) -> pz.QueryProcessorConfig:
    """Build the benchmark query-processor configuration."""
    del include_small_model
    all_optimizations = all_optimizations or os.getenv(
        "PZ_ALL_OPTIMIZATIONS",
        "0",
    ).strip().lower() in {"1", "true", "yes", "on"}
    text_model = _text_model(max_tokens)
    embedding_model = _embedding_model()
    if not all_optimizations and FORCE_RAG and embedding_model is None:
        raise ValueError("PZ_FORCE_RAG requires PZ_EMBED_MODEL_ID")
    available_models = [text_model]
    if embedding_model is not None:
        available_models.append(embedding_model)

    if all_optimizations:
        return pz.QueryProcessorConfig(
            policy=pz.MaxQuality(),
            available_models=available_models,
            execution_strategy="parallel",
            optimizer_strategy="pareto",
            progress=True,
            verbose=False,
            allow_bonded_query=True,
            allow_model_selection=True,
            allow_rag_reduction=embedding_model is not None,
            allow_mixtures=True,
            allow_critic=True,
            allow_split_merge=True,
            force_rag=False,
            sentinel_execution_strategy="mab",
            k=SENTINEL_K,
            j=SENTINEL_J,
            sample_budget=SENTINEL_SAMPLE_BUDGET,
            seed=SENTINEL_SEED,
        )

    return pz.QueryProcessorConfig(
        policy=pz.MaxQuality(),
        available_models=available_models,
        execution_strategy="parallel",
        optimizer_strategy="none",
        progress=True,
        verbose=False,
        allow_bonded_query=not FORCE_RAG,
        allow_model_selection=False,
        # Keep RAG isolated to explicit SEC-style runs. The embedding model
        # remains available to native embedding join implementations.
        allow_rag_reduction=FORCE_RAG,
        allow_mixtures=not FORCE_RAG,
        allow_critic=not FORCE_RAG,
        allow_split_merge=False,
        force_rag=FORCE_RAG,
    )


def run_optimized_plan(
    plan,
    config: pz.QueryProcessorConfig,
    *,
    dynamic_learning_supported: bool = True,
):
    """Run a plan with optional Sentinel/MAB learning on its current inputs."""
    if not DYNAMIC_OPTIMIZATION:
        return plan.run(config)
    if not dynamic_learning_supported:
        print(
            "[DYNAMIC OPTIMIZATION] skipped: Palimpzest has no native "
            "Sentinel validator for semantic aggregate operators",
            flush=True,
        )
        return plan.run(config)

    validator_model = next(
        (
            model
            for model in config.available_models or []
            if isinstance(model, pz.Model) and model.value == TEXT_MODEL_ID
        ),
        None,
    )
    if validator_model is None:
        raise ValueError(
            "dynamic optimization requires the primary text model in "
            "available_models"
        )

    train_dataset = plan._get_root_datasets()
    print(
        "[DYNAMIC OPTIMIZATION] "
        f"strategy=sentinel/mab train_roots={len(train_dataset)} "
        f"validator={validator_model.value} "
        f"sample_budget={config.sample_budget}",
        flush=True,
    )
    return plan.optimize_and_run(
        config,
        train_dataset=train_dataset,
        validator=pz.Validator(model=validator_model),
    )


def _dataset_path(dataset: str, *parts: str) -> Path:
    path = (DATA_ROOT / dataset / Path(*parts)).resolve()
    dataset_root = (DATA_ROOT / dataset).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {dataset_root}")
    if os.path.commonpath([dataset_root, path]) != str(dataset_root):
        raise ValueError(f"path escapes data/{dataset}: {path}")
    return path


def load_table(
    dataset: str,
    filename: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    path = _dataset_path(dataset, filename)
    return pd.read_csv(path, usecols=columns)


def load_jsonl(dataset: str, filename: str) -> pd.DataFrame:
    path = _dataset_path(dataset, filename)
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}: {error.msg}"
                ) from error
    return pd.json_normalize(records, sep="_")


def load_text_documents(
    dataset: str,
    directory: str,
    pattern: str = "*.txt",
    id_column: str = "file_id",
    text_column: str = "body",
) -> pd.DataFrame:
    document_root = _dataset_path(dataset, directory)
    if not document_root.is_dir():
        raise FileNotFoundError(f"document directory not found: {document_root}")

    records = []
    for path in sorted(document_root.glob(pattern)):
        resolved = path.resolve()
        if not resolved.is_file():
            continue
        if os.path.commonpath([document_root, resolved]) != str(document_root):
            raise ValueError(f"document path escapes {document_root}: {resolved}")
        records.append(
            {
                id_column: resolved.name,
                text_column: resolved.read_text(
                    encoding="utf-8",
                    errors="replace",
                ),
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=[id_column, text_column],
    )


def load_html_documents(
    dataset: str,
    directory: str = ".",
    filenames: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Materialize native HTMLFileDataset text without its broken timestamp field."""
    document_root = _dataset_path(dataset, directory)
    if not document_root.is_dir():
        raise FileNotFoundError(f"document directory not found: {document_root}")

    selected = None if filenames is None else {str(name) for name in filenames}
    source = pz.HTMLFileDataset(
        id=f"{dataset}-html-materialization",
        path=str(document_root),
    )
    records = []
    for index in range(len(source)):
        item = source[index]
        if selected is not None and item["filename"] not in selected:
            continue
        records.append(
            {
                "filename": item["filename"],
                "text": item["text"],
            }
        )

    frame = pd.DataFrame.from_records(records, columns=["filename", "text"])
    if selected is not None:
        loaded = set(frame["filename"])
        if loaded != selected:
            missing = sorted(selected - loaded)
            unexpected = sorted(loaded - selected)
            raise ValueError(
                "HTML document selection mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
    return frame


def load_mixed_documents(
    dataset: str,
    directory: str = ".",
) -> pd.DataFrame:
    """Load TXT/HTML/PDF documents from one dataset directory."""
    from bs4 import BeautifulSoup
    from palimpzest.tools.pdfparser import get_text_from_pdf

    document_root = _dataset_path(dataset, directory)
    if not document_root.is_dir():
        raise FileNotFoundError(f"document directory not found: {document_root}")

    cache_root = BASE_DIR / ".document_text_cache" / dataset
    cache_root.mkdir(parents=True, exist_ok=True)
    records = []
    supported_suffixes = {".txt", ".htm", ".html", ".pdf"}
    for path in sorted(document_root.iterdir(), key=lambda item: item.name):
        resolved = path.resolve()
        suffix = resolved.suffix.lower()
        if not resolved.is_file() or suffix not in supported_suffixes:
            continue
        if os.path.commonpath([document_root, resolved]) != str(document_root):
            raise ValueError(f"document path escapes {document_root}: {resolved}")

        if suffix == ".pdf":
            stat = resolved.stat()
            cache_key = hashlib.sha256(
                (
                    f"{resolved.relative_to(document_root)}:"
                    f"{stat.st_size}:{stat.st_mtime_ns}"
                ).encode("utf-8")
            ).hexdigest()
            cache_path = cache_root / f"{cache_key}.txt"
            if cache_path.is_file():
                text = cache_path.read_text(encoding="utf-8", errors="replace")
            else:
                text = get_text_from_pdf(
                    resolved.name,
                    resolved.read_bytes(),
                    pdfprocessor="pypdf",
                )
                temporary = cache_path.with_suffix(".tmp")
                temporary.write_text(text, encoding="utf-8")
                os.replace(temporary, cache_path)
        elif suffix in {".htm", ".html"}:
            markup = resolved.read_text(encoding="utf-8", errors="replace")
            text = BeautifulSoup(markup, "html.parser").get_text(
                separator="\n",
                strip=True,
            )
        else:
            text = resolved.read_text(encoding="utf-8", errors="replace")

        records.append(
            {
                "document_id": resolved.name,
                "text": text,
                "word_count": len(text.split()),
            }
        )

    return pd.DataFrame.from_records(
        records,
        columns=["document_id", "text", "word_count"],
    )


def load_selected_texts(
    dataset: str,
    records: pd.DataFrame,
    path_column: str = "text_file",
    output_column: str = "text",
) -> pd.DataFrame:
    if path_column not in records.columns:
        raise ValueError(f"records must contain {path_column}")

    loaded = []
    for record in records.to_dict(orient="records"):
        relative_path = str(record[path_column])
        if os.path.isabs(relative_path):
            raise ValueError(f"absolute document path is not allowed: {relative_path}")
        text_path = _dataset_path(dataset, relative_path)
        if not text_path.is_file():
            raise FileNotFoundError(f"document not found: {text_path}")
        item = dict(record)
        item[output_column] = text_path.read_text(
            encoding="utf-8", errors="replace"
        )
        loaded.append(item)
    return pd.DataFrame.from_records(
        loaded,
        columns=[*records.columns, output_column],
    )


class LazySelectedTextDataset(pz.IterDataset):
    """Expose selected text files through Palimpzest's indexed lazy scan."""

    def __init__(
        self,
        *,
        id: str,
        dataset: str,
        records: pd.DataFrame,
        path_column: str,
        output_column: str,
        metadata_columns: Iterable[str] | None = None,
    ) -> None:
        if path_column not in records.columns:
            raise ValueError(f"records must contain {path_column}")

        columns = (
            [column for column in records.columns if column != path_column]
            if metadata_columns is None
            else list(metadata_columns)
        )
        if output_column in columns:
            raise ValueError(
                f"output column must not also be metadata: {output_column}"
            )
        missing = [column for column in columns if column not in records.columns]
        if missing:
            raise ValueError(f"metadata columns are missing: {missing}")

        source_columns = list(dict.fromkeys([*columns, path_column]))
        self.dataset_name = dataset
        self.path_column = path_column
        self.output_column = output_column
        self.metadata_columns = tuple(columns)
        self.records = records.loc[:, source_columns].reset_index(drop=True).copy()

        schema_frame = self.records.loc[:, columns].head(0).copy()
        schema_frame[output_column] = pd.Series(dtype="object")
        super().__init__(id=id, schema=create_schema_from_df(schema_frame))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records.iloc[idx]
        relative_path = str(record[self.path_column])
        if os.path.isabs(relative_path):
            raise ValueError(
                f"absolute document path is not allowed: {relative_path}"
            )
        text_path = _dataset_path(self.dataset_name, relative_path)
        if not text_path.is_file():
            raise FileNotFoundError(f"document not found: {text_path}")

        item = {
            column: record[column]
            for column in self.metadata_columns
        }
        item[self.output_column] = text_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
        return item


def lazy_selected_text_dataset(
    task_id: str,
    dataset: str,
    records: pd.DataFrame,
    *,
    path_column: str,
    output_column: str,
    metadata_columns: Iterable[str] | None = None,
) -> LazySelectedTextDataset:
    """Create a root dataset which reads each selected document on demand."""
    return LazySelectedTextDataset(
        id=f"{task_id}-semantic-input",
        dataset=dataset,
        records=records,
        path_column=path_column,
        output_column=output_column,
        metadata_columns=metadata_columns,
    )


def memory_dataset(task_id: str, records: pd.DataFrame) -> pz.MemoryDataset:
    normalized = records.copy(deep=False)
    for column, dtype in records.dtypes.items():
        if isinstance(dtype, pd.StringDtype):
            normalized[column] = records[column].astype(object)
    return pz.MemoryDataset(
        id=f"{task_id}-semantic-input",
        vals=normalized,
    )


def result_frame(
    result,
    source: pd.DataFrame | None = None,
    generated_columns: Iterable[str] = (),
) -> pd.DataFrame:
    frame = result.to_df().reset_index(drop=True)
    if not frame.empty or source is None:
        return frame

    frame = source.head(0).copy()
    for column in generated_columns:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object")
    return frame


def df_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", force_ascii=False))


@dataclass
class SemanticAggregateCall:
    """One native Palimpzest sem_agg invocation within a bounded reduction."""

    stage: str
    input_rows: int
    output: pd.DataFrame
    result: Any
    elapsed_seconds: float
    estimated_prompt_tokens: int


@dataclass
class SemanticAggregateExecution:
    """Final output and all physical sem_agg calls used to produce it."""

    output: pd.DataFrame
    calls: list[SemanticAggregateCall]
    chunked: bool
    resumed_elapsed_seconds: float = 0.0


def _accumulate_semantic_aggregate_usage(
    tracker: StepTracker | None,
    call: SemanticAggregateCall,
) -> None:
    if tracker is None:
        return
    tracker.record_semantic(
        "sem_agg",
        call.input_rows,
        call.output,
        call.result,
        call.elapsed_seconds,
        estimated_prompt_tokens=call.estimated_prompt_tokens,
        semantic_stage=call.stage,
    )


def _count_model_tokens(text: str) -> int:
    """Count with LiteLLM and fall back to a conservative UTF-8 bound."""
    try:
        return int(litellm.token_counter(model=TEXT_MODEL_ID, text=text))
    except Exception:
        return len(text.encode("utf-8"))


def _aggregate_static_tokens(
    col: dict,
    agg: str,
    depends_on: list[str],
) -> int:
    output_desc = str(col.get("desc", col.get("description", "")))
    static_text = "\n".join(
        [
            str(agg),
            str(col.get("name", "")),
            output_desc,
            *depends_on,
        ]
    )
    return _count_model_tokens(static_text)


def _aggregate_record_tokens(record: dict) -> int:
    serialized = json.dumps(
        record,
        ensure_ascii=True,
        indent=2,
        allow_nan=False,
        default=str,
    )
    # Palimpzest indents every object again when it serializes the enclosing
    # list. Account for those separators and indentation boundaries.
    formatting_tokens = max(8, serialized.count("\n") * 2)
    return _count_model_tokens(serialized) + formatting_tokens


def _aggregate_prompt_token_bound(
    records: pd.DataFrame,
    col: dict,
    agg: str,
    depends_on: list[str],
) -> int:
    selected = records.loc[:, depends_on]
    payload_tokens = _aggregate_static_tokens(col, agg, depends_on) + 8
    payload_tokens += sum(
        _aggregate_record_tokens(record) for record in df_records(selected)
    )
    return int(
        math.ceil(payload_tokens * SEM_AGG_TOKEN_SAFETY_FACTOR)
        + SEM_AGG_PROMPT_TOKEN_RESERVE
    )


def _aggregate_fits_context(
    records: pd.DataFrame,
    col: dict,
    agg: str,
    depends_on: list[str],
) -> tuple[bool, int]:
    estimate = _aggregate_prompt_token_bound(records, col, agg, depends_on)
    available = SEM_AGG_CONTEXT_TOKENS - SEM_AGG_OUTPUT_TOKEN_RESERVE
    return estimate <= available, estimate


def _chunk_aggregate_inputs(
    records: pd.DataFrame,
    col: dict,
    agg: str,
    depends_on: list[str],
    *,
    max_rows: int | None = None,
) -> list[pd.DataFrame]:
    """Greedily split rows so every aggregate prompt stays in context."""
    missing = [field for field in depends_on if field not in records.columns]
    if missing:
        raise ValueError(f"semantic aggregate input is missing fields: {missing}")
    if records.empty:
        return []
    if max_rows is not None and max_rows < 1:
        raise ValueError("semantic aggregate max_rows must be positive")

    static_tokens = _aggregate_static_tokens(col, agg, depends_on) + 8
    raw_budget = int(
        (
            SEM_AGG_CONTEXT_TOKENS
            - SEM_AGG_OUTPUT_TOKEN_RESERVE
            - SEM_AGG_PROMPT_TOKEN_RESERVE
        )
        / SEM_AGG_TOKEN_SAFETY_FACTOR
    ) - static_tokens
    if raw_budget < 1:
        raise ValueError("semantic aggregate prompt leaves no room for input")

    selected_records = df_records(records.loc[:, depends_on])
    record_tokens = [
        _aggregate_record_tokens(record) for record in selected_records
    ]
    chunks: list[pd.DataFrame] = []
    start = 0
    current_tokens = 0

    for position, token_count in enumerate(record_tokens):
        if token_count > raw_budget:
            raise ValueError(
                "one semantic aggregate input row exceeds the safe 128K "
                f"budget at row {position}: estimated payload tokens "
                f"{token_count} > {raw_budget}; the row was not truncated"
            )
        row_limit_reached = (
            max_rows is not None and position - start >= max_rows
        )
        if position > start and (
            current_tokens + token_count > raw_budget or row_limit_reached
        ):
            chunks.append(records.iloc[start:position].reset_index(drop=True))
            start = position
            current_tokens = 0
        current_tokens += token_count

    chunks.append(records.iloc[start:].reset_index(drop=True))
    return chunks


def _chunk_grouped_aggregate_inputs(
    records: pd.DataFrame,
    group_by: list[str],
    max_rows: int,
) -> list[pd.DataFrame]:
    """Partition sorted groups without splitting one group across calls."""
    if max_rows < 1:
        raise ValueError("semantic aggregate grouped max_rows must be positive")
    if not group_by:
        raise ValueError("semantic aggregate grouped chunking needs group fields")
    missing = [field for field in group_by if field not in records.columns]
    if missing:
        raise ValueError(
            f"semantic aggregate grouped fields are missing: {missing}"
        )
    if records.empty:
        return []

    ordered = records.sort_values(
        by=group_by,
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    group_key = group_by[0] if len(group_by) == 1 else group_by
    groups = [
        group.reset_index(drop=True)
        for _, group in ordered.groupby(group_key, sort=False, dropna=False)
    ]

    chunks: list[pd.DataFrame] = []
    current_groups: list[pd.DataFrame] = []
    current_rows = 0
    for group in groups:
        if current_groups and current_rows + len(group) > max_rows:
            chunks.append(pd.concat(current_groups, ignore_index=True))
            current_groups = []
            current_rows = 0
        current_groups.append(group)
        current_rows += len(group)
    if current_groups:
        chunks.append(pd.concat(current_groups, ignore_index=True))
    return chunks


def _run_semantic_aggregate_call(
    task_id: str,
    stage: str,
    records: pd.DataFrame,
    config: pz.QueryProcessorConfig,
    *,
    col: dict,
    agg: str,
    depends_on: list[str],
) -> SemanticAggregateCall:
    fits, estimated_tokens = _aggregate_fits_context(
        records,
        col,
        agg,
        depends_on,
    )
    if not fits:
        raise ValueError(
            f"{task_id} {stage}: aggregate prompt estimate "
            f"{estimated_tokens} exceeds the safe input limit "
            f"{SEM_AGG_CONTEXT_TOKENS - SEM_AGG_OUTPUT_TOKEN_RESERVE}"
        )

    print(
        f"[SEM_AGG] {task_id} {stage} | rows={len(records)} | "
        f"prompt_tokens<={estimated_tokens}/"
        f"{SEM_AGG_CONTEXT_TOKENS - SEM_AGG_OUTPUT_TOKEN_RESERVE}",
        flush=True,
    )
    plan = memory_dataset(
        f"{task_id}-{stage}",
        records,
    ).sem_agg(
        col=dict(col),
        agg=agg,
        depends_on=list(depends_on),
    )
    started = time.time()
    # Palimpzest has no Sentinel validator for SemanticAggregate. Running the
    # plan directly still lets its configured optimizer choose the native
    # physical aggregate implementation and model.
    result = plan.run(config)
    elapsed = time.time() - started
    output = result_frame(
        result,
        generated_columns=[str(col["name"])],
    )
    if output.empty or str(col["name"]) not in output.columns:
        raise ValueError(f"{task_id} {stage}: sem_agg returned no output")
    output_field = str(col["name"])
    output.at[0, output_field] = _normalize_typed_dict_keys(
        output.iloc[0][output_field]
    )

    return SemanticAggregateCall(
        stage=stage,
        input_rows=len(records),
        output=output,
        result=result,
        elapsed_seconds=elapsed,
        estimated_prompt_tokens=estimated_tokens,
    )


def _normalize_typed_dict_keys(value):
    """Remove model-added scalar type prefixes from nested dictionary keys."""
    if isinstance(value, list):
        return [_normalize_typed_dict_keys(item) for item in value]
    if not isinstance(value, dict):
        return value

    normalized = {}
    for key, nested_value in value.items():
        normalized_key = key
        if isinstance(key, str):
            for prefix in ("integer_", "float_", "string_", "boolean_", "list_"):
                candidate = key.removeprefix(prefix)
                if candidate != key and candidate not in value:
                    normalized_key = candidate
                    break
        normalized[normalized_key] = _normalize_typed_dict_keys(nested_value)
    return normalized


def _flatten_partial_aggregate(
    call: SemanticAggregateCall,
    field_name: str,
    depends_on: list[str],
) -> pd.DataFrame:
    items = parse_record_list(call.output.iloc[0][field_name])
    call.output.at[0, field_name] = items
    if not items:
        return pd.DataFrame(columns=depends_on)
    frame = pd.DataFrame.from_records(items)
    # Nested dictionaries are represented as list[dict], so Palimpzest cannot
    # enforce their field names through the output type. Some models prefix a
    # requested field with its described scalar type (for example,
    # ``integer_incident_count``). Normalize only that schema-level variation;
    # the values and aggregation result remain untouched.
    aliases = {}
    for field in depends_on:
        if field in frame.columns:
            continue
        candidates = [
            f"{prefix}_{field}"
            for prefix in ("integer", "float", "string", "boolean", "list")
            if f"{prefix}_{field}" in frame.columns
        ]
        if len(candidates) == 1:
            aliases[candidates[0]] = field
    if aliases:
        frame = frame.rename(columns=aliases)
    missing = [field for field in depends_on if field not in frame.columns]
    if missing:
        raise ValueError(
            f"{call.stage}: partial sem_agg output is missing fields: {missing}"
        )
    return frame.loc[:, depends_on].reset_index(drop=True)


def run_bounded_semantic_aggregate(
    task_id: str,
    records: pd.DataFrame,
    config: pz.QueryProcessorConfig,
    *,
    tracker: StepTracker | None = None,
    col: dict,
    agg: str,
    depends_on: list[str],
    partial_col: dict,
    partial_agg: str,
    partial_depends_on: list[str],
    partial_merge_agg: str,
    merge_agg: str,
    partial_max_rows: int | None = None,
    merge_max_rows: int | None = None,
    merge_sort_by: list[str] | None = None,
    final_max_rows: int | None = None,
    final_additive_fields: list[str] | None = None,
    final_numeric_minimums: dict[str, float] | None = None,
    final_candidate_max_rows: int | None = None,
    final_group_by: list[str] | None = None,
    resume_partial_calls: bool = False,
    max_merge_levels: int = 8,
) -> SemanticAggregateExecution:
    """Run sem_agg once when possible, otherwise use bounded LLM reduction."""
    if records.empty:
        raise ValueError(f"{task_id}: cannot sem_agg an empty relation")

    direct_fits, direct_estimate = _aggregate_fits_context(
        records,
        col,
        agg,
        depends_on,
    )
    if direct_fits:
        call = _run_semantic_aggregate_call(
            task_id,
            "direct",
            records,
            config,
            col=col,
            agg=agg,
            depends_on=depends_on,
        )
        _accumulate_semantic_aggregate_usage(tracker, call)
        return SemanticAggregateExecution(
            output=call.output,
            calls=[call],
            chunked=False,
        )

    chunks = _chunk_aggregate_inputs(
        records,
        partial_col,
        partial_agg,
        depends_on,
        max_rows=partial_max_rows,
    )
    bounded_config = config.copy()
    bounded_config.reasoning_effort = "disable"
    del resume_partial_calls
    print(
        f"[SEM_AGG] {task_id} direct prompt estimate={direct_estimate}; "
        f"using {len(chunks)} bounded partial calls with native "
        "AGG_NO_REASONING",
        flush=True,
    )
    calls: list[SemanticAggregateCall] = []
    partial_frames = []
    partial_field = str(partial_col["name"])

    def execute_call(
        stage: str,
        frame: pd.DataFrame,
        *,
        output_col: dict,
        instruction: str,
        input_fields: list[str],
    ) -> SemanticAggregateCall:
        call = _run_semantic_aggregate_call(
            task_id,
            stage,
            frame,
            bounded_config,
            col=output_col,
            agg=instruction,
            depends_on=input_fields,
        )
        _accumulate_semantic_aggregate_usage(tracker, call)
        calls.append(call)
        return call

    for index, chunk in enumerate(chunks, start=1):
        stage = f"partial-l0-c{index:03d}-of-{len(chunks):03d}"
        call = execute_call(
            stage,
            chunk,
            output_col=partial_col,
            instruction=partial_agg,
            input_fields=depends_on,
        )
        partial_frames.append(
            _flatten_partial_aggregate(
                call,
                partial_field,
                partial_depends_on,
            )
        )

    partials = pd.concat(partial_frames, ignore_index=True)
    if partials.empty:
        raise ValueError(f"{task_id}: all partial sem_agg calls returned empty")

    merge_level = 1
    while True:
        final_fits, _ = _aggregate_fits_context(
            partials,
            col,
            merge_agg,
            partial_depends_on,
        )
        final_row_limit_met = (
            final_max_rows is None or len(partials) <= final_max_rows
        )
        if final_fits and final_row_limit_met:
            break
        if merge_level > max_merge_levels:
            raise ValueError(
                f"{task_id}: semantic aggregate exceeded {max_merge_levels} "
                "bounded merge levels"
            )

        merge_inputs = partials
        if merge_sort_by:
            missing_sort_fields = [
                field for field in merge_sort_by if field not in partials.columns
            ]
            if missing_sort_fields:
                raise ValueError(
                    f"{task_id}: merge sort fields are missing: "
                    f"{missing_sort_fields}"
                )
            merge_inputs = partials.sort_values(
                by=merge_sort_by,
                kind="stable",
                na_position="last",
            ).reset_index(drop=True)

        merge_chunks = _chunk_aggregate_inputs(
            merge_inputs,
            partial_col,
            partial_merge_agg,
            partial_depends_on,
            max_rows=merge_max_rows,
        )
        merged_frames = []
        for index, chunk in enumerate(merge_chunks, start=1):
            stage = (
                f"merge-l{merge_level}-c{index:03d}-of-"
                f"{len(merge_chunks):03d}"
            )
            call = execute_call(
                stage,
                chunk,
                output_col=partial_col,
                instruction=partial_merge_agg,
                input_fields=partial_depends_on,
            )
            merged_frames.append(
                _flatten_partial_aggregate(
                    call,
                    partial_field,
                    partial_depends_on,
                )
            )

        merged = pd.concat(merged_frames, ignore_index=True)
        if merged.empty:
            raise ValueError(f"{task_id}: merge sem_agg returned empty")
        if len(merge_chunks) == len(partials) and len(merged) >= len(partials):
            raise ValueError(
                f"{task_id}: bounded sem_agg merge made no reduction progress"
            )
        partials = merged
        merge_level += 1

    if final_additive_fields:
        if not final_group_by:
            raise ValueError(
                f"{task_id}: final additive merge needs group fields"
            )
        required_fields = [*final_group_by, *final_additive_fields]
        missing_fields = [
            field for field in required_fields if field not in partials.columns
        ]
        if missing_fields:
            raise ValueError(
                f"{task_id}: final additive fields are missing: "
                f"{missing_fields}"
            )
        additive = partials.loc[:, required_fields].copy()
        for field in final_additive_fields:
            additive[field] = pd.to_numeric(additive[field], errors="raise")
        input_rows = len(additive)
        partials = (
            additive.groupby(
                final_group_by,
                as_index=False,
                sort=False,
                dropna=False,
            )[final_additive_fields]
            .sum()
            .reset_index(drop=True)
        )
        print(
            f"[SEM_AGG] {task_id} structurally consolidated additive "
            f"statistics: {input_rows} -> {len(partials)} rows",
            flush=True,
        )

    if final_numeric_minimums:
        selected = pd.Series(True, index=partials.index)
        for field, minimum in final_numeric_minimums.items():
            if field not in partials.columns:
                raise ValueError(
                    f"{task_id}: final minimum field is missing: {field}"
                )
            selected &= pd.to_numeric(
                partials[field],
                errors="raise",
            ).ge(minimum)
        input_rows = len(partials)
        partials = partials.loc[selected].reset_index(drop=True)
        print(
            f"[SEM_AGG] {task_id} applied structural minimums: "
            f"{input_rows} -> {len(partials)} rows",
            flush=True,
        )
        if partials.empty:
            raise ValueError(
                f"{task_id}: no aggregate group satisfies final minimums"
            )

    final_instruction = merge_agg
    if final_candidate_max_rows is not None:
        if not final_group_by:
            raise ValueError(
                f"{task_id}: final candidate chunking needs group fields"
            )
        final_instruction = (
            f"{merge_agg}\n"
            "Silently verify every constraint before answering. Emit exactly "
            "one final JSON answer and do not emit a provisional answer, "
            "explanation, recalculation, or revision."
        )
        if len(partials) > final_candidate_max_rows:
            candidate_chunks = _chunk_grouped_aggregate_inputs(
                partials,
                final_group_by,
                final_candidate_max_rows,
            )
            print(
                f"[SEM_AGG] {task_id} using {len(candidate_chunks)} local "
                "candidate calls before the final aggregate",
                flush=True,
            )
            candidate_frames = []
            final_field = str(col["name"])
            candidate_instruction = (
                f"{final_instruction}\nThis is one bounded candidate pass."
            )
            for index, chunk in enumerate(candidate_chunks, start=1):
                stage = (
                    f"final-candidates-c{index:03d}-of-"
                    f"{len(candidate_chunks):03d}"
                )
                call = execute_call(
                    stage,
                    chunk,
                    output_col=col,
                    instruction=candidate_instruction,
                    input_fields=partial_depends_on,
                )
                candidate_frames.append(
                    _flatten_partial_aggregate(
                        call,
                        final_field,
                        partial_depends_on,
                    )
                )
            partials = pd.concat(candidate_frames, ignore_index=True)
            if partials.empty:
                raise ValueError(
                    f"{task_id}: final candidate aggregates returned empty"
                )

    final_call = execute_call(
        "final",
        partials,
        output_col=col,
        instruction=final_instruction,
        input_fields=partial_depends_on,
    )
    return SemanticAggregateExecution(
        output=final_call.output,
        calls=calls,
        chunked=True,
        resumed_elapsed_seconds=0.0,
    )


def _parse_structured(value):
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(stripped)
        except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return stripped


def normalize_scalar_value(value):
    """Unwrap singleton outputs; ambiguous multi-value scalars stay missing."""
    while not pd.api.types.is_scalar(value):
        if getattr(value, "ndim", None) == 0:
            value = value.item()
        elif (
            pd.api.types.is_list_like(value)
            and not isinstance(value, dict)
            and len(value) == 1
        ):
            value = next(iter(value))
        else:
            return None
    return None if value is None or pd.isna(value) else value


def normalize_text_value(
    value, *, null_markers: Iterable[str] = ()
) -> str | None:
    """Keep all list-valued text without expanding rows or inferring content."""
    null_markers = frozenset(marker.casefold() for marker in null_markers)
    if getattr(value, "ndim", None) == 0 and not pd.api.types.is_scalar(value):
        value = value.item()
    if pd.api.types.is_list_like(value) and not isinstance(value, dict):
        parts = [
            text
            for item in value
            if (text := normalize_text_value(item, null_markers=null_markers))
            is not None
        ]
        if isinstance(value, (set, frozenset)):
            parts.sort()
        return "; ".join(parts) or None
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return None
    text = " ".join(str(value).strip().split())
    return None if not text or text.casefold() in null_markers else text


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        return value == 1
    return str(value).strip().lower() in {"true", "yes", "1"}


def normalize_enum(value, allowed_labels: Iterable[str]) -> str | None:
    allowed = list(allowed_labels)
    normalized = re.sub(
        r"[^a-z0-9]+", "_", str(value).strip().lower()
    ).strip("_")
    return normalized if normalized in allowed else None


def parse_label_list(value, allowed_labels: Iterable[str]) -> list[str]:
    allowed = list(allowed_labels)
    parsed = _parse_structured(value)
    if isinstance(parsed, str):
        parsed = re.split(r"[,;|\n]+", parsed)
    if not isinstance(parsed, (list, tuple, set)):
        parsed = [parsed]
    present = {
        label
        for item in parsed
        if (label := normalize_enum(item, allowed)) is not None
    }
    return [label for label in allowed if label in present]


def parse_record_list(value) -> list[dict]:
    """Normalize a semantic aggregate's list-valued field."""
    parsed = _parse_structured(value)
    if isinstance(parsed, dict) and isinstance(parsed.get("rows"), list):
        parsed = parsed["rows"]
    if not isinstance(parsed, list) or not all(
        isinstance(item, dict) for item in parsed
    ):
        raise ValueError("semantic aggregate did not return a list of objects")
    return parsed


def normalize_free_label(value, default: str = "unspecified") -> str:
    normalized = re.sub(
        r"[^a-z0-9]+", "_", str(value).strip().lower()
    ).strip("_")
    if normalized in {"", "none", "null", "not_applicable", "n_a"}:
        return default
    return normalized


def stable_mode(values: pd.Series) -> str:
    cleaned = [str(value) for value in values if pd.notna(value) and str(value)]
    if not cleaned:
        return "unspecified"
    counts = Counter(cleaned)
    highest = max(counts.values())
    return sorted(label for label, count in counts.items() if count == highest)[0]


def _json_safe(value):
    if isinstance(value, pd.DataFrame):
        return df_records(value)
    if isinstance(value, pd.Series):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _json_safe(value),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _dataset_scoped_dir(kind: str) -> Path:
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    if main_file:
        script_dir = Path(main_file).resolve().parent
        if script_dir.name.startswith("pipelines-"):
            dataset = script_dir.name.removeprefix("pipelines-")
            return script_dir.parent / f"{kind}-{dataset}"
    return BASE_DIR / kind


def _resolve_results_dir() -> Path:
    return Path(
        os.getenv("PZ_RESULTS_DIR", str(_dataset_scoped_dir("results")))
    ).resolve()


def _semantic_totals(result) -> dict[str, int | float]:
    """Read only aggregate cost and text-token totals from Palimpzest."""
    execution_stats = getattr(result, "execution_stats", None)
    if execution_stats is None:
        return {"cost_usd": 0.0, "total_tokens": 0}

    input_tokens = float(
        getattr(execution_stats, "input_text_tokens", 0.0) or 0.0
    )
    output_tokens = float(
        getattr(execution_stats, "output_text_tokens", 0.0) or 0.0
    )
    cost = float(
        getattr(execution_stats, "total_execution_cost", 0.0) or 0.0
    )

    if input_tokens == 0.0 and output_tokens == 0.0:
        sum_field = getattr(execution_stats, "sum_plan_stats_field", None)
        if callable(sum_field):
            input_tokens = float(sum_field("input_text_tokens") or 0.0)
            output_tokens = float(sum_field("output_text_tokens") or 0.0)
    if cost == 0.0:
        sum_sentinel = getattr(
            execution_stats,
            "sum_sentinel_plan_costs",
            None,
        )
        sum_plans = getattr(execution_stats, "sum_plan_costs", None)
        if callable(sum_sentinel) and callable(sum_plans):
            cost = float(sum_sentinel() or 0.0) + float(sum_plans() or 0.0)

    return {
        "cost_usd": cost,
        "total_tokens": int(round(input_tokens + output_tokens)),
    }



class StepTracker:
    """Aggregate-only accounting compatible with existing pipeline calls."""

    def __init__(self, task_id: str, optimizer_strategy: str | None = None):
        self.task_id = task_id
        self.optimizer_strategy = optimizer_strategy
        self.cost_usd = 0.0
        self.total_tokens = 0

    def record(self, *_args, **_kwargs) -> None:
        return None

    def record_semantic(
        self,
        _name: str,
        _input_rows,
        _output,
        result,
        _elapsed: float,
        **_kwargs,
    ) -> None:
        usage = _semantic_totals(result)
        self.cost_usd += float(usage["cost_usd"])
        self.total_tokens += int(usage["total_tokens"])

    def get_execution_metrics(self) -> dict:
        return {
            "totals": {
                "cost_usd": round(self.cost_usd, 8),
                "total_tokens": self.total_tokens,
            }
        }



def save_output(
    task_id: str,
    answer,
    elapsed: float,
    tracker: StepTracker,
) -> Path:
    if answer is None or (isinstance(answer, str) and not answer.strip()):
        raise ValueError(f"{task_id}: invalid empty answer")
    metrics = tracker.get_execution_metrics()
    result = {
        "task_id": task_id,
        "answer": answer,
        "elapsed_seconds": round(elapsed, 2),
        "cost_usd": round(metrics["totals"]["cost_usd"], 6),
        "total_tokens": int(metrics["totals"]["total_tokens"]),
    }
    output_path = _resolve_results_dir() / f"{task_id}.json"
    _write_json(output_path, result)
    print(f"Result: {output_path}")
    return output_path


class Timer:
    def __init__(self):
        self.started = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.started = time.time()
        return self

    def __exit__(self, *_args):
        self.elapsed = time.time() - self.started
