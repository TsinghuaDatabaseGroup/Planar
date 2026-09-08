"""Shared LOTUS helpers for the testbed pipelines."""

import ast
import json
import math
import os
import re
import sys
import time
from collections.abc import Iterable

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import litellm
import pandas as pd
from dotenv import load_dotenv

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTBED_RUNNER_DIR = os.path.dirname(_BASE_DIR)
_REPO_ROOT = os.path.abspath(os.path.join(_BASE_DIR, "..", "..", ".."))
_BASELINE_DIR = os.path.join(_REPO_ROOT, "baseline", "lotus")
load_dotenv(os.path.join(_TESTBED_RUNNER_DIR, ".env"))

import lotus  # noqa: E402
from lotus.cache import CacheConfig, CacheFactory, CacheType  # noqa: E402
from lotus.models import LM  # noqa: E402
from lotus.models.litellm_rm import LiteLLMRM  # noqa: E402
from lotus.vector_store import FaissVS  # noqa: E402

API_BASE = os.getenv("LLM_API_BASE", "").strip()
if API_BASE:
    os.environ["OPENAI_API_BASE"] = API_BASE
    os.environ["OPENAI_BASE_URL"] = API_BASE

DATA_ROOT = os.getenv("LOTUS_DATA_ROOT", os.path.join(_BASELINE_DIR, "data"))
CACHE_DIR = os.path.join(_BASE_DIR, ".lotus_cache")
JOIN_OPTIMIZATION_ENV = "LOTUS_JOIN_OPTIMIZATION_ENABLED"
CACHE_DIR_ENV = "LOTUS_CACHE_DIR"
PIPELINE_BATCHING_ENV = "LOTUS_PIPELINE_BATCHING_ENABLED"
EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}

LM_MODEL = "Qwen3.5-397B-A17B"
LM_MAX_INPUT_TOKENS = 128000
LM_MAX_OUTPUT_TOKENS = 32768
EMBED_MODEL = "Qwen3-Embedding-8B"
EMBED_MAX_INPUT_TOKENS = 32768
DEFAULT_RM_BATCH_SIZE = 64

_STRUCTURED_NULL_VALUES = {
    "",
    "none",
    "null",
    "n/a",
    "na",
    "not_applicable",
}
_COMPANY_LEGAL_SUFFIXES = {
    "co",
    "company",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "limited",
    "llc",
    "llp",
    "lp",
    "ltd",
    "plc",
}

litellm.register_model(
    {
        LM_MODEL: {
            "max_tokens": LM_MAX_INPUT_TOKENS,
            "max_input_tokens": LM_MAX_INPUT_TOKENS,
            "max_output_tokens": LM_MAX_OUTPUT_TOKENS,
            "input_cost_per_token": 0.00000039,
            "output_cost_per_token": 0.00000234,
            "litellm_provider": "openai",
        },
        EMBED_MODEL: {
            "max_tokens": EMBED_MAX_INPUT_TOKENS,
            "max_input_tokens": EMBED_MAX_INPUT_TOKENS,
            "input_cost_per_token": 0.00000001,
            "output_cost_per_token": 0.0,
            "litellm_provider": "openai",
            "mode": "embedding",
            "output_vector_size": 4096,
        },
    }
)


def _configured_api_base() -> str:
    if not API_BASE:
        raise RuntimeError(
            "LLM_API_BASE is required; create testbed/runner/.env from "
            "testbed/runner/.env.example"
        )
    return API_BASE


def get_lm(
    max_tokens: int = 4096,
    max_batch_size: int = 10,
    task_prefix: str | None = None,
) -> LM:
    """Create the single text model used by every testbed dataset."""
    del task_prefix
    return LM(
        model=LM_MODEL,
        api_base=_configured_api_base(),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.0,
        max_ctx_len=LM_MAX_INPUT_TOKENS,
        max_tokens=min(max_tokens, LM_MAX_OUTPUT_TOKENS),
        max_batch_size=max_batch_size,
        timeout=300,
        num_retries=3,
        extra_body=EXTRA_BODY,
    )


def get_rm(
    max_batch_size: int = DEFAULT_RM_BATCH_SIZE,
    task_prefix: str | None = None,
) -> LiteLLMRM:
    """Create the embedding model used by native LOTUS operators."""
    del task_prefix
    _configured_api_base()
    return LiteLLMRM(model=EMBED_MODEL, max_batch_size=max_batch_size)


def setup(
    max_tokens: int = 4096,
    max_batch_size: int = 10,
    rm_batch_size: int = DEFAULT_RM_BATCH_SIZE,
    use_rm: bool = False,
    task_prefix: str | None = None,
):
    """Configure one LM and, when required, the native RM/vector store."""
    max_batch_size = max(
        int(os.getenv("LOTUS_LLM_MAX_CONCURRENCY", str(max_batch_size))),
        1,
    )
    join_optimization_enabled = os.environ.get(
        JOIN_OPTIMIZATION_ENV,
        "",
    ).strip().lower() in {"1", "true", "yes", "on"}
    effective_use_rm = use_rm or join_optimization_enabled
    cache_dir = os.environ.get(CACHE_DIR_ENV, CACHE_DIR)
    cache = CacheFactory.create_cache(
        CacheConfig(
            CacheType.SQLITE,
            max_size=4096,
            cache_dir=cache_dir,
        )
    )
    lm = get_lm(
        max_tokens=max_tokens,
        max_batch_size=max_batch_size,
        task_prefix=task_prefix,
    )
    lm.cache = cache
    rm = (
        get_rm(max_batch_size=rm_batch_size, task_prefix=task_prefix)
        if effective_use_rm
        else None
    )
    vs = FaissVS() if effective_use_rm else None
    lotus.settings.configure(
        lm=lm,
        rm=rm,
        vs=vs,
        enable_cache=True,
    )
    return lm, rm

# ---------------------------------------------------------------------------
# 数据加载（数据集无关，根目录为 baseline/lotus/data/<dataset>/）
# ---------------------------------------------------------------------------


def _dataset_path(dataset: str, *parts: str) -> str:
    return os.path.join(DATA_ROOT, dataset, *parts)


def load_table(dataset: str, filename: str, **read_csv_kwargs) -> pd.DataFrame:
    """加载 CSV 表格。布尔字面量列（如 true/false）由 pandas 解析为 bool。"""
    return pd.read_csv(_dataset_path(dataset, filename), **read_csv_kwargs)


def load_jsonl(dataset: str, filename: str, sep: str = "_") -> pd.DataFrame:
    """加载 JSONL，嵌套字段用 sep 扁平化。

    例如 risk.defect_summary -> risk_defect_summary（langex 的 .format()
    不支持带点的列名）。
    """
    path = _dataset_path(dataset, filename)
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return pd.json_normalize(records, sep=sep)


def load_docs(dataset: str, subdir: str = "") -> pd.DataFrame:
    """加载文档目录（txt/pdf/htm/html）为 {doc_id, contents} DataFrame。"""
    doc_dir = _dataset_path(dataset, subdir) if subdir else _dataset_path(dataset)
    records = []
    for fname in sorted(os.listdir(doc_dir)):
        fpath = os.path.join(doc_dir, fname)
        if not os.path.isfile(fpath):
            continue
        lower = fname.lower()
        if lower.endswith(".pdf"):
            import fitz

            try:
                doc = fitz.open(fpath)
                text = "\n".join(page.get_text() for page in doc)
                doc.close()
            except Exception:
                text = f"[PDF: {fname}]"
        elif lower.endswith((".htm", ".html")):
            with open(fpath, encoding="utf-8", errors="replace") as f:
                raw = f.read()
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(raw, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
        else:
            with open(fpath, encoding="utf-8", errors="replace") as f:
                text = f.read()
        records.append({"doc_id": fname, "contents": text})
    return pd.DataFrame(records)


def load_document_corpus(dataset: str, subdir: str = "") -> pd.DataFrame:
    """Load a document corpus with the metadata used by document benchmarks."""
    documents = load_docs(dataset, subdir).rename(
        columns={"doc_id": "document_id", "contents": "text"}
    )
    if documents.empty:
        return pd.DataFrame(
            columns=["document_id", "file_extension", "word_count", "text"]
        )

    documents["file_extension"] = documents["document_id"].map(
        lambda value: os.path.splitext(str(value))[1].lower()
    )
    documents["word_count"] = documents["text"].map(
        lambda value: len(str(value).split())
    )
    return documents[
        ["document_id", "file_extension", "word_count", "text"]
    ].reset_index(drop=True)


def load_selected_texts(
    dataset: str,
    records: pd.DataFrame,
    path_column: str = "text_file",
    output_column: str = "text",
) -> pd.DataFrame:
    """Load exactly the dataset-relative text files selected by a dataframe."""
    if path_column not in records.columns:
        raise ValueError(f"records dataframe must contain {path_column}")

    dataset_dir = os.path.abspath(_dataset_path(dataset))
    loaded_records = []
    for record in records.to_dict(orient="records"):
        relative_path = str(record[path_column])
        if os.path.isabs(relative_path):
            raise ValueError(f"absolute dataset text path is not allowed: {relative_path}")

        text_path = os.path.abspath(os.path.join(dataset_dir, relative_path))
        if os.path.commonpath([dataset_dir, text_path]) != dataset_dir:
            raise ValueError(f"text path escapes data/{dataset}: {relative_path}")
        if not os.path.isfile(text_path):
            raise FileNotFoundError(f"dataset text file not found: {text_path}")

        loaded = dict(record)
        with open(text_path, encoding="utf-8", errors="replace") as handle:
            loaded[output_column] = handle.read()
        loaded_records.append(loaded)

    return pd.DataFrame.from_records(
        loaded_records,
        columns=[*records.columns, output_column],
    )


def iter_selected_texts(
    dataset: str,
    records: pd.DataFrame,
    path_column: str = "text_file",
    output_column: str = "text",
    batch_size: int = 100,
):
    """Yield selected documents in bounded row batches without changing row semantics."""
    for batch in _iter_record_batches(records, batch_size):
        yield load_selected_texts(
            dataset,
            batch,
            path_column=path_column,
            output_column=output_column,
        )


def _pipeline_batching_enabled() -> bool:
    value = os.environ.get(PIPELINE_BATCHING_ENV, "0").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _iter_record_batches(records: pd.DataFrame, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if records.empty:
        return
    if not _pipeline_batching_enabled():
        yield records.copy()
        return
    for start in range(0, len(records), batch_size):
        yield records.iloc[start : start + batch_size].copy()


def _concat_batch_frames(
    source: pd.DataFrame, parts: list[pd.DataFrame], added_columns: Iterable[str] = ()
) -> pd.DataFrame:
    if parts:
        return pd.concat(parts, ignore_index=True)
    empty = source.iloc[0:0].copy()
    for column in added_columns:
        if column not in empty.columns:
            empty[column] = pd.Series(dtype="object")
    return empty


def sem_filter_in_batches(
    records: pd.DataFrame,
    instruction: str,
    batch_size: int = 100,
    **kwargs,
) -> pd.DataFrame:
    """Run LOTUS sem_filter in bounded row batches with identical row semantics."""
    parts = []
    for batch in _iter_record_batches(records, batch_size):
        parts.append(batch.sem_filter(instruction, **kwargs))
    return _concat_batch_frames(records, parts)


def sem_map_in_batches(
    records: pd.DataFrame,
    instruction: str,
    suffix: str,
    batch_size: int = 100,
    **kwargs,
) -> pd.DataFrame:
    """Run LOTUS sem_map in bounded row batches with identical row semantics."""
    parts = []
    for batch in _iter_record_batches(records, batch_size):
        parts.append(batch.sem_map(instruction, suffix=suffix, **kwargs))
    return _concat_batch_frames(records, parts, added_columns=[suffix])


def sem_extract_in_batches(
    records: pd.DataFrame,
    input_cols: list[str],
    output_cols: dict[str, str | None],
    batch_size: int = 100,
    **kwargs,
) -> pd.DataFrame:
    """Run LOTUS sem_extract in bounded row batches with identical row semantics."""
    parts = []
    for batch in _iter_record_batches(records, batch_size):
        parts.append(
            batch.sem_extract(
                input_cols=input_cols,
                output_cols=output_cols,
                **kwargs,
            )
        )
    return _concat_batch_frames(records, parts, added_columns=output_cols)


def parse_bool(value) -> bool:
    """Conservatively parse a structured extraction boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    return str(value).strip().lower() in {"true", "yes", "1"}


def parse_optional_bool(value) -> bool | None:
    """Parse an extracted boolean while preserving null or malformed values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    return None


def normalize_enum(value, allowed_labels: Iterable[str]) -> str | None:
    """Normalize a model value only when it matches an allowed enum label."""
    allowed = set(allowed_labels)
    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")
    return normalized if normalized in allowed else None


def _parse_structured_value(value):
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(stripped)
        except (TypeError, ValueError, SyntaxError):
            continue
    return stripped


def parse_label_list(value, allowed_labels: Iterable[str]) -> list[str]:
    """Parse a JSON/Python-style list and retain only allowed labels."""
    allowed = list(allowed_labels)
    parsed = _parse_structured_value(value)
    if isinstance(parsed, str):
        parsed = re.split(r"[,;|\n]+", parsed)

    if not isinstance(parsed, (list, tuple, set)):
        parsed = [parsed]

    present = {label for item in parsed if (label := normalize_enum(item, allowed)) is not None}
    return [label for label in allowed if label in present]


def parse_string_list(value) -> list[str]:
    """Parse a structured list of scalar strings, preserving source order."""
    parsed = _parse_structured_value(value)
    if isinstance(parsed, str):
        parsed = re.split(r"[;|\n]+", parsed)
    if not isinstance(parsed, (list, tuple, set)):
        parsed = [parsed]

    output = []
    seen = set()
    for item in parsed:
        if isinstance(item, (dict, list, tuple, set)):
            continue
        text = str(item).strip().strip("`\"'")
        if text.lower() in _STRUCTURED_NULL_VALUES or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def parse_record_list(value) -> list[dict]:
    """Parse a structured object or list of objects from sem_extract output."""
    parsed = _parse_structured_value(value)
    if isinstance(parsed, dict):
        return [parsed]
    if not isinstance(parsed, (list, tuple)):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def parse_number(value) -> int | float | None:
    """Parse the first decimal number from a scalar value."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return int(value) if float(value).is_integer() else float(value)

    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", str(value))
    if match is None:
        return None
    parsed = float(match.group(0).replace(",", ""))
    return int(parsed) if parsed.is_integer() else parsed


def parse_number_after(value, marker: str) -> int | float | None:
    """Parse the first number appearing after a literal marker."""
    text = str(value)
    marker_index = text.find(marker)
    if marker_index < 0:
        return None
    return parse_number(text[marker_index + len(marker) :])


def clean_text(value, default: str = "unspecified") -> str:
    """Trim a free-form phrase while preserving its natural word boundaries."""
    text = " ".join(str(value).strip().strip("`\"'").split())
    return default if text.lower() in _STRUCTURED_NULL_VALUES else text


def clean_optional_text(value) -> str | None:
    """Trim a nullable extracted phrase without inventing a fallback value."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    text = " ".join(str(value).strip().strip("`\"'").split())
    return None if text.lower() in _STRUCTURED_NULL_VALUES else text


def normalize_iso_date(value) -> str | None:
    """Normalize a date-like extraction to YYYY-MM-DD, preserving invalids as null."""
    text = clean_optional_text(value)
    if text is None:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def normalize_company_name(value) -> str | None:
    """Case-fold a company name, remove punctuation, and trim legal suffixes."""
    text = clean_optional_text(value)
    if text is None:
        return None
    tokens = re.sub(r"[^a-z0-9]+", " ", text.casefold()).split()
    while tokens and tokens[-1] in _COMPANY_LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens) or None


def normalize_free_label(value, default: str = "unspecified") -> str:
    """Normalize an open short label without assigning semantic content."""
    text = str(value).strip().strip("`\"'")
    if text.lower() in _STRUCTURED_NULL_VALUES:
        return default
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return normalized or default


def stable_mode(values: pd.Series) -> str:
    """Return a deterministic mode, using lexical order only to break ties."""
    cleaned = [str(value) for value in values if pd.notna(value) and str(value)]
    if not cleaned:
        return "unspecified"
    counts = pd.Series(cleaned).value_counts()
    max_count = int(counts.max())
    return sorted(counts[counts == max_count].index.tolist())[0]


def stable_mode_optional(values: pd.Series) -> str | None:
    """Return an ignore-null deterministic mode, or null when no value exists."""
    cleaned = [str(value) for value in values if pd.notna(value) and str(value)]
    if not cleaned:
        return None
    counts = pd.Series(cleaned).value_counts()
    max_count = int(counts.max())
    return sorted(counts[counts == max_count].index.tolist())[0]


# ---------------------------------------------------------------------------
# Compact final accounting
# ---------------------------------------------------------------------------


def _physical_lm_totals() -> dict[str, int | float]:
    model = getattr(lotus.settings, "lm", None)
    usage = getattr(getattr(model, "stats", None), "physical_usage", None)
    if usage is None:
        return {"cost_usd": 0.0, "total_tokens": 0}
    return {
        "cost_usd": float(getattr(usage, "total_cost", 0.0) or 0.0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


class StepTracker:
    """Compatibility shim; testbed copies do not record per-operation details."""

    def __init__(self, task_id: str | None = None):
        self.task_id = task_id

    def record(self, *_args, **_kwargs) -> None:
        return None

    def step(self, *_args, **_kwargs):
        return _SemStep()

    def get_execution_metrics(self) -> dict:
        return {"totals": _physical_lm_totals()}


class _SemStep:
    def set_output(self, _output) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

# ---------------------------------------------------------------------------
# 结果输出
# ---------------------------------------------------------------------------


def df_records(df: pd.DataFrame) -> list:
    """DataFrame 转 JSON 安全的记录列表（numpy 类型转原生类型）。"""
    return json.loads(df.to_json(orient="records", force_ascii=False))


def clean_label(text: str) -> str:
    """清理标签型答案的首尾空白与句号。"""
    return str(text).strip().strip(".").strip()


def _is_valid_answer(answer) -> bool:
    """检查 answer 是否可保存；空表和空字典是合法查询结果。"""
    if answer is None:
        return False
    if isinstance(answer, str):
        return len(answer.strip()) > 0
    if isinstance(answer, (list, dict)):
        return True
    if isinstance(answer, (int, float, bool)):
        return True
    return answer is not None


def _dataset_scoped_dir(kind: str) -> str:
    """根据 pipelines-<dataset> 主脚本位置推导同级的 kind-<dataset>。"""
    main_mod = sys.modules.get("__main__")
    main_file = getattr(main_mod, "__file__", None)
    if main_file:
        script_dir = os.path.dirname(os.path.abspath(main_file))
        dirname = os.path.basename(script_dir)
        if dirname.startswith(("pipelines-", "pipeline-")):
            prefix = "pipelines-" if dirname.startswith("pipelines-") else "pipeline-"
            dataset = dirname.removeprefix(prefix)
            return os.path.join(os.path.dirname(script_dir), f"{kind}-{dataset}")
        return os.path.join(script_dir, kind)
    return os.path.join(os.getcwd(), kind)


def _resolve_results_dir() -> str:
    """最终结果目录；LOTUS_OUTPUT_DIR 仅作为旧调用的兼容回退。"""
    env_dir = os.environ.get("LOTUS_RESULTS_DIR")
    if env_dir:
        return env_dir
    legacy_dir = os.environ.get("LOTUS_OUTPUT_DIR")
    if legacy_dir:
        return legacy_dir
    return _dataset_scoped_dir("results")


def _json_safe(value):
    """将 pandas/numpy 值转换为可稳定写入 JSON 的原生对象。"""
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
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: str, value) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temp_path, path)


def save_output(
    task_id: str,
    answer,
    elapsed: float = 0.0,
    cost: float = 0.0,
    tracker: StepTracker | None = None,
):
    """Persist one compact testbed result with aggregate text-LLM usage."""
    del tracker
    if not _is_valid_answer(answer):
        raise ValueError(f"{task_id}: answer 为空或无效 (type={type(answer).__name__}, value={answer!r})，拒绝保存")
    totals = _physical_lm_totals()
    if cost == 0.0:
        cost = float(totals["cost_usd"])
    result = {
        "task_id": task_id,
        "answer": answer,
        "elapsed_seconds": round(elapsed, 2),
        "cost_usd": round(cost, 6),
        "total_tokens": int(totals["total_tokens"]),
    }

    out_dir = _resolve_results_dir()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task_id}.json")
    _write_json(out_path, result)
    print(f"Result: {out_path}")
    return out_path


class Timer:
    """简单计时器，用于记录 pipeline 执行时间。"""

    def __init__(self):
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
