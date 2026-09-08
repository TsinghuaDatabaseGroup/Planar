#!/usr/bin/env python3
"""Run Planar tasks with the local Claude Code agent.

Runtime state, generated helpers, caches, and diagnostic outputs are preserved
under this launcher's persistent per-domain run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
RUNNER_DIR = ROOT_DIR / "runner"
DEFAULT_RESULTS_ROOT = ROOT_DIR / "results"
DEFAULT_RUN_ROOT = Path(__file__).resolve().parent / "runs"
DEFAULT_ENV_PATH = RUNNER_DIR / ".env"
DEFAULT_CLAUDE_COMMAND = "claude"
TEXT_MODEL = "Qwen3.5-397B-A17B"
DATASET_NAMES = ("aviation_safety", "vehicle_safety", "finance", "legal_contracts")
DEFAULT_OPERATOR_IMPLEMENTATION_DATA_DIR = ROOT_DIR / "data"
DEFAULT_PLAN_OPTIMIZATION_RUN_DIR = ROOT_DIR / "claudecode_run_plan_optimization_all"
DEFAULT_OPTIMIZED_PLAN_SELECTION_NAME = "four_type_optimization_tasks.json"
DEFAULT_AGENT_ENV_PATH = DEFAULT_ENV_PATH

CACHE_VERSION = 4
DEFAULT_TIMEOUT_SECS = 14400
DEFAULT_AGENT_TIMEOUT_SECS = 3600
DEFAULT_DATASET_CACHE_TIMEOUT_SECS = 1800
DEFAULT_TASK_MAX_ATTEMPTS = 3
DEFAULT_AGENT_EXIT_GRACE_SECS = 120
DEFAULT_TOOL_MODEL_MAX_CONCURRENCY = 10
DEFAULT_TOOL_MODEL_MAX_TOKENS = 2048
DEFAULT_RETRY_SESSION_MODE = "fresh"
RETRY_SESSION_MODES = ("fresh", "resume")
DEFAULT_TASK_MODE = "benchmark"
TASK_MODES = (
    "benchmark",
    "operator-implementation",
    "plan-optimization",
    "optimized-plan-execution",
)
OPTIMIZATION_TYPES = (
    "relational_semantic_reordering",
    "multi_semantic_reordering",
    "multi_join_reordering",
    "semantic_to_relational_rewrite",
)
RESUMED_SESSION_RETENTION_DAYS = 30
MAX_RETRY_STDERR_TAIL_CHARS = 8000
MAX_RETRY_STDOUT_TAIL_CHARS = 4000

SECRET_KEY_PATTERN = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)", re.IGNORECASE)
SCRIPT_SUFFIXES = {".py", ".sh", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".ipynb"}
ENV_REFERENCE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
AGENT_ENV_MAPPINGS: dict[str, tuple[str, ...]] = {
    "AGENT_MAIN_MODEL": ("ANTHROPIC_MODEL", "LLM_MAIN_MODEL", "LLM_MODEL_NAME"),
    "AGENT_API_BASE": ("ANTHROPIC_BASE_URL", "LLM_API_BASE"),
    "AGENT_API_KEY": ("ANTHROPIC_API_KEY",),
    "AGENT_AUTH_TOKEN": ("ANTHROPIC_AUTH_TOKEN",),
    "LLM_INPUT_COST_PER_TOKEN": ("LLM_MODEL_USD_PER_INPUT_TOKEN",),
    "LLM_OUTPUT_COST_PER_TOKEN": ("LLM_MODEL_USD_PER_OUTPUT_TOKEN",),
    "LLM_CACHE_READ_COST_PER_TOKEN": ("LLM_MODEL_USD_PER_CACHE_READ_TOKEN",),
    "LLM_CACHE_WRITE_COST_PER_TOKEN": ("LLM_MODEL_USD_PER_CACHE_WRITE_TOKEN",),
}
AGENT_REQUIRED_ENV_GROUPS: dict[str, tuple[str, ...]] = {
    "agent model": ("AGENT_MAIN_MODEL", "ANTHROPIC_MODEL", "LLM_MAIN_MODEL", "LLM_MODEL_NAME"),
    "agent auth": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "AGENT_API_KEY", "AGENT_AUTH_TOKEN"),
}
CLAUDE_PROVIDER_ENV_FLAGS = (
    "CLAUDE_CODE_USE_OPENAI",
    "CLAUDE_CODE_USE_GEMINI",
    "CLAUDE_CODE_USE_GROK",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
)
OPERATOR_IMPLEMENTATION_DATASET_DIRS = {
    "CONTRACTEXHIBIT": "contract-exhibit",
    "NASA_ASRS": "asrs",
    "NHTSA Vehicle Safety": "nhtsa",
    "SEC10K": "SEC",
}

_ACTIVE_CHILDREN: dict[int, tuple[subprocess.Popen[str], str]] = {}
_ACTIVE_CHILDREN_LOCK = threading.Lock()


class HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self.skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip_depth and data.strip():
            self.parts.append(data)

    def get_text(self) -> str:
        return " ".join(part.strip() for part in self.parts if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Planar tasks with Claude Code while preserving the original agent workflow."
    )
    parser.add_argument("--dataset", choices=DATASET_NAMES, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Persistent runtime directory; defaults to runner/claude_code/runs/<dataset>.",
    )
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument(
        "--task-mode",
        choices=TASK_MODES,
        default=DEFAULT_TASK_MODE,
        help=(
            "Task input contract: benchmark (default) preserves the existing behavior; "
            "operator-implementation executes supplied reference plans; "
            "plan-optimization only optimizes supplied plans without execution; "
            "optimized-plan-execution executes audited optimized plans from a prior optimization run."
        ),
    )
    parser.add_argument(
        "--plan-optimization-run-dir",
        type=Path,
        default=DEFAULT_PLAN_OPTIMIZATION_RUN_DIR,
        help=(
            "Prior plan-optimization run containing results.jsonl and output/. "
            "Used only by optimized-plan-execution."
        ),
    )
    parser.add_argument(
        "--optimized-plan-selection",
        type=Path,
        help=(
            "Machine-readable audited task selection. Defaults to "
            "<plan-optimization-run-dir>/four_type_optimization_tasks.json."
        ),
    )
    parser.add_argument(
        "--claude-command",
        default=os.environ.get("CLAUDE_COMMAND", DEFAULT_CLAUDE_COMMAND),
        help="Claude Code CLI command; defaults to CLAUDE_COMMAND or 'claude'.",
    )
    parser.add_argument(
        "--agent-env-path",
        type=Path,
        default=DEFAULT_AGENT_ENV_PATH,
        help="Shared testbed model configuration; defaults to testbed/runner/.env.",
    )
    parser.add_argument("--task", help="Run a single task_id")
    parser.add_argument(
        "--skip-task",
        action="append",
        default=[],
        help="Task ID(s) to skip. Repeat the flag or pass comma-separated IDs.",
    )
    parser.add_argument("--limit", type=int, help="Run at most N selected pending tasks")
    parser.add_argument("--force", action="store_true", help="Rerun tasks even if already successful")
    parser.add_argument("--dry-run", action="store_true", help="Print selected pending tasks only")
    parser.add_argument("--clean", action="store_true", help="Delete claudecode_run/ before starting")
    parser.add_argument("--rebuild-dataset-cache", action="store_true")
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECS,
        help="Foreground task-script timeout in seconds (default: 14400).",
    )
    parser.add_argument(
        "--agent-timeout",
        type=int,
        default=DEFAULT_AGENT_TIMEOUT_SECS,
        help="Claude Code script-authoring timeout in seconds (default: 3600).",
    )
    parser.add_argument("--dataset-cache-timeout", type=int, default=DEFAULT_DATASET_CACHE_TIMEOUT_SECS)
    parser.add_argument("--task-max-attempts", type=int, default=DEFAULT_TASK_MAX_ATTEMPTS)
    parser.add_argument(
        "--retry-session-mode",
        choices=RETRY_SESSION_MODES,
        default=DEFAULT_RETRY_SESSION_MODE,
        help=(
            "Claude Code session behavior across attempts of one task: "
            "fresh (default) starts a new session for every attempt; "
            "resume continues the first attempt's conversation for retries."
        ),
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop immediately when a task exhausts its attempts. By default, failed tasks are recorded and skipped.",
    )
    parser.add_argument("--agent-exit-grace-secs", type=int, default=DEFAULT_AGENT_EXIT_GRACE_SECS)
    parser.add_argument(
        "--tool-model-max-concurrency",
        type=int,
        default=DEFAULT_TOOL_MODEL_MAX_CONCURRENCY,
        help="Tool-side model concurrency cap. Values above 10 are rejected.",
    )
    parser.add_argument(
        "--tool-model-max-tokens",
        type=int,
        default=DEFAULT_TOOL_MODEL_MAX_TOKENS,
        help="Tool-side per-call output token cap.",
    )
    parser.add_argument("--no-agent-log", action="store_true", help="Do not print live agent/tool summaries")
    args = parser.parse_args()
    args.input = args.input or ROOT_DIR / "queries" / args.dataset / "queries.json"
    args.data_dir = args.data_dir or ROOT_DIR / "data" / args.dataset
    args.run_dir = args.run_dir or DEFAULT_RUN_ROOT / args.dataset
    return args


def parse_dotenv(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        env[key.strip()] = value
    return env


def expand_env_references(env: dict[str, str]) -> None:
    for _ in range(8):
        changed = False
        for key, value in list(env.items()):
            if not isinstance(value, str):
                continue

            def replace(match: re.Match[str]) -> str:
                return env.get(match.group(1), match.group(0))

            expanded = ENV_REFERENCE_RE.sub(replace, value)
            if expanded != value:
                env[key] = expanded
                changed = True
        if not changed:
            return


def env_value_is_set(value: str | None) -> bool:
    return bool(value) and ENV_REFERENCE_RE.search(value) is None


def agent_env_with_process_overrides(file_env: dict[str, str]) -> dict[str, str]:
    env = dict(file_env)
    for key, value in os.environ.items():
        if key.startswith("ANTHROPIC_") or key.startswith("AGENT_"):
            env[key] = value
    return env


def sync_env_mappings(env: dict[str, str], mappings: dict[str, tuple[str, ...]]) -> list[str]:
    expand_env_references(env)
    synced: list[str] = []
    for canonical, aliases in mappings.items():
        keys = (canonical, *aliases)
        source_key = next((key for key in keys if env_value_is_set(env.get(key))), None)
        if source_key is None:
            continue
        source_value = env[source_key]
        for key in keys:
            if env.get(key) != source_value:
                env[key] = source_value
                synced.append(f"{key}<-{source_key}")
    expand_env_references(env)
    return synced


def sync_agent_env_mappings(env: dict[str, str]) -> list[str]:
    return sync_env_mappings(env, AGENT_ENV_MAPPINGS)


def missing_env_groups(env: dict[str, str], groups: dict[str, tuple[str, ...]]) -> list[str]:
    missing: list[str] = []
    for label, keys in groups.items():
        if not any(env_value_is_set(env.get(key)) for key in keys):
            missing.append(f"{label} ({' or '.join(keys)})")
    return missing


def require_env_value(env: dict[str, str], key: str) -> str:
    value = env.get(key)
    if not env_value_is_set(value):
        raise SystemExit(f"agent env/config is missing required field: {key}")
    return str(value)


def require_path(label: str, path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")


def require_under_root(label: str, path: Path) -> None:
    try:
        path.resolve().relative_to(ROOT_DIR.resolve())
    except ValueError as exc:
        raise SystemExit(f"{label} must stay under project root: {path}") from exc


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        ("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n")
        if records
        else "",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def env_key_is_secret(key: str) -> bool:
    upper_key = key.upper()
    non_secret_markers = (
        "TOKENS",
        "MAX_INPUT_TOKEN",
        "MAX_OUTPUT_TOKEN",
        "COST_PER_TOKEN",
        "USD_PER_INPUT_TOKEN",
        "USD_PER_OUTPUT_TOKEN",
        "TPM",
    )
    token_is_secret = "TOKEN" in upper_key and not any(marker in upper_key for marker in non_secret_markers)
    return token_is_secret or bool(SECRET_KEY_PATTERN.search(upper_key))


def secret_values(env: dict[str, str]) -> list[str]:
    values = []
    for key, value in env.items():
        if env_key_is_secret(key) and value and len(value) >= 6:
            values.append(value)
    return sorted(set(values), key=len, reverse=True)


def secret_values_from_envs(*envs: dict[str, str]) -> list[str]:
    values: list[str] = []
    for env in envs:
        values.extend(secret_values(env))
    return sorted(set(values), key=len, reverse=True)


def redact_text(text: str, secrets: list[str]) -> str:
    redacted = text
    for value in secrets:
        redacted = redacted.replace(value, "<redacted>")
    return redacted


def redact_payload(payload: Any, secrets: list[str]) -> Any:
    if isinstance(payload, str):
        return redact_text(payload, secrets)
    if isinstance(payload, list):
        return [redact_payload(item, secrets) for item in payload]
    if isinstance(payload, dict):
        return {key: redact_payload(value, secrets) for key, value in payload.items()}
    return payload


def assert_evaluation_input(input_path: Path) -> None:
    if not input_path.is_file():
        raise SystemExit(f"evaluation input is not a file: {input_path}")


def scoped_data_files(data_dir: Path, data_paths: list[Path] | None = None) -> list[Path]:
    data_root = data_dir.resolve()
    roots = [data_root] if data_paths is None else [path.resolve() for path in data_paths]
    files: set[Path] = set()
    for root in roots:
        try:
            root.relative_to(data_root)
        except ValueError as exc:
            raise SystemExit(f"data source is outside dataset directory: {root}") from exc
        if not root.exists():
            raise SystemExit(f"data source not found: {root}")
        if root.is_file():
            files.add(root)
        else:
            files.update(path.resolve() for path in root.rglob("*") if path.is_file())
    return sorted(files)


def assert_no_source_scripts(
    input_path: Path,
    data_dir: Path,
    data_paths: list[Path] | None = None,
) -> None:
    offenders: list[str] = []
    if input_path.suffix.lower() in SCRIPT_SUFFIXES:
        offenders.append(str(input_path.relative_to(ROOT_DIR)))
    for path in scoped_data_files(data_dir, data_paths):
        if path.suffix.lower() in SCRIPT_SUFFIXES:
            offenders.append(str(path.relative_to(ROOT_DIR)))
    if offenders:
        raise SystemExit("evaluation input or data directory contains script-like files: " + ", ".join(offenders[:20]))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_source_dirs(
    input_path: Path,
    data_dir: Path,
    *,
    include_hash: bool,
    data_paths: list[Path] | None = None,
) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    source_files = [input_path, *scoped_data_files(data_dir, data_paths)]
    for path in source_files:
        relative = str(path.relative_to(ROOT_DIR))
        stat = path.stat()
        item: dict[str, Any] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if include_hash:
            item["sha256"] = sha256_file(path)
        files[relative] = item
    return {"include_hash": include_hash, "files": files}


def assert_source_dirs_unchanged(
    expected: dict[str, Any],
    input_path: Path,
    data_dir: Path,
    *,
    include_hash: bool = False,
    data_paths: list[Path] | None = None,
) -> None:
    current = snapshot_source_dirs(
        input_path,
        data_dir,
        include_hash=include_hash,
        data_paths=data_paths,
    )
    expected_files = expected.get("files", {})
    current_files = current.get("files", {})
    if set(expected_files) != set(current_files):
        missing = sorted(set(expected_files) - set(current_files))
        added = sorted(set(current_files) - set(expected_files))
        raise SystemExit(f"evaluation input or data directory changed. missing={missing[:5]} added={added[:5]}")
    for name, expected_item in expected_files.items():
        current_item = current_files[name]
        for key in ("size", "mtime_ns"):
            if expected_item.get(key) != current_item.get(key):
                raise SystemExit(f"evaluation input or data directory changed: {name} ({key})")
        if include_hash and expected_item.get("sha256") != current_item.get("sha256"):
            raise SystemExit(f"evaluation input or data directory changed: {name} (sha256)")


def clean_run_dir(run_dir: Path) -> None:
    resolved = run_dir.resolve()
    require_under_root("run dir", resolved)
    if resolved.name != "claudecode_run":
        raise SystemExit(f"--clean refuses to delete non-standard run dir: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def dataset_cache_helper_source() -> str:
    return r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

CACHE_VERSION = 4
PDF_TIMEOUT_SECS = 120
ROOT_DIR = Path(
    os.environ.get("CLAUDECODE_PROJECT_ROOT")
    or os.environ.get("CONTRACT_EXHIBIT_ROOT")
    or Path(__file__).resolve().parents[2]
).resolve()
TEXT_PATH_COLUMN_CANDIDATES = (
    "text",
    "text_path",
    "text_file",
    "document_text",
    "document_text_path",
    "file_path",
    "filepath",
    "path",
    "source_path",
)
DOCUMENT_ID_COLUMN_CANDIDATES = (
    "document_id",
    "doc_id",
    "docid",
    "incident_id",
    "complaint_id",
    "recall_id",
    "id",
    "accession",
    "accession_number",
)
INLINE_TEXT_MIN_AVG_CHARS = 200
JSON_RECORD_SAMPLE_LIMIT = 50
JSON_RECORD_FILE_SUFFIXES = {".jsonl", ".ndjson"}
JSON_RECORD_MAX_ARRAY_BYTES = 256 * 1024 * 1024

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


class HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self.skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip_depth and data.strip():
            self.parts.append(data)

    def get_text(self) -> str:
        return " ".join(part.strip() for part in self.parts if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build complete dataset text cache.")
    parser.add_argument("--ensure", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--include-path", action="append", default=[])
    parser.add_argument("--partition-file-collections-by-top-level", action="store_true")
    return parser.parse_args()


def normalize_include_paths(data_dir: Path, values: list[str] | None = None) -> list[Path]:
    data_root = data_dir.resolve()
    if not values:
        return [data_root]
    candidates: list[Path] = []
    for value in values:
        candidate = (data_root / value).resolve()
        try:
            candidate.relative_to(data_root)
        except ValueError as exc:
            raise SystemExit(f"cache include path is outside data dir: {value}") from exc
        if not candidate.exists():
            raise SystemExit(f"cache include path not found: {value}")
        candidates.append(candidate)
    roots: list[Path] = []
    for candidate in sorted(set(candidates), key=lambda path: (len(path.parts), str(path).lower())):
        if any(candidate == root or root in candidate.parents for root in roots):
            continue
        roots.append(candidate)
    return roots


def include_path_labels(data_dir: Path, include_paths: list[Path]) -> list[str]:
    labels: list[str] = []
    for path in include_paths:
        relative = path.relative_to(data_dir)
        labels.append(relative.as_posix() if relative.parts else ".")
    return labels


def dataset_files(data_dir: Path, include_paths: list[Path] | None = None) -> list[Path]:
    roots = normalize_include_paths(data_dir) if include_paths is None else include_paths
    files: set[Path] = set()
    for root in roots:
        if root.is_file():
            files.add(root)
        else:
            files.update(path for path in root.rglob("*") if path.is_file())
    return sorted(files, key=lambda path: str(path.relative_to(data_dir)).lower())


def file_inventory(data_dir: Path, include_paths: list[Path] | None = None) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in dataset_files(data_dir, include_paths):
        stat = path.stat()
        rel = str(path.relative_to(data_dir))
        files.append({
            "document_id": path.name,
            "filename": path.name,
            "relative_path": rel,
            "extension": path.suffix.lower(),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    return files


def normalize_csv_key(key: str | None) -> str:
    return (key or "").lstrip("\ufeff").strip()


def normalize_csv_row(row: dict[str | None, str | None]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in row.items():
        normalized_key = normalize_csv_key(key)
        if normalized_key:
            normalized[normalized_key] = "" if value is None else str(value).strip()
    return normalized


def resolve_dataset_reference(data_dir: Path, csv_path: Path, value: str) -> Path | None:
    raw = str(value or "").strip()
    if not raw or len(raw) > 1024 or "\n" in raw or "\r" in raw or "\x00" in raw:
        return None
    candidate = Path(raw)
    data_root = data_dir.resolve()
    if candidate.is_absolute():
        candidates = [candidate]
    else:
        candidates = []
        ancestor = csv_path.parent.resolve()
        while ancestor == data_root or data_root in ancestor.parents:
            candidates.append(ancestor / candidate)
            if ancestor == data_root:
                break
            ancestor = ancestor.parent
    for path in candidates:
        try:
            resolved = path.resolve()
            resolved.relative_to(data_root)
            if resolved.is_file():
                return resolved
        except (OSError, ValueError):
            continue
    return None


def read_csv_sample(csv_path: Path, limit: int = 50) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = [normalize_csv_key(field) for field in (reader.fieldnames or []) if normalize_csv_key(field)]
            rows: list[dict[str, str]] = []
            for row in reader:
                rows.append(normalize_csv_row(row))
                if len(rows) >= limit:
                    break
    except Exception:
        return [], []
    return fieldnames, rows


def count_csv_rows(csv_path: Path) -> int:
    try:
        with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    except Exception:
        return 0


def detect_csv_text_path_table(data_dir: Path, csv_path: Path) -> dict[str, Any] | None:
    fieldnames, sample_rows = read_csv_sample(csv_path)
    if not fieldnames or not sample_rows:
        return None
    candidates = [
        column
        for column in fieldnames
        if column.lower() in TEXT_PATH_COLUMN_CANDIDATES or "text" in column.lower()
    ]
    best_column = None
    best_valid = 0
    best_nonempty = 0
    for column in candidates:
        nonempty = sum(1 for row in sample_rows if row.get(column))
        valid = sum(
            1
            for row in sample_rows
            if row.get(column) and resolve_dataset_reference(data_dir, csv_path, row[column]) is not None
        )
        if valid > best_valid:
            best_column = column
            best_valid = valid
            best_nonempty = nonempty
    threshold = max(1, min(best_nonempty, len(sample_rows)) // 2)
    if not best_column or best_valid < threshold:
        return None
    return {
        "csv_path": csv_path,
        "relative_path": csv_path.relative_to(data_dir).as_posix(),
        "columns": fieldnames,
        "text_path_column": best_column,
        "sample_rows": len(sample_rows),
        "sample_valid_text_paths": best_valid,
        "row_count": count_csv_rows(csv_path),
    }


def detect_csv_inline_text_table(data_dir: Path, csv_path: Path) -> dict[str, Any] | None:
    fieldnames, sample_rows = read_csv_sample(csv_path)
    if not fieldnames or not sample_rows:
        return None
    best_column = None
    best_avg_chars = 0.0
    for column in fieldnames:
        nonempty = [row[column] for row in sample_rows if row.get(column)]
        if len(nonempty) < max(1, len(sample_rows) // 2):
            continue
        avg_chars = sum(len(value) for value in nonempty) / len(nonempty)
        if avg_chars >= INLINE_TEXT_MIN_AVG_CHARS and avg_chars > best_avg_chars:
            best_column = column
            best_avg_chars = avg_chars
    if best_column is None:
        return None
    return {
        "csv_path": csv_path,
        "relative_path": csv_path.relative_to(data_dir).as_posix(),
        "columns": fieldnames,
        "inline_text_column": best_column,
        "avg_text_chars": round(best_avg_chars, 1),
        "sample_rows": len(sample_rows),
        "row_count": count_csv_rows(csv_path),
    }


def describe_csv_metadata_table(data_dir: Path, csv_path: Path) -> dict[str, Any] | None:
    fieldnames, sample_rows = read_csv_sample(csv_path)
    if not fieldnames:
        return None
    return {
        "csv_path": csv_path,
        "relative_path": csv_path.relative_to(data_dir).as_posix(),
        "columns": fieldnames,
        "sample_rows": len(sample_rows),
        "row_count": count_csv_rows(csv_path),
    }


def flatten_record(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in record.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_record(value, name))
        elif isinstance(value, list):
            flat[name] = "; ".join(
                json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item)
                for item in value
            )
        else:
            flat[name] = value
    return flat


def record_text_from_flat(flat: dict[str, Any]) -> str:
    return "\n".join(f"{key}: {value}" for key, value in flat.items() if value not in (None, ""))


def detect_json_record_file(data_dir: Path, path: Path) -> dict[str, Any] | None:
    suffix = path.suffix.lower()
    sample: list[dict[str, Any]] = []
    record_count = 0
    if suffix in JSON_RECORD_FILE_SUFFIXES:
        record_format = "jsonl"
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        parsed = json.loads(stripped)
                    except json.JSONDecodeError:
                        return None
                    if not isinstance(parsed, dict):
                        return None
                    record_count += 1
                    if len(sample) < JSON_RECORD_SAMPLE_LIMIT:
                        sample.append(parsed)
        except OSError:
            return None
    elif suffix == ".json":
        record_format = "json"
        try:
            if path.stat().st_size > JSON_RECORD_MAX_ARRAY_BYTES:
                return None
            loaded = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(loaded, list) or not loaded or not all(isinstance(item, dict) for item in loaded):
            return None
        record_count = len(loaded)
        sample = loaded[:JSON_RECORD_SAMPLE_LIMIT]
    else:
        return None
    if record_count == 0:
        return None
    fields = sorted({key for record in sample for key in flatten_record(record)})
    return {
        "path": path,
        "relative_path": path.relative_to(data_dir).as_posix(),
        "format": record_format,
        "record_count": record_count,
        "fields": fields[:80],
    }


def classify_dataset_sources(data_dir: Path, files: list[Path]) -> dict[str, Any]:
    csv_text_path_tables: list[dict[str, Any]] = []
    csv_inline_text_tables: list[dict[str, Any]] = []
    csv_metadata_tables: list[dict[str, Any]] = []
    json_record_files: list[dict[str, Any]] = []
    container_paths: set[Path] = set()
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            table = detect_csv_text_path_table(data_dir, path)
            if table is not None:
                csv_text_path_tables.append(table)
                container_paths.add(path)
                continue
            inline_table = detect_csv_inline_text_table(data_dir, path)
            if inline_table is not None:
                csv_inline_text_tables.append(inline_table)
                container_paths.add(path)
                continue
            metadata_table = describe_csv_metadata_table(data_dir, path)
            if metadata_table is not None:
                csv_metadata_tables.append(metadata_table)
                container_paths.add(path)
        elif suffix in JSON_RECORD_FILE_SUFFIXES or suffix == ".json":
            record_file = detect_json_record_file(data_dir, path)
            if record_file is not None:
                json_record_files.append(record_file)
                container_paths.add(path)
    return {
        "csv_text_path_tables": csv_text_path_tables,
        "csv_inline_text_tables": csv_inline_text_tables,
        "csv_metadata_tables": csv_metadata_tables,
        "json_record_files": json_record_files,
        "document_files": [path for path in files if path not in container_paths],
    }


def cache_paths(cache_dir: Path) -> tuple[Path, Path]:
    return cache_dir / "dataset_texts.jsonl", cache_dir / "manifest.json"


def cache_is_fresh(
    data_dir: Path,
    cache_dir: Path,
    include_paths: list[Path] | None = None,
    partition_file_collections_by_top_level: bool = False,
) -> bool:
    jsonl_path, manifest_path = cache_paths(cache_dir)
    if not jsonl_path.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("version") != CACHE_VERSION:
        return False
    if Path(str(manifest.get("data_dir", ""))).resolve() != data_dir.resolve():
        return False
    normalized_includes = normalize_include_paths(data_dir) if include_paths is None else include_paths
    if manifest.get("included_paths", ["."]) != include_path_labels(data_dir, normalized_includes):
        return False
    if bool(manifest.get("partition_file_collections_by_top_level")) != partition_file_collections_by_top_level:
        return False
    current = file_inventory(data_dir, normalized_includes)
    if manifest.get("files") != current:
        return False
    if manifest.get("file_count", manifest.get("total_files")) != len(current):
        return False
    try:
        with jsonl_path.open("r", encoding="utf-8") as handle:
            line_count = sum(1 for line in handle if line.strip())
    except OSError:
        return False
    return line_count == manifest.get("record_count")


def read_text_file(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore"), {"ok": True, "method": "text", "error": None}
    except OSError as exc:
        return "", {"ok": False, "method": "text", "error": str(exc)}


def read_html_file(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return "", {"ok": False, "method": "html", "error": str(exc)}
    parser = HTMLTextExtractor()
    try:
        parser.feed(raw)
        text = html.unescape(parser.get_text())
        return text, {"ok": bool(text.strip()), "method": "html-parser", "error": None}
    except Exception as exc:
        text = html.unescape(raw)
        return text, {"ok": bool(text.strip()), "method": "html-fallback-raw", "error": str(exc)}


def extract_pdf_with_pdftotext(path: Path) -> tuple[str, str | None]:
    executable = shutil.which("pdftotext")
    if not executable:
        return "", "pdftotext not found"
    try:
        completed = subprocess.run(
            [executable, "-layout", "-enc", "UTF-8", str(path), "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=PDF_TIMEOUT_SECS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "", f"pdftotext timed out after {PDF_TIMEOUT_SECS}s"
    except OSError as exc:
        return "", str(exc)
    if completed.returncode != 0:
        return "", completed.stderr.strip() or f"pdftotext exited {completed.returncode}"
    return completed.stdout, None


def extract_pdf_with_pypdf(path: Path) -> tuple[str, str | None]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:
        return "", f"pypdf unavailable: {exc}"
    try:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages), None
    except Exception as exc:
        return "", f"pypdf failed: {exc}"


def extract_pdf_with_pdfminer(path: Path) -> tuple[str, str | None]:
    try:
        from pdfminer.high_level import extract_text  # type: ignore
    except Exception as exc:
        return "", f"pdfminer unavailable: {exc}"
    try:
        return extract_text(str(path)) or "", None
    except Exception as exc:
        return "", f"pdfminer failed: {exc}"


def read_pdf_file(path: Path) -> tuple[str, dict[str, Any]]:
    errors: list[str] = []
    for method, extractor in (
        ("pdftotext", extract_pdf_with_pdftotext),
        ("pypdf", extract_pdf_with_pypdf),
        ("pdfminer", extract_pdf_with_pdfminer),
    ):
        text, error = extractor(path)
        if text.strip():
            return text, {"ok": True, "method": method, "error": None}
        errors.append(error or f"{method} produced empty text")
    return "", {"ok": False, "method": "pdf", "error": " | ".join(errors)}


def extract_text(path: Path) -> tuple[str, dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf_file(path)
    if suffix in {".htm", ".html"}:
        return read_html_file(path)
    if suffix == ".txt":
        return read_text_file(path)
    text, meta = read_text_file(path)
    meta["method"] = f"unknown{suffix}-as-text"
    return text, meta


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def write_progress(message: str) -> None:
    print(message, flush=True)
    progress_path = os.environ.get("CLAUDECODE_TASK_PROGRESS_LOG")
    if progress_path:
        with Path(progress_path).open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
            handle.flush()


def row_value_case_insensitive(row: dict[str, Any], target: str) -> Any:
    target_lower = target.lower()
    for key, value in row.items():
        if str(key).lower() == target_lower:
            return value
    return None


def record_id_base(row: dict[str, Any]) -> str:
    for column in DOCUMENT_ID_COLUMN_CANDIDATES:
        value = row_value_case_insensitive(row, column)
        if value not in (None, ""):
            return str(value)
    id_columns = [
        column
        for column, value in row.items()
        if str(column).lower().endswith("_id") and value not in (None, "")
    ]
    if id_columns:
        id_columns.sort(key=lambda column: str(column).count("."))
        return str(row[id_columns[0]])
    return ""


def unique_document_id(base: str, collision_suffix: str, seen: dict[str, int]) -> str:
    normalized = base or collision_suffix
    count = seen.get(normalized, 0)
    seen[normalized] = count + 1
    return normalized if count == 0 else f"{normalized}::{collision_suffix}"


def file_document_collection(data_dir: Path, path: Path, partition_by_top_level: bool) -> str:
    if not partition_by_top_level:
        return "files"
    relative = path.relative_to(data_dir)
    top_level = relative.parts[0] if len(relative.parts) > 1 else "<root>"
    return f"files:{top_level}"


def write_file_document_cache(
    data_dir: Path,
    handle: Any,
    files: list[Path],
    seen_document_ids: dict[str, int],
    partition_by_top_level: bool = False,
) -> tuple[int, int]:
    failures = 0
    for index, path in enumerate(files, start=1):
        relative_path = path.relative_to(data_dir).as_posix()
        text, extraction = extract_text(path)
        if not extraction.get("ok"):
            failures += 1
        record = {
            "document_id": unique_document_id(relative_path, relative_path, seen_document_ids),
            "filename": path.name,
            "relative_path": relative_path,
            "extension": path.suffix.lower(),
            "source_path": str(path.resolve()),
            "text": text,
            "text_chars": len(text),
            "text_words": len(text.split()),
            "text_sha256": sha256_text(text),
            "extraction": extraction,
            "cache_record_type": "file_document",
            "collection": file_document_collection(data_dir, path, partition_by_top_level),
        }
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if index % 250 == 0 or index == len(files):
            write_progress(f"dataset-cache cached {index}/{len(files)} standalone files failures={failures}")
    return len(files), failures


def write_csv_text_path_cache(
    data_dir: Path,
    handle: Any,
    tables: list[dict[str, Any]],
    seen_document_ids: dict[str, int],
    referenced_paths: set[Path],
) -> tuple[int, int]:
    failures = 0
    records = 0
    for table in tables:
        csv_path = Path(table["csv_path"])
        text_column = table["text_path_column"]
        metadata_source = csv_path.relative_to(data_dir).as_posix()
        with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as csv_handle:
            reader = csv.DictReader(csv_handle)
            for row_number, raw_row in enumerate(reader, start=2):
                row = normalize_csv_row(raw_row)
                text_reference = row.get(text_column, "")
                text_path = resolve_dataset_reference(data_dir, csv_path, text_reference)
                if text_path is None:
                    failures += 1
                    text = ""
                    extraction = {
                        "ok": False,
                        "method": "csv_text_path",
                        "error": f"text path not found: {text_reference}",
                    }
                    text_relative_path = text_reference
                    source_path = ""
                    extension = ""
                else:
                    referenced_paths.add(text_path.resolve())
                    text_relative_path = text_path.relative_to(data_dir).as_posix()
                    source_path = str(text_path.resolve())
                    extension = text_path.suffix.lower()
                    text, extraction = extract_text(text_path)
                    if not extraction.get("ok"):
                        failures += 1
                metadata = {key: value for key, value in row.items() if key != text_column}
                base = record_id_base(row) or Path(text_relative_path).with_suffix("").as_posix()
                document_id = unique_document_id(base, f"{metadata_source}::row{row_number}", seen_document_ids)
                record: dict[str, Any] = {
                    "document_id": document_id,
                    "filename": row.get("filename") or row.get("file") or Path(text_relative_path).name,
                    "relative_path": text_relative_path,
                    "extension": extension,
                    "source_path": source_path,
                    "text_path": text_reference,
                    "text_relative_path": text_relative_path,
                    "metadata_source": metadata_source,
                    "metadata_row_number": row_number,
                    "metadata": metadata,
                    "text": text,
                    "text_chars": len(text),
                    "text_words": len(text.split()),
                    "extraction": extraction,
                    "cache_record_type": "csv_metadata_text_file",
                    "collection": f"csv:{metadata_source}",
                }
                for key, value in metadata.items():
                    if key not in record:
                        record[key] = value
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records += 1
                if records % 250 == 0:
                    write_progress(f"dataset-cache cached {records} CSV text-path rows")
    return records, failures


def write_csv_inline_text_cache(
    data_dir: Path,
    handle: Any,
    tables: list[dict[str, Any]],
    seen_document_ids: dict[str, int],
) -> tuple[int, int]:
    records = 0
    for table in tables:
        csv_path = Path(table["csv_path"])
        text_column = table["inline_text_column"]
        metadata_source = csv_path.relative_to(data_dir).as_posix()
        with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as csv_handle:
            reader = csv.DictReader(csv_handle)
            for row_number, raw_row in enumerate(reader, start=2):
                row = normalize_csv_row(raw_row)
                text = row.get(text_column, "")
                metadata = {key: value for key, value in row.items() if key != text_column}
                base = record_id_base(row) or f"{metadata_source}::row{row_number}"
                document_id = unique_document_id(base, f"{metadata_source}::row{row_number}", seen_document_ids)
                record: dict[str, Any] = {
                    "document_id": document_id,
                    "filename": csv_path.name,
                    "relative_path": metadata_source,
                    "extension": csv_path.suffix.lower(),
                    "source_path": str(csv_path.resolve()),
                    "text_column": text_column,
                    "metadata_source": metadata_source,
                    "metadata_row_number": row_number,
                    "metadata": metadata,
                    "text": text,
                    "text_chars": len(text),
                    "text_words": len(text.split()),
                    "extraction": {"ok": True, "method": "csv_inline_text", "error": None},
                    "cache_record_type": "csv_inline_text_row",
                    "collection": f"csv:{metadata_source}",
                }
                for key, value in metadata.items():
                    if key not in record:
                        record[key] = value
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records += 1
                if records % 250 == 0:
                    write_progress(f"dataset-cache cached {records} CSV inline-text rows")
    return records, 0


def iter_json_records(path: Path, record_format: str) -> Any:
    if record_format == "jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            record_number = 0
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                record_number += 1
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    yield record_number, None
                    continue
                yield record_number, parsed if isinstance(parsed, dict) else None
        return
    loaded = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    for record_number, item in enumerate(loaded if isinstance(loaded, list) else [], start=1):
        yield record_number, item if isinstance(item, dict) else None


def write_json_record_cache(
    data_dir: Path,
    handle: Any,
    record_files: list[dict[str, Any]],
    seen_document_ids: dict[str, int],
) -> tuple[int, int]:
    failures = 0
    records = 0
    for info in record_files:
        path = Path(info["path"])
        metadata_source = path.relative_to(data_dir).as_posix()
        record_format = info["format"]
        for record_number, record_object in iter_json_records(path, record_format):
            if record_object is None:
                failures += 1
                continue
            flat = flatten_record(record_object)
            text = record_text_from_flat(flat)
            base = record_id_base(flat) or f"{metadata_source}::record{record_number}"
            document_id = unique_document_id(base, f"{metadata_source}::record{record_number}", seen_document_ids)
            record: dict[str, Any] = {
                "document_id": document_id,
                "filename": path.name,
                "relative_path": metadata_source,
                "extension": path.suffix.lower(),
                "source_path": str(path.resolve()),
                "metadata_source": metadata_source,
                "metadata_row_number": record_number,
                "metadata": flat,
                "record": record_object,
                "text": text,
                "text_chars": len(text),
                "text_words": len(text.split()),
                "extraction": {"ok": True, "method": "json_record", "error": None},
                "cache_record_type": "json_record",
                "collection": f"{record_format}:{metadata_source}",
            }
            for key, value in flat.items():
                if key not in record:
                    record[key] = value
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records += 1
            if records % 250 == 0:
                write_progress(f"dataset-cache cached {records} JSON records")
    return records, failures


def table_summaries(tables: list[dict[str, Any]], *, kind: str) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for table in tables:
        summary = {
            "relative_path": table["relative_path"],
            "columns": table["columns"],
            "row_count": table["row_count"],
        }
        if kind == "text_path":
            summary["text_path_column"] = table["text_path_column"]
            summary["sample_valid_text_paths"] = table["sample_valid_text_paths"]
        elif kind == "inline_text":
            summary["inline_text_column"] = table["inline_text_column"]
            summary["avg_text_chars"] = table["avg_text_chars"]
        summaries.append(summary)
    return summaries


def json_record_summaries(record_files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": info["relative_path"],
            "format": info["format"],
            "record_count": info["record_count"],
            "fields": info["fields"],
        }
        for info in record_files
    ]


def collection_summaries(sources: dict[str, Any]) -> list[dict[str, Any]]:
    collections: list[dict[str, Any]] = []
    for table in sources["csv_text_path_tables"]:
        collections.append({
            "collection": f"csv:{table['relative_path']}",
            "kind": "csv_text_path_table",
            "cache_record_type": "csv_metadata_text_file",
            "relative_path": table["relative_path"],
            "columns": table["columns"],
            "text_path_column": table["text_path_column"],
            "row_count": table["row_count"],
        })
    for table in sources["csv_inline_text_tables"]:
        collections.append({
            "collection": f"csv:{table['relative_path']}",
            "kind": "csv_inline_text_table",
            "cache_record_type": "csv_inline_text_row",
            "relative_path": table["relative_path"],
            "columns": table["columns"],
            "inline_text_column": table["inline_text_column"],
            "avg_text_chars": table["avg_text_chars"],
            "row_count": table["row_count"],
        })
    for info in sources["json_record_files"]:
        collections.append({
            "collection": f"{info['format']}:{info['relative_path']}",
            "kind": "json_records",
            "cache_record_type": "json_record",
            "relative_path": info["relative_path"],
            "format": info["format"],
            "fields": info["fields"],
            "row_count": info["record_count"],
        })
    return collections


def choose_cache_mode(sources: dict[str, Any], file_records: int) -> str:
    has_text_path_tables = bool(sources["csv_text_path_tables"])
    has_record_collections = bool(sources["csv_inline_text_tables"] or sources["json_record_files"])
    if has_text_path_tables and not has_record_collections and file_records == 0:
        return "csv_metadata_text_files"
    if not has_text_path_tables and not has_record_collections:
        return "file_per_document"
    return "mixed_sources"


CACHE_RECORD_BASE_FIELDS = [
    "document_id",
    "collection",
    "cache_record_type",
    "filename",
    "relative_path",
    "extension",
    "source_path",
    "text",
    "text_chars",
    "text_words",
    "extraction",
]
CACHE_RECORD_EXTRA_FIELDS_BY_TYPE = {
    "csv_metadata_text_file": ["text_path", "text_relative_path", "metadata_source", "metadata_row_number", "metadata"],
    "csv_inline_text_row": ["text_column", "metadata_source", "metadata_row_number", "metadata"],
    "json_record": ["record", "metadata_source", "metadata_row_number", "metadata"],
    "file_document": [],
}


def cache_record_field_union() -> list[str]:
    fields = list(CACHE_RECORD_BASE_FIELDS)
    for extras in CACHE_RECORD_EXTRA_FIELDS_BY_TYPE.values():
        for field in extras:
            if field not in fields:
                fields.append(field)
    return fields


def build_cache(
    data_dir: Path,
    cache_dir: Path,
    include_paths: list[Path] | None = None,
    partition_file_collections_by_top_level: bool = False,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path, manifest_path = cache_paths(cache_dir)
    tmp_jsonl = jsonl_path.with_suffix(".jsonl.tmp")
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    normalized_includes = (
        normalize_include_paths(data_dir)
        if include_paths is None
        else normalize_include_paths(data_dir, [str(path) for path in include_paths])
    )
    files = dataset_files(data_dir, normalized_includes)
    inventory = file_inventory(data_dir, normalized_includes)
    sources = classify_dataset_sources(data_dir, files)
    extension_counts = Counter(item["extension"] or "<none>" for item in inventory)
    top_level_counts = Counter(
        path.relative_to(data_dir).parts[0]
        if len(path.relative_to(data_dir).parts) > 1
        else "<root>"
        for path in files
    )
    started_at = time.time()
    seen_document_ids: dict[str, int] = {}
    referenced_paths: set[Path] = set()
    with tmp_jsonl.open("w", encoding="utf-8") as handle:
        text_path_records, text_path_failures = write_csv_text_path_cache(
            data_dir,
            handle,
            sources["csv_text_path_tables"],
            seen_document_ids,
            referenced_paths,
        )
        inline_records, inline_failures = write_csv_inline_text_cache(
            data_dir,
            handle,
            sources["csv_inline_text_tables"],
            seen_document_ids,
        )
        json_records, json_failures = write_json_record_cache(
            data_dir,
            handle,
            sources["json_record_files"],
            seen_document_ids,
        )
        document_files = [
            path
            for path in sources["document_files"]
            if path.resolve() not in referenced_paths
        ]
        file_records, file_failures = write_file_document_cache(
            data_dir,
            handle,
            document_files,
            seen_document_ids,
            partition_file_collections_by_top_level,
        )
    record_count = text_path_records + inline_records + json_records + file_records
    failures = text_path_failures + inline_failures + json_failures + file_failures
    cache_mode = choose_cache_mode(sources, file_records)
    text_path_tables = table_summaries(sources["csv_text_path_tables"], kind="text_path")
    inline_text_tables = table_summaries(sources["csv_inline_text_tables"], kind="inline_text")
    metadata_only_tables = table_summaries(sources["csv_metadata_tables"], kind="metadata")
    json_record_files = json_record_summaries(sources["json_record_files"])
    collections = collection_summaries(sources)
    if file_records:
        file_collection_counts = Counter(
            file_document_collection(data_dir, path, partition_file_collections_by_top_level)
            for path in document_files
        )
        for collection_name, row_count in sorted(file_collection_counts.items()):
            collections.append({
                "collection": collection_name,
                "kind": "document_files",
                "cache_record_type": "file_document",
                "row_count": row_count,
            })
    manifest = {
        "version": CACHE_VERSION,
        "created_at": time.time(),
        "elapsed_seconds": round(time.time() - started_at, 3),
        "data_dir": str(data_dir.resolve()),
        "included_paths": include_path_labels(data_dir, normalized_includes),
        "partition_file_collections_by_top_level": partition_file_collections_by_top_level,
        "jsonl_path": str(jsonl_path.resolve()),
        "cache_mode": cache_mode,
        "record_count": record_count,
        "file_count": len(inventory),
        "total_files": len(inventory),
        "by_extension": dict(sorted(extension_counts.items())),
        "failures": failures,
        "extraction_failures": failures,
        "record_counts_by_group": {
            "csv_text_path_rows": text_path_records,
            "csv_inline_text_rows": inline_records,
            "json_records": json_records,
            "document_files": file_records,
        },
        "collections": collections,
        "metadata_tables": text_path_tables,
        "csv_tables": text_path_tables,
        "inline_text_tables": inline_text_tables,
        "json_record_files": json_record_files,
        "metadata_only_tables": metadata_only_tables,
        "cache_record_fields": cache_record_field_union(),
        "cache_record_base_fields": CACHE_RECORD_BASE_FIELDS,
        "cache_record_fields_by_type": CACHE_RECORD_EXTRA_FIELDS_BY_TYPE,
        "dataset_profile": {
            "data_dir": str(data_dir.resolve()),
            "cache_mode": cache_mode,
            "record_count": record_count,
            "source_file_count": len(files),
            "total_bytes": sum(path.stat().st_size for path in files),
            "top_level_entries": dict(sorted(top_level_counts.items())),
            "extension_counts": dict(sorted(extension_counts.items())),
            "sample_files": [path.relative_to(data_dir).as_posix() for path in files[:30]],
            "csv_tables": text_path_tables[:20],
            "inline_text_tables": inline_text_tables[:20],
            "json_record_files": json_record_files[:20],
            "metadata_only_tables": metadata_only_tables[:20],
            "document_file_count": len(sources["document_files"]),
        },
        "files": inventory,
    }
    tmp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_jsonl.replace(jsonl_path)
    tmp_manifest.replace(manifest_path)
    write_progress(
        f"dataset-cache ready mode={cache_mode} records={record_count} "
        f"files={len(inventory)} failures={failures} jsonl={jsonl_path}"
    )


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    if not data_dir.exists():
        raise SystemExit(f"data dir not found: {data_dir}")
    include_paths = normalize_include_paths(data_dir, args.include_path)
    if args.ensure and not args.rebuild and cache_is_fresh(
        data_dir,
        cache_dir,
        include_paths,
        args.partition_file_collections_by_top_level,
    ):
        _, manifest_path = cache_paths(cache_dir)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        write_progress(
            f"dataset-cache fresh mode={manifest.get('cache_mode')} "
            f"records={manifest.get('record_count')} files={manifest.get('file_count')} "
            f"by_extension={manifest.get('by_extension')}"
        )
        return 0
    build_cache(
        data_dir,
        cache_dir,
        include_paths,
        args.partition_file_collections_by_top_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def model_call_helper_source() -> str:
    return r'''#!/usr/bin/env python3
"""OpenAI-compatible helper for tool-side batched semantic processing.

Use from task scripts:
    from model_call_helper import batch_call_model
    results = batch_call_model(prompts, max_tokens=256)
    text = results[0]["text"]

The helper always sends temperature=0 by default and disables thinking with:
    chat_template_kwargs={"enable_thinking": false}
"""

from __future__ import annotations

import concurrent.futures
from contextlib import contextmanager
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT_DIR = Path(
    os.environ.get("CLAUDECODE_PROJECT_ROOT")
    or os.environ.get("CONTRACT_EXHIBIT_ROOT")
    or Path(__file__).resolve().parents[2]
).resolve()
MAX_MODEL_CONCURRENCY = 10
DEFAULT_MODEL_OUTPUT_TOKENS = 2048
DEFAULT_MODEL_SLOT_TTL_SECS = 3600


def parse_dotenv(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        env[key.strip()] = value
    return env


def merged_env() -> dict[str, str]:
    env = parse_dotenv(ROOT_DIR / ".env")
    env.update(os.environ)
    return env


def to_int(value: Any) -> int:
    try:
        return max(int(float(str(value))), 0)
    except (TypeError, ValueError):
        return 0


def normalize_million_token_cost(value: Any) -> float:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return 0.0
    if parsed < 0:
        return 0.0
    return parsed * 1_000_000 if 0 < parsed < 0.01 else parsed


def configured_max_concurrency(env: dict[str, str]) -> int:
    configured = to_int(env.get("CLAUDECODE_TOOL_MODEL_MAX_CONCURRENCY"))
    if configured < 1:
        configured = MAX_MODEL_CONCURRENCY
    return min(configured, MAX_MODEL_CONCURRENCY)


def configured_max_tokens(env: dict[str, str]) -> int:
    configured = to_int(env.get("CLAUDECODE_TOOL_MODEL_MAX_TOKENS"))
    if configured < 1:
        configured = DEFAULT_MODEL_OUTPUT_TOKENS
    return min(configured, DEFAULT_MODEL_OUTPUT_TOKENS)


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def slot_is_stale(path: Path, ttl_secs: int = DEFAULT_MODEL_SLOT_TTL_SECS) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    pid = to_int(data.get("pid"))
    started_at = float(data.get("started_at", 0) or 0)
    if started_at and time.time() - started_at > ttl_secs:
        return True
    return bool(pid and not process_is_alive(pid))


def acquire_model_slot(env: dict[str, str]) -> Path | None:
    semaphore_dir = env.get("CLAUDECODE_TOOL_MODEL_SEMAPHORE_DIR")
    if not semaphore_dir:
        return None
    root = Path(semaphore_dir)
    root.mkdir(parents=True, exist_ok=True)
    limit = configured_max_concurrency(env)
    deadline = time.time() + 600
    while True:
        for index in range(limit):
            slot_path = root / f"slot-{index}.json"
            if slot_path.exists() and slot_is_stale(slot_path):
                try:
                    slot_path.unlink()
                except FileNotFoundError:
                    pass
            try:
                fd = os.open(str(slot_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"pid": os.getpid(), "started_at": time.time()}, handle)
            return slot_path
        if time.time() > deadline:
            raise TimeoutError(f"timed out waiting for model concurrency slot, limit={limit}")
        time.sleep(0.1)


def release_model_slot(slot_path: Path | None) -> None:
    if slot_path is None:
        return
    try:
        slot_path.unlink()
    except FileNotFoundError:
        pass


@contextmanager
def model_slot(env: dict[str, str]):
    slot_path = acquire_model_slot(env)
    try:
        yield
    finally:
        release_model_slot(slot_path)


def usage_from_response(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
    return {
        "input": to_int(usage.get("input") or usage.get("prompt_tokens") or usage.get("promptTokens") or usage.get("input_tokens")),
        "output": to_int(usage.get("output") or usage.get("completion_tokens") or usage.get("completionTokens") or usage.get("output_tokens")),
        "cacheRead": to_int(usage.get("cacheRead") or usage.get("cache_read") or usage.get("cache_read_input_tokens")),
        "cacheWrite": to_int(usage.get("cacheWrite") or usage.get("cache_write") or usage.get("cache_creation_input_tokens")),
    }


def calculate_cost_usd(usage: dict[str, int], env: dict[str, str]) -> float:
    input_per_million = normalize_million_token_cost(env.get("LLM_MODEL_USD_PER_INPUT_TOKEN"))
    output_per_million = normalize_million_token_cost(env.get("LLM_MODEL_USD_PER_OUTPUT_TOKEN"))
    cache_read_per_million = normalize_million_token_cost(env.get("LLM_MODEL_USD_PER_CACHE_READ_TOKEN")) or input_per_million
    cache_write_per_million = normalize_million_token_cost(env.get("LLM_MODEL_USD_PER_CACHE_WRITE_TOKEN")) or output_per_million
    return round(
        (
            usage.get("input", 0) * input_per_million
            + usage.get("output", 0) * output_per_million
            + usage.get("cacheRead", 0) * cache_read_per_million
            + usage.get("cacheWrite", 0) * cache_write_per_million
        )
        / 1_000_000,
        10,
    )


def add_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {
        "input": left.get("input", 0) + right.get("input", 0),
        "output": left.get("output", 0) + right.get("output", 0),
        "cacheRead": left.get("cacheRead", 0) + right.get("cacheRead", 0),
        "cacheWrite": left.get("cacheWrite", 0) + right.get("cacheWrite", 0),
    }


def append_usage(response: dict[str, Any], env: dict[str, str], endpoint: str, started_at: float, ended_at: float) -> dict[str, Any]:
    usage = usage_from_response(response)
    cost_usd = calculate_cost_usd(usage, env)
    record = {
        "endpoint": endpoint,
        "model": env.get("LLM_MODEL_NAME"),
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_secs": round(ended_at - started_at, 6),
        "thread_id": threading.get_ident(),
        "usage": usage,
        "cost_usd": cost_usd,
    }
    usage_path = env.get("CLAUDECODE_TOOL_MODEL_USAGE_PATH")
    if usage_path:
        path = Path(usage_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    return record


def write_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
    progress_path = os.environ.get("CLAUDECODE_TASK_PROGRESS_LOG")
    if progress_path:
        with Path(progress_path).open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
            handle.flush()


def post_json(url: str, payload: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    api_key = env.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"model API HTTP {exc.code}: {body[:1000]}") from exc


def _call_model(
    prompt: str | None = None,
    *,
    messages: list[dict[str, str]] | None = None,
    temperature: float = 0,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    env = merged_env()
    base_url = env.get("LLM_API_BASE", "").rstrip("/")
    model = env.get("LLM_MODEL_NAME")
    if not base_url or not model:
        raise RuntimeError("LLM_API_BASE or LLM_MODEL_NAME is missing")
    if messages is None:
        messages = [{"role": "user", "content": prompt or ""}]
    effective_max_tokens = configured_max_tokens(env)
    if max_tokens is not None:
        requested = to_int(max_tokens)
        if requested >= 1:
            effective_max_tokens = min(effective_max_tokens, requested)
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": effective_max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    endpoint = f"{base_url}/chat/completions"
    started_at = time.time()
    with model_slot(env):
        response = post_json(endpoint, payload, env)
    ended_at = time.time()
    usage_record = append_usage(response, env, endpoint, started_at, ended_at)
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    text = message.get("content") if isinstance(message, dict) else choice.get("text", "")
    return {"text": text or "", "usage": usage_record["usage"], "cost_usd": usage_record["cost_usd"], "raw": response}


def batch_call_model(
    prompts: list[str],
    *,
    temperature: float = 0,
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    if not prompts:
        return []
    env = merged_env()
    workers = max(1, min(configured_max_concurrency(env), len(prompts)))
    results: list[dict[str, Any] | None] = [None] * len(prompts)
    total_usage = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
    total_cost = 0.0
    completed = 0
    write_progress(f"tool-model batch start prompts={len(prompts)} workers={workers}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(_call_model, prompt, temperature=temperature, max_tokens=max_tokens): index
            for index, prompt in enumerate(prompts)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            result = future.result()
            results[index] = result
            completed += 1
            total_usage = add_usage(total_usage, result.get("usage", {}))
            total_cost += float(result.get("cost_usd") or 0)
            write_progress(
                "tool-model completed "
                f"{completed}/{len(prompts)} input={total_usage['input']} "
                f"output={total_usage['output']} cost_usd={round(total_cost, 6)}"
            )
    return [item or {"text": "", "usage": {}, "raw": {}} for item in results]


def main() -> int:
    prompt = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else sys.stdin.read().strip()
    result = batch_call_model([prompt])[0]
    print(result["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def agent_anthropic_base_url(agent_env: dict[str, str]) -> str | None:
    base = agent_env.get("AGENT_API_BASE") or agent_env.get("ANTHROPIC_BASE_URL") or agent_env.get("LLM_API_BASE")
    if not env_value_is_set(base):
        return None
    normalized = str(base).rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3].rstrip("/")
    return normalized or None


def agent_api_key(agent_env: dict[str, str]) -> str | None:
    for key in ("ANTHROPIC_API_KEY", "AGENT_API_KEY"):
        value = agent_env.get(key)
        if env_value_is_set(value):
            return str(value)
    return None


def agent_auth_token(agent_env: dict[str, str]) -> str | None:
    for key in ("ANTHROPIC_AUTH_TOKEN", "AGENT_AUTH_TOKEN"):
        value = agent_env.get(key)
        if env_value_is_set(value):
            return str(value)
    return None


def build_claude_settings(
    agent_env: dict[str, str],
    *,
    persist_sessions: bool = False,
) -> dict[str, Any]:
    settings_env: dict[str, str] = {
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    }
    base_url = agent_anthropic_base_url(agent_env)
    if base_url:
        settings_env["ANTHROPIC_BASE_URL"] = base_url
    return {
        "env": settings_env,
        "cleanupPeriodDays": RESUMED_SESSION_RETENTION_DAYS if persist_sessions else 0,
        "autoMemoryEnabled": False,
        "skipDangerousModePermissionPrompt": True,
    }


def ensure_runtime_files(tool_env: dict[str, str], agent_env: dict[str, str], args: argparse.Namespace) -> dict[str, Path]:
    run_dir = args.run_dir.resolve()
    paths = {
        "run_dir": run_dir,
        "state_dir": run_dir / "state",
        "home_dir": run_dir / "home",
        "workspace_dir": run_dir / "workspace",
        "workspace_tmp_dir": run_dir / "workspace" / "tmp",
        "workspace_intermediate_dir": run_dir / "workspace" / "intermediate",
        "dataset_cache_dir": run_dir / "workspace" / "dataset_cache",
        "raw_dir": run_dir / "raw",
        "output_dir": run_dir / "output",
        "results_dir": args.results_root.resolve() / args.dataset,
        "tool_model_usage_dir": run_dir / "tool_model_usage",
        "tool_model_semaphore_dir": run_dir / "tool_model_semaphore",
        "logs_dir": run_dir / "logs",
        "results_jsonl": run_dir / "results.jsonl",
    }
    for path in paths.values():
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)

    helper_path = paths["workspace_dir"] / "dataset_text_cache_helper.py"
    helper_path.write_text(dataset_cache_helper_source(), encoding="utf-8")
    helper_path.chmod(0o755)
    model_helper_path = paths["workspace_dir"] / "model_call_helper.py"
    model_helper_path.write_text(model_call_helper_source(), encoding="utf-8")
    model_helper_path.chmod(0o755)

    settings = build_claude_settings(
        agent_env,
        persist_sessions=args.retry_session_mode == "resume",
    )
    settings_path = paths["state_dir"] / "claude_settings.json"
    write_json(settings_path, settings)
    mcp_path = paths["state_dir"] / "empty_mcp.json"
    write_json(mcp_path, {"mcpServers": {}})

    paths["dataset_cache_helper"] = helper_path
    paths["model_call_helper"] = model_helper_path
    paths["dataset_cache_jsonl"] = paths["dataset_cache_dir"] / "dataset_texts.jsonl"
    paths["dataset_cache_manifest"] = paths["dataset_cache_dir"] / "manifest.json"
    paths["settings_path"] = settings_path
    paths["mcp_path"] = mcp_path
    return paths


def run_capture_process(
    *,
    command: list[str],
    cwd: Path,
    env: dict[str, str] | None,
    timeout: int,
    label: str,
    secrets: list[str],
) -> dict[str, Any]:
    popen_kwargs: dict[str, Any] = {}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **popen_kwargs,
    )
    register_child_process(process, label)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        terminate_process_group(process, label, reason=f"timeout after {timeout}s")
        stdout, stderr = process.communicate()
    except KeyboardInterrupt:
        terminate_process_group(process, label, reason="keyboard interrupt")
        raise
    finally:
        unregister_child_process(process)
    return {
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "stdout": redact_text(stdout or "", secrets),
        "stderr": redact_text(stderr or "", secrets),
    }


def ensure_dataset_text_cache(paths: dict[str, Path], args: argparse.Namespace, env: dict[str, str], secrets: list[str]) -> None:
    command = [
        sys.executable,
        str(paths["dataset_cache_helper"]),
        "--ensure",
        "--data-dir",
        str(args.data_dir),
        "--cache-dir",
        str(paths["dataset_cache_dir"]),
    ]
    for include_path in getattr(args, "dataset_include_paths", None) or []:
        command.extend(["--include-path", include_path.relative_to(args.data_dir).as_posix()])
    if getattr(args, "task_mode", DEFAULT_TASK_MODE) in {
        "operator-implementation",
        "optimized-plan-execution",
    }:
        command.append("--partition-file-collections-by-top-level")
    if args.rebuild_dataset_cache:
        command.append("--rebuild")
    child_env = os.environ.copy()
    child_env["CLAUDECODE_PROJECT_ROOT"] = str(ROOT_DIR)
    child_env["CONTRACT_EXHIBIT_ROOT"] = str(ROOT_DIR)
    child_env["CLAUDECODE_TASK_PROGRESS_LOG"] = str(paths["logs_dir"] / "dataset_cache.progress.log")
    print("Ensuring structured multi-format dataset cache...", flush=True)
    completed = run_capture_process(
        command=command,
        cwd=ROOT_DIR,
        env=child_env,
        timeout=args.dataset_cache_timeout,
        label="dataset-cache",
        secrets=secrets,
    )
    if completed["stdout"].strip():
        print(completed["stdout"].strip(), flush=True)
    if completed["timed_out"] or completed["exit_code"] != 0:
        if completed["stderr"].strip():
            print(completed["stderr"].strip(), file=sys.stderr, flush=True)
        if completed["timed_out"]:
            raise SystemExit(f"dataset cache build timed out after {args.dataset_cache_timeout}s")
        raise SystemExit(f"dataset cache build failed, exit_code={completed['exit_code']}")


def dataset_context_for_prompt(paths: dict[str, Path]) -> str:
    manifest_path = paths["dataset_cache_manifest"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {}
    if not isinstance(manifest, dict) or not manifest:
        return json.dumps(
            {
                "cache_manifest_available": False,
                "cache_jsonl": str(paths["dataset_cache_jsonl"]),
                "cache_manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    profile = manifest.get("dataset_profile") if isinstance(manifest.get("dataset_profile"), dict) else {}
    context = {
        "cache_manifest_available": True,
        "cache_jsonl": str(paths["dataset_cache_jsonl"]),
        "cache_manifest": str(manifest_path),
        "data_dir": manifest.get("data_dir"),
        "included_paths": manifest.get("included_paths", ["."]),
        "cache_mode": manifest.get("cache_mode"),
        "record_count": manifest.get("record_count"),
        "source_file_count": manifest.get("file_count"),
        "failures": manifest.get("failures"),
        "record_counts_by_group": manifest.get("record_counts_by_group", {}),
        "cache_record_fields": manifest.get("cache_record_fields", []),
        "collections": manifest.get("collections", []),
        "metadata_tables": manifest.get("metadata_tables", []),
        "inline_text_tables": manifest.get("inline_text_tables", []),
        "json_record_files": manifest.get("json_record_files", []),
        "metadata_only_tables": manifest.get("metadata_only_tables", []),
        "source_structure": {
            "top_level_entries": profile.get("top_level_entries", {}),
            "extension_counts": profile.get("extension_counts", {}),
            "sample_files": profile.get("sample_files", []),
        },
        "usage_notes": [
            "Every cache row has collection and cache_record_type; select only the source collections needed by the query.",
            "csv_metadata_text_file rows join one metadata CSV row to its referenced text file.",
            "csv_inline_text_row rows carry source CSV text inline and copy other columns top-level and under metadata.",
            "json_record rows preserve the original nested object in record and expose flattened fields top-level and under metadata.",
            "metadata_only_tables are not duplicated into the text cache; task scripts should read those source CSVs directly for exact joins and predicates.",
            "For mixed_sources datasets, filter streamed JSONL rows by collection before semantic model calls.",
            "For large caches, stream JSONL rather than loading every full-text row into memory.",
        ],
    }
    return json.dumps(context, ensure_ascii=False, indent=2)


def normalize_task_for_mode(task: dict[str, Any], task_mode: str) -> dict[str, Any]:
    if task_mode not in {
        "operator-implementation",
        "plan-optimization",
        "optimized-plan-execution",
    }:
        return task
    task_id = task.get("task_id", "<unknown>")
    query = task.get("query")
    inputs = task.get("inputs")
    plans = task.get("reference_semantic_plan")
    if not isinstance(query, str) or not query.strip():
        raise SystemExit(f"{task_mode} task {task_id} has no query")
    if not isinstance(inputs, list) or not inputs or not all(isinstance(item, dict) for item in inputs):
        raise SystemExit(f"{task_mode} task {task_id} has invalid inputs")
    if not isinstance(plans, list) or not plans or not all(isinstance(plan, dict) for plan in plans):
        raise SystemExit(f"{task_mode} task {task_id} has invalid reference_semantic_plan")
    if task_mode == "plan-optimization":
        return dict(task)
    answer_schema = task.get("answer_schema")
    if not isinstance(answer_schema, dict):
        gold = task.get("gold")
        answer_schema = gold.get("answer_schema") if isinstance(gold, dict) else None
    if not isinstance(answer_schema, dict):
        raise SystemExit(f"{task_mode} task {task_id} has no answer_schema")
    normalized = dict(task)
    normalized["answer_schema"] = answer_schema
    return normalized


def normalize_source_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def operator_dataset_dir(data_dir: Path, dataset: str) -> Path:
    relative = OPERATOR_IMPLEMENTATION_DATASET_DIRS.get(dataset)
    if relative is None:
        raise SystemExit(f"unsupported operator implementation dataset: {dataset}")
    candidate = (data_dir / relative).resolve()
    if data_dir.name == relative and not candidate.exists():
        candidate = data_dir.resolve()
    try:
        candidate.relative_to(data_dir.resolve())
    except ValueError as exc:
        raise SystemExit(f"operator dataset directory is outside --data-dir: {candidate}") from exc
    if not candidate.is_dir():
        raise SystemExit(f"operator dataset directory not found for {dataset}: {candidate}")
    return candidate


def resolve_operator_input(data_dir: Path, item: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    dataset = item.get("dataset")
    source_id = item.get("source_id")
    if not isinstance(dataset, str) or not dataset.strip():
        raise SystemExit(f"operator input has invalid dataset: {item}")
    if not isinstance(source_id, str) or not source_id.strip():
        raise SystemExit(f"operator input has invalid source_id: {item}")
    data_root = data_dir.resolve()
    dataset_dir = operator_dataset_dir(data_root, dataset)
    source_id = source_id.strip()
    selection_kind = "file"

    source_parts = Path(source_id).parts
    if source_parts and source_parts[0] == "data":
        resolved = data_root.joinpath(*source_parts[1:]).resolve()
        selection_kind = "directory" if resolved.is_dir() else "file"
    elif source_id.startswith("operator_implement/"):
        resolved = (data_root / source_id).resolve()
    elif source_id.lower().endswith("_corpus"):
        resolved = dataset_dir
        selection_kind = "directory"
    elif any(marker in source_id for marker in ("*", "?", "[")):
        wildcard_index = min(
            index for marker in ("*", "?", "[") if (index := source_id.find(marker)) >= 0
        )
        prefix = source_id[:wildcard_index].rstrip("/")
        resolved = (dataset_dir / prefix).resolve()
        selection_kind = "glob_parent"
    else:
        candidates = [dataset_dir / source_id]
        if source_id.endswith("_csv"):
            candidates.append(dataset_dir / f"{source_id[:-4]}.csv")
        if source_id.endswith("_jsonl"):
            candidates.append(dataset_dir / f"{source_id[:-6]}.jsonl")
        if source_id.endswith("_txt"):
            candidates.extend(
                [
                    dataset_dir / source_id[:-4],
                    dataset_dir / f"{source_id[:-4]}.txt",
                ]
            )
        resolved_candidate = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
        if resolved_candidate is None:
            normalized_source = normalize_source_name(source_id)
            matches = [
                path.resolve()
                for path in dataset_dir.rglob("*")
                if path.is_file() and normalize_source_name(path.name) == normalized_source
            ]
            if len(matches) != 1:
                raise SystemExit(
                    f"could not uniquely resolve operator input {source_id!r} under {dataset_dir}; "
                    f"matches={len(matches)}"
                )
            resolved = matches[0]
        else:
            resolved = resolved_candidate

    try:
        relative_path = resolved.relative_to(data_root)
    except ValueError as exc:
        raise SystemExit(f"operator input resolves outside --data-dir: {source_id} -> {resolved}") from exc
    if not resolved.exists():
        raise SystemExit(f"operator input source not found: {source_id} -> {resolved}")
    if resolved.is_dir() and selection_kind == "file":
        selection_kind = "directory"
    descriptor = dict(item)
    descriptor.update(
        {
            "dataset_directory": str(dataset_dir),
            "resolved_path": str(resolved),
            "data_relative_path": relative_path.as_posix(),
            "selection_kind": selection_kind,
            "requested_selector": source_id,
        }
    )
    return descriptor, resolved


def collapse_data_paths(paths: list[Path]) -> list[Path]:
    collapsed: list[Path] = []
    for candidate in sorted(set(path.resolve() for path in paths), key=lambda path: (len(path.parts), str(path))):
        if any(candidate == parent or parent in candidate.parents for parent in collapsed):
            continue
        collapsed.append(candidate)
    return collapsed


def resolve_operator_task_inputs(tasks: list[dict[str, Any]], data_dir: Path) -> list[Path]:
    selected_paths: list[Path] = []
    for task in tasks:
        resolved_inputs: list[dict[str, Any]] = []
        for item in task.get("inputs", []):
            descriptor, resolved = resolve_operator_input(data_dir, item)
            resolved_inputs.append(descriptor)
            selected_paths.append(resolved)
        task["_resolved_inputs"] = resolved_inputs
    return collapse_data_paths(selected_paths)


def operator_task_contract_for_prompt(task: dict[str, Any]) -> str:
    contract = {
        "inputs": task.get("inputs", []),
        "resolved_inputs": task.get("_resolved_inputs", []),
        "reference_semantic_plan": task.get("reference_semantic_plan", []),
    }
    return json.dumps(contract, ensure_ascii=False, indent=2)


def optimized_plan_execution_contract_for_prompt(task: dict[str, Any]) -> str:
    contract = {
        "inputs": task.get("inputs", []),
        "resolved_inputs": task.get("_resolved_inputs", []),
        "optimization_types": task.get("_optimization_types", []),
        "optimized_semantic_plan": task.get("optimized_semantic_plan", []),
    }
    return json.dumps(contract, ensure_ascii=False, indent=2)


def plan_optimization_response_contract(task_id: str) -> str:
    return "\n".join(
        [
            "Return exactly one JSON object and no markdown:",
            "{",
            f'  "task_id": "{task_id}",',
            '  "optimization_analysis": "concise explanation of the optimization",',
            '  "optimized_semantic_plan": [',
            '    {"type": "operator_dag", "program": "optimized plan"}',
            "  ]",
            "}",
        ]
    )


def build_plan_optimization_prompt(
    task: dict[str, Any],
    attempt: int,
    max_attempts: int,
    previous_attempts: list[dict[str, Any]],
) -> str:
    task_id = task["task_id"]
    sections = [
        "Optimize the following executable semantic query plan.",
        "The plan needs optimization to reduce the language-model processing cost of semantic operators. Preserve correctness; do not reduce cost in a way that changes the query semantics or makes the plan incorrect.",
        "",
        f"Task ID: {task_id}",
        f"Query: {task.get('query', '')}",
        "Inputs:",
        json.dumps(task.get("inputs", []), ensure_ascii=False, indent=2),
        "Current reference semantic plan:",
        json.dumps(task.get("reference_semantic_plan", []), ensure_ascii=False, indent=2),
    ]
    if attempt > 1 and previous_attempts:
        latest = previous_attempts[-1]
        sections.extend(
            [
                "",
                f"Retry attempt: {attempt}/{max_attempts}",
                "The previous response was invalid:",
                json.dumps(
                    {
                        "failure_reason": latest.get("failure_reason"),
                        "validation_warnings": latest.get("validation_warnings", []),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            ]
        )
    sections.extend(["", plan_optimization_response_contract(task_id)])
    return "\n".join(sections)


def build_resumed_plan_optimization_retry_prompt(
    task: dict[str, Any],
    attempt: int,
    max_attempts: int,
    previous_attempts: list[dict[str, Any]],
) -> str:
    if attempt <= 1 or not previous_attempts:
        raise ValueError("resumed plan optimization retry requires a previous attempt")
    latest = previous_attempts[-1]
    return "\n".join(
        [
            f"Retry attempt: {attempt}/{max_attempts}.",
            "The previous plan-optimization response was invalid:",
            json.dumps(
                {
                    "failure_reason": latest.get("failure_reason"),
                    "validation_warnings": latest.get("validation_warnings", []),
                },
                ensure_ascii=False,
                indent=2,
            ),
            "Return a corrected response. The original query, inputs, plan, cost objective, and correctness requirement remain unchanged.",
            "",
            plan_optimization_response_contract(task["task_id"]),
        ]
    )


def load_tasks(
    input_path: Path,
    task_id: str | None,
    limit: int | None,
    task_mode: str = DEFAULT_TASK_MODE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    tasks = raw.get("tasks") if isinstance(raw, dict) else raw
    if not isinstance(tasks, list):
        raise SystemExit("evaluation input must be a list or contain a tasks list")
    normalized = [
        normalize_task_for_mode(task, task_mode)
        for task in tasks
        if isinstance(task, dict) and isinstance(task.get("task_id"), str)
    ]
    selected = normalized
    if task_id:
        selected = [task for task in selected if task.get("task_id") == task_id]
        if not selected:
            raise SystemExit(f"task not found: {task_id}")
    if limit is not None:
        selected = selected[: max(limit, 0)]
    return normalized, selected


def normalize_task_ids(values: list[str] | None) -> set[str]:
    task_ids: set[str] = set()
    for value in values or []:
        for part in str(value).split(","):
            task_id = part.strip()
            if task_id:
                task_ids.add(task_id)
    return task_ids


def task_difficulty(task: dict[str, Any]) -> str:
    value = str(task.get("operator_complexity") or "").strip().lower()
    aliases = {"l1": "easy", "l2": "medium", "l3": "hard"}
    difficulty = aliases.get(value, value)
    if difficulty not in {"easy", "medium", "hard"}:
        raise SystemExit(f"task {task.get('task_id')} has invalid operator_complexity: {value!r}")
    return difficulty


def select_testbed_tasks(tasks: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    requested = normalize_task_ids(args.task_id)
    if args.task:
        requested.add(args.task)
    known = {str(task.get("task_id")) for task in tasks}
    unknown = sorted(requested - known)
    if unknown:
        raise SystemExit("task not found: " + ", ".join(unknown))
    start = args.start or 1
    end = args.end or len(tasks)
    if start < 1 or end < start:
        raise SystemExit("--start and --end must define a valid 1-based inclusive range")
    selected = [
        task
        for position, task in enumerate(tasks, start=1)
        if start <= position <= end and (not requested or task.get("task_id") in requested)
    ]
    if args.limit is not None:
        selected = selected[: max(args.limit, 0)]
    return selected


def compact_result_path(results_dir: Path, task: dict[str, Any]) -> Path:
    return results_dir / task_difficulty(task) / "claude_code" / f"{safe_filename(task['task_id'])}.json"


def read_compact_results(results_dir: Path, tasks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    for task in tasks:
        path = compact_result_path(results_dir, task)
        payload = read_json_object(path)
        if payload is None or payload.get("task_id") != task.get("task_id"):
            continue
        total_tokens = to_nonnegative_int(payload.get("total_tokens"))
        existing[task["task_id"]] = {
            "task_id": task["task_id"],
            "task_mode": DEFAULT_TASK_MODE,
            "ok": payload.get("answer") is not None,
            "answer": payload.get("answer"),
            "elapsed_secs": payload.get("elapsed_seconds"),
            "cost_usd": payload.get("cost_usd"),
            "total_tokens": total_tokens,
            "usage": {"input": total_tokens, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        }
    return existing


def read_existing_results(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return results
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        task_id = entry.get("task_id") if isinstance(entry, dict) else None
        if isinstance(task_id, str):
            results[task_id] = entry
    return results


def stored_usage(value: Any) -> dict[str, int]:
    usage = value if isinstance(value, dict) else {}
    return {
        "input": to_nonnegative_int(usage.get("input")),
        "output": to_nonnegative_int(usage.get("output")),
        "cacheRead": to_nonnegative_int(usage.get("cacheRead")),
        "cacheWrite": to_nonnegative_int(usage.get("cacheWrite")),
    }


def load_optimized_plan_selection(path: Path) -> dict[str, list[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"optimized plan selection could not be read: {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise SystemExit("optimized plan selection must be an object containing a tasks list")

    selected: dict[str, list[str]] = {}
    allowed_types = set(OPTIMIZATION_TYPES)
    for index, item in enumerate(payload["tasks"]):
        if not isinstance(item, dict):
            raise SystemExit(f"optimized plan selection tasks[{index}] must be an object")
        task_id = item.get("task_id")
        optimization_types = item.get("optimization_types")
        if not isinstance(task_id, str) or not task_id.strip():
            raise SystemExit(f"optimized plan selection tasks[{index}] has invalid task_id")
        if task_id in selected:
            raise SystemExit(f"optimized plan selection contains duplicate task_id: {task_id}")
        if (
            not isinstance(optimization_types, list)
            or not optimization_types
            or not all(isinstance(value, str) and value in allowed_types for value in optimization_types)
        ):
            raise SystemExit(
                f"optimized plan selection task {task_id} must contain one or more recognized optimization_types"
            )
        selected[task_id] = list(dict.fromkeys(optimization_types))

    declared_count = payload.get("task_count")
    if not isinstance(declared_count, int) or declared_count != len(selected):
        raise SystemExit(
            f"optimized plan selection task_count mismatch: declared={declared_count!r} actual={len(selected)}"
        )
    return selected


def read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def summarize_plan_optimization_phase(
    source_run_dir: Path,
    task_id: str,
    source_entry: dict[str, Any],
) -> dict[str, Any]:
    attempts = source_entry.get("attempts") if isinstance(source_entry.get("attempts"), list) else []
    agent_usage = stored_usage({})
    tool_model_usage = stored_usage({})
    agent_cost_usd = 0.0
    tool_model_cost_usd = 0.0
    elapsed_secs = 0.0
    loaded_attempt_usage = False
    attempt_costs: list[dict[str, Any]] = []

    for item in attempts:
        if not isinstance(item, dict):
            continue
        attempt = item.get("attempt")
        if not isinstance(attempt, int) or attempt < 1:
            continue
        output_path = source_run_dir / "output" / f"{safe_filename(task_id)}.attempt-{attempt}.json"
        output = read_json_object(output_path) or {}
        if output:
            agent_usage = add_usage(agent_usage, stored_usage(output.get("agent_usage")))
            output_usage = stored_usage(output.get("usage"))
            output_agent_usage = stored_usage(output.get("agent_usage"))
            tool_model_usage = add_usage(
                tool_model_usage,
                {
                    key: max(output_usage[key] - output_agent_usage[key], 0)
                    for key in output_usage
                },
            )
            loaded_attempt_usage = True
        cost = item.get("cost_usd")
        if not isinstance(cost, (int, float)):
            cost = output.get("cost_usd", 0.0)
        agent_cost = output.get("agent_cost_usd")
        if not isinstance(agent_cost, (int, float)):
            agent_cost = cost
        cost_value = max(float(cost or 0.0), 0.0)
        agent_cost_value = max(float(agent_cost or 0.0), 0.0)
        agent_cost_usd += agent_cost_value
        tool_model_cost_usd += max(cost_value - agent_cost_value, 0.0)
        if isinstance(item.get("elapsed_secs"), (int, float)):
            elapsed_secs += max(float(item["elapsed_secs"]), 0.0)
        attempt_costs.append(
            {
                "attempt": attempt,
                "ok": item.get("ok") is True,
                "cost_usd": round(cost_value, 10),
                "output_file": str(output_path),
            }
        )

    if not attempts:
        agent_usage = stored_usage(source_entry.get("agent_usage"))
        total_usage = stored_usage(source_entry.get("usage"))
        tool_model_usage = {
            key: max(total_usage[key] - agent_usage[key], 0)
            for key in total_usage
        }
        agent_cost_usd = max(float(source_entry.get("agent_cost_usd") or 0.0), 0.0)
        tool_model_cost_usd = max(float(source_entry.get("tool_model_cost_usd") or 0.0), 0.0)
        elapsed_secs = max(float(source_entry.get("elapsed_secs") or 0.0), 0.0)
    elif not loaded_attempt_usage:
        agent_usage = stored_usage(source_entry.get("agent_usage"))
        total_usage = stored_usage(source_entry.get("usage"))
        tool_model_usage = {
            key: max(total_usage[key] - agent_usage[key], 0)
            for key in total_usage
        }

    total_usage = add_usage(agent_usage, tool_model_usage)
    total_cost_usd = round(agent_cost_usd + tool_model_cost_usd, 10)
    return {
        "source_run_dir": str(source_run_dir),
        "source_results_jsonl": str(source_run_dir / "results.jsonl"),
        "source_output_file": str(source_run_dir / "output" / f"{safe_filename(task_id)}.json"),
        "attempt_count": len(attempt_costs) or int(source_entry.get("attempt") or 1),
        "attempts": attempt_costs,
        "elapsed_secs": round(elapsed_secs, 3),
        "cost_usd": total_cost_usd,
        "agent_cost_usd": round(agent_cost_usd, 10),
        "tool_model_cost_usd": round(tool_model_cost_usd, 10),
        "cost_breakdown": {
            "agent_usd": round(agent_cost_usd, 10),
            "tool_model_usd": round(tool_model_cost_usd, 10),
            "total_usd": total_cost_usd,
        },
        "usage": total_usage,
        "agent_usage": agent_usage,
        "tool_model_usage": tool_model_usage,
    }


def prepare_optimized_plan_execution_tasks(
    tasks: list[dict[str, Any]],
    source_run_dir: Path,
    selection_path: Path,
) -> list[dict[str, Any]]:
    selection = load_optimized_plan_selection(selection_path)
    source_results_path = source_run_dir / "results.jsonl"
    source_results = read_existing_results(source_results_path)
    if not source_results:
        raise SystemExit(f"plan optimization results are missing or empty: {source_results_path}")
    tasks_by_id = {task["task_id"]: task for task in tasks}
    unknown_tasks = sorted(set(selection) - set(tasks_by_id))
    if unknown_tasks:
        raise SystemExit(
            "optimized plan selection contains task(s) absent from evaluation input: "
            + ", ".join(unknown_tasks)
        )

    prepared: list[dict[str, Any]] = []
    for task in tasks:
        task_id = task["task_id"]
        optimization_types = selection.get(task_id)
        if optimization_types is None:
            continue
        source_entry = source_results.get(task_id)
        if not isinstance(source_entry, dict):
            raise SystemExit(f"plan optimization result is missing for selected task: {task_id}")
        if source_entry.get("task_mode") != "plan-optimization" or source_entry.get("ok") is not True:
            raise SystemExit(f"selected task does not have a successful plan-optimization result: {task_id}")
        raw_plans = source_entry.get("optimized_semantic_plan")
        if not isinstance(raw_plans, list) or not raw_plans:
            raise SystemExit(f"selected task has no optimized_semantic_plan: {task_id}")
        optimized_plans: list[dict[str, str]] = []
        for index, plan in enumerate(raw_plans):
            if (
                not isinstance(plan, dict)
                or plan.get("type") != "operator_dag"
                or not isinstance(plan.get("program"), str)
                or not plan["program"].strip()
            ):
                raise SystemExit(f"invalid optimized_semantic_plan[{index}] for selected task: {task_id}")
            optimized_plans.append({"type": "operator_dag", "program": plan["program"].strip()})
        prepared_task = dict(task)
        prepared_task["optimized_semantic_plan"] = optimized_plans
        prepared_task["optimization_analysis"] = source_entry.get("optimization_analysis", "")
        prepared_task["_optimization_types"] = optimization_types
        prepared_task["_plan_optimization"] = summarize_plan_optimization_phase(
            source_run_dir,
            task_id,
            source_entry,
        )
        prepared.append(prepared_task)

    if len(prepared) != len(selection):
        raise SystemExit(
            f"prepared optimized task count mismatch: expected={len(selection)} actual={len(prepared)}"
        )
    return prepared


def write_results_jsonl(path: Path, all_tasks: list[dict[str, Any]], existing: dict[str, dict[str, Any]]) -> None:
    task_ids = {task.get("task_id") for task in all_tasks}
    ordered = [existing[task["task_id"]] for task in all_tasks if task.get("task_id") in existing]
    ordered.extend(entry for task_id, entry in existing.items() if task_id not in task_ids)
    append_jsonl(path, ordered)


def final_result_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    total_usage = entry.get("usage") if isinstance(entry.get("usage"), dict) else {}
    total_tokens = to_nonnegative_int(entry.get("total_tokens")) or sum(
        to_nonnegative_int(total_usage.get(key))
        for key in ("input", "output", "cacheRead", "cacheWrite")
    )
    return {
        "task_id": entry.get("task_id"),
        "answer": entry.get("answer"),
        "elapsed_seconds": entry.get("elapsed_secs", entry.get("elapsed_seconds")),
        "cost_usd": entry.get("cost_usd"),
        "total_tokens": total_tokens,
    }


def write_result_files(results_dir: Path, all_tasks: list[dict[str, Any]], existing: dict[str, dict[str, Any]]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    for task in all_tasks:
        entry = existing.get(task.get("task_id"))
        if not entry:
            continue
        task_id = entry.get("task_id")
        if isinstance(task_id, str):
            write_json(compact_result_path(results_dir, task), final_result_from_entry(entry))


def task_answer_file_path(paths: dict[str, Path], task_id: str) -> Path:
    return paths["workspace_intermediate_dir"] / f"{safe_filename(task_id)}.answer.json"


def task_progress_log_path(paths: dict[str, Path], task_id: str) -> Path:
    return paths["workspace_intermediate_dir"] / f"{safe_filename(task_id)}.progress.log"


def task_tmp_dir(paths: dict[str, Path], task_id: str) -> Path:
    return paths["workspace_tmp_dir"] / safe_filename(task_id)


def task_attempt_tmp_dir(paths: dict[str, Path], task_id: str, attempt: int) -> Path:
    return task_tmp_dir(paths, task_id) / f"attempt-{attempt}"


def task_script_mtimes(script_dir: Path) -> dict[str, float]:
    if not script_dir.exists():
        return {}
    mtimes: dict[str, float] = {}
    for path in script_dir.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            mtimes[str(path.resolve())] = path.stat().st_mtime
        except OSError:
            continue
    return mtimes


def path_is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def select_task_script(script_dir: Path, execute_response: dict[str, Any], before_mtimes: dict[str, float]) -> tuple[Path | None, str | None]:
    response_keys = ("script_file", "script_path", "pipeline_script", "task_script", "script")
    for key in response_keys:
        value = execute_response.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = script_dir / candidate
        if candidate.suffix == ".py" and candidate.exists() and path_is_under(candidate, script_dir):
            try:
                resolved = str(candidate.resolve())
                mtime = candidate.stat().st_mtime
            except OSError:
                return None, f"script path is not readable: {candidate}"
            if before_mtimes.get(resolved) == mtime:
                return None, f"script was not created or updated in this attempt: {candidate}"
            return candidate.resolve(), None
        if candidate.suffix == ".py" and candidate.exists():
            return None, f"script path is outside this attempt scratch directory: {candidate}; required under {script_dir}"

    scripts = [
        path
        for path in script_dir.rglob("*.py")
        if "__pycache__" not in path.parts and path.is_file()
    ]
    if not scripts:
        return None, f"no Python task script found under {script_dir}"

    changed_or_new: list[Path] = []
    for path in scripts:
        try:
            resolved = str(path.resolve())
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if before_mtimes.get(resolved) != mtime:
            changed_or_new.append(path)
    if not changed_or_new and before_mtimes:
        return None, f"no Python task script was created or updated under {script_dir}"
    candidates = changed_or_new or scripts
    candidates.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0.0, reverse=True)
    return candidates[0].resolve(), None


def task_process_markers(paths: dict[str, Path], task_id: str) -> list[str]:
    tmp_dir = task_tmp_dir(paths, task_id)
    markers = [str(tmp_dir)]
    try:
        markers.append(str(tmp_dir.relative_to(ROOT_DIR)))
    except ValueError:
        pass
    return markers


def list_task_processes(paths: dict[str, Path], task_id: str) -> list[dict[str, Any]]:
    markers = task_process_markers(paths, task_id)
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,pgid=,stat=,args="],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError:
        return []
    current_pid = os.getpid()
    matches: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        pid_s, ppid_s, pgid_s, stat, cmd = parts
        try:
            pid = int(pid_s)
            ppid = int(ppid_s)
            pgid = int(pgid_s)
        except ValueError:
            continue
        if pid == current_pid or "ps -eo" in cmd:
            continue
        if any(marker and marker in cmd for marker in markers):
            matches.append({"pid": pid, "ppid": ppid, "pgid": pgid, "stat": stat, "cmd": cmd})
    return matches


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def summarize_task_processes(processes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "pid": process["pid"],
            "ppid": process["ppid"],
            "pgid": process["pgid"],
            "stat": process["stat"],
            "cmd": shorten_one_line(process["cmd"], 240),
        }
        for process in processes
    ]


def terminate_task_processes(processes: list[dict[str, Any]], *, label: str, reason: str) -> list[dict[str, Any]]:
    if not processes:
        return []
    current_pgid = os.getpgrp() if os.name == "posix" else None
    pgids = sorted({process["pgid"] for process in processes if process.get("pgid")})
    print(f"[cleanup] terminating {label} lingering task subprocesses pgids={pgids} reason={reason}", file=sys.stderr, flush=True)
    if os.name == "posix":
        for pgid in pgids:
            if current_pgid is not None and pgid == current_pgid:
                continue
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    else:
        for process in processes:
            try:
                os.kill(process["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.time() + 10
    while time.time() < deadline:
        if not any(process_exists(process["pid"]) for process in processes):
            return summarize_task_processes(processes)
        time.sleep(0.25)
    if os.name == "posix":
        for pgid in pgids:
            if current_pgid is not None and pgid == current_pgid:
                continue
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    else:
        for process in processes:
            try:
                os.kill(process["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
    return summarize_task_processes(processes)


def handle_lingering_task_processes(paths: dict[str, Path], task_id: str, *, reason: str) -> dict[str, Any]:
    label = f"task {task_id}"
    initial = list_task_processes(paths, task_id)
    if not initial:
        return {"found": False, "terminated": []}
    terminated = terminate_task_processes(initial, label=label, reason=reason)
    return {
        "found": True,
        "initial": summarize_task_processes(initial),
        "terminated": terminated,
    }


def compact_attempt_for_retry(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "retry_diagnostics"}


def build_retry_section(attempt: int, max_attempts: int, previous_attempts: list[dict[str, Any]]) -> str:
    if attempt <= 1 or not previous_attempts:
        return ""
    latest = previous_attempts[-1]
    diagnostics = latest.get("retry_diagnostics") if isinstance(latest.get("retry_diagnostics"), dict) else {}
    text_fields = ("stderr_tail", "stdout_tail")
    diagnostic_metadata = {key: value for key, value in diagnostics.items() if key not in text_fields}
    payload = {
        "attempt": f"{attempt}/{max_attempts}",
        "previous_attempt_summaries": [compact_attempt_for_retry(item) for item in previous_attempts[-3:]],
        "latest_failure_diagnostics": diagnostic_metadata,
    }
    source = str(diagnostics.get("diagnostic_source") or "previous attempt")
    artifact_paths = diagnostics.get("artifact_paths") if isinstance(diagnostics.get("artifact_paths"), dict) else {}
    raw_path = artifact_paths.get("raw_attempt") or artifact_paths.get("raw_latest")
    sections = [
        "Retry context:",
        json.dumps(payload, ensure_ascii=False, indent=2),
    ]
    stderr_tail = str(diagnostics.get("stderr_tail") or "")
    if stderr_tail:
        sections.extend([f"Relevant {source} stderr/traceback tail:", stderr_tail])
    stdout_tail = str(diagnostics.get("stdout_tail") or "")
    if stdout_tail:
        sections.extend([f"Relevant {source} stdout tail:", stdout_tail])
    if raw_path:
        sections.append(
            f"Before rewriting the solution, you MUST inspect the complete previous-attempt raw artifact at `{raw_path}` "
            "with Read or Grep. Inspect the stderr and script_run.stderr fields first; read stdout only when needed."
        )
    else:
        sections.append(
            "Before rewriting the solution, diagnose the previous failure from the bounded diagnostic tail above."
        )
    sections.extend(
        [
            "The inline diagnostic is intentionally bounded; the artifact paths are authoritative.",
            "Do not repeat the same failing implementation. Produce a corrected, faster, and more robust script.",
        ]
    )
    return "\n".join(sections)


def build_resumed_retry_prompt(
    task: dict[str, Any],
    paths: dict[str, Path],
    attempt: int,
    max_attempts: int,
    previous_attempts: list[dict[str, Any]],
) -> str:
    if attempt <= 1 or not previous_attempts:
        raise ValueError("resumed retry prompt requires a previous attempt")

    task_id = task["task_id"]
    answer_path = task_answer_file_path(paths, task_id)
    progress_path = task_progress_log_path(paths, task_id)
    attempt_tmp = task_attempt_tmp_dir(paths, task_id, attempt)
    latest = previous_attempts[-1]
    diagnostics = latest.get("retry_diagnostics") if isinstance(latest.get("retry_diagnostics"), dict) else {}
    artifact_paths = diagnostics.get("artifact_paths") if isinstance(diagnostics.get("artifact_paths"), dict) else {}
    raw_path = artifact_paths.get("raw_attempt") or artifact_paths.get("raw_latest")
    previous_script = artifact_paths.get("previous_script") or latest.get("script_path")
    diagnostic_metadata = {
        key: value
        for key, value in diagnostics.items()
        if key not in {"stderr_tail", "stdout_tail", "artifact_paths"}
    }
    payload = {
        "attempt": f"{attempt}/{max_attempts}",
        "previous_attempt": compact_attempt_for_retry(latest),
        "latest_failure": diagnostic_metadata,
        "artifact_paths": artifact_paths,
    }
    sections = [
        "You are continuing the same benchmark task in the same Claude Code conversation.",
        "The original task query, answer schema, dataset contract, and hard constraints remain in the conversation and still apply.",
        "The previous task script was executed externally by the runner after your last turn, so its runtime failure is not otherwise present in this transcript.",
        "",
        f"Retry attempt: {attempt}/{max_attempts}",
        f"New script/scratch directory for this attempt: {attempt_tmp}",
        f"Task progress log: {progress_path}",
        f"Authoritative answer JSON file: {answer_path}",
        "Retry metadata:",
        json.dumps(payload, ensure_ascii=False, indent=2),
    ]
    if raw_path:
        sections.append(
            f"Before writing the replacement, you MUST inspect the complete previous-attempt raw artifact at `{raw_path}` "
            "with Read or Grep. Inspect stderr and script_run.stderr first; read stdout only when needed."
        )
    else:
        stderr_tail = str(diagnostics.get("stderr_tail") or "")
        stdout_tail = str(diagnostics.get("stdout_tail") or "")
        if stderr_tail:
            sections.extend(["Previous failure stderr/traceback tail:", stderr_tail])
        elif stdout_tail:
            sections.extend(["Previous failure stdout tail:", stdout_tail])
    if previous_script:
        sections.append(f"Inspect the previous script at `{previous_script}` before correcting it.")
    sections.extend(
        [
            f"Write the corrected script and any new scratch files only under `{attempt_tmp}`.",
            "Do not edit or delete files in any previous attempt directory.",
            "Do not repeat the same failing implementation. Correct the diagnosed failure and preserve valid work from the prior attempt.",
            "Write the task script only; do not execute it yourself. The runner will execute it synchronously after you return.",
            f"The task script must write exactly this JSON shape to `{answer_path}`:",
            f'  {{"task_id": "{task_id}", "answer": <value matching the original answer_schema>, "evidence": [<short filenames or notes>]}}',
            "",
            "Final assistant response must be exactly one JSON object and no markdown:",
            "{",
            f'  "task_id": "{task_id}",',
            f'  "answer_file": "{answer_path}",',
            '  "script_file": "absolute path to the Python script you wrote",',
            '  "evidence": ["short list of filenames or concise notes"]',
            "}",
        ]
    )
    return "\n".join(sections)


def build_prompt(task: dict[str, Any], args: argparse.Namespace, paths: dict[str, Path], attempt: int, max_attempts: int, previous_attempts: list[dict[str, Any]]) -> str:
    task_id = task["task_id"]
    answer_path = task_answer_file_path(paths, task_id)
    progress_path = task_progress_log_path(paths, task_id)
    task_tmp = task_tmp_dir(paths, task_id)
    attempt_tmp = task_attempt_tmp_dir(paths, task_id, attempt)
    retry_section = build_retry_section(attempt, max_attempts, previous_attempts)
    dataset_context = dataset_context_for_prompt(paths)
    operator_contract_section: list[str] = []
    operator_constraints: list[str] = []
    if getattr(args, "task_mode", DEFAULT_TASK_MODE) == "operator-implementation":
        operator_contract_section = [
            "Operator implementation experiment contract:",
            operator_task_contract_for_prompt(task),
        ]
        operator_constraints = [
            "- Implement the supplied reference_semantic_plan as the authoritative execution plan; do not redesign or replace it with a different plan.",
            "- Read data only from the listed resolved_inputs. Do not use other datasets, collections, or auxiliary files even if they are present under --data-dir or in the cache.",
            "- Preserve the reference plan's predicates, extraction schema, join condition, grouping keys, and aggregation semantics exactly.",
        ]
    elif getattr(args, "task_mode", DEFAULT_TASK_MODE) == "optimized-plan-execution":
        operator_contract_section = [
            "Optimized plan execution experiment contract:",
            optimized_plan_execution_contract_for_prompt(task),
        ]
        operator_constraints = [
            "- Implement the supplied optimized_semantic_plan as the authoritative execution plan; do not redesign, re-optimize, or replace it with the reference plan or another plan.",
            "- Read data only from the listed resolved_inputs. Do not use other datasets, collections, or auxiliary files even if they are present under --data-dir or in the cache.",
            "- Preserve the optimized plan's operator order, predicates, extraction schema, join condition, grouping keys, and aggregation semantics exactly.",
        ]
    return "\n".join(
        [
            "You are processing exactly one benchmark evaluation task.",
            "",
            f"Project root: {ROOT_DIR}",
            f"Dataset directory (read-only): {args.data_dir}",
            f"Evaluation input file (read-only): {args.input}",
            f"Runtime workspace: {paths['workspace_dir']}",
            f"Task scratch root: {task_tmp}",
            f"Script/scratch directory for this attempt: {attempt_tmp}",
            f"Intermediate directory: {paths['workspace_intermediate_dir']}",
            f"Complete dataset cache JSONL: {paths['dataset_cache_jsonl']}",
            f"Dataset cache manifest: {paths['dataset_cache_manifest']}",
            f"Tool-side model helper: {paths['model_call_helper']}",
            f"Task progress log: {progress_path}",
            f"Authoritative answer JSON file: {answer_path}",
            f"Task ID: {task_id}",
            f"Task query: {task.get('query', '')}",
            f"Answer schema: {json.dumps(task.get('answer_schema'), ensure_ascii=False)}",
            *operator_contract_section,
            "Dataset structure and cache contract:",
            dataset_context,
            retry_section,
            "",
            "Hard constraints:",
            "- Stay within the project root. Do not read, write, or execute against paths outside it.",
            "- Never create, edit, delete, or clean up the dataset directory or evaluation input file.",
            f"- Write the task script and scratch files for this attempt under `{attempt_tmp}` only.",
            f"- Write checkpoints and intermediate outputs under `{paths['workspace_dir']}` only.",
            "- Start from the injected dataset structure and cache contract, then select only the collection(s) required by the query.",
            "- Use cache_record_type and collection correctly; the cache may mix file documents, CSV text rows, and JSON records.",
            "- Pure metadata CSV tables listed under metadata_only_tables are source tables for exact joins and are intentionally not duplicated into the text cache.",
            "- Task scripts may read CLAUDECODE_DATA_DIR, CLAUDECODE_DATASET_CACHE_JSONL, and CLAUDECODE_DATASET_CACHE_MANIFEST from the environment instead of hard-coding paths.",
            "- Do not use the Read tool on the evaluation input or dataset_texts.jsonl; they may exceed Read limits.",
            f"- For large JSON/JSONL files, write streaming Python scripts under `{attempt_tmp}` instead of loading whole files through Read.",
            "- The task query and answer schema are already included in this prompt; do not read any portion of the evaluation input file.",
            "- Use each cache row's document_id and collection fields; do not assume the filename is the identifier for structured rows.",
            "- Process this task independently; do not use previous task answers.",
            "- Never print, log, or include OPENAI_API_KEY or any key/token/secret.",
            *operator_constraints,
            "",
            "Semantic model-use rules:",
            "- Deterministic scripts are appropriate for indexing, candidate filtering, exact matching, counting, and verification.",
            "- For semantic classification or extraction, use model_call_helper.batch_call_model(prompts, temperature=0).",
            f"- Tool-side model concurrency is capped by the runner at {args.tool_model_max_concurrency}, never above 10.",
            f"- Tool-side output tokens are capped at {args.tool_model_max_tokens}.",
            '- model_call_helper always sends chat_template_kwargs={"enable_thinking": false}. Do not bypass it for semantic batch calls.',
            "- For multiple documents/snippets, collect prompts and call batch_call_model once per semantic step, or in large checkpointed chunks. Do not do one-by-one serial model calls.",
            "- batch_call_model returns dictionaries; read result['text'].",
            "- Your job in this phase is to write the task script only. Do not run the script yourself; the runner will execute it synchronously after you return.",
            "",
            "Long-running process rules:",
            f"- Task timeout is {args.timeout}s. Do not kill a child process just because it is slow while still running without an error.",
            "- If a tool call fails because a file is too large, recover by writing a streaming script, then continue the task.",
            "- If a script may take minutes, write progress lines to CLAUDECODE_TASK_PROGRESS_LOG and flush them.",
            f"- For model-heavy scripts, write checkpoint files under `{attempt_tmp}` after each chunk.",
            "- Only terminate a script if it clearly failed or the task-level timeout is reached.",
            "- Run task scripts synchronously in the foreground. Do not append &, use nohup/disown, launch tmux/screen, or daemonize.",
            "- Do not use persistent log monitors such as tail -f, tail -F, tail --follow, watch, or sleep/ps polling loops.",
            "",
            "Answer file rules:",
            "- The task is incomplete until the authoritative answer JSON file exists and matches the answer schema.",
            "- Do not finish after exploratory commands or file-header inspection; continue by writing the task script under this attempt directory.",
            f"- You must create or update at least one Python task script under `{attempt_tmp}` before returning.",
            "- The task script must write the authoritative answer JSON file when the runner executes it.",
            f"- The task script must write exactly this JSON shape to {answer_path}:",
            f'  {{"task_id": "{task_id}", "answer": <value matching answer_schema>, "evidence": [<short filenames or notes>]}}',
            "- For table schemas, answer must be an array of row objects, each row containing exactly the requested columns.",
            "- For scalar schemas, answer must be one scalar of value_type.",
            "- If no table matches are found, return an empty array.",
            "- Use empty string for missing strings and false for absent booleans; do not use null for schema fields.",
            "",
            "Final assistant response must be exactly one JSON object and no markdown:",
            "{",
            f'  "task_id": "{task_id}",',
            f'  "answer_file": "{answer_path}",',
            '  "script_file": "absolute path to the Python script you wrote",',
            '  "evidence": ["short list of filenames or concise notes"]',
            "}",
        ]
    )


def build_child_env(
    tool_env: dict[str, str],
    agent_env: dict[str, str],
    args: argparse.Namespace,
    paths: dict[str, Path],
    task_id: str | None = None,
) -> dict[str, str]:
    child_env = os.environ.copy()
    child_env.update(tool_env)
    child_env["CLAUDECODE_PROJECT_ROOT"] = str(ROOT_DIR)
    child_env["CONTRACT_EXHIBIT_ROOT"] = str(ROOT_DIR)
    child_env["CLAUDECODE_DATA_DIR"] = str(args.data_dir)
    child_env["CLAUDECODE_DATASET_CACHE_JSONL"] = str(paths["dataset_cache_jsonl"])
    child_env["CLAUDECODE_DATASET_CACHE_MANIFEST"] = str(paths["dataset_cache_manifest"])
    child_env["HOME"] = str(paths["home_dir"])
    child_env["CLAUDE_CONFIG_DIR"] = str(paths["state_dir"] / "claude_config")
    child_env["XDG_CONFIG_HOME"] = str(paths["home_dir"] / ".config")
    child_env["XDG_CACHE_HOME"] = str(paths["home_dir"] / ".cache")
    for key in CLAUDE_PROVIDER_ENV_FLAGS:
        child_env.pop(key, None)
    child_env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    child_env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] = "1"
    child_env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] = "1"
    base_url = agent_anthropic_base_url(agent_env)
    if base_url:
        child_env["ANTHROPIC_BASE_URL"] = base_url
    auth_token = agent_auth_token(agent_env)
    if auth_token:
        child_env["ANTHROPIC_AUTH_TOKEN"] = auth_token
    api_key = agent_api_key(agent_env) or auth_token
    if api_key:
        child_env["ANTHROPIC_API_KEY"] = api_key
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = (
        f"{paths['workspace_dir']}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(paths["workspace_dir"])
    )
    child_env["CLAUDECODE_TOOL_MODEL_MAX_CONCURRENCY"] = str(args.tool_model_max_concurrency)
    child_env["CLAUDECODE_TOOL_MODEL_MAX_TOKENS"] = str(args.tool_model_max_tokens)
    child_env["CLAUDECODE_TOOL_MODEL_DISABLE_THINKING"] = "1"
    if task_id:
        safe_task_id = safe_filename(task_id)
        child_env["CLAUDECODE_TOOL_MODEL_USAGE_PATH"] = str(paths["tool_model_usage_dir"] / f"{safe_task_id}.jsonl")
        child_env["CLAUDECODE_TOOL_MODEL_SEMAPHORE_DIR"] = str(paths["tool_model_semaphore_dir"] / safe_task_id)
        child_env["CLAUDECODE_TASK_PROGRESS_LOG"] = str(task_progress_log_path(paths, task_id))
    return child_env


def build_task_script_env(tool_env: dict[str, str], args: argparse.Namespace, paths: dict[str, Path], task_id: str) -> dict[str, str]:
    child_env = os.environ.copy()
    child_env.update(tool_env)
    child_env["CLAUDECODE_PROJECT_ROOT"] = str(ROOT_DIR)
    child_env["CONTRACT_EXHIBIT_ROOT"] = str(ROOT_DIR)
    child_env["CLAUDECODE_DATA_DIR"] = str(args.data_dir)
    child_env["CLAUDECODE_DATASET_CACHE_JSONL"] = str(paths["dataset_cache_jsonl"])
    child_env["CLAUDECODE_DATASET_CACHE_MANIFEST"] = str(paths["dataset_cache_manifest"])
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = (
        f"{paths['workspace_dir']}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(paths["workspace_dir"])
    )
    child_env["CLAUDECODE_TOOL_MODEL_MAX_CONCURRENCY"] = str(args.tool_model_max_concurrency)
    child_env["CLAUDECODE_TOOL_MODEL_MAX_TOKENS"] = str(args.tool_model_max_tokens)
    child_env["CLAUDECODE_TOOL_MODEL_DISABLE_THINKING"] = "1"
    safe_task_id = safe_filename(task_id)
    child_env["CLAUDECODE_TOOL_MODEL_USAGE_PATH"] = str(paths["tool_model_usage_dir"] / f"{safe_task_id}.jsonl")
    child_env["CLAUDECODE_TOOL_MODEL_SEMAPHORE_DIR"] = str(paths["tool_model_semaphore_dir"] / safe_task_id)
    child_env["CLAUDECODE_TASK_PROGRESS_LOG"] = str(task_progress_log_path(paths, task_id))
    return child_env


def build_agent_command(
    task: dict[str, Any],
    agent_env: dict[str, str],
    args: argparse.Namespace,
    paths: dict[str, Path],
    attempt: int,
    max_attempts: int,
    previous_attempts: list[dict[str, Any]],
    session_id: str,
    resume_session: bool,
) -> list[str]:
    tool_csv = (
        ""
        if getattr(args, "task_mode", DEFAULT_TASK_MODE) == "plan-optimization"
        else "Read,Write,Edit,Glob,Grep"
    )
    session_args = ["--resume", session_id] if resume_session else ["--session-id", session_id]
    return [
        *shlex.split(args.claude_command),
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        require_env_value(agent_env, "AGENT_MAIN_MODEL"),
        "--settings",
        str(paths["settings_path"]),
        "--mcp-config",
        str(paths["mcp_path"]),
        "--strict-mcp-config",
        "--dangerously-skip-permissions",
        *session_args,
        "--add-dir",
        str(ROOT_DIR),
        "--tools",
        tool_csv,
        "--allowedTools",
        tool_csv,
    ]


def register_child_process(process: subprocess.Popen[str], label: str) -> None:
    with _ACTIVE_CHILDREN_LOCK:
        _ACTIVE_CHILDREN[process.pid] = (process, label)


def unregister_child_process(process: subprocess.Popen[str]) -> None:
    with _ACTIVE_CHILDREN_LOCK:
        _ACTIVE_CHILDREN.pop(process.pid, None)


def terminate_process_group(process: subprocess.Popen[str], label: str, reason: str) -> None:
    if process.poll() is not None:
        return
    print(f"[runner] terminating {label}: {reason}", file=sys.stderr, flush=True)
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=15)
    except Exception:
        if process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except Exception:
                pass


def terminate_active_processes(reason: str) -> None:
    with _ACTIVE_CHILDREN_LOCK:
        children = list(_ACTIVE_CHILDREN.values())
    for process, label in children:
        terminate_process_group(process, label, reason)


def install_shutdown_handlers() -> None:
    def handler(signum: int, frame: Any) -> None:
        terminate_active_processes(f"signal {signum}")
        raise SystemExit(128 + signum)

    if os.name == "posix":
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)


def shorten_one_line(value: str, max_len: int = 220) -> str:
    one_line = re.sub(r"\s+", " ", value).strip()
    if len(one_line) <= max_len:
        return one_line
    return one_line[: max_len - 3] + "..."


def replace_self_reported_model_label(text: str, model: str | None) -> str:
    stripped = text.strip()
    if not model:
        return stripped
    return re.sub(
        r"^\s*(?:\[[^\]\n]*model\s*:\s*[^\]\n]+\]\s*)+",
        f"[Model: {model}] ",
        stripped,
        count=1,
        flags=re.IGNORECASE,
    ).strip()


def iter_json_lines(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or not stripped.startswith("{"):
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def summarize_stream_event(event: dict[str, Any]) -> list[str]:
    messages: list[str] = []
    event_type = event.get("type")
    if event_type == "system" and event.get("subtype") == "init" and isinstance(event.get("model"), str):
        messages.append(f"[agent][model] {event['model']}")
    if event_type in {"assistant", "message"}:
        message = event.get("message") if isinstance(event.get("message"), dict) else event
        model = message.get("model") or event.get("model")
        if not isinstance(model, str):
            model = None
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type in {"tool_use", "toolCall", "tool_call"}:
                    name = item.get("name") or item.get("toolName") or "tool"
                    args = item.get("input") or item.get("arguments") or {}
                    messages.append(f"[agent][tool] {name}: {shorten_one_line(json.dumps(args, ensure_ascii=False), 220)}")
                elif item_type == "text" and isinstance(item.get("text"), str):
                    text = replace_self_reported_model_label(item["text"], model)
                    if text:
                        messages.append(f"[agent][assistant] {shorten_one_line(text, 220)}")
    if event_type in {"tool_result", "toolResult"}:
        name = event.get("name") or event.get("toolName") or "tool"
        status = event.get("status") or "completed"
        messages.append(f"[agent][result] {name} {status}")
    if event_type == "result" and isinstance(event.get("result"), str):
        messages.append(f"[agent][result] {shorten_one_line(event['result'], 220)}")
    return messages


def bash_tool_commands_from_event(event: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    message = event.get("message") if isinstance(event.get("message"), dict) else event
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return commands
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in {"tool_use", "toolCall", "tool_call"}:
            continue
        name = str(item.get("name") or item.get("toolName") or "")
        if name != "Bash":
            continue
        tool_input = item.get("input") or item.get("arguments") or {}
        if isinstance(tool_input, dict) and isinstance(tool_input.get("command"), str):
            commands.append(tool_input["command"])
    return commands


FORBIDDEN_BASH_COMMAND_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bwatch\b", re.IGNORECASE), "watch monitor"),
    (re.compile(r"(?s)\bsleep\s+[0-9.]+[smhd]?\s*(?:&&|;|\n)\s*(?:\(\s*)?(?:tail|grep|ps|cat|test|ls|python3?)\b", re.IGNORECASE), "delayed monitoring command"),
    (re.compile(r"(?s)\bwhile\b.+\bdo\b.+\b(?:sleep|tail|grep|ps)\b", re.IGNORECASE), "polling loop"),
    (re.compile(r"(?s)\buntil\b.+\bdo\b.+\b(?:sleep|tail|grep|ps)\b", re.IGNORECASE), "polling loop"),
    (re.compile(r"(?s)\bfor\b.+\bdo\b.+\b(?:sleep|tail|grep|ps)\b", re.IGNORECASE), "polling loop"),
    (re.compile(r"(?<![&>])&(?![&>])\s*(?:$|[;\n])"), "background command"),
    (re.compile(r"\b(?:nohup|disown|tmux|screen)\b", re.IGNORECASE), "detached process/session command"),
]


def tail_command_uses_follow(command: str) -> bool:
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return bool(re.search(r"\btail\s+(?:-[A-Za-z]*[fF][A-Za-z]*(?=\s|$)|--follow(?:=|\b))", command))

    separators = {";", "&&", "||", "|"}
    for idx, token in enumerate(tokens):
        if Path(token).name == "tailf":
            return True
        if Path(token).name != "tail":
            continue
        for arg in tokens[idx + 1 :]:
            if arg in separators:
                break
            if arg == "--":
                continue
            if arg.startswith("--follow"):
                return True
            if arg.startswith("--"):
                continue
            if arg.startswith("-") and any(ch in arg for ch in ("f", "F")):
                return True
    return False


def forbidden_bash_command_reason(command: str) -> str | None:
    if tail_command_uses_follow(command):
        return "persistent tail monitor"
    for pattern, reason in FORBIDDEN_BASH_COMMAND_PATTERNS:
        if pattern.search(command):
            return reason
    return None


def forbidden_bash_command_from_event(event: dict[str, Any]) -> dict[str, str] | None:
    for command in bash_tool_commands_from_event(event):
        reason = forbidden_bash_command_reason(command)
        if reason:
            return {"reason": reason, "command": command}
    return None


def print_new_progress_events(progress_log_path: Path, position: int) -> int:
    if not progress_log_path.exists():
        return position
    try:
        with progress_log_path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(position)
            while True:
                line = handle.readline()
                if not line:
                    break
                position = handle.tell()
                text = shorten_one_line(line.strip(), 260)
                if text:
                    print(f"[agent][progress] {text}", flush=True)
    except OSError:
        return position
    return position


def to_nonnegative_int(value: Any) -> int:
    try:
        return max(int(float(str(value))), 0)
    except (TypeError, ValueError):
        return 0


def add_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {
        "input": to_nonnegative_int(left.get("input")) + to_nonnegative_int(right.get("input")),
        "output": to_nonnegative_int(left.get("output")) + to_nonnegative_int(right.get("output")),
        "cacheRead": to_nonnegative_int(left.get("cacheRead")) + to_nonnegative_int(right.get("cacheRead")),
        "cacheWrite": to_nonnegative_int(left.get("cacheWrite")) + to_nonnegative_int(right.get("cacheWrite")),
    }


def usage_from_dict(usage: dict[str, Any]) -> dict[str, int]:
    return {
        "input": to_nonnegative_int(usage.get("input") or usage.get("prompt_tokens") or usage.get("promptTokens") or usage.get("input_tokens")),
        "output": to_nonnegative_int(usage.get("output") or usage.get("completion_tokens") or usage.get("completionTokens") or usage.get("output_tokens")),
        "cacheRead": to_nonnegative_int(usage.get("cacheRead") or usage.get("cache_read") or usage.get("cache_read_input_tokens") or usage.get("cacheReadInputTokens")),
        "cacheWrite": to_nonnegative_int(usage.get("cacheWrite") or usage.get("cache_write") or usage.get("cache_creation_input_tokens") or usage.get("cacheCreationInputTokens")),
    }


def usage_from_model_usage(raw: Any) -> dict[str, int] | None:
    if not isinstance(raw, dict):
        return None
    tokens = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
    saw_usage = False
    for value in raw.values():
        if not isinstance(value, dict):
            continue
        saw_usage = True
        tokens["input"] += to_nonnegative_int(value.get("inputTokens") or value.get("input_tokens") or value.get("input"))
        tokens["output"] += to_nonnegative_int(value.get("outputTokens") or value.get("output_tokens") or value.get("output"))
        tokens["cacheRead"] += to_nonnegative_int(value.get("cacheReadInputTokens") or value.get("cache_read_input_tokens") or value.get("cacheRead") or value.get("cache_read"))
        tokens["cacheWrite"] += to_nonnegative_int(value.get("cacheCreationInputTokens") or value.get("cache_creation_input_tokens") or value.get("cacheWrite") or value.get("cache_write"))
    if not saw_usage or sum(tokens.values()) == 0:
        return None
    return tokens


def normalize_million_token_cost(value: Any) -> float:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return 0.0
    if parsed < 0:
        return 0.0
    return parsed * 1_000_000 if 0 < parsed < 0.01 else parsed


def calculate_cost_usd(usage: dict[str, int], env: dict[str, str]) -> float:
    input_per_million = normalize_million_token_cost(env.get("LLM_INPUT_COST_PER_TOKEN") or env.get("LLM_MODEL_USD_PER_INPUT_TOKEN"))
    output_per_million = normalize_million_token_cost(env.get("LLM_OUTPUT_COST_PER_TOKEN") or env.get("LLM_MODEL_USD_PER_OUTPUT_TOKEN"))
    cache_read_per_million = (
        normalize_million_token_cost(env.get("LLM_CACHE_READ_COST_PER_TOKEN") or env.get("LLM_MODEL_USD_PER_CACHE_READ_TOKEN"))
        or input_per_million
    )
    cache_write_per_million = (
        normalize_million_token_cost(env.get("LLM_CACHE_WRITE_COST_PER_TOKEN") or env.get("LLM_MODEL_USD_PER_CACHE_WRITE_TOKEN"))
        or output_per_million
    )
    return round(
        (
            usage.get("input", 0) * input_per_million
            + usage.get("output", 0) * output_per_million
            + usage.get("cacheRead", 0) * cache_read_per_million
            + usage.get("cacheWrite", 0) * cache_write_per_million
        )
        / 1_000_000,
        10,
    )


def print_tool_usage_heartbeat(tool_usage_path: Path, previous_lines: int, last_heartbeat: float, *, force: bool = False) -> tuple[int, float]:
    if not tool_usage_path.exists():
        return previous_lines, last_heartbeat
    try:
        lines = [line for line in tool_usage_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return previous_lines, last_heartbeat
    current_lines = len(lines)
    now = time.time()
    changed = current_lines != previous_lines
    due = now - last_heartbeat >= 10
    if current_lines and (force or (changed and due)):
        total_cost = 0.0
        total_usage = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            usage = usage_from_dict(record.get("usage", {}) if isinstance(record.get("usage"), dict) else record)
            total_usage = add_usage(total_usage, usage)
            if isinstance(record.get("cost_usd"), (int, float)):
                total_cost += float(record["cost_usd"])
        print(
            "[agent][tool-model] "
            f"calls={current_lines} input={total_usage['input']} output={total_usage['output']} "
            f"cost_usd={round(total_cost, 6)}",
            flush=True,
        )
        last_heartbeat = now
    return current_lines, last_heartbeat


def monitor_progress(tool_usage_path: Path, progress_log_path: Path, stop_event: threading.Event) -> None:
    progress_position = 0
    tool_usage_lines = 0
    last_heartbeat = 0.0
    while not stop_event.is_set():
        progress_position = print_new_progress_events(progress_log_path, progress_position)
        tool_usage_lines, last_heartbeat = print_tool_usage_heartbeat(
            tool_usage_path, tool_usage_lines, last_heartbeat
        )
        time.sleep(0.5)
    print_new_progress_events(progress_log_path, progress_position)
    print_tool_usage_heartbeat(tool_usage_path, tool_usage_lines, 0.0, force=True)


def run_task_script_foreground(
    *,
    task_id: str,
    script_path: Path,
    tool_env: dict[str, str],
    args: argparse.Namespace,
    paths: dict[str, Path],
    secrets: list[str],
) -> dict[str, Any]:
    label = f"task-script-{task_id}"
    python_command = shlex.split(tool_env.get("CLAUDE_PIPELINE_PYTHON", "")) or [sys.executable]
    command = [*python_command, "-u", str(script_path)]
    child_env = build_task_script_env(tool_env, args, paths, task_id)
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    safe_task_id = safe_filename(task_id)
    tool_usage_path = paths["tool_model_usage_dir"] / f"{safe_task_id}.jsonl"
    progress_log_path = task_progress_log_path(paths, task_id)
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=monitor_progress,
        args=(tool_usage_path, progress_log_path, stop_event),
        daemon=True,
    )
    monitor.start()
    popen_kwargs: dict[str, Any] = {}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    started = time.time()
    try:
        process = subprocess.Popen(
            command,
            cwd=str(script_path.parent),
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            **popen_kwargs,
        )
    except OSError as exc:
        stop_event.set()
        monitor.join(timeout=2)
        return {
            "label": label,
            "command": command,
            "cwd": str(script_path.parent),
            "script_path": str(script_path),
            "exit_code": 127,
            "timed_out": False,
            "elapsed_ms": 0,
            "stdout": "",
            "stderr": str(exc),
            "error": str(exc),
        }

    register_child_process(process, label)

    def read_stream(stream: Any, parts: list[str], prefix: str) -> None:
        for line in stream:
            safe_line = redact_text(line, secrets)
            parts.append(safe_line)
            if not args.no_agent_log and safe_line.strip():
                print(f"[script][{prefix}] {shorten_one_line(safe_line.strip(), 220)}", flush=True)

    stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, stdout_parts, "stdout"), daemon=True)
    stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, stderr_parts, "stderr"), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    timed_out = False
    try:
        process.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        terminate_process_group(process, label, reason=f"timeout after {args.timeout}s")
    except KeyboardInterrupt:
        terminate_process_group(process, label, reason="keyboard interrupt")
        raise
    finally:
        unregister_child_process(process)
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        stop_event.set()
        monitor.join(timeout=2)

    return {
        "label": label,
        "command": command,
        "cwd": str(script_path.parent),
        "script_path": str(script_path),
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "elapsed_ms": int((time.time() - started) * 1000),
        "stdout": "".join(stdout_parts),
        "stderr": "".join(stderr_parts),
    }


def run_agent_task(
    task: dict[str, Any],
    tool_env: dict[str, str],
    agent_env: dict[str, str],
    args: argparse.Namespace,
    paths: dict[str, Path],
    secrets: list[str],
    *,
    attempt: int,
    max_attempts: int,
    previous_attempts: list[dict[str, Any]],
    session_id: str | None = None,
    resume_session: bool = False,
) -> dict[str, Any]:
    task_id = task["task_id"]
    safe_task_id = safe_filename(task_id)
    if resume_session and session_id is None:
        raise ValueError("resume_session requires an existing session_id")
    session_id = session_id or str(uuid.uuid4())
    for path in (task_answer_file_path(paths, task_id), task_progress_log_path(paths, task_id)):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    tool_usage_path = paths["tool_model_usage_dir"] / f"{safe_task_id}.jsonl"
    if attempt == 1:
        try:
            tool_usage_path.unlink()
        except FileNotFoundError:
            pass

    command = build_agent_command(
        task,
        agent_env,
        args,
        paths,
        attempt,
        max_attempts,
        previous_attempts,
        session_id,
        resume_session,
    )
    child_env = build_child_env(tool_env, agent_env, args, paths, task_id)
    started = time.time()
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    parsed_event_count = 0
    policy_violation: dict[str, str] | None = None
    policy_lock = threading.Lock()
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=monitor_progress,
        args=(tool_usage_path, task_progress_log_path(paths, task_id), stop_event),
        daemon=True,
    )
    monitor.start()
    popen_kwargs: dict[str, Any] = {}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    process = subprocess.Popen(
        command,
        cwd=str(ROOT_DIR),
        env=child_env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        **popen_kwargs,
    )
    register_child_process(process, f"claudecode-task-{task_id}")

    def read_stdout() -> None:
        nonlocal parsed_event_count, policy_violation
        assert process.stdout is not None
        for line in process.stdout:
            safe_line = redact_text(line, secrets)
            stdout_parts.append(safe_line)
            try:
                event = json.loads(safe_line)
            except json.JSONDecodeError:
                if not args.no_agent_log and safe_line.strip():
                    print(f"[agent][stdout] {shorten_one_line(safe_line.strip(), 220)}", flush=True)
                continue
            if isinstance(event, dict):
                parsed_event_count += 1
                violation = forbidden_bash_command_from_event(event)
                if violation is not None:
                    with policy_lock:
                        if policy_violation is None:
                            policy_violation = violation
                            if not args.no_agent_log:
                                print(
                                    "[agent][policy] rejected Bash command: "
                                    f"{violation['reason']}: {shorten_one_line(violation['command'], 220)}",
                                    flush=True,
                                )
                            terminate_process_group(
                                process,
                                f"claudecode-task-{task_id}",
                                reason=f"forbidden Bash command: {violation['reason']}",
                            )
                if not args.no_agent_log:
                    for message in summarize_stream_event(event):
                        print(message, flush=True)

    def read_stderr() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            safe_line = redact_text(line, secrets)
            stderr_parts.append(safe_line)
            if not args.no_agent_log and safe_line.strip():
                print(f"[agent][stderr] {shorten_one_line(safe_line.strip(), 220)}", file=sys.stderr, flush=True)

    stdout_thread = threading.Thread(target=read_stdout, daemon=True)
    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    if process.stdin is not None:
        if getattr(args, "task_mode", DEFAULT_TASK_MODE) == "plan-optimization":
            prompt = (
                build_resumed_plan_optimization_retry_prompt(
                    task,
                    attempt,
                    max_attempts,
                    previous_attempts,
                )
                if resume_session
                else build_plan_optimization_prompt(
                    task,
                    attempt,
                    max_attempts,
                    previous_attempts,
                )
            )
        else:
            prompt = (
                build_resumed_retry_prompt(task, paths, attempt, max_attempts, previous_attempts)
                if resume_session
                else build_prompt(task, args, paths, attempt, max_attempts, previous_attempts)
            )
        process.stdin.write(prompt)
        process.stdin.close()
    timed_out = False
    timeout = args.agent_timeout + args.agent_exit_grace_secs
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        terminate_process_group(process, f"claudecode-task-{task_id}", reason=f"timeout after {timeout}s")
    except KeyboardInterrupt:
        terminate_process_group(process, f"claudecode-task-{task_id}", reason="keyboard interrupt")
        raise
    finally:
        unregister_child_process(process)
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        stop_event.set()
        monitor.join(timeout=2)

    exit_code = process.returncode
    if policy_violation is not None and (exit_code is None or exit_code == 0):
        exit_code = 126
    return {
        "task_id": task_id,
        "session_id": session_id,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "command": redact_payload(command, secrets),
        "cwd": str(ROOT_DIR),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "elapsed_ms": int((time.time() - started) * 1000),
        "stdout": "".join(stdout_parts),
        "stderr": "".join(stderr_parts),
        "parsed_event_count": parsed_event_count,
        "policy_violation": policy_violation,
    }


def extract_assistant_text(stdout: str, stderr: str) -> str:
    events = iter_json_lines(stdout)
    result_texts: list[str] = []
    assistant_texts: list[str] = []
    for event in events:
        if event.get("type") == "result" and isinstance(event.get("result"), str):
            result_texts.append(event["result"])
        message = event.get("message") if isinstance(event.get("message"), dict) else event
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                    assistant_texts.append(item["text"])
        if isinstance(event.get("text"), str):
            assistant_texts.append(event["text"])
    if result_texts:
        return "\n".join(result_texts).strip()
    if assistant_texts:
        return "\n".join(assistant_texts).strip()
    return (stdout.strip() or stderr.strip()).strip()


def strip_json_fence(text: str) -> str:
    trimmed = text.strip()
    match = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", trimmed, re.IGNORECASE)
    return match.group(1).strip() if match else trimmed


def find_balanced_json_object(text: str) -> str | None:
    source = strip_json_fence(text)
    start = source.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(source)):
        char = source[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    return None


def parse_assistant_response(text: str) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    stripped = strip_json_fence(text)
    if stripped:
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed, warnings
            return {"answer": parsed}, warnings
        except json.JSONDecodeError:
            pass
    candidate = find_balanced_json_object(text)
    if candidate:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, warnings
            return {"answer": parsed}, warnings
        except json.JSONDecodeError as exc:
            warnings.append(f"assistant JSON parse failed: {exc}")
    warnings.append("assistant did not return a JSON object; used raw text as answer")
    return {"answer": stripped}, warnings


def validate_plan_optimization_response(
    task: dict[str, Any],
    response: dict[str, Any],
    parse_warnings: list[str],
) -> dict[str, Any]:
    warnings = list(parse_warnings)
    task_id = task["task_id"]
    if response.get("task_id") != task_id:
        warnings.append(f"task_id must equal {task_id}")

    analysis = response.get("optimization_analysis")
    if not isinstance(analysis, str) or not analysis.strip():
        warnings.append("optimization_analysis must be a non-empty string")
        analysis = "" if analysis is None else str(analysis)

    raw_plans = response.get("optimized_semantic_plan")
    optimized_plans: list[dict[str, str]] = []
    if not isinstance(raw_plans, list) or not raw_plans:
        warnings.append("optimized_semantic_plan must be a non-empty list")
    else:
        for index, plan in enumerate(raw_plans):
            if not isinstance(plan, dict):
                warnings.append(f"optimized_semantic_plan[{index}] must be an object")
                continue
            plan_type = plan.get("type")
            program = plan.get("program")
            if plan_type != "operator_dag":
                warnings.append(f"optimized_semantic_plan[{index}].type must be operator_dag")
            if not isinstance(program, str) or not program.strip():
                warnings.append(f"optimized_semantic_plan[{index}].program must be a non-empty string")
                continue
            optimized_plans.append(
                {
                    "type": str(plan_type or ""),
                    "program": program.strip(),
                }
            )

    current_plans = [
        {
            "type": str(plan.get("type") or ""),
            "program": str(plan.get("program") or "").strip(),
        }
        for plan in task.get("reference_semantic_plan", [])
        if isinstance(plan, dict)
    ]
    if optimized_plans and optimized_plans == current_plans:
        warnings.append("optimized_semantic_plan is unchanged from reference_semantic_plan")

    return {
        "task_id": task_id,
        "optimization_analysis": analysis.strip(),
        "optimized_semantic_plan": optimized_plans,
        "valid": not warnings,
        "validation_warnings": warnings,
    }


def resolve_answer_file(value: Any, workspace_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = Path(value.strip())
    candidates = [raw] if raw.is_absolute() else [ROOT_DIR / raw, workspace_dir / raw]
    workspace = workspace_dir.resolve()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            resolved.relative_to(workspace)
        except (OSError, ValueError):
            continue
        return resolved
    return None


def load_response_from_answer_file(task: dict[str, Any], response: dict[str, Any], paths: dict[str, Path]) -> tuple[dict[str, Any], str | None, list[str]]:
    warnings: list[str] = []
    task_id = task["task_id"]
    candidates: list[Path] = []
    for key in ("answer_file", "answer_path", "final_answer_file"):
        resolved = resolve_answer_file(response.get(key), paths["workspace_dir"])
        if resolved is not None:
            candidates.append(resolved)
        elif isinstance(response.get(key), str):
            warnings.append(f"{key} is outside workspace or invalid")
    candidates.append(task_answer_file_path(paths, task_id))
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.exists():
            continue
        try:
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"answer_file read failed: {candidate}: {exc}")
            continue
        if isinstance(loaded, dict):
            loaded.setdefault("task_id", task_id)
            if "evidence" not in loaded and "evidence" in response:
                loaded["evidence"] = response["evidence"]
            return loaded, str(candidate), warnings
        return {"task_id": task_id, "answer": loaded, "evidence": response.get("evidence", [])}, str(candidate), warnings
    return response, None, warnings


def extract_agent_usage(raw: dict[str, Any]) -> dict[str, int]:
    result_candidates: list[dict[str, int]] = []
    message_candidates: list[dict[str, int]] = []
    for event in iter_json_lines(raw.get("stdout", "")):
        if event.get("type") == "result":
            model_usage = usage_from_model_usage(event.get("modelUsage"))
            result_usage = usage_from_dict(event.get("usage")) if isinstance(event.get("usage"), dict) else None
            tokens = model_usage if model_usage and sum(model_usage.values()) else result_usage
            if tokens and sum(tokens.values()):
                result_candidates.append(tokens)
        stack: list[Any] = [event]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                usage = item.get("usage")
                if isinstance(usage, dict):
                    parsed = usage_from_dict(usage)
                    if any(parsed.values()):
                        message_candidates.append(parsed)
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
    candidates = result_candidates or message_candidates
    if not candidates:
        return {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
    return max(candidates, key=lambda item: sum(item.values()))


def read_tool_model_usage(tool_usage_path: Path | None) -> dict[str, Any]:
    summary = {
        "path": str(tool_usage_path) if tool_usage_path else None,
        "calls": 0,
        "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "cost_usd": 0.0,
    }
    if tool_usage_path is None or not tool_usage_path.exists():
        return summary
    for line in tool_usage_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        usage = usage_from_dict(record.get("usage", {}) if isinstance(record.get("usage"), dict) else record)
        summary["calls"] += 1
        summary["usage"] = add_usage(summary["usage"], usage)
        if isinstance(record.get("cost_usd"), (int, float)):
            summary["cost_usd"] += float(record["cost_usd"])
    summary["cost_usd"] = round(summary["cost_usd"], 10)
    return summary


def extract_run_metrics(raw: dict[str, Any], agent_env: dict[str, str], tool_usage_path: Path) -> dict[str, Any]:
    elapsed_ms = raw.get("elapsed_ms")
    elapsed_secs = round(float(elapsed_ms) / 1000, 3) if isinstance(elapsed_ms, (int, float)) else None
    agent_usage = extract_agent_usage(raw)
    tool_model_usage = read_tool_model_usage(tool_usage_path)
    usage = add_usage(agent_usage, tool_model_usage["usage"])
    agent_cost_usd = calculate_cost_usd(agent_usage, agent_env)
    tool_model_cost_usd = round(float(tool_model_usage["cost_usd"]), 10)
    cost_usd = round(agent_cost_usd + tool_model_cost_usd, 10)
    return {
        "elapsed_secs": elapsed_secs,
        "cost_usd": cost_usd,
        "agent_cost_usd": agent_cost_usd,
        "tool_model_cost_usd": tool_model_cost_usd,
        "cost_breakdown": {"agent_usd": agent_cost_usd, "tool_model_usd": tool_model_cost_usd, "total_usd": cost_usd},
        "usage": usage,
        "agent_usage": agent_usage,
        "tool_model_usage": tool_model_usage,
    }


def aggregate_execution_attempt_metrics(attempt_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not attempt_metrics:
        raise ValueError("execution metrics require at least one attempt")
    agent_usage = stored_usage({})
    agent_cost_usd = 0.0
    elapsed_secs = 0.0
    for metrics in attempt_metrics:
        agent_usage = add_usage(agent_usage, stored_usage(metrics.get("agent_usage")))
        agent_cost_usd += max(float(metrics.get("agent_cost_usd") or 0.0), 0.0)
        elapsed_secs += max(float(metrics.get("elapsed_secs") or 0.0), 0.0)

    # The tool usage file is cumulative across retries, so the latest summary already
    # contains every tool-model call and must not be summed once per attempt.
    latest_tool_summary = attempt_metrics[-1].get("tool_model_usage")
    tool_summary = dict(latest_tool_summary) if isinstance(latest_tool_summary, dict) else {}
    tool_usage = stored_usage(tool_summary.get("usage"))
    tool_model_cost_usd = max(float(tool_summary.get("cost_usd") or 0.0), 0.0)
    cost_usd = round(agent_cost_usd + tool_model_cost_usd, 10)
    tool_summary.update(
        {
            "usage": tool_usage,
            "cost_usd": round(tool_model_cost_usd, 10),
        }
    )
    return {
        "attempt_count": len(attempt_metrics),
        "elapsed_secs": round(elapsed_secs, 3),
        "cost_usd": cost_usd,
        "agent_cost_usd": round(agent_cost_usd, 10),
        "tool_model_cost_usd": round(tool_model_cost_usd, 10),
        "cost_breakdown": {
            "agent_usd": round(agent_cost_usd, 10),
            "tool_model_usd": round(tool_model_cost_usd, 10),
            "total_usd": cost_usd,
        },
        "usage": add_usage(agent_usage, tool_usage),
        "agent_usage": agent_usage,
        "tool_model_usage": tool_summary,
    }


def combine_optimization_and_execution_metrics(
    optimization: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, Any]:
    optimization_agent_usage = stored_usage(optimization.get("agent_usage"))
    execution_agent_usage = stored_usage(execution.get("agent_usage"))
    optimization_tool_usage = stored_usage(optimization.get("tool_model_usage"))
    execution_tool_summary = (
        execution.get("tool_model_usage")
        if isinstance(execution.get("tool_model_usage"), dict)
        else {}
    )
    execution_tool_usage = stored_usage(execution_tool_summary.get("usage"))
    agent_usage = add_usage(optimization_agent_usage, execution_agent_usage)
    tool_usage = add_usage(optimization_tool_usage, execution_tool_usage)
    agent_cost_usd = round(
        max(float(optimization.get("agent_cost_usd") or 0.0), 0.0)
        + max(float(execution.get("agent_cost_usd") or 0.0), 0.0),
        10,
    )
    tool_model_cost_usd = round(
        max(float(optimization.get("tool_model_cost_usd") or 0.0), 0.0)
        + max(float(execution.get("tool_model_cost_usd") or 0.0), 0.0),
        10,
    )
    cost_usd = round(agent_cost_usd + tool_model_cost_usd, 10)
    return {
        "elapsed_secs": round(
            max(float(optimization.get("elapsed_secs") or 0.0), 0.0)
            + max(float(execution.get("elapsed_secs") or 0.0), 0.0),
            3,
        ),
        "cost_usd": cost_usd,
        "agent_cost_usd": agent_cost_usd,
        "tool_model_cost_usd": tool_model_cost_usd,
        "cost_breakdown": {
            "plan_optimization_usd": round(float(optimization.get("cost_usd") or 0.0), 10),
            "execution_usd": round(float(execution.get("cost_usd") or 0.0), 10),
            "agent_usd": agent_cost_usd,
            "tool_model_usd": tool_model_cost_usd,
            "total_usd": cost_usd,
        },
        "usage": add_usage(agent_usage, tool_usage),
        "agent_usage": agent_usage,
        "tool_model_usage": {
            "path": execution_tool_summary.get("path"),
            "calls": int(execution_tool_summary.get("calls") or 0),
            "usage": tool_usage,
            "cost_usd": tool_model_cost_usd,
        },
    }


def normalize_response(task: dict[str, Any], response: dict[str, Any], parse_warnings: list[str], metrics: dict[str, Any], answer_file: str | None) -> dict[str, Any]:
    schema = task.get("answer_schema", {})
    answer = response.get("answer") if isinstance(response, dict) and "answer" in response else response
    schema_warnings: list[str] = []
    normalized = normalize_answer_by_schema(answer, schema, schema_warnings)
    return {
        "task_id": task["task_id"],
        "answer_schema": schema,
        "answer": normalized,
        "elapsed_secs": metrics["elapsed_secs"],
        "cost_usd": metrics["cost_usd"],
        "agent_cost_usd": metrics["agent_cost_usd"],
        "tool_model_cost_usd": metrics["tool_model_cost_usd"],
        "cost_breakdown": metrics["cost_breakdown"],
        "usage": metrics["usage"],
        "agent_usage": metrics["agent_usage"],
        "tool_model_usage": metrics["tool_model_usage"],
        "answer_file": answer_file,
        "evidence": normalize_evidence(response.get("evidence") if isinstance(response, dict) else None),
        "schema_valid": len(schema_warnings) == 0,
        "schema_warnings": schema_warnings,
        "parse_warnings": parse_warnings,
    }


def normalize_answer_by_schema(answer: Any, schema: Any, warnings: list[str]) -> Any:
    if not isinstance(schema, dict):
        warnings.append("answer_schema is not an object")
        return answer
    schema_type = schema.get("type")
    if "columns" in schema and ("types" in schema or schema_type == "table"):
        return normalize_table_answer(answer, schema, warnings)
    if schema_type == "scalar":
        return coerce_value(answer, schema.get("value_type"), warnings, "answer")
    if schema_type == "dictionary":
        return normalize_dictionary_answer(answer, schema, warnings)
    value_type = schema.get("value_type")
    if value_type is not None:
        return coerce_value(answer, value_type, warnings, "answer")
    warnings.append(f"unrecognized answer_schema shape: {schema}")
    return answer


def normalize_table_schema_columns(schema: dict[str, Any], warnings: list[str]) -> tuple[list[str], list[Any]]:
    columns = schema.get("columns", [])
    types = schema.get("types", [])
    if isinstance(columns, dict):
        return [str(key) for key in columns.keys()], [type_from_column_descriptor(value) for value in columns.values()]
    if not isinstance(columns, list):
        warnings.append("table schema columns is not a list")
        return [], []
    if not isinstance(types, list):
        warnings.append("table schema types is not a list")
        types = []
    return [str(column) for column in columns], types


def normalize_table_answer(answer: Any, schema: dict[str, Any], warnings: list[str]) -> list[dict[str, Any]]:
    columns, types = normalize_table_schema_columns(schema, warnings)
    if not columns:
        return []
    rows_source = answer
    if isinstance(answer, dict):
        for key in ("rows", "data", "answer", "results"):
            if isinstance(answer.get(key), list):
                rows_source = answer[key]
                break
        else:
            rows_source = [answer]
    if rows_source is None:
        return []
    if not isinstance(rows_source, list):
        warnings.append("table answer is not an array; wrapped one row")
        rows_source = [rows_source]
    rows: list[dict[str, Any]] = []
    for row_index, raw_row in enumerate(rows_source):
        row: dict[str, Any] = {}
        if isinstance(raw_row, dict):
            extras = sorted(set(str(key) for key in raw_row.keys()) - set(columns))
            if extras:
                warnings.append(f"row {row_index} has extra columns: {extras}")
            for col_index, column in enumerate(columns):
                if column not in raw_row:
                    warnings.append(f"row {row_index} missing column {column}")
                value_type = types[col_index] if col_index < len(types) else None
                row[column] = coerce_value(raw_row.get(column), value_type, warnings, f"answer[{row_index}].{column}")
        elif isinstance(raw_row, list):
            if len(raw_row) != len(columns):
                warnings.append(f"row {row_index} column count mismatch")
            for col_index, column in enumerate(columns):
                value = raw_row[col_index] if col_index < len(raw_row) else None
                value_type = types[col_index] if col_index < len(types) else None
                row[column] = coerce_value(value, value_type, warnings, f"answer[{row_index}].{column}")
        else:
            warnings.append(f"row {row_index} is not an object or array")
            for col_index, column in enumerate(columns):
                value_type = types[col_index] if col_index < len(types) else None
                row[column] = coerce_value(raw_row if col_index == 0 else None, value_type, warnings, f"answer[{row_index}].{column}")
        rows.append(row)
    return rows


def type_from_column_descriptor(descriptor: Any) -> str | None:
    text = str(descriptor or "").strip().lower()
    aliases = {
        "integer": "integer",
        "int": "integer",
        "float": "float",
        "number": "float",
        "double": "float",
        "boolean": "boolean",
        "bool": "boolean",
        "string": "string",
        "text": "string",
        "date": "date",
    }
    first = re.match(r"[a-z_]+", text)
    if first and first.group(0) in aliases:
        return aliases[first.group(0)]
    for alias, normalized in aliases.items():
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return normalized
    return None


def normalize_dictionary_answer(answer: Any, schema: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    keys = schema.get("keys", [])
    value_types = schema.get("value_types", [])
    raw = parse_json_string(answer) if isinstance(answer, str) else answer
    if not isinstance(raw, dict):
        warnings.append("dictionary answer is not an object")
        return {}
    if not isinstance(keys, list) or not keys:
        return raw
    result: dict[str, Any] = {}
    for index, key in enumerate(keys):
        value_type = value_types[index] if isinstance(value_types, list) and index < len(value_types) else None
        if key not in raw:
            warnings.append(f"dictionary answer missing key {key}")
        result[str(key)] = coerce_value(raw.get(key), value_type, warnings, f"answer.{key}")
    extras = sorted(set(str(key) for key in raw.keys()) - set(str(key) for key in keys))
    if extras:
        warnings.append(f"dictionary answer has extra keys: {extras}")
    return result


def parse_json_string(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def coerce_value(value: Any, value_type: Any, warnings: list[str], path: str) -> Any:
    if value_type is None:
        return value
    normalized_type = str(value_type).strip().lower()
    if normalized_type in {"", "any", "unknown"}:
        return value
    list_match = re.match(r"^(?:list|array)\[(.+)]$", normalized_type)
    if list_match:
        inner_type = list_match.group(1)
        items = parse_json_string(value) if isinstance(value, str) else value
        if items is None:
            return []
        if not isinstance(items, list):
            items = [items]
        return [coerce_value(item, inner_type, warnings, f"{path}[{index}]") for index, item in enumerate(items)]
    if normalized_type in {"integer", "int"}:
        if isinstance(value, bool):
            warnings.append(f"{path} expected integer, got boolean")
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if not value.is_integer():
                warnings.append(f"{path} expected integer, got non-integer float")
            return int(value)
        if isinstance(value, str):
            match = re.fullmatch(r"\s*-?\d+\s*", value.replace(",", ""))
            if match:
                return int(value.replace(",", "").strip())
        warnings.append(f"{path} expected integer")
        return None
    if normalized_type in {"float", "number", "double"}:
        if isinstance(value, bool):
            warnings.append(f"{path} expected float, got boolean")
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.replace(",", "").strip())
            except ValueError:
                pass
        warnings.append(f"{path} expected float")
        return None
    if normalized_type in {"boolean", "bool"}:
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in {0, 1}:
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "y", "1"}:
                return True
            if lowered in {"false", "no", "n", "0", "", "null", "none", "unknown", "n/a", "na"}:
                return False
        warnings.append(f"{path} expected boolean")
        return None
    if normalized_type in {"string", "text", "date"}:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)
    if normalized_type in {"dictionary", "dict", "object"}:
        parsed = parse_json_string(value) if isinstance(value, str) else value
        if isinstance(parsed, dict):
            return parsed
        warnings.append(f"{path} expected object")
        return {}
    warnings.append(f"{path} has unsupported value_type {value_type}")
    return value


def normalize_evidence(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item if isinstance(item, str) else json.dumps(item, ensure_ascii=False) for item in value]
    if isinstance(value, str):
        return [value]
    return [json.dumps(value, ensure_ascii=False)]


def bounded_retry_diagnostic(value: Any, limit: int, *, keep_tail: bool = True) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    retained = text[-limit:] if keep_tail else text[:limit]
    position = "tail" if keep_tail else "head"
    return f"[diagnostic truncated: omitted {omitted} chars; retained {position}; read artifact path for full content]\n{retained}"


def build_failure_diagnostics(
    raw: dict[str, Any],
    output: dict[str, Any],
    artifact_paths: dict[str, str | None],
    failure_reason: str,
) -> dict[str, Any]:
    script_run = raw.get("script_run") if isinstance(raw.get("script_run"), dict) else {}
    if script_run:
        diagnostic_source = "task-script"
        stderr = str(script_run.get("stderr") or "")
        stdout = str(script_run.get("stdout") or "")
    else:
        diagnostic_source = "Claude Code agent"
        stderr = str(raw.get("stderr") or "")
        stdout = str(raw.get("stdout") or "")

    output_format_failure = not bool(output.get("schema_valid")) or failure_reason == "answer_file was not found"
    include_stdout = not stderr.strip() or output_format_failure
    return {
        "failure_summary": failure_reason,
        "artifact_paths": artifact_paths,
        "diagnostic_source": diagnostic_source,
        "agent_exit_code": raw.get("agent_exit_code"),
        "agent_timed_out": raw.get("agent_timed_out"),
        "task_exit_code": raw.get("exit_code"),
        "task_timed_out": raw.get("timed_out"),
        "script_selection_error": raw.get("script_selection_error"),
        "policy_violation": raw.get("policy_violation"),
        "schema_warnings": output.get("schema_warnings", []),
        "parse_warnings": output.get("parse_warnings", []),
        "stderr_tail": bounded_retry_diagnostic(stderr, MAX_RETRY_STDERR_TAIL_CHARS),
        "stdout_tail": bounded_retry_diagnostic(stdout, MAX_RETRY_STDOUT_TAIL_CHARS)
        if include_stdout
        else "",
    }


def build_attempt_summary(
    attempt: int,
    raw: dict[str, Any],
    output: dict[str, Any],
    metrics: dict[str, Any],
    process_ok: bool,
    answer_file: str | None,
    ok: bool,
    artifact_paths: dict[str, str | None],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "attempt": attempt,
        "ok": ok,
        "process_ok": process_ok,
        "schema_valid": output.get("schema_valid"),
        "exit_code": raw.get("exit_code"),
        "timed_out": raw.get("timed_out"),
        "elapsed_secs": metrics.get("elapsed_secs"),
        "cost_usd": metrics.get("cost_usd"),
        "answer_file": answer_file,
    }
    tool_model_usage = metrics.get("tool_model_usage")
    if isinstance(tool_model_usage, dict):
        summary["tool_model_calls"] = tool_model_usage.get("calls")
        summary["tool_model_cost_usd"] = tool_model_usage.get("cost_usd")
    if output.get("schema_warnings"):
        summary["schema_warnings"] = output["schema_warnings"][:5]
    if output.get("parse_warnings"):
        summary["parse_warnings"] = output["parse_warnings"][:5]
    if raw.get("script_path"):
        summary["script_path"] = raw.get("script_path")
    if raw.get("script_selection_error"):
        summary["script_selection_error"] = raw.get("script_selection_error")
    if isinstance(raw.get("script_run"), dict):
        script_run = raw["script_run"]
        summary["script_run_exit_code"] = script_run.get("exit_code")
        summary["script_run_timed_out"] = script_run.get("timed_out")
    if isinstance(raw.get("preexisting_task_processes"), dict):
        summary["preexisting_task_processes"] = raw["preexisting_task_processes"]
    if isinstance(raw.get("lingering_task_processes"), dict):
        summary["lingering_task_processes"] = raw["lingering_task_processes"]
    if raw.get("timed_out"):
        summary["failure_reason"] = "task timed out"
    elif raw.get("script_lingering_processes"):
        summary["failure_reason"] = "task script left subprocesses running"
    elif raw.get("script_selection_error"):
        summary["failure_reason"] = "task script was not created or selected"
    elif not process_ok:
        summary["failure_reason"] = "agent or task script exited non-zero"
    elif not output.get("schema_valid"):
        summary["failure_reason"] = "answer did not match answer_schema"
    elif not answer_file:
        summary["failure_reason"] = "answer_file was not found"
    if not ok:
        failure_reason = str(summary.get("failure_reason") or "task attempt failed")
        summary["retry_diagnostics"] = build_failure_diagnostics(
            raw,
            output,
            artifact_paths,
            failure_reason,
        )
    return summary


def run_plan_optimization_tasks(
    *,
    all_tasks: list[dict[str, Any]],
    selected_tasks: list[dict[str, Any]],
    pending: list[dict[str, Any]],
    existing: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    paths: dict[str, Path],
    tool_env: dict[str, str],
    agent_env: dict[str, str],
    secrets: list[str],
    source_snapshot: dict[str, Any],
) -> int:
    for task in pending:
        assert_evaluation_input(args.input)
        assert_source_dirs_unchanged(
            source_snapshot,
            args.input,
            args.data_dir,
            include_hash=False,
            data_paths=args.dataset_include_paths,
        )
        task_id = task["task_id"]
        safe_task_id = safe_filename(task_id)
        attempt_summaries: list[dict[str, Any]] = []
        final_entry: dict[str, Any] | None = None
        task_session_id = str(uuid.uuid4()) if args.retry_session_mode == "resume" else None

        for attempt in range(1, args.task_max_attempts + 1):
            action = "optimizing" if attempt == 1 else "retrying"
            print(
                f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {action} {task_id} "
                f"attempt {attempt}/{args.task_max_attempts}",
                flush=True,
            )
            raw = run_agent_task(
                task,
                tool_env,
                agent_env,
                args,
                paths,
                secrets,
                attempt=attempt,
                max_attempts=args.task_max_attempts,
                previous_attempts=attempt_summaries,
                session_id=task_session_id,
                resume_session=args.retry_session_mode == "resume" and attempt > 1,
            )
            raw["agent_exit_code"] = raw.get("exit_code")
            raw["agent_timed_out"] = raw.get("timed_out")
            raw = redact_payload(raw, secrets)
            raw_file = paths["raw_dir"] / f"{safe_task_id}.json"
            raw_attempt_file = paths["raw_dir"] / f"{safe_task_id}.attempt-{attempt}.json"
            write_json(raw_file, raw)
            write_json(raw_attempt_file, raw)

            assistant_text = extract_assistant_text(raw.get("stdout", ""), raw.get("stderr", ""))
            response, parse_warnings = parse_assistant_response(assistant_text)
            output = validate_plan_optimization_response(task, response, parse_warnings)
            tool_usage_path = paths["tool_model_usage_dir"] / f"{safe_task_id}.jsonl"
            metrics = extract_run_metrics(raw, agent_env, tool_usage_path)
            output.update(
                {
                    "elapsed_secs": metrics["elapsed_secs"],
                    "cost_usd": metrics["cost_usd"],
                    "agent_cost_usd": metrics["agent_cost_usd"],
                    "usage": metrics["usage"],
                    "agent_usage": metrics["agent_usage"],
                }
            )
            output = redact_payload(output, secrets)
            output_file = paths["output_dir"] / f"{safe_task_id}.json"
            output_attempt_file = paths["output_dir"] / f"{safe_task_id}.attempt-{attempt}.json"
            write_json(output_file, output)
            write_json(output_attempt_file, output)

            process_ok = raw.get("exit_code") == 0 and raw.get("timed_out") is False
            ok = process_ok and bool(output.get("valid"))
            if raw.get("timed_out"):
                failure_reason = "Claude Code agent timed out"
            elif not process_ok:
                failure_reason = "Claude Code agent exited non-zero"
            elif not output.get("valid"):
                failure_reason = "plan optimization response failed validation"
            else:
                failure_reason = None
            attempt_summary = {
                "attempt": attempt,
                "ok": ok,
                "process_ok": process_ok,
                "exit_code": raw.get("exit_code"),
                "timed_out": raw.get("timed_out"),
                "elapsed_secs": metrics["elapsed_secs"],
                "cost_usd": metrics["cost_usd"],
                "validation_warnings": output.get("validation_warnings", []),
                "failure_reason": failure_reason,
                "raw_attempt_file": str(raw_attempt_file),
                "output_attempt_file": str(output_attempt_file),
            }
            attempt_summaries.append(attempt_summary)
            entry = {
                "task_id": task_id,
                "task_mode": "plan-optimization",
                "query": task.get("query", ""),
                "inputs": task.get("inputs", []),
                "reference_semantic_plan": task.get("reference_semantic_plan", []),
                "ok": ok,
                "attempt": attempt,
                "max_attempts": args.task_max_attempts,
                "attempts": attempt_summaries,
                "process_ok": process_ok,
                "exit_code": raw.get("exit_code"),
                "timed_out": raw.get("timed_out"),
                "elapsed_ms": raw.get("elapsed_ms"),
                "elapsed_secs": metrics["elapsed_secs"],
                "cost_usd": metrics["cost_usd"],
                "agent_cost_usd": metrics["agent_cost_usd"],
                "tool_model_cost_usd": metrics["tool_model_cost_usd"],
                "cost_breakdown": metrics["cost_breakdown"],
                "usage": metrics["usage"],
                "agent_usage": metrics["agent_usage"],
                "tool_model_usage": metrics["tool_model_usage"],
                "optimization_analysis": output.get("optimization_analysis", ""),
                "optimized_semantic_plan": output.get("optimized_semantic_plan", []),
                "validation_warnings": output.get("validation_warnings", []),
                "output_file": str(output_file),
                "raw_file": str(raw_file),
                "raw_attempt_file": str(raw_attempt_file),
                "output_attempt_file": str(output_attempt_file),
            }
            existing[task_id] = entry
            final_entry = entry
            write_results_jsonl(paths["results_jsonl"], all_tasks, existing)
            write_result_files(paths["results_dir"], all_tasks, existing)
            status = "ok" if ok else "failed"
            print(
                f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {task_id} "
                f"attempt {attempt}/{args.task_max_attempts} {status}",
                flush=True,
            )
            if ok:
                break

        if not final_entry or final_entry.get("ok") is not True:
            message = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {task_id} failed after {args.task_max_attempts} attempts"
            if args.stop_on_failure:
                print(f"{message}; stopping.", file=sys.stderr, flush=True)
                return 1
            print(f"{message}; skipping to next task.", file=sys.stderr, flush=True)

    assert_source_dirs_unchanged(
        source_snapshot,
        args.input,
        args.data_dir,
        include_hash=True,
        data_paths=args.dataset_include_paths,
    )
    selected_ids = {task.get("task_id") for task in selected_tasks}
    failed = [entry for task_id, entry in existing.items() if task_id in selected_ids and entry.get("ok") is not True]
    if failed:
        failed_ids = ", ".join(str(entry.get("task_id")) for entry in failed)
        print(
            f"Completed with {len(failed)} failed plan optimizations: {failed_ids}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(f"Completed plan optimization. Final results dir: {paths['results_dir']}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    args.input = args.input.resolve()
    args.data_dir = args.data_dir.resolve()
    args.results_root = args.results_root.resolve()
    args.run_dir = args.run_dir.resolve()
    args.plan_optimization_run_dir = args.plan_optimization_run_dir.resolve()
    if args.optimized_plan_selection is None:
        args.optimized_plan_selection = (
            args.plan_optimization_run_dir / DEFAULT_OPTIMIZED_PLAN_SELECTION_NAME
        )
    args.optimized_plan_selection = args.optimized_plan_selection.resolve()
    args.agent_env_path = args.agent_env_path.resolve()

    if args.tool_model_max_concurrency < 1 or args.tool_model_max_concurrency > 10:
        raise SystemExit("--tool-model-max-concurrency must be between 1 and 10")
    if args.tool_model_max_tokens < 1 or args.tool_model_max_tokens > DEFAULT_TOOL_MODEL_MAX_TOKENS:
        raise SystemExit(f"--tool-model-max-tokens must be between 1 and {DEFAULT_TOOL_MODEL_MAX_TOKENS}")
    if args.task_max_attempts < 1:
        raise SystemExit("--task-max-attempts must be at least 1")
    if args.timeout < 1:
        raise SystemExit("--timeout must be at least 1")
    if args.agent_timeout < 1:
        raise SystemExit("--agent-timeout must be at least 1")

    require_path("evaluation input", args.input)
    require_path("dataset directory", args.data_dir)
    require_path("agent env/config file", args.agent_env_path)
    require_under_root("evaluation input", args.input)
    require_under_root("dataset directory", args.data_dir)
    require_under_root("run dir", args.run_dir)
    require_under_root("results root", args.results_root)
    require_under_root("agent env/config file", args.agent_env_path)
    if args.task_mode == "optimized-plan-execution":
        require_path("plan optimization run", args.plan_optimization_run_dir)
        require_path("optimized plan selection", args.optimized_plan_selection)
        require_path(
            "plan optimization results",
            args.plan_optimization_run_dir / "results.jsonl",
        )
        require_under_root("plan optimization run", args.plan_optimization_run_dir)
        require_under_root("optimized plan selection", args.optimized_plan_selection)
        if path_is_under(args.plan_optimization_run_dir, args.run_dir) or path_is_under(
            args.run_dir,
            args.plan_optimization_run_dir,
        ):
            raise SystemExit(
                "--run-dir and --plan-optimization-run-dir must be disjoint; "
                "--clean must not be able to delete source optimization artifacts"
            )

    all_tasks, selected_tasks = load_tasks(
        args.input,
        None,
        None,
        task_mode=args.task_mode,
    )
    if args.task_mode == "optimized-plan-execution":
        prepared_tasks = prepare_optimized_plan_execution_tasks(
            all_tasks,
            args.plan_optimization_run_dir,
            args.optimized_plan_selection,
        )
        selected_tasks = select_testbed_tasks(prepared_tasks, args)
        all_tasks = prepared_tasks
    else:
        selected_tasks = select_testbed_tasks(all_tasks, args)
    skip_task_ids = normalize_task_ids(args.skip_task)
    if skip_task_ids:
        selected_task_ids = {task.get("task_id") for task in selected_tasks}
        unknown_skips = sorted(task_id for task_id in skip_task_ids if task_id not in selected_task_ids)
        if unknown_skips:
            print(f"[warning] skip task(s) not in selected input: {', '.join(unknown_skips)}", flush=True)
        selected_tasks = [task for task in selected_tasks if task.get("task_id") not in skip_task_ids]
    args.dataset_include_paths = (
        resolve_operator_task_inputs(selected_tasks, args.data_dir)
        if args.task_mode in {"operator-implementation", "optimized-plan-execution"}
        else ([] if args.task_mode == "plan-optimization" else None)
    )

    assert_evaluation_input(args.input)
    assert_no_source_scripts(args.input, args.data_dir, args.dataset_include_paths)
    source_snapshot = snapshot_source_dirs(
        args.input,
        args.data_dir,
        include_hash=True,
        data_paths=args.dataset_include_paths,
    )

    tool_env = parse_dotenv(DEFAULT_ENV_PATH)
    expand_env_references(tool_env)
    tool_env.update(os.environ)
    tool_env["LLM_MODEL_NAME"] = TEXT_MODEL
    tool_env["LLM_MAIN_MODEL"] = TEXT_MODEL
    missing = [key for key in ("OPENAI_API_KEY", "LLM_API_BASE", "LLM_MODEL_NAME") if not env_value_is_set(tool_env.get(key))]
    if missing and args.task_mode != "plan-optimization":
        raise SystemExit(f".env is missing required config: {', '.join(missing)}")

    agent_env = agent_env_with_process_overrides(parse_dotenv(args.agent_env_path))
    agent_env.setdefault("AGENT_API_BASE", tool_env.get("LLM_API_BASE", ""))
    agent_env.setdefault("AGENT_API_KEY", tool_env.get("OPENAI_API_KEY", ""))
    agent_env["AGENT_MAIN_MODEL"] = TEXT_MODEL
    synced_agent_mappings = sync_agent_env_mappings(agent_env)
    missing_agent = missing_env_groups(agent_env, AGENT_REQUIRED_ENV_GROUPS)
    if missing_agent:
        raise SystemExit("agent env/config is missing recognizable config: " + ", ".join(missing_agent))
    secrets = secret_values_from_envs(tool_env, agent_env)

    if args.clean:
        clean_run_dir(args.run_dir)
    paths = ensure_runtime_files(tool_env, agent_env, args)
    existing = read_existing_results(paths["results_jsonl"])
    for task_id, entry in read_compact_results(paths["results_dir"], all_tasks).items():
        existing.setdefault(task_id, entry)
    write_results_jsonl(paths["results_jsonl"], all_tasks, existing)
    write_result_files(paths["results_dir"], all_tasks, existing)

    if args.force:
        pending = selected_tasks
    elif args.task_mode in {"plan-optimization", "optimized-plan-execution"}:
        expected_mode = args.task_mode
        pending = [
            task
            for task in selected_tasks
            if existing.get(task.get("task_id"), {}).get("ok") is not True
            or existing.get(task.get("task_id"), {}).get("task_mode") != expected_mode
        ]
    else:
        pending = [
            task for task in selected_tasks if existing.get(task.get("task_id"), {}).get("ok") is not True
        ]
    print(f"Claude Code command: {args.claude_command}", flush=True)
    print(f"Agent env/config: {args.agent_env_path}", flush=True)
    if synced_agent_mappings:
        print(f"[agent-env] applied {len(synced_agent_mappings)} agent env mappings", flush=True)
    if env_value_is_set(agent_env.get("AGENT_MAIN_MODEL")):
        print(f"Agent model: {agent_env['AGENT_MAIN_MODEL']}", flush=True)
    if args.task_mode != "plan-optimization":
        print(f"Tool-side model: {tool_env.get('LLM_MODEL_NAME')}", flush=True)
    print(f"Task mode: {args.task_mode}", flush=True)
    print(f"Dataset: {args.data_dir}", flush=True)
    if args.task_mode in {"operator-implementation", "optimized-plan-execution"}:
        included = [path.relative_to(args.data_dir).as_posix() for path in args.dataset_include_paths]
        print(f"Selected data sources: {', '.join(included) if included else '<none>'}", flush=True)
    if args.task_mode == "plan-optimization":
        print("Plan execution: disabled", flush=True)
    elif args.task_mode == "optimized-plan-execution":
        print(f"Plan optimization source: {args.plan_optimization_run_dir}", flush=True)
        print(f"Four-type task selection: {args.optimized_plan_selection}", flush=True)
    print(f"Run dir: {paths['run_dir']}", flush=True)
    if args.retry_session_mode == "resume":
        print("Retry agent session mode: resume", flush=True)
    print(f"Selected tasks: {len(selected_tasks)}; pending: {len(pending)}", flush=True)
    if skip_task_ids:
        print(f"Skipped tasks: {', '.join(sorted(skip_task_ids))}", flush=True)

    if args.dry_run:
        for task in pending:
            print(f"{task['task_id']}: {task.get('query', '')}", flush=True)
        assert_source_dirs_unchanged(
            source_snapshot,
            args.input,
            args.data_dir,
            include_hash=True,
            data_paths=args.dataset_include_paths,
        )
        return 0

    if args.task_mode == "plan-optimization":
        return run_plan_optimization_tasks(
            all_tasks=all_tasks,
            selected_tasks=selected_tasks,
            pending=pending,
            existing=existing,
            args=args,
            paths=paths,
            tool_env=tool_env,
            agent_env=agent_env,
            secrets=secrets,
            source_snapshot=source_snapshot,
        )

    if pending:
        ensure_dataset_text_cache(paths, args, tool_env, secrets)

    for task in pending:
        assert_evaluation_input(args.input)
        assert_source_dirs_unchanged(
            source_snapshot,
            args.input,
            args.data_dir,
            include_hash=False,
            data_paths=args.dataset_include_paths,
        )
        task_id = task["task_id"]
        attempt_summaries: list[dict[str, Any]] = []
        execution_attempt_metrics: list[dict[str, Any]] = []
        final_entry: dict[str, Any] | None = None
        task_session_id = str(uuid.uuid4()) if args.retry_session_mode == "resume" else None
        for attempt in range(1, args.task_max_attempts + 1):
            if attempt == 1:
                print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] running {task_id}", flush=True)
            else:
                print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] retrying {task_id} attempt {attempt}/{args.task_max_attempts}", flush=True)
            preexisting_task_processes = handle_lingering_task_processes(
                paths,
                task_id,
                reason="before starting a new task attempt",
            )
            task_attempt_dir = task_attempt_tmp_dir(paths, task_id, attempt)
            shutil.rmtree(task_attempt_dir, ignore_errors=True)
            task_attempt_dir.mkdir(parents=True, exist_ok=True)
            before_script_mtimes = task_script_mtimes(task_attempt_dir)
            raw = run_agent_task(
                task,
                tool_env,
                agent_env,
                args,
                paths,
                secrets,
                attempt=attempt,
                max_attempts=args.task_max_attempts,
                previous_attempts=attempt_summaries,
                session_id=task_session_id,
                resume_session=args.retry_session_mode == "resume" and attempt > 1,
            )
            assistant_text = extract_assistant_text(raw.get("stdout", ""), raw.get("stderr", ""))
            response, parse_warnings = parse_assistant_response(assistant_text)
            script_path, script_selection_error = select_task_script(task_attempt_dir, response, before_script_mtimes)
            agent_exit_code = raw.get("exit_code")
            agent_timed_out = raw.get("timed_out")
            agent_process_ok = agent_exit_code == 0 and agent_timed_out is False
            raw["agent_exit_code"] = agent_exit_code
            raw["agent_timed_out"] = agent_timed_out
            raw["script_path"] = str(script_path) if script_path else None
            raw["script_selection_error"] = script_selection_error
            if agent_process_ok and script_path is not None and script_selection_error is None:
                print(f"[runner] executing {task_id} script in foreground: {script_path}", flush=True)
                script_run = run_task_script_foreground(
                    task_id=task_id,
                    script_path=script_path,
                    tool_env=tool_env,
                    args=args,
                    paths=paths,
                    secrets=secrets,
                )
                raw["script_run"] = script_run
                raw["script_run_exit_code"] = script_run.get("exit_code")
                raw["script_run_timed_out"] = script_run.get("timed_out")
                raw["exit_code"] = script_run.get("exit_code")
                raw["timed_out"] = script_run.get("timed_out")
                raw["elapsed_ms"] = to_nonnegative_int(raw.get("elapsed_ms")) + to_nonnegative_int(script_run.get("elapsed_ms"))
            elif agent_process_ok:
                raw["exit_code"] = 1
                raw["timed_out"] = False
                raw["script_run"] = None
            lingering_task_processes = handle_lingering_task_processes(
                paths,
                task_id,
                reason="task attempt finished or failed",
            )
            raw["preexisting_task_processes"] = preexisting_task_processes
            raw["lingering_task_processes"] = lingering_task_processes
            if lingering_task_processes.get("found") and raw.get("exit_code") == 0 and raw.get("timed_out") is False:
                raw["exit_code"] = 1
                raw["script_lingering_processes"] = True
            raw = redact_payload(raw, secrets)
            safe_task_id = safe_filename(task_id)
            raw_file = paths["raw_dir"] / f"{safe_task_id}.json"
            raw_attempt_file = paths["raw_dir"] / f"{safe_task_id}.attempt-{attempt}.json"
            write_json(raw_file, raw)
            write_json(raw_attempt_file, raw)

            tool_usage_path = paths["tool_model_usage_dir"] / f"{safe_task_id}.jsonl"
            metrics = extract_run_metrics(raw, agent_env, tool_usage_path)
            execution_attempt_metrics.append(metrics)
            response, answer_file, answer_file_warnings = load_response_from_answer_file(task, response, paths)
            parse_warnings.extend(answer_file_warnings)
            if script_selection_error:
                parse_warnings.append(script_selection_error)
            output = normalize_response(task, response, parse_warnings, metrics, answer_file)
            output = redact_payload(output, secrets)
            output_file = paths["output_dir"] / f"{safe_task_id}.json"
            output_attempt_file = paths["output_dir"] / f"{safe_task_id}.attempt-{attempt}.json"
            write_json(output_file, output)
            write_json(output_attempt_file, output)

            process_ok = raw.get("exit_code") == 0 and raw.get("timed_out") is False
            ok = bool(output.get("schema_valid")) and process_ok and bool(answer_file)
            attempt_summary = build_attempt_summary(
                attempt,
                raw,
                output,
                metrics,
                process_ok,
                answer_file,
                ok,
                {
                    "raw_attempt": str(raw_attempt_file),
                    "raw_latest": str(raw_file),
                    "normalized_output_attempt": str(output_attempt_file),
                    "normalized_output_latest": str(output_file),
                    "previous_script": str(script_path) if script_path else None,
                    "authoritative_answer": answer_file or str(task_answer_file_path(paths, task_id)),
                    "task_scratch_root": str(task_tmp_dir(paths, task_id)),
                },
            )
            attempt_summaries.append(attempt_summary)
            entry_metrics = metrics
            optimization_metrics: dict[str, Any] | None = None
            execution_metrics: dict[str, Any] | None = None
            if args.task_mode == "optimized-plan-execution":
                optimization_metrics = task.get("_plan_optimization")
                if not isinstance(optimization_metrics, dict):
                    raise SystemExit(f"missing plan optimization provenance for task: {task_id}")
                execution_metrics = aggregate_execution_attempt_metrics(execution_attempt_metrics)
                entry_metrics = combine_optimization_and_execution_metrics(
                    optimization_metrics,
                    execution_metrics,
                )
                output.update(
                    {
                        "optimization_types": task.get("_optimization_types", []),
                        "optimized_semantic_plan": task.get("optimized_semantic_plan", []),
                        "plan_optimization": optimization_metrics,
                        "execution": execution_metrics,
                        "elapsed_secs": entry_metrics["elapsed_secs"],
                        "cost_usd": entry_metrics["cost_usd"],
                        "agent_cost_usd": entry_metrics["agent_cost_usd"],
                        "tool_model_cost_usd": entry_metrics["tool_model_cost_usd"],
                        "cost_breakdown": entry_metrics["cost_breakdown"],
                        "usage": entry_metrics["usage"],
                        "agent_usage": entry_metrics["agent_usage"],
                        "tool_model_usage": entry_metrics["tool_model_usage"],
                    }
                )
                output = redact_payload(output, secrets)
                write_json(output_file, output)
                write_json(output_attempt_file, output)
            entry = {
                "task_id": task_id,
                "task_mode": args.task_mode,
                "query": task.get("query", ""),
                "answer_schema": task.get("answer_schema"),
                "ok": ok,
                "attempt": attempt,
                "max_attempts": args.task_max_attempts,
                "attempts": attempt_summaries,
                "process_ok": process_ok,
                "schema_valid": output["schema_valid"],
                "schema_warnings": output["schema_warnings"],
                "parse_warnings": output["parse_warnings"],
                "exit_code": raw.get("exit_code"),
                "timed_out": raw.get("timed_out"),
                "elapsed_ms": (
                    int(round(entry_metrics["elapsed_secs"] * 1000))
                    if args.task_mode == "optimized-plan-execution"
                    else raw.get("elapsed_ms")
                ),
                "elapsed_secs": entry_metrics["elapsed_secs"],
                "cost_usd": entry_metrics["cost_usd"],
                "agent_cost_usd": entry_metrics["agent_cost_usd"],
                "tool_model_cost_usd": entry_metrics["tool_model_cost_usd"],
                "cost_breakdown": entry_metrics["cost_breakdown"],
                "usage": entry_metrics["usage"],
                "agent_usage": entry_metrics["agent_usage"],
                "tool_model_usage": entry_metrics["tool_model_usage"],
                "script_path": raw.get("script_path"),
                "script_selection_error": raw.get("script_selection_error"),
                "preexisting_task_processes": raw.get("preexisting_task_processes"),
                "lingering_task_processes": raw.get("lingering_task_processes"),
                "answer_file": answer_file,
                "answer": output["answer"],
                "evidence": output["evidence"],
                "output_file": str(output_file),
                "raw_file": str(raw_file),
                "raw_attempt_file": str(raw_attempt_file),
                "output_attempt_file": str(output_attempt_file),
            }
            if args.task_mode == "optimized-plan-execution":
                assert optimization_metrics is not None and execution_metrics is not None
                entry.update(
                    {
                        "inputs": task.get("inputs", []),
                        "resolved_inputs": task.get("_resolved_inputs", []),
                        "optimization_types": task.get("_optimization_types", []),
                        "optimization_analysis": task.get("optimization_analysis", ""),
                        "optimized_semantic_plan": task.get("optimized_semantic_plan", []),
                        "plan_optimization": optimization_metrics,
                        "execution": execution_metrics,
                        "plan_optimization_cost_usd": optimization_metrics.get("cost_usd"),
                        "execution_cost_usd": execution_metrics.get("cost_usd"),
                    }
                )
            existing[task_id] = entry
            final_entry = entry
            write_results_jsonl(paths["results_jsonl"], all_tasks, existing)
            write_result_files(paths["results_dir"], all_tasks, existing)
            status = "ok" if ok else "failed"
            print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {task_id} attempt {attempt}/{args.task_max_attempts} {status}", flush=True)
            assert_evaluation_input(args.input)
            assert_source_dirs_unchanged(
                source_snapshot,
                args.input,
                args.data_dir,
                include_hash=False,
                data_paths=args.dataset_include_paths,
            )
            if ok:
                break
        if not final_entry or final_entry.get("ok") is not True:
            message = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {task_id} failed after {args.task_max_attempts} attempts"
            if args.stop_on_failure:
                print(f"{message}; stopping.", file=sys.stderr, flush=True)
                assert_source_dirs_unchanged(
                    source_snapshot,
                    args.input,
                    args.data_dir,
                    include_hash=True,
                    data_paths=args.dataset_include_paths,
                )
                return 1
            print(f"{message}; skipping to next task.", file=sys.stderr, flush=True)
            assert_source_dirs_unchanged(
                source_snapshot,
                args.input,
                args.data_dir,
                include_hash=False,
                data_paths=args.dataset_include_paths,
            )
            continue

    assert_evaluation_input(args.input)
    assert_source_dirs_unchanged(
        source_snapshot,
        args.input,
        args.data_dir,
        include_hash=True,
        data_paths=args.dataset_include_paths,
    )
    selected_ids = {task.get("task_id") for task in selected_tasks}
    failed = [entry for task_id, entry in existing.items() if task_id in selected_ids and entry.get("ok") is not True]
    if failed:
        failed_ids = ", ".join(str(entry.get("task_id")) for entry in failed[:20])
        suffix = " ..." if len(failed) > 20 else ""
        print(
            f"Completed with {len(failed)} failed or invalid selected tasks: {failed_ids}{suffix}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(f"Completed. Final results dir: {paths['results_dir']}", flush=True)
    return 0


if __name__ == "__main__":
    install_shutdown_handlers()
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        terminate_active_processes("keyboard interrupt")
        print("\nInterrupted. Runner-owned child processes were terminated.", file=sys.stderr, flush=True)
        raise SystemExit(130)
