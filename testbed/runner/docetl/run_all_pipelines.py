#!/usr/bin/env python3
"""Run DocETL testbed pipelines and persist compact results."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
import litellm


BASE_DIR = Path(__file__).resolve().parent
TESTBED_ROOT = BASE_DIR.parents[1]
REPO_ROOT = BASE_DIR.parents[2]
BASELINE_DIR = REPO_ROOT / "baseline" / "docetl"
DATASETS = (
    "aviation_safety",
    "vehicle_safety",
    "finance",
    "legal_contracts",
)
RESULT_FIELDS = (
    "task_id",
    "answer",
    "elapsed_seconds",
    "cost_usd",
    "total_tokens",
)
INTERNAL_FIELDS = {"_counts_prereduce", "doc_format", "contents"}


load_dotenv(BASE_DIR.parent / ".env")
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
litellm.register_model(
    {
        "Qwen3.5-397B-A17B": {
            "max_tokens": 128000,
            "max_input_tokens": 128000,
            "max_output_tokens": 32768,
            "input_cost_per_token": 0.00000039,
            "output_cost_per_token": 0.00000234,
            "litellm_provider": "openai",
        },
        "Qwen3-Embedding-8B": {
            "max_tokens": 32768,
            "max_input_tokens": 32768,
            "input_cost_per_token": 0.00000001,
            "output_cost_per_token": 0.0,
            "litellm_provider": "openai",
            "mode": "embedding",
            "output_vector_size": 4096,
        },
    }
)
sys.path.insert(0, str(BASELINE_DIR))
if os.getenv("LLM_API_BASE") and not os.getenv("EMBEDDING_API_BASE"):
    os.environ["EMBEDDING_API_BASE"] = os.environ["LLM_API_BASE"]

_ENV_REFERENCE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_metadata(dataset: str) -> dict[str, dict]:
    query_path = TESTBED_ROOT / "queries" / dataset / "queries.json"
    gold_path = TESTBED_ROOT / "gold" / dataset / "queries.json"
    queries = json.loads(query_path.read_text(encoding="utf-8"))
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    metadata = {item["task_id"]: dict(item) for item in queries}
    for item in gold:
        metadata[item["task_id"]]["gold_answer"] = item["gold_answer"]
    return metadata


def _clean_record(record: dict, columns: list[str] | None = None) -> dict:
    clean = {
        key: value
        for key, value in record.items()
        if not key.startswith("_") and key not in INTERNAL_FIELDS
    }
    if not columns:
        return clean
    aliases = {"document_id": "doc_id", "doc_id": "document_id"}
    projected = {}
    for column in columns:
        if column in clean:
            projected[column] = clean[column]
        elif aliases.get(column) in clean:
            projected[column] = clean[aliases[column]]
    return projected


def _answer_from_raw(raw_output: list, gold_answer):
    if isinstance(gold_answer, list):
        columns = None
        if gold_answer and isinstance(gold_answer[0], dict):
            columns = list(gold_answer[0])
        return [
            _clean_record(item, columns)
            for item in raw_output
            if isinstance(item, dict)
        ]

    if isinstance(gold_answer, dict):
        if len(raw_output) == 1 and isinstance(raw_output[0], dict):
            return _clean_record(raw_output[0], list(gold_answer))
        return {}

    if isinstance(gold_answer, str):
        if len(raw_output) != 1 or not isinstance(raw_output[0], dict):
            return ""
        values = [
            value
            for value in _clean_record(raw_output[0]).values()
            if isinstance(value, str) and value
        ]
        return values[0] if values else ""

    if isinstance(gold_answer, (int, float)) and not isinstance(
        gold_answer, bool
    ):
        if len(raw_output) != 1 or not isinstance(raw_output[0], dict):
            return len(raw_output)
        record = raw_output[0]
        for key in ("count", "result", "value"):
            if isinstance(record.get(key), (int, float)):
                return record[key]
        values = [
            value
            for value in _clean_record(record).values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        return values[0] if values else 0

    if len(raw_output) == 1 and isinstance(raw_output[0], dict):
        return _clean_record(raw_output[0])
    return raw_output


def _substitute_environment(value):
    if isinstance(value, dict):
        return {
            key: _substitute_environment(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_substitute_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match) -> str:
        variable = match.group(1)
        configured = os.getenv(variable)
        if not configured:
            raise RuntimeError(
                f"{variable} is required; create testbed/runner/.env from "
                "testbed/runner/.env.example"
            )
        return configured

    return _ENV_REFERENCE.sub(replace, value)


def _load_pipeline_config(pipeline_path: Path) -> dict:
    config = yaml.safe_load(pipeline_path.read_text(encoding="utf-8")) or {}
    return _substitute_environment(config)


def _pipeline_output_path(config: dict, pipeline_path: Path) -> Path:
    path = config.get("pipeline", {}).get("output", {}).get("path")
    if not path:
        raise ValueError(f"pipeline output path is missing: {pipeline_path}")
    output_path = Path(path)
    return output_path if output_path.is_absolute() else BASE_DIR / output_path


def _pipeline_tasks(dataset: str, requested: list[str] | None) -> list[tuple[str, Path]]:
    pipeline_dir = BASE_DIR / dataset
    discovered = []
    for path in sorted(pipeline_dir.glob(f"pipeline_{dataset}-*.yaml")):
        match = re.fullmatch(
            rf"pipeline_({re.escape(dataset)}-(\d{{3}}))\.yaml",
            path.name,
        )
        if match:
            discovered.append((match.group(1), path))
    if requested:
        by_id = dict(discovered)
        missing = [task_id for task_id in requested if task_id not in by_id]
        if missing:
            raise ValueError(f"unknown task IDs: {', '.join(missing)}")
        return [(task_id, by_id[task_id]) for task_id in requested]
    return discovered


def _valid_result(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return set(payload) == set(RESULT_FIELDS) and payload["answer"] is not None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int)
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument("--max-threads", type=int, default=10)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=TESTBED_ROOT / "results",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.max_threads < 1:
        parser.error("--max-threads must be at least 1")
    metadata = _load_metadata(args.dataset)
    tasks = _pipeline_tasks(args.dataset, args.task_ids)
    tasks = [
        (task_id, path)
        for task_id, path in tasks
        if args.start <= int(task_id.rsplit("-", 1)[1])
        and (args.end is None or int(task_id.rsplit("-", 1)[1]) <= args.end)
    ]
    if not tasks:
        parser.error("no pipelines matched the requested range")

    os.chdir(BASE_DIR)
    from docetl.runner import DSLRunner

    counts = {"success": 0, "skipped": 0, "failed": 0}
    for index, (task_id, pipeline_path) in enumerate(tasks, start=1):
        task_meta = metadata[task_id]
        result_path = (
            args.results_root.resolve()
            / args.dataset
            / task_meta["operator_complexity"]
            / "docetl"
            / f"{task_id}.json"
        )
        if not args.force and _valid_result(result_path):
            counts["skipped"] += 1
            print(f"[SKIP] {task_id}", flush=True)
            continue

        print(f"[RUN]  {task_id}", flush=True)
        started = time.time()
        runner = None
        raw_path = None
        try:
            config = _load_pipeline_config(pipeline_path)
            raw_path = _pipeline_output_path(config, pipeline_path)
            if raw_path.exists():
                raw_path.unlink()
            runner = DSLRunner(
                config,
                max_threads=args.max_threads,
                base_name=str(pipeline_path.with_suffix("")),
                yaml_file_suffix=pipeline_path.stem,
            )
            runner.load_run_save()
            raw_output = json.loads(raw_path.read_text(encoding="utf-8"))
            metrics = runner.get_execution_metrics()
            totals = metrics.get("totals", {})
            payload = {
                "task_id": task_id,
                "answer": _answer_from_raw(
                    raw_output,
                    task_meta["gold_answer"],
                ),
                "elapsed_seconds": round(time.time() - started, 2),
                "cost_usd": round(float(runner.total_cost), 6),
                "total_tokens": int(
                    totals.get(
                        "total_tokens",
                        totals.get("prompt_tokens", 0)
                        + totals.get("completion_tokens", 0),
                    )
                ),
            }
            _atomic_write(result_path, payload)
            counts["success"] += 1
            print(f"[DONE] {task_id} -> {result_path}", flush=True)
        except Exception as exc:
            metrics = (
                runner.get_execution_metrics()
                if runner is not None
                else {}
            )
            totals = metrics.get("totals", {})
            payload = {
                "task_id": task_id,
                "answer": None,
                "elapsed_seconds": round(time.time() - started, 2),
                "cost_usd": round(
                    float(getattr(runner, "total_cost", 0.0)),
                    6,
                ),
                "total_tokens": int(totals.get("total_tokens", 0)),
            }
            _atomic_write(result_path, payload)
            counts["failed"] += 1
            print(f"[FAIL] {task_id}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if raw_path is not None:
                raw_path.unlink(missing_ok=True)
        print(
            f"[PROGRESS] {index}/{len(tasks)} "
            f"success={counts['success']} skipped={counts['skipped']} "
            f"failed={counts['failed']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
