#!/usr/bin/env python3
"""Run LOTUS testbed pipelines and persist compact results."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
TESTBED_ROOT = BASE_DIR.parents[1]
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


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _metadata(dataset: str) -> dict[str, dict]:
    path = TESTBED_ROOT / "queries" / dataset / "queries.json"
    return {
        item["task_id"]: item
        for item in json.loads(path.read_text(encoding="utf-8"))
    }


def _tasks(dataset: str, requested: list[str] | None) -> list[tuple[str, Path]]:
    discovered = []
    for path in sorted((BASE_DIR / dataset).glob(f"{dataset}_*.py")):
        match = re.fullmatch(
            rf"({re.escape(dataset)})_(\d{{3}})\.py",
            path.name,
        )
        if match:
            discovered.append((f"{dataset}-{match.group(2)}", path))
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


def _compact_result(path: Path, task_id: str, elapsed: float) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        "task_id": task_id,
        "answer": payload.get("answer"),
        "elapsed_seconds": round(
            float(payload.get("elapsed_seconds", elapsed)),
            2,
        ),
        "cost_usd": round(float(payload.get("cost_usd", 0.0)), 6),
        "total_tokens": int(payload.get("total_tokens", 0) or 0),
    }
    _atomic_write(path, compact)


def _run(command: list[str], env: dict[str, str], timeout: float) -> tuple[int, bool]:
    process = subprocess.Popen(
        command,
        cwd=BASE_DIR,
        env=env,
        start_new_session=True,
    )
    try:
        return process.wait(timeout=timeout), False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        return 124, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int)
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--task-timeout-hours", type=float, default=4.0)
    parser.add_argument("--auto-optimize-joins", action="store_true")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=TESTBED_ROOT / "results",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if not 0 < args.task_timeout_hours <= 4:
        parser.error("--task-timeout-hours must be in (0, 4]")
    metadata = _metadata(args.dataset)
    tasks = _tasks(args.dataset, args.task_ids)
    tasks = [
        (task_id, path)
        for task_id, path in tasks
        if args.start <= int(task_id.rsplit("-", 1)[1])
        and (args.end is None or int(task_id.rsplit("-", 1)[1]) <= args.end)
    ]
    if not tasks:
        parser.error("no pipelines matched the requested range")

    counts = {"success": 0, "skipped": 0, "failed": 0}
    timeout = args.task_timeout_hours * 3600.0
    for index, (task_id, script) in enumerate(tasks, start=1):
        result_dir = (
            args.results_root.resolve()
            / args.dataset
            / metadata[task_id]["operator_complexity"]
            / "lotus"
        )
        result_path = result_dir / f"{task_id}.json"
        if not args.force and _valid_result(result_path):
            counts["skipped"] += 1
            print(f"[SKIP] {task_id}", flush=True)
            continue

        result_dir.mkdir(parents=True, exist_ok=True)
        if result_path.exists():
            result_path.unlink()
        source = script.read_text(encoding="utf-8")
        env = os.environ.copy()
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "LOTUS_RESULTS_DIR": str(result_dir),
                "LOTUS_TASK_ID": task_id,
                "LOTUS_CACHE_DIR": str(BASE_DIR / ".cache" / args.dataset),
                "LOTUS_LLM_MAX_CONCURRENCY": str(args.concurrency),
                "LOTUS_JOIN_OPTIMIZATION_ENABLED": (
                    "1"
                    if args.auto_optimize_joins and ".sem_join(" in source
                    else "0"
                ),
            }
        )

        print(f"[RUN]  {task_id}", flush=True)
        started = time.time()
        return_code, timed_out = _run(
            [sys.executable, "-u", str(BASE_DIR / "execute_pipeline.py"), str(script)],
            env,
            timeout,
        )
        elapsed = time.time() - started
        if return_code == 0 and result_path.is_file():
            _compact_result(result_path, task_id, elapsed)
            counts["success"] += 1
            print(f"[DONE] {task_id} -> {result_path}", flush=True)
        else:
            _atomic_write(
                result_path,
                {
                    "task_id": task_id,
                    "answer": None,
                    "elapsed_seconds": round(timeout if timed_out else elapsed, 2),
                    "cost_usd": 0.0,
                    "total_tokens": 0,
                },
            )
            counts["failed"] += 1
            reason = "timeout" if timed_out else f"exit={return_code}"
            print(f"[FAIL] {task_id}: {reason}", flush=True)
        print(
            f"[PROGRESS] {index}/{len(tasks)} "
            f"success={counts['success']} skipped={counts['skipped']} "
            f"failed={counts['failed']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
