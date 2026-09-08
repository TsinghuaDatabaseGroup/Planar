#!/usr/bin/env python3
"""
Run Planar tasks through an operator-native system using Claude Code.

The runner keeps the three-phase operator-system workflow:
0. Record or verify the target system environment with env-setup-skill.
1. Analyze the target system with new-system-operator-skill-creator and write a
   system-specific pipeline skill.
2. For each benchmark task, ask Claude Code to write a query plan, then execute
   that plan while using the target system's native operators/API.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shutil
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any


# Inline document-text CSV columns can exceed csv's default 128KB field limit.
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

ROOT_DIR = Path(__file__).resolve().parents[2]
RUNNER_DIR = ROOT_DIR / "runner"
SYSTEM_RUNNER_DIR = RUNNER_DIR / "lotus"
DEFAULT_RESULTS_ROOT = ROOT_DIR / "results"
DEFAULT_RUN_ROOT = Path(__file__).resolve().parent / "runs"
DEFAULT_SYSTEM_PATH = SYSTEM_RUNNER_DIR
DEFAULT_AGENT_ENV_PATH = RUNNER_DIR / ".env"
DEFAULT_ANALYSIS_SKILL = ROOT_DIR / "utils" / "new-system-operator-skill-creator" / "SKILL.md"
DEFAULT_ENV_SETUP_SKILL = ROOT_DIR / "utils" / "env-setup-skill" / "SKILL.md"
DEFAULT_ENV_PATH = RUNNER_DIR / ".env"
DEFAULT_PIPELINE_SKILL = Path(__file__).resolve().parent / "skills" / "lotus" / "SKILL.md"
DEFAULT_CLAUDE_COMMAND = "claude"
TEXT_MODEL = "Qwen3.5-397B-A17B"
DATASET_NAMES = ("aviation_safety", "vehicle_safety", "finance", "legal_contracts")
DEFAULT_TIMEOUT_SECS = 14400
DEFAULT_AGENT_EXIT_GRACE_SECS = 300
DATASET_TEXT_CACHE_VERSION = 3
PROCESS_TERMINATE_GRACE_SECS = 10

FORBIDDEN_BYPASS_PATTERNS = [
    r"\bmodel_call_helper\b",
    r"\bfrom\s+openai\b",
    r"\bimport\s+openai\b",
    r"\bfrom\s+anthropic\b",
    r"\bimport\s+anthropic\b",
    r"\bfrom\s+litellm\b",
    r"\bimport\s+litellm\b",
    r"\brequests\.post\s*\(",
    r"\burllib\.request\b",
    r"\bhttpx\.post\s*\(",
    r"/chat/completions",
    r"/responses",
]

RUNNER_ENV_MAPPINGS: dict[str, tuple[str, ...]] = {
    "LLM_API_BASE": ("OPENAI_API_BASE", "OPENAI_BASE_URL"),
    "LLM_API_KEY": ("OPENAI_API_KEY",),
    "LLM_MAIN_MODEL": ("LLM_MODEL_NAME",),
    "LLM_COST_UNIT": ("LLM_MODEL_COST_UNIT",),
    "LLM_INPUT_COST_PER_TOKEN": ("LLM_MODEL_USD_PER_INPUT_TOKEN",),
    "LLM_OUTPUT_COST_PER_TOKEN": ("LLM_MODEL_USD_PER_OUTPUT_TOKEN",),
    "LLM_CACHE_READ_COST_PER_TOKEN": ("LLM_MODEL_USD_PER_CACHE_READ_TOKEN",),
    "LLM_CACHE_WRITE_COST_PER_TOKEN": ("LLM_MODEL_USD_PER_CACHE_WRITE_TOKEN",),
    "LLM_MAX_CTX_TOKENS": ("LLM_MODEL_MAX_INPUT_TOKENS",),
    "LLM_MAX_OUTPUT_TOKENS": ("LLM_MODEL_MAX_OUTPUT_TOKENS",),
}
AGENT_ENV_MAPPINGS: dict[str, tuple[str, ...]] = {
    "AGENT_MAIN_MODEL": ("ANTHROPIC_MODEL", "LLM_MAIN_MODEL", "LLM_MODEL_NAME"),
    "AGENT_API_BASE": ("ANTHROPIC_BASE_URL", "LLM_API_BASE"),
    "AGENT_API_KEY": ("ANTHROPIC_API_KEY",),
    "AGENT_AUTH_TOKEN": ("ANTHROPIC_AUTH_TOKEN",),
    "LLM_COST_UNIT": ("LLM_MODEL_COST_UNIT",),
    "LLM_INPUT_COST_PER_TOKEN": ("LLM_MODEL_USD_PER_INPUT_TOKEN",),
    "LLM_OUTPUT_COST_PER_TOKEN": ("LLM_MODEL_USD_PER_OUTPUT_TOKEN",),
    "LLM_CACHE_READ_COST_PER_TOKEN": ("LLM_MODEL_USD_PER_CACHE_READ_TOKEN",),
    "LLM_CACHE_WRITE_COST_PER_TOKEN": ("LLM_MODEL_USD_PER_CACHE_WRITE_TOKEN",),
}
RUNNER_REQUIRED_ENV_GROUPS: dict[str, tuple[str, ...]] = {
    "API base": ("LLM_API_BASE", "OPENAI_API_BASE", "OPENAI_BASE_URL"),
    "API key": ("LLM_API_KEY", "OPENAI_API_KEY"),
    "model": ("LLM_MAIN_MODEL", "LLM_MODEL_NAME"),
}
AGENT_REQUIRED_ENV_GROUPS: dict[str, tuple[str, ...]] = {
    "agent model": ("AGENT_MAIN_MODEL", "ANTHROPIC_MODEL", "LLM_MAIN_MODEL", "LLM_MODEL_NAME"),
    "agent auth": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "AGENT_API_KEY", "AGENT_AUTH_TOKEN"),
}
ENV_RECORD_RECOMMENDED_GROUPS: dict[str, tuple[str, ...]] = {
    "max context tokens": ("LLM_MAX_CTX_TOKENS", "LLM_MODEL_MAX_INPUT_TOKENS"),
    "max output tokens": ("LLM_MAX_OUTPUT_TOKENS", "LLM_MODEL_MAX_OUTPUT_TOKENS"),
    "input cost per token": ("LLM_INPUT_COST_PER_TOKEN", "LLM_MODEL_USD_PER_INPUT_TOKEN"),
    "output cost per token": ("LLM_OUTPUT_COST_PER_TOKEN", "LLM_MODEL_USD_PER_OUTPUT_TOKEN"),
}

ENV_REFERENCE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
SECRET_KEY_RE = re.compile(
    r"(API_KEY|SECRET|PASSWORD|CREDENTIAL|AUTH_TOKEN|ACCESS_TOKEN|REFRESH_TOKEN|TOKEN)",
    re.I,
)
CLAUDE_PROVIDER_ENV_FLAGS = (
    "CLAUDE_CODE_USE_OPENAI",
    "CLAUDE_CODE_USE_GEMINI",
    "CLAUDE_CODE_USE_GROK",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
)
ACTIVE_PROCESSES_LOCK = threading.RLock()
ACTIVE_PROCESSES: dict[int, tuple[subprocess.Popen[str], str]] = {}
SHUTDOWN_STARTED = False


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Planar tasks through Claude Code and the native LOTUS operator system.",
    )
    parser.add_argument(
        "--mode",
        choices=("env-setup", "analyze-system", "run-tasks", "all"),
        default="run-tasks",
        help="env-setup writes ENV.md; analyze-system writes the pipeline skill; run-tasks executes benchmark tasks; all requires Phase 0 and runs Phase 1+2.",
    )
    parser.add_argument("--system-path", type=Path, default=DEFAULT_SYSTEM_PATH)
    parser.add_argument("--system-name")
    parser.add_argument("--system-package", help="Python package that task scripts must import; defaults to --system-name")
    parser.add_argument("--pipeline-skill-path", type=Path, default=DEFAULT_PIPELINE_SKILL)
    parser.add_argument("--analysis-skill", type=Path, default=DEFAULT_ANALYSIS_SKILL)
    parser.add_argument("--env-setup-skill", type=Path, default=DEFAULT_ENV_SETUP_SKILL)
    parser.add_argument(
        "--env-record",
        type=Path,
        help="Defaults to testbed/runner/lotus/ENV.md.",
    )
    parser.add_argument("--env-path", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument(
        "--agent-env-path",
        type=Path,
        default=DEFAULT_AGENT_ENV_PATH,
        help="Shared testbed model configuration; defaults to testbed/runner/.env.",
    )
    parser.add_argument("--system-env-note", help='Example: "conda activate lotus"')
    parser.add_argument("--allow-missing-env-record", action="store_true")
    parser.add_argument("--task")
    parser.add_argument(
        "--skip-task",
        action="append",
        default=[],
        help="Task ID(s) to skip in run-tasks mode. Repeat the flag or pass comma-separated IDs, e.g. --skip-task SEC10K-011,SEC10K-015.",
    )
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=positive_int, default=DEFAULT_TIMEOUT_SECS)
    parser.add_argument("--agent-exit-grace-secs", type=positive_int, default=DEFAULT_AGENT_EXIT_GRACE_SECS)
    parser.add_argument("--task-max-attempts", type=positive_int, default=3)
    parser.add_argument("--dataset", choices=DATASET_NAMES, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Persistent runtime directory; defaults to runner/claude_code_plus_lotus/runs/<dataset>.",
    )
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--rebuild-dataset-cache", action="store_true")
    parser.add_argument(
        "--claude-command",
        default=os.environ.get("CLAUDE_COMMAND", DEFAULT_CLAUDE_COMMAND),
        help="Claude Code CLI command; defaults to CLAUDE_COMMAND or 'claude'.",
    )
    parser.add_argument("--model-concurrency", type=positive_int, default=10)
    parser.add_argument("--show-agent-log", action="store_true", help="Print Claude Code stream-json summaries")
    parser.add_argument("--disable-bypass-scan", action="store_true")
    parser.add_argument("--disable-operator-use-scan", action="store_true")
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


def agent_env_with_process_overrides(file_env: dict[str, str]) -> dict[str, str]:
    env = dict(file_env)
    for key, value in os.environ.items():
        if key.startswith("ANTHROPIC_") or key.startswith("AGENT_"):
            env[key] = value
    return env


def env_value_is_set(value: str | None) -> bool:
    return bool(value) and ENV_REFERENCE_RE.search(value) is None


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


def sync_runner_env_mappings(env: dict[str, str]) -> list[str]:
    return sync_env_mappings(env, RUNNER_ENV_MAPPINGS)


def sync_agent_env_mappings(env: dict[str, str]) -> list[str]:
    return sync_env_mappings(env, AGENT_ENV_MAPPINGS)


def missing_runner_env_groups(env: dict[str, str]) -> list[str]:
    missing: list[str] = []
    for label, keys in RUNNER_REQUIRED_ENV_GROUPS.items():
        if not any(env_value_is_set(env.get(key)) for key in keys):
            missing.append(f"{label} ({' or '.join(keys)})")
    return missing


def missing_agent_env_groups(env: dict[str, str]) -> list[str]:
    missing: list[str] = []
    for label, keys in AGENT_REQUIRED_ENV_GROUPS.items():
        if not any(env_value_is_set(env.get(key)) for key in keys):
            missing.append(f"{label} ({' or '.join(keys)})")
    return missing


def require_env(env: dict[str, str], key: str) -> str:
    value = env.get(key)
    if not env_value_is_set(value):
        raise SystemExit(f"env/config is missing required runner config: {key}")
    return str(value)


def require_path(label: str, path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")


def require_under_root(label: str, path: Path) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT_DIR)
    except ValueError as exc:
        raise SystemExit(f"{label} must stay under workspace root: {resolved}") from exc


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def shell_quote(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_@%+=:,./-]+", value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def clean_stream_text(value: str) -> str:
    return ANSI_ESCAPE_RE.sub("", value).strip()


def first_present_key(env: dict[str, str], keys: tuple[str, ...]) -> str | None:
    return next((key for key in keys if env_value_is_set(env.get(key))), None)


def missing_env_groups(env: dict[str, str], groups: dict[str, tuple[str, ...]]) -> list[str]:
    missing: list[str] = []
    for label, keys in groups.items():
        if first_present_key(env, keys) is None:
            missing.append(f"{label} ({' or '.join(keys)})")
    return missing


def redact_env_value(key: str, value: str) -> str:
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
    if token_is_secret or SECRET_KEY_RE.search(upper_key):
        return "present, secret redacted"
    return value[:157] + "..." if len(value) > 160 else value


def secret_values(env: dict[str, str]) -> list[str]:
    values = []
    for key, value in env.items():
        if SECRET_KEY_RE.search(key) and value and len(value) >= 6:
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def text_tail(text: Any, limit: int = 4000) -> str:
    if not isinstance(text, str) or not text:
        return ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def infer_system_name(system_path: Path) -> str:
    name = safe_filename(system_path.resolve().name.lower())
    if not name:
        raise SystemExit(f"cannot infer system name from {system_path}")
    return name


def default_pipeline_skill_path(system_name: str) -> Path:
    if system_name == "lotus":
        return DEFAULT_PIPELINE_SKILL
    return Path(__file__).resolve().parent / "skills" / system_name / "SKILL.md"


def default_env_record_path(system_path: Path) -> Path:
    del system_path
    return SYSTEM_RUNNER_DIR / "ENV.md"


def needs_system_analysis(mode: str) -> bool:
    return mode in {"analyze-system", "all"}


def needs_task_run(mode: str) -> bool:
    return mode in {"run-tasks", "all"}


def markdown_list(lines: list[str]) -> str:
    return "\n".join(f"- {line}" for line in lines) if lines else "- None"


def env_field_lines(file_env: dict[str, str]) -> list[str]:
    if not file_env:
        return ["No fields found."]
    return [f"`{key}`: {redact_env_value(key, value)}" for key, value in sorted(file_env.items())]


def group_lines(env: dict[str, str], groups: dict[str, tuple[str, ...]]) -> list[str]:
    lines: list[str] = []
    for label, keys in groups.items():
        key = first_present_key(env, keys)
        if key:
            lines.append(f"{label}: `{key}` = `{redact_env_value(key, str(env[key]))}`")
        else:
            lines.append(f"{label}: missing (`{'` or `'.join(keys)}`)")
    return lines


def system_env_prefix(system_name: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9]+", "_", system_name).strip("_").upper()
    return prefix or "SYSTEM"


def concurrency_env_groups(system_name: str) -> dict[str, tuple[str, ...]]:
    prefix = system_env_prefix(system_name)
    return {
        "max model-call concurrency or batch size": (
            f"{prefix}_MAX_BATCH_SIZE",
            f"{prefix}_MAX_CONCURRENCY",
            "LLM_MODEL_MAX_BATCH_SIZE",
            "LLM_MAX_BATCH_SIZE",
            "LLM_MAX_CONCURRENCY",
            "MAX_BATCH_SIZE",
            "MAX_CONCURRENCY",
            "CONCURRENCY",
            "NUM_WORKERS",
        ),
        "request rate limit": (
            f"{prefix}_RATE_LIMIT_RPM",
            f"{prefix}_RPM_LIMIT",
            "LLM_RATE_LIMIT_RPM",
            "RATE_LIMIT_RPM",
            "RPM_LIMIT",
        ),
        "token rate limit": (f"{prefix}_TPM_LIMIT", "LLM_TPM_LIMIT", "TPM_LIMIT"),
    }


def print_env_setup_guidance(args: argparse.Namespace) -> None:
    print("[phase0] env setup should be run with env-setup-skill.")
    print(f"[phase0] system: {args.system_name} ({args.system_path})")
    print(f"[phase0] skill: {args.env_setup_skill}")
    print(f"[phase0] env/config: {args.env_path}")
    print(f"[phase0] expected record: {args.env_record}")
    if not args.env_record.exists():
        print("[phase0] ENV.md is missing. Run --mode env-setup first, or pass --allow-missing-env-record only for debugging.")


def require_env_record(args: argparse.Namespace) -> bool:
    if args.env_record.exists():
        return True
    print_env_setup_guidance(args)
    if args.allow_missing_env_record:
        print("[phase0] warning: continuing without ENV.md because --allow-missing-env-record was set.")
        return False
    raise SystemExit(f"Phase 0 is incomplete: missing {args.env_record}")


def build_import_probe(args: argparse.Namespace) -> tuple[list[str] | None, str, str | None]:
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(args.system_path)!r}); "
        f"import {args.system_package}; "
        f"pkg={args.system_package}; "
        "print(sys.executable); "
        "print(getattr(pkg, '__version__', 'ok'))"
    )
    note = (args.system_env_note or "").strip()
    match = re.search(r"\bconda\s+(?:activate|run\s+-n)\s+([A-Za-z0-9_.-]+)", note)
    if match:
        env_name = match.group(1)
        command = ["conda", "run", "-n", env_name, "python", "-c", code]
        return command, " ".join(shell_quote(part) for part in command), None
    if not note or note.startswith("current shell:"):
        command = [sys.executable, "-c", code]
        return command, " ".join(shell_quote(part) for part in command), None
    return None, f"<manual probe required after: {note}>", "activation note cannot be safely executed non-interactively"


def run_import_probe(args: argparse.Namespace, file_env: dict[str, str]) -> dict[str, Any]:
    command, display_command, skipped_reason = build_import_probe(args)
    if command is None:
        return {
            "command": display_command,
            "ok": False,
            "skipped": True,
            "warning": skipped_reason,
            "stdout": "",
            "stderr": "",
            "exit_code": None,
        }
    child_env = os.environ.copy()
    child_env.update(file_env)
    child_env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(args.system_path), str(ROOT_DIR), child_env.get("PYTHONPATH", "")) if path
    )
    try:
        completed = subprocess.run(
            command,
            cwd=str(ROOT_DIR),
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": display_command,
            "ok": False,
            "skipped": False,
            "warning": str(exc),
            "stdout": "",
            "stderr": "",
            "exit_code": None,
        }
    return {
        "command": display_command,
        "ok": completed.returncode == 0,
        "skipped": False,
        "warning": None if completed.returncode == 0 else "import probe failed",
        "stdout": clean_stream_text(completed.stdout),
        "stderr": clean_stream_text(completed.stderr),
        "exit_code": completed.returncode,
    }


def collect_env_setup_warnings(args: argparse.Namespace, file_env: dict[str, str], probe: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for missing in missing_env_groups(file_env, RUNNER_REQUIRED_ENV_GROUPS):
        warnings.append(f"missing required runner/model field: {missing}")
    for missing in missing_env_groups(file_env, ENV_RECORD_RECOMMENDED_GROUPS):
        warnings.append(f"missing recommended cost/token field: {missing}")
    concurrency_groups = concurrency_env_groups(args.system_name)
    concurrency_missing = missing_env_groups(file_env, concurrency_groups)
    if len(concurrency_missing) == len(concurrency_groups):
        prefix = system_env_prefix(args.system_name)
        warnings.append(
            "missing explicit concurrency/rate-limit fields: consider "
            f"{prefix}_MAX_BATCH_SIZE / {prefix}_RATE_LIMIT_RPM / {prefix}_TPM_LIMIT, "
            "or record the library default in ENV.md."
        )
    if not probe["ok"]:
        warnings.append("import probe did not pass; ENV.md should record the failure for repair.")
    return warnings


def build_env_setup_prompt(args: argparse.Namespace, *, file_env: dict[str, str], warnings: list[str], probe: dict[str, Any]) -> str:
    activation = args.system_env_note or f"current shell: {sys.executable}"
    return "\n".join(
        [
            "You are in Phase 0: record or verify one SQPE system environment by following env-setup-skill.",
            "",
            f"Workspace root: {ROOT_DIR}",
            f"System source path: {args.system_path}",
            f"System name: {args.system_name}",
            f"Target Python package/import root: {args.system_package}",
            f"Env-setup skill to follow: {args.env_setup_skill}",
            f"Required ENV.md output path: {args.env_record}",
            f"User-provided environment activation note: {activation}",
            f"User-provided env/config path: {args.env_path}",
            "",
            "The command-line arguments above are the user's confirmed non-interactive Phase 0 inputs. Do not ask the user to repeat them.",
            "",
            "Runner-collected redacted env/config fields:",
            markdown_list(env_field_lines(file_env)),
            "",
            "Required runner/model field status:",
            markdown_list(group_lines(file_env, RUNNER_REQUIRED_ENV_GROUPS)),
            "",
            "Recommended cost/token field status:",
            markdown_list(group_lines(file_env, ENV_RECORD_RECOMMENDED_GROUPS)),
            "",
            "Concurrency/rate-limit field status:",
            markdown_list(group_lines(file_env, concurrency_env_groups(args.system_name))),
            "",
            "Runner warnings:",
            markdown_list(warnings if warnings else ["None."]),
            "",
            "Import probe already performed by the runner:",
            f"- Command: `{probe['command']}`",
            f"- Exit code: `{probe.get('exit_code')}`",
            f"- OK: `{probe['ok']}`",
            f"- Skipped: `{probe.get('skipped')}`",
            f"- Warning: `{probe.get('warning')}`",
            "- stdout:",
            "```text",
            probe.get("stdout") or "",
            "```",
            "- stderr:",
            "```text",
            probe.get("stderr") or "",
            "```",
            "",
            "Hard requirements:",
            "- First read the env-setup-skill file at the path above and follow it as the governing workflow.",
            "- Also inspect the target system's actual source/docs/examples before writing ENV.md. At minimum find the package's documented model constructor, global settings/configure API, and usage/cost tracking API.",
            "- Treat this as a fast path for an existing environment. Do not install dependencies, repair packages, rewrite env/config files, or create aliases unless the prompt explicitly asks. It does not.",
            "- Write the ENV.md handoff exactly at the required output path.",
            "- Use actual env/config field names. Do not invent aliases or normalize names just for this runner.",
            "- Redact secrets in ENV.md. Do not include API key values. Do not redact non-secret token/cost/context/output/concurrency values such as max token counts, per-token prices, max batch size, RPM, or TPM.",
            "- Record provider/model adaptation, model-call kwargs, concurrency/rate-limit policy, and cost/token tracking only as discovered from env/config plus the target system source/docs/examples.",
            "- The Native setup section is invalid if it only assigns env vars to local variables. It must show the target system's native imports and configuration call(s), for example the system's model constructor and settings/configure API if those exist.",
            "- The Cost/token tracking section must name where downstream scripts read accumulated system usage/cost after a run, or explicitly record that no native source was found after inspecting source/docs.",
            "- If the import probe stdout includes a resolved Python executable path, record it and prefer `<resolved-python-executable> -u <pipeline.py>` as the downstream non-interactive launch form.",
            "- Include a Native setup section with the system's native setup pattern.",
            "- Avoid live model endpoint calls unless needed to verify a documented setup helper.",
            "- Include a short Evidence / inspected files note listing the source/docs files used for the native setup and cost tracking conclusions.",
            "- Return exactly one JSON object and no markdown after writing ENV.md.",
            "",
            "Return JSON shape:",
            "{",
            f'  "system_name": "{args.system_name}",',
            f'  "env_record": "{args.env_record}",',
            '  "summary": "short summary of what was recorded",',
            '  "warnings": ["warnings or caveats"]',
            "}",
        ]
    )


COST_UNIT_PER_TOKEN = "per_token"
COST_UNIT_PER_MILLION = "per_million"
COST_UNIT_ALIASES: dict[str, str] = {
    "per_token": COST_UNIT_PER_TOKEN,
    "usd_per_token": COST_UNIT_PER_TOKEN,
    "token": COST_UNIT_PER_TOKEN,
    "per_million": COST_UNIT_PER_MILLION,
    "per_million_tokens": COST_UNIT_PER_MILLION,
    "usd_per_million_tokens": COST_UNIT_PER_MILLION,
    "per_mtok": COST_UNIT_PER_MILLION,
    "usd_per_mtok": COST_UNIT_PER_MILLION,
    "mtok": COST_UNIT_PER_MILLION,
    "million": COST_UNIT_PER_MILLION,
}
_COST_UNIT_WARNINGS_EMITTED: set[str] = set()


def _warn_cost_unit_once(key: str, message: str) -> None:
    if key in _COST_UNIT_WARNINGS_EMITTED:
        return
    _COST_UNIT_WARNINGS_EMITTED.add(key)
    print(f"[cost][warning] {message}", file=sys.stderr, flush=True)


def resolve_cost_unit(env: dict[str, str]) -> str | None:
    raw = env.get("LLM_COST_UNIT") or env.get("LLM_MODEL_COST_UNIT")
    if not env_value_is_set(raw):
        return None
    normalized = COST_UNIT_ALIASES.get(str(raw).strip().lower())
    if normalized is None:
        _warn_cost_unit_once(
            f"unknown:{raw}",
            f"LLM_COST_UNIT={raw!r} is not recognized (expected one of: "
            f"{', '.join(sorted(set(COST_UNIT_ALIASES)))}); falling back to magnitude heuristic.",
        )
    return normalized


def normalize_million_token_cost(value: Any, unit: str | None = None) -> float:
    """Return the USD-per-million-tokens rate for a configured price value.

    An explicit `unit` (from the optional LLM_COST_UNIT) always wins: per_token
    multiplies by 1e6, per_million passes through. Without it, the config field
    names (*_COST_PER_TOKEN / *_USD_PER_*_TOKEN) promise per-token values, so
    small values (< 0.01) convert silently. Values >= 0.01 would be absurd
    per-token prices (>= $10,000/M); they keep the historical tolerance of being
    read as already-per-million, with a one-time warning because that reading
    contradicts the field name.
    """
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return 0.0
    if parsed < 0:
        return 0.0
    if unit == COST_UNIT_PER_TOKEN:
        return parsed * 1_000_000
    if unit == COST_UNIT_PER_MILLION:
        return parsed
    if 0 < parsed < 0.01:
        return parsed * 1_000_000
    if parsed >= 0.01:
        _warn_cost_unit_once(
            f"per_million_guess:{parsed}",
            f"cost value {parsed} in a per-token config field looks like a USD-per-million price "
            "and is treated as per-million; set LLM_COST_UNIT=per_million to confirm, or "
            "LLM_COST_UNIT=per_token if it really is a per-token price.",
        )
    return parsed


def to_float(value: Any, fallback: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed >= 0 else fallback


def to_int(value: Any, fallback: int = 0) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed >= 0 else fallback


def round_cost(value: Any) -> float:
    return round(to_float(value), 10)


def round_seconds(value: Any) -> float:
    return round(to_float(value), 3)


def calculate_cost_usd(tokens: dict[str, int], env: dict[str, str]) -> dict[str, float]:
    cost_unit = resolve_cost_unit(env)
    input_rate = normalize_million_token_cost(env.get("LLM_INPUT_COST_PER_TOKEN"), cost_unit)
    output_rate = normalize_million_token_cost(env.get("LLM_OUTPUT_COST_PER_TOKEN"), cost_unit)
    cache_read_rate = normalize_million_token_cost(env.get("LLM_CACHE_READ_COST_PER_TOKEN"), cost_unit) or input_rate
    cache_write_rate = normalize_million_token_cost(env.get("LLM_CACHE_WRITE_COST_PER_TOKEN"), cost_unit) or output_rate

    input_cost = tokens.get("input", 0) * input_rate / 1_000_000
    output_cost = tokens.get("output", 0) * output_rate / 1_000_000
    cache_read_cost = tokens.get("cache_read", 0) * cache_read_rate / 1_000_000
    cache_write_cost = tokens.get("cache_write", 0) * cache_write_rate / 1_000_000
    total = input_cost + output_cost + cache_read_cost + cache_write_cost
    return {
        "input": round_cost(input_cost),
        "output": round_cost(output_cost),
        "cache_read": round_cost(cache_read_cost),
        "cache_write": round_cost(cache_write_cost),
        "total": round_cost(total),
    }


def empty_agent_usage() -> dict[str, Any]:
    return {
        "available": False,
        "source": "none",
        "calls": 0,
        "tokens": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
        "cost_usd": {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0, "total": 0.0},
    }


def empty_system_usage(source: str = "none") -> dict[str, Any]:
    return {
        "available": False,
        "source": source,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
    }


def usage_from_dict(raw: dict[str, Any]) -> dict[str, int]:
    tokens = {
        "input": to_int(raw.get("input") or raw.get("prompt_tokens") or raw.get("input_tokens") or raw.get("inputTokens")),
        "output": to_int(raw.get("output") or raw.get("completion_tokens") or raw.get("output_tokens") or raw.get("outputTokens")),
        "cache_read": to_int(
            raw.get("cacheRead")
            or raw.get("cache_read")
            or raw.get("cache_read_input_tokens")
            or raw.get("cacheReadInputTokens")
        ),
        "cache_write": to_int(
            raw.get("cacheWrite")
            or raw.get("cache_write")
            or raw.get("cache_creation_input_tokens")
            or raw.get("cacheCreationInputTokens")
        ),
    }
    tokens["total"] = to_int(raw.get("totalTokens") or raw.get("total_tokens")) or sum(tokens.values())
    return tokens


def usage_from_model_usage(raw: Any) -> dict[str, int] | None:
    if not isinstance(raw, dict):
        return None

    tokens = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0}
    saw_usage = False
    for value in raw.values():
        if not isinstance(value, dict):
            continue
        saw_usage = True
        tokens["input"] += to_int(value.get("inputTokens") or value.get("input_tokens") or value.get("input"))
        tokens["output"] += to_int(value.get("outputTokens") or value.get("output_tokens") or value.get("output"))
        tokens["cache_read"] += to_int(
            value.get("cacheReadInputTokens") or value.get("cache_read_input_tokens") or value.get("cache_read")
        )
        tokens["cache_write"] += to_int(
            value.get("cacheCreationInputTokens")
            or value.get("cache_creation_input_tokens")
            or value.get("cache_write")
        )

    tokens["total"] = sum(tokens.values())
    if not saw_usage or tokens["total"] == 0:
        return None
    return tokens


def normalize_system_usage(raw: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return empty_system_usage(source)

    nested = raw.get("tokens") if isinstance(raw.get("tokens"), dict) else {}
    input_tokens = to_int(
        raw.get("prompt_tokens")
        or raw.get("input_tokens")
        or raw.get("prompt")
        or raw.get("input")
        or nested.get("prompt_tokens")
        or nested.get("input_tokens")
        or nested.get("prompt")
        or nested.get("input")
    )
    output_tokens = to_int(
        raw.get("completion_tokens")
        or raw.get("output_tokens")
        or raw.get("completion")
        or raw.get("output")
        or nested.get("completion_tokens")
        or nested.get("output_tokens")
        or nested.get("completion")
        or nested.get("output")
    )
    total_tokens = to_int(
        raw.get("total_tokens")
        or raw.get("total_token")
        or raw.get("totalTokens")
        or raw.get("total")
        or nested.get("total_tokens")
        or nested.get("total_token")
        or nested.get("totalTokens")
        or nested.get("total")
    )
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    cost_usd = round_cost(
        raw.get("cost_usd")
        or raw.get("total_cost_usd")
        or raw.get("system_cost_usd")
        or raw.get("total_cost")
        or 0.0
    )
    return {
        "available": bool(input_tokens or output_tokens or total_tokens or cost_usd),
        "source": str(raw.get("source") or source),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
    }


def system_usage_from_output(output: dict[str, Any] | None) -> dict[str, Any]:
    if not output:
        return empty_system_usage()
    for key in ("system_usage", "system_token_usage", "system_tokens", "usage"):
        if isinstance(output.get(key), dict):
            usage = normalize_system_usage(output[key], source=f"output.{key}")
            if usage["available"]:
                return usage
    direct = normalize_system_usage(output, source="output")
    if direct["available"] and direct["total_tokens"]:
        return direct
    return empty_system_usage()


def extract_structured_system_usage(stdout: str) -> dict[str, Any]:
    usage = empty_system_usage("stdout.structured")
    pattern = re.compile(r"SQPE_SYSTEM_USAGE_JSON\s*=\s*(\{[^\n]*\})")
    for match in pattern.finditer(stdout):
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        normalized = normalize_system_usage(parsed, source="stdout.SQPE_SYSTEM_USAGE_JSON")
        if normalized["available"]:
            usage = normalized
    return usage


def extract_lotus_print_usage(stdout: str) -> dict[str, Any]:
    physical_tokens_matches = re.findall(r"Physical Tokens:\s*([0-9,]+)", stdout)
    physical_cost_matches = re.findall(r"Physical Cost:\s*\$?([0-9,]+(?:\.[0-9]+)?)", stdout)
    if not physical_tokens_matches and not physical_cost_matches:
        return empty_system_usage("stdout.lotus_print_total_usage")
    total_tokens = to_int(physical_tokens_matches[-1].replace(",", "")) if physical_tokens_matches else 0
    cost_usd = round_cost(physical_cost_matches[-1].replace(",", "")) if physical_cost_matches else 0.0
    return {
        "available": bool(total_tokens or cost_usd),
        "source": "stdout.lotus_print_total_usage",
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
    }


def system_usage_from_execution(output: dict[str, Any] | None, execute_raw: dict[str, Any] | None) -> dict[str, Any]:
    usage = system_usage_from_output(output)
    stdout_candidates: list[str] = []
    if execute_raw and isinstance(execute_raw.get("stdout"), str):
        stdout_candidates.append(execute_raw["stdout"])
    script_run = execute_raw.get("script_run") if execute_raw else None
    if isinstance(script_run, dict) and isinstance(script_run.get("stdout"), str):
        stdout_candidates.append(script_run["stdout"])

    for stdout in stdout_candidates:
        if not usage["available"]:
            usage = extract_structured_system_usage(stdout)
        if not usage["available"]:
            usage = extract_lotus_print_usage(stdout)

    output_cost = to_float(output.get("cost_usd")) if output else 0.0
    if usage["available"] and usage["cost_usd"] == 0.0 and output_cost > 0:
        usage["cost_usd"] = round_cost(output_cost)
    if not usage["available"] and output_cost > 0:
        usage = empty_system_usage("output.cost_usd")
        usage["available"] = True
        usage["cost_usd"] = round_cost(output_cost)
    return usage


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


def extract_claude_usage(stdout: str, env: dict[str, str]) -> dict[str, Any]:
    message_candidates: list[dict[str, int]] = []
    result_candidates: list[tuple[dict[str, int], str]] = []
    events = iter_json_lines(stdout)
    for event in events:
        if event.get("type") == "result":
            model_usage = usage_from_model_usage(event.get("modelUsage"))
            result_usage = usage_from_dict(event.get("usage")) if isinstance(event.get("usage"), dict) else None
            tokens = model_usage if model_usage and model_usage["total"] else result_usage
            if tokens and tokens["total"]:
                source = "claude_result_model_usage" if model_usage else "claude_result_usage"
                result_candidates.append((tokens, source))

        stack: list[Any] = [event]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                usage = item.get("usage")
                if isinstance(usage, dict):
                    parsed = usage_from_dict(usage)
                    if parsed["total"] or parsed["input"] or parsed["output"]:
                        message_candidates.append(parsed)
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)

    if result_candidates:
        tokens, source = max(result_candidates, key=lambda item: item[0]["total"])
        cost_usd = calculate_cost_usd(tokens, env)
        return {
            "available": True,
            "source": source,
            "calls": len(message_candidates) or len(result_candidates),
            "tokens": tokens,
            "cost_usd": cost_usd,
        }

    if not message_candidates:
        return empty_agent_usage()
    tokens = max(message_candidates, key=lambda item: item["total"] or sum(item.values()))
    return {
        "available": True,
        "source": "claude_stream_json",
        "calls": len(message_candidates),
        "tokens": tokens,
        "cost_usd": calculate_cost_usd(tokens, env),
    }


def agent_phase_metrics(phase: str, raw: dict[str, Any] | None, *, raw_file: Path | None = None) -> dict[str, Any]:
    raw = raw or {}
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else empty_agent_usage()
    elapsed_ms = to_int(raw.get("elapsed_ms"))
    return {
        "phase": phase,
        "elapsed_ms": elapsed_ms,
        "elapsed_seconds": round_seconds(elapsed_ms / 1000),
        "agent_cost_usd": round_cost(get_nested_number(usage, ("cost_usd", "total"))),
        "agent_usage": usage,
        "exit_code": raw.get("exit_code"),
        "timed_out": raw.get("timed_out"),
        "session_id": raw.get("session_id"),
        "raw_file": str(raw_file) if raw_file else None,
    }


def get_nested_number(payload: dict[str, Any], keys: tuple[str, ...], fallback: float = 0.0) -> float:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return fallback
        current = current.get(key)
    return to_float(current, fallback)


def dataset_cache_files(data_dir: Path) -> list[Path]:
    return sorted(path for path in data_dir.rglob("*") if path.is_file())


def dataset_file_record(data_dir: Path, path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "relative_path": path.relative_to(data_dir).as_posix(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def dataset_cache_manifest_fingerprint(data_dir: Path) -> list[dict[str, Any]]:
    return [dataset_file_record(data_dir, path) for path in dataset_cache_files(data_dir)]


TEXT_PATH_COLUMN_CANDIDATES = (
    "text",
    "text_path",
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
    "id",
    "accession",
    "accession_number",
)
INLINE_TEXT_MIN_AVG_CHARS = 200
JSON_RECORD_SAMPLE_LIMIT = 50
JSON_RECORD_FILE_SUFFIXES = {".jsonl", ".ndjson"}
JSON_RECORD_MAX_ARRAY_BYTES = 256 * 1024 * 1024


def normalize_csv_key(key: str | None) -> str:
    return (key or "").lstrip("\ufeff").strip()


def normalize_csv_row(row: dict[str | None, str | None]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in row.items():
        normalized_key = normalize_csv_key(key)
        if not normalized_key:
            continue
        normalized[normalized_key] = "" if value is None else str(value).strip()
    return normalized


def resolve_dataset_reference(data_dir: Path, csv_path: Path, value: str) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    # Inline document text is never a path: long values or embedded newlines/NULs
    # would raise ENAMETOOLONG/ValueError from the filesystem probes below.
    if len(raw) > 1024 or "\n" in raw or "\r" in raw or "\x00" in raw:
        return None
    candidates: list[Path]
    candidate = Path(raw)
    if candidate.is_absolute():
        candidates = [candidate]
    else:
        candidates = [data_dir / candidate, csv_path.parent / candidate]
    for path in candidates:
        try:
            resolved = path.resolve()
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
            reader = csv.DictReader(handle)
            return sum(1 for _ in reader)
    except Exception:
        return 0


def detect_csv_text_table(data_dir: Path, csv_path: Path) -> dict[str, Any] | None:
    fieldnames, sample_rows = read_csv_sample(csv_path)
    if not fieldnames or not sample_rows:
        return None
    candidate_columns = [
        column
        for column in fieldnames
        if column.lower() in TEXT_PATH_COLUMN_CANDIDATES or "text" in column.lower()
    ]
    best_column = None
    best_valid = 0
    best_nonempty = 0
    for column in candidate_columns:
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
    if not best_column or best_valid == 0:
        return None
    threshold = max(1, min(best_nonempty, len(sample_rows)) // 2)
    if best_valid < threshold:
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
    """Detect CSVs that carry document text inline in one column instead of file paths."""
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
    """Detect JSONL/NDJSON streams and JSON arrays made of object records."""
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
    """Split dataset files into cache source groups.

    Priority per CSV: text-path-reference table > inline-text table > metadata-only
    table. JSONL/NDJSON streams and JSON arrays of objects become record
    collections. Everything else stays a per-file document.
    """
    csv_text_path_tables: list[dict[str, Any]] = []
    csv_inline_text_tables: list[dict[str, Any]] = []
    csv_metadata_tables: list[dict[str, Any]] = []
    json_record_files: list[dict[str, Any]] = []
    container_paths: set[Path] = set()
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            table = detect_csv_text_table(data_dir, path)
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
    document_files = [path for path in files if path not in container_paths]
    return {
        "csv_text_path_tables": csv_text_path_tables,
        "csv_inline_text_tables": csv_inline_text_tables,
        "csv_metadata_tables": csv_metadata_tables,
        "json_record_files": json_record_files,
        "document_files": document_files,
    }


def record_id_base(row: dict[str, Any]) -> str:
    for column in DOCUMENT_ID_COLUMN_CANDIDATES:
        if row.get(column) not in (None, ""):
            return str(row[column])
    id_columns = [
        column
        for column, value in row.items()
        if str(column).lower().endswith("_id") and value not in (None, "")
    ]
    if id_columns:
        # Prefer top-level ids over nested (dotted) ones; stable sort keeps column order within a depth.
        id_columns.sort(key=lambda column: str(column).count("."))
        return str(row[id_columns[0]])
    return ""


def unique_document_id(base: str, collision_suffix: str, seen: dict[str, int]) -> str:
    count = seen.get(base, 0)
    seen[base] = count + 1
    if count == 0:
        return base
    return f"{base}::{collision_suffix}"


def choose_document_id(row: dict[str, str], text_rel_path: str, seen: dict[str, int]) -> str:
    base = record_id_base(row)
    if not base:
        base = Path(text_rel_path).with_suffix("").as_posix()
    return unique_document_id(base, text_rel_path, seen)


def dataset_profile(data_dir: Path, files: list[Path], sources: dict[str, Any], cache_mode: str, record_count: int) -> dict[str, Any]:
    extension_counts = Counter(path.suffix.lower() or "<none>" for path in files)
    top_level_counts = Counter(
        path.relative_to(data_dir).parts[0] if len(path.relative_to(data_dir).parts) > 1 else "<root>"
        for path in files
    )
    total_bytes = sum(path.stat().st_size for path in files)
    return {
        "data_dir": str(data_dir),
        "cache_mode": cache_mode,
        "record_count": record_count,
        "source_file_count": len(files),
        "total_bytes": total_bytes,
        "top_level_entries": dict(sorted(top_level_counts.items())),
        "extension_counts": dict(sorted(extension_counts.items())),
        "sample_files": [path.relative_to(data_dir).as_posix() for path in files[:30]],
        "csv_tables": metadata_table_summaries(sources["csv_text_path_tables"][:20]),
        "inline_text_tables": inline_text_table_summaries(sources["csv_inline_text_tables"][:20]),
        "json_record_files": json_record_file_summaries(sources["json_record_files"][:20]),
        "metadata_only_tables": metadata_only_table_summaries(sources["csv_metadata_tables"][:20]),
        "document_file_count": len(sources["document_files"]),
    }


def metadata_table_summaries(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": table["relative_path"],
            "columns": table["columns"],
            "text_path_column": table["text_path_column"],
            "row_count": table["row_count"],
            "sample_valid_text_paths": table["sample_valid_text_paths"],
        }
        for table in tables
    ]


def inline_text_table_summaries(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": table["relative_path"],
            "columns": table["columns"],
            "inline_text_column": table["inline_text_column"],
            "avg_text_chars": table["avg_text_chars"],
            "row_count": table["row_count"],
        }
        for table in tables
    ]


def json_record_file_summaries(record_files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": info["relative_path"],
            "format": info["format"],
            "record_count": info["record_count"],
            "fields": info["fields"],
        }
        for info in record_files
    ]


def metadata_only_table_summaries(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": table["relative_path"],
            "columns": table["columns"],
            "row_count": table["row_count"],
        }
        for table in tables
    ]


def collection_summaries(sources: dict[str, Any]) -> list[dict[str, Any]]:
    collections: list[dict[str, Any]] = []
    for table in sources["csv_text_path_tables"]:
        collections.append(
            {
                "collection": f"csv:{table['relative_path']}",
                "kind": "csv_text_path_table",
                "cache_record_type": "csv_metadata_text_file",
                "relative_path": table["relative_path"],
                "columns": table["columns"],
                "text_path_column": table["text_path_column"],
                "row_count": table["row_count"],
            }
        )
    for table in sources["csv_inline_text_tables"]:
        collections.append(
            {
                "collection": f"csv:{table['relative_path']}",
                "kind": "csv_inline_text_table",
                "cache_record_type": "csv_inline_text_row",
                "relative_path": table["relative_path"],
                "columns": table["columns"],
                "inline_text_column": table["inline_text_column"],
                "avg_text_chars": table["avg_text_chars"],
                "row_count": table["row_count"],
            }
        )
    for info in sources["json_record_files"]:
        collections.append(
            {
                "collection": f"{info['format']}:{info['relative_path']}",
                "kind": "json_records",
                "cache_record_type": "json_record",
                "relative_path": info["relative_path"],
                "format": info["format"],
                "fields": info["fields"],
                "row_count": info["record_count"],
            }
        )
    return collections


def choose_cache_mode(sources: dict[str, Any], cached_document_files: int) -> str:
    has_csv_join_tables = bool(sources["csv_text_path_tables"])
    has_record_collections = bool(sources["csv_inline_text_tables"] or sources["json_record_files"])
    if has_csv_join_tables and not has_record_collections and cached_document_files == 0:
        return "csv_metadata_text_files"
    if not has_csv_join_tables and not has_record_collections:
        return "file_per_document"
    return "mixed_sources"


def normalize_dataset_manifest_aliases(manifest: dict[str, Any]) -> bool:
    changed = False
    metadata_tables = manifest.get("metadata_tables")
    if isinstance(metadata_tables, list) and "csv_tables" not in manifest:
        manifest["csv_tables"] = metadata_tables
        changed = True
    return changed


def dataset_cache_is_fresh(args: argparse.Namespace, paths: dict[str, Path]) -> bool:
    manifest_path = paths["dataset_cache_manifest"]
    cache_path = paths["dataset_cache_jsonl"]
    if args.rebuild_dataset_cache or not manifest_path.exists() or not cache_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("version") != DATASET_TEXT_CACHE_VERSION:
        return False
    if Path(str(manifest.get("data_dir", ""))).resolve() != args.data_dir.resolve():
        return False
    return manifest.get("files") == dataset_cache_manifest_fingerprint(args.data_dir)


def strip_html_to_text(raw_html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw_html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return html.unescape(re.sub(r"\s+", " ", text)).strip()


def extract_dataset_text(path: Path) -> tuple[str, dict[str, Any]]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            try:
                import fitz  # type: ignore
            except Exception as exc:
                return "", {"ok": False, "method": "pdf", "error": f"PyMuPDF unavailable: {exc}"}
            doc = fitz.open(path)
            try:
                return "\n".join(page.get_text() for page in doc), {"ok": True, "method": "pymupdf", "error": None}
            finally:
                doc.close()
        raw = path.read_text(encoding="utf-8", errors="replace")
        if suffix in {".htm", ".html"}:
            return strip_html_to_text(raw), {"ok": True, "method": "html", "error": None}
        return raw, {"ok": True, "method": "text", "error": None}
    except Exception as exc:
        return "", {"ok": False, "method": suffix.lstrip(".") or "unknown", "error": str(exc)}


def write_file_per_document_cache(
    args: argparse.Namespace,
    handle: Any,
    files: list[Path],
    seen_document_ids: dict[str, int],
) -> tuple[int, int]:
    failures = 0
    for index, path in enumerate(files, start=1):
        rel_path = path.relative_to(args.data_dir).as_posix()
        text, extraction = extract_dataset_text(path)
        if not extraction.get("ok"):
            failures += 1
        record = {
            "document_id": unique_document_id(rel_path, rel_path, seen_document_ids),
            "filename": path.name,
            "relative_path": rel_path,
            "extension": path.suffix.lower(),
            "source_path": str(path),
            "text": text,
            "text_chars": len(text),
            "text_words": len(text.split()),
            "extraction": extraction,
            "cache_record_type": "file_document",
            "collection": "files",
        }
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if index % 250 == 0:
            print(f"[dataset-cache] cached {index}/{len(files)} files", flush=True)
    return len(files), failures


def write_csv_metadata_text_cache(
    args: argparse.Namespace,
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
        metadata_source = csv_path.relative_to(args.data_dir).as_posix()
        with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as csv_handle:
            reader = csv.DictReader(csv_handle)
            for row_number, raw_row in enumerate(reader, start=2):
                row = normalize_csv_row(raw_row)
                text_ref = row.get(text_column, "")
                text_path = resolve_dataset_reference(args.data_dir, csv_path, text_ref)
                if text_path is None:
                    failures += 1
                    text = ""
                    extraction = {
                        "ok": False,
                        "method": "csv_text_path",
                        "error": f"text path not found: {text_ref}",
                    }
                    text_rel_path = text_ref
                    source_path = ""
                    extension = ""
                else:
                    referenced_paths.add(text_path)
                    text_rel_path = text_path.relative_to(args.data_dir).as_posix()
                    source_path = str(text_path)
                    extension = text_path.suffix.lower()
                    text, extraction = extract_dataset_text(text_path)
                    if not extraction.get("ok"):
                        failures += 1
                metadata = {key: value for key, value in row.items() if key != text_column}
                document_id = choose_document_id(row, text_rel_path, seen_document_ids)
                record: dict[str, Any] = {
                    "document_id": document_id,
                    "filename": row.get("filename") or row.get("file") or (Path(text_rel_path).name if text_rel_path else ""),
                    "relative_path": text_rel_path,
                    "extension": extension,
                    "source_path": source_path,
                    "text_path": text_ref,
                    "text_relative_path": text_rel_path,
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
                    print(f"[dataset-cache] cached {records} metadata rows", flush=True)
    return records, failures


def write_csv_inline_text_cache(
    args: argparse.Namespace,
    handle: Any,
    tables: list[dict[str, Any]],
    seen_document_ids: dict[str, int],
) -> tuple[int, int]:
    failures = 0
    records = 0
    for table in tables:
        csv_path = Path(table["csv_path"])
        text_column = table["inline_text_column"]
        metadata_source = csv_path.relative_to(args.data_dir).as_posix()
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
                    "source_path": str(csv_path),
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
                    print(f"[dataset-cache] cached {records} inline-text rows", flush=True)
    return records, failures


def iter_json_record_objects(path: Path, record_format: str) -> Any:
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


def write_json_records_cache(
    args: argparse.Namespace,
    handle: Any,
    record_files: list[dict[str, Any]],
    seen_document_ids: dict[str, int],
) -> tuple[int, int]:
    failures = 0
    records = 0
    for info in record_files:
        path = Path(info["path"])
        metadata_source = path.relative_to(args.data_dir).as_posix()
        record_format = info["format"]
        for record_number, record_obj in iter_json_record_objects(path, record_format):
            if record_obj is None:
                failures += 1
                continue
            flat = flatten_record(record_obj)
            text = record_text_from_flat(flat)
            base = record_id_base(flat) or f"{metadata_source}::record{record_number}"
            document_id = unique_document_id(base, f"{metadata_source}::record{record_number}", seen_document_ids)
            record: dict[str, Any] = {
                "document_id": document_id,
                "filename": path.name,
                "relative_path": metadata_source,
                "extension": path.suffix.lower(),
                "source_path": str(path),
                "metadata_source": metadata_source,
                "metadata_row_number": record_number,
                "metadata": {key: value for key, value in flat.items()},
                "record": record_obj,
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
                print(f"[dataset-cache] cached {records} json records", flush=True)
    return records, failures


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
CACHE_RECORD_EXTRA_FIELDS_BY_TYPE: dict[str, list[str]] = {
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


def write_dataset_text_cache(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    cache_dir = paths["dataset_cache_dir"]
    cache_path = paths["dataset_cache_jsonl"]
    manifest_path = paths["dataset_cache_manifest"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = dataset_cache_files(args.data_dir)
    sources = classify_dataset_sources(args.data_dir, files)
    started = time.time()
    seen_document_ids: dict[str, int] = {}
    referenced_paths: set[Path] = set()
    tmp_cache_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_cache_path.open("w", encoding="utf-8") as handle:
        join_records, join_failures = write_csv_metadata_text_cache(
            args, handle, sources["csv_text_path_tables"], seen_document_ids, referenced_paths
        )
        inline_records, inline_failures = write_csv_inline_text_cache(
            args, handle, sources["csv_inline_text_tables"], seen_document_ids
        )
        json_records, json_failures = write_json_records_cache(
            args, handle, sources["json_record_files"], seen_document_ids
        )
        # Files already cached through a CSV text-path join must not be duplicated
        # as standalone documents.
        if referenced_paths:
            document_files = [path for path in sources["document_files"] if path.resolve() not in referenced_paths]
        else:
            document_files = list(sources["document_files"])
        file_records, file_failures = write_file_per_document_cache(args, handle, document_files, seen_document_ids)
    tmp_cache_path.replace(cache_path)
    record_count = join_records + inline_records + json_records + file_records
    failures = join_failures + inline_failures + json_failures + file_failures
    cache_mode = choose_cache_mode(sources, file_records)
    profile = dataset_profile(args.data_dir, files, sources, cache_mode, record_count)
    metadata_tables = metadata_table_summaries(sources["csv_text_path_tables"])
    collections = collection_summaries(sources)
    if file_records:
        collections.append(
            {
                "collection": "files",
                "kind": "document_files",
                "cache_record_type": "file_document",
                "row_count": file_records,
            }
        )
    manifest = {
        "version": DATASET_TEXT_CACHE_VERSION,
        "created_at": time.time(),
        "elapsed_seconds": round_seconds(time.time() - started),
        "data_dir": str(args.data_dir),
        "jsonl_path": str(cache_path),
        "cache_mode": cache_mode,
        "record_count": record_count,
        "file_count": len(files),
        "failures": failures,
        "record_counts_by_group": {
            "csv_text_path_rows": join_records,
            "csv_inline_text_rows": inline_records,
            "json_records": json_records,
            "document_files": file_records,
        },
        "collections": collections,
        "metadata_only_tables": metadata_only_table_summaries(sources["csv_metadata_tables"]),
        "metadata_tables": metadata_tables,
        "csv_tables": metadata_tables,
        "dataset_profile": profile,
        "cache_record_fields": cache_record_field_union(),
        "cache_record_fields_by_type": CACHE_RECORD_EXTRA_FIELDS_BY_TYPE,
        "cache_record_base_fields": CACHE_RECORD_BASE_FIELDS,
        "files": [dataset_file_record(args.data_dir, path) for path in files],
    }
    write_json(manifest_path, manifest)
    return manifest


def ensure_dataset_text_cache(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    if dataset_cache_is_fresh(args, paths):
        manifest = json.loads(paths["dataset_cache_manifest"].read_text(encoding="utf-8"))
        if normalize_dataset_manifest_aliases(manifest):
            write_json(paths["dataset_cache_manifest"], manifest)
        print(
            f"[dataset-cache] reusing dataset cache: mode={manifest.get('cache_mode', 'file_per_document')} "
            f"records={manifest.get('record_count', manifest.get('file_count'))} files={manifest.get('file_count')} "
            f"failures={manifest.get('failures')} jsonl={paths['dataset_cache_jsonl']}",
            flush=True,
        )
        return manifest
    print(f"[dataset-cache] building dataset cache from {args.data_dir}", flush=True)
    manifest = write_dataset_text_cache(args, paths)
    print(
        f"[dataset-cache] ready: mode={manifest.get('cache_mode', 'file_per_document')} "
        f"records={manifest.get('record_count', manifest.get('file_count'))} files={manifest.get('file_count')} "
        f"failures={manifest.get('failures')} jsonl={paths['dataset_cache_jsonl']}",
        flush=True,
    )
    return manifest


def dataset_context_for_prompt(paths: dict[str, Path]) -> str:
    manifest = load_json_object(paths["dataset_cache_manifest"])
    if not manifest:
        context = {
            "cache_manifest_available": False,
            "cache_jsonl": str(paths["dataset_cache_jsonl"]),
            "cache_manifest": str(paths["dataset_cache_manifest"]),
        }
        return json.dumps(context, ensure_ascii=False, indent=2)
    profile = manifest.get("dataset_profile") if isinstance(manifest.get("dataset_profile"), dict) else {}
    metadata_tables = manifest.get("metadata_tables", [])
    csv_tables = manifest.get("csv_tables", metadata_tables)
    context = {
        "cache_manifest_available": True,
        "cache_jsonl": str(paths["dataset_cache_jsonl"]),
        "cache_manifest": str(paths["dataset_cache_manifest"]),
        "data_dir": manifest.get("data_dir"),
        "cache_mode": manifest.get("cache_mode", "file_per_document"),
        "record_count": manifest.get("record_count", manifest.get("file_count")),
        "source_file_count": manifest.get("file_count"),
        "failures": manifest.get("failures"),
        "record_counts_by_group": manifest.get("record_counts_by_group", {}),
        "cache_record_fields": manifest.get("cache_record_fields", []),
        "collections": manifest.get("collections", []),
        "metadata_only_tables": manifest.get("metadata_only_tables", []),
        "metadata_tables": metadata_tables,
        "csv_tables": csv_tables,
        "source_structure": {
            "top_level_entries": profile.get("top_level_entries", {}),
            "extension_counts": profile.get("extension_counts", {}),
            "sample_files": profile.get("sample_files", []),
            "csv_tables": profile.get("csv_tables", []),
            "inline_text_tables": profile.get("inline_text_tables", []),
            "json_record_files": profile.get("json_record_files", []),
            "metadata_only_tables": profile.get("metadata_only_tables", []),
        },
        "usage_notes": [
            "Use SQPE_DATASET_CACHE_MANIFEST first to inspect original data_dir, cache_mode, collections, source structure, CSV schemas, and cache record fields.",
            "Every cache record carries `collection` and `cache_record_type`; the manifest `collections` array lists each source group with its schema and row count.",
            "If cache_mode is mixed_sources, the cache JSONL mixes several collections; filter rows by `collection` for the collection(s) the task needs before any full scan.",
            "`csv_inline_text_row` records carry document text inline from the source CSV column named by the collection's `inline_text_column`; the other CSV columns are copied top-level and under `metadata`.",
            "`json_record` records come from JSONL/JSON record files: `record` holds the original nested object, flattened scalar fields are copied top-level and under `metadata`, and `text` is a deterministic `key: value` rendering.",
            "`metadata_only_tables` are auxiliary/relational CSVs with no document text; they are not cached as rows. Read those source CSVs directly for exact joins/predicates on shared id columns, then join against cache records if needed.",
            "For csv_metadata_text_files, manifest top-level metadata_tables and csv_tables both describe source CSV metadata tables.",
            "For file_per_document cache_mode, load SQPE_DATASET_CACHE_JSONL as JSONL with one cached record per benchmark document/filing.",
            "If cache_mode is csv_metadata_text_files, each cache row joins one metadata CSV row with its referenced text file; source CSV files are metadata tables, not documents.",
            "For csv_metadata_text_files, prefer source CSV metadata tables for exact predicates such as year, word_count, filing type, company, or numeric thresholds; load referenced text only for filtered candidates that need semantic operators.",
            "Do not materialize the full JSONL text cache into one DataFrame for large csv_metadata_text_files or mixed_sources datasets. If the JSONL cache is unavoidable, stream it with chunksize and discard full text as early as possible.",
            "For file_per_document cache_mode, each cache row corresponds to one source file under data_dir.",
        ],
    }
    return json.dumps(context, ensure_ascii=False, indent=2)


def ensure_runtime_dirs(args: argparse.Namespace) -> dict[str, Path]:
    run_dir = args.run_dir.resolve()
    paths = {
        "run_dir": run_dir,
        "state_dir": run_dir / "state",
        "raw_dir": run_dir / "raw",
        "output_dir": run_dir / "output",
        "results_dir": args.results_root.resolve() / args.dataset,
        "workspace_dir": run_dir / "workspace",
        "workspace_tmp_dir": run_dir / "workspace" / "tmp",
        "workspace_intermediate_dir": run_dir / "workspace" / "intermediate",
        "skill_mount_dir": run_dir / "skill_mount",
        "dataset_cache_dir": run_dir / "workspace" / "dataset_cache",
        "dataset_cache_jsonl": run_dir / "workspace" / "dataset_cache" / "dataset_texts.jsonl",
        "dataset_cache_manifest": run_dir / "workspace" / "dataset_cache" / "manifest.json",
        "plans_dir": run_dir / "plans",
        "metrics_dir": run_dir / "metrics",
        "task_metrics_dir": run_dir / "metrics" / "tasks",
        "results_jsonl": run_dir / "results.jsonl",
        "env_setup_raw": run_dir / "raw" / "_env_setup.json",
        "env_setup_output": run_dir / "output" / "_env_setup.json",
        "env_setup_metrics": run_dir / "metrics" / "_env_setup.metrics.json",
        "analysis_raw": run_dir / "raw" / "_system_analysis.json",
        "analysis_output": run_dir / "output" / "_system_analysis.json",
        "analysis_metrics": run_dir / "metrics" / "_system_analysis.metrics.json",
        "metrics_summary": run_dir / "metrics" / "summary.json",
    }
    for path in paths.values():
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
    return paths


def ensure_claude_skill_mount(paths: dict[str, Path]) -> None:
    """Expose repo skills/ through Claude Code's native .claude/skills layout.

    Claude Code loads skills from explicit --add-dir roots, expecting
    <dir>/.claude/skills/<name>/SKILL.md.
    The repository stores skills in ./skills/<name>/SKILL.md, so mount them into
    a runner-owned add-dir without moving or rewriting the source skill files.
    """
    source_root = Path(__file__).resolve().parent / "skills"
    mount_skills_dir = paths["skill_mount_dir"] / ".claude" / "skills"
    mount_skills_dir.mkdir(parents=True, exist_ok=True)
    if not source_root.exists():
        return
    for source_dir in sorted(source_root.iterdir()):
        if not source_dir.is_dir() or not (source_dir / "SKILL.md").exists():
            continue
        target_dir = mount_skills_dir / source_dir.name
        if target_dir.exists() or target_dir.is_symlink():
            if target_dir.is_symlink() or target_dir.is_file():
                target_dir.unlink()
            else:
                shutil.rmtree(target_dir)
        try:
            target_dir.symlink_to(source_dir.resolve(), target_is_directory=True)
        except OSError:
            shutil.copytree(source_dir, target_dir)


def register_child_process(process: subprocess.Popen[str], label: str) -> None:
    with ACTIVE_PROCESSES_LOCK:
        ACTIVE_PROCESSES[process.pid] = (process, label)


def unregister_child_process(process: subprocess.Popen[str]) -> None:
    with ACTIVE_PROCESSES_LOCK:
        ACTIVE_PROCESSES.pop(process.pid, None)


def terminate_process_group(
    process: subprocess.Popen[str],
    label: str,
    *,
    reason: str,
    grace_secs: int = PROCESS_TERMINATE_GRACE_SECS,
) -> None:
    if process.poll() is not None:
        return
    print(f"[cleanup] terminating {label} pid={process.pid} reason={reason}", file=sys.stderr, flush=True)
    pgid: int | None = None
    if os.name == "posix":
        try:
            pgid = os.getpgid(process.pid)
        except ProcessLookupError:
            pgid = None
        if pgid is not None and pgid != os.getpgrp():
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            process.terminate()
    else:
        process.terminate()
    try:
        process.wait(timeout=grace_secs)
        return
    except subprocess.TimeoutExpired:
        pass
    print(f"[cleanup] killing {label} pid={process.pid} after grace period", file=sys.stderr, flush=True)
    if os.name == "posix" and pgid is not None and pgid != os.getpgrp():
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        process.kill()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print(f"[cleanup] warning: {label} pid={process.pid} did not exit after SIGKILL", file=sys.stderr, flush=True)


def terminate_active_processes(reason: str) -> None:
    with ACTIVE_PROCESSES_LOCK:
        processes = list(ACTIVE_PROCESSES.values())
    for process, label in processes:
        terminate_process_group(process, label, reason=reason)


def install_shutdown_handlers() -> None:
    def handle_signal(signum: int, _frame: Any) -> None:
        global SHUTDOWN_STARTED
        signal_name = signal.Signals(signum).name
        if not SHUTDOWN_STARTED:
            SHUTDOWN_STARTED = True
            print(f"\nReceived {signal_name}; cleaning runner-owned child processes...", file=sys.stderr, flush=True)
            terminate_active_processes(signal_name)
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


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


def build_claude_settings(agent_env: dict[str, str]) -> dict[str, Any]:
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
        "cleanupPeriodDays": 0,
        "autoMemoryEnabled": False,
        "skipDangerousModePermissionPrompt": True,
    }


def build_claude_command(
    args: argparse.Namespace,
    agent_env: dict[str, str],
    paths: dict[str, Path],
    session_id: str,
    settings_path: Path,
    mcp_path: Path,
    *,
    tools: list[str] | None = None,
) -> list[str]:
    tool_names = tools or ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]
    tool_csv = ",".join(tool_names)
    command = [
        *shlex.split(args.claude_command),
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        require_env(agent_env, "AGENT_MAIN_MODEL"),
        "--settings",
        str(settings_path),
        "--mcp-config",
        str(mcp_path),
        "--strict-mcp-config",
        "--dangerously-skip-permissions",
        "--session-id",
        session_id,
        "--add-dir",
        str(ROOT_DIR),
        str(paths["skill_mount_dir"]),
        "--tools",
        tool_csv,
        "--allowedTools",
        tool_csv,
    ]
    if "Bash" in tool_names:
        command.extend(
            [
                "--disallowedTools",
                ",".join(
                    [
                        "Bash(tail -f *)",
                        "Bash(tail -F *)",
                        "Bash(tail --follow *)",
                        "Bash(watch *)",
                        "Bash(while *)",
                        "Bash(until *)",
                        "Bash(nohup *)",
                        "Bash(disown *)",
                        "Bash(tmux *)",
                        "Bash(screen *)",
                    ]
                ),
            ]
        )
    return command


def build_child_env(system_env: dict[str, str], agent_env: dict[str, str], args: argparse.Namespace, paths: dict[str, Path], state_dir: Path, skill_names: list[str]) -> dict[str, str]:
    child_env = os.environ.copy()
    child_env.update(system_env)
    child_env["HOME"] = str(state_dir / "home")
    child_env["CLAUDE_CONFIG_DIR"] = str(state_dir / "claude_config")
    child_env["XDG_CONFIG_HOME"] = str(state_dir / "home" / ".config")
    child_env["XDG_CACHE_HOME"] = str(state_dir / "home" / ".cache")
    for key in CLAUDE_PROVIDER_ENV_FLAGS:
        child_env.pop(key, None)
    child_env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    child_env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] = "1"
    # claude-code loads "project memory" by walking every ancestor of cwd up to /,
    # which reaches the operator's real ~/.claude/CLAUDE.md regardless of the
    # HOME/CLAUDE_CONFIG_DIR redirection below. Hard-disable all CLAUDE.md loading;
    # task instructions arrive via prompts and skills mount through --add-dir.
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
    child_env["SQPE_SYSTEM_ROOT"] = str(args.system_path)
    child_env["SQPE_SYSTEM_NAME"] = args.system_name
    child_env["SQPE_SYSTEM_PACKAGE"] = args.system_package
    child_env["SQPE_PIPELINE_SKILL_PATH"] = str(args.pipeline_skill_path)
    child_env["SQPE_ENV_SETUP_SKILL_PATH"] = str(args.env_setup_skill)
    child_env["SQPE_ENV_RECORD_PATH"] = str(args.env_record)
    child_env["SQPE_ENV_PATH"] = str(args.env_path)
    child_env["SQPE_DATASET_CACHE_JSONL"] = str(paths["dataset_cache_jsonl"])
    child_env["SQPE_DATASET_CACHE_MANIFEST"] = str(paths["dataset_cache_manifest"])
    child_env["SQPE_DATA_DIR"] = str(args.data_dir)
    child_env["LOTUS_DATA_ROOT"] = str(args.data_dir.parent)
    child_env["LOTUS_LLM_MAX_CONCURRENCY"] = str(args.model_concurrency)
    child_env["LOTUS_CACHE_DIR"] = str(paths["workspace_dir"] / "lotus_cache")
    child_env["SQPE_ALLOWED_SKILL_NAMES"] = ",".join(skill_names)
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(args.system_path), str(ROOT_DIR), existing_pythonpath) if path
    )
    return child_env


def summarize_stream_event(event: dict[str, Any]) -> list[str]:
    messages: list[str] = []
    event_type = event.get("type")
    if event_type == "system" and event.get("subtype") == "init" and isinstance(event.get("model"), str):
        messages.append(f"[agent][model] {event['model']}")
    if event_type in {"assistant", "message"}:
        message = event.get("message") if isinstance(event.get("message"), dict) else event
        model = message.get("model") if isinstance(message, dict) else event.get("model")
        content = message.get("content") if isinstance(message, dict) else None
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
                    text = replace_self_reported_model_label(item["text"], model if isinstance(model, str) else None)
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
    (re.compile(r"\bwatch\b", re.I), "watch monitor"),
    (re.compile(r"(?s)\bsleep\s+[0-9.]+[smhd]?\s*(?:&&|;|\n)\s*(?:\(\s*)?(?:tail|grep|ps|cat|test|ls|python3?)\b", re.I), "delayed monitoring command"),
    (re.compile(r"(?s)\bwhile\b.+\bdo\b.+\b(?:sleep|tail|grep|ps)\b", re.I), "polling loop"),
    (re.compile(r"(?s)\buntil\b.+\bdo\b.+\b(?:sleep|tail|grep|ps)\b", re.I), "polling loop"),
    (re.compile(r"(?s)\bfor\b.+\bdo\b.+\b(?:sleep|tail|grep|ps)\b", re.I), "polling loop"),
    (re.compile(r"(?<![&>])&(?![&>])\s*(?:$|[;\n])"), "background command"),
    (re.compile(r"\b(?:nohup|disown|tmux|screen)\b", re.I), "detached process/session command"),
]


HEREDOC_START_RE = re.compile(
    r"<<(?P<strip_tabs>-)?(?!<)\s*(?:'(?P<single>[^']+)'|\"(?P<double>[^\"]+)\"|(?P<bare>[^\s;|&()<>]+))"
)


def _mask_heredoc_bodies(command: str, shell_syntax: str) -> str:
    command_lines = command.splitlines(keepends=True)
    syntax_lines = shell_syntax.splitlines(keepends=True)
    if len(command_lines) != len(syntax_lines):
        return shell_syntax

    pending: list[tuple[str, bool]] = []
    output: list[str] = []
    for command_line, syntax_line in zip(command_lines, syntax_lines, strict=True):
        if pending:
            delimiter, strip_tabs = pending[0]
            candidate = command_line.rstrip("\r\n")
            if strip_tabs:
                candidate = candidate.lstrip("\t")
            output.append("".join(char if char in "\r\n" else " " for char in syntax_line))
            if candidate == delimiter:
                pending.pop(0)
            continue

        output.append(syntax_line)
        for match in HEREDOC_START_RE.finditer(command_line):
            if syntax_line[match.start() : match.start() + 2] != "<<":
                continue
            delimiter = match.group("single") or match.group("double") or match.group("bare")
            if delimiter:
                pending.append((delimiter.replace("\\", ""), bool(match.group("strip_tabs"))))
    return "".join(output)


def unquoted_shell_syntax(command: str) -> str:
    """Return shell syntax while masking quoted arguments, comments, and heredocs."""
    chars = list(command)
    masked = list(command)
    quote: str | None = None
    index = 0
    while index < len(chars):
        char = chars[index]
        if quote is not None:
            masked[index] = " "
            if char == "\\" and quote == '"' and index + 1 < len(chars):
                masked[index + 1] = " "
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue

        if char in {"'", '"'}:
            quote = char
            masked[index] = " "
            index += 1
            continue
        if char == "\\" and index + 1 < len(chars):
            masked[index] = " "
            masked[index + 1] = " "
            index += 2
            continue
        if char == "#" and (index == 0 or chars[index - 1].isspace() or chars[index - 1] in ";|&(){}"):
            while index < len(chars) and chars[index] != "\n":
                masked[index] = " "
                index += 1
            continue
        index += 1
    return _mask_heredoc_bodies(command, "".join(masked))


def nested_shell_scripts(command: str) -> list[str]:
    """Extract scripts passed to a nested shell's -c option."""
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return []

    scripts: list[str] = []
    shell_names = {"bash", "dash", "ksh", "sh", "zsh"}
    for index, token in enumerate(tokens):
        if Path(token).name not in shell_names:
            continue
        option_index = index + 1
        while option_index < len(tokens):
            option = tokens[option_index]
            if option == "--":
                break
            if not option.startswith("-") or option == "-":
                break
            if "c" in option[1:]:
                if option_index + 1 < len(tokens):
                    scripts.append(tokens[option_index + 1])
                break
            option_index += 1
    return scripts


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


def forbidden_bash_command_reason(command: str, *, _depth: int = 0) -> str | None:
    shell_syntax = unquoted_shell_syntax(command)
    if tail_command_uses_follow(shell_syntax):
        return "persistent tail monitor"
    for pattern, reason in FORBIDDEN_BASH_COMMAND_PATTERNS:
        if pattern.search(shell_syntax):
            return reason
    if _depth < 3:
        for nested_script in nested_shell_scripts(command):
            reason = forbidden_bash_command_reason(nested_script, _depth=_depth + 1)
            if reason:
                return reason
    return None


def forbidden_bash_command_from_event(event: dict[str, Any]) -> dict[str, str] | None:
    for command in bash_tool_commands_from_event(event):
        reason = forbidden_bash_command_reason(command)
        if reason:
            return {"reason": reason, "command": command}
    return None


def shorten_one_line(value: str, max_len: int) -> str:
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
        flags=re.I,
    ).strip()


def run_claude_message(
    *,
    label: str,
    message: str,
    system_env: dict[str, str],
    agent_env: dict[str, str],
    args: argparse.Namespace,
    paths: dict[str, Path],
    skill_names: list[str],
    session_prefix: str,
    tools: list[str] | None = None,
) -> dict[str, Any]:
    # Claude Code validates --session-id as a UUID. Keep the filesystem state
    # directory readable, but pass only the raw UUID to the CLI.
    session_id = str(uuid.uuid4())
    state_dir = paths["state_dir"] / safe_filename(f"{session_prefix}-{session_id}")
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "home").mkdir(parents=True, exist_ok=True)
    (state_dir / "claude_config").mkdir(parents=True, exist_ok=True)
    settings_path = state_dir / "claude_settings.json"
    mcp_path = state_dir / "empty_mcp.json"
    write_json(settings_path, build_claude_settings(agent_env))
    write_json(mcp_path, {"mcpServers": {}})
    ensure_claude_skill_mount(paths)
    command = build_claude_command(args, agent_env, paths, session_id, settings_path, mcp_path, tools=tools)
    child_env = build_child_env(system_env, agent_env, args, paths, state_dir, skill_names)
    secrets = secret_values(child_env)

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    parsed_event_count = 0
    policy_violation: dict[str, str] | None = None
    policy_lock = threading.Lock()
    popen_kwargs: dict[str, Any] = {}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    started = time.time()
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
    register_child_process(process, label)

    def read_stdout() -> None:
        nonlocal parsed_event_count, policy_violation
        assert process.stdout is not None
        for line in process.stdout:
            safe_line = redact_text(line, secrets)
            stdout_parts.append(safe_line)
            try:
                event = json.loads(safe_line)
            except json.JSONDecodeError:
                if args.show_agent_log and safe_line.strip():
                    print(f"[agent][stdout] {shorten_one_line(safe_line.strip(), 220)}", flush=True)
                continue
            if isinstance(event, dict):
                parsed_event_count += 1
                violation = forbidden_bash_command_from_event(event)
                if violation is not None:
                    with policy_lock:
                        if policy_violation is None:
                            policy_violation = violation
                            if args.show_agent_log:
                                print(
                                    "[agent][policy] rejected Bash command: "
                                    f"{violation['reason']}: {shorten_one_line(violation['command'], 220)}",
                                    flush=True,
                                )
                            terminate_process_group(
                                process,
                                label,
                                reason=f"forbidden Bash command: {violation['reason']}",
                            )
                if args.show_agent_log:
                    for summary in summarize_stream_event(event):
                        print(summary, flush=True)

    def read_stderr() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            safe_line = redact_text(line, secrets)
            stderr_parts.append(safe_line)
            if args.show_agent_log and safe_line.strip():
                print(f"[agent][stderr] {shorten_one_line(safe_line.strip(), 220)}", file=sys.stderr, flush=True)

    stdout_thread = threading.Thread(target=read_stdout, daemon=True)
    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    if process.stdin is not None:
        process.stdin.write(message)
        process.stdin.close()

    timed_out = False
    timeout = args.timeout + args.agent_exit_grace_secs
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        terminate_process_group(process, label, reason=f"timeout after {timeout}s")
    except KeyboardInterrupt:
        terminate_process_group(process, label, reason="keyboard interrupt")
        raise
    finally:
        unregister_child_process(process)
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

    stdout = "".join(stdout_parts)
    stderr = "".join(stderr_parts)
    usage = extract_claude_usage(stdout, agent_env)
    exit_code = process.returncode
    if policy_violation is not None and (exit_code is None or exit_code == 0):
        exit_code = 126
    return {
        "label": label,
        "session_id": session_id,
        "state_dir": str(state_dir),
        "settings_path": str(settings_path),
        "mcp_path": str(mcp_path),
        "command": command,
        "cwd": str(ROOT_DIR),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "elapsed_ms": int((time.time() - started) * 1000),
        "elapsed_seconds": round_seconds(time.time() - started),
        "usage": usage,
        "cost_usd": round_cost(get_nested_number(usage, ("cost_usd", "total"))),
        "stdout": stdout,
        "stderr": stderr,
        "parsed_event_count": parsed_event_count,
        "policy_violation": policy_violation,
    }


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def build_system_analysis_prompt(args: argparse.Namespace) -> str:
    return "\n".join(
        [
            "You are in Phase 1: analyze one new operator-native system and produce a pipeline skill.",
            "",
            f"Workspace root: {ROOT_DIR}",
            f"System name: {args.system_name}",
            f"System source path: {args.system_path}",
            f"Meta-skill to follow: {args.analysis_skill}",
            f"Phase 0 env setup record: {args.env_record}",
            f"Configured env/config path: {args.env_path}",
            f"Required output pipeline skill path: {args.pipeline_skill_path}",
            "",
            "Hard requirements:",
            "- First read the meta-skill file at the path above. Follow it as the governing workflow.",
            "- Read only the reference files the meta-skill directs you to load.",
            "- Treat environment setup as Phase 0. Do not run env-setup-skill or ask interactive env questions here.",
            "- Read ENV.md if it exists, and make the produced pipeline skill follow the recorded environment, native setup notes, and cost/token tracking source.",
            "- Before drafting the pipeline skill, follow ENV.md and run an import/setup probe plus one tiny operator-native model-call smoke test through the target system itself.",
            "- The smoke test must use the target system's own operators/API. Do not use direct OpenAI/Anthropic/LiteLLM imports or raw HTTP calls.",
            "- If the ENV handoff is wrong, patch ENV.md first and rerun the smoke test. Do not generate the pipeline skill until the handoff works or the limitation is documented.",
            "- When patching ENV.md, update every affected section consistently: model/provider notes, native setup code, cost/token tracking, caveats, and downstream launch guidance. Do not patch only one mention of a model or provider string.",
            "- After patching ENV.md, re-read the changed sections and verify the smoke-tested native setup exactly matches the setup shown in ENV.md.",
            "- If ENV.md contains per-token prices and the target system records token usage but reports zero native cost, treat that as broken cost wiring. Patch the documented setup or system-local pricing registration when feasible; otherwise document a precise limitation and fallback.",
            "- Analyze actual source, docs, examples, and operator implementations. Do not rely only on README claims.",
            "- Produce exactly one pipeline skill for this system at the required output path.",
            "- The produced skill frontmatter name must equal the system name.",
            "- The produced skill must teach downstream agents to use the system's operator-native API for benchmark tasks.",
            "- The produced skill must explain that the runner provides SQPE_DATASET_CACHE_JSONL and SQPE_DATASET_CACHE_MANIFEST.",
            "- Do not include API key values or endpoint secrets in the produced skill.",
            "- Do not solve benchmark tasks in this phase.",
            "",
            "Return exactly one JSON object and no markdown:",
            "{",
            f'  "system_name": "{args.system_name}",',
            f'  "pipeline_skill": "{args.pipeline_skill_path}",',
            '  "summary": "short summary of the produced skill",',
            '  "env_smoke_test": {"ok": true, "summary": "operator-native model-call smoke test result", "elapsed_seconds": 0.0, "cost_usd": 0.0},',
            '  "operators_covered": ["operator names"],',
            '  "caveats": ["remaining caveats"]',
            "}",
        ]
    )


def load_pipeline_skill(path: Path, system_name: str) -> str:
    if not path.exists():
        raise SystemExit(f"pipeline skill does not exist: {path}. Run --mode analyze-system first.")
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---", text, re.S)
    if not match:
        raise SystemExit(f"pipeline skill lacks frontmatter: {path}")
    name_match = re.search(r"^name:\s*(.+)$", match.group(1), re.M)
    if not name_match:
        raise SystemExit(f"pipeline skill frontmatter lacks name: {path}")
    actual_name = name_match.group(1).strip().strip("'\"")
    if actual_name != system_name:
        raise SystemExit(f"pipeline skill name={actual_name!r}, but --system-name={system_name!r}")
    return text


def task_output_file(paths: dict[str, Path], task_id: str) -> Path:
    return paths["workspace_intermediate_dir"] / f"output_{safe_filename(task_id)}.json"


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


def python_executable_from_env_record(env_record: Path) -> str:
    if env_record.exists():
        text = env_record.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(r"(?m)^(/[^`\s]+/python(?:3(?:\.\d+)?)?)\s*$", text):
            candidate = match.group(1)
            if Path(candidate).exists():
                return candidate
    return sys.executable


def build_pipeline_env(
    system_env: dict[str, str],
    args: argparse.Namespace,
    paths: dict[str, Path],
    task_id: str,
) -> dict[str, str]:
    child_env = os.environ.copy()
    child_env.update(system_env)
    child_env["SQPE_SYSTEM_ROOT"] = str(args.system_path)
    child_env["SQPE_SYSTEM_NAME"] = args.system_name
    child_env["SQPE_SYSTEM_PACKAGE"] = args.system_package
    child_env["SQPE_PIPELINE_SKILL_PATH"] = str(args.pipeline_skill_path)
    child_env["SQPE_ENV_SETUP_SKILL_PATH"] = str(args.env_setup_skill)
    child_env["SQPE_ENV_RECORD_PATH"] = str(args.env_record)
    child_env["SQPE_ENV_PATH"] = str(args.env_path)
    child_env["SQPE_DATASET_CACHE_JSONL"] = str(paths["dataset_cache_jsonl"])
    child_env["SQPE_DATASET_CACHE_MANIFEST"] = str(paths["dataset_cache_manifest"])
    child_env["SQPE_DATA_DIR"] = str(args.data_dir)
    child_env["PLANAR_TASK_ID"] = task_id
    child_env["PLANAR_OUTPUT_PATH"] = str(task_output_file(paths, task_id))
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(args.system_path), str(ROOT_DIR), existing_pythonpath) if path
    )
    return child_env


def run_task_script_foreground(
    *,
    task_id: str,
    script_path: Path,
    system_env: dict[str, str],
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> dict[str, Any]:
    label = f"operator-script-{task_id}"
    python_command = shlex.split(system_env.get("LOTUS_PIPELINE_PYTHON", ""))
    if not python_command:
        python_command = [python_executable_from_env_record(args.env_record)]
    command = [*python_command, "-u", str(script_path)]
    child_env = build_pipeline_env(system_env, args, paths, task_id)
    secrets = secret_values(child_env)
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
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
        return {
            "label": label,
            "command": command,
            "cwd": str(script_path.parent),
            "script_path": str(script_path),
            "exit_code": 127,
            "timed_out": False,
            "elapsed_ms": 0,
            "elapsed_seconds": 0.0,
            "stdout": "",
            "stderr": str(exc),
            "error": str(exc),
        }

    register_child_process(process, label)

    def read_stream(stream: Any, parts: list[str], prefix: str) -> None:
        for line in stream:
            safe_line = redact_text(line, secrets)
            parts.append(safe_line)
            if args.show_agent_log and safe_line.strip():
                print(f"[system][{prefix}] {shorten_one_line(safe_line.strip(), 220)}", flush=True)

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

    return {
        "label": label,
        "command": command,
        "cwd": str(script_path.parent),
        "script_path": str(script_path),
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "elapsed_ms": int((time.time() - started) * 1000),
        "elapsed_seconds": round_seconds(time.time() - started),
        "stdout": "".join(stdout_parts),
        "stderr": "".join(stderr_parts),
    }


def build_task_plan_prompt(
    task: dict[str, Any],
    args: argparse.Namespace,
    paths: dict[str, Path],
    pipeline_skill_text: str,
    attempt: int,
    max_attempts: int,
    previous_attempts: list[dict[str, Any]],
) -> str:
    task_id = task["task_id"]
    plan_file = paths["plans_dir"] / f"{safe_filename(task_id)}.plan.md"
    dataset_context = dataset_context_for_prompt(paths)
    retry_lines: list[str] = []
    if attempt > 1:
        retry_lines = [
            "",
            "Retry instructions:",
            f"- This is attempt {attempt}/{max_attempts} for the same task.",
            "- Inspect previous attempt summaries and revise the operator-native plan. Do not switch to direct LLM calls.",
            json.dumps(previous_attempts[-3:], ensure_ascii=False),
        ]
    return "\n".join(
        [
            "You are in Phase 2A: generate a query plan through the generated pipeline skill and target system.",
            "",
            f"Workspace root: {ROOT_DIR}",
            f"Target system name: {args.system_name}",
            f"Target Python package/import root: {args.system_package}",
            f"Target system source path: {args.system_path}",
            f"Phase 0 env setup record: {args.env_record}",
            f"Configured env/config path: {args.env_path}",
            f"Pipeline skill path: {args.pipeline_skill_path}",
            f"Dataset directory: {args.data_dir}",
            f"Dataset cache JSONL: {paths['dataset_cache_jsonl']}",
            f"Dataset cache manifest: {paths['dataset_cache_manifest']}",
            "Dataset source/cache context:",
            "```json",
            dataset_context,
            "```",
            f"Task file: {args.input}",
            f"Task ID: {task_id}",
            f"Query: {task.get('query', '')}",
            f"Answer schema: {json.dumps(task.get('answer_schema'), ensure_ascii=False)}",
            f"Gold program hint, if present: {json.dumps(task.get('gold_program'), ensure_ascii=False)}",
            f"Required plan file to write: {plan_file}",
            *retry_lines,
            "",
            "Planning requirements:",
            "- Read the pipeline skill file and ENV.md first. The pipeline skill content is also embedded below for cache stability.",
            "- Treat environment setup as already handled by Phase 0.",
            f"- Plan must use the {args.system_name} system and its operators/API as described in the pipeline skill.",
            "- Write only the plan file. Do not run the target system, write task scripts, or create the benchmark output file in this phase.",
            "- You must create the required plan file with the Write/Edit tool before returning. A final JSON response is only a completion signal and does not satisfy the plan-file contract.",
            "- The plan must name target-system operators/API methods, input files, intermediate artifacts, output/save strategy, and cost/token accounting source.",
            "- The output/save strategy must write exactly `task_id`, `answer`, `elapsed_seconds`, `cost_usd`, and aggregate `total_tokens`.",
            "- Use the dataset manifest/cache handoff as the default data-loading source.",
            "- Use the dataset source/cache context above to understand whether cached rows are source files or CSV metadata rows joined to text files.",
            "- If `cache_mode` is `file_per_document`, plan to read `SQPE_DATASET_CACHE_JSONL` directly unless the task needs raw file inspection.",
            "- If `cache_mode` is `csv_metadata_text_files`, plan exact metadata predicates against the source CSV table(s) first and load referenced text only for the reduced candidate set. Do not plan a full-cache `pd.read_json(..., lines=True)` load when CSV metadata columns can perform the narrowing.",
            "- If `cache_mode` is `mixed_sources`, read the manifest `collections` list first, pick only the collection(s) the task needs, and plan filters on the cache rows' `collection` field; `csv_inline_text_row` and `json_record` rows already contain their text inline, and `metadata_only_tables` must be queried from their source CSVs for exact joins/predicates.",
            "- For large text-backed tables, plan bounded candidate batches and avoid accumulating thousands of full-text rows in memory before semantic operators.",
            "- For predicates about whether a document mentions explicit terms, phrases, entities, or acronyms, plan a deterministic high-recall keyword/regex scan first, then use semantic operators only on reduced or ambiguous candidates. Do not plan semantic filters over thousands of full documents when literal-term filtering can safely narrow the input.",
            "- For long row-wise documents, plan a separate evidence column with a conservative safety margin. For 128k-context models, default to roughly 60k-80k evidence tokens instead of near-context-limit payloads.",
            "- Build long-document evidence by local retrieval/selection first: relevant sections/headings, query-term windows, synonyms, and metadata-derived names. Token trimming is only the final guardrail after evidence is selected and ordered by relevance; do not plan blind first-N truncation.",
            "- Pure Python may be planned only for cache loading, deterministic non-LLM handling, file extraction fallback, lightweight answer normalization, and final JSON writing.",
            "- The plan must explicitly reject direct LLM/API bypasses.",
            "- The eventual pipeline script must run synchronously in the foreground. No &, nohup, disown, tmux/screen, daemonization, or detached subprocesses.",
            "- The eventual execution must not wrap the main pipeline in shell `timeout` or choose a shorter timeout than the runner.",
            "- The eventual execution must not use persistent log monitors such as `tail -f`, `tail -F`, `tail --follow`, `watch`, or endless polling loops.",
            "- Do not read gold answers or evaluator internals.",
            "",
            "Pipeline skill content starts here:",
            "```markdown",
            pipeline_skill_text,
            "```",
            "",
            "After the required plan file exists, return exactly one JSON object and no markdown:",
            "{",
            f'  "task_id": "{task_id}",',
            f'  "system_name": "{args.system_name}",',
            f'  "plan_file": "{plan_file}",',
            '  "operators_planned": ["operator names or API methods planned"],',
            '  "summary": "short execution plan summary"',
            "}",
        ]
    )


def build_task_execute_prompt(
    task: dict[str, Any],
    args: argparse.Namespace,
    paths: dict[str, Path],
    pipeline_skill_text: str,
    attempt: int,
    max_attempts: int,
    previous_attempts: list[dict[str, Any]],
) -> str:
    task_id = task["task_id"]
    output_file = task_output_file(paths, task_id)
    plan_file = paths["plans_dir"] / f"{safe_filename(task_id)}.plan.md"
    task_tmp_dir_path = task_tmp_dir(paths, task_id)
    task_attempt_tmp_dir_path = task_attempt_tmp_dir(paths, task_id, attempt)
    dataset_context = dataset_context_for_prompt(paths)
    retry_lines: list[str] = []
    if attempt > 1:
        retry_lines = [
            "",
            "Retry context:",
            f"- This is attempt {attempt}/{max_attempts}.",
            "- The plan file was regenerated for this attempt. Avoid previous failed implementation details.",
            "- If previous attempts show `output file missing`, `answer` is null, or no Python task script was found, the fix is to write a better task script. The runner, not you, will run it.",
            json.dumps(previous_attempts[-3:], ensure_ascii=False),
        ]
    return "\n".join(
        [
            "You are in Phase 2B: write one benchmark pipeline script by following the generated query plan and pipeline skill.",
            "",
            f"Workspace root: {ROOT_DIR}",
            f"Target system name: {args.system_name}",
            f"Target Python package/import root: {args.system_package}",
            f"Target system source path: {args.system_path}",
            f"Phase 0 env setup record: {args.env_record}",
            f"Configured env/config path: {args.env_path}",
            f"Pipeline skill path: {args.pipeline_skill_path}",
            f"Dataset directory: {args.data_dir}",
            f"Dataset cache JSONL: {paths['dataset_cache_jsonl']}",
            f"Dataset cache manifest: {paths['dataset_cache_manifest']}",
            "Dataset source/cache context:",
            "```json",
            dataset_context,
            "```",
            f"Task file: {args.input}",
            f"Task ID: {task_id}",
            f"Query: {task.get('query', '')}",
            f"Answer schema: {json.dumps(task.get('answer_schema'), ensure_ascii=False)}",
            f"Gold program hint, if present: {json.dumps(task.get('gold_program'), ensure_ascii=False)}",
            f"Plan file to follow: {plan_file}",
            f"Required benchmark output file: {output_file}",
            f"Task scratch root: {task_tmp_dir_path}",
            f"Script/scratch directory for this attempt: {task_attempt_tmp_dir_path}",
            *retry_lines,
            "",
            "Execution artifact contract:",
            "- Your first actions in this phase must be tool calls, not the final JSON response.",
            "- You must Read the plan file before implementing the task.",
            f"- You must create or update at least one Python task script under `{task_attempt_tmp_dir_path}` using Write/Edit.",
            "- Do not edit, reuse, or return scripts from another attempt directory. This attempt directory is clean and is the only accepted script location.",
            "- The task script must execute the target system's native operators/API and write the required benchmark output JSON.",
            "- Do not run the task script yourself. The runner will execute the selected Python script synchronously in the foreground after you return.",
            "- Do not use Bash in this phase. You only need Read/Write/Edit/Glob/Grep to inspect inputs and write the script.",
            f"- The script must write `{output_file}` when the runner executes it.",
            "- The output JSON must contain exactly `task_id`, a non-null `answer`, `elapsed_seconds`, `cost_usd`, and `total_tokens`.",
            "- If the task answer is legitimately empty, write the schema-correct empty value (`[]` for list/table answers or `{}` for dictionary answers), never null.",
            "- The final JSON response is only a completion signal for the runner. It does not replace the task script or the required output JSON file.",
            "",
            "Mandatory operator-native rule:",
            "- Read the plan file first and execute that plan. If a minor detail changes, document it under the task scratch directory.",
            "- Read the pipeline skill file and ENV.md first. The pipeline skill content is also embedded below for cache stability.",
            "- Treat environment setup as already handled by Phase 0.",
            f"- You must use the {args.system_name} system and its operators/API as described in the pipeline skill.",
            f"- Task scripts must explicitly import the target Python package, e.g. `import {args.system_package}` or `from {args.system_package} ...`.",
            "- You may write Python scripts, but those scripts must use the target system package and its documented native setup pattern.",
            "- Pure Python is allowed only for dataset loading, deterministic non-LLM handling, file extraction fallback, lightweight answer normalization, and final JSON writing.",
            "- Use the dataset manifest/cache handoff as the primary dataset input unless the plan requires raw file inspection.",
            "- Use the dataset source/cache context above to understand whether cached rows are source files or CSV metadata rows joined to text files.",
            "- If the script reads the manifest JSON, use the exact field names shown in the manifest/context; do not invent top-level aliases. For CSV-backed datasets, prefer direct source CSV paths from the plan when available.",
            "- If `cache_mode` is `file_per_document`, read `SQPE_DATASET_CACHE_JSONL` directly unless the task needs raw file inspection.",
            "- If `cache_mode` is `csv_metadata_text_files`, use source CSV metadata tables for exact deterministic filtering/sorting/grouping whenever possible, then resolve and read text files only for rows that require semantic operators. Never load or concat the entire full-text JSONL cache just to filter metadata columns.",
            "- If `cache_mode` is `mixed_sources`, filter cache rows by their `collection` field per the plan before any semantic operators; `csv_inline_text_row` and `json_record` rows carry text inline (no extra file loading), and `metadata_only_tables` are only available from their source CSVs for exact joins.",
            "- If the JSONL cache must be scanned, stream it with `pd.read_json(..., lines=True, chunksize=...)` or line iteration, immediately drop non-candidate rows and unused `text`, and process semantic candidates in bounded batches. Do not append unbounded chunks containing full document text to a list.",
            "- If the task asks whether documents mention explicit terms, phrases, entities, or acronyms, implement a deterministic high-recall keyword/regex pass before semantic operators. Use semantic operators only for reduced/ambiguous candidates or for judgments that cannot be resolved by local text matching.",
            "- For long row-wise documents, create a separate evidence column with a conservative input-token budget. For 128k-context models, default to roughly 60k-80k evidence tokens, for example `min(max_ctx_len - max_tokens - max(32768, int(max_ctx_len * 0.25)), 80000)`.",
            "- Do not pass raw full 10-K text to row-wise semantic operators when it is long. Retrieve/select relevant sections, headings, query-term windows, synonyms, and metadata-derived names first; apply token trimming only after selected evidence is ordered by relevance.",
            "- Do not bypass the target system by writing direct LLM/API scripts.",
            "- Forbidden in task scripts: direct OpenAI/Anthropic/LiteLLM imports, direct requests/httpx/urllib calls to model endpoints, model_call_helper, /chat/completions, /responses.",
            "- If the target system internally calls an LLM through its own operators, that is allowed and expected.",
            "- Save the target-system result with exactly `task_id`, `answer`, `elapsed_seconds`, `cost_usd`, and `total_tokens`.",
            "- `total_tokens` is the aggregate token count from the target system's native model calls. The runner adds Claude Code agent usage when it writes the persistent result.",
            "- `cost_usd` must equal the target system cost from the system's cost accounting when available. If native cost is zero but token counters are non-zero, use the pipeline skill's deterministic token-price fallback.",
            "- Run on the full dataset unless the query itself narrows scope.",
            "- Do not read gold answers or evaluator internals.",
            "",
            "Process lifecycle requirements:",
            "- The task script itself must run synchronously in the foreground when invoked by the runner.",
            "- The task script must not append &, use nohup/disown, launch tmux/screen, daemonize, or leave detached subprocesses.",
            "- Use the ENV.md-documented non-interactive launch form for the target environment.",
            "- Do not wrap the main pipeline command in shell `timeout`.",
            "- Do not put sleep/tail/watch/ps monitoring loops in the script. Progress prints are fine; monitoring is owned by the runner.",
            "- The script is complete only after it writes the exact five-field output JSON.",
            "",
            "Pipeline skill content starts here:",
            "```markdown",
            pipeline_skill_text,
            "```",
            "",
            "After the required script file exists, return exactly one JSON object and no markdown:",
            "{",
            f'  "task_id": "{task_id}",',
            f'  "system_name": "{args.system_name}",',
            f'  "plan_file": "{plan_file}",',
            '  "script_file": "absolute path to the Python script you wrote",',
            f'  "output_file": "{output_file}",',
            '  "operators_used": ["operator names or API methods used"]',
            "}",
        ]
    )


def load_tasks(input_path: Path, task_id: str | None, limit: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks", []) if isinstance(payload, dict) else payload
    if not isinstance(tasks, list):
        raise SystemExit(f"input file lacks a tasks array: {input_path}")
    normalized = [task for task in tasks if isinstance(task, dict) and isinstance(task.get("task_id"), str)]
    selected = normalized
    if task_id:
        selected = [task for task in normalized if task.get("task_id") == task_id]
        if not selected:
            raise SystemExit(f"task not found: {task_id}")
    if limit is not None:
        selected = selected[:limit]
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
        selected = selected[:args.limit]
    return selected


def compact_result_path(results_dir: Path, task: dict[str, Any]) -> Path:
    return (
        results_dir
        / task_difficulty(task)
        / "claude_code_plus_lotus"
        / f"{safe_filename(task['task_id'])}.json"
    )


def read_compact_results(results_dir: Path, tasks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    for task in tasks:
        payload = load_json_object(compact_result_path(results_dir, task))
        if not payload or payload.get("task_id") != task.get("task_id"):
            continue
        total_tokens = to_int(payload.get("total_tokens"))
        existing[task["task_id"]] = {
            "task_id": task["task_id"],
            "ok": payload.get("answer") is not None,
            "answer": payload.get("answer"),
            "elapsed_secs": payload.get("elapsed_seconds"),
            "cost_usd": payload.get("cost_usd"),
            "total_elapsed_seconds": payload.get("elapsed_seconds"),
            "total_cost_usd": payload.get("cost_usd"),
            "total_tokens": total_tokens,
        }
    return existing


def read_existing_results(path: Path) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return existing
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        task_id = entry.get("task_id") if isinstance(entry, dict) else None
        if isinstance(task_id, str):
            backfill_entry_system_usage(entry)
            existing[task_id] = entry
    return existing


def load_entry_path(entry: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = entry.get(key)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        loaded = load_json_object(path)
        if loaded:
            return loaded
    return {}


def backfill_entry_system_usage(entry: dict[str, Any]) -> None:
    if to_int(entry.get("system_total_token")) > 0 and isinstance(entry.get("final_system_usage"), dict):
        return
    output = load_entry_path(entry, "output_file")
    execute_raw = load_entry_path(entry, "execute_raw_attempt_file", "execute_raw_file")
    usage = system_usage_from_execution(output if output else None, execute_raw if execute_raw else None)
    if not usage["available"]:
        return

    entry["final_system_usage"] = usage
    entry["final_system_total_token"] = usage["total_tokens"]
    if to_int(entry.get("system_total_token")) == 0:
        entry["system_total_token"] = usage["total_tokens"]
        entry["system_input_tokens"] = usage["input_tokens"]
        entry["system_output_tokens"] = usage["output_tokens"]
    entry.setdefault("system_usage", usage)


def write_results_jsonl(path: Path, all_tasks: list[dict[str, Any]], existing: dict[str, dict[str, Any]]) -> None:
    task_ids = {task.get("task_id") for task in all_tasks}
    ordered = [existing[task["task_id"]] for task in all_tasks if task.get("task_id") in existing]
    ordered.extend(entry for task_id, entry in existing.items() if task_id not in task_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    lines = [json.dumps(entry, ensure_ascii=False) for entry in ordered]
    tmp_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    tmp_path.replace(path)


def write_result_files(results_dir: Path, all_tasks: list[dict[str, Any]], existing: dict[str, dict[str, Any]]) -> None:
    for task in all_tasks:
        entry = existing.get(task.get("task_id"))
        if not entry:
            continue
        task_id = entry.get("task_id")
        if isinstance(task_id, str):
            write_json(
                compact_result_path(results_dir, task),
                {
                    "task_id": task_id,
                    "answer": entry.get("answer"),
                    "elapsed_seconds": entry.get("total_elapsed_seconds", entry.get("elapsed_secs")),
                    "cost_usd": entry.get("total_cost_usd", entry.get("cost_usd")),
                    "total_tokens": to_int(entry.get("total_tokens")),
                },
            )


def parse_assistant_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.S)
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {"answer": parsed}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", stripped)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {"answer": parsed}
        except json.JSONDecodeError:
            pass
    return {}


def extract_assistant_text(raw: dict[str, Any]) -> str:
    stdout = raw.get("stdout", "")
    if isinstance(stdout, str):
        result_texts: list[str] = []
        assistant_texts: list[str] = []
        for event in iter_json_lines(stdout):
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
    for stream_name in ("stdout", "stderr"):
        text = raw.get(stream_name)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return ""


def load_task_output(output_file: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not output_file.exists():
        return None, f"output file missing: {output_file}"
    try:
        loaded = json.loads(output_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"output JSON parse failed: {exc}"
    if not isinstance(loaded, dict):
        return None, "output JSON is not an object"
    required = ("task_id", "answer", "elapsed_seconds", "cost_usd", "total_tokens")
    missing = [key for key in required if key not in loaded]
    if missing:
        return loaded, f"output JSON missing keys: {', '.join(missing)}"
    extra = sorted(set(loaded) - set(required))
    if extra:
        return loaded, f"output JSON contains unsupported keys: {', '.join(extra)}"
    return loaded, None


def validate_answer_schema(answer: Any, schema: Any) -> list[str]:
    warnings: list[str] = []
    if not isinstance(schema, dict):
        return warnings
    schema_type = schema.get("type")
    if schema_type == "scalar":
        if not isinstance(answer, (int, float)) or isinstance(answer, bool):
            warnings.append("scalar answer must be a number")
    elif schema_type in {"named_entity", "label", "report"}:
        if not isinstance(answer, str):
            warnings.append(f"{schema_type} answer must be a string")
    elif schema_type in {"table", "ordered_list"}:
        if not isinstance(answer, list):
            warnings.append(f"{schema_type} answer must be an array")
        elif schema.get("columns"):
            columns = schema["columns"]
            if isinstance(columns, dict):
                columns = list(columns)
            if isinstance(columns, list):
                for idx, row in enumerate(answer):
                    if isinstance(row, dict):
                        for column in columns:
                            if column not in row:
                                warnings.append(f"row {idx} missing column {column}")
                    elif len(columns) != 1:
                        warnings.append(f"row {idx} is not object for multi-column table")
    elif schema_type == "dictionary":
        if not isinstance(answer, dict):
            warnings.append("dictionary answer must be an object")
    return warnings


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
    deadline = time.time() + PROCESS_TERMINATE_GRACE_SECS
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


def handle_lingering_task_processes(paths: dict[str, Path], task_id: str, *, wait_if_running: bool, wait_timeout_secs: int, reason: str) -> dict[str, Any]:
    label = f"task {task_id}"
    initial = list_task_processes(paths, task_id)
    if not initial:
        return {"found": False, "waited_seconds": 0.0, "timed_out": False, "terminated": []}
    started = time.time()
    if wait_if_running:
        print(f"[cleanup] waiting for {len(initial)} lingering subprocess(es) from {label}", file=sys.stderr, flush=True)
        deadline = started + wait_timeout_secs
        while time.time() < deadline:
            time.sleep(min(15, max(0.5, deadline - time.time())))
            current = list_task_processes(paths, task_id)
            if not current:
                return {
                    "found": True,
                    "waited_seconds": round_seconds(time.time() - started),
                    "timed_out": False,
                    "initial": summarize_task_processes(initial),
                    "terminated": [],
                }
        current = list_task_processes(paths, task_id)
        terminated = terminate_task_processes(current, label=label, reason=f"{reason}; lingering wait exceeded")
        return {
            "found": True,
            "waited_seconds": round_seconds(time.time() - started),
            "timed_out": True,
            "initial": summarize_task_processes(initial),
            "terminated": terminated,
        }
    terminated = terminate_task_processes(initial, label=label, reason=reason)
    return {
        "found": True,
        "waited_seconds": round_seconds(time.time() - started),
        "timed_out": False,
        "initial": summarize_task_processes(initial),
        "terminated": terminated,
    }


def scan_forbidden_bypass(paths: dict[str, Path], task_id: str, script_dir: Path | None = None) -> list[str]:
    findings: list[str] = []
    tmp_dir = script_dir or task_tmp_dir(paths, task_id)
    if not tmp_dir.exists():
        return findings
    compiled = [(pattern, re.compile(pattern, re.I)) for pattern in FORBIDDEN_BYPASS_PATTERNS]
    for path in sorted(tmp_dir.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for pattern, regex in compiled:
            if regex.search(text):
                findings.append(f"{display_path(path)} matches forbidden bypass pattern {pattern}")
    return findings


def scan_operator_native_use(paths: dict[str, Path], task_id: str, package_name: str, script_dir: Path | None = None) -> list[str]:
    tmp_dir = script_dir or task_tmp_dir(paths, task_id)
    if not tmp_dir.exists():
        return [f"task scratch directory missing: {display_path(tmp_dir)}"]
    scripts = sorted(tmp_dir.rglob("*.py"))
    if not scripts:
        return [f"no Python task script found under {display_path(tmp_dir)}"]
    import_re = re.compile(
        rf"^\s*(?:import\s+{re.escape(package_name)}(?:\s|\.|$)|from\s+{re.escape(package_name)}(?:\s|\.|$))",
        re.M,
    )
    for path in scripts:
        try:
            if import_re.search(path.read_text(encoding="utf-8", errors="replace")):
                return []
        except OSError:
            continue
    return [f"no task script imports target package {package_name!r} under {display_path(tmp_dir)}"]


def script_run_failure_summary(execute_raw: dict[str, Any] | None) -> dict[str, Any]:
    if not execute_raw:
        return {}
    script_run = execute_raw.get("script_run")
    if not isinstance(script_run, dict):
        return {}
    summary: dict[str, Any] = {
        "script_run_exit_code": script_run.get("exit_code"),
        "script_run_timed_out": script_run.get("timed_out"),
        "script_run_elapsed_seconds": script_run.get("elapsed_seconds"),
        "script_run_script_path": script_run.get("script_path"),
    }
    stderr_tail = text_tail(script_run.get("stderr"), 5000).strip()
    stdout_tail = text_tail(script_run.get("stdout"), 3000).strip()
    if stderr_tail:
        summary["script_run_stderr_tail"] = stderr_tail
    if stdout_tail:
        summary["script_run_stdout_tail"] = stdout_tail
    return {key: value for key, value in summary.items() if value not in (None, "")}


def build_task_attempt_metrics(
    *,
    attempt_started: float,
    attempt_finished: float,
    plan_raw: dict[str, Any] | None,
    execute_raw: dict[str, Any] | None,
    output: dict[str, Any] | None,
    plan_raw_file: Path | None,
    execute_raw_file: Path | None,
) -> dict[str, Any]:
    plan = agent_phase_metrics("task_plan_generation", plan_raw, raw_file=plan_raw_file)
    execute_agent = agent_phase_metrics("task_execution_agent", execute_raw, raw_file=execute_raw_file)
    system_usage = system_usage_from_execution(output, execute_raw)
    system_elapsed_seconds = to_float(output.get("elapsed_seconds")) if output else 0.0
    output_cost_usd = to_float(output.get("cost_usd")) if output else 0.0
    system_cost_usd = output_cost_usd if output_cost_usd > 0 else system_usage["cost_usd"]
    agent_elapsed_seconds = plan["elapsed_seconds"] + execute_agent["elapsed_seconds"]
    agent_cost_usd = plan["agent_cost_usd"] + execute_agent["agent_cost_usd"]
    plan_agent_tokens = to_int(get_nested_number(plan, ("agent_usage", "tokens", "total")))
    execute_agent_tokens = to_int(
        get_nested_number(execute_agent, ("agent_usage", "tokens", "total"))
    )
    agent_total_tokens = plan_agent_tokens + execute_agent_tokens
    total_elapsed_seconds = max(0.0, attempt_finished - attempt_started)
    total_cost_usd = agent_cost_usd + system_cost_usd
    return {
        "plan_generation": plan,
        "execution": {
            "agent": execute_agent,
            "system_elapsed_seconds": round_seconds(system_elapsed_seconds),
            "system_cost_usd": round_cost(system_cost_usd),
            "system_usage": system_usage,
            "system_total_token": system_usage["total_tokens"],
        },
        "total": {
            "elapsed_seconds": round_seconds(total_elapsed_seconds),
            "agent_elapsed_seconds": round_seconds(agent_elapsed_seconds),
            "system_elapsed_seconds": round_seconds(system_elapsed_seconds),
            "cost_usd": round_cost(total_cost_usd),
            "agent_cost_usd": round_cost(agent_cost_usd),
            "agent_total_tokens": agent_total_tokens,
            "system_cost_usd": round_cost(system_cost_usd),
            "system_total_token": system_usage["total_tokens"],
            "total_tokens": agent_total_tokens + system_usage["total_tokens"],
        },
    }


def cumulative_attempt_metrics(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    def sum_attempt(key: str) -> float:
        return sum(to_float(attempt.get(key)) for attempt in attempts)

    def sum_attempt_int(key: str) -> int:
        return sum(to_int(attempt.get(key)) for attempt in attempts)

    plan_elapsed = sum_attempt("plan_agent_elapsed_seconds")
    plan_cost = sum_attempt("plan_agent_cost_usd")
    execute_elapsed = sum_attempt("execute_agent_elapsed_seconds")
    execute_cost = sum_attempt("execute_agent_cost_usd")
    system_elapsed = sum_attempt("system_elapsed_seconds")
    system_cost = sum_attempt("system_cost_usd")
    system_input_tokens = sum_attempt_int("system_input_tokens")
    system_output_tokens = sum_attempt_int("system_output_tokens")
    system_total_tokens = sum_attempt_int("system_total_token")
    agent_total_tokens = sum_attempt_int("agent_total_tokens")
    total_elapsed = sum_attempt("total_elapsed_seconds")
    total_cost = sum_attempt("total_cost_usd")
    return {
        "plan_agent_elapsed_seconds": round_seconds(plan_elapsed),
        "plan_agent_cost_usd": round_cost(plan_cost),
        "execute_agent_elapsed_seconds": round_seconds(execute_elapsed),
        "execute_agent_cost_usd": round_cost(execute_cost),
        "agent_elapsed_seconds": round_seconds(plan_elapsed + execute_elapsed),
        "agent_cost_usd": round_cost(plan_cost + execute_cost),
        "agent_total_tokens": agent_total_tokens,
        "system_elapsed_seconds": round_seconds(system_elapsed),
        "system_cost_usd": round_cost(system_cost),
        "system_input_tokens": system_input_tokens,
        "system_output_tokens": system_output_tokens,
        "system_total_token": system_total_tokens,
        "total_tokens": agent_total_tokens + system_total_tokens,
        "total_elapsed_seconds": round_seconds(total_elapsed),
        "total_cost_usd": round_cost(total_cost),
    }


def task_metrics_file(paths: dict[str, Path], task_id: str) -> Path:
    return paths["task_metrics_dir"] / f"{safe_filename(task_id)}.metrics.json"


def write_metrics_summary(paths: dict[str, Path], args: argparse.Namespace, all_tasks: list[dict[str, Any]], existing: dict[str, dict[str, Any]]) -> None:
    ordered_entries = [
        existing[task["task_id"]]
        for task in all_tasks
        if isinstance(task.get("task_id"), str) and task.get("task_id") in existing
    ]
    completed = [entry for entry in ordered_entries if entry.get("ok") is True]
    phase0 = load_json_object(paths["env_setup_metrics"])
    phase1 = load_json_object(paths["analysis_metrics"])

    def sum_key(entries: list[dict[str, Any]], key: str) -> float:
        return sum(to_float(entry.get(key)) for entry in entries)

    tasks_plan_cost = sum_key(ordered_entries, "plan_agent_cost_usd")
    tasks_execute_agent_cost = sum_key(ordered_entries, "execute_agent_cost_usd")
    tasks_system_cost = sum_key(ordered_entries, "system_cost_usd")
    tasks_system_total_token = sum(to_int(entry.get("system_total_token")) for entry in ordered_entries)
    tasks_total_cost = sum_key(ordered_entries, "total_cost_usd")
    summary = {
        "system_name": args.system_name,
        "env_record": str(args.env_record),
        "pipeline_skill": str(args.pipeline_skill_path),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phase0_env_setup": {
            "metrics_file": str(paths["env_setup_metrics"]) if paths["env_setup_metrics"].exists() else None,
            "elapsed_seconds": round_seconds(to_float(phase0.get("elapsed_seconds"))),
            "agent_cost_usd": round_cost(to_float(phase0.get("agent_cost_usd"))),
        },
        "phase1_pipeline_skill_generation": {
            "metrics_file": str(paths["analysis_metrics"]) if paths["analysis_metrics"].exists() else None,
            "elapsed_seconds": round_seconds(to_float(phase1.get("elapsed_seconds"))),
            "agent_cost_usd": round_cost(to_float(phase1.get("agent_cost_usd"))),
            "env_smoke_test_elapsed_seconds": round_seconds(to_float(phase1.get("env_smoke_test_elapsed_seconds"))),
            "env_smoke_test_cost_usd": round_cost(to_float(phase1.get("env_smoke_test_cost_usd"))),
        },
        "phase2_tasks": {
            "completed_tasks": len(completed),
            "recorded_tasks": len(ordered_entries),
            "plan_generation": {"agent_cost_usd": round_cost(tasks_plan_cost)},
            "execution": {
                "agent_cost_usd": round_cost(tasks_execute_agent_cost),
                "system_cost_usd": round_cost(tasks_system_cost),
                "system_total_token": tasks_system_total_token,
            },
            "total": {
                "cost_usd": round_cost(tasks_total_cost),
                "agent_cost_usd": round_cost(tasks_plan_cost + tasks_execute_agent_cost),
                "system_cost_usd": round_cost(tasks_system_cost),
                "system_total_token": tasks_system_total_token,
            },
        },
    }
    write_json(paths["metrics_summary"], summary)


def refresh_metrics_summary(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    all_tasks: list[dict[str, Any]] = []
    if args.input.exists():
        try:
            all_tasks, _ = load_tasks(args.input, None, None)
        except Exception as exc:
            print(f"[metrics] warning: cannot load task input for summary: {exc}", file=sys.stderr, flush=True)
    existing = read_existing_results(paths["results_jsonl"])
    write_metrics_summary(paths, args, all_tasks, existing)


def run_env_setup(args: argparse.Namespace, system_env: dict[str, str], agent_env: dict[str, str], paths: dict[str, Path]) -> int:
    print(f"[phase0] checking system={args.system_name} path={args.system_path}")
    file_env = parse_dotenv(args.env_path)
    expand_env_references(file_env)
    probe = run_import_probe(args, file_env)
    warnings = collect_env_setup_warnings(args, file_env, probe)
    for warning in warnings:
        print(f"[phase0][warning] {warning}")
    if args.dry_run:
        print(f"[phase0][dry-run] would launch Claude Code and write {args.env_record}")
        return 0
    raw = run_claude_message(
        label=f"env-setup-{args.system_name}",
        message=build_env_setup_prompt(args, file_env=file_env, warnings=warnings, probe=probe),
        system_env=system_env,
        agent_env=agent_env,
        args=args,
        paths=paths,
        skill_names=["env-setup-skill"],
        session_prefix=f"env-setup-{args.system_name}",
    )
    raw = redact_payload(raw, secret_values_from_envs(system_env, agent_env))
    write_json(paths["env_setup_raw"], raw)
    metrics = agent_phase_metrics("env_setup", raw, raw_file=paths["env_setup_raw"])
    metrics.update({"system_name": args.system_name, "system_path": str(args.system_path), "env_record": str(args.env_record), "warnings": warnings})
    write_json(paths["env_setup_metrics"], metrics)
    response = parse_assistant_response(extract_assistant_text(raw))
    response.update({"env_record_exists": args.env_record.exists(), "metrics": metrics, "runner_warnings": warnings, "import_probe": probe})
    write_json(paths["env_setup_output"], response)
    refresh_metrics_summary(args, paths)
    if raw.get("exit_code") != 0 or raw.get("timed_out"):
        raise SystemExit(f"Phase 0 env setup agent failed, exit_code={raw.get('exit_code')} timed_out={raw.get('timed_out')}")
    if not args.env_record.exists():
        raise SystemExit(f"Phase 0 env setup agent did not write ENV.md: {args.env_record}")
    print(f"[phase0] ENV.md ready: {args.env_record}")
    return 0


def run_system_analysis(args: argparse.Namespace, system_env: dict[str, str], agent_env: dict[str, str], paths: dict[str, Path]) -> None:
    print(f"[phase1] analyzing system={args.system_name} path={args.system_path}")
    raw = run_claude_message(
        label=f"analyze-system-{args.system_name}",
        message=build_system_analysis_prompt(args),
        system_env=system_env,
        agent_env=agent_env,
        args=args,
        paths=paths,
        skill_names=["new-system-operator-skill-creator"],
        session_prefix=f"analyze-{args.system_name}",
    )
    raw = redact_payload(raw, secret_values_from_envs(system_env, agent_env))
    write_json(paths["analysis_raw"], raw)
    response = parse_assistant_response(extract_assistant_text(raw))
    env_smoke_test = response.get("env_smoke_test") if isinstance(response.get("env_smoke_test"), dict) else {}
    metrics = agent_phase_metrics("pipeline_skill_generation", raw, raw_file=paths["analysis_raw"])
    metrics.update(
        {
            "system_name": args.system_name,
            "system_path": str(args.system_path),
            "pipeline_skill": str(args.pipeline_skill_path),
            "env_smoke_test": env_smoke_test,
            "env_smoke_test_elapsed_seconds": round_seconds(to_float(env_smoke_test.get("elapsed_seconds"))),
            "env_smoke_test_cost_usd": round_cost(to_float(env_smoke_test.get("cost_usd"))),
        }
    )
    write_json(paths["analysis_metrics"], metrics)
    response["pipeline_skill_exists"] = args.pipeline_skill_path.exists()
    response["metrics"] = metrics
    write_json(paths["analysis_output"], response)
    refresh_metrics_summary(args, paths)
    if raw.get("exit_code") != 0 or raw.get("timed_out"):
        raise SystemExit(f"system analysis failed, exit_code={raw.get('exit_code')} timed_out={raw.get('timed_out')}")
    if not args.pipeline_skill_path.exists():
        raise SystemExit(f"system analysis did not write pipeline skill: {args.pipeline_skill_path}")
    print(f"[phase1] pipeline skill ready: {args.pipeline_skill_path}")


def run_tasks(args: argparse.Namespace, system_env: dict[str, str], agent_env: dict[str, str], paths: dict[str, Path]) -> int:
    pipeline_skill_text = load_pipeline_skill(args.pipeline_skill_path, args.system_name)
    all_tasks, _ = load_tasks(args.input, None, None)
    selected_tasks = select_testbed_tasks(all_tasks, args)
    skip_task_ids = normalize_task_ids(args.skip_task)
    if skip_task_ids:
        selected_task_ids = {task.get("task_id") for task in selected_tasks}
        unknown_skips = sorted(task_id for task_id in skip_task_ids if task_id not in selected_task_ids)
        if unknown_skips:
            print(f"[phase2][warning] skip task(s) not in selected input: {', '.join(unknown_skips)}", flush=True)
        selected_tasks = [task for task in selected_tasks if task.get("task_id") not in skip_task_ids]
    existing = read_existing_results(paths["results_jsonl"])
    for task_id, entry in read_compact_results(paths["results_dir"], all_tasks).items():
        existing.setdefault(task_id, entry)
    pending = selected_tasks if args.force else [task for task in selected_tasks if existing.get(task.get("task_id"), {}).get("ok") is not True]
    write_results_jsonl(paths["results_jsonl"], all_tasks, existing)
    write_result_files(paths["results_dir"], all_tasks, existing)
    write_metrics_summary(paths, args, all_tasks, existing)
    print(f"[phase2] system={args.system_name} pipeline_skill={args.pipeline_skill_path}")
    print(f"[phase2] selected={len(selected_tasks)} pending={len(pending)}")
    if skip_task_ids:
        print(f"[phase2] skipped={', '.join(sorted(skip_task_ids))}", flush=True)
    if args.dry_run:
        print(f"[phase2] dataset_cache={paths['dataset_cache_jsonl']} (not built in dry-run)")
        for task in pending:
            print(f"{task['task_id']}: {task.get('query', '')}")
        return 0
    if not pending:
        print(f"[phase2] dataset_cache={paths['dataset_cache_jsonl']} (not built: no pending tasks)")
        return 0
    dataset_cache_manifest = ensure_dataset_text_cache(args, paths)
    print(f"[phase2] dataset_cache={paths['dataset_cache_jsonl']} files={dataset_cache_manifest.get('file_count')} failures={dataset_cache_manifest.get('failures')}")

    for task in pending:
        task_id = task["task_id"]
        attempt_summaries: list[dict[str, Any]] = []
        final_entry: dict[str, Any] | None = None
        for attempt in range(1, args.task_max_attempts + 1):
            print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] running {task_id} attempt {attempt}/{args.task_max_attempts}", flush=True)
            attempt_started = time.time()
            task_tmp_dir_path = task_tmp_dir(paths, task_id)
            task_attempt_tmp_dir_path = task_attempt_tmp_dir(paths, task_id, attempt)
            task_tmp_dir_path.mkdir(parents=True, exist_ok=True)
            preexisting_task_processes = handle_lingering_task_processes(paths, task_id, wait_if_running=False, wait_timeout_secs=0, reason="before starting a new task attempt")
            shutil.rmtree(task_attempt_tmp_dir_path, ignore_errors=True)
            task_attempt_tmp_dir_path.mkdir(parents=True, exist_ok=True)
            output_file = task_output_file(paths, task_id)
            plan_file = paths["plans_dir"] / f"{safe_filename(task_id)}.plan.md"
            for generated_file in (output_file, plan_file):
                try:
                    generated_file.unlink()
                except FileNotFoundError:
                    pass

            safe_task_id = safe_filename(task_id)
            plan_raw = run_claude_message(
                label=f"operator-plan-{task_id}",
                message=build_task_plan_prompt(task, args, paths, pipeline_skill_text, attempt, args.task_max_attempts, attempt_summaries),
                system_env=system_env,
                agent_env=agent_env,
                args=args,
                paths=paths,
                skill_names=[args.system_name],
                session_prefix=f"{args.system_name}-{safe_task_id.lower()}-plan",
            )
            plan_raw = redact_payload(plan_raw, secret_values_from_envs(system_env, agent_env))
            plan_raw_file = paths["raw_dir"] / f"{safe_task_id}.plan.json"
            plan_raw_attempt_file = paths["raw_dir"] / f"{safe_task_id}.plan.attempt-{attempt}.json"
            write_json(plan_raw_file, plan_raw)
            write_json(plan_raw_attempt_file, plan_raw)
            plan_response = parse_assistant_response(extract_assistant_text(plan_raw))
            plan_process_ok = plan_raw.get("exit_code") == 0 and plan_raw.get("timed_out") is False
            plan_ok = plan_process_ok and plan_file.exists()
            plan_error = None
            if not plan_process_ok:
                plan_error = f"plan agent failed: exit_code={plan_raw.get('exit_code')} timed_out={plan_raw.get('timed_out')}"
            elif not plan_file.exists():
                plan_error = f"plan file missing: {plan_file}"

            execute_raw: dict[str, Any] | None = None
            execute_response: dict[str, Any] = {}
            execute_raw_file: Path | None = None
            execute_raw_attempt_file: Path | None = None
            lingering_task_processes: dict[str, Any] = {"found": False}
            script_path: Path | None = None
            script_selection_error: str | None = None
            if plan_ok:
                before_script_mtimes = task_script_mtimes(task_attempt_tmp_dir_path)
                execute_agent_raw = run_claude_message(
                    label=f"operator-execute-{task_id}",
                    message=build_task_execute_prompt(task, args, paths, pipeline_skill_text, attempt, args.task_max_attempts, attempt_summaries),
                    system_env=system_env,
                    agent_env=agent_env,
                    args=args,
                    paths=paths,
                    skill_names=[args.system_name],
                    session_prefix=f"{args.system_name}-{safe_task_id.lower()}-execute",
                    tools=["Read", "Write", "Edit", "Glob", "Grep"],
                )
                execute_agent_raw = redact_payload(execute_agent_raw, secret_values_from_envs(system_env, agent_env))
                execute_raw_file = paths["raw_dir"] / f"{safe_task_id}.execute.json"
                execute_raw_attempt_file = paths["raw_dir"] / f"{safe_task_id}.execute.attempt-{attempt}.json"
                execute_response = parse_assistant_response(extract_assistant_text(execute_agent_raw))
                script_path, script_selection_error = select_task_script(task_attempt_tmp_dir_path, execute_response, before_script_mtimes)

                execute_raw = dict(execute_agent_raw)
                execute_raw["script_agent_exit_code"] = execute_agent_raw.get("exit_code")
                execute_raw["script_agent_timed_out"] = execute_agent_raw.get("timed_out")
                execute_raw["script_path"] = str(script_path) if script_path else None
                execute_raw["script_selection_error"] = script_selection_error

                execute_agent_ok = execute_agent_raw.get("exit_code") == 0 and execute_agent_raw.get("timed_out") is False
                if execute_agent_ok and script_path is not None and script_selection_error is None:
                    print(f"[runner] executing {task_id} script in foreground: {script_path}", flush=True)
                    script_run_raw = run_task_script_foreground(
                        task_id=task_id,
                        script_path=script_path,
                        system_env=system_env,
                        args=args,
                        paths=paths,
                    )
                    script_run_raw = redact_payload(script_run_raw, secret_values(system_env))
                    execute_raw["script_run"] = script_run_raw
                    execute_raw["script_run_exit_code"] = script_run_raw.get("exit_code")
                    execute_raw["script_run_timed_out"] = script_run_raw.get("timed_out")
                    execute_raw["exit_code"] = script_run_raw.get("exit_code")
                    execute_raw["timed_out"] = script_run_raw.get("timed_out")
                elif execute_agent_ok:
                    execute_raw["exit_code"] = 1
                    execute_raw["timed_out"] = False

            if execute_raw is None:
                output, output_error = None, plan_error or "execution not started"
            else:
                execute_finished_ok = execute_raw.get("exit_code") == 0 and execute_raw.get("timed_out") is False
                lingering_task_processes = handle_lingering_task_processes(
                    paths,
                    task_id,
                    wait_if_running=False,
                    wait_timeout_secs=0,
                    reason="task script left subprocesses running" if execute_finished_ok else "execute phase failed or timed out",
                )
                if lingering_task_processes.get("found"):
                    execute_raw["lingering_task_processes"] = lingering_task_processes
                    if execute_finished_ok:
                        execute_raw["exit_code"] = 1
                        execute_raw["script_lingering_processes"] = True
                if lingering_task_processes.get("timed_out"):
                    execute_raw["timed_out"] = True
                    execute_raw["lingering_task_processes_timed_out"] = True
                if execute_raw_file is not None:
                    write_json(execute_raw_file, execute_raw)
                if execute_raw_attempt_file is not None:
                    write_json(execute_raw_attempt_file, execute_raw)
                output, output_error = load_task_output(output_file)
                if output_error and script_selection_error:
                    output_error = f"{output_error}; {script_selection_error}"
                script_failure = script_run_failure_summary(execute_raw)
                if output_error and script_failure.get("script_run_stderr_tail"):
                    output_error = f"{output_error}; script stderr tail: {shorten_one_line(str(script_failure['script_run_stderr_tail']), 500)}"

            schema_warnings = validate_answer_schema(output.get("answer") if output else None, task.get("answer_schema"))
            bypass_findings = [] if args.disable_bypass_scan or execute_raw is None else scan_forbidden_bypass(paths, task_id, task_attempt_tmp_dir_path)
            operator_use_findings = [] if args.disable_operator_use_scan or execute_raw is None else scan_operator_native_use(paths, task_id, args.system_package, task_attempt_tmp_dir_path)
            execute_process_ok = execute_raw is not None and execute_raw.get("exit_code") == 0 and execute_raw.get("timed_out") is False
            process_ok = plan_process_ok and execute_process_ok
            ok = plan_ok and process_ok and output is not None and output_error is None and not schema_warnings and not bypass_findings and not operator_use_findings
            attempt_finished = time.time()
            metrics = build_task_attempt_metrics(
                attempt_started=attempt_started,
                attempt_finished=attempt_finished,
                plan_raw=plan_raw,
                execute_raw=execute_raw,
                output=output,
                plan_raw_file=plan_raw_attempt_file,
                execute_raw_file=execute_raw_attempt_file,
            )
            raw_file = paths["raw_dir"] / f"{safe_task_id}.json"
            raw_attempt_file = paths["raw_dir"] / f"{safe_task_id}.attempt-{attempt}.json"
            write_json(
                raw_file,
                {
                    "task_id": task_id,
                    "attempt": attempt,
                    "plan_response": plan_response,
                    "execute_response": execute_response,
                    "plan_raw_file": str(plan_raw_attempt_file),
                    "execute_raw_file": str(execute_raw_attempt_file) if execute_raw_attempt_file else None,
                    "metrics": metrics,
                },
            )
            write_json(raw_attempt_file, load_json_object(raw_file))

            answer = output.get("answer") if output else None
            elapsed_secs = output.get("elapsed_seconds") if output else None
            cost_usd = output.get("cost_usd") if output else None
            system_usage = metrics["execution"]["system_usage"]
            attempt_summary = {
                "attempt": attempt,
                "ok": ok,
                "process_ok": process_ok,
                "plan_ok": plan_ok,
                "plan_error": plan_error,
                "execute_process_ok": execute_process_ok,
                "output_error": output_error,
                "schema_warnings": schema_warnings,
                "bypass_findings": bypass_findings[:10],
                "operator_use_findings": operator_use_findings[:10],
                "plan_exit_code": plan_raw.get("exit_code"),
                "plan_timed_out": plan_raw.get("timed_out"),
                "execute_exit_code": execute_raw.get("exit_code") if execute_raw else None,
                "execute_timed_out": execute_raw.get("timed_out") if execute_raw else None,
                "script_path": str(script_path) if script_path else None,
                "script_selection_error": script_selection_error,
                "attempt_scratch_dir": str(task_attempt_tmp_dir_path),
                "script_run": script_run_failure_summary(execute_raw),
                "preexisting_task_processes": preexisting_task_processes,
                "lingering_task_processes": lingering_task_processes,
                "elapsed_ms": to_int(metrics["total"]["elapsed_seconds"] * 1000),
                "plan_agent_elapsed_seconds": metrics["plan_generation"]["elapsed_seconds"],
                "plan_agent_cost_usd": metrics["plan_generation"]["agent_cost_usd"],
                "execute_agent_elapsed_seconds": metrics["execution"]["agent"]["elapsed_seconds"],
                "execute_agent_cost_usd": metrics["execution"]["agent"]["agent_cost_usd"],
                "agent_elapsed_seconds": metrics["total"]["agent_elapsed_seconds"],
                "agent_cost_usd": metrics["total"]["agent_cost_usd"],
                "agent_total_tokens": metrics["total"]["agent_total_tokens"],
                "system_elapsed_seconds": metrics["execution"]["system_elapsed_seconds"],
                "system_cost_usd": metrics["execution"]["system_cost_usd"],
                "system_usage": system_usage,
                "system_input_tokens": system_usage["input_tokens"],
                "system_output_tokens": system_usage["output_tokens"],
                "system_total_token": system_usage["total_tokens"],
                "total_elapsed_seconds": metrics["total"]["elapsed_seconds"],
                "total_cost_usd": metrics["total"]["cost_usd"],
                "total_tokens": metrics["total"]["total_tokens"],
                "output_file": str(output_file),
            }
            attempt_summaries.append(attempt_summary)
            cumulative_metrics = cumulative_attempt_metrics(attempt_summaries)
            metrics_file = task_metrics_file(paths, task_id)
            entry = {
                "task_id": task_id,
                "query": task.get("query", ""),
                "answer_schema": task.get("answer_schema"),
                "system_name": args.system_name,
                "pipeline_skill": str(args.pipeline_skill_path),
                "ok": ok,
                "attempt": attempt,
                "max_attempts": args.task_max_attempts,
                "attempts": attempt_summaries,
                "process_ok": process_ok,
                "output_error": output_error,
                "schema_valid": not schema_warnings,
                "schema_warnings": schema_warnings,
                "bypass_valid": not bypass_findings,
                "bypass_findings": bypass_findings[:20],
                "operator_use_valid": not operator_use_findings,
                "operator_use_findings": operator_use_findings[:20],
                "plan_ok": plan_ok,
                "plan_error": plan_error,
                "exit_code": execute_raw.get("exit_code") if execute_raw else plan_raw.get("exit_code"),
                "timed_out": execute_raw.get("timed_out") if execute_raw else plan_raw.get("timed_out"),
                "elapsed_ms": to_int(cumulative_metrics["total_elapsed_seconds"] * 1000),
                "elapsed_secs": elapsed_secs,
                "cost_usd": cost_usd,
                "final_system_elapsed_seconds": elapsed_secs,
                "final_system_cost_usd": cost_usd,
                "final_system_usage": system_usage,
                "final_system_total_token": system_usage["total_tokens"],
                "system_elapsed_seconds": cumulative_metrics["system_elapsed_seconds"],
                "system_cost_usd": cumulative_metrics["system_cost_usd"],
                "system_input_tokens": cumulative_metrics["system_input_tokens"],
                "system_output_tokens": cumulative_metrics["system_output_tokens"],
                "system_total_token": cumulative_metrics["system_total_token"],
                "plan_agent_elapsed_seconds": cumulative_metrics["plan_agent_elapsed_seconds"],
                "plan_agent_cost_usd": cumulative_metrics["plan_agent_cost_usd"],
                "execute_agent_elapsed_seconds": cumulative_metrics["execute_agent_elapsed_seconds"],
                "execute_agent_cost_usd": cumulative_metrics["execute_agent_cost_usd"],
                "agent_elapsed_seconds": cumulative_metrics["agent_elapsed_seconds"],
                "agent_cost_usd": cumulative_metrics["agent_cost_usd"],
                "total_elapsed_seconds": cumulative_metrics["total_elapsed_seconds"],
                "total_cost_usd": cumulative_metrics["total_cost_usd"],
                "total_tokens": cumulative_metrics["total_tokens"],
                "metrics_file": str(metrics_file),
                "answer": answer,
                "output_file": str(output_file),
                "raw_file": str(raw_file),
                "raw_attempt_file": str(raw_attempt_file),
                "plan_file": str(plan_file),
                "plan_raw_file": str(plan_raw_file),
                "plan_raw_attempt_file": str(plan_raw_attempt_file),
                "execute_raw_file": str(execute_raw_file) if execute_raw_file else None,
                "execute_raw_attempt_file": str(execute_raw_attempt_file) if execute_raw_attempt_file else None,
                "script_path": str(script_path) if script_path else None,
            }
            existing[task_id] = entry
            final_entry = entry
            write_json(metrics_file, {"task_id": task_id, "system_name": args.system_name, "attempt": attempt, "ok": ok, "metrics": metrics, "attempts": attempt_summaries})
            write_results_jsonl(paths["results_jsonl"], all_tasks, existing)
            write_result_files(paths["results_dir"], all_tasks, existing)
            write_metrics_summary(paths, args, all_tasks, existing)
            print(
                f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {task_id} attempt {attempt}/{args.task_max_attempts} "
                f"{'ok' if ok else 'failed'} plan_cost=${metrics['plan_generation']['agent_cost_usd']:.6f} "
                f"exec_agent_cost=${metrics['execution']['agent']['agent_cost_usd']:.6f} "
                f"system_cost=${metrics['execution']['system_cost_usd']:.6f} "
                f"attempt_total=${metrics['total']['cost_usd']:.6f} cumulative_total=${cumulative_metrics['total_cost_usd']:.6f}",
                flush=True,
            )
            if ok:
                break

        if not final_entry or final_entry.get("ok") is not True:
            print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {task_id} failed after {args.task_max_attempts} attempts; stopping before next task.", file=sys.stderr, flush=True)
            return 1
    print(f"Completed. Final results dir: {paths['results_dir']}")
    return 0


def main() -> int:
    args = parse_args()
    args.system_path = args.system_path.resolve()
    args.system_name = args.system_name or infer_system_name(args.system_path)
    args.system_package = args.system_package or args.system_name
    args.pipeline_skill_path = (args.pipeline_skill_path or default_pipeline_skill_path(args.system_name)).resolve()
    args.analysis_skill = args.analysis_skill.resolve()
    args.env_setup_skill = args.env_setup_skill.resolve()
    args.env_record = (args.env_record or default_env_record_path(args.system_path)).resolve()
    args.env_path = args.env_path.resolve()
    args.agent_env_path = args.agent_env_path.resolve()
    args.results_root = args.results_root.resolve()
    args.run_dir = args.run_dir.resolve()
    args.input = args.input.resolve()
    args.data_dir = args.data_dir.resolve()

    require_path("target system directory", args.system_path)
    require_path("Phase 0 env setup skill", args.env_setup_skill)
    require_path("env/config file", args.env_path)
    require_path("agent env/config file", args.agent_env_path)
    if needs_system_analysis(args.mode):
        require_path("system analysis skill", args.analysis_skill)
    if needs_task_run(args.mode):
        require_path("benchmark input file", args.input)
        require_path("data directory", args.data_dir)
    for label, path in (
        ("target system directory", args.system_path),
        ("pipeline skill directory", args.pipeline_skill_path.parent),
        ("Phase 0 env record", args.env_record),
        ("env/config file", args.env_path),
        ("agent env/config file", args.agent_env_path),
        ("benchmark input file", args.input),
        ("data directory", args.data_dir),
        ("run output directory", args.run_dir),
        ("results root", args.results_root),
    ):
        require_under_root(label, path)

    paths = ensure_runtime_dirs(args)
    print("Claude Code operator runner")
    print(f"Mode: {args.mode}")
    print(f"System: {args.system_name} ({args.system_path})")
    print(f"Env/config: {args.env_path}")
    print(f"Agent env/config: {args.agent_env_path}")
    print(f"Env record: {args.env_record}")
    print(f"Pipeline skill: {args.pipeline_skill_path}")
    print(f"Claude Code command: {args.claude_command}")
    print(f"Run dir: {paths['run_dir']}")

    system_env = parse_dotenv(args.env_path)
    system_env.update(os.environ)
    system_env["LLM_MODEL_NAME"] = TEXT_MODEL
    system_env["LLM_MAIN_MODEL"] = TEXT_MODEL
    system_env["LLM_MAX_BATCH_SIZE"] = str(args.model_concurrency)
    synced_mappings = sync_runner_env_mappings(system_env)
    missing = missing_runner_env_groups(system_env)
    if missing and not args.dry_run:
        raise SystemExit("runner is missing recognizable config: " + ", ".join(missing))
    if synced_mappings:
        print(f"[env] applied {len(synced_mappings)} runner env mappings")
    agent_env = agent_env_with_process_overrides(parse_dotenv(args.agent_env_path))
    agent_env.setdefault("AGENT_API_BASE", system_env.get("LLM_API_BASE", ""))
    agent_env.setdefault("AGENT_API_KEY", system_env.get("OPENAI_API_KEY", ""))
    agent_env["AGENT_MAIN_MODEL"] = TEXT_MODEL
    synced_agent_mappings = sync_agent_env_mappings(agent_env)
    missing_agent = missing_agent_env_groups(agent_env)
    if missing_agent and not args.dry_run:
        raise SystemExit("agent env/config is missing recognizable config: " + ", ".join(missing_agent))
    if synced_agent_mappings:
        print(f"[agent-env] applied {len(synced_agent_mappings)} agent env mappings")
    if env_value_is_set(agent_env.get("AGENT_MAIN_MODEL")):
        print(f"Agent model: {agent_env['AGENT_MAIN_MODEL']}")

    if args.mode == "env-setup":
        return run_env_setup(args, system_env, agent_env, paths)

    if args.dry_run:
        if args.env_record.exists():
            print("[phase0] ENV.md ready")
        else:
            print_env_setup_guidance(args)
    else:
        env_record_present = require_env_record(args)
        if env_record_present:
            print("[phase0] ENV.md ready")

    if args.dry_run:
        if needs_system_analysis(args.mode):
            print(f"[dry-run] would analyze system and write {args.pipeline_skill_path}")
        if needs_task_run(args.mode):
            all_tasks, _ = load_tasks(args.input, None, None)
            selected_tasks = select_testbed_tasks(all_tasks, args)
            existing = read_existing_results(paths["results_jsonl"])
            for task_id, entry in read_compact_results(paths["results_dir"], all_tasks).items():
                existing.setdefault(task_id, entry)
            pending = selected_tasks if args.force else [task for task in selected_tasks if existing.get(task.get("task_id"), {}).get("ok") is not True]
            print(f"[dry-run] selected={len(selected_tasks)} pending={len(pending)}")
            for task in pending:
                print(f"{task['task_id']}: {task.get('query', '')}")
        return 0

    if needs_system_analysis(args.mode):
        run_system_analysis(args, system_env, agent_env, paths)
    if needs_task_run(args.mode):
        return run_tasks(args, system_env, agent_env, paths)
    return 0


if __name__ == "__main__":
    install_shutdown_handlers()
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        terminate_active_processes("keyboard interrupt")
        print("\nInterrupted. Runner-owned child process groups were cleaned up.", file=sys.stderr, flush=True)
        raise SystemExit(130)
