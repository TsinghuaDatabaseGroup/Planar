# Testbed Runners

This directory contains query pipelines and launchers for five systems:

- `docetl`
- `lotus`
- `palimpzest`
- `claude_code`
- `claude_code_plus_lotus`

The launchers cover four domains:

| Domain | Queries per system |
| --- | ---: |
| `aviation_safety` | 48 |
| `vehicle_safety` | 47 |
| `finance` | 42 |
| `legal_contracts` | 106 |
| **Total** | **243** |

## Configuration

Create the local runtime configuration before executing a pipeline:

```bash
cp testbed/runner/.env.example testbed/runner/.env
```

Set the API key and compatible API endpoints in `.env`. The text model is
`Qwen3.5-397B-A17B`.

## Result Contract

All runners write results to:

```text
testbed/results/<domain>/<difficulty>/<system>/<task_id>.json
```

Every result contains exactly these fields:

```json
{
  "task_id": "aviation_safety-001",
  "answer": null,
  "elapsed_seconds": 0.0,
  "cost_usd": 0.0,
  "total_tokens": 0
}
```

## Running Pipelines

Run commands from the testbed repository root:

```bash
python testbed/runner/docetl/run_all_pipelines.py \
  --dataset aviation_safety --start 1 --end 48

python testbed/runner/lotus/run_all_pipelines.py \
  --dataset vehicle_safety --start 1 --end 47

python testbed/runner/palimpzest/run_all_pipelines.py \
  --dataset finance --start 1 --end 42

python testbed/runner/claude_code/run_all_pipelines.py \
  --dataset legal_contracts --start 1 --end 106

python testbed/runner/claude_code_plus_lotus/run_all_pipelines.py \
  --dataset aviation_safety --start 1 --end 48
```

Each runner defaults to a model concurrency limit of 10. Repeated
`--task-id` arguments select individual tasks, and `--force` replaces existing
successful results.

The Claude Code launchers preserve generated plans, scripts, diagnostics,
retry state, caches, and aggregate runtime state under each launcher's
`runs/<domain>` directory.
