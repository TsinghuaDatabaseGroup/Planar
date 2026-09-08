# Planar

Planar is a benchmark for semantic query processing over heterogeneous data. It provides real-world workloads with gold answers, reference plans, and optimization opportunities. These enable unified evaluation of SQPEs, agents, and hybrid systems, with analysis of planning, optimization, and operator execution.

The benchmark contains **243 queries** across four real-world domains and three complexity levels (`easy`, `medium`, and `hard`).

## Community

We deeply appreciate the invaluable effort contributed by our dedicated team of developers and esteemed industry partners.

- [Tsinghua University](https://www.tsinghua.edu.cn/en)
- [Ant Digital Technologies, Ant Group](https://intl.antdigital.com/en)
- [National University of Singapore](https://nus.edu.sg/)

## Benchmark Data

- **Datasets:** [Download from Google Drive](https://drive.google.com/drive/folders/1wMy6zqkUfnq8VQP28U09H8GeW85qHh22?usp=sharing). The included [data guide](data/README.md) describes the additional inputs shipped with this repository.
- **Queries:** [`testbed/queries/`](testbed/queries/) contains the public task ID, complexity, and natural-language query for each domain.
- **Ground Truth & Annotated Semantic Plans:** [`testbed/gold/`](testbed/gold/) contains gold answers, annotated semantic plans.

| Domain | Queries |
| --- | ---: |
| Aviation Safety | 48 |
| Vehicle Safety | 47 |
| Legal Contracts | 106 |
| Finance | 42 |
| **Total** | **243** |

## Quickstart

### 1. Download the datasets

Download the four domain datasets from the [Planar data folder](https://drive.google.com/drive/folders/1wMy6zqkUfnq8VQP28U09H8GeW85qHh22?usp=sharing) and extract them under `data/`. See [`data/README.md`](data/README.md) for the included operator-level inputs.

### 2. Install

Create a separate environment for each execution backend because their native dependency versions are not mutually compatible.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r testbed/requirements.txt
```

### 3. Configure model endpoints

To rerun pipelines, create the shared runner configuration:

```bash
cp testbed/runner/.env.example testbed/runner/.env
```

Set `OPENAI_API_KEY`, `LLM_API_BASE`, and `EMBEDDING_API_BASE` in `testbed/runner/.env`.

### 4. Run a pipeline

For example, run one LOTUS task:

```bash
python testbed/runner/lotus/run_all_pipelines.py \
  --dataset aviation_safety \
  --task-id aviation_safety-001
```

The available domain names are `aviation_safety`, `vehicle_safety`, `legal_contracts`, and `finance`. See [`testbed/runner/README.md`](testbed/runner/README.md) for all five launchers, concurrency controls, task ranges, timeouts, and result paths.

Every runner emits the same result contract:

```json
{
  "task_id": "aviation_safety-001",
  "answer": null,
  "elapsed_seconds": 0.0,
  "cost_usd": 0.0,
  "total_tokens": 0
}
```

Results are written to:

```text
testbed/results/<domain>/<difficulty>/<system>/<task_id>.json
```

Use `--force` when an existing successful result should be replaced.
