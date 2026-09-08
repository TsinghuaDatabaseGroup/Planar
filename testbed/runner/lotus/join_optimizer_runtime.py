"""Enable LOTUS native semantic-join optimization without audit artifacts."""

from __future__ import annotations

import functools
import importlib
import math
import os
from pathlib import Path

import pandas as pd

import pipeline_helpers
from lotus.sem_ops.sem_join import SemJoinDataframe
from lotus.types import CascadeArgs

JOIN_OPTIMIZATION_ENV = pipeline_helpers.JOIN_OPTIMIZATION_ENV
CACHE_DIR_ENV = pipeline_helpers.CACHE_DIR_ENV

_ENABLED = False
_PREVIOUS_ENV: dict[str, str | None] = {}

PAPER_RECALL_TARGET = 0.9
PAPER_PRECISION_TARGET = 0.9
PAPER_FAILURE_PROBABILITY = 0.2
PAPER_SAMPLE_FRACTION = 0.0001
PAPER_MIN_SAMPLE_SIZE = 100


def _paper_join_sample_size(candidate_pairs: int) -> int:
    if candidate_pairs <= 0:
        return 0
    percentage_sample = math.ceil(PAPER_SAMPLE_FRACTION * candidate_pairs)
    return min(candidate_pairs, max(PAPER_MIN_SAMPLE_SIZE, percentage_sample))


def _paper_join_cascade_args(candidate_pairs: int) -> CascadeArgs:
    sample_size = _paper_join_sample_size(candidate_pairs)
    sampling_percentage = (
        sample_size / candidate_pairs if candidate_pairs > 0 else 0.0
    )
    if (
        candidate_pairs > 0
        and int(sampling_percentage * candidate_pairs) < sample_size
    ):
        sampling_percentage = math.nextafter(
            sampling_percentage,
            math.inf,
        )
    return CascadeArgs(
        recall_target=PAPER_RECALL_TARGET,
        precision_target=PAPER_PRECISION_TARGET,
        sampling_percentage=sampling_percentage,
        failure_probability=PAPER_FAILURE_PROBABILITY,
        cascade_IS_max_sample_range=candidate_pairs,
    )


def _normalize_native_join_input(
    value: pd.DataFrame | pd.Series,
) -> pd.DataFrame | pd.Series:
    expected_index = pd.RangeIndex(len(value))
    reset_index = not value.index.equals(expected_index)
    invalid_index_dirs = value.attrs.get("index_dirs", {}) is None
    if not reset_index and not invalid_index_dirs:
        return value

    normalized = (
        value.reset_index(drop=True) if reset_index else value.copy(deep=False)
    )
    normalized.attrs = dict(value.attrs)
    normalized.attrs.pop("index_dirs", None)
    return normalized


def configure(cache_dir: str | os.PathLike[str]) -> None:
    """Enable the native join cascade for this process."""
    global _ENABLED, _PREVIOUS_ENV

    if not _ENABLED:
        _PREVIOUS_ENV = {
            JOIN_OPTIMIZATION_ENV: os.environ.get(JOIN_OPTIMIZATION_ENV),
            CACHE_DIR_ENV: os.environ.get(CACHE_DIR_ENV),
        }
    os.environ[JOIN_OPTIMIZATION_ENV] = "1"
    os.environ[CACHE_DIR_ENV] = str(Path(cache_dir).expanduser().resolve())
    _ENABLED = True
    _install_join_plan_budget_patch()
    _install_sem_join_patch()


def disable() -> None:
    """Disable the runtime switch and restore its environment."""
    global _ENABLED, _PREVIOUS_ENV

    _ENABLED = False
    for key in (JOIN_OPTIMIZATION_ENV, CACHE_DIR_ENV):
        previous = _PREVIOUS_ENV.get(key)
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous
    _PREVIOUS_ENV = {}


def _install_join_plan_budget_patch() -> None:
    sem_join_module = importlib.import_module("lotus.sem_ops.sem_join")
    current = sem_join_module.join_optimizer
    if getattr(current, "_lotus_join_plan_budget_checked", False):
        return

    @functools.wraps(current)
    def budget_checked_join_optimizer(*args, **kwargs):
        result = current(*args, **kwargs)
        if not _ENABLED:
            return result
        model = args[4] if len(args) > 4 else kwargs.get("model")
        enforce_budget = getattr(model, "_enforce_task_time_budget", None)
        if callable(enforce_budget):
            enforce_budget(len(result[1]))
        return result

    budget_checked_join_optimizer._lotus_join_plan_budget_checked = True  # type: ignore[attr-defined]
    sem_join_module.join_optimizer = budget_checked_join_optimizer


def _install_sem_join_patch() -> None:
    if getattr(SemJoinDataframe.__call__, "_lotus_native_join_optimized", False):
        return

    original_sem_join = SemJoinDataframe.__call__

    @functools.wraps(original_sem_join)
    def optimized_sem_join(self, other, join_instruction, *args, **kwargs):
        if not _ENABLED:
            return original_sem_join(
                self,
                other,
                join_instruction,
                *args,
                **kwargs,
            )

        candidate_pairs = len(self._obj) * len(other)
        call_args = list(args)
        cascade_position = 6
        if len(call_args) > cascade_position:
            if call_args[cascade_position] is None:
                call_args[cascade_position] = _paper_join_cascade_args(
                    candidate_pairs
                )
        elif kwargs.get("cascade_args") is None:
            kwargs["cascade_args"] = _paper_join_cascade_args(candidate_pairs)

        normalized_left = _normalize_native_join_input(self._obj)
        normalized_other = _normalize_native_join_input(other)
        normalized_accessor = (
            self
            if normalized_left is self._obj
            else SemJoinDataframe(normalized_left)
        )
        return original_sem_join(
            normalized_accessor,
            normalized_other,
            join_instruction,
            *call_args,
            **kwargs,
        )

    optimized_sem_join._lotus_native_join_optimized = True  # type: ignore[attr-defined]
    SemJoinDataframe.__call__ = optimized_sem_join
