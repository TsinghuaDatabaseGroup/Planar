#!/usr/bin/env python3
"""Self-contained output-aware scoring for semantic-query answers.

Numeric values use clipped one-minus relative absolute error. Structured rows
are aligned one-to-one by query-specific or deterministically inferred keys,
and non-key cell quality forms a soft table-level F1. Ordered tables multiply
that soft F1 by NDCG over the relative order of matched rows.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


DEFAULT_PROFILE = "soft_f1_ndcg_final_v1"
SCORING_PROFILES = (DEFAULT_PROFILE,)
ORDERED_OUTPUT_TYPES = {"ordered_list", "ordered_table"}
ORDERED_METRIC = "soft_table_f1_x_matched_order_ndcg"

MEASURE_TOKENS = {
    "avg",
    "average",
    "count",
    "median",
    "percentage",
    "rate",
    "share",
    "sum",
}
CATEGORICAL_FIELD_TOKENS = {
    "actor",
    "category",
    "label",
    "mechanism",
    "mode",
    "pattern",
    "phase",
    "role",
    "sentiment",
    "severity",
    "status",
    "style",
    "theme",
    "type",
}
STOPWORDS = {"a", "an", "the"}
ENTITY_SUFFIX_TOKENS = {
    "co",
    "companies",
    "company",
    "corp",
    "corporation",
    "group",
    "holding",
    "holdings",
    "inc",
    "incorporated",
    "limited",
    "llc",
    "llp",
    "ltd",
    "plc",
}
KEY_DESCRIPTION_STOPWORDS = {
    "a",
    "an",
    "dictionary",
    "group",
    "id",
    "key",
    "name",
    "of",
    "or",
    "string",
    "the",
    "type",
}


def strip_answer_prefix(value: str) -> str:
    return re.sub(
        r"^\s*(?:final\s+)?answer\s*[:：-]\s*", "", value, flags=re.IGNORECASE
    )


def normalize_text(value: Any) -> str:
    value = strip_answer_prefix(str(value)).lower().replace("_", " ")
    value = value.translate(str.maketrans({char: " " for char in string.punctuation}))
    tokens = [token for token in value.split() if token not in STOPWORDS]
    return " ".join(tokens)


def field_tokens(name: str) -> list[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return re.findall(r"[a-z0-9]+", expanded.lower())


def base_type(type_name: Any) -> str:
    parts = [normalize_text(part) for part in str(type_name or "unknown").split("|")]
    parts = [part for part in parts if part and part not in {"null", "none"}]
    value = parts[0] if parts else "unknown"
    aliases = {
        "int": "integer",
        "double": "float",
        "number": "float",
        "numeric": "float",
        "bool": "boolean",
        "str": "string",
        "text": "string",
        "array": "list",
        "dict": "dictionary",
        "object": "dictionary",
    }
    if value.startswith(("list ", "array ")):
        return "list"
    for candidate in (
        "integer",
        "float",
        "boolean",
        "date",
        "string",
        "dictionary",
        "list",
    ):
        if value.startswith(candidate + " "):
            return candidate
    return aliases.get(value, value)


def schema_columns(schema: dict[str, Any]) -> tuple[list[str], list[str]]:
    raw = schema.get("columns") or []
    if isinstance(raw, dict):
        return list(raw), [base_type(value) for value in raw.values()]
    columns = [str(value) for value in raw]
    types = [base_type(value) for value in schema.get("types") or []]
    types.extend(["unknown"] * (len(columns) - len(types)))
    return columns, types[: len(columns)]


def parse_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).strip().replace(",", ""))
    return float(match.group()) if match else None


def relative_accuracy(gold: Any, pred: Any) -> float:
    """Return clipped one-minus gold-relative absolute error."""
    left, right = parse_number(gold), parse_number(pred)
    if left is None or right is None:
        return 1.0 if normalize_text(gold) == normalize_text(pred) else 0.0
    if "%" in str(pred) and "%" not in str(gold) and abs(left) <= 1.0:
        right /= 100.0
    if "%" in str(gold) and "%" not in str(pred) and abs(right) <= 1.0:
        left /= 100.0
    if left == 0.0:
        return 1.0 if right == 0.0 else 0.0
    return max(0.0, 1.0 - abs(left - right) / abs(left))


def numeric_similarity(gold: Any, pred: Any) -> float:
    return relative_accuracy(gold, pred)


def exact_numeric_similarity(gold: Any, pred: Any) -> float:
    left, right = parse_number(gold), parse_number(pred)
    if left is None or right is None:
        return float(normalize_text(gold) == normalize_text(pred))
    if "%" in str(pred) and "%" not in str(gold) and abs(left) <= 1.0:
        right /= 100.0
    if "%" in str(gold) and "%" not in str(pred) and abs(right) <= 1.0:
        left /= 100.0
    return float(left == right)


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = normalize_text(value)
    if normalized in {"true", "yes", "y", "1"}:
        return True
    if normalized in {"false", "no", "n", "0"}:
        return False
    return None


def parse_date(value: Any) -> str | None:
    text = str(value).strip()
    for pattern in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y"):
        try:
            return dt.datetime.strptime(text, pattern).date().isoformat()
        except ValueError:
            pass
    return None


def is_identifier_field(name: str) -> bool:
    tokens = field_tokens(name)
    if not tokens:
        return False
    if tokens[-1] in {"id", "ids"}:
        return True
    return tokens[-1] in {"number", "numbers"} and bool(
        set(tokens) & {"campaign", "recall"}
    )


def canonical_identifier(value: Any, field_name: str) -> str:
    text = strip_answer_prefix(str(value)).strip().replace("\\", "/")
    tokens = set(field_tokens(field_name))
    if tokens & {"campaign", "recall"}:
        compact = re.sub(r"[^A-Z0-9]", "", text.upper())
        match = re.fullmatch(r"(?:20)?(\d{2})V(\d{1,6})", compact)
        if match:
            year, sequence = match.groups()
            if len(sequence) < 6:
                number = int(sequence)
                sequence = f"{number:03d}000" if number <= 999 else f"{number:06d}"
            return f"{year}v{sequence}"
    if tokens & {"file", "report", "document"}:
        text = text.rsplit("/", 1)[-1]
        text = re.sub(r"\.txt$", "", text, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]", "", text.lower())


def identifier_similarity(gold: Any, pred: Any, field_name: str) -> float:
    return float(
        canonical_identifier(gold, field_name)
        == canonical_identifier(pred, field_name)
    )


def harmonic(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def token_f1(gold: Any, pred: Any) -> float:
    left_tokens = normalize_text(gold).split()
    right_tokens = normalize_text(pred).split()
    if left_tokens == right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = sum((Counter(left_tokens) & Counter(right_tokens)).values())
    if overlap == 0:
        return 0.0
    return harmonic(overlap / len(right_tokens), overlap / len(left_tokens))


def initialism(value: Any) -> str:
    raw = strip_answer_prefix(str(value)).strip()
    words = re.findall(r"[A-Za-z]+", raw)
    if not words:
        return ""
    if len(words) == 1:
        token = words[0]
        return token.lower() if 2 <= len(token) <= 12 and token.isupper() else ""
    if all(len(word) == 1 for word in words):
        return "".join(words).lower()
    return ""


def name_initials(value: Any) -> str:
    words = [
        word
        for word in normalize_text(value).split()
        if word not in ENTITY_SUFFIX_TOKENS
    ]
    return "".join(word[0] for word in words) if len(words) >= 2 else ""


def is_initialism_alias(left: Any, right: Any) -> bool:
    left_initialism = initialism(left)
    right_initialism = initialism(right)
    return bool(
        (left_initialism and left_initialism == name_initials(right))
        or (right_initialism and right_initialism == name_initials(left))
    )


def short_text_similarity(gold: Any, pred: Any) -> float:
    if normalize_text(gold) == normalize_text(pred):
        return 1.0
    if is_initialism_alias(gold, pred):
        return 1.0
    return token_f1(gold, pred)


def named_entity_similarity(gold: Any, pred: Any) -> float:
    if normalize_text(gold) == normalize_text(pred):
        return 1.0
    if is_initialism_alias(gold, pred):
        return 1.0
    left_tokens = normalize_text(gold).split()
    right_tokens = normalize_text(pred).split()
    left_core = [token for token in left_tokens if token not in ENTITY_SUFFIX_TOKENS]
    right_core = [token for token in right_tokens if token not in ENTITY_SUFFIX_TOKENS]
    if left_core and left_core == right_core:
        return 1.0
    if not (Counter(left_core) & Counter(right_core)):
        return 0.0
    return token_f1(gold, pred)


def is_categorical_field(field_name: str, gold: Any) -> bool:
    text = str(gold or "")
    if "_" in text:
        return True
    if not set(field_tokens(field_name)) & CATEGORICAL_FIELD_TOKENS:
        return False
    return " " not in text.strip()


def text_similarity(gold: Any, pred: Any, *, field_name: str = "") -> float:
    left_bool, right_bool = parse_bool(gold), parse_bool(pred)
    if left_bool is not None and right_bool is not None:
        return float(left_bool == right_bool)
    if is_categorical_field(field_name, gold):
        return float(normalize_text(gold) == normalize_text(pred))
    return token_f1(gold, pred)


def coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        if ";" in text or "\n" in text:
            return [part.strip() for part in re.split(r";|\n", text) if part.strip()]
    return [value]


def maximum_weight_pairs(
    weights: Sequence[Sequence[float]],
) -> list[tuple[int, int, float]]:
    """Return a deterministic maximum-weight one-to-one assignment."""
    if not weights or not weights[0]:
        return []
    original = [list(row) for row in weights]
    matrix = original
    transposed = False
    if len(matrix) > len(matrix[0]):
        matrix = [list(row) for row in zip(*matrix)]
        transposed = True

    row_count, column_count = len(matrix), len(matrix[0])
    maximum = max(max(row) for row in matrix)
    costs = [[maximum - value for value in row] for row in matrix]
    row_potential = [0.0] * (row_count + 1)
    column_potential = [0.0] * (column_count + 1)
    matching = [0] * (column_count + 1)
    path = [0] * (column_count + 1)

    for row_idx in range(1, row_count + 1):
        matching[0] = row_idx
        column_idx = 0
        minimum = [math.inf] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[column_idx] = True
            current_row = matching[column_idx]
            delta = math.inf
            next_column = 0
            for candidate in range(1, column_count + 1):
                if used[candidate]:
                    continue
                reduced = (
                    costs[current_row - 1][candidate - 1]
                    - row_potential[current_row]
                    - column_potential[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    path[candidate] = column_idx
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(column_count + 1):
                if used[candidate]:
                    row_potential[matching[candidate]] += delta
                    column_potential[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column_idx = next_column
            if matching[column_idx] == 0:
                break
        while True:
            previous = path[column_idx]
            matching[column_idx] = matching[previous]
            column_idx = previous
            if column_idx == 0:
                break

    pairs = []
    for column_idx in range(1, column_count + 1):
        if matching[column_idx] == 0:
            continue
        left, right = matching[column_idx] - 1, column_idx - 1
        gold_idx, pred_idx = (right, left) if transposed else (left, right)
        value = original[gold_idx][pred_idx]
        if value > 0.0:
            pairs.append((gold_idx, pred_idx, value))
    return sorted(pairs)


def infer_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dictionary"
    return "string"


def list_similarity(
    gold: Any,
    pred: Any,
    field_name: str = "",
    *,
    exact: bool = False,
) -> float:
    gold_items, pred_items = coerce_list(gold), coerce_list(pred)
    if not gold_items and not pred_items:
        return 1.0
    if not gold_items or not pred_items:
        return 0.0
    weights = [
        [
            value_similarity(
                gold_item,
                pred_item,
                infer_type(gold_item),
                field_name=field_name,
                exact=exact,
            )
            for pred_item in pred_items
        ]
        for gold_item in gold_items
    ]
    matched_mass = sum(weight for _, _, weight in maximum_weight_pairs(weights))
    return harmonic(matched_mass / len(pred_items), matched_mass / len(gold_items))


def value_similarity(
    gold: Any,
    pred: Any,
    value_type: str = "unknown",
    *,
    field_name: str = "",
    exact: bool = False,
) -> float:
    if gold is None and pred is None:
        return 1.0
    if gold is None or pred is None:
        return 0.0
    value_type = base_type(value_type)
    if value_type == "unknown":
        value_type = infer_type(gold)
    if is_identifier_field(field_name):
        return identifier_similarity(gold, pred, field_name)
    if value_type in {"integer", "float"}:
        return (
            exact_numeric_similarity(gold, pred)
            if exact
            else relative_accuracy(gold, pred)
        )
    if value_type == "boolean":
        left, right = parse_bool(gold), parse_bool(pred)
        return float(left is not None and left == right)
    if value_type == "date":
        left, right = parse_date(gold), parse_date(pred)
        if left is not None and right is not None:
            return float(left == right)
        return float(normalize_text(gold) == normalize_text(pred))
    if value_type == "list" or isinstance(gold, list):
        return list_similarity(gold, pred, field_name, exact=exact)
    if value_type == "dictionary" or isinstance(gold, dict):
        return dictionary_similarity(gold, pred, {}, exact=exact)["score"]
    if exact:
        return float(normalize_text(gold) == normalize_text(pred))
    return text_similarity(gold, pred, field_name=field_name)


def normalize_rows(value: Any, columns: Sequence[str]) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            return normalize_rows(json.loads(value), columns)
        except json.JSONDecodeError:
            return [{columns[0]: value}] if len(columns) == 1 else []
    if isinstance(value, dict):
        for key in ("rows", "data", "items", "results", "answer"):
            if isinstance(value.get(key), list):
                return normalize_rows(value[key], columns)
        if any(column in value for column in columns):
            return [value]
        return []
    if not isinstance(value, list):
        return [{columns[0]: value}] if len(columns) == 1 else []
    rows = []
    for item in value:
        if isinstance(item, dict):
            rows.append(item)
        elif isinstance(item, (list, tuple)):
            rows.append(
                {
                    column: item[idx] if idx < len(item) else None
                    for idx, column in enumerate(columns)
                }
            )
        elif len(columns) == 1:
            rows.append({columns[0]: item})
    return rows


def normalize_dictionary_prediction(value: Any, schema: dict[str, Any]) -> Any:
    """Convert an unambiguous row-oriented dictionary representation."""
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(row, dict) for row in value)
    ):
        return value
    description_tokens = set(field_tokens(schema.get("key_description", "")))
    description_tokens -= KEY_DESCRIPTION_STOPWORDS
    if not description_tokens:
        return value

    candidates = list(value[0])
    ranked = []
    for candidate in candidates:
        candidate_tokens = set(field_tokens(candidate))
        overlap = len(candidate_tokens & description_tokens)
        coverage = overlap / len(candidate_tokens) if candidate_tokens else 0.0
        ranked.append(((coverage, overlap), candidate))
    best_score = max(score for score, _ in ranked)
    key_fields = [candidate for score, candidate in ranked if score == best_score]
    if best_score[1] == 0 or len(key_fields) != 1:
        return value
    key_field = key_fields[0]
    if any(key_field not in row or row[key_field] is None for row in value):
        return value

    normalized: dict[str, Any] = {}
    for row in value:
        key = str(row[key_field])
        if key in normalized:
            return value
        normalized[key] = {
            name: item for name, item in row.items() if name != key_field
        }
    return normalized


def unique_with_columns(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> bool:
    signatures = [
        tuple(normalize_text(row.get(column)) for column in columns) for row in rows
    ]
    return len(signatures) == len(set(signatures)) and all(
        any(signature) for signature in signatures
    )


def declared_row_keys(
    schema: dict[str, Any], columns: Sequence[str]
) -> list[str]:
    """Read query-specific row keys when the answer schema declares them."""
    for field in ("row_keys", "key_columns", "primary_key"):
        raw = schema.get(field)
        if isinstance(raw, str):
            requested = [raw]
        elif isinstance(raw, list):
            requested = [str(value) for value in raw]
        else:
            continue
        selected = [value for value in requested if value in columns and value != "rank"]
        if selected:
            return selected
    return []


def infer_row_keys(
    columns: Sequence[str],
    types: Sequence[str],
    gold_rows: Sequence[dict[str, Any]],
) -> list[str]:
    explicit_ids = []
    for column, value_type in zip(columns, types):
        tokens = field_tokens(column)
        if (
            column != "rank"
            and base_type(value_type) != "list"
            and tokens
            and tokens[-1] not in {"ids", "numbers"}
            and is_identifier_field(column)
        ):
            explicit_ids.append(column)
    if explicit_ids:
        selected: list[str] = []
        for column in explicit_ids:
            selected.append(column)
            if unique_with_columns(gold_rows, selected):
                break
        return selected

    dimensions: list[str] = []
    for column, value_type in zip(columns, types):
        if column == "rank":
            continue
        tokens = set(field_tokens(column))
        if tokens & MEASURE_TOKENS or base_type(value_type) in {"integer", "float"}:
            break
        dimensions.append(column)
    if dimensions:
        selected = []
        for column in dimensions:
            selected.append(column)
            if unique_with_columns(gold_rows, selected):
                break
        return selected
    return [column for column in columns if column != "rank"][:1]


def key_similarity(
    gold_row: dict[str, Any],
    pred_row: dict[str, Any],
    keys: Sequence[str],
    types: dict[str, str],
) -> float:
    for key in keys:
        gold, pred = gold_row.get(key), pred_row.get(key)
        value_type = base_type(types.get(key, "unknown"))
        if is_identifier_field(key):
            score = identifier_similarity(gold, pred, key)
        elif value_type in {"integer", "float"}:
            left, right = parse_number(gold), parse_number(pred)
            score = float(left is not None and left == right)
        elif value_type == "boolean":
            left, right = parse_bool(gold), parse_bool(pred)
            score = float(left is not None and left == right)
        elif value_type == "date":
            left, right = parse_date(gold), parse_date(pred)
            score = (
                float(left == right)
                if left is not None and right is not None
                else float(normalize_text(gold) == normalize_text(pred))
            )
        else:
            score = float(normalize_text(gold) == normalize_text(pred))
        if score < 1.0:
            return 0.0
    return 1.0 if keys else 0.0


def row_similarity(
    gold_row: dict[str, Any],
    pred_row: dict[str, Any],
    columns: Sequence[str],
    keys: Sequence[str],
    types: dict[str, str],
) -> float:
    scored_columns = [
        column for column in columns if column != "rank" and column not in keys
    ]
    if not scored_columns:
        return 1.0
    return sum(
        value_similarity(
            gold_row.get(column),
            pred_row.get(column),
            types.get(column, "unknown"),
            field_name=column,
        )
        for column in scored_columns
    ) / len(scored_columns)


def align_rows(
    gold_rows: Sequence[dict[str, Any]],
    pred_rows: Sequence[dict[str, Any]],
    keys: Sequence[str],
    columns: Sequence[str],
    types: dict[str, str],
) -> list[tuple[int, int, float, float]]:
    weights = [[0.0 for _ in pred_rows] for _ in gold_rows]
    for gold_idx, gold_row in enumerate(gold_rows):
        for pred_idx, pred_row in enumerate(pred_rows):
            if key_similarity(gold_row, pred_row, keys, types) < 1.0:
                continue
            value_score = row_similarity(
                gold_row, pred_row, columns, keys, types
            )
            weights[gold_idx][pred_idx] = 1.0 + value_score
    matches = []
    for gold_idx, pred_idx, _ in maximum_weight_pairs(weights):
        value_score = row_similarity(
            gold_rows[gold_idx], pred_rows[pred_idx], columns, keys, types
        )
        matches.append((gold_idx, pred_idx, value_score, 1.0))
    return matches


def score_table(gold: Any, pred: Any, schema: dict[str, Any]) -> dict[str, Any]:
    columns, raw_types = schema_columns(schema)
    gold_rows = normalize_rows(gold, columns)
    pred_rows = normalize_rows(pred, columns)
    keys = declared_row_keys(schema, columns) or infer_row_keys(
        columns, raw_types, gold_rows
    )
    if not gold_rows and not pred_rows:
        return {
            "score": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "structure_score": 1.0,
            "value_score": 1.0,
            "detail": {
                "n_gold": 0,
                "n_pred": 0,
                "n_aligned": 0,
                "row_keys": keys,
            },
            "_matches": [],
            "_columns": columns,
        }
    if not gold_rows or not pred_rows:
        return {
            "score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "structure_score": 0.0,
            "value_score": 0.0,
            "detail": {
                "n_gold": len(gold_rows),
                "n_pred": len(pred_rows),
                "n_aligned": 0,
                "row_keys": keys,
            },
            "_matches": [],
            "_columns": columns,
        }

    types = dict(zip(columns, raw_types))
    matches = align_rows(gold_rows, pred_rows, keys, columns, types)
    structure_mass = float(len(matches))
    value_mass = sum(similarity for _, _, similarity, _ in matches)
    structure_precision = structure_mass / len(pred_rows)
    structure_recall = structure_mass / len(gold_rows)
    structure_score = harmonic(structure_precision, structure_recall)
    value_precision = value_mass / len(pred_rows)
    value_recall = value_mass / len(gold_rows)
    value_score = harmonic(value_precision, value_recall)
    return {
        "score": value_score,
        "precision": value_precision,
        "recall": value_recall,
        "structure_score": structure_score,
        "value_score": value_score,
        "detail": {
            "n_gold": len(gold_rows),
            "n_pred": len(pred_rows),
            "n_aligned": len(matches),
            "row_keys": keys,
            "structure_precision": structure_precision,
            "structure_recall": structure_recall,
            "structure_f1": structure_score,
            "value_f1": value_score,
        },
        "_matches": matches,
        "_columns": columns,
    }


def dictionary_similarity(
    gold: Any,
    pred: Any,
    schema: dict[str, Any],
    *,
    exact: bool = False,
) -> dict[str, Any]:
    if not isinstance(gold, dict) or not isinstance(pred, dict):
        return {
            "score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "structure_score": 0.0,
            "value_score": 0.0,
            "detail": {"reason": "incompatible dictionary structure"},
        }
    gold_index = {normalize_text(key): (key, value) for key, value in gold.items()}
    pred_index = {normalize_text(key): (key, value) for key, value in pred.items()}
    declared_keys = schema.get("keys") or list(gold)
    value_types = schema.get("value_types") or []
    type_map = {
        str(key): base_type(value_types[idx]) if idx < len(value_types) else "unknown"
        for idx, key in enumerate(declared_keys)
    }
    if not gold_index and not pred_index:
        return {
            "score": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "structure_score": 1.0,
            "value_score": 1.0,
            "detail": {"n_gold": 0, "n_pred": 0, "n_aligned": 0},
        }

    value_mass = 0.0
    matched = 0
    for normalized, (gold_key, gold_value) in gold_index.items():
        if normalized not in pred_index:
            continue
        _, pred_value = pred_index[normalized]
        item_schema = (
            (schema.get("items") or {}).get(gold_key)
            or schema.get("value_schema")
            or {}
        )
        if isinstance(gold_value, dict):
            similarity = dictionary_similarity(
                gold_value, pred_value, item_schema, exact=exact
            )["score"]
        else:
            similarity = value_similarity(
                gold_value,
                pred_value,
                type_map.get(gold_key, "unknown"),
                field_name=str(gold_key),
                exact=exact,
            )
        value_mass += float(similarity == 1.0) if exact else similarity
        matched += 1

    structure_precision = matched / len(pred_index) if pred_index else 0.0
    structure_recall = matched / len(gold_index) if gold_index else 0.0
    structure_score = harmonic(structure_precision, structure_recall)
    value_precision = value_mass / len(pred_index) if pred_index else 0.0
    value_recall = value_mass / len(gold_index) if gold_index else 0.0
    value_score = harmonic(value_precision, value_recall)
    return {
        "score": harmonic(structure_score, value_score) if exact else value_score,
        "precision": value_precision,
        "recall": value_recall,
        "structure_score": structure_score,
        "value_score": value_score,
        "detail": {
            "n_gold": len(gold_index),
            "n_pred": len(pred_index),
            "n_aligned": matched,
            "structure_f1": structure_score,
            "value_f1": value_score,
        },
    }


def discounted_cumulative_gain(gains: Sequence[float]) -> float:
    return sum(
        gain / math.log2(predicted_rank + 2)
        for predicted_rank, gain in enumerate(gains)
    )


def matched_row_order_ndcg(
    matches: Sequence[tuple[int, int, float, float]],
) -> dict[str, Any]:
    """Evaluate the relative order of aligned rows without rescoring values."""
    if not matches:
        return {
            "score": 0.0,
            "dcg": 0.0,
            "ideal_dcg": 0.0,
            "matched_rows": 0,
            "predicted_gold_order": [],
        }

    gold_order = sorted(gold_idx for gold_idx, _, _, _ in matches)
    matched_count = len(gold_order)
    relevance_by_gold = {
        gold_idx: (matched_count - rank) / matched_count
        for rank, gold_idx in enumerate(gold_order)
    }
    predicted_gold_order = [
        gold_idx
        for gold_idx, _, _, _ in sorted(matches, key=lambda item: item[1])
    ]
    predicted_gains = [
        relevance_by_gold[gold_idx] for gold_idx in predicted_gold_order
    ]
    ideal_gains = [relevance_by_gold[gold_idx] for gold_idx in gold_order]
    dcg = discounted_cumulative_gain(predicted_gains)
    ideal_dcg = discounted_cumulative_gain(ideal_gains)
    return {
        "score": dcg / ideal_dcg if ideal_dcg else 0.0,
        "dcg": dcg,
        "ideal_dcg": ideal_dcg,
        "matched_rows": matched_count,
        "predicted_gold_order": predicted_gold_order,
    }


def score_ordered_table(
    gold: Any,
    pred: Any,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Multiply soft table F1 by NDCG over matched-row relative order."""
    table = score_table(gold, pred, schema)
    matches = table.pop("_matches", [])
    table.pop("_columns", None)
    n_gold = int(table["detail"].get("n_gold", 0))
    n_pred = int(table["detail"].get("n_pred", 0))

    if n_gold == 0 and n_pred == 0:
        order = {
            "score": 1.0,
            "dcg": 0.0,
            "ideal_dcg": 0.0,
            "matched_rows": 0,
            "predicted_gold_order": [],
        }
    else:
        order = matched_row_order_ndcg(matches)

    soft_table_f1 = float(table.get("value_score", 0.0))
    order_ndcg = float(order["score"])
    table["score"] = soft_table_f1 * order_ndcg
    table["soft_table_f1"] = soft_table_f1
    table["order_score"] = order_ndcg
    table["ndcg_score"] = order_ndcg
    table["detail"].update(
        {
            "soft_table_f1": soft_table_f1,
            "order_ndcg": order_ndcg,
            "ordered_score_formula": "soft_table_f1 * order_ndcg",
            "order_gain": "linear_rank_within_matched_gold_rows",
            "order_scope": "relative_order_of_matched_rows_only",
            "order_dcg": order["dcg"],
            "order_ideal_dcg": order["ideal_dcg"],
            "order_matched_rows": order["matched_rows"],
            "predicted_gold_order": order["predicted_gold_order"],
        }
    )
    return table


def score_scalar(
    gold: Any,
    pred: Any,
    schema: dict[str, Any],
    metric: str,
) -> dict[str, Any]:
    value_type = base_type(schema.get("value_type") or infer_type(gold))
    if value_type in {"integer", "float"}:
        score = relative_accuracy(gold, pred)
    elif metric == "exact_match" and value_type == "string":
        score = short_text_similarity(gold, pred)
    elif metric == "exact_match":
        score = value_similarity(gold, pred, value_type, exact=True)
    else:
        score = value_similarity(gold, pred, value_type)
    return {
        "score": score,
        "detail": {"gold": gold, "pred": pred, "value_type": value_type},
    }


def score_label(gold: Any, pred: Any) -> dict[str, Any]:
    score = float(normalize_text(gold) == normalize_text(pred))
    return {"score": score, "detail": {"gold": gold, "pred": pred}}


def unwrap_report(value: Any) -> Any:
    if isinstance(value, dict) and isinstance(value.get("report"), str):
        return value["report"]
    return value


def split_report_claims(value: Any) -> list[str]:
    value = unwrap_report(value)
    if not isinstance(value, str):
        return []
    return [
        claim.strip()
        for claim in re.split(r"(?<=[.!?])\s+|\n+", value)
        if claim.strip()
    ]


def score_report(gold: Any, pred: Any, metric: str) -> dict[str, Any]:
    gold_claims = split_report_claims(gold)
    pred_claims = split_report_claims(pred)
    if not gold_claims and not pred_claims:
        score = 1.0
    elif not gold_claims or not pred_claims:
        score = 0.0
    elif metric == "claim_recall":
        score = sum(
            max(token_f1(gold_claim, pred_claim) for pred_claim in pred_claims)
            for gold_claim in gold_claims
        ) / len(gold_claims)
    else:
        score = token_f1(gold, pred)
    return {
        "score": score,
        "detail": {
            "gold_claims": len(gold_claims),
            "pred_claims": len(pred_claims),
        },
    }


def score_task(
    task: dict[str, Any],
    pred_answer: Any,
    profile: str = DEFAULT_PROFILE,
) -> dict[str, Any]:
    if profile not in SCORING_PROFILES:
        raise ValueError(f"unknown scoring profile: {profile}")
    task_id = str(task.get("task_id") or task.get("id") or "unknown")
    output_type = str((task.get("tags") or {}).get("output_type", "unknown"))
    declared_metric = str(
        (task.get("evaluation") or {}).get("primary_metric", "unknown")
    )
    gold_block = task.get("gold") or {}
    gold = gold_block.get("gold_answer")
    schema = gold_block.get("answer_schema") or {}

    if pred_answer is None:
        result: dict[str, Any] = {
            "score": 0.0,
            "detail": {"reason": "no prediction"},
        }
        if output_type in {"table", "dictionary", *ORDERED_OUTPUT_TYPES}:
            result.update(
                {
                    "precision": 0.0,
                    "recall": 0.0,
                    "structure_score": 0.0,
                    "value_score": 0.0,
                }
            )
        if output_type in ORDERED_OUTPUT_TYPES:
            result.update(
                {
                    "soft_table_f1": 0.0,
                    "order_score": 0.0,
                    "ndcg_score": 0.0,
                }
            )
            result["detail"]["declared_metric"] = declared_metric
    elif output_type == "scalar":
        result = score_scalar(gold, pred_answer, schema, declared_metric)
    elif output_type == "label":
        result = score_label(gold, pred_answer)
    elif output_type == "named_entity":
        result = {
            "score": named_entity_similarity(gold, pred_answer),
            "detail": {"gold": gold, "pred": pred_answer},
        }
    elif output_type == "table":
        result = score_table(gold, pred_answer, schema)
        result.pop("_matches", None)
        result.pop("_columns", None)
    elif output_type in ORDERED_OUTPUT_TYPES:
        result = score_ordered_table(gold, pred_answer, schema)
        result["detail"]["declared_metric"] = declared_metric
    elif output_type == "dictionary":
        normalized_prediction = normalize_dictionary_prediction(pred_answer, schema)
        result = dictionary_similarity(gold, normalized_prediction, schema)
    elif output_type == "report":
        result = score_report(gold, pred_answer, declared_metric)
    else:
        result = {
            "score": value_similarity(gold, pred_answer, infer_type(gold)),
            "detail": {"reason": f"fallback for {output_type}"},
        }

    metric = ORDERED_METRIC if output_type in ORDERED_OUTPUT_TYPES else declared_metric
    scored = {
        "task_id": task_id,
        "output_type": output_type,
        "metric": metric,
        "profile": profile,
        "score": round(float(result.get("score", 0.0)), 6),
        "precision": result.get("precision"),
        "recall": result.get("recall"),
        "structure_score": result.get("structure_score"),
        "value_score": result.get("value_score"),
        "order_score": result.get("order_score"),
        "detail": result.get("detail", {}),
    }
    if "soft_table_f1" in result:
        scored["soft_table_f1"] = result["soft_table_f1"]
    if "ndcg_score" in result:
        scored["ndcg_score"] = result["ndcg_score"]
    return scored


def unwrap_answer(answer: Any) -> Any:
    while isinstance(answer, dict) and set(answer) == {"answer"}:
        answer = answer["answer"]
    return answer


def extract_answer(path: str | Path) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return data
    answer = data.get(
        "final_answer_parsed",
        data.get("answer", data.get("prediction", data.get("result"))),
    )
    return unwrap_answer(answer)


# The scoring implementation above is unchanged from the benchmark's final
# scorer. The code below only loads the testbed layout and aggregates scores
# and resources into one report.
import argparse
import csv
import io


TESTBED_ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    ("aviation_safety", "Aviation Safety"),
    ("vehicle_safety", "Vehicle Safety"),
    ("legal_contracts", "Legal Contracts"),
    ("finance", "Finance"),
)
DIFFICULTIES = ("easy", "medium", "hard")
DIFFICULTY_LABELS = {
    "easy": "Easy",
    "medium": "Medium",
    "hard": "Hard",
}
SYSTEMS = (
    ("docetl", "DocETL"),
    ("lotus", "LOTUS"),
    ("palimpzest", "Palimpzest"),
    ("claude_code", "Claude"),
    ("claude_code_plus_lotus", "Claude+LOTUS"),
)
RESOURCE_FIELDS = ("cost_usd", "total_tokens", "elapsed_seconds")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json_array(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise ValueError(f"{path}: expected an array of objects")
    return payload


def index_unique(
    rows: Sequence[dict[str, Any]], path: Path
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            raise ValueError(f"{path}: every row must have a task_id")
        if task_id in indexed:
            raise ValueError(f"{path}: duplicate task_id {task_id}")
        indexed[task_id] = row
    return indexed


def infer_output_type(gold_row: dict[str, Any]) -> str:
    task_id = str(gold_row["task_id"])
    schema = gold_row.get("answer_schema") or {}
    declared = str(schema.get("type") or "").lower()
    if declared == "ordered_table":
        return "ordered_list"
    if declared in {
        "scalar",
        "label",
        "named_entity",
        "table",
        "ordered_list",
        "dictionary",
        "report",
    }:
        return declared
    raise ValueError(f"{task_id}: answer_schema has invalid type {declared!r}")


def primary_metric(output_type: str, schema: dict[str, Any]) -> str:
    if output_type == "scalar":
        return "rae" if base_type(schema.get("value_type")) == "float" else "exact_match"
    return {
        "label": "label_accuracy",
        "named_entity": "semantic_equivalence",
        "table": "table_f1",
        "ordered_list": "ordered_f1",
        "dictionary": "f1",
        "report": "claim_recall",
    }.get(output_type, "unknown")


def numeric_resource(value: Any, field: str, path: Path) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{path}: {field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {field} must be numeric") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{path}: {field} must be finite and non-negative")
    return number


def build_task(gold_row: dict[str, Any]) -> dict[str, Any]:
    schema = gold_row.get("answer_schema") or {}
    output_type = infer_output_type(gold_row)
    return {
        "task_id": gold_row["task_id"],
        "tags": {"output_type": output_type},
        "evaluation": {
            "primary_metric": primary_metric(output_type, schema),
        },
        "gold": {
            "gold_answer": gold_row.get("gold_answer"),
            "answer_schema": schema,
        },
    }


def evaluate(testbed_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, _ in DATASETS:
        query_path = testbed_root / "queries" / dataset / "queries.json"
        gold_path = testbed_root / "gold" / dataset / "queries.json"
        query_rows = load_json_array(query_path)
        gold_rows = load_json_array(gold_path)
        gold_by_id = index_unique(gold_rows, gold_path)
        query_by_id = index_unique(query_rows, query_path)
        if query_by_id.keys() != gold_by_id.keys():
            missing_gold = sorted(query_by_id.keys() - gold_by_id.keys())
            missing_query = sorted(gold_by_id.keys() - query_by_id.keys())
            raise ValueError(
                f"{dataset}: missing_gold={missing_gold}, missing_query={missing_query}"
            )

        for query_row in query_rows:
            task_id = str(query_row["task_id"])
            difficulty = str(query_row.get("operator_complexity") or "").lower()
            if difficulty not in DIFFICULTIES:
                raise ValueError(f"{task_id}: invalid difficulty {difficulty!r}")
            task = build_task(gold_by_id[task_id])

            for system, _ in SYSTEMS:
                result_path = (
                    testbed_root
                    / "results"
                    / dataset
                    / difficulty
                    / system
                    / f"{task_id}.json"
                )
                result = load_json(result_path)
                if not isinstance(result, dict):
                    raise ValueError(f"{result_path}: expected a JSON object")
                if str(result.get("task_id") or "") != task_id:
                    raise ValueError(f"{result_path}: task_id does not match its path")
                scored = score_task(task, unwrap_answer(result.get("answer")))
                rows.append(
                    {
                        "dataset": dataset,
                        "task_id": task_id,
                        "difficulty": difficulty,
                        "system": system,
                        "output_type": scored["output_type"],
                        "metric": scored["metric"],
                        "score": scored["score"],
                        "cost_usd": numeric_resource(
                            result.get("cost_usd"), "cost_usd", result_path
                        ),
                        "total_tokens": numeric_resource(
                            result.get("total_tokens"), "total_tokens", result_path
                        ),
                        "elapsed_seconds": numeric_resource(
                            result.get("elapsed_seconds"),
                            "elapsed_seconds",
                            result_path,
                        ),
                    }
                )
    return rows


def average(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for dataset, _ in DATASETS:
        for system, _ in SYSTEMS:
            selected = [
                row
                for row in rows
                if row["dataset"] == dataset and row["system"] == system
            ]
            summaries.append(
                {
                    "dataset": dataset,
                    "system": system,
                    "queries": len(selected),
                    "avg_score": average([row["score"] for row in selected]),
                    **{
                        f"{difficulty}_score": average(
                            [
                                row["score"]
                                for row in selected
                                if row["difficulty"] == difficulty
                            ]
                        )
                        for difficulty in DIFFICULTIES
                    },
                    "avg_cost_usd": average(
                        [row["cost_usd"] for row in selected]
                    ),
                    "avg_tokens": average(
                        [row["total_tokens"] for row in selected]
                    ),
                    "avg_latency_seconds": average(
                        [row["elapsed_seconds"] for row in selected]
                    ),
                }
            )
    return summaries


def render_markdown(
    rows: Sequence[dict[str, Any]], summaries: Sequence[dict[str, Any]]
) -> str:
    unique_tasks = {(row["dataset"], row["task_id"]) for row in rows}
    counts = {
        difficulty: len(
            {
                (row["dataset"], row["task_id"])
                for row in rows
                if row["difficulty"] == difficulty
            }
        )
        for difficulty in DIFFICULTIES
    }
    output = [
        f"# Testbed {len(unique_tasks)} Query Evaluation Report",
        "",
        "Difficulty distribution: "
        + ", ".join(
            f"{DIFFICULTY_LABELS[difficulty]} {counts[difficulty]}"
            for difficulty in DIFFICULTIES
        )
        + ".",
    ]

    system_labels = dict(SYSTEMS)
    for dataset, dataset_label in DATASETS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        task_ids = {row["task_id"] for row in dataset_rows}
        dataset_counts = {
            difficulty: len(
                {
                    row["task_id"]
                    for row in dataset_rows
                    if row["difficulty"] == difficulty
                }
            )
            for difficulty in DIFFICULTIES
        }
        output.extend(
            [
                "",
                f"## {dataset_label} ({len(task_ids)}Q)",
                "",
                "Difficulty distribution: "
                + ", ".join(
                    f"{DIFFICULTY_LABELS[difficulty]} {dataset_counts[difficulty]}"
                    for difficulty in DIFFICULTIES
                )
                + ".",
                "",
                "| System | Avg Score | Easy | Medium | Hard | Avg. Cost ($/Q) | Avg. token | Avg. Latency (s/Q) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for summary in summaries:
            if summary["dataset"] != dataset:
                continue
            output.append(
                "| {} | {:.2%} | {:.2%} | {:.2%} | {:.2%} | {:.4f} | {:,.0f} | {:,.2f} |".format(
                    system_labels[summary["system"]],
                    summary["avg_score"],
                    summary["easy_score"],
                    summary["medium_score"],
                    summary["hard_score"],
                    summary["avg_cost_usd"],
                    summary["avg_tokens"],
                    summary["avg_latency_seconds"],
                )
            )
    return "\n".join(output) + "\n"


def render_csv(summaries: Sequence[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    fieldnames = [
        "dataset",
        "system",
        "queries",
        "avg_score",
        "easy_score",
        "medium_score",
        "hard_score",
        "avg_cost_usd",
        "avg_tokens",
        "avg_latency_seconds",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(summaries)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--testbed-root",
        type=Path,
        default=TESTBED_ROOT,
        help="testbed directory containing queries, gold, and results",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json", "csv"),
        default="markdown",
    )
    args = parser.parse_args()

    rows = evaluate(args.testbed_root.resolve())
    summaries = summarize(rows)
    if args.format == "json":
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
    elif args.format == "csv":
        print(render_csv(summaries), end="")
    else:
        print(render_markdown(rows, summaries), end="")


if __name__ == "__main__":
    main()
