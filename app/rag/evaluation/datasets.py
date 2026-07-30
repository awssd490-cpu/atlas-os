"""Evaluation datasets — immutable collections of query-relevance pairs.

Provides ``EvaluationSample``, ``EvaluationDataset``, and
``DatasetLoader`` for loading, saving, and manipulating ground-truth
datasets used to evaluate retrieval quality.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from app.rag.evaluation.errors import EvaluationError


# ---------------------------------------------------------------------------
# EvaluationSample
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationSample:
    """A single query with its ground-truth relevant document IDs.

    Attributes:
        query: The search query string.
        relevant_ids: A frozenset of document or chunk IDs that are
            considered relevant for this query.
        metadata: Optional metadata attached to the sample (e.g.
            difficulty, category, source).
    """

    query: str = ""
    relevant_ids: frozenset[str] = field(default_factory=frozenset)
    metadata: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# EvaluationDataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationDataset:
    """An immutable collection of ``EvaluationSample`` objects.

    Attributes:
        name: Human-readable dataset name.
        samples: Tuple of samples in the dataset.
        metadata: Optional dataset-level metadata (e.g. description,
            version, source).
    """

    name: str = ""
    samples: tuple[EvaluationSample, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    @property
    def is_empty(self) -> bool:
        """Return ``True`` if the dataset contains no samples."""
        return len(self.samples) == 0

    # ------------------------------------------------------------------
    # Query access
    # ------------------------------------------------------------------

    def queries(self) -> list[str]:
        """Return all query strings in order."""
        return [s.query for s in self.samples]

    def relevant_sets(self) -> list[frozenset[str]]:
        """Return all relevant ID sets in order."""
        return [s.relevant_ids for s in self.samples]

    def sample(self, index: int) -> EvaluationSample:
        """Return the sample at *index*.

        Args:
            index: Zero-based index.

        Returns:
            The ``EvaluationSample`` at that index.

        Raises:
            IndexError: If *index* is out of range.
        """
        return self.samples[index]


# ---------------------------------------------------------------------------
# DatasetLoader
# ---------------------------------------------------------------------------


class DatasetLoader:
    """Loads and saves evaluation datasets from/to Python dicts and JSON.

    Usage::

        loader = DatasetLoader()

        # From Python dict
        dataset = loader.from_dict({
            "name": "my-dataset",
            "samples": [
                {"query": "capital of France", "relevant_ids": ["doc1"]},
            ],
        })

        # From JSON file
        dataset = loader.from_json("path/to/dataset.json")

        # To JSON file
        loader.to_json(dataset, "path/to/dataset.json")
    """

    # ------------------------------------------------------------------
    # Deserialisation
    # ------------------------------------------------------------------

    @staticmethod
    def from_dict(data: dict[str, Any]) -> EvaluationDataset:
        """Build an ``EvaluationDataset`` from a Python dict.

        Expected structure::

            {
                "name": "...",
                "metadata": { ... },
                "samples": [
                    {
                        "query": "...",
                        "relevant_ids": ["doc1", "doc2"],
                        "metadata": { ... }
                    },
                    ...
                ]
            }

        Args:
            data: The dict to deserialise.

        Returns:
            A new ``EvaluationDataset``.

        Raises:
            InvalidEvaluationConfiguration: If the dict is missing
                required fields or has an invalid structure.
        """
        from app.rag.evaluation.errors import InvalidEvaluationConfiguration

        if not isinstance(data, dict):
            raise InvalidEvaluationConfiguration(
                "Dataset root must be a JSON object",
                details={"received_type": type(data).__name__},
            )

        name = data.get("name", "")
        if not isinstance(name, str):
            raise InvalidEvaluationConfiguration(
                "Dataset 'name' must be a string",
                details={"received_type": type(name).__name__},
            )

        meta = data.get("metadata", {})
        if not isinstance(meta, dict):
            raise InvalidEvaluationConfiguration(
                "Dataset 'metadata' must be an object",
                details={"received_type": type(meta).__name__},
            )

        raw_samples = data.get("samples")
        if raw_samples is None:
            raise InvalidEvaluationConfiguration(
                "Missing required field: samples",
            )
        if not isinstance(raw_samples, list):
            raise InvalidEvaluationConfiguration(
                "Dataset 'samples' must be a list",
                details={"received_type": type(raw_samples).__name__},
            )

        samples: list[EvaluationSample] = []
        for i, item in enumerate(raw_samples):
            if not isinstance(item, dict):
                raise InvalidEvaluationConfiguration(
                    f"Sample at index {i} must be an object",
                    details={"index": i, "received_type": type(item).__name__},
                )

            query = item.get("query", "")
            if not isinstance(query, str):
                raise InvalidEvaluationConfiguration(
                    f"Sample at index {i} 'query' must be a string",
                    details={"index": i, "received_type": type(query).__name__},
                )

            raw_ids = item.get("relevant_ids", [])
            if not isinstance(raw_ids, list):
                raise InvalidEvaluationConfiguration(
                    f"Sample at index {i} 'relevant_ids' must be a list",
                    details={"index": i, "received_type": type(raw_ids).__name__},
                )

            relevant_ids: set[str] = set()
            for j, rid in enumerate(raw_ids):
                if not isinstance(rid, str):
                    raise InvalidEvaluationConfiguration(
                        f"Sample at index {i} 'relevant_ids[{j}]' must be a string",
                        details={"index": i, "item_index": j, "received_type": type(rid).__name__},
                    )
                relevant_ids.add(rid)

            sample_meta = item.get("metadata", {})
            if not isinstance(sample_meta, dict):
                raise InvalidEvaluationConfiguration(
                    f"Sample at index {i} 'metadata' must be an object",
                    details={"index": i, "received_type": type(sample_meta).__name__},
                )

            samples.append(EvaluationSample(
                query=query,
                relevant_ids=frozenset(relevant_ids),
                metadata=sample_meta,
            ))

        return EvaluationDataset(
            name=name,
            samples=tuple(samples),
            metadata=meta,
        )

    @staticmethod
    def from_json(path: str) -> EvaluationDataset:
        """Load a dataset from a JSON file.

        Args:
            path: Path to a JSON file.

        Returns:
            A new ``EvaluationDataset``.

        Raises:
            EvaluationError: If the file cannot be read or parsed.
            InvalidEvaluationConfiguration: If the JSON structure is
                invalid.
        """
        target = Path(path)
        if not target.exists():
            raise EvaluationError(
                f"Dataset file does not exist: {path}",
                details={"path": path},
            )

        try:
            raw = target.read_bytes()
            data = json.loads(raw)
        except (json.JSONDecodeError, OSError) as exc:
            raise EvaluationError(
                f"Failed to parse dataset file: {exc}",
                details={"path": path},
            ) from exc

        if not isinstance(data, dict):
            from app.rag.evaluation.errors import InvalidEvaluationConfiguration
            raise InvalidEvaluationConfiguration(
                "Dataset file must contain a JSON object",
                details={"path": path, "received_type": type(data).__name__},
            )

        return DatasetLoader.from_dict(data)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def to_dict(dataset: EvaluationDataset) -> dict[str, Any]:
        """Serialise an ``EvaluationDataset`` to a plain Python dict.

        The output is deterministic and ready for ``json.dumps``.

        Args:
            dataset: The dataset to serialise.

        Returns:
            A dict with sorted sample entries.
        """
        samples_list: list[dict[str, Any]] = []
        for sample in dataset.samples:
            samples_list.append({
                "query": sample.query,
                "relevant_ids": sorted(sample.relevant_ids),
                "metadata": dict(sample.metadata),
            })

        return {
            "name": dataset.name,
            "metadata": dict(dataset.metadata),
            "samples": samples_list,
        }

    @staticmethod
    def to_json(dataset: EvaluationDataset, path: str) -> None:
        """Serialise an ``EvaluationDataset`` to a JSON file.

        The output is UTF-8 with pretty-printing and sorted keys for
        deterministic output.

        Args:
            dataset: The dataset to serialise.
            path: Target file path.

        Raises:
            EvaluationError: On write failures.
        """
        data = DatasetLoader.to_dict(dataset)

        try:
            json_bytes = json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise EvaluationError(
                f"Failed to serialise dataset: {exc}",
                details={"path": path},
            ) from exc

        try:
            Path(path).write_bytes(json_bytes)
        except OSError as exc:
            raise EvaluationError(
                f"Failed to write dataset file: {exc}",
                details={"path": path},
            ) from exc
