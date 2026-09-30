"""The benchmark driver: ``datasets x systems``, scored by ``semantica.evals``.

Two scopes, defined by the dataset (see :mod:`semantica.benchmarks.types`):

``per_case``
    Each case ships its own evidence. The system is reset and fed only that
    case's passages before it answers — SQuAD-style reading comprehension.

``corpus``
    The union of every case's passages is ingested once and every question then
    queries the same long-lived memory — the setting a *memory* benchmark
    (LoCoMo) actually cares about.

Scoring is deliberately not reimplemented here. Predictions are handed to
:func:`semantica.evals.evaluate`, so the harness and the repo's evaluator
library can never drift apart.
"""

import statistics
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from ..evals import evaluate
from .datasets import load_dataset
from .systems import SystemUnavailable, get_system
from .types import (
    CORPUS,
    PER_CASE,
    BenchmarkReport,
    Dataset,
    Prediction,
    SystemResult,
)

EXACT_MATCH = "normalized_exact_match"


def _corpus_texts(dataset: Dataset) -> List[str]:
    """Order-preserving union of every case's passages."""
    return list(dict.fromkeys(str(p) for case in dataset.cases for p in case.context))


def run_system(
    system_name: str,
    dataset: Dataset,
    *,
    primary_metric: str = "token_f1",
    system_options: Optional[Dict[str, Any]] = None,
    progress=None,
) -> SystemResult:
    """Run one system over one dataset and score the predictions.

    A system that raises while answering is recorded as a prediction with an
    ``error`` rather than aborting the run: one flaky question should not throw
    away the other 199, and the error count is reported next to the score.
    """
    system = get_system(system_name, **(system_options or {}))
    predictions: List[Prediction] = []

    started = time.perf_counter()
    if dataset.scope == CORPUS:
        system.reset()
        system.ingest(_corpus_texts(dataset), case_id=dataset.name)

    for index, case in enumerate(dataset.cases):
        if dataset.scope == PER_CASE:
            system.reset()
            if case.context:
                system.ingest(case.context, case_id=case.case_id)

        case_started = time.perf_counter()
        answer, error = "", None
        try:
            answer = system.answer(case.question, case_id=case.case_id)
        except Exception as exc:  # noqa: BLE001 - one bad case must not kill the run
            error = f"{type(exc).__name__}: {exc}"
        predictions.append(
            Prediction(
                case_id=case.case_id,
                answer=answer if error is None else "",
                latency_s=time.perf_counter() - case_started,
                error=error,
            )
        )
        if progress is not None:
            progress(system_name, dataset.name, index + 1, len(dataset.cases))
    wall_s = time.perf_counter() - started

    by_id = {case.case_id: case for case in dataset.cases}
    eval_cases = [
        {
            "id": prediction.case_id,
            "expected": by_id[prediction.case_id].answers,
            "actual": prediction.answer,
        }
        for prediction in predictions
    ]
    evaluators = list(dict.fromkeys([primary_metric, EXACT_MATCH]))
    summary = evaluate(eval_cases, evaluators)

    scores = [
        result.metrics[primary_metric].score
        for result in summary.cases
        if primary_metric in result.metrics
    ]
    exact = [
        result.metrics[EXACT_MATCH].score
        for result in summary.cases
        if EXACT_MATCH in result.metrics
    ]

    return SystemResult(
        system=system_name,
        dataset=dataset.name,
        n=len(predictions),
        mean_score=statistics.fmean(scores) if scores else 0.0,
        exact_match_rate=statistics.fmean(exact) if exact else 0.0,
        errors=sum(1 for prediction in predictions if prediction.error),
        wall_s=wall_s,
        primary_metric=primary_metric,
        summary=summary,
        predictions=predictions,
    )


def run_benchmark(
    datasets: Sequence[Dataset],
    systems: Sequence[str],
    *,
    primary_metric: str = "token_f1",
    system_options: Optional[Dict[str, Any]] = None,
    on_error: str = "skip",
    progress=None,
) -> BenchmarkReport:
    """Run every (system, dataset) pair and return one report.

    ``on_error`` controls what happens when a system cannot start (missing SDK
    or credential). ``"skip"`` records it in :attr:`BenchmarkReport.skipped` and
    continues, which is what you want for a laptop run where only one of the
    three hosted backends is configured; ``"raise"`` propagates the error.
    """
    if on_error not in {"skip", "raise"}:
        raise ValueError(f"on_error must be 'skip' or 'raise', got {on_error!r}")

    report = BenchmarkReport(
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        primary_metric=primary_metric,
        datasets=[dataset.name for dataset in datasets],
        systems=list(systems),
    )

    for system_name in systems:
        for dataset in datasets:
            try:
                result = run_system(
                    system_name,
                    dataset,
                    primary_metric=primary_metric,
                    system_options=(system_options or {}).get(system_name),
                    progress=progress,
                )
            except SystemUnavailable as exc:
                if on_error == "raise":
                    raise
                report.skipped.append(
                    {"system": system_name, "dataset": dataset.name, "reason": str(exc)}
                )
                continue
            report.results.append(result)
            report.scores[f"{system_name}/{dataset.name}"] = result.mean_score

    return report


def run_from_spec(
    dataset_specs: Sequence[Dict[str, Any]],
    systems: Sequence[str],
    **kwargs: Any,
) -> BenchmarkReport:
    """Convenience wrapper: load datasets from ``{"name": ..., **options}`` specs."""
    datasets = [
        load_dataset(spec["name"], **{k: v for k, v in spec.items() if k != "name"})
        for spec in dataset_specs
    ]
    return run_benchmark(datasets, systems, **kwargs)
