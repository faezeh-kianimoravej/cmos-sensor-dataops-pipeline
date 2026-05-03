"""Pipeline entrypoints package for the notebook-aligned 7-stage workflow."""


def run_ml_pipeline(*args, **kwargs):
    from pipelines.run_pipeline import run_ml_pipeline as _run_ml_pipeline

    return _run_ml_pipeline(*args, **kwargs)


def run_ingestion(*args, **kwargs):
    from pipelines.ingest_flow import run_ingestion as _run_ingestion

    return _run_ingestion(*args, **kwargs)


def run_feature_engineering(*args, **kwargs):
    from pipelines.feature_engineering_flow import (
        run_feature_engineering as _run_feature_engineering,
    )

    return _run_feature_engineering(*args, **kwargs)


def run_train_stage1_mixture_detection(*args, **kwargs):
    from pipelines.train_stage1_mixture_detection_flow import (
        run_train_stage1_mixture_detection as _run_train_stage1_mixture_detection,
    )

    return _run_train_stage1_mixture_detection(*args, **kwargs)


def run_prepare_stage2_subset(*args, **kwargs):
    from pipelines.prepare_stage2_single_gas_subset_flow import (
        run_prepare_stage2_subset as _run_prepare_stage2_subset,
    )

    return _run_prepare_stage2_subset(*args, **kwargs)


def run_train_stage2_single_gas_probability(*args, **kwargs):
    from pipelines.train_stage2_single_gas_probability_flow import (
        run_train_stage2_single_gas_probability as _run_train_stage2_single_gas_probability,
    )

    return _run_train_stage2_single_gas_probability(*args, **kwargs)


def run_evaluate_stage1(*args, **kwargs):
    from pipelines.evaluate_stage1_flow import (
        run_evaluate_stage1 as _run_evaluate_stage1,
    )

    return _run_evaluate_stage1(*args, **kwargs)


def run_evaluate_stage2(*args, **kwargs):
    from pipelines.evaluate_stage2_flow import (
        run_evaluate_stage2 as _run_evaluate_stage2,
    )

    return _run_evaluate_stage2(*args, **kwargs)


def run_evaluation(*args, **kwargs):
    from pipelines.evaluate_flow import run_evaluation as _run_evaluation

    return _run_evaluation(*args, **kwargs)


def run_batch_prediction(*args, **kwargs):
    from pipelines.batch_predict_flow import (
        run_batch_prediction as _run_batch_prediction,
    )

    return _run_batch_prediction(*args, **kwargs)


__all__ = [
    "run_ml_pipeline",
    "run_ingestion",
    "run_feature_engineering",
    "run_train_stage1_mixture_detection",
    "run_prepare_stage2_subset",
    "run_train_stage2_single_gas_probability",
    "run_evaluate_stage1",
    "run_evaluate_stage2",
    "run_evaluation",
    "run_batch_prediction",
]
