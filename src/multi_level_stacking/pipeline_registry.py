"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from multi_level_stacking.pipelines.data_processing import data_processing_raw_pipeline
from multi_level_stacking.pipelines.model_train import model_train_pipeline
from multi_level_stacking.pipelines.data_generation import data_generation  


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values())
    pipelines['preprocessing_pipeline'] = data_processing_raw_pipeline.preprocessing_data_pipeline
    pipelines['model_training_pipeline'] = model_train_pipeline.model_train_pipeline
    pipelines['art_data_generation_pipeline'] = data_generation.generating_art_data_pipeline
    
    return pipelines
