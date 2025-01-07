from kedro.io import DataCatalog
from kedro.pipeline import node, pipeline
from kedro.runner import SequentialRunner

from multi_level_stacking.nodes.data_processing.data_processing_raw import preprocess_raw_data
from multi_level_stacking.constants import DATASETS

nodes = [
    node(
        func=preprocess_raw_data,
        inputs=[f'raw_{dataset}', f'params:preprocessing_params.{dataset}'],
        outputs=f"int_{dataset}",
        name=f"preprocess_{dataset}"
    )
    for dataset in DATASETS
]

preprocessing_data_pipeline = pipeline(nodes)