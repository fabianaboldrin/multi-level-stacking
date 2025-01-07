from kedro.pipeline import node, pipeline

from multi_level_stacking.nodes.model_train.model_train import run_multi_level_stacking
from multi_level_stacking.constants import DATASETS, ART_DATASETS

ALL_DATASETS = DATASETS + ART_DATASETS


nodes_tree = [
    node(
        func=run_multi_level_stacking,
        inputs=[f'int_{dataset}',
                f'params:preprocessing_params.{dataset}.name',
                'params:modeling_params', 
                'params:validation_params',
                'params:modeling_params.tree_meta_algorithm'],
        outputs=f"model_{dataset}_tree",
        name=f"model_train_{dataset}_tree"
    )
    for dataset in ALL_DATASETS
]

nodes_nb = [
    node(
        func=run_multi_level_stacking,
        inputs=[f'int_{dataset}', 
                f'params:preprocessing_params.{dataset}.name',
                'params:modeling_params', 
                'params:validation_params',
                'params:modeling_params.naive_bayes_meta_algorithm'],
        outputs=f"model_{dataset}_naive_bayes",
        name=f"model_train_{dataset}_naive_bayes"
    )
    for dataset in ALL_DATASETS
]

nodes_svm = [
    node(
        func=run_multi_level_stacking,
        inputs=[f'int_{dataset}', 
                f'params:preprocessing_params.{dataset}.name',
                'params:modeling_params',
                'params:validation_params',
                'params:modeling_params.svm_meta_algorithm'],
        outputs=f"model_{dataset}_svm",
        name=f"model_train_{dataset}_svm"
    )
    for dataset in ALL_DATASETS
]

model_train_pipeline = pipeline(nodes_tree + nodes_nb + nodes_svm)   # Combine all nodes into a single pipeline