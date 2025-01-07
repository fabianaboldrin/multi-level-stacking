from kedro.pipeline import node, pipeline

from multi_level_stacking.constants import ART_DATASETS
from multi_level_stacking.nodes.artificial_data_generation.artificial_data_generation import generate_data_circles, generate_data_moons, generate_data_blobs, generate_data_classification, generate_gaussian_quantiles


nodes = [
    node(
        func=generate_data_circles,
        inputs=f'params:preprocessing_params.circles.args',
        outputs=f"int_circles",
        name=f"generate_circles_dataset"
    ),
    node(
        func=generate_data_moons,
        inputs=f'params:preprocessing_params.moons.args',
        outputs=f"int_moons",
        name=f"generate_moons_dataset"
    ),
    
    ]


nodes_blobs_features = [
    node(
            func=generate_data_blobs,
            inputs=f'params:preprocessing_params.blobs_{n_features}.args',
            outputs=f"int_blobs_{n_features}",
            name=f"generate_blobs_{n_features}_dataset"
        )
            for n_features in ['2', '3', '5', '7']
    
    ]

nodes_classification_features = [
    node(
            func=generate_data_classification,
            inputs=f'params:preprocessing_params.classification_{n_features}.args',
            outputs=f"int_classification_{n_features}",
            name=f"generate_classification_{n_features}_dataset"
        )
            for n_features in [ '5', '7']
    ]

nodes_gaussian_quantile_features = [
    node(
            func=generate_gaussian_quantiles,
            inputs=f'params:preprocessing_params.gaussian_quantiles_{n_features}.args',
            outputs=f"int_gaussian_quantiles_{n_features}",
            name=f"generate_gaussian_quantiles_{n_features}_dataset"
        )
            for n_features in ['2', '3', '5', '7']
    ]



generating_art_data_pipeline = pipeline(nodes + 
                                        nodes_blobs_features + 
                                        nodes_classification_features +
                                        nodes_gaussian_quantile_features)