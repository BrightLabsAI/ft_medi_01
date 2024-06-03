from kedro.pipeline import Pipeline, node, pipeline

from .nodes import download_from_huggingface, pre_process


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=download_from_huggingface,
                inputs="params:dataset_name",
                outputs="raw_medical",
                name="download_data",
            ),
            node(
                func=pre_process,
                inputs=["params:data_percent", "raw_medical"],
                outputs="processed_medical",
                name="pre_process_data",
            ),
        ]
    )
