from kedro.pipeline import Pipeline, node, pipeline

from .nodes import download_from_huggingface, pre_process


def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates a pipeline for downloading and preprocessing medical data.

    Args:
        **kwargs: Additional keyword arguments.

    Returns:
        Pipeline: The created pipeline.

    The function creates a pipeline with the following nodes:
    - download_from_huggingface: Downloads medical data from Hugging Face's Datasets library.
    - pre_process: Preprocesses the downloaded data.

    The pipeline is returned as a `Pipeline` object.
    """
    return pipeline(
        [
            node(
                func=download_from_huggingface,
                inputs="params:dataset",
                outputs="raw_medical",
                name="download_data",
            ),
            node(
                func=pre_process,
                inputs=["params:dataset", "raw_medical"],
                outputs="processed_medical",
                name="pre_process_data",
            ),
        ]
    )
