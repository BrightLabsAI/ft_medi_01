from kedro.pipeline import Pipeline, node, pipeline

from .download import download_model, download_tokenizer
from .train import tokenize_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates a pipeline for data science tasks.

    Args:
        **kwargs: Additional keyword arguments.

    Returns:
        Pipeline: The created pipeline.

    The function creates a pipeline with the following nodes:
    - download_model: Downloads the model specified in the 'params:model_name' input.
    - download_tokenizer: Downloads the tokenizer for the model specified in the 'params:model_name' input.
    - tokenize_data: Tokenizes the 'processed_medical' dataset and the tokenizer.
    - train_model: Trains the model using the tokenized training data and the raw model.

    The pipeline is returned as a `Pipeline` object.
    """

    return pipeline(
        [
            node(
                func=download_model,
                inputs="params:model_name",
                outputs="training_model_raw",
                name="download_model",
            ),
            node(
                func=download_tokenizer,
                inputs="params:model_name",
                outputs="tokenizer",
                name="download_tokenizer",
            ),
            node(
                func=tokenize_data,
                inputs=["processed_medical", "tokenizer"],
                outputs="tokenized_training_data",
                name="tokenize_training_data",
            ),
            node(
                func=train_model,
                inputs=["tokenized_training_data", "training_model_raw"],
                outputs=None,
                name="train_model",
            ),
        ]
    )
