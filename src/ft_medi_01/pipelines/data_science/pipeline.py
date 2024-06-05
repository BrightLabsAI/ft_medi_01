from kedro.pipeline import Pipeline, node, pipeline

from .download import download_tokenizer
from .train import prepare_model, tokenize_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=download_tokenizer,
                inputs="params:model",
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
                func=prepare_model,
                inputs=["params:model", "params:model_optimization"],
                outputs="training_model",
                name="prepare_training_model",
            ),
            node(
                func=train_model,
                inputs=[
                    "tokenized_training_data",
                    "training_model",
                    "tokenizer",
                    "params:dataset",
                ],
                outputs=None,
                name="train_model",
            ),
        ]
    )
