from kedro.pipeline import Pipeline, pipeline, node
from .download import download_model, download_tokenizer
from .train import tokenize_data, train_model

def create_pipeline(**kwargs) -> Pipeline:
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
