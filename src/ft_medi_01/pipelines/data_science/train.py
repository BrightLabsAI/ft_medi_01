import logging
import sys

import torch
import transformers
from datasets import Dataset
from pandas import DataFrame

logger = logging.getLogger(__name__)


def tokenize_data(df: DataFrame, tokenizer: transformers.AutoTokenizer) -> Dataset:
    """
    Tokenizes the given DataFrame using the provided tokenizer.

    Args:
        df (DataFrame): The DataFrame containing the text data to be tokenized.
        tokenizer (transformers.AutoTokenizer): The tokenizer to be used for tokenization.

    Returns:
        Dataset: The tokenized data as a Dataset object.

    Raises:
        None

    Examples:
        >>> df = pd.DataFrame({'text': ['Hello world', 'How are you']})
        >>> tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
        >>> tokenize_data(df, tokenizer)
        Tokenizing dataset
        <Dataset>
    """
    logger.info("Tokenizing dataset")
    tokenizer.pad_token = tokenizer.eos_token
    text_list = df["text"].tolist()
    tokenized_data = tokenizer(
        text_list, truncation=True, padding="max_length", return_tensors="pt"
    )

    return tokenized_data


def train_model(
    tokenized_data: Dataset, model: transformers.AutoModelForCausalLM
) -> None:
    """
    Trains a causal language model using the given tokenized data and model.

    Args:
        tokenized_data (Dataset): The tokenized dataset to train the model on.
        model (transformers.AutoModelForCausalLM): The causal language model to train.

    Returns:
        None

    Raises:
        None

    Examples:
        >>> tokenized_data = ...
        >>> model = transformers.AutoModelForCausalLM.from_pretrained('bert-base-uncased')
        >>> train_model(tokenized_data, model)
        Training model
        <AutoModelForCausalLM>
    """

    logger.info("Setting training arguments")
    training_args = transformers.TrainingArguments(
        output_dir="model/output",
        eval_strategy="steps",
        load_best_model_at_end=True,
        eval_steps=5,
        learning_rate=2e-5,
        weight_decay=0.01,
        no_cuda=False if not torch.cuda.is_available() else True,
        logging_dir="model/logs",
    )
    #
    trainer = transformers.Trainer(model, training_args, train_dataset=[tokenized_data])
    result = trainer.train()

    logger.info("Training completed")

    return None
