import logging
from typing import Dict

import torch
import transformers
from datasets import Dataset
from pandas import DataFrame
from peft import LoraConfig, TaskType, get_peft_model, peft_model
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling

logger = logging.getLogger(__name__)


def tokenize_data(df: DataFrame, tokenizer: transformers.AutoTokenizer) -> Dataset:
    """
    Tokenizes the data using the given tokenizer.

    Args:
        df (DataFrame): The data to tokenize.
        tokenizer (transformers.AutoTokenizer): The tokenizer to use.

    Returns:
        Dataset: The tokenized data.

    Raises:
        None

    Examples:
        >>> tokenize_data(df, tokenizer)
        Tokenizing dataset
        <None>
    """
    logger.info("Tokenizing dataset")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        return tokenizer(
            example["text"], truncation=True, max_length=tokenizer.model_max_length
        )

    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    return tokenized_dataset


def prepare_model(
    params_model: Dict,
    params_model_optimization: Dict,
) -> peft_model.PeftModelForCausalLM:
    """
    Prepares the model for training by loading the model, setting the peft configuration, and printing the number of trainable parameters.

    Args:
        params_model (Dict): The parameters for the model.
        params_model_optimization (Dict): The parameters for the model optimization.

    Returns:
        peft_model.PeftModelForCausalLM: The prepared model.

    Raises:
        None

    Examples:
        >>> prepare_model("model_name", "model_optimization")
        Downloading model model_name
        Setting peft configuration
        <peft.model.peft_model.PeftModelForCausalLM>
    """
    logger.info("Downloading model %s", params_model["model_name"])
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=params_model_optimization["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=params_model_optimization["quantization"][
            "bnb_4bit_quant_type"
        ],
        bnb_4bit_use_double_quant=params_model_optimization["quantization"][
            "bnb_4bit_use_double_quant"
        ],
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        params_model["model_name"],
        trust_remote_code=True,
        quantization_config=nf4_config,
    )

    # Set peft configuration for model
    # https://huggingface.co/docs/peft/en/index
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=params_model_optimization["lora"]["r"],
        lora_alpha=params_model_optimization["lora"]["lora_alpha"],
        lora_dropout=params_model_optimization["lora"]["lora_dropout"],
    )
    model = get_peft_model(model, peft_config)
    # Print number of trainable parameters when using peft
    model.print_trainable_parameters()
    return model


def train_model(
    tokenized_data: Dataset,
    model: peft_model.PeftModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    params_model_training: Dict,
) -> None:
    """
    Trains the model on the tokenized data.

    Args:
        tokenized_data (Dataset): The tokenized data to train on.
        model (peft_model.PeftModelForCausalLM): The model to train.
        tokenizer (transformers.AutoTokenizer): The tokenizer used to tokenize the data.
        params_model_training (Dict): The parameters for the model training.

    Returns:
        None

    Raises:
        None

    Examples:
        >>> train_model(tokenized_data, model, tokenizer, "model_optimization")
        Setting training arguments
        Training model
        <None>
    """
    # Free up some memory
    torch.cuda.empty_cache()

    # Create data collator for training.
    # The data collator is used to pad the inputs and targets during the training process.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=False, return_tensors="pt"
    )

    # Set training arguments
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Setting training arguments")

    training_args = transformers.TrainingArguments(
        output_dir="model/output",
        eval_strategy="steps",
        eval_steps=params_model_training["eval_steps"],
        learning_rate=params_model_training["learning_rate"],
        weight_decay=params_model_training["weight_decay"],
        logging_dir="model/logs",
        per_device_train_batch_size=params_model_training["batch_size"],
        gradient_accumulation_steps=params_model_training[
            "gradient_accumulation_steps"
        ],
        fp16=params_model_training["fp16"],
        # report_to="wandb",
        # run_name="FT-Medi-01",
    )

    # Train the model
    trainer = transformers.Trainer(
        model,
        training_args,
        train_dataset=tokenized_data,
        data_collator=data_collator,
        strategy="ddp",
    )
    logger.info("Training model")
    # trainer.train()
    logger.info("Training completed")

    return None
