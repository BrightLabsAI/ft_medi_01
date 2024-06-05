import logging
import os
from datetime import datetime
from typing import Dict

import optuna
import torch
import transformers
from datasets import Dataset
from pandas import DataFrame
from peft import LoraConfig, TaskType, get_peft_model, peft_model
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, STAR_WARS

import wandb

logger = logging.getLogger(__name__)
os.environ["WANDB_PROJECT"] = "ft_medi_01"


def get_random_run_name() -> str:
    run_name = get_random_name(
        combo=[ADJECTIVES, STAR_WARS], separator="_", style="lowercase"
    ).replace(" ", "_")
    return run_name


def tokenize_data(df: DataFrame, tokenizer: transformers.AutoTokenizer) -> Dataset:
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
    logger.info("Downloading model %s", params_model["model_name"])
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=params_model_optimization["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=params_model_optimization["quantization"][
            "bnb_4bit_quant_type"
        ],
        bnb_4bit_use_double_quant=params_model_optimization["quantization"][
            "bnb_4bit_use_double_quant"
        ],
        bnb_4bit_compute_dtype=torch.float16,
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


def objective(
    trial: optuna.Trial,
    tokenized_data: Dataset,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.AutoTokenizer,
) -> float:
    params_model_training = {
        "no_of_epochs": trial.suggest_int("no_of_epochs", 1, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.001, 0.05),
        "per_device_train_batch_size": trial.suggest_int(
            "per_device_train_batch_size", 2, 4
        ),
        "fp16": trial.suggest_categorical("fp16", [True, False]),
        "gradient_accumulation_steps": trial.suggest_int(
            "gradient_accumulation_steps", 1, 4
        ),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
    }

    tokenizer.pad_token = tokenizer.eos_token

    # Create data collator for training.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=False, return_tensors="pt"
    )

    wandb.init(config=params_model_training, name=get_random_run_name())
    # Set training arguments
    training_args = transformers.TrainingArguments(
        output_dir="model/output/trial",
        logging_dir="model/logs",
        num_train_epochs=params_model_training["no_of_epochs"],
        learning_rate=params_model_training["learning_rate"],
        weight_decay=params_model_training["weight_decay"],
        per_device_train_batch_size=params_model_training[
            "per_device_train_batch_size"
        ],
        fp16=params_model_training["fp16"],
        gradient_accumulation_steps=params_model_training[
            "gradient_accumulation_steps"
        ],
        warmup_ratio=params_model_training["warmup_ratio"],
        report_to="wandb",
        logging_steps=50,
    )

    # Train the model
    trainer = transformers.Trainer(
        model,
        training_args,
        train_dataset=tokenized_data,
        data_collator=data_collator,
    )
    logger.info("Training model")
    train_result = trainer.train()
    logger.info("Training completed")

    train_loss = train_result.metrics["train_loss"]

    # Finish the W&B run
    wandb.run.finish()

    # Return the best training loss to minimize
    return train_loss


def train_model(
    tokenized_data: Dataset,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.AutoTokenizer,
    params_dataset: Dict,
) -> None:
    torch.cuda.empty_cache()

    # tokenized_parameter_data is 20% of the tokenized_data
    tokenized_parameter_data = tokenized_data.shuffle().select(
        range(
            0,
            int(
                params_dataset["hyperparameter_search_percentage"] * len(tokenized_data)
            ),
        )
    )
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=30, interval_steps=10
        ),
    )
    study.optimize(
        lambda trial: objective(trial, tokenized_parameter_data, model, tokenizer),
        n_trials=20,
    )

    logger.info(f"Best trial: {study.best_trial.params}")
    best_params = study.best_trial.params
    data_collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=False, return_tensors="pt"
    )

    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Setting training arguments")

    wandb.init(name=get_random_run_name())
    training_args = transformers.TrainingArguments(
        output_dir="model/output/final",
        num_train_epochs=best_params["no_of_epochs"],
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
        per_device_train_batch_size=best_params["per_device_train_batch_size"],
        fp16=best_params["fp16"],
        report_to="wandb",
        logging_steps=100,
    )

    trainer = transformers.Trainer(
        model,
        training_args,
        train_dataset=tokenized_data,
        data_collator=data_collator,
    )
    logger.info("Training model with best hyperparameters")
    trainer.train()
    logger.info("Training completed")

    wandb.run.finish()

    logger.info("Saving model and tokenizer")
    model.save_pretrained("model/output/model")
    tokenizer.save_pretrained("model/output/tokenizer")

    logger.info("Model and tokenizer saved")

    return None
