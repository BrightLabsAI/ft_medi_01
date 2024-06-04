import logging
from typing import Dict

import torch
import transformers
from datasets import Dataset
from pandas import DataFrame
from peft import LoraConfig, TaskType, get_peft_model, peft_model
from transformers import BitsAndBytesConfig

logger = logging.getLogger(__name__)


def tokenize_data(df: DataFrame, tokenizer: transformers.AutoTokenizer) -> Dataset:
    logger.info("Tokenizing dataset")
    tokenizer.pad_token = tokenizer.eos_token
    text_list = df["text"].tolist()
    tokenized_data = tokenizer(
        text_list, truncation=True, padding="max_length", return_tensors="pt"
    )
    return tokenized_data


def prepare_model(params_model: Dict) -> peft_model.PeftModelForCausalLM:
    logger.info("Downloading model %s", params_model["model_name"])
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        params_model["model_name"], quantization_config=nf4_config
    )
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def train_model(
    tokenized_data: Dataset, model: peft_model.PeftModelForCausalLM
) -> None:
    torch.cuda.empty_cache()

    logger.info("Setting training arguments")
    training_args = transformers.TrainingArguments(
        output_dir="model/output",
        eval_strategy="steps",
        load_best_model_at_end=True,
        eval_steps=5,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="model/logs",
        per_device_train_batch_size=1,
    )
    trainer = transformers.Trainer(model, training_args, train_dataset=[tokenized_data])
    result = trainer.train()
    logger.info("Training completed")

    return None
