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
    logger.info("Tokenizing dataset")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        return tokenizer(
            example["text"], truncation=True, max_length=tokenizer.model_max_length
        )

    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    return tokenized_dataset


def prepare_model(params_model: Dict) -> peft_model.PeftModelForCausalLM:
    logger.info("Downloading model %s", params_model["model_name"])
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        params_model["model_name"],
        trust_remote_code=True,
        quantization_config=nf4_config,
    )

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
    tokenized_data: Dataset,
    model: peft_model.PeftModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
) -> None:
    torch.cuda.empty_cache()

    # create a data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=False, return_tensors="pt"
    )

    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Setting training arguments")

    training_args = transformers.TrainingArguments(
        output_dir="model/output",
        eval_strategy="steps",
        eval_steps=5000,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="model/logs",
        per_device_train_batch_size=16,  # Reduced batch size
        gradient_accumulation_steps=2,  # Added gradient accumulation
    )

    trainer = transformers.Trainer(
        model, training_args, train_dataset=tokenized_data, data_collator=data_collator
    )
    trainer.train()
    logger.info("Training completed")

    return None
