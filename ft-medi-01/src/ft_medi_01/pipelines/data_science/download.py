import logging

import transformers
from peft import LoraConfig, TaskType, get_peft_model

logger = logging.getLogger(__name__)


def download_model(model_name: str) -> transformers.AutoModelForCausalLM:
    logger.info("Downloading model %s", model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)
    logger.info(model.print_trainable_parameters())

    return model


def download_tokenizer(model_name: str) -> transformers.AutoTokenizer:
    logger.info("Downloading tokenizer %s", model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    logging.info(tokenizer)
    return tokenizer
