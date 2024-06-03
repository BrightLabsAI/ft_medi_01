import logging

import transformers
from peft import LoraConfig, TaskType, get_peft_model

logger = logging.getLogger(__name__)


def download_model(model_name: str) -> transformers.AutoModelForCausalLM:
    """
    Downloads a pre-trained model from Hugging Face's Transformers library and applies the PEFT model to it.

    Args:
        model_name (str): The name of the pre-trained model to download.

    Returns:
        transformers.AutoModelForCausalLM: The downloaded and transformed model.

    Raises:
        None

    Examples:
        >>> download_model("gpt2")
        Downloading model gpt2
        Model parameters: ...
        <AutoModelForCausalLM>

    """
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
    """
    Downloads a tokenizer for the specified model from the Hugging Face Transformers library.

    Args:
        model_name (str): The name of the pre-trained model whose tokenizer will be downloaded.

    Returns:
        transformers.AutoTokenizer: The downloaded tokenizer.

    Raises:
        None

    Examples:
        >>> download_tokenizer("bert-base-uncased")
        Downloading tokenizer bert-base-uncased
        <AutoTokenizer>
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    logging.info(tokenizer)
    return tokenizer
