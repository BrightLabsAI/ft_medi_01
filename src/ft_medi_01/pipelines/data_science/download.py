import logging
from typing import Dict

import transformers

logger = logging.getLogger(__name__)


def download_tokenizer(params_model: Dict) -> transformers.AutoTokenizer:
    """
    Downloads a tokenizer from Hugging Face's transformers library.

    Args:
        params_model (Dict): The parameters for the model.

    Returns:
        transformers.AutoTokenizer: The downloaded tokenizer.

    Raises:
        None

    Examples:
        >>> download_tokenizer("model_name")
        Downloading tokenizer model_name
        <transformers.tokenization_utils_base.PreTrainedTokenizer>
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(params_model["model_name"])
    logging.info(tokenizer)
    return tokenizer
