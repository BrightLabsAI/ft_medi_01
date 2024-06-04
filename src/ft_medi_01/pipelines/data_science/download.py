import logging
from typing import Dict

import transformers

logger = logging.getLogger(__name__)


def download_tokenizer(params_model: Dict) -> transformers.AutoTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(params_model["model_name"])
    logging.info(tokenizer)
    return tokenizer
