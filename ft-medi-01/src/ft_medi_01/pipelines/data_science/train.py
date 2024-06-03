import logging
import transformers
from datasets import Dataset
from pandas import DataFrame

logger = logging.getLogger(__name__)


def tokenize_data(df: DataFrame, tokenizer: transformers.AutoTokenizer) -> Dataset:
    logger.info('Tokenizing dataset')
    tokenizer.pad_token = tokenizer.eos_token
    text_list = df['text'].tolist()
    tokenized_data = tokenizer(text_list, 
                               truncation=True, 
                               padding="max_length", 
                               return_tensors="pt")

    return tokenized_data


def train_model(tokenized_data: Dataset, 
                model: transformers.AutoModelForCausalLM) -> None:

    training_args = transformers.TrainingArguments(
        output_dir="test-trainer",
        eval_strategy="steps",
        eval_steps=5,
        learning_rate=2e-5,
        weight_decay=0.01,
        )
    trainer = transformers.Trainer(
        model,
        training_args,
        train_dataset=[tokenized_data]
    )

    trainer.train()

    logger.info('Training model') 
    logger.info(model)

    return None
