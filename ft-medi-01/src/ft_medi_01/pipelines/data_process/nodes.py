# logging
import logging

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)


def download_from_huggingface(dataset_name: str) -> pd.DataFrame:
    logger.info("Downloading dataset %s", dataset_name)
    ds = load_dataset(dataset_name)
    logger.info("Converting dataset to pandas and saving to catalog")
    df = pd.DataFrame(ds["train"])
    return df


def pre_process(data_percent: float, df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Pre-processing dataset")
    df = df.sample(frac=data_percent, random_state=42, axis=0)
    logger.warning(f"Using {data_percent*100}% of the dataset")

    df["output"] = df["output"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["input"] = df["input"].str.replace(r"\s+", " ", regex=True).str.strip()

    df["text"] = "question: " + df["input"] + "answer: " + df["output"]
    sdf = df.drop(["input", "output"], axis=1)
    return sdf
