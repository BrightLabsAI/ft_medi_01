# logging
import logging

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)


def download_from_huggingface(dataset_name: str) -> pd.DataFrame:
    """
    Downloads a dataset from Hugging Face's Datasets library and converts it to a pandas DataFrame.

    Args:
        dataset_name (str): The name of the dataset to download.

    Returns:
        pd.DataFrame: The downloaded dataset as a pandas DataFrame.

    Raises:
        None

    Examples:
        >>> download_from_huggingface("glue")
        Downloading dataset glue
        Converting dataset to pandas and saving to catalog
        <pandas.core.frame.DataFrame>
    """
    logger.info("Downloading dataset %s", dataset_name)
    ds = load_dataset(dataset_name)
    logger.info("Converting dataset to pandas and saving to catalog")
    df = pd.DataFrame(ds["train"])
    return df


def pre_process(data_percent: float, df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-processes a given DataFrame by sampling a percentage of the data, removing extra whitespace from the 'output' and 'input' columns, concatenating the 'input' and 'output' columns with a separator, and dropping the 'input' and 'output' columns.

    Args:
        data_percent (float): The percentage of the data to sample.
        df (pd.DataFrame): The DataFrame to pre-process.

    Returns:
        pd.DataFrame: The pre-processed DataFrame.

    Raises:
        None

    Examples:
        >>> pre_process(0.5, df)
        Pre-processing dataset
        Using 50.0% of the dataset
        <pandas.core.frame.DataFrame>
    """
    logger.info("Pre-processing dataset")
    df = df.sample(frac=data_percent, random_state=42, axis=0)
    logger.warning(f"Using {data_percent*100}% of the dataset")

    df["output"] = df["output"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["input"] = df["input"].str.replace(r"\s+", " ", regex=True).str.strip()

    df["text"] = "question: " + df["input"] + "answer: " + df["output"]
    sdf = df.drop(["input", "output"], axis=1)
    return sdf
