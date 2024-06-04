import pandas as pd
from unittest.mock import MagicMock, patch
from ft_medi_01.pipelines.data_process.nodes import download_from_huggingface

class TestDownloadFromHuggingface:

    @patch("datasets.load_dataset")
    def test_download_from_huggingface(self, mock_load_dataset):
        # Test case: Downloading dataset and converting to pandas DataFrame
        mock_ds = MagicMock()
        mock_ds.__getitem__.return_value = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        mock_load_dataset.return_value = mock_ds
        params_dataset = {"dataset_name": "ShenRuililin/MedicalQnA"}
        df = download_from_huggingface(params_dataset)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (16412, 2)
