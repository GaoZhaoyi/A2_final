import random

import numpy as np
import torch
from datasets import DatasetDict

TEST_DATASET_FINGERPRINT = "04ab8679dce4fdf4"
TEST_DATASET_ROW_NUMBER = 3981
TEST_DATASET_SIZE_IN_BYTES = 10007859714
SEED = 42


def not_change_test_dataset(raw_datasets: DatasetDict) -> bool:
    """
    Check if the test dataset is not changed.

    Args:
        raw_datasets: Raw datasets.

    Returns:
        True if the test dataset is not changed, False otherwise.
    """
    raw_datasets_test = raw_datasets["test"]
    return (
        raw_datasets_test.num_rows == TEST_DATASET_ROW_NUMBER
        and raw_datasets_test.size_in_bytes == TEST_DATASET_SIZE_IN_BYTES
        and raw_datasets_test._fingerprint == TEST_DATASET_FINGERPRINT
    )


def set_random_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
