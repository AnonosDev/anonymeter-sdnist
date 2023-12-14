import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

home_path = os.path.expanduser("~")

original_data_path = os.path.join(
    home_path, "./SDNist/nist diverse communities data excerpts"
)
base_path = "./crc_data_and_metric_bundle_1.1"


def cap_dataset_size(df: pd.DataFrame, max_n_rows: Optional[int]) -> pd.DataFrame:
    """Eventually reduce the size of a dataset by drawing max_rows random samples."""
    if max_n_rows is not None and len(df) > max_n_rows:
        return df.sample(n=max_n_rows, random_state=42, replace=False)
    else:
        return df


def cleanup_datasets(
    original: pd.DataFrame,
    protected: pd.DataFrame,
    control: pd.DataFrame,
    max_n_rows: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Make sure that the datasets are consistent."""
    common_columns = list(
        set.intersection(*[set(df.columns) for df in [original, control, protected]])
    )

    original = original[common_columns]
    protected = protected[common_columns]
    control = control[common_columns]

    protected = protected.astype(original.dtypes)

    print("Original", original.shape, protected.shape, control.shape)

    if len(control) > len(original):
        control = control.sample(n=len(original), random_state=42, replace=False)

    original = cap_dataset_size(df=original, max_n_rows=max_n_rows)
    protected = cap_dataset_size(df=protected, max_n_rows=max_n_rows)
    control = cap_dataset_size(df=control, max_n_rows=max_n_rows)

    return original, protected, control


def append_submission_info_to_results(
    results: List[Dict], submission_info: pd.Series
) -> List[Dict]:
    """Append submission info to the results."""
    info_dict = submission_info.to_dict()

    for res in results:
        res.update(info_dict)

    return results


def load_original_and_control(dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the original and control datasets.

    Parameters
    ----------
    dataset : str
        The name of the dataset to load.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        The original and control datasets.

    """
    if dataset == "ma2019":
        subdir = "massachusetts"
        original_dataset_name = "ma2019.txt"
        control_dataset_name = "ma2018.txt"
    elif dataset == "national2019":
        subdir = "national"
        original_dataset_name = "national2019.txt"
        control_dataset_name = "national2018.txt"
    elif dataset == "tx2019":
        subdir = "texas"
        original_dataset_name = "tx2019.txt"
        control_dataset_name = "tx2018.txt"
    else:
        raise ValueError(f"Invalid dataset name: {dataset}")

    original = pd.read_csv(
        os.path.join(original_data_path, subdir, original_dataset_name)
    )
    control = pd.read_csv(
        os.path.join(original_data_path, subdir, control_dataset_name)
    )

    return original, control
