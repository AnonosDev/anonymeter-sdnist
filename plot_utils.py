import json
import os
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed

repository_path = "./crc_data_and_metric_bundle_1.1"

label_size = 24
fig_type = "pdf"
risks = ["singlingout", "linkability", "inference"]
risk_names_labels = {
    "inference": "Inference",
    "singlingout": "Singling out",
    "linkability": "Linkability",
}


def _compact_label(label):
    old_text = label.get_text()
    new_text = old_text.replace("_", "\n")
    new_text = new_text.replace("-", "\n")
    label.set_text(new_text)
    return label


def compact_labels(ax: plt.Axes) -> plt.Axes:
    xlabels = [_compact_label(ll) for ll in ax.get_xticklabels()]
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(xlabels)
    return ax


def _open_json(json_path: str) -> Dict:
    with open(json_path) as ff:
        return json.load(ff)


def replace_xticks_labels(ax: plt.Axes) -> None:
    dataset_names = {
        "ma2019": "Massachusetts",
        "tx2019": "Texas",
        "national2019": "National",
    }

    old_labels = [label.get_text() for label in ax.get_xticklabels()]
    new_labels = [dataset_names[old_label] for old_label in old_labels]

    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(new_labels)


def remove_legend_duplicates(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())


def combine_library_algo_names(row: pd.Series) -> str:
    lib_name = row["library name"]
    algo_name = row["algorithm name"]

    if lib_name == algo_name:
        return f"{lib_name}"

    # simplify some long names
    if algo_name == "bayesian_network":
        algo_name = "BN"

    return f"{lib_name}-{algo_name}"


def make_axis_bigger(
    ax: plt.Axes,
    tick_size: int = 14,
    label_size: int = 16,
    font_size: int = 18,
) -> plt.Axes:
    for label in ax.get_xticklabels():
        label.set_fontsize(tick_size)
    for label in ax.get_yticklabels():
        label.set_fontsize(tick_size)
    ax.set_xlabel(ax.get_xlabel(), fontsize=label_size)
    ax.set_ylabel(ax.get_ylabel(), fontsize=label_size)
    ax.set_title(ax.get_title(), fontsize=font_size)

    return ax


def load_metrics_from_report(submission_details: pd.Series) -> Dict[str, float]:
    """Extract more metrics form the report of the given submission.

    This function will go and look at the report.json file accompanying each
    submission and extract metrics such as the k_marginal utility score.

    Parameters
    ----------
    submission_details : pd.Series
        A series containing the details of a submission.

    Returns
    -------
    Dict[str, float]
        A dictionary containing the metrics extracted from the report.

    """
    report_path = os.path.join(
        repository_path, submission_details["report path"], "report.json"
    )
    report = _open_json(report_path)

    # this dict maps the name of a metric to the set of
    # set of levels that identify it in the json file report
    metrics = {
        "unique_records": [
            "unique_exact_matches",
            "percent records matched in target data",
        ],
        "k_marginal": ["k_marginal", "k_marginal_synopsys", "k_marginal_score"],
    }

    out = {}
    for name, levels in metrics.items():
        try:
            value = report
            for level in levels:
                value = value[level]
            out[name] = value
        except KeyError:
            print(f"Could not find metric {name} in report {report_path}")

    # print("----------------------")
    # print(json.dumps(report["unique_exact_matches"], indent=4))
    # print("----------------------")
    # print(json.dumps(report["k_marginal"], indent=4))
    # print("----------------------")
    # print(out)
    out["deid data id"] = submission_details["deid data id"]
    return out


def combine_with_report(results: pd.DataFrame) -> pd.DataFrame:
    """Combine the results of the Anonymeter analysis with metrics from the report.

    Parameters
    ----------
    res : pd.DataFrame
        A dataframe containing the results of the analysis.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the results of the analysis and the metrics
        extracted from the report.


    """
    print("extracting info from the reports...")

    with Parallel(n_jobs=-1) as executor:
        report_metrics = executor(
            delayed(load_metrics_from_report)(
                submission_details=submission_details,
            )
            for idx, submission_details in results.reset_index().iterrows()
        )

    report_metrics = pd.DataFrame(report_metrics)
    report_metrics.set_index("deid data id", inplace=True)

    return pd.concat([results, report_metrics], axis=1)


def load_results(base_results_dir: str) -> pd.DataFrame:
    """Combine results of the Anonymeter analysis for different datasets.

    Parameters
    ----------
    base_results_dir : str
        Path to the directory containing the results of the analysis.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the results of the analysis for plotting.

    """
    res_ma = pd.read_csv(
        os.path.join(base_results_dir, "results_ma2019_all_features.csv")
    )
    res_tx = pd.read_csv(
        os.path.join(base_results_dir, "results_tx2019_all_features.csv")
    )
    res_na = pd.read_csv(
        os.path.join(base_results_dir, "results_national2019_all_features.csv")
    )

    res_ma = res_ma.set_index("deid data id")
    res_tx = res_tx.set_index("deid data id")
    res_na = res_na.set_index("deid data id")

    res = pd.concat([res_ma, res_tx, res_na])

    return res


def clean_nanmean(array: np.array) -> np.array:
    """Sometimes all risks are nans, in that case return nan.
    This avoids warnings from np.nanmean.

    There are ~3000 evaluations out of 36000 with all nans
    risks. They comes form a few libraries only. To inspect
    further:

    risks = ["singlingout", "linkability", "inference"]
    problems = res[res[[f"{risk}_risk" for risk in risks]].isna().all(axis=1)]
    problems.groupby(['target dataset', 'library name', 'algorithm name'])['aux_cols'].count()

    """
    out = np.empty(array.shape[0])
    mask = np.isnan(array).all(axis=1)
    out[mask] = np.nan
    out[~mask] = np.nanmean(array[~mask], axis=1)
    return out


def combine_privacy_risks(results: pd.DataFrame) -> pd.DataFrame:
    """Add a column with all the risks to the dataset.

    Exploit the fact that only one risk is not None in each row.
    """
    # aggregate the risk (only one risk is != none at the time)
    privacy_risks = clean_nanmean(results[[f"{r}_risk" for r in risks]].values)

    data = results.copy()
    data = data.assign(privacy_risk=privacy_risks)
    data = data.assign(risk_name="fuffa")
    data.rename(columns={"risk_name": "risk name"}, inplace=True)

    data.loc[data["privacy_risk"] == data["inference_risk"], "risk name"] = "Inference"
    data.loc[
        data["privacy_risk"] == data["singlingout_risk"], "risk name"
    ] = "Singling out"
    data.loc[
        data["privacy_risk"] == data["linkability_risk"], "risk name"
    ] = "Linkability"

    return data
