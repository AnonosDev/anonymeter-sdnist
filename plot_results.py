"""Plot results of the privacy analysis of SDNIST."""


import os

import matplotlib.pyplot as plt
from plot_functions import (
    plot_correlation_with_nist_metrics,
    plot_dp_comparison,
    plot_dp_comparison_in_depth,
    plot_summary_aggregates,
    plot_privacy_utility,
    plot_risks_for_different_algos,
    plot_super_aggregated_risk,
    plot_total_risks,
    plot_total_risks_for_feature_sets_combined,
    plot_total_risks_max_rows,
    plot_worse_libraries_combined,
    print_mean_risks_for_datasets,
)
from plot_utils import combine_with_report, load_results

# --- analysis settings and paths --- #
max_rows = 7634
max_rows = None

base_results_dir = os.path.join(
    "./tiny_paper_results/",
    f"1000_attacks_results_all-features-simple-features-demographic-focused_max_n_rows_{max_rows}",
)

repository_path = "./crc_data_and_metric_bundle_1.1"

base_plot_dir = "./plots_review/"

exclude_libraries = [
    "LostInTheNoise",
    "subsample_1pcnt",
    "subsample_5pcnt",
    "Sarus SDG",
]  # these have only national


plt.rc("legend", fontsize=12)
plt.rc("legend", title_fontsize=12)

n_jobs = -2

# ----------------------------------- #


if __name__ == "__main__":
    plt.switch_backend(
        "agg"
    )  # otherwise: Tcl_AsyncDelete: async handler deleted by the wrong thread

    # load results
    res = load_results(base_results_dir)
    res = combine_with_report(res)
    res = res.reset_index()

    # apply quality cuts
    res = res.query("`library name` not in @exclude_libraries")

    # plot total risks for feature sets
    plot_dir = os.path.join(
        base_plot_dir, os.path.basename(base_results_dir.split("/")[-1])
    )
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    # print aggregated risks per dataset
    print_mean_risks_for_datasets(results=res)

    # algo comparison
    plot_risks_for_different_algos(results=res, max_rows=max_rows, plot_dir=plot_dir)

    # DP comparison
    plot_dp_comparison(
        results=res, max_rows=max_rows, plot_dir=plot_dir, log_epsilon=True
    )
    plot_dp_comparison(
        results=res, max_rows=max_rows, plot_dir=plot_dir, log_epsilon=False
    )

    plot_dp_comparison_in_depth(
        results=load_results(base_results_dir),
        log_epsilon=True,
        max_rows=max_rows,
        plot_dir=plot_dir,
    )
    plot_dp_comparison_in_depth(
        results=load_results(base_results_dir),
        log_epsilon=False,
        max_rows=max_rows,
        plot_dir=plot_dir,
    )

    # overall: total risk
    plot_total_risks(results=res, max_rows=max_rows, plot_dir=plot_dir)
    plot_summary_aggregates(results=res, max_rows=max_rows, plot_dir=plot_dir)
    plot_super_aggregated_risk(results=res, max_rows=max_rows, plot_dir=plot_dir)

    # for the appendix: effect of dataset resizing
    plot_total_risks_max_rows(plot_dir=plot_dir)

    # compare risk between libraries
    plot_worse_libraries_combined(results=res, max_rows=max_rows, plot_dir=plot_dir)
    plot_total_risks_for_feature_sets_combined(
        results=res, max_rows=max_rows, plot_dir=plot_dir
    )

    # correlate with NIST metrics
    plot_correlation_with_nist_metrics(
        results=res, max_rows=max_rows, plot_dir=plot_dir
    )

    # privacy-utility scatterplot
    for feature_set in ["all-features", "simple-features", "demographic-focused", None]:
        plot_privacy_utility(
            results=res,
            feature_set=feature_set,
            max_rows=max_rows,
            plot_dir=plot_dir,
        )
