"""Collection of functions to create the tiny paper plots."""

import os
import warnings
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plot_utils import (
    clean_nanmean,
    combine_library_algo_names,
    combine_privacy_risks,
    compact_labels,
    fig_type,
    load_results,
    make_axis_bigger,
    remove_legend_duplicates,
    replace_xticks_labels,
    risk_names_labels,
    risks,
)


def plot_risks_for_different_algos(results: pd.DataFrame, max_rows: int, plot_dir: str):
    res = results.reset_index()

    # figure out the libraries with more than one algo
    counts = res.groupby(["library name"])["algorithm name"].nunique()
    libraries = counts[counts > 1].index

    fig, axes = plt.subplots(len(libraries), 3, figsize=(12, 25))

    for i_row, library_name in enumerate(libraries):
        data = res.query("`library name` == @library_name")

        for i_col, risk in enumerate(risks):
            ax = axes[i_row, i_col]

            sns.barplot(
                x="algorithm name",
                y=f"{risk}_risk",
                hue="target dataset",
                data=data,
                ax=ax,
            )

            ax = compact_labels(ax=ax)

            tag = library_name
            if tag == "smartnoise-synth":
                tag = "smartnoise"
            ax.set_title(f"{tag} - {risk}")

            if i_col == 0:
                ax.set_ylabel("Privacy Risk")
            else:
                ax.set_ylabel("")
                ax.get_legend().remove()

            if i_row != len(libraries) - 1:
                if ax.get_legend() is not None:
                    ax.get_legend().remove()

            ax.set_xlabel("")
            ax.tick_params(axis="x", labelrotation=90)
            make_axis_bigger(ax)

    fig.tight_layout()
    fig.savefig(
        os.path.join(
            plot_dir,
            f"algo_comparison_{max_rows}.{fig_type}",
        ),
        dpi=300,
    )

    plt.close(fig)


def plot_total_risks_max_rows(plot_dir: str) -> None:
    """For each feature set make a bar plot of the total risk for each dataset."""

    results_no_cap = load_results(
        base_results_dir=os.path.join(
            "./tiny_paper_results/",
            "1000_attacks_results_all-features-simple-features-demographic-focused_max_n_rows_None",
        )
    )
    results_no_cap["resized"] = "No"

    results_cap = load_results(
        base_results_dir=os.path.join(
            "./tiny_paper_results/",
            "1000_attacks_results_all-features-simple-features-demographic-focused_max_n_rows_7634",
        )
    )
    results_cap["resized"] = "Yes"

    results = pd.concat([results_cap, results_no_cap]).reset_index()

    # aggregate the risk
    avg_risks = clean_nanmean(results[[f"{r}_risk" for r in risks]].values)
    results["avg_risk"] = avg_risks

    fig, ax = plt.subplots()
    sns.barplot(
        x="target dataset",
        y="avg_risk",
        hue="resized",
        data=results,
        ax=ax,
    )
    ax.set_ylabel("Privacy risk")

    replace_xticks_labels(ax)
    ax.set_ylim(0, 0.07)
    ax.set_xlabel("")

    make_axis_bigger(ax)

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            plot_dir,
            f"all_aggregated_risk_maxrows_comparison.{fig_type}",
        ),
        dpi=300,
    )
    plt.close(fig)


def plot_super_aggregated_risk(
    results: pd.DataFrame, max_rows: int, plot_dir: str
) -> None:
    """Most aggregated plot of the all."""

    fig, ax = plt.subplots()

    # aggregate the risk (only one risk is != none at the time)
    avg_risks = clean_nanmean(results[[f"{r}_risk" for r in risks]].values)

    data = results.copy()
    data["avg_risk"] = avg_risks

    sns.barplot(
        x="target dataset",
        y="avg_risk",
        color="tab:blue",
        data=data,
        ax=ax,
    )
    ax.set_ylabel("Privacy risk")

    replace_xticks_labels(ax)
    ax.set_ylim(0, 0.07)
    ax.set_xlabel("")

    make_axis_bigger(ax)

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            plot_dir,
            f"all_aggregated_risk_maxrows_{max_rows}.{fig_type}",
        ),
        dpi=300,
    )
    plt.close(fig)


def plot_summary_aggregates(
    results: pd.DataFrame, max_rows: int, plot_dir: str
) -> None:
    """Most aggregated plot of the all."""

    fig, ax = plt.subplots()

    data = combine_privacy_risks(results=results)

    sns.barplot(
        x="target dataset",
        y="privacy_risk",
        hue="risk name",
        hue_order=["Singling out", "Inference", "Linkability"],
        data=data,
        palette="Set1",
        ax=ax,
    )
    ax.set_ylabel("Privacy risk")

    replace_xticks_labels(ax)
    ax.set_ylim(0, 0.11)
    ax.set_xlabel("")

    make_axis_bigger(ax)

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            plot_dir,
            f"all_moneyplot_maxrows_{max_rows}.{fig_type}",
        ),
        dpi=300,
    )
    plt.close(fig)


def plot_worse_libraries_combined(
    results: pd.DataFrame,
    max_rows: int,
    plot_dir: str,
):
    """Make a barplot ranking the libraries by their average risk."""

    included_feature_sets = ["all-features", "demographic-focused", "simple-features"]

    data = combine_privacy_risks(
        results=results.query("`library name` != 'subsample_40pcnt'")
    )
    data = data.query("`feature set name` in @included_feature_sets")

    data.loc[:, "full_name"] = [
        combine_library_algo_names(row) for _, row in data.iterrows()
    ]

    # make the color consistent among algos
    names = data["full_name"].unique()
    palette = sns.color_palette(n_colors=len(names))
    color_mapping = dict(zip(names, palette))

    y_var = "privacy_risk"
    order = data.groupby("full_name")[y_var].mean().sort_values(ascending=False).index

    fig, ax = plt.subplots()

    with warnings.catch_warnings():
        # Passing `palette` without assigning `hue` is deprecated
        warnings.simplefilter(action="ignore", category=FutureWarning)
        sns.barplot(
            x="full_name",
            y=y_var,
            palette=[color_mapping[name] for name in order],
            data=data,
            order=order,
            ax=ax,
        )

    ax.set_ylabel("Privacy risk")
    ax.set_xlabel("")

    ax.tick_params(axis="x", labelrotation=90)

    make_axis_bigger(ax)

    fig.tight_layout()
    fig.savefig(
        os.path.join(
            plot_dir,
            f"combined_privacy_risks_ranking_maxrows_{max_rows}_fsets_{''.join(included_feature_sets)}.{fig_type}",
        ),
        dpi=300,
    )


def plot_dp_comparison_in_depth(
    results: pd.DataFrame, log_epsilon: bool, max_rows: int, plot_dir: str
):
    # do not exclude any library
    data = results.query("`privacy category` == 'dp'").reset_index(drop=True)

    # data = data.query("dp < 11")
    data = data.query("`library name` != 'rsynthpop'")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    risk_names_labels = {
        "inference": "Inference",
        "singlingout": "Singling out",
        "linkability": "Linkability",
    }

    data = data.assign(loge=np.log(data["epsilon"]))

    for ii, risk in enumerate(risks):
        y_var = f"{risk}_risk"

        ax = axes[ii]
        sns.pointplot(
            x="loge" if log_epsilon else "epsilon",
            y=y_var,
            hue="library name",
            data=data,
            native_scale=True,
            legend=None,
            ax=ax,
        )
        sns.lineplot(
            x="loge" if log_epsilon else "epsilon",
            y=y_var,
            hue="library name",
            data=data,
            ax=ax,
        )
        ax.set_title(f"{risk_names_labels[risk]}")

        if ii == 0:
            ax.set_ylabel("Privacy Risk")
        else:
            ax.set_ylabel("")
            ax.get_legend().remove()

        if log_epsilon:
            ax.set_xlabel("$\ln(\epsilon$)")
        else:
            ax.set_xlabel("$\epsilon$")

        # ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))

        make_axis_bigger(ax)

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                plot_dir,
                f"dp_comparison_in_depth_maxrows_{max_rows}_log{log_epsilon}.{fig_type}",
            ),
            dpi=300,
        )

        plt.close(fig)


def plot_dp_comparison(
    results: pd.DataFrame, max_rows: int, plot_dir: str, log_epsilon: bool = False
):
    data = results.query("`privacy category` == 'dp'").reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    risk_names_labels = {
        "inference": "Inference",
        "singlingout": "Singling out",
        "linkability": "Linkability",
    }

    for ii, risk in enumerate(risks):
        y_var = f"{risk}_risk"

        ax = axes[ii]
        sns.lineplot(
            x="epsilon",
            y=y_var,
            data=data,
            ax=ax,
        )
        ax.set_title(f"{risk_names_labels[risk]}")

        if ii == 0:
            ax.set_ylabel("Privacy Risk")
        else:
            ax.set_ylabel("")

        if log_epsilon:
            ax.set_xscale("log")
            ax.set_xlabel("$\ln(\epsilon$)")
        else:
            ax.set_xlabel("$\epsilon$")

        make_axis_bigger(ax)

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                plot_dir,
                f"dp_comparison_maxrows_{max_rows}_logE_{log_epsilon}.{fig_type}",
            ),
            dpi=300,
        )

        fig.tight_layout()


def plot_privacy_utility(
    results: pd.DataFrame, feature_set: List[str], max_rows: int, plot_dir: str
):
    """Make privacy utility plot for each library."""
    if feature_set is not None:
        data = results.query("`feature set name` == @feature_set")
    else:
        data = results.copy()

    data = data.assign(
        full_name=[combine_library_algo_names(row) for _, row in data.iterrows()]
    )
    data = combine_privacy_risks(results=data)

    # colors distinguish libraries, markers algorithms
    names = data["library name"].unique()
    palette = sns.color_palette(n_colors=len(names))
    color_mapping = dict(zip(names, palette))

    markers = [".", ",", "o", "v", "^", "<", ">"]

    fig, ax = plt.subplots()
    legend_handles = {}

    for library, library_group in data.groupby("library name"):
        color = color_mapping[library]

        plots = []

        for ii, (algo, group) in enumerate(library_group.groupby("algorithm name")):
            x, xerr = group["k_marginal"].mean(), group["k_marginal"].std()
            y, yerr = group["privacy_risk"].mean(), group["privacy_risk"].std()

            scatterplot = ax.scatter(x, y, color=color, marker=markers[ii])

            ax.errorbar(x=x, y=y, xerr=xerr, yerr=yerr, color=color, label=library)
            plots.append(scatterplot)

        legend_handles[library] = plots
    ax.legend()
    remove_legend_duplicates(ax)

    ax.set_xlabel("K-marginal utility score")
    ax.set_ylabel("Privacy risk")
    # ax.set_ylim(-0.005, 0.03)
    # ax.set_xlim(250, 1050)
    make_axis_bigger(ax)

    fig.tight_layout()
    fig.savefig(
        os.path.join(
            plot_dir,
            f"privacy_utility_maxrows_{max_rows}_fset_{feature_set}.{fig_type}",
        ),
        dpi=300,
    )
    plt.close(fig)


def plot_correlation_with_nist_metrics(
    results: pd.DataFrame, max_rows: int, plot_dir: str
) -> None:
    """Plot the correlation between the number of unique records and the risk."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    risks_names = {
        "singling out": "singlingout_risk",
        "linkability": "linkability_risk",
        "inference": "inference_risk",
    }
    n_bins = 10

    # plot the risk vs the number of unique records
    for label, column in risks_names.items():
        sns.regplot(
            x="unique_records",
            y=column,
            x_bins=n_bins,
            label=label,
            data=results,
            ax=axes[0],
            truncate=False,
        )

    # now the risks vs the k-marginal utility score
    for label, column in risks_names.items():
        sns.regplot(
            x="k_marginal",
            y=column,
            x_bins=n_bins,
            label=label,
            data=results,
            ax=axes[1],
            truncate=False,
        )

    axes[0].set_xlabel("Fraction of unique records")
    axes[1].set_xlabel("K-marginal utility score")

    axes[0].set_ylabel("Privacy risk")
    axes[1].set_ylabel("")

    axes[0].set_xlim(-1, 45)
    axes[1].set_xlim(500, 1000)

    for ax in axes:
        ax.legend(loc="upper left")
        ax.set_ylim(-0.01, 0.35)
        make_axis_bigger(ax)

    fig.tight_layout()
    fig.savefig(
        os.path.join(
            plot_dir,
            f"risks_correaltions_report_maxrows_{max_rows}.{fig_type}",
        ),
        dpi=300,
    )
    plt.close(fig)

    from scipy.stats import pearsonr

    print("-------------- correlations:")
    coeff, _ = pearsonr(results["unique_records"], results["k_marginal"])
    print("unique vs k marginal", round(coeff, 2))

    for risk in risks:
        values = results[f"{risk}_risk"].to_numpy()
        mask = ~np.isnan(values)
        from scipy.stats import pearsonr

        coeff, _ = pearsonr(results["unique_records"].to_numpy()[mask], values[mask])

        print(risk, "pearson coeff", round(coeff, 2))

    print(f"Plots correlating risk with report metrics saved to {plot_dir}")


def plot_total_risks_for_feature_sets_combined(
    results: pd.DataFrame, max_rows: int, plot_dir: str
) -> None:
    """For each privacy risk make a bar plot of the total risk
    for each dataset colored for feature set."""

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    for ii, risk in enumerate(risks):
        ax = axes[ii]
        sns.barplot(
            x="target dataset",
            y=f"{risk}_risk",
            hue="feature set name",
            data=results,
            ax=ax,
        )

        if ii != 2:
            ax.get_legend().remove()

        if ii == 0:
            ax.set_ylabel("Privacy risk")
        else:
            ax.set_ylabel("")

        ax.set_title(risk_names_labels[risk])
        ax.set_xlabel("")
        replace_xticks_labels(ax)
        ax.set_ylim(0, 0.15)

        make_axis_bigger(ax)

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            plot_dir,
            f"total_risks_feature_sets_maxrows_{max_rows}.{fig_type}",
        ),
        dpi=300,
    )
    plt.close(fig)

    print(f"Plots for total risk per datasets are saved to {plot_dir}")


def plot_total_risks(results: pd.DataFrame, max_rows: int, plot_dir: str) -> None:
    """For each feature set make a bar plot of the total risk for each dataset."""

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    risk_names_labels = {
        "inference": "Inference",
        "singlingout": "Singling out",
        "linkability": "Linkability",
    }

    for ii, risk in enumerate(risks):
        ax = axes[ii]
        sns.barplot(
            x="target dataset",
            y=f"{risk}_risk",
            data=results,
            ax=ax,
            color="steelblue",
        )

        if ii == 0:
            ax.set_ylabel("Privacy risk")
        else:
            ax.set_ylabel("")

        ax.set_title(risk_names_labels[risk])
        ax.set_xlabel("")
        replace_xticks_labels(ax)
        ax.set_ylim(0, 0.1)

        make_axis_bigger(ax)

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            plot_dir,
            f"total_risks_maxrows_{max_rows}.{fig_type}",
        ),
        dpi=300,
    )
    plt.close(fig)

    print(f"Plots for total risk per datasets are saved to {plot_dir}")


def print_mean_risks_for_datasets(results: pd.DataFrame) -> None:
    """Print average risk and standard error of the mean for each dataset."""

    for dataset, group in results.groupby("target dataset"):
        all_risks = []
        for risk in risks:
            all_risks.extend(group[f"{risk}_risk"].dropna().to_list())

        all_risks = np.array(all_risks)

        mean_risk = np.mean(all_risks)
        seo_risk = np.std(all_risks) / np.sqrt(len(all_risks))

        print(
            f"For dataset {dataset} the mean risks is: {mean_risk:.1e} +- {seo_risk:.0e}"
        )
