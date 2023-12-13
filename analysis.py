import os
import warnings
from typing import Dict, List, Optional

import pandas as pd
import tqdm
from analysis_utils import (
    append_submission_info_to_results,
    base_path,
    cleanup_datasets,
    load_original_and_control,
)
from anonymeter_evaluation_lib import (
    InferenceRiskEvaluation,
    LinkabilityRiskEvaluation,
    SinglingOutRiskEvaluation,
)

warnings.filterwarnings("ignore")

n_attacks = 1000
n_jobs = -1
n_steps = 10
max_n_aux_cols = None
datasets = ["ma2019", "tx2019", "national2019"]
max_n_rows = 7634  # make it None to analyze all the datasets
feature_sets = ["all-features", "simple-features", "demographic-focused"]


results_dir = f"./results_{'-'.join(feature_sets)}_max_n_rows_{max_n_rows}"


def analyze_dataset(
    original: pd.DataFrame,
    protected: pd.DataFrame,
    control: pd.DataFrame,
    n_attacks: int,
    n_steps: int,
    n_jobs: int,
    max_n_rows: Optional[int],
) -> List[Dict]:
    """Run the privacy risks analysis for a given dataset."""
    original, protected, control = cleanup_datasets(
        original, protected, control, max_n_rows=max_n_rows
    )

    inference_results = InferenceRiskEvaluation(
        original=original,
        protected=protected,
        control=control,
        n_attacks=n_attacks,
        n_steps=n_steps,
        n_jobs=n_jobs,
        max_n_aux_cols=max_n_aux_cols,
    ).run()

    linkability_results = LinkabilityRiskEvaluation(
        original=original,
        protected=protected,
        control=control,
        n_attacks=n_attacks,
        n_steps=n_steps,
        n_jobs=n_jobs,
        max_n_neighbors=5,
    ).run()

    # we don't want to run the SO with too many columns
    singling_out_max_n_cols = min(10, len(original.columns) - 1)

    singling_out_results = SinglingOutRiskEvaluation(
        original=original,
        protected=protected,
        control=control,
        n_attacks=n_attacks,
        n_steps=n_steps,
        n_jobs=n_jobs,
        max_n_aux_cols=singling_out_max_n_cols
        if max_n_aux_cols is None
        else max_n_aux_cols,
    ).run()

    return inference_results + linkability_results + singling_out_results


if __name__ == "__main__":
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    file_index = pd.read_csv(os.path.join(base_path, "index.csv"))
    file_index = file_index.drop(columns=["Unnamed: 0"])

    all_feature_index = file_index.query("`feature set name` in @feature_sets")

    for dataset in datasets:
        group = all_feature_index.query("`target dataset` == @dataset")

        print(f"=========== Analyzing dataset {dataset} ===========")

        original, control = load_original_and_control(dataset=dataset)

        all_results = []

        for _, row in tqdm.tqdm(group.iterrows(), total=len(group)):
            protected = pd.read_csv(os.path.join(base_path, row["data path"]))

            results = analyze_dataset(
                original=original,
                protected=protected,
                control=control,
                n_attacks=n_attacks,
                n_steps=n_steps,
                n_jobs=n_jobs,
                max_n_rows=max_n_rows,
            )

            results = append_submission_info_to_results(
                results=results, submission_info=row
            )

            all_results.extend(results)

        results_df = pd.DataFrame(all_results)

        fout_name = os.path.join(results_dir, f"results_{dataset}_all_features.csv")
        results_df.to_csv(fout_name, index=False)
        print(f"Results saved to {fout_name}")
