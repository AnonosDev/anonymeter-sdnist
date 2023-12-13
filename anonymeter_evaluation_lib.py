"""High level evaluation classes to mass-produce Anonymeter evaluations."""


import functools
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from anonymeter.evaluators import (
    InferenceEvaluator,
    LinkabilityEvaluator,
    SinglingOutEvaluator,
)
from anonymeter.stats.confidence import EvaluationResults
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")


def _extract_results(results: Optional[EvaluationResults]) -> Dict[str, Any]:
    if results is not None:
        risk, ci = results.risk().value, results.risk().ci
        attack_rate, attack_rate_err = results.attack_rate
        control_rate, control_rate_err = results.control_rate
        baseline_rate, baseline_rate_err = results.baseline_rate
    else:
        risk, ci = None, (None, None)
        attack_rate, attack_rate_err = None, None
        control_rate, control_rate_err = None, None
        baseline_rate, baseline_rate_err = None, None

    return {
        "risk": risk,
        "risk_l": ci[0],
        "risk_h": ci[1],
        "attack_rate": attack_rate,
        "attack_rate_err": attack_rate_err,
        "control_rate": control_rate,
        "control_rate_err": control_rate_err,
        "baseline_rate": baseline_rate,
        "baseline_rate_err": baseline_rate_err,
    }


def _generate_n_aux_cols_values(
    min_n_aux_cols: int, max_n_aux_cols: int, n_steps: Optional[int]
) -> List[int]:
    """Generate the number of auxiliary columns to use in the privacy attacks."""
    if n_steps is None:
        return np.arange(min_n_aux_cols, max_n_aux_cols).astype(int).tolist()
    else:
        return sorted(
            set(
                np.linspace(min_n_aux_cols, max_n_aux_cols, n_steps)
                .astype(int)
                .tolist()
            )
        )


class InferenceRiskEvaluation:
    """Evaluate the inference risk of a dataset."""

    def __init__(
        self,
        original: pd.DataFrame,
        protected: pd.DataFrame,
        control: pd.DataFrame,
        n_jobs: int,
        n_attacks: int,
        max_n_aux_cols: Optional[int] = None,
        n_steps: Optional[int] = 10,
    ):
        self._original = original
        self._protected = protected
        self._control = control
        self._n_attacks = n_attacks
        self._n_jobs = n_jobs

        columns = self._original.columns
        if max_n_aux_cols is None:
            max_n_aux_cols = len(columns) - 1
        elif max_n_aux_cols > len(columns):
            max_n_aux_cols = len(columns) - 1

        self._n_aux_cols = _generate_n_aux_cols_values(
            min_n_aux_cols=1,
            max_n_aux_cols=max_n_aux_cols,
            n_steps=n_steps,
        )

    def _evaluate_one_column(self, secret: str, aux_cols: List[str]) -> Dict:
        try:
            evaluator = InferenceEvaluator(
                ori=self._original,
                syn=self._protected,
                control=self._control,
                aux_cols=aux_cols,
                secret=secret,
                n_attacks=self._n_attacks,
            )

            evaluator.evaluate(n_jobs=self._n_jobs)

            results = evaluator.results()

        except Exception as ex:
            warnings.warn(f"Inference evaluation failed with {ex}.")
            results = None

        res_dict = _extract_results(results=results)
        res_dict = {f"inference_{k}": v for k, v in res_dict.items()}
        res_dict["risk_name"] = "inference"
        res_dict["aux_cols"] = aux_cols
        res_dict["n_aux"] = len(aux_cols)
        res_dict["secret"] = secret

        return res_dict

    def run(self) -> List[Dict]:
        """Run inference evaluation for different secret columns and with different number of auxiliary columns."""
        secrets = self._original.nunique().sort_values(ascending=False).index.to_list()

        results = []
        for secret in secrets:
            possible_aux = [c for c in self._original.columns if c != secret]

            for n_aux in self._n_aux_cols:
                aux_cols = possible_aux[:n_aux]
                results.append(
                    self._evaluate_one_column(secret=secret, aux_cols=aux_cols)
                )

        return results


def _extract_linkability_results(
    evaluator: LinkabilityEvaluator,
    aux_cols: Tuple[List[str], List[str]],
    n_neighbors: int,
    has_failed: bool,
) -> Dict:
    results = None if has_failed else evaluator.results(n_neighbors=n_neighbors)

    res_dict = _extract_results(results=results)

    res_dict = {f"linkability_{k}": v for k, v in res_dict.items()}
    res_dict["risk_name"] = "linkability"
    res_dict["aux_cols_0"] = aux_cols[0]
    res_dict["aux_cols_1"] = aux_cols[1]
    res_dict["n_neighbors"] = n_neighbors
    res_dict["n_aux_0"] = len(aux_cols[0])
    res_dict["n_aux_1"] = len(aux_cols[1])

    return res_dict


class LinkabilityRiskEvaluation:
    """Evaluate the linkability risk of a dataset."""

    def __init__(
        self,
        original: pd.DataFrame,
        protected: pd.DataFrame,
        control: pd.DataFrame,
        n_jobs: int,
        n_attacks: int,
        max_n_neighbors: int,
        max_n_aux_cols: Optional[int] = None,
        n_steps: Optional[int] = 10,
    ):
        self._original = original
        self._protected = protected
        self._control = control
        self._n_attacks = n_attacks
        self._n_jobs = n_jobs
        self._max_n_neighbors = max_n_neighbors

        columns = self._original.columns
        if max_n_aux_cols is None:
            max_n_aux_cols = len(columns) - 1
        elif max_n_aux_cols > len(columns):
            max_n_aux_cols = len(columns) - 1

        self._n_aux_cols = _generate_n_aux_cols_values(
            min_n_aux_cols=2,
            max_n_aux_cols=max_n_aux_cols,
            n_steps=n_steps,
        )

    def run(self) -> List[Dict]:
        """Run the linkability analysis for a set of auxiliary columns."""
        sorted_columns = (
            self._original.nunique().sort_values(ascending=False).index.to_list()
        )

        results = []
        for n_aux in self._n_aux_cols:
            all_aux_cols = sorted_columns[:n_aux]
            aux_cols = (all_aux_cols[::2], all_aux_cols[1::2])

            try:
                evaluator = LinkabilityEvaluator(
                    ori=self._original,
                    syn=self._protected,
                    control=self._control,
                    aux_cols=aux_cols,
                    n_neighbors=self._max_n_neighbors,
                    n_attacks=self._n_attacks,
                )
                evaluator.evaluate(n_jobs=self._n_jobs)
                has_failed = False

            except Exception as ex:
                warnings.warn(f"Linkability evaluation failed with {ex}.")
                has_failed = True

            for n_neighbors in range(1, self._max_n_neighbors + 1):
                results.append(
                    _extract_linkability_results(
                        evaluator=evaluator,
                        aux_cols=aux_cols,
                        n_neighbors=n_neighbors,
                        has_failed=has_failed,
                    )
                )

        return results


def _so_attack(
    original: pd.DataFrame,
    protected: pd.DataFrame,
    control: pd.DataFrame,
    n_aux_cols: int,
    n_attacks: int,
) -> Optional[Dict[str, float]]:
    mode = "univariate" if n_aux_cols == 1 else "multivariate"

    try:
        evaluator = SinglingOutEvaluator(
            ori=original,
            syn=protected,
            control=control,
            n_cols=n_aux_cols,
            n_attacks=n_attacks,
            max_attempts=1000000,
        )

        evaluator.evaluate(mode=mode)

        results = evaluator.results()

    except Exception as ex:
        print(f"Singling out evaluation failed with {ex}.")
        results = None

    res_dict = _extract_results(results=results)
    res_dict = {f"singlingout_{k}": v for k, v in res_dict.items()}
    res_dict["risk_name"] = "singlingout"
    res_dict["mode"] = mode
    res_dict["n_aux_cols"] = n_aux_cols

    return res_dict


class SinglingOutRiskEvaluation:
    """Evaluate singling out risk of a dataset."""

    def __init__(
        self,
        original: pd.DataFrame,
        protected: pd.DataFrame,
        control: pd.DataFrame,
        n_jobs: int,
        n_attacks: int,
        max_n_aux_cols: Optional[int] = None,
        n_steps: Optional[int] = 10,
    ):
        self._original = original
        self._protected = protected
        self._control = control
        self._n_attacks = n_attacks
        self._n_jobs = n_jobs

        if max_n_aux_cols is None:
            max_n_aux_cols = len(self._original.columns) - 1

        self._n_aux_cols = _generate_n_aux_cols_values(
            min_n_aux_cols=1,
            max_n_aux_cols=max_n_aux_cols,
            n_steps=n_steps,
        )

    def _attack(self, n_aux_cols: int) -> Dict[str, float]:
        job = functools.partial(
            _so_attack,
            original=self._original,
            protected=self._protected,
            control=self._control,
            n_aux_cols=n_aux_cols,
            n_attacks=self._n_attacks,
        )

        job_results = job(n_aux_cols=n_aux_cols)

        if job_results is None:
            job_results = {}

        return job_results

    def run(self) -> List[Dict]:
        """Run singling out risk evaluation for a set of auxiliary columns."""
        results = []

        with Parallel(n_jobs=self._n_jobs) as executor:
            results = executor(
                delayed(self._attack)(n_aux_cols=n_aux_cols)
                for n_aux_cols in self._n_aux_cols
            )

        return results
