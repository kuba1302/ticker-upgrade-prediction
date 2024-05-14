import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from ticket_upgrade_prediction.models import BaseModel


@dataclass
class Metrics:
    accuracy: Optional[float] = None
    roc_auc: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    pr_auc: Optional[float] = None
    epoch: Optional[int] = None

    def to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_multiple_metrics(cls, *args):
        metrics_list = [metric.to_dict() for metric in args]
        metrics_df = pd.DataFrame(metrics_list).mean()
        return cls(**metrics_df.to_dict())

    def get_metric_from_string(self, metric_name: str) -> float:
        """
        Select one metric from string.
        Possible metric_name values:
        accuracy, roc_auc, precision
        recall, f1, pr_auc, epoch
        """
        metric_mapping = {
            "accuracy": self.accuracy,
            "roc_auc": self.roc_auc,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "pr_auc": self.pr_auc,
            "epoch": self.epoch,
        }

        if metric_name not in metric_mapping.keys():
            raise ValueError(
                f"Wrong metric name provided! Accepted names: {list(metric_mapping.keys())}"
            )

        return metric_mapping[metric_name]


class BaseEvaluator(ABC):
    """Interface for Evaluator"""

    @abstractmethod
    def get_all_metrics(self, to_mlflow: bool):
        """Method for getting all metrics"""

    @abstractmethod
    def plot_all_plots(self, save_path: Path, to_mlflow: bool):
        """Methods for plotting everything"""


class Evaluator(BaseEvaluator):
    def __init__(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: np.ndarray,
        device: Optional[str] = None
    ) -> None:
        self.model = model
        self._assert_model_has_proper_methods()
        self.device = device
        self.X = X
        self.y = y
        self._assert_y_has_proper_type()

        self.preds = self._get_preds()
        self.proba = self._get_proba()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(X shape: {self.X.shape},"
            f"y shape{self.y.shape}"
        )

    def _assert_y_has_proper_type(self) -> None:
        if not isinstance(self.y, np.ndarray):
            try:
                self.y = self.y.values
            except AttributeError as e:
                raise AttributeError(
                    f"{e}. It must be either np.ndarray, pd.Series or 1 dimensional pd.DataFrame"
                )

    def _assert_model_has_proper_methods(self) -> None:
        # TODO! Add support for torch
        required_methods = ["predict", "predict_proba"]
        methods = dir(self.model)

        if not all(
            required_method in methods for required_method in required_methods
        ):
            raise AttributeError(
                f"Wrong model class provided! Model needs to have {required_methods} methods"
            )

    def _get_preds(self) -> np.ndarray:
        return self.model.predict(self.X)

    def _get_proba(self) -> np.ndarray:
        proba = self.model.predict_proba(self.X)

        if isinstance(proba, torch.Tensor):
            if self.device:
                proba = proba.cpu()
            return proba.numpy().reshape(-1)

        else:
            return proba[:, 1]

    def get_accuracy(self) -> float:
        return accuracy_score(y_true=self.y, y_pred=self.preds)  # type: ignore

    def get_roc_auc(self) -> float:
        return roc_auc_score(y_true=self.y, y_score=self.proba)  # type: ignore

    def _get_pr_curve_properties(self) -> tuple:
        precision, recall, thresholds = precision_recall_curve(
            self.y, self.proba
        )
        # code is repeated to be clear what is returned
        return precision, recall, thresholds

    def get_precision(self) -> float:
        return precision_score(y_true=self.y, y_pred=self.preds)  # type: ignore

    def get_recall(self) -> float:
        return recall_score(y_true=self.y, y_pred=self.preds)  # type: ignore

    def get_f1_score(self) -> float:
        return f1_score(y_true=self.y, y_pred=self.preds)  # type: ignore

    def get_pr_auc(self) -> float:
        precision, recall, _ = self._get_pr_curve_properties()
        return auc(recall, precision)

    def get_all_metrics(
        self, to_mlflow: bool = False, epoch: Optional[int] = None
    ) -> Metrics:
        metrics = Metrics(
            accuracy=self.get_accuracy(),
            roc_auc=self.get_roc_auc(),
            precision=self.get_precision(),
            recall=self.get_recall(),
            f1=self.get_f1_score(),
            pr_auc=self.get_pr_auc(),
            epoch=epoch,
        )

        if to_mlflow:
            mlflow.log_metrics(metrics.to_dict())

        return metrics

    def plot_roc_curve(self, save_path: Optional[Path] = None) -> Figure:
        fpr, tpr, thresholds = roc_curve(y_true=self.y, y_score=self.proba)
        auc_score = auc(fpr, tpr)

        gmeans = np.sqrt(tpr * (1 - fpr))
        ix = np.argmax(gmeans)

        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(15, 7)

        ax.plot([0, 1], [0, 1], linestyle="--", label="Random model")
        ax.plot(fpr, tpr, marker=".", label=f"Model - AUC: {auc_score:.3f}")
        ax.scatter(
            fpr[ix],
            tpr[ix],
            marker="o",  # type: ignore
            color="black",
            label=f"Best Threshold={thresholds[ix]:.2f}, G-Mean={gmeans[ix]:.2f}",
        )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="upper left")

        if save_path:
            fig.savefig(str(save_path / "roc_curve.png"))

        return fig

    def plot_precision_recall_curve(
        self, save_path: Optional[Path] = None
    ) -> Figure:
        precision, recall, thresholds = precision_recall_curve(
            self.y, self.proba
        )
        auc_score = self.get_pr_auc()

        fscore = (2 * precision * recall) / (precision + recall)
        ix = np.argmax(fscore)

        no_skill = len(self.y[self.y == 1]) / len(self.y)

        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(15, 7)

        ax.plot([0, 1], [no_skill, no_skill], linestyle="--", label="No Skill")
        ax.plot(
            recall,
            precision,
            marker=".",
            label=f"Model - AUC: {auc_score:.3f}",
        )
        ax.scatter(
            recall[ix],
            precision[ix],
            marker="o",  # type: ignore
            color="black",
            label=f"Best Threshold={thresholds[ix]:.2f}, F-Score={fscore[ix]:.2f}",
        )

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="upper left")

        if save_path:
            fig.savefig(str(save_path / "precision_recall_curve.png"))

        return fig

    def plot_partial_dependency_plot(
        self, save_path: Optional[Path] = None, kind: str = "average"
    ) -> Figure:

        cols = self.X.columns
        pdp = PartialDependenceDisplay.from_estimator(
            self.model, self.X, cols, kind=kind
        )

        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(1 * len(cols), 1.5 * len(cols))
        fig.suptitle("Partial dependency plot", size=15)
        pdp.plot(ax=ax)

        if save_path:
            fig.savefig(str(save_path / "partial_dependence_plot.png"))

        return fig

    def plot_all_plots(
        self, save_path: Optional[Path] = None, to_mlflow: bool = False
    ) -> None:
        if save_path:
            os.makedirs(save_path, exist_ok=True)

        self.plot_partial_dependency_plot(save_path=save_path)
        self.plot_precision_recall_curve(save_path=save_path)
        self.plot_roc_curve(save_path=save_path)

        if to_mlflow:
            mlflow.log_artifacts(str(save_path))


if __name__ == "__main__":
    # Just for testing purposes, to be removed later
    save_path = Path(__file__).parents[1] / "plots"
    X, y = make_classification(n_samples=10000, weights=[0.5])
    X = pd.DataFrame(data=X, columns=[f"col_{x}" for x in range(X.shape[1])])
    y = pd.DataFrame(data=y, columns=["y"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    evaluator = Evaluator(model=model, X=X_test, y=y_test)
    evaluator.plot_precision_recall_curve(save_path=save_path)
    evaluator.plot_roc_curve(save_path=save_path)
    evaluator.plot_partial_dependency_plot(save_path=save_path)

    # print(evaluator.get_all_metrics())
    # mlflow.set_tracking_uri("http://localhost:5000")
    # with mlflow.start_run():
    #     model = RandomForestClassifier()
    #     model.fit(X_train, y_train)
    #     evaluator = Evaluator(model=model, X=X_test, y=y_test)
    #     evaluator.get_all_metrics(to_mlflow=True)
    #     mlflow.sklearn.log_model(
    #         sk_model=model,
    #         artifact_path="sklearn-model",
    #         registered_model_name="rf_random_data",
    #     )
