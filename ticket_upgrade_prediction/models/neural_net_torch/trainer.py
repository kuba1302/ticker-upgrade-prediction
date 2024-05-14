from copyreg import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import mlflow
import pandas as pd
import torch
from data_loader import UpgradeDataset
from loguru import logger
from neural_net_model import Network
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from ticket_upgrade_prediction import Evaluator, Metrics
from ticket_upgrade_prediction.config.env_config import EXPERIMENT_NAME
from ticket_upgrade_prediction.pipeline import Dataset


# def mlflow_run_start_handle(method):
#     def wrapper(*args, **kwargs):
#         mlflow_run_name = kwargs.get("mlflow_run_name", None)
#         to_mlflow = True if mlflow_run_name else False

#         if to_mlflow:
#             client = mlflow.MlflowClient()
#             experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

#             with mlflow.start_run(
#                 run_name=mlflow_run_name,
#                 experiment_id=experiment.experiment_id,
#             ):
#                 method(*args, **kwargs)

#         else:
#             method(*args, **kwargs)

#     return wrapper


@dataclass
class HyperParams:
    layers: List[int]
    optimizer_name: str
    learning_rate: float

    def to_dict(self) -> dict:
        return self.__dict__


class NetworkTrainer:
    def __init__(
        self,
        dataset: Dataset,
        hparams: HyperParams = HyperParams(
            layers=[256, 128, 64], optimizer_name="Adam", learning_rate=0.001
        ),
        epochs: int = 2,
        batch_size: int = 64,
        criterion: Any = BCEWithLogitsLoss(),
        plots_save_path: Optional[Path] = None,
    ) -> None:
        self.dataset = dataset
        self.hparams = hparams
        self.plot_save_path = plots_save_path
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.model = Network(
            input_size=dataset.X_train.shape[1],
            hidden_layers_sizes=self.hparams.layers,
            device=self.device
        )
        self.model = self.model.to(self.device)
        
        self.optimizer = self._get_optimizer(
            optimizer_name=self.hparams.optimizer_name,
            learning_rate=self.hparams.learning_rate,
        )
        self.epochs = epochs
        self.criterion = criterion
        self.batch_size = batch_size

        self.training_results = []

    def _get_optimizer(self, optimizer_name: str, learning_rate: float) -> Any:
        if optimizer_name.lower() == "adam":
            return Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == "sgd":
            return SGD(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError("Wrong type of optimizer provided!")

    def _one_batch(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X = X.to(self.device)
        y = y.to(self.device)

        preds_logit = self.model(X.float())
        loss = self.criterion(preds_logit.float(), y.float())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # @mlflow_run_start_handle
    def fit(self, mlflow_run_name: Optional[str] = None) -> None:
        to_mlflow = bool(mlflow_run_name)

        train_dataset = UpgradeDataset(
            X=self.dataset.X_train, y=self.dataset.y_train.values
        )
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True,
        )

        if to_mlflow:
            mlflow.log_params(self.hparams.to_dict())
        
        for epoch in tqdm(range(self.epochs), desc="Epoch"):
            for X, y in tqdm(iter(train_loader), desc="Batch"):
                self._one_batch(X=X, y=y)

            with torch.no_grad():
                self._evaluate(to_mlflow=to_mlflow, epoch=epoch)

    def _evaluate(self, to_mlflow: bool, epoch: int) -> None:
        evaluator = Evaluator(
            model=self.model, X=self.dataset.X_test, y=self.dataset.y_test, device=self.device
        )
        metrics = evaluator.get_all_metrics(epoch=epoch, to_mlflow=to_mlflow)
        # _ = evaluator.plot_all_plots(
        #     save_path=self.plot_save_path, to_mlflow=to_mlflow
        # )
        logger.info(metrics)
        self.training_results.append(metrics)

    def get_results(self) -> List[Metrics]:
        return self.training_results


if __name__ == "__main__":
    import pickle

    data_path = Path(__file__).parents[3] / "data" / "dataset.pickle"
    with open(data_path, "rb") as file:
        dataset = pickle.load(file)
    
    # dataset = dataset.get_sample()
    # hparams = HyperParams(
    #         layers=[256], optimizer_name="Adam", learning_rate=0.001
    #     ),
    trainer = NetworkTrainer(dataset=dataset, epochs=5)
    trainer.fit(mlflow_run_name="Neural-net-test")

    # data_path_csv = Path(__file__).parents[3] / "data" / "dataset.csv"
    # df = (
    #     pd.read_csv(data_path_csv, index_col=False)
    #     .sample(10000)
    #     .reset_index(drop=True)
    # )
    # params = {
    #     "layers": [[5], [5, 10], [5, 10, 15]],
    #     "optimizer_name": ["adam"],
    #     "learning_rate": [0.001, 0.01, 0.0001],
    # }

    # hyperopt = NeuralNetHyperopt(data=df, hyper_params=params)
    # hyperopt.hyperopt(
    #     target_metric="roc_auc", number_of_hparams_combinations=3
    # )

    # print(hyperopt.get_metrics())
