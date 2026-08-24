"""
Base experiment class for probabilistic forecasting.

Extends torch_timeseries.experiments.ForecastExp with probabilistic metrics
(CRPS, QICE, PICP, ...) computed over sampled prediction ensembles. Model
experiments subclass this and implement _init_model / _process_train_batch /
_process_val_batch.
"""
from dataclasses import asdict, dataclass
import datetime
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.multiprocessing as mp
from torchmetrics import MetricCollection
from tqdm import tqdm

from src.metrics import CRPS, CRPSSum, QICE, PICP
from src.metrics import ProbMAE, ProbMSE, ProbRMSE
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible
from torch_timeseries.experiments import ForecastExp


def update_metrics(preds, truths, metrics):
    """Update metrics in a worker process."""
    metrics.update(preds, truths)


@dataclass
class ProbForecastExp(ForecastExp):
    loss_func_type: str = 'mse'
    epochs: int = 10

    def _init_metrics(self):
        self.metrics = MetricCollection(
            metrics={
                "crps": CRPS(),
                "crps_sum": CRPSSum(),
                "qice": QICE(),
                "picp": PICP(),
                "mse": ProbMSE(),
                "mae": ProbMAE(),
                "rmse": ProbRMSE(),
            }
        )
        self.metrics.to("cpu")
        ctx = mp.get_context("spawn")
        self.task_pool = ctx.Pool(processes=32)

    def _evaluate(self, dataloader):
        self.model.eval()
        self.metrics.reset()
        results = []
        with torch.no_grad(), tqdm(total=len(dataloader.dataset)) as progress_bar:
            for batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc, *rest in dataloader:
                origin_y = origin_y.to(self.device)
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()
                preds, truths = self._process_val_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )
                # Denormalize so probabilistic metrics are computed on the real scale
                preds = self.scaler.inverse_transform(preds)
                truths = origin_y
                results.append(self.task_pool.apply_async(
                    update_metrics,
                    (preds.contiguous().cpu().detach(),
                     truths.contiguous().cpu().detach(),
                     self.metrics)
                ))
                progress_bar.update(batch_x.shape[0])

        for result in results:
            result.get()

        return {name: float(metric.compute()) for name, metric in self.metrics.items()}

    def _test(self) -> Dict[str, float]:
        print("Testing .... ")
        test_result = self._evaluate(self.test_loader)
        self._run_print(f"test_results: {test_result}")
        return test_result

    def _val(self):
        print("Validating .... ")
        val_result = self._evaluate(self.val_loader)
        self._run_print(f"vali_results: {val_result}")
        return val_result

    def _check_run_exist(self, seed: str):
        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir)
            print(f"Creating running results saving dir: '{self.run_save_dir}'.")
        else:
            print(f"result directory exists: {self.run_save_dir}")
        with open(
            os.path.join(self.run_save_dir, "args.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=4)

        return os.path.exists(self.run_checkpoint_filepath)

    def _load_best_model(self):
        self.model.load_state_dict(
            torch.load(self.best_checkpoint_filepath, map_location=self.device)
        )

    def _run_print(self, *args, **kwargs):
        time_str = "[" + str(datetime.datetime.now())[:19] + "] -"
        print(*args, **kwargs)
        with open(os.path.join(self.run_save_dir, "output.log"), "a+") as f:
            print(time_str, *args, flush=True, file=f)

    def _resume_run(self, seed):
        run_checkpoint_filepath = os.path.join(self.run_save_dir, "run_checkpoint.pth")
        print(f"resuming from {run_checkpoint_filepath}")

        check_point = torch.load(run_checkpoint_filepath, map_location=self.device)
        self.model.load_state_dict(check_point["model"])
        self.model_optim.load_state_dict(check_point["optimizer"])
        self.current_epoch = check_point["current_epoch"]
        self.early_stopper.set_state(check_point["early_stopping"])

    def run(self, seed=42) -> Dict[str, float]:
        self._setup_run(seed)
        if self._check_run_exist(seed):
            self._resume_run(seed)

        self._run_print(f"run : {self.current_run} in seed: {seed}")

        parameter_tables, model_parameters_num = count_parameters(self.model)
        self._run_print(f"parameter_tables: {parameter_tables}")
        self._run_print(f"model parameters: {model_parameters_num}")

        while self.current_epoch < self.epochs:
            epoch_start_time = time.time()
            if self.early_stopper.early_stop is True:
                self._run_print(
                    f"val loss no decreased for patience={self.patience} epochs, early stopping ...."
                )
                break

            # seed depends on epoch so training remains reproducible after resume
            reproducible(seed + self.current_epoch)
            train_losses = self._train()
            self._run_print(
                "Epoch: {} cost time: {}s".format(
                    self.current_epoch + 1, time.time() - epoch_start_time
                )
            )
            self._run_print(f"Training loss : {np.mean(train_losses)}")

            val_result = self._val()
            self._test()

            self.current_epoch = self.current_epoch + 1
            self.early_stopper(val_result['crps'], model=self.model)
            self._save_run_check_point(seed)

        self._load_best_model()
        return self._test()

    def runs(self, seeds: List[int] = [1, 2, 3, 4, 5]):
        return [self.run(seed=seed) for seed in seeds]

    def _save_run_check_point(self, seed):
        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir)
        print(f"Saving run checkpoint to '{self.run_save_dir}'.")

        self.run_state = {
            "model": self.model.state_dict(),
            "current_epoch": self.current_epoch,
            "optimizer": self.model_optim.state_dict(),
            "rng_state": torch.get_rng_state(),
            "early_stopping": self.early_stopper.get_state(),
        }
        torch.save(self.run_state, f"{self.run_checkpoint_filepath}")
        print("Run state saved ... ")
