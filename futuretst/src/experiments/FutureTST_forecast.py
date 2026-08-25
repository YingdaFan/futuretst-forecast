"""
FutureTST forecasting experiment.

FutureTST is encoder-decoder and explicitly uses the known future exogenous
inputs: X_history and X_future are concatenated along the time axis, while
Y_history is the endogenous input.

Full input construction:
  batch_x:   (B, window,   F_exo+1)  last column is Y_history
  x_future:  (B, pred_len, F_exo)
  ->
  X_hist     = batch_x[:, :, :-1]                                 (B, window, F_exo)
  Y_hist     = batch_x[:, :, -1:]                                 (B, window, 1)
  X_full     = cat([X_hist, x_future], dim=time)                  (B, window+pred, F_exo)
  Y_full     = cat([Y_hist, zeros],    dim=time)                  (B, window+pred, 1)
  full_input = cat([X_full, Y_full],   dim=feat)                  (B, window+pred, F_exo+1)
  pred       = model(full_input)                                  (B, pred, 1)
"""

from dataclasses import dataclass
from typing import Dict
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MetricCollection
import torch.multiprocessing as mp

from src.experiments.prob_forecast import ProbForecastExp, update_metrics
from src.datasets.forecast_dataset import ForecastMeta
from src.dataloader.forecast_loader import ForecastLoader
from src.models.FutureTST import FutureTST
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible


def enable_dropout(model):
    """Enable dropout layers during inference for MC Dropout."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


@dataclass
class FutureTSTParameters:
    d_model: int = 256
    num_heads: int = 8
    num_transformer_layers: int = 2
    mlp_size: int = 128
    patch_size: int = 16
    stride_len: int = 8
    mlp_dropout: float = 0.2
    embedding_dropout: float = 0.1
    num_samples: int = 100
    npz_path: str = '../data_processing/data/prepped.npz'


@dataclass
class FutureTSTForecast(ProbForecastExp, FutureTSTParameters):
    model_type: str = "FutureTST_forecast"
    dataset_type: str = "CAMELS"
    optm_type: str = "AdamW"

    def _init_dataset(self):
        self.dataset = ForecastMeta(npz_path=self.npz_path)

    def _init_data_loader(self, shuffle=True, fast_test=True, fast_val=True):
        self._init_dataset()
        from torch_timeseries.scaler import StandardScaler
        self.scaler = StandardScaler()
        self.dataloader = ForecastLoader(
            dataset=self.dataset,
            scaler=self.scaler,
            window=self.windows,
            horizon=self.horizon,
            steps=self.pred_len,
            shuffle_train=shuffle,
            freq=self.dataset.freq,
            batch_size=None,
            num_worker=self.num_worker,
            fast_test=fast_test,
            fast_val=fast_val,
            npz_path=self.npz_path,
            masked_basins=None,
        )
        self.train_loader = self.dataloader.train_loader
        self.val_loader = self.dataloader.val_loader
        self.test_loader = self.dataloader.test_loader

    def _init_model(self):
        # input_channels = batch_x feature dim; ForecastMeta.num_features
        # already accounts for the optional y_imputed column
        N = self.dataset.num_features
        self.model = FutureTST(
            context_window_size=self.windows,
            patch_size=self.patch_size,
            stride_len=self.stride_len,
            d_model=self.d_model,
            num_transformer_layers=self.num_transformer_layers,
            mlp_size=self.mlp_size,
            num_heads=self.num_heads,
            mlp_dropout=self.mlp_dropout,
            pred_size=self.pred_len,
            embedding_dropout=self.embedding_dropout,
            input_channels=N,
        ).to(self.device)

    def _init_optimizer(self):
        # AdamW with lr=3e-4 and cosine decay to lr/10; overrides the --lr CLI flag
        self.model_optim = AdamW(
            self.model.parameters(),
            lr=3e-4,
            weight_decay=self.l2_weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.model_optim,
            T_max=self.epochs,
            eta_min=3e-5,
        )

    @staticmethod
    def _build_full_input(batch_x: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        """
        Build the FutureTST input from batch_x / x_future.

        batch_x  : (B, window,   C)     last column = Y_history
        x_future : (B, pred_len, C-1)
        returns:
            full_input : (B, window+pred_len, C)
                         last channel:   [Y_history, 0...0]
                         other channels: [exogenous history, exogenous future]
        """
        assert batch_x.dim() == 3 and x_future.dim() == 3, \
            f"expected 3-D tensors, got {batch_x.shape}, {x_future.shape}"
        assert batch_x.shape[-1] - 1 == x_future.shape[-1], \
            (f"feature mismatch: batch_x has {batch_x.shape[-1]} cols (expect F_exo+1), "
             f"x_future has {x_future.shape[-1]} cols (expect F_exo)")

        X_hist = batch_x[:, :, :-1]                                          # (B, window, C-1)
        Y_hist = batch_x[:, :, -1:]                                          # (B, window, 1)

        X_full = torch.cat([X_hist, x_future], dim=1)                        # (B, T, C-1)
        Y_pad = torch.zeros(
            X_full.shape[0], x_future.shape[1], 1,
            device=X_full.device, dtype=X_full.dtype,
        )
        Y_full = torch.cat([Y_hist, Y_pad], dim=1)                            # (B, T, 1)
        return torch.cat([X_full, Y_full], dim=2)                             # (B, T, C)

    def _process_train_batch(self, batch_x, batch_y, x_future):
        full_input = self._build_full_input(batch_x, x_future)               # (B, T, C)
        pred = self.model(full_input)                                         # (B, pred_len, 1)
        return F.mse_loss(pred, batch_y)

    def _process_val_batch(self, batch_x, batch_y, x_future, batch_x_date_enc, batch_y_date_enc):
        """
        MC Dropout sampling to estimate the predictive distribution.
        returns: outs (B, pred_len, 1, S), batch_y_target (B, pred_len, 1)
        """
        self.model.eval()
        enable_dropout(self.model)

        full_input = self._build_full_input(batch_x, x_future)
        samples = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                samples.append(self.model(full_input))                        # (B, pred_len, 1)
        outs = torch.stack(samples, dim=-1)                                   # (B, pred_len, 1, S)
        batch_y_target = batch_y[:, -self.pred_len:, :].to(self.device)
        return outs, batch_y_target

    def _train(self):
        self.model.train()
        with torch.enable_grad():
            train_loss = []
            for i, (
                batch_x, batch_y, x_future, origin_x, origin_y,
                batch_x_date_enc, batch_y_date_enc, is_masked,
            ) in enumerate(self.train_loader):
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                x_future = x_future.to(self.device).float()

                # Drop basins with any NaN in this window; a single NaN basin
                # would poison the whole batch's gradient through the MSE.
                nan_mask = (torch.isnan(batch_x).any(dim=(1, 2)) |
                            torch.isnan(batch_y).any(dim=(1, 2)) |
                            torch.isnan(x_future).any(dim=(1, 2)))
                if nan_mask.all():
                    continue
                if nan_mask.any():
                    valid = ~nan_mask
                    batch_x = batch_x[valid]
                    batch_y = batch_y[valid]
                    x_future = x_future[valid]

                self.model_optim.zero_grad()
                loss = self._process_train_batch(batch_x, batch_y, x_future)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                train_loss.append(loss.item())
                self.model_optim.step()

        self.model.eval()
        return train_loss

    def run(self, seed=42) -> Dict[str, float]:
        self._setup_run(seed)
        self._check_run_exist(seed)
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

            reproducible(seed + self.current_epoch)
            train_losses = self._train()
            self._run_print(
                "Epoch: {} cost time: {}s".format(
                    self.current_epoch + 1, time.time() - epoch_start_time
                )
            )
            self._run_print(f"Training loss : {np.mean(train_losses)}")

            self.current_epoch += 1
            val_loss = np.mean(train_losses)
            self.early_stopper(val_loss, model=self.model)
            self._save_run_check_point(seed)
            self.scheduler.step()

        self._load_best_model()
        best_test_result = self._test_and_save()
        return best_test_result

    def _test_and_save(self):
        print("Testing and saving predictions...")
        self.model.eval()
        self.metrics.reset()

        all_preds, all_truths, metric_results = [], [], []
        with torch.no_grad():
            for batch_x, batch_y, x_future, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc, *rest in self.test_loader:
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                x_future = x_future.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                # Filter NaN basins, run the model on the valid ones, then
                # re-insert NaN placeholders so tst.npy keeps basin-aligned
                # ordering for postprocessing.
                nan_mask = (torch.isnan(batch_x).any(dim=(1, 2)) |
                            torch.isnan(batch_y).any(dim=(1, 2)) |
                            torch.isnan(x_future).any(dim=(1, 2)))
                full_size = batch_x.shape[0]
                if nan_mask.all():
                    nan_pred = np.full((full_size, self.pred_len, 1), np.nan, dtype=np.float32)
                    all_preds.append(nan_pred)
                    all_truths.append(batch_y.cpu().numpy())
                    continue
                has_nan = nan_mask.any()
                if has_nan:
                    valid = ~nan_mask
                    batch_x = batch_x[valid]
                    batch_y = batch_y[valid]
                    x_future = x_future[valid]
                    batch_x_date_enc = batch_x_date_enc[valid]
                    batch_y_date_enc = batch_y_date_enc[valid]

                preds, truths = self._process_val_batch(
                    batch_x, batch_y, x_future, batch_x_date_enc, batch_y_date_enc
                )
                metric_results.append(self.task_pool.apply_async(
                    update_metrics,
                    (preds.contiguous().cpu().detach(),
                     truths.contiguous().cpu().detach(),
                     self.metrics)
                ))
                pred_mean = preds.mean(dim=-1).cpu().numpy()                   # (valid, pred_len, 1)
                truths_out = truths.cpu().numpy()

                if has_nan:
                    full_pred = np.full((full_size, self.pred_len, 1), np.nan, dtype=np.float32)
                    full_truth = np.full((full_size, self.pred_len, 1), np.nan, dtype=np.float32)
                    valid_np = valid.cpu().numpy()
                    full_pred[valid_np] = pred_mean
                    full_truth[valid_np] = truths_out
                    pred_mean = full_pred
                    truths_out = full_truth

                all_preds.append(pred_mean)
                all_truths.append(truths_out)

        for r in metric_results:
            r.get()
        test_result = {name: float(metric.compute()) for name, metric in self.metrics.items()}
        self._run_print(f"test_results: {test_result}")

        # Write predictions to <project_root>/output/pred/tst.npy
        # (meta.json is written by ForecastLoader)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        pred_dir = os.path.join(project_root, 'output', 'pred')
        os.makedirs(pred_dir, exist_ok=True)
        all_preds = np.concatenate(all_preds, axis=0)
        pred_path = os.path.join(pred_dir, 'tst.npy')
        np.save(pred_path, all_preds)
        print(f"Predictions saved: {pred_path}, shape: {all_preds.shape}")
        return test_result


if __name__ == "__main__":
    import fire
    fire.Fire(FutureTSTForecast)
