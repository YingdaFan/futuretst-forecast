import torch
from torchmetrics import Metric
import numpy as np


class QICE(Metric):
    """
    Quantile Interval Coverage Error: mean absolute deviation between the
    empirical coverage of each predicted quantile bin and the ideal 1/n_bins.
    """

    def __init__(self, n_bins: int = 10, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.n_bins = n_bins
        self.add_state("quantile_bin_counts", default=torch.zeros(self.n_bins), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds: predicted samples, shape (B, O, N, S).
            targets: true values, shape (B, O, N).
        """
        preds = preds.view(-1, preds.size(3))  # (B * O * N, S)
        targets = targets.view(-1)             # (B * O * N,)

        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy().T

        quantile_list = np.arange(self.n_bins + 1) * (100 / self.n_bins)
        y_pred_quantiles = np.percentile(preds_np, q=quantile_list, axis=1)  # (n_bins+1, N)

        # Which quantile interval each true value falls into
        quantile_membership_array = ((targets_np - y_pred_quantiles) > 0).astype(int)
        y_true_quantile_membership = quantile_membership_array.sum(axis=0)   # (N,)

        y_true_quantile_bin_count = np.array(
            [(y_true_quantile_membership == v).sum() for v in np.arange(self.n_bins + 2)]
        )
        # Fold outliers into the first and last bins
        y_true_quantile_bin_count[1] += y_true_quantile_bin_count[0]
        y_true_quantile_bin_count[-2] += y_true_quantile_bin_count[-1]
        y_true_quantile_bin_count_ = y_true_quantile_bin_count[1:-1]

        self.quantile_bin_counts += torch.tensor(y_true_quantile_bin_count_).to(self.device)
        self.total_samples += preds.size(0)

    def compute(self):
        y_true_ratio_by_bin = self.quantile_bin_counts.float() / self.total_samples.item()
        assert torch.abs(
            torch.sum(y_true_ratio_by_bin) - 1) < 1e-5, "Sum of quantile coverage ratios shall be 1!"
        return torch.abs(torch.ones(self.n_bins) / self.n_bins - y_true_ratio_by_bin).mean()
