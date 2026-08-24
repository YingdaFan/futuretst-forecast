import torch
from torchmetrics import Metric


class ProbMAE(Metric):
    """MAE of the ensemble-mean prediction."""

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total_mae", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, true: torch.Tensor):
        """
        Args:
            pred: predicted samples, shape (B, O, N, S).
            true: true values, shape (B, O, N).
        """
        pred_mean = pred.mean(dim=-1)  # (B, O, N)
        assert true.shape == pred_mean.shape, "Shapes of true values and pred_mean must match"

        absolute_error = torch.abs(pred_mean - true)
        self.total_mae += absolute_error.sum()
        self.total_samples += absolute_error.numel()

    def compute(self):
        return self.total_mae / self.total_samples
