import torch
from torchmetrics import Metric
import CRPS.CRPS as pscore


class CRPS(Metric):
    """Continuous Ranked Probability Score over sampled prediction ensembles."""

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total_crps", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, true: torch.Tensor):
        """
        Args:
            pred: predicted samples, shape (B, O, N, S).
            true: true values, shape (B, O, N).
        """
        pred = pred.view(-1, pred.shape[3])  # (B * O * N, S)
        true = true.view(-1)                 # (B * O * N,)

        pred_np = pred.cpu().numpy()
        true_np = true.cpu().numpy()

        crps_sum = 0.0
        for i in range(len(true_np)):
            res = pscore(pred_np[i], true_np[i]).compute()
            crps_sum += res[0]

        self.total_crps += torch.tensor(crps_sum).to(self.device)
        self.total_samples += pred.size(0)

    def compute(self):
        return self.total_crps / self.total_samples
