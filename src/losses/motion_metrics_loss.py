import torch
import torch.nn as nn


class MotionMetricsLoss(nn.Module):
    """
    Not needed yet. May got removed at the end.
    """
    def __init__(self):
        raise NotImplementedError

    def forward(self, train_metric_values, metric_names):
        loss = 0
        # for i, m in enumerate(
        #         ['min_ade', 'min_fde', 'miss_rate', 'overlap_rate', 'map']):
        #     for j, n in enumerate(metric_names):
        #         print('{}/{}: {}'.format(m, n, train_metric_values[i, j]))

        for j, n in enumerate(metric_names):
            loss += train_metric_values[0, j]

        return loss