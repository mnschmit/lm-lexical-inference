from pytorch_lightning.metrics.functional import auc
import torch


def compute_auc(precisions: torch.FloatTensor, recalls: torch.FloatTensor,
                filter_threshold: float = 0.5) -> torch.FloatTensor:
    xs, ys = [], []
    for p, r in zip(precisions, recalls):
        if p >= filter_threshold:
            xs.append(r)
            ys.append(p)

    return auc(
        torch.cat([x.unsqueeze(0) for x in xs], 0),
        torch.cat([y.unsqueeze(0) for y in ys], 0)
    )
