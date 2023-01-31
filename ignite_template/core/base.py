import ignite.distributed as idist
import torch
from torch.distributed import nn

_EPS = torch.finfo(torch.half).eps


def to_index(pred: torch.Tensor,
             true: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
    """
    Convert `pred` of logits with shape [B, C, ...] to [B, ...] of indices,
    i.e. tensors of long.
    """
    assert pred.shape[0] == true.shape[0]
    assert pred.shape[2:] == true.shape[1:]

    c = pred.shape[1]
    pred = pred.argmax(dim=1)

    return c, pred, true


def to_prob(pred: torch.Tensor,
            true: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
    """
    Convert `pred` of logits with shape [B, C, ...] to probs,
    i.e. tensors of float.
    """
    assert pred.shape[0] == true.shape[0]
    assert pred.shape[2:] == true.shape[1:]

    c = pred.shape[1]
    pred = pred.softmax(dim=1)

    return c, pred, true


def confusion_mat(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    c, pred, true = to_index(pred, true)

    # flatten
    pred = pred.view(-1)  # (b n)
    true = true.view(-1)  # (b n)

    mat = torch.zeros(c, c, dtype=torch.long)
    return mat.index_put_((true, pred), torch.tensor(1), accumulate=True)


def confusion_mat_grad(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    c, pred, true = to_prob(pred, true)

    # flatten
    b = true.shape[0]
    pred = pred.view(b, c, -1).permute(0, 2, 1).reshape(-1, c)  # (b n) c
    true = true.view(-1)  # (b n)

    mat = pred.new_zeros(c, c).index_add(0, true, pred)

    if idist.get_world_size() > 1:
        mat = nn.all_reduce(mat)

    return mat


def dice(mat: torch.Tensor) -> torch.Tensor:
    return 2 * mat.diag() / (mat.sum(0) + mat.sum(1)).clamp(_EPS)
