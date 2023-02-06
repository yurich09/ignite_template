import ignite.distributed as idist
import torch
from torch.distributed import nn

_EPS = torch.finfo(torch.half).eps


def to_indices(pred: torch.Tensor,
               true: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
    """
    Convert `pred` of logits with shape `(B C *)` to `(B *)` of indices,
    i.e. tensors of long.
    """
    assert pred.shape[0] == true.shape[0]
    assert pred.shape[2:] == true.shape[1:]

    c = pred.shape[1]
    pred = pred.argmax(dim=1)

    return c, pred, true


def to_probs(pred: torch.Tensor,
             true: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
    """
    Convert `pred` of logits with shape `(B C *)` to probs,
    i.e. tensors of float32.
    """
    assert pred.shape[0] == true.shape[0]
    assert pred.shape[2:] == true.shape[1:]
    c = pred.shape[1]

    with torch.autocast('cuda'):  # Softmax is always fp32
        pred = pred.softmax(dim=1)

    assert pred.dtype == torch.float32
    return c, pred, true


def confusion(c: int, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """
    Compute `(C C)` confusion matrix from `(B *)` predicted and
    `(B *)` ground-truth index tensors.
    """
    assert pred.shape == true.shape
    assert pred.dtype == true.dtype == torch.long

    # flatten
    pred = pred.view(-1)  # (b n)
    true = true.view(-1)  # (b n)

    mat = torch.zeros(c, c, dtype=torch.long)
    return mat.index_put_((true, pred), torch.tensor(1), accumulate=True)


def soft_confusion(c: int, pred: torch.Tensor,
                   true: torch.Tensor) -> torch.Tensor:
    """
    Compute differentiable `(C C)` confusion matrix from `(B C *)` probs
    and `(B *)` indices tensors.
    """
    assert pred.shape[0] == true.shape[0]
    assert pred.shape[2:] == true.shape[1:]

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
