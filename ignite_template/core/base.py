import torch

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


def dice(pred: torch.Tensor,
         true: torch.Tensor,
         macro: bool = True) -> torch.Tensor:
    # TODO: Add docs
    c, pred, true = to_index(pred, true)

    def _dice(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        true = true.view(-1)
        pred = pred.view(-1)
        tp, t, p = (x.bincount(minlength=c).clamp_(1).double()
                    for x in (true[true == pred], true, pred))
        return 2 * tp / (t + p)

    if macro:
        return _dice(pred, true)

    b = pred.shape[0]
    *scores, = map(_dice, pred.view(b, -1).unbind(), true.view(b, -1).unbind())
    return torch.mean(torch.stack(scores), dim=0)


def confusion_mat(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    c, pred, true = to_prob(pred, true)

    # flatten
    b = true.shape[0]
    pred = pred.view(b, c, -1).permute(0, 2, 1).reshape(-1, c)  # (b n) c
    true = true.view(-1)  # (b n)

    mat = torch.zeros(c, c, device=pred.device, dtype=pred.dtype).index_add(0, true, pred)
    return mat.double() / mat.sum()


def dice_grad(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    mat = confusion_mat(pred, true)
    return 2 * mat.diag() / (mat.sum(0) + mat.sum(1)).clamp(_EPS)
