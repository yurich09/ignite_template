import torch


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
