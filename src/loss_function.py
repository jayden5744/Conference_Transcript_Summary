import torch.nn.functional as F


def label_smoothed_nll_loss(probs, target, epsilon=0.1, ignore_index=0, reduce=True):
    if target.dim() == probs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -probs.gather(dim=-1, index=target)
    smooth_loss = -probs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()
    eps_i = epsilon / (probs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss


def compute_kl_loss(p, q, pad_mask=None, use_function: str = "mean"):

    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level task
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function 'sum' and 'mean' depending on your task
    if use_function == "mean":
        p_loss = p_loss.mean()
        q_loss = q_loss.mean()

    elif use_function == "sum":
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

    else:
        raise ValueError(f"param `use_function` must be one of [mean, sum], but your param is {use_function} ")

    loss = (p_loss + q_loss) / 2
    return loss
