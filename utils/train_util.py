import torch
from accelerate.optimizer import AcceleratedOptimizer


def infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def compute_grad_norm(model):
    grads = [p.grad.detach() for p in model.parameters() if p.grad is not None and p.requires_grad]
    if len(grads) == 0:
        total_norm = 0.0
    else:
        norms = [torch.linalg.vector_norm(g, 2.0) for g in grads]
        total_norm = torch.linalg.vector_norm(torch.stack(norms), 2.0).item()
    return total_norm


def compute_step_and_weight_norm(optimizer, eps: float = 1e-10):
    step_norm_sq = 0.0
    weight_norm_sq = 0.0

    if isinstance(optimizer, AcceleratedOptimizer):
        unwrapped_optimizer = optimizer.optimizer
    else:
        unwrapped_optimizer = optimizer

    for group in optimizer.param_groups:
        lr = group["lr"]
        beta1, beta2 = group.get("betas", (0.9, 0.999))  # Adam defaults
        eps_opt = group.get("eps", 1e-8)  # for Adam-type

        for p in group["params"]:
            if p.grad is None:
                continue

            # ‖∇θ‖²
            g = p.grad.detach()

            # --- approximate update Δθ --------------------------------------
            if isinstance(unwrapped_optimizer, torch.optim.SGD):
                dstep = -lr * g  # momentum not included; add if you use it
            elif isinstance(unwrapped_optimizer, (torch.optim.Adam, torch.optim.AdamW)):
                state = optimizer.state[p]
                # First and second moment estimates up to this point
                m = state.get("exp_avg", torch.zeros_like(p))
                v = state.get("exp_avg_sq", torch.zeros_like(p))
                denom = (v.sqrt() / (1 - beta2) ** 0.5) + eps_opt
                dstep = -lr * (m / (1 - beta1) + g) / denom  # 1-step look-ahead
            else:
                # Fall back to first-order estimate
                dstep = -lr * g

            step_norm_sq += torch.sum(dstep**2).item()
            weight_norm_sq += torch.sum(p.data**2).item()

    step_norm = step_norm_sq**0.5
    weight_norm = max(weight_norm_sq**0.5, eps)  # avoid /0

    return step_norm, weight_norm
