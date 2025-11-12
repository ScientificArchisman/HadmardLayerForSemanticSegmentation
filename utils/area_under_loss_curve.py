import numpy as np

def aulc(loss_vals: np.ndarray, time_steps: np.ndarray, *, normalize=False) -> float:
    """
    Area under loss curve w.r.t. time (or steps).
    Lower is better for loss; for accuracy use 1-accuracy or flip the sign externally.
    Args:
        loss_vals (np.ndarray): Array of loss values recorded at different time steps.
        time_steps (np.ndarray): Array of time steps corresponding to the loss values.
        normalize (bool): If True, normalize the area by the total time span.
    Returns:
        float: Area under loss curve w.r.t. time (or steps).
    """
    loss = np.asarray(loss_vals, dtype=float)
    t = np.asarray(time_steps, dtype=float)

    if loss.shape != t.shape:
        raise ValueError("loss_vals and time_steps must have the same shape.")
    if loss.size < 2:
        return float(loss[0])

    order = np.argsort(t)
    t, loss = t[order], loss[order]
    t_unique, idx = np.unique(t, return_index=True)
    t, loss = t_unique, loss[idx]

    area = np.trapz(loss, x=t)  

    if normalize:
        span = t[-1] - t[0]
        if span <= 0:
            raise ValueError("time_steps must span a positive range to normalize.")
        area /= span  

    return float(area)
