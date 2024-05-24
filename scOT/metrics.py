import numpy as np


def lp_error(preds: np.ndarray, targets: np.ndarray, p=1):
    num_samples, num_channels, _, _ = preds.shape
    preds = preds.reshape(num_samples, num_channels, -1)
    targets = targets.reshape(num_samples, num_channels, -1)
    errors = np.sum(np.abs(preds - targets) ** p, axis=-1)
    return np.sum(errors, axis=-1) ** (1 / p)


def relative_lp_error(
    preds: np.ndarray,
    targets: np.ndarray,
    p=1,
    return_percent=True,
):
    num_samples, num_channels, _, _ = preds.shape
    preds = preds.reshape(num_samples, num_channels, -1)
    targets = targets.reshape(num_samples, num_channels, -1)
    errors = np.sum(np.abs(preds - targets) ** p, axis=-1)
    normalization_factor = np.sum(np.abs(targets) ** p, axis=-1)

    # catch 0 division
    normalization_factor = np.sum(normalization_factor, axis=-1)
    normalization_factor = np.where(
        normalization_factor == 0, 1e-10, normalization_factor
    )

    errors = (np.sum(errors, axis=-1) / normalization_factor) ** (1 / p)

    if return_percent:
        errors *= 100

    return errors


def mean_relative_lp_error(
    preds: np.ndarray,
    targets: np.ndarray,
    p=1,
    return_percent=True,
):
    errors = relative_lp_error(preds, targets, p, return_percent)
    return np.mean(errors, axis=0)


def median_relative_lp_error(
    preds: np.ndarray,
    targets: np.ndarray,
    p=1,
    return_percent=True,
):
    errors = relative_lp_error(preds, targets, p, return_percent)
    return np.median(errors, axis=0)
