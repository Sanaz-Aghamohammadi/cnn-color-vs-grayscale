import numpy as np


def luma_grayscale(img_rgb: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image array to grayscale using the luma formula.

    Parameters
    ----------
    img_rgb : np.ndarray
        RGB image array of shape (H, W, 3).

    Returns
    -------
    np.ndarray
        Grayscale image array of shape (H, W), dtype uint8.
    """
    r = img_rgb[:, :, 0]
    g = img_rgb[:, :, 1]
    b = img_rgb[:, :, 2]

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray, 0, 255).astype(np.uint8)