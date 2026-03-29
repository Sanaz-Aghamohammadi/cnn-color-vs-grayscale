import numpy as np


def srgb_channel_to_linear(channel: np.ndarray) -> np.ndarray:
    """
    Convert one sRGB channel from [0, 1] to linear RGB.

    Parameters
    ----------
    channel : np.ndarray
        One sRGB channel in [0, 1].

    Returns
    -------
    np.ndarray
        Linearized channel in [0, 1].
    """
    threshold = 0.04045

    return np.where(
        channel <= threshold,
        channel / 12.92,
        ((channel + 0.055) / 1.055) ** 2.4
    )


def cie_y_grayscale(img_rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Convert an 8-bit RGB image to grayscale using linear CIE Y.

    Steps
    -----
    1. Normalize RGB from [0, 255] to [0, 1]
    2. Linearize each sRGB channel separately
    3. Compute linear Y:
           Y = 0.2126 R_lin + 0.7152 G_lin + 0.0722 B_lin
    4. Scale Y to [0, 255] and save as uint8 WITHOUT converting back to sRGB

    Parameters
    ----------
    img_rgb_uint8 : np.ndarray
        RGB image array of shape (H, W, 3), dtype uint8.

    Returns
    -------
    np.ndarray
        Grayscale image array of shape (H, W), dtype uint8.
    """
    # Normalize to [0, 1]
    rgb = img_rgb_uint8.astype(np.float32) / 255.0

    # Separate normalized sRGB channels
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    # Linearize each channel separately
    r_lin = srgb_channel_to_linear(r)
    g_lin = srgb_channel_to_linear(g)
    b_lin = srgb_channel_to_linear(b)

    # Compute linear CIE Y
    y_linear = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

    # Convert linear Y directly to uint8 without re-encoding to sRGB
    gray_uint8 = np.clip(y_linear * 255.0, 0, 255).astype(np.uint8)

    return gray_uint8