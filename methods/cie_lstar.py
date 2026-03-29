import numpy as np


def srgb_channel_to_linear(channel: np.ndarray) -> np.ndarray:
    threshold = 0.04045

    return np.where(
        channel <= threshold,
        channel / 12.92,
        ((channel + 0.055) / 1.055) ** 2.4
    )


def cie_lstar_grayscale(img_rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale using CIE L*.

    Steps:
    - sRGB → linear RGB
    - compute Y
    - convert Y → L*
    - scale to [0,255]
    """

    # Normalize
    rgb = img_rgb_uint8.astype(np.float32) / 255.0

    # Split channels
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    # Linearize
    r_lin = srgb_channel_to_linear(r)
    g_lin = srgb_channel_to_linear(g)
    b_lin = srgb_channel_to_linear(b)

    # Compute Y
    Y = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

    # Compute L*
    epsilon = 0.008856
    kappa = 903.3

    L_star = np.where(
        Y > epsilon,
        116 * np.cbrt(Y) - 16,
        kappa * Y
    )

    # Scale to [0,255]
    gray_uint8 = np.clip((L_star / 100) * 255.0, 0, 255).astype(np.uint8)

    return gray_uint8