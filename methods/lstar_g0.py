import numpy as np


def srgb_channel_to_linear(channel: np.ndarray) -> np.ndarray:
    """
    Convert one sRGB channel from [0,1] to linear RGB.
    """
    threshold = 0.04045

    return np.where(
        channel <= threshold,
        channel / 12.92,
        ((channel + 0.055) / 1.055) ** 2.4
    )


def linear_rgb_to_xyz(img_rgb_linear: np.ndarray) -> np.ndarray:
    """
    Convert linear RGB image to XYZ using the sRGB / D65 matrix.

    Parameters
    ----------
    img_rgb_linear : np.ndarray
        Shape (H, W, 3), values in [0,1]

    Returns
    -------
    np.ndarray
        Shape (H, W, 3), channels are X, Y, Z
    """
    r = img_rgb_linear[:, :, 0]
    g = img_rgb_linear[:, :, 1]
    b = img_rgb_linear[:, :, 2]

    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    return np.stack([X, Y, Z], axis=2)


def xyz_to_xy(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Convert XYZ channels to xy chromaticity.

    Returns
    -------
    np.ndarray
        Shape (H*W, 2)
    """
    denom = X + Y + Z

    x = np.divide(X, denom, out=np.zeros_like(X), where=denom != 0)
    y = np.divide(Y, denom, out=np.zeros_like(Y), where=denom != 0)

    xy = np.column_stack([x.ravel(), y.ravel()])
    return xy


def lstar_g0_grayscale(img_rgb_uint8: np.ndarray, g0_model) -> np.ndarray:
    """
    Convert an RGB image to grayscale using the L*G0 method.

    Parameters
    ----------
    img_rgb_uint8 : np.ndarray
        RGB image of shape (H, W, 3), dtype uint8
    g0_model : G0Model
        Precomputed G0 luminance lookup model

    Returns
    -------
    np.ndarray
        Grayscale image of shape (H, W), dtype uint8
    """
    # Step 1: normalize sRGB to [0,1]
    rgb = img_rgb_uint8.astype(np.float32) / 255.0

    # Step 2: linearize each channel
    r_lin = srgb_channel_to_linear(rgb[:, :, 0])
    g_lin = srgb_channel_to_linear(rgb[:, :, 1])
    b_lin = srgb_channel_to_linear(rgb[:, :, 2])

    rgb_linear = np.stack([r_lin, g_lin, b_lin], axis=2)

    # Step 3: convert linear RGB to XYZ
    xyz = linear_rgb_to_xyz(rgb_linear)
    X = xyz[:, :, 0]
    Y = xyz[:, :, 1]
    Z = xyz[:, :, 2]

    # Step 4: compute xy for each pixel
    xy_pixels = xyz_to_xy(X, Y, Z)

    # Step 5: lookup YG0 for each pixel chromaticity
    yg0 = g0_model.predict_yg0(xy_pixels)

    # Reshape back to image shape
    yg0 = yg0.reshape(Y.shape)

    # Step 6: compute relative luminance with respect to G0
    # Avoid divide-by-zero just in case
    ratio = np.divide(Y, yg0, out=np.zeros_like(Y), where=yg0 > 0)

    epsilon = 0.008856
    kappa = 903.3

    lstar_g0 = np.where(
        ratio > epsilon,
        116 * np.cbrt(ratio) - 16,
        kappa * ratio
    )

    # Optional: clip to valid lightness range for image storage
    lstar_g0 = np.clip(lstar_g0, 0, 100)

    # Step 7: scale [0,100] to [0,255]
    gray_uint8 = np.clip((lstar_g0 / 100.0) * 255.0, 0, 255).astype(np.uint8)

    return gray_uint8