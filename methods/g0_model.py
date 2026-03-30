import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


class G0Model:
    """
    G0 luminance lookup model for fixed conditions:
    - CIE 1931 2-degree observer
    - D65 illuminant
    - 400-700 nm, 1 nm step
    """

    def __init__(
        self,
        cmf_path="data/reference/CMF2deg.xlsx",
        d65_path="data/reference/D65.xlsx",
        lut_path="data/reference/LUT.xlsx",
        white_y=1.0,
    ):
        self.white_y = float(white_y)

        # Load reference data
        self.cmf = pd.read_excel(cmf_path, header=None).to_numpy(dtype=np.float64)
        self.d65 = pd.read_excel(d65_path, header=None).to_numpy(dtype=np.float64).reshape(-1)
        self.lut = pd.read_excel(lut_path, header=None).to_numpy(dtype=np.float64)

        # Basic checks
        if self.cmf.shape != (301, 3):
            raise ValueError(f"CMF must have shape (301, 3), got {self.cmf.shape}")

        if self.d65.shape[0] != 301:
            raise ValueError(f"D65 must have length 301, got {self.d65.shape[0]}")

        if self.lut.shape[1] != 301:
            raise ValueError(f"LUT must have 301 columns, got {self.lut.shape[1]}")

        # Precompute lookup XYZ and xy
        self.xyz_lut = self.reflectance_to_xyz(self.lut, self.d65, self.cmf, self.white_y)
        self.xyy_lut = self.xyz_to_xyy(self.xyz_lut)

        # Build nearest-neighbor search using xy only
        self.tree = cKDTree(self.xyy_lut[:, :2])

    @staticmethod
    def reflectance_to_xyz(reflectance, illuminant, observer, white_y=1.0):
        """
        Convert reflectance spectra to XYZ.
        """
        ybar = observer[:, 1]

        # Normalization constant
        k = white_y / np.sum(illuminant * ybar)

        reflected_spd = reflectance * illuminant[None, :]
        xyz = k * (reflected_spd @ observer)

        return xyz

    @staticmethod
    def xyz_to_xyy(xyz):
        """
        Convert XYZ to xyY.
        """
        X = xyz[:, 0]
        Y = xyz[:, 1]
        Z = xyz[:, 2]

        denom = X + Y + Z

        x = np.divide(X, denom, out=np.zeros_like(X), where=denom != 0)
        y = np.divide(Y, denom, out=np.zeros_like(Y), where=denom != 0)

        return np.column_stack([x, y, Y])

    def predict_yg0(self, xy_input):
        """
        Find nearest xy in the LUT and return its YG0.
        """
        xy_input = np.asarray(xy_input, dtype=np.float64)

        if xy_input.ndim != 2:
            raise ValueError("xy_input must be a 2D array")

        if xy_input.shape[1] != 2:
            xy_input = xy_input.T
            if xy_input.shape[1] != 2:
                raise ValueError("xy_input must have shape (m, 2) or (2, m)")

        _, indices = self.tree.query(xy_input, k=1)
        yg0 = self.xyz_lut[indices, 1]

        return yg0