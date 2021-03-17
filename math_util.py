"""
Mathematical utility functions for fully Ambisonic domain SELD estimation tasks
2019/12, Ryotaro SATO
"""
from logging import getLogger
from typing import Union
import numpy as np
import torch

logger = getLogger(__name__)


def WXYZ2Sph(signal: np.ndarray) -> np.ndarray:
    """
    FOA (W, X, Y, Z) -> spherical harmonics (l, m) = (0, 0), (1, -1), (1, 0), (1, 1)
    :param signal: shape [4, TIME, FREQ, Re/Im]
    :return: shape [4, TIME, FREQ, Re/Im]
    """
    assert len(signal.shape) == 4
    assert signal.shape[0] == 4
    assert signal.shape[3] == 2
    ret = signal.copy()
    # (1, -1)
    ret[1, :, :, 0] = (signal[1, :, :, 1] + signal[3, :, :, 0]) / np.sqrt(2.0)
    ret[1, :, :, 1] = (-signal[1, :, :, 0] + signal[3, :, :, 1]) / np.sqrt(2.0)
    # (1, 0)
    ret[2, :, :, :] = signal[2, :, :, :]
    # (1, 1)
    ret[3, :, :, 0] = (signal[1, :, :, 1] - signal[3, :, :, 0]) / np.sqrt(2.0)
    ret[3, :, :, 1] = (-signal[1, :, :, 0] - signal[3, :, :, 1]) / np.sqrt(2.0)
    return ret


def rotate_vec(v: np.ndarray, rotate_info: list) -> np.ndarray:
    """
    Apply ZXZ rotation to 3-d vector `v`
    :param v: shape [3, X, Y] (X, Y: arbitrary)
    :param rotate_info: (a, b, c), similar to Euler's angle
    :return: shape [3, X, Y]
    """
    assert v.shape[0] == 3
    a, b, c = rotate_info
    v[1], v[0] = v[1] * np.cos(a) + v[0] * np.sin(a), -v[1] * np.sin(a) + v[0] * np.cos(a)
    v[2], v[1] = v[2] * np.cos(b) + v[1] * np.sin(b), -v[2] * np.sin(b) + v[1] * np.cos(b)
    v[1], v[0] = v[1] * np.cos(c) + v[0] * np.sin(c), -v[1] * np.sin(c) + v[0] * np.cos(c)
    return v


def rotate_foa(foa: np.ndarray, rotate_info: list) -> np.ndarray:
    """
    Apply ZXZ rotation to FOA signals `foa`
    :param foa: shape [4, X, Y] (X, Y: arbitrary) foa[0]: W, foa[1]: X, foa[2]: Y, foa[3]: Z
    :param rotate_info: (a, b, c), similar to Euler's angle
    :return: shape [4, X, Y]
    """
    assert foa.shape[0] == 4
    a, b, c = rotate_info
    foa[1], foa[3] = foa[1] * np.cos(a) + foa[3] * np.sin(a), -foa[1] * np.sin(a) + foa[3] * np.cos(a)
    foa[2], foa[1] = foa[2] * np.cos(b) + foa[1] * np.sin(b), -foa[2] * np.sin(b) + foa[1] * np.cos(b)
    foa[1], foa[3] = foa[1] * np.cos(c) + foa[3] * np.sin(c), -foa[1] * np.sin(c) + foa[3] * np.cos(c)
    return foa


def unit_vec_distance(ea, eb, labelonoff) -> torch.Tensor:
    """
    Great-circular distance between two points on unit sphere
    :param ea: shape [(arbitrary), 3]
    :param eb: shape [(arbitrary), 3]
    :param labelonoff: shape [(arbitrary)] , whether each data is used for calculation or not
    :return: float
    """
    assert ea.shape == eb.shape and ea.shape[-1] == 3 and ea.shape[:-1] == labelonoff.shape
    inner_prod = torch.sum(ea * eb, dim=-1)
    theta = torch.acos(inner_prod * 0.999) * labelonoff
    return torch.sum(theta) / torch.sum(labelonoff)


def unitvec2azimuthelevation(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert 3-D unit vectors to (azimuth, elevation) values
    :param x: shape [BATCH_SIZE, TIME, CLASS, 3]
    :return: shape [BATCH_SIZE, TIME, CLASS * 2]
    """
    assert len(x.shape) == 4 and x.shape[-1] == 3
    if type(x) == np.ndarray:
        azim = np.arctan2(x[:, :, :, 1], x[:, :, :, 0])
        elev = np.arctan(x[:, :, :, 2])
        return np.concatenate((azim, elev), -1)
    elif type(x) == torch.Tensor:
        azim = torch.atan2(x[:, :, :, 1], x[:, :, :, 0])
        elev = torch.atan(x[:, :, :, 2])
        return torch.cat((azim, elev), -1)


def phase_alignment(x: np.ndarray, tdiff) -> np.ndarray:
    """
    For time-frequency domain ambisonic signals x, divide $x_{t, f}$ by the
    phase component of Wch of $x_{t, f}$ for each time frame t, each frequency bin f.
    This operation makes the x time-translation invariant while maintaining rotation equivariance.
    :param x: [CH, TIME, FREQ, Re/Im]
    :return: shape [CH, TIME, FREQ, Re/Im]
    """
    assert len(x.shape) == 4 and x.shape[0] == 4 and x.shape[-1] == 2
    if tdiff is None:
        return x
    x_complex = x[:, :, :, 0] + x[:, :, :, 1] * 1j
    l0elem_norm = np.expand_dims(x_complex[0] / np.abs(x_complex[0]), axis=0)

    ret = x_complex[:, tdiff:]
    ret /= l0elem_norm[:, :l0elem_norm.shape[1] - tdiff]
    ret = np.stack([ret.real, ret.imag], axis=-1)
    return ret


def decide_rotation(azi_rad: np.ndarray, class_: np.ndarray, width: int):
    """
    Find the angle such that the rotation by this azimuth angle moves the most sound
    events to the specific direction range of `width` degrees width.
    """
    assert width >= 0 and width < 360
    cnt = [0 for i in range(360 * 3)]
    for az, val in zip(*np.unique(azi_rad * class_, return_counts=True)):
        az_deg = int(round(az * 180 / np.pi))
        if az_deg < 0:
            az_deg += 360
        cnt[az_deg] += val
        cnt[az_deg + width] -= val
        cnt[az_deg + 360] += val
        cnt[az_deg + 360 + width] -= val
    cnt_zero = np.size(class_) - np.count_nonzero(class_)
    cnt[0] -= cnt_zero
    cnt[width] += cnt_zero
    cnt[360] -= cnt_zero
    cnt[360 + width] += cnt_zero
    for i in range(len(cnt) - 1):
        cnt[i + 1] += cnt[i]
    rotate = [-np.argmax(cnt) * np.pi / 180, 0, 0]

    return rotate
