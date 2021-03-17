"""
Module functions for the proposed DNN implementation
"""

from logging import getLogger
from numpy import trim_zeros
from typing import List, Tuple, Union
import torch
import torch.nn as nn
from fCGModule import fCGModule

logger = getLogger(__name__)


def tau_decompose(taus: List[int]) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    From the list of tau_l's, calculate beggining and ending indices of each
    feature vector.

    Parameters
    ----------
    taus : 1-D array of integers
        [tau_0, tau_1, ..., tau_L]

    Returns
    -------
    mlos / mhis : list of integers
        [description]
    lbegin / lend : list of integers
        The result shows that the feature vectors with dim=l is in the range
        of [lbegin[l], lend[l]) for each valid l.

    Examples
    --------
    >>> tau = [1, 2]
    >>> tau_decompose(tau)
    ([0, 1, 4], [1, 4, 7], [0, 1], [1, 7])

    tau = [1, 2] means that there exists one l=0 feature vector and two l=1
    feature vectors. The total dimension is 1 * 0 + 2 * 3 = 7.
    mlos and mhis show that the l=0 vector is in [0, 1), and two l=1 vectors
    belong to [1, 4) and [4, 7), respectively.

    """
    mlos = []
    mhis = []
    l_begin = []
    l_end = []
    for l, t in enumerate(taus):
        l_begin.append(0 if not mhis else mhis[-1])
        for _ in range(t):
            mlos.append(0 if not mhis else mhis[-1])
            mhis.append(mlos[-1] + 2 * l + 1)
        l_end.append(0 if not mhis else mhis[-1])
    return mlos, mhis, l_begin, l_end


class SphHarmDistributedComplexConv2d(nn.Module):
    """
    Time-Frequency convoluton layer (Section IV.C)
    based on CompexConv2d
    """

    def __init__(self, taus_in: List[int], taus_out: List[int],
                 kernel_size: Union[Tuple[int], int],
                 padding: Union[Tuple[int], int],
                 skip_l0: bool = False):
        """
        taus_in: tau = [tau_0, tau_1, ..., tau_Lmax] of input variable.
        taus_out: tau of output variable.
        kernel_size: (kernel_time, kernel_freq)
        padding: (padding_time, padding_freq)
        skip_l0: boolean. If `true`, no operations done for `l = 0` variables.
        """

        super(SphHarmDistributedComplexConv2d, self).__init__()
        self.taus_in = trim_zeros(taus_in, 'b')
        self.taus_out = trim_zeros(taus_out, 'b')
        self.Lmax = len(self.taus_in) - 1
        self.cconvs = nn.ModuleList([ComplexConv2d(tin, tout, kernel_size, bias=(l == 0), padding=padding) for l, (tin, tout) in enumerate(zip(self.taus_in, self.taus_out))])
        self.skip_l0 = skip_l0

        if self.skip_l0 and self.Lmax >= 0:
            assert self.taus_in[0] >= self.taus_out[0]

        _, _, self.l_begin, self.l_end = tau_decompose(self.taus_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [BATCH, (j, l, m), TIME, FREQ, Re/Im]
        """
        BATCH, _, TIME_IN, FREQ_IN, _ = x.shape
        assert self.l_end[-1] == x.shape[1] and x.shape[-1] == 2 and x.is_contiguous()
        ret = []
        for l, (tin, tout) in enumerate(zip(self.taus_in, self.taus_out)):
            if tin == 0 or tout == 0:
                continue
            x_in = x[:, self.l_begin[l]:self.l_end[l]].reshape(BATCH, tin, 2 * l + 1, TIME_IN, FREQ_IN, 2).permute(0, 2, 1, 3, 4, 5).reshape(BATCH * (2 * l + 1), tin, TIME_IN, FREQ_IN, 2)
            x_out = self.cconvs[l](x_in) if (l > 0 or self.skip_l0 is False) else x_in[:, :tout]
            _, _, TIME_OUT, FREQ_OUT, _ = x_out.shape
            x_out = x_out.reshape(BATCH, 2 * l + 1, tout, TIME_OUT, FREQ_OUT, 2).permute(0, 2, 1, 3, 4, 5).reshape(BATCH, tout * (2 * l + 1), TIME_OUT, FREQ_OUT, 2)
            ret.append(x_out)
        return torch.cat(ret, dim=1)


class SphericalSigmaBN(nn.Module):
    """
    Variance normalization layer (Section IV.D)
    """
    def __init__(self, taus: List[int], momentum: float, eps: float):
        """
        taus: tau of input/output variables.
        """
        super(SphericalSigmaBN, self).__init__()
        self.mlos, self.mhis, _, _ = tau_decompose(taus)
        self.momentum = nn.Parameter(torch.tensor(momentum), requires_grad=False)
        self.sigma2 = nn.Parameter(torch.zeros(len(self.mlos)), requires_grad=False)
        self.eps = nn.Parameter(torch.tensor(eps), requires_grad=False)

    def forward(self, x: torch.Tensor, update: bool, *args, **kwargs) -> torch.Tensor:
        """
        x: [BATCH, (j, l, m), TIME, FREQ, Re/Im]
        """
        BATCH, CH, TIME, FREQ, RIDIM = x.shape
        assert CH == self.mhis[-1] and RIDIM == 2

        if update:
            with torch.no_grad():
                for i, (mlo, mhi) in enumerate(zip(self.mlos, self.mhis)):
                    s2 = torch.mean(torch.sqrt(x[:, mlo:mhi] ** 2))
                    if self.sigma2[i] == 0:
                        self.sigma2[i] = s2
                    else:
                        self.sigma2[i] = self.sigma2[i] * (1.0 - self.momentum) + s2 * self.momentum

        ret = []
        for i, (mlo, mhi) in enumerate(zip(self.mlos, self.mhis)):
            ret.append(x[:, mlo:mhi] / (torch.sqrt(self.sigma2[i]) + self.eps))
        ret = torch.cat(ret, 1)
        assert ret.shape == x.shape
        return ret


class GroupAvgPool1dBCTF2(nn.Module):
    """
    Average pooling layer (Section IV.F)
    """
    def __init__(self, stride: int, taus: List[int]):
        super(GroupAvgPool1dBCTF2, self).__init__()
        self.stride = stride
        self.taus = taus
        self.mlos, self.mhis, _, _ = tau_decompose(taus)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [BATCH, (j, l, m), TIME, FREQ, Re/Im]
        """
        BATCH, CH, TIME, FREQ, RIDIM = x.shape
        assert RIDIM == 2 and CH == self.mhis[-1]
        x = x.reshape(BATCH, CH, TIME, FREQ // self.stride, self.stride, 2)
        x = torch.mean(x, dim=4)
        assert x.shape == torch.Size([BATCH, CH, TIME, FREQ // self.stride, 2])
        return x


class ComplexConv2d(nn.Module):
    """
    torch.nn.Conv2d extended to complex convolution
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int]], bias: bool,
                 padding: Union[int, Tuple[int]]):
        super(ComplexConv2d, self).__init__()
        if in_channels and out_channels:
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.conv_re = nn.Conv2d(in_channels, out_channels, kernel_size,
                                     padding=padding, bias=bias)
            self.conv_im = nn.Conv2d(in_channels, out_channels, kernel_size,
                                     padding=padding, bias=bias)
            self.dummy = False
        else:
            self.dummy = True

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # x: [BATCH, CH, TIME, FREQ, real/imag]
        if self.dummy:
            return None
        assert len(x.shape) == 5 and x.shape[4] == 2
        x = x.contiguous()
        ret_re = self.conv_re(x[:, :, :, :, 0]) - self.conv_im(x[:, :, :, :, 1])
        ret_im = self.conv_im(x[:, :, :, :, 0]) + self.conv_re(x[:, :, :, :, 1])
        ret = torch.stack((ret_re, ret_im), dim=4)
        assert len(ret.shape) == 5 and ret.shape[0] == x.shape[0] and \
               ret.shape[1] == self.out_channels and ret.shape[4] == 2
        return ret


class SphericalHarmonicsDomainActivation(nn.Module):
    """
    Spherical Harmonic Domain Activation
    """
    def __init__(self, taus: List[int], l_use_list: List[int]):
        super(SphericalHarmonicsDomainActivation, self).__init__()
        self.taus = taus
        _, _, self.l_begin, self.l_end = tau_decompose(taus)
        self.linear_layers = nn.ModuleList(
            [nn.Linear(taus[0] * 2, tau) for l, tau in enumerate(self.taus)]
        )
        self.activation_function = nn.Sigmoid()
        self.l_use = [0 for l in range(len(self.taus))]
        for l in l_use_list:
            self.l_use[l] = 1
        assert self.l_use[0] == 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [BATCH, (j, l, m), T, F, 2]
        """
        in_shape = x.shape
        BATCH, CH, T, F, RIDIM = x.shape
        assert RIDIM == 2 and CH == self.l_end[-1]
        assert x.is_contiguous()
        x = x.permute(0, 2, 3, 1, 4).contiguous().reshape(BATCH * T * F, CH, 2)  # BatchTimeFreq, CH, Re/Im

        ret = []  # [[BTF, CH, Re/Im]]
        for l, tau in enumerate(self.taus):
            if l == 0:
                ret.append(x[:, self.l_begin[0]:self.l_end[0]])
                x0 = x[:, self.l_begin[0]:self.l_end[0]].reshape(BATCH * T * F, self.taus[0] * 2)
            elif self.taus[l]:
                xl = x[:, self.l_begin[l]:self.l_end[l]]
                if self.l_use[l]:
                    xl = xl.reshape(BATCH * T * F, self.taus[l], (2 * l + 1) * 2)  # [BTF, tau, (m, Re/Im)]
                    activ = self.linear_layers[l](x0).unsqueeze(-1)  # [BTF, tau, 1]
                    xl = xl * self.activation_function(activ)
                    xl = xl.reshape(BATCH * T * F, self.taus[l] * (2 * l + 1), 2)
                ret.append(xl)

        ret = torch.cat(ret, 1)
        assert ret.shape == torch.Size([BATCH * T * F, CH, 2])
        ret = ret.reshape(BATCH, T, F, CH, 2).permute(0, 3, 1, 2, 4).contiguous()
        assert ret.shape == in_shape
        return ret


class ScaleInvariantSphericalHarmonicsDomainActivation(nn.Module):
    """
    CGD (B) operation in Fig. 1.
    Scale Invariant Spherical Harmonic Domain Activation
    """
    def __init__(self, taus: List[int], l_use_list: List[int], eps: float):
        super(ScaleInvariantSphericalHarmonicsDomainActivation, self).__init__()
        self.taus = trim_zeros(taus, 'b')
        _, _, self.l_begin, self.l_end = tau_decompose(taus)
        self.linear_layers_re = nn.ModuleList([nn.Linear(self.taus[0], tau) for tau in self.taus])
        self.linear_layers_im = nn.ModuleList([nn.Linear(self.taus[0], tau) for tau in self.taus])
        self.l_use = [0 for l in range(len(self.taus))]
        for l in l_use_list:
            self.l_use[l] = 1
        assert self.l_use[0] == 0
        self.eps = nn.Parameter(torch.tensor(eps), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [BATCH, (j, l, m), Time, Freq, Re/Im]
        """
        in_shape = x.shape
        BATCH, CH, T, F, RIDIM = x.shape
        assert RIDIM == 2 and CH == self.l_end[-1]
        x = x.contiguous().permute(0, 2, 3, 1, 4).reshape(BATCH * T * F, CH, 2)  # BTF, CH, 2

        ret = []  # [[BTF, CH, 2]]
        x0 = x[:, self.l_begin[0]:self.l_end[0]]
        rex0 = x0[:, :, 0]
        imx0 = x0[:, :, 1]
        for l, tau in enumerate(self.taus):
            if l == 0:
                ret.append(x0)
            elif self.taus[l]:
                xl = x[:, self.l_begin[l]:self.l_end[l]]
                if self.l_use[l]:
                    xl = xl.reshape(BATCH * T * F, self.taus[l], 2 * l + 1, 2)  # [BTF, tau, m, Re/Im]
                    activ_re = (self.linear_layers_re[l](rex0) - self.linear_layers_im[l](imx0)).unsqueeze(-1)  # [BTF, tau, 1]
                    activ_im = (self.linear_layers_im[l](rex0) + self.linear_layers_re[l](imx0)).unsqueeze(-1)  # [BTF, tau, 1]
                    xl_re = xl[:, :, :, 0] * activ_re - xl[:, :, :, 1] * activ_im
                    xl_im = xl[:, :, :, 1] * activ_re + xl[:, :, :, 0] * activ_im
                    xl = torch.stack([xl_re, xl_im], dim=-1)  # BTF, tau, m, Re/Im
                    norm = torch.sqrt(torch.sum(xl * xl, (2, 3)))
                    xl = xl / (self.eps + torch.sqrt(norm).unsqueeze(-1).unsqueeze(-1))
                    xl = xl.reshape(BATCH * T * F, self.taus[l] * (2 * l + 1), 2)
                ret.append(xl)

        ret = torch.cat(ret, 1)
        assert ret.shape == torch.Size([BATCH * T * F, CH, 2])
        ret = ret.reshape(BATCH, T, F, CH, 2).permute(0, 3, 1, 2, 4).contiguous()
        assert ret.shape == in_shape
        return ret


def gen_noduplicate_tauouts(tau_in: List[int], Lout_max: int) -> Tuple[List[int], List[bool]]:
    """
    Utility function for `CGElementProduct`.

    Plain `fCGModule` outputs a little verbose information, so
    we need to remove unwanted channels for faster comutation time.

    Parameters
    ----------
    tau_in : list
        tau of the data input to the CGD (A) layer.
    Lout_max : int
        Max L of the output of the CGD (A) layer.

    Returns
    -------
    trim_zeros(tau_out, 'b')
        tau of the effective output of the CGD (A) layer.
    mask_eachl : list of boolean
        `False` channels of the output of `fCGModule` are verbose and should be eliminated.
    """
    tau_out = [0] * (Lout_max + 1)
    mask_eachl = [[] for _ in range(Lout_max + 1)]
    for l, T in enumerate(tau_in):
        tau_out[l] += T
        mask_eachl[l] += [True] * T
        if l > 0:
            for t1 in range(T):
                for t2 in range(T):
                    for lout in range(min(l * 2, Lout_max) + 1):
                        flg = t1 < t2 or (t1 == t2 and lout % 2 == 0)
                        mask_eachl[lout] += [flg]
                        tau_out[lout] += flg

    return trim_zeros(tau_out, 'b'), mask_eachl


class CGElementProduct(nn.Module):
    """
    CGD (A) operation in Fig. 1.
    Clebsch-Gordan decomposition-based bilinear calculation. (Section IV.B)
    """
    def __init__(self, tau_in: List[int], Lout_max: int, eps: float,
                 scale_invariance: bool = True):
        """
        Parameters
        ----------
        tau_in : List[int]
            integer list [tau_0, tau_1, ...]
        Lout_max : int
            Max degree of spherical harmonics contained in output.
        eps : float
            Used for normalization.
        scale_invariance : bool, optional
            Impose scale invariance or not. Defaults to True.
        """
        super(CGElementProduct, self).__init__()
        self.tau_in = trim_zeros(tau_in, 'b')
        self.Lout_max = min((len(self.tau_in) - 1) * 2, Lout_max)
        self.tau_out, self.mask_eachl = gen_noduplicate_tauouts(self.tau_in, self.Lout_max)

        self.cglayers = [None] * len(self.tau_in)
        for l, t in enumerate(self.tau_in):
            if l > 0:  # l=0ではCG変換をしない
                tau_in_onehot = [0] * (self.Lout_max + 1)
                tau_in_onehot[l] += t
                self.cglayers[l] = fCGModule(self.Lout_max, tau_in_onehot)
        self.cglayers = nn.ModuleList(self.cglayers)

        _, _, self.l_begin, self.l_end = tau_decompose(self.tau_in)
        _, _, self.l_out_begin, self.l_out_end = tau_decompose(self.tau_out)

        self.scale_invariance = scale_invariance
        self.eps = nn.Parameter(torch.tensor(eps), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BATCH, CH_IN, T, F, RIDIM = x.shape  # [BATCH, CH, T, F, 2]
        assert RIDIM == 2 and CH_IN == self.l_end[-1]
        x = x.permute(0, 2, 3, 1, 4).contiguous().reshape(BATCH * T * F, CH_IN, 2)  # BTF, CH, 2

        ret = [[] for l in range(self.Lout_max + 1)]  # BTF, tau[l], 2l+1, 2

        for l, tau in enumerate(self.tau_in):
            if tau:
                xl = x[:, self.l_begin[l]:self.l_end[l]].contiguous()
                xl = xl.reshape(BATCH * T * F, tau, 2 * l + 1, 2).contiguous()
                ret[l].append(xl)
                if l:
                    xl = xl.reshape(BATCH * T * F, tau * (2 * l + 1), 2)
                    xl = self.cglayers[l](xl.contiguous())
                    for ll in range(min(l * 2, self.Lout_max) + 1):
                        newxll = xl[:, self.cglayers[l].cum_new_tau[ll]:self.cglayers[l].cum_new_tau[ll + 1]]
                        newxll = newxll.reshape(BATCH * T * F, tau * tau, 2 * ll + 1, 2)
                        if self.scale_invariance:
                            norm = torch.sqrt(torch.sum(newxll * newxll, (2, 3)))
                            newxll = newxll / (self.eps + torch.sqrt(norm).reshape((norm.shape[0], norm.shape[1], 1, 1)))
                        ret[ll].append(newxll.contiguous())

        ret2 = []
        for l in range(self.Lout_max + 1):
            if len(ret[l]):
                tmp = torch.cat(ret[l], dim=1).contiguous()
                tmp = tmp[:, self.mask_eachl[l]].contiguous()
                assert tmp.shape == torch.Size([BATCH * T * F, self.tau_out[l], 2 * l + 1, 2])
                ret2.append(tmp.reshape(BATCH, T, F, self.tau_out[l] * (2 * l + 1), 2).contiguous())
        ret2 = torch.cat(ret2, dim=3).contiguous()  # B, T, F, CH_out, 2
        ret2 = ret2.permute(0, 3, 1, 2, 4).contiguous()  # B, CH_out, T, F, 2
        assert ret2.shape == torch.Size([BATCH, self.l_out_end[-1], T, F, 2])
        return ret2


class FirstOrderSpherical2UnitVector(nn.Module):
    """
    Rotation equivariant mapping of 3-d complex vectors in l=1 spherical harmonics domain
    to unit vectors in R^3. (Section IV.G)
    """
    def __init__(self):
        super(FirstOrderSpherical2UnitVector, self).__init__()
        self.eps = 1e-12

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, [*, 3, 2]
            3-d complex vectors in l=1 spherical harmonic domain.
            x[*, 0, *] : l=1, m=-1
            x[*, 1, *] : l=1, m=0
            x[*, 2, *] : l=1, m=1
            x[*, *, 0] : real part, x[*, *, 1] : imaginary part

        Returns
        -------
        ret : torch.Tensor, [*, 3]
            3-d real unit vectors in R^3.
            Intuitively, each 3-d vector in ret points to the argmax of the
            function :math:`f(\\theta, \\phi) = \\Re( \\sum_m x_{l=1}^m Y_{l=1}^m (\\theta, \\phi) )`.

        Examples
        --------
        >>> import torch
        >>> layer = FirstOrderSpherical2UnitVector()
        >>> x = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        >>> layer(x)
        tensor([ 0.7071, -0.7071,  0.0000])

        Elements of input vector x is 0 except for m=-1.
        The direction which maximizes the function
        :math:`\\Re( (1+\\sqrt{-1}) Y_{l=1}^{m=-1}(\\theta, \\phi) )`
        is :math:`[1/\\sqrt{2}, -1/\\sqrt{2}, 0]` (= the output value).

        """
        input_shape = x.shape
        assert input_shape[-2] == 3 and input_shape[-1] == 2
        x = x.reshape(-1, 3, 2)
        re_x = (x[:, 0, 0] - x[:, 2, 0]) / 1.4142136  # TODO: fix to make consistent with the formula in docstring
        re_y = (-x[:, 0, 1] - x[:, 2, 1]) / 1.4142136
        re_z = x[:, 1, 0]
        ret = torch.stack([re_x, re_y, re_z], dim=-1)
        norm = torch.sqrt(torch.sum(ret ** 2, dim=-1))
        ret = ret / (norm.unsqueeze(-1) + self.eps)
        return ret.reshape(input_shape[:-1])
