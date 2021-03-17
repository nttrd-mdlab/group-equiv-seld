from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import SphHarmDistributedComplexConv2d, \
    SphericalSigmaBN, \
    GroupAvgPool1dBCTF2, \
    CGElementProduct, \
    FirstOrderSpherical2UnitVector, \
    ScaleInvariantSphericalHarmonicsDomainActivation

logger = getLogger(__name__)


class StrictlyCovariantCG(nn.Module):
    def __init__(self, cgnet_params):
        super(StrictlyCovariantCG, self).__init__()
        self.nb_class = cgnet_params['nb_class']
        self.nb_layer = cgnet_params['nb_layer']
        self.Lmax = cgnet_params['Lmax']
        self.nb_skip_bin = cgnet_params['nb_skip_bin']
        self.half_dense_feature = cgnet_params['half_dense_feature']
        self.half_fc_feature = cgnet_params['half_fc_feature']
        self.nb_skip_l0_features = cgnet_params['nb_skip_l0_features']
        ladd = [0 for _ in range(self.Lmax + 1)]
        ladd[0] = (self.half_dense_feature - sum(self.nb_skip_l0_features[1:]) * self.nb_skip_bin)
        ladd[1] = self.nb_class
        self.taus_cgin = cgnet_params['taus_cgins'] + [ladd]
        self.poolns = cgnet_params['pooling_ns']
        self.cnn_paddings = cgnet_params['cnn_paddings']  # (Time, Freq)
        self.cnn_kernel_sizes = cgnet_params['cnn_kernel_sizes']  # (Time, Freq)
        self.cnn_skip_l0s = cgnet_params['cnn_skip_l0s']
        self.sphharm_activation_enabled_ls = cgnet_params['sphharm_activation_enabled_ls']

        cgeps = cgnet_params['cgbilinear_eps'] if 'cgbilinear_eps' in cgnet_params.keys() else 1e-2
        self.cgs = nn.ModuleList([
            CGElementProduct(tau_in=self.taus_cgin[layer], Lout_max=self.Lmax, eps=cgeps, scale_invariance=cgnet_params['scale_equivariance'])
            for layer in range(self.nb_layer)
        ])

        self.sphharmactivation = nn.ModuleList([
            ScaleInvariantSphericalHarmonicsDomainActivation(
                taus=self.cgs[layer].tau_out,
                l_use_list=self.sphharm_activation_enabled_ls[layer],
                eps=cgnet_params['sphharmactivation_eps']
            )
            for layer in range(self.nb_layer)
        ])

        self.convolution = nn.ModuleList([
            SphHarmDistributedComplexConv2d(
                taus_in=self.cgs[layer].tau_out,
                taus_out=self.taus_cgin[layer + 1],
                kernel_size=self.cnn_kernel_sizes[layer],
                padding=self.cnn_paddings[layer],
                skip_l0=self.cnn_skip_l0s[layer]
            )
            for layer in range(self.nb_layer)
        ])

        self.bn_preconv = nn.ModuleList(
            [SphericalSigmaBN(self.cgs[layer].tau_out, momentum=cgnet_params['sphstdbatchnorm_momentum'], eps=cgnet_params['sphstdbatchnorm_eps'])
             for layer in range(self.nb_layer)]
        )
        self.pooling = nn.ModuleList([GroupAvgPool1dBCTF2(stride=self.poolns[layer], taus=self.taus_cgin[layer + 1]) for layer in range(self.nb_layer)])

        self.bn_sed = nn.BatchNorm1d(2 * self.half_dense_feature)
        self.fc_sed1 = nn.Linear(2 * self.half_dense_feature, self.half_fc_feature * 2)
        self.dropout = nn.Dropout(p=cgnet_params['dropout'])
        self.fc_sed2 = nn.Linear(self.half_fc_feature * 2, self.nb_class)

        self.doa = FirstOrderSpherical2UnitVector()

        self.use_gru = cgnet_params['use_GRU']
        if self.use_gru:
            self.nb_gru_layer = cgnet_params['nb_gru_layer']
            self.rnn = nn.GRU(input_size=2 * self.half_dense_feature, hidden_size=self.half_fc_feature, num_layers=self.nb_gru_layer,
                              batch_first=True, dropout=cgnet_params['dropout_gru'], bidirectional=True)

    def forward(self, x, update):
        BATCH, _, _, _, _ = x.shape  # [BATCH, CH, TIME, FREQ, real/imag]
        assert x.shape[1] == 8 and x.shape[4] == 2

        skipsl0 = []
        for layer in range(self.nb_layer):
            x = self.cgs[layer](x)
            x = self.bn_preconv[layer](x, update)
            x = self.sphharmactivation[layer](x)
            x = self.convolution[layer](x)
            x0 = x[:, :self.taus_cgin[layer + 1][0]]
            xl = x[:, self.taus_cgin[layer + 1][0]:]
            x = torch.cat([F.relu(x0), xl], dim=1)
            x = self.pooling[layer](x)
            if layer < self.nb_layer - 1:
                nb_f_now = x.shape[3]
                fs_use = [i * nb_f_now // self.nb_skip_bin for i in range(self.nb_skip_bin)]  # Pick up self.nb_skip_bin frequency bins
                skipsl0.append(x[:, :self.nb_skip_l0_features[layer + 1], :, fs_use])  # [BATCH, (j, l, m), TIME, FREQ_USE, real/imag] in skipsl0's

        TIME_OUT = x.shape[2]
        assert x.shape[0] == BATCH and x.shape[4] == 2

        x_sed = [x[:, :self.taus_cgin[-1][0]].permute(0, 2, 1, 3, 4).reshape(BATCH, TIME_OUT, -1)]
        for s in skipsl0:
            s = s.permute(0, 2, 1, 3, 4)
            T_BEGIN = (s.shape[1] - TIME_OUT) // 2
            s = s[:, T_BEGIN:T_BEGIN + TIME_OUT].reshape(BATCH, TIME_OUT, -1)
            x_sed.append(s)
        x_sed = torch.cat(x_sed, dim=2)
        x_sed = self.bn_sed(x_sed.permute(0, 2, 1)).permute(0, 2, 1)
        assert x_sed.shape == torch.Size([BATCH, TIME_OUT, self.half_dense_feature * 2])

        if self.use_gru:
            x_sed, _ = self.rnn(x_sed)
        else:
            x_sed = self.fc_sed1(x_sed)
        assert x_sed.shape == torch.Size([BATCH, TIME_OUT, self.half_fc_feature * 2])

        x_sed = self.dropout(x_sed)
        x_sed = self.fc_sed2(x_sed)
        assert x_sed.shape == torch.Size([BATCH, TIME_OUT, self.nb_class])

        x_doa = x[:, self.taus_cgin[-1][0]:].contiguous().reshape(BATCH, self.nb_class, 3, TIME_OUT, 2).permute(0, 3, 1, 2, 4)
        x_doa = self.doa(x_doa)
        assert x_doa.shape == torch.Size([BATCH, TIME_OUT, self.nb_class, 3])

        return x_sed, x_doa


class ConventionalNet(nn.Module):
    def __init__(self, cgnet_params):
        super(ConventionalNet, self).__init__()
        self.nb_class = cgnet_params['nb_class']
        self.nb_gru_layer = cgnet_params['nb_gru_layer']
        self.conv1 = nn.Conv2d(8, 64, (3, 3), padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((1, 8))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((1, 8))
        self.conv3 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d((1, 4))

        self.gru = nn.GRU(256, 128, num_layers=self.nb_gru_layer, batch_first=True, dropout=cgnet_params['dropout_gru'], bidirectional=True)
        self.fc_sed1 = nn.Linear(256, 128)
        self.fc_sed2 = nn.Linear(128, self.nb_class)
        self.fc_doa1 = nn.Linear(256, 128)
        self.fc_doa2 = nn.Linear(128, self.nb_class * 3)
        self.dropout = nn.Dropout(p=cgnet_params['dropout'])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.tensor, *args, **kwargs):
        BATCH, _, TIME, FREQ, _ = x.shape  # [BATCH, CH, TIME, FREQ, real/imag]
        assert x.shape[1] == 8 and x.shape[4] == 2
        x = x[:, [0, 2, 3, 4]].permute(0, 2, 3, 1, 4).reshape(BATCH, TIME, FREQ, 8).permute(0, 3, 1, 2)
        # (BATCH, feature, TIME, FREQ)
        x = self.conv1(x)
        x = F.relu(self.batchnorm1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.batchnorm2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.batchnorm3(x))
        x = self.pool3(x)

        x = x.permute(0, 2, 1, 3).reshape(BATCH, TIME, 256)
        x, _ = self.gru(x)

        x_sed = self.fc_sed1(x)
        x_sed = self.dropout(x_sed)
        x_sed = self.fc_sed2(x_sed)

        x_doa = self.fc_doa1(x)
        x_doa = self.dropout(x_doa)
        x_doa = self.fc_doa2(x_doa).reshape(BATCH, TIME, self.nb_class, 3)
        doa_norm = torch.sqrt(torch.sum(x_doa ** 2, dim=-1))
        x_doa = x_doa / (doa_norm.reshape(*doa_norm.shape, 1) + 1e-12)

        return x_sed, x_doa


def get_model(s: str):
    if s == 'Conventional':
        return ConventionalNet
    elif s == 'Proposed':
        return StrictlyCovariantCG
    else:
        raise NotImplementedError()
