"""
adversarial attack for trained model
"""
import argparse
import datetime
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import sys
import torch
import torch.nn as nn
from torch.utils import data

from dcase19_dataset import TorchDataSet
from math_util import unit_vec_distance, unitvec2azimuthelevation
from models import get_model
import importlib
sys.path.append('seld-dcase2019/metrics')
evaluation_metrics = importlib.import_module('evaluation_metrics')
from evaluation_metrics import reshape_3Dto2D, compute_sed_scores, compute_doa_scores_regr, compute_seld_metric

mpl.use('Agg')

plt.rcParams["font.size"] = 18
plt.tight_layout()
parser = argparse.ArgumentParser(description='options')
parser.add_argument('--resume', type=str, default=None, help='checkpoint file path to resume')
parser.add_argument('--eid', type=str, default=None, help='experiment id for log')


class AdversarialRotation1d(nn.Module):
    """
    Rotation to ambisonic signals / 3-d real vectors.
    """
    def __init__(self):
        super(AdversarialRotation1d, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x: torch.tensor):
        if len(x.shape) == 4:
            # x is doa estimation result
            assert x.shape[-1] == 3
            X, Y, Z = x[:, :, :, 0], x[:, :, :, 1], x[:, :, :, 2]
            X, Y = X * torch.cos(self.alpha) - Y * torch.sin(self.alpha), Y * torch.cos(self.alpha) + X * torch.sin(self.alpha)
            Z, X = Z * torch.cos(self.beta) - X * torch.sin(-self.beta), X * torch.cos(self.beta) + Z * torch.sin(-self.beta)
            X, Y = X * torch.cos(self.gamma) - Y * torch.sin(self.gamma), Y * torch.cos(self.gamma) + X * torch.sin(self.gamma)
            ret = torch.stack([X, Y, Z], -1)
            return ret

        if len(x.shape) == 5:
            # x is observed STFT spectrogram
            BATCH, CH, TIME, FREQ, RIDIM = x.shape
            assert CH == 8 and RIDIM == 2

            ret = [x[:, 0], x[:, 1]]
            for l in [3, 6]:
                for m in range(-1, 2):
                    z_m_re = x[:, l + m, :, :, 0] * torch.cos(self.gamma * m) + x[:, l + m, :, :, 1] * torch.sin(self.gamma * m)
                    z_m_im = x[:, l + m, :, :, 1] * torch.cos(self.gamma * m) - x[:, l + m, :, :, 0] * torch.sin(self.gamma * m)
                    ret.append(torch.stack([z_m_re, z_m_im], dim=-1))
            x = torch.stack(ret, dim=1)

            ret = [x[:, 0], x[:, 1]]
            for l in [3, 6]:
                dpp = (1.0 + torch.cos(self.beta)) / 2.0
                dp0 = -torch.sin(self.beta) / 1.4142136
                dpm = (1.0 - torch.cos(self.beta)) / 2.0
                d00 = torch.cos(self.beta)
                ret.append(x[:, l - 1] * dpp - x[:, l] * dp0 + x[:, l + 1] * dpm)
                ret.append(x[:, l - 1] * dp0 + x[:, l] * d00 - x[:, l + 1] * dp0)
                ret.append(x[:, l - 1] * dpm + x[:, l] * dp0 + x[:, l + 1] * dpp)
            x = torch.stack(ret, dim=1)

            ret = [x[:, 0], x[:, 1]]
            for l in [3, 6]:
                for m in range(-1, 2):
                    z_m_re = x[:, l + m, :, :, 0] * torch.cos(self.alpha * m) + x[:, l + m, :, :, 1] * torch.sin(self.alpha * m)
                    z_m_im = x[:, l + m, :, :, 1] * torch.cos(self.alpha * m) - x[:, l + m, :, :, 0] * torch.sin(self.alpha * m)
                    ret.append(torch.stack([z_m_re, z_m_im], dim=-1))
            x = torch.stack(ret, dim=1)
            return x


if __name__ == '__main__':
    args = parser.parse_args()

    launched_time_str = '{0:%Y-%m-%d}_{0:%H-%M-%S}_{0:%f}'.format(datetime.datetime.now())
    experiment_id = launched_time_str if args.eid is None else args.eid
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s %(name)s,l%(lineno)03d[%(levelname)s]%(message)s')
    filehandler = logging.FileHandler(filename='ret_adv/' + experiment_id + '.log')
    filehandler.setLevel(logging.DEBUG)
    filehandler.setFormatter(file_formatter)
    logger.addHandler(filehandler)

    checkpoint = torch.load(args.resume)
    params = checkpoint['params']
    model = get_model(params['model'])

    net = model(cgnet_params=params['cgnet_params']).cuda()
    net.load_state_dict(checkpoint['model_state_dict'])

    total_nb_param = 0
    for i, k in enumerate(checkpoint['model_state_dict'].keys()):
        prm = checkpoint['model_state_dict'][k]
        nb_param = np.prod(np.array(prm.shape))
        logger.info(str(k) + ' ' + str(prm.shape) + ' ' + str(nb_param))
        total_nb_param += int(nb_param)
    logger.info('total parameters: {:d}'.format(total_nb_param))

    dataset = TorchDataSet(
        len_restrict=0,
        seq_len=params['seq_len'],
        splits=params['evaluation_splits'],
        with_conj=True,
        output_trim=params['label_trim'],
        nb_freq_bins_use=params['nb_freq_bin_use'],
        direction_bias=params['train_rotation_bias'],
        direction_bias_additional=180,
        single_source_case_only=(params['dataset_type'] == 'singlesource'))
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)

    rotationLayer = AdversarialRotation1d().cuda()
    optimizer = torch.optim.SGD(rotationLayer.parameters(), lr=1e-1)
    criterion_sed = nn.BCEWithLogitsLoss().cuda()

    highest_loss = 0.0
    dirpath = os.path.join('./ret_adv/', os.path.split(args.resume)[1])
    try:
        os.mkdir(dirpath)
    except FileExistsError:
        pass

    loss_sed_validation = list()
    loss_doa_validation = list()
    doa_preds = []
    doa_gts = []
    sed_preds = []
    sed_gts = []

    for i, (input_, label) in enumerate(dataloader):
        logger.info('Data {:4d}'.format(i))

        net.train()
        for param in net.parameters():
            param.requires_grad = False
        rotationLayer.train()

        for m in net.modules():
            if m.__class__.__name__.startswith('Dropout'):
                print(m.__class__.__name__, 'Ignored.')
                m.eval()

        input_ = input_.cuda()
        label_sed = label[0].cuda()
        label_doa = label[1].cuda()
        for i_train in range(100):
            X = input_.clone()
            X = rotationLayer(X)
            y_sed, y_doa = net(X, update=False)
            y_doa = rotationLayer(y_doa)
            optimizer.zero_grad()
            loss_sed = criterion_sed(y_sed, label_sed)
            loss_doa = unit_vec_distance(y_doa, label_doa, label_sed)
            # loss = -loss_sed - loss_doa
            # loss = - loss_doa
            loss = - loss_sed

            if i == 0:
                sed_est = reshape_3Dto2D(torch.sigmoid(y_sed).cpu().detach().numpy())
                if highest_loss < loss_sed.item() + loss_doa.item():
                    highest_loss = loss_sed.item() + loss_doa.item()
                    ax_t = np.arange(sed_est.shape[0]) * 0.02
                    for j, v in enumerate(sed_est.T):
                        plt.plot(ax_t, v + j)
                    plt.xlabel('Time [sec]')
                    plt.ylabel('On / Off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(dirpath, 'sed_est_' + str(i_train + 1) + '.png'))
                    plt.close()
                if i_train == 0:
                    sed_gt = reshape_3Dto2D(label[0].numpy())
                    for j, v in enumerate(sed_gt.T):
                        plt.plot(ax_t, v + j)
                    plt.xlabel('Time [sec]')
                    plt.ylabel('On / Off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(dirpath, 'sed_gt.png'))
                    plt.close()

            loss.backward()
            optimizer.step()

        net.eval()
        rotationLayer.eval()
        with torch.no_grad():
            X = rotationLayer(input_)
            ret_sed, ret_doa = net(X, update=False)

            doa_preds.append(reshape_3Dto2D(unitvec2azimuthelevation(ret_doa.cpu().numpy())))
            doa_gts.append(reshape_3Dto2D(unitvec2azimuthelevation(label[1].numpy())))
            loss_doa_validation.append(float(unit_vec_distance(ret_doa, label_doa, label_sed)))

            loss_sed_validation.append(float(criterion_sed(ret_sed, label_sed)))
            sed_preds.append(reshape_3Dto2D(torch.sigmoid(ret_sed).cpu().numpy()) > 0.5)
            sed_gts.append(reshape_3Dto2D(label[0].numpy()) > 0.5)

    sed_preds = np.concatenate(sed_preds)
    sed_gts = np.concatenate(sed_gts)
    er_score, f1_score = compute_sed_scores(sed_preds, sed_gts, 50)
    loss_sed_validation = np.mean(loss_sed_validation)

    doa_preds = np.concatenate(doa_preds)
    doa_gts = np.concatenate(doa_gts)
    doa_metric = compute_doa_scores_regr(doa_preds, doa_gts, sed_preds, sed_gts)
    seld_metric = compute_seld_metric([er_score, f1_score], doa_metric)
    loss_doa_validation = np.mean(loss_doa_validation)

    logger.info('EvaluationSED: {:.4f}, EvaluationDOA: {:.4f}, SED ER: {:.4f}, F1: {:.4f}, DOA: {:s}, SELD: {:.4f}'.format(
        loss_sed_validation, loss_doa_validation, er_score, f1_score, str(doa_metric), seld_metric))
