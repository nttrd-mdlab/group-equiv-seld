"""
Generate Fig.4 of the article
"""
import argparse
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import torch

from dcase19_dataset import TorchDataSet
from logging import Formatter, StreamHandler, getLogger, INFO
from models import get_model

logger = getLogger(__name__)
matplotlib.rcParams.update({'font.size': 24})

parser = argparse.ArgumentParser(description='options')
parser.add_argument('--eid', type=str, default='taslp', help='experiment id for log')

torch.manual_seed(8)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


classes = [
    'Knock',
    'Drawer',
    'Clearthroat',
    'Phone',
    'KeysDrop',
    'Speech',
    'Keyboard',
    'PageTurn',
    'Cough',
    'Doorslam',
    'Laughter'
]
markers = [
    'o', 'v', '^', '<', '>', 's', '*', '+', 'x', 'd', 'o', 'v'
]
styles = [
    'solid', (0, (2, 2)), 'dotted',
    'dotted', (0, (3, 1)), (0, (4, 1, 1, 1)),
    (0, (2, 1, 1, 1)), (0, (4, 2)), (0, (5, 1)),
    'dotted', (0, (3, 1, 1, 1, 1, 1))
]
cmap = plt.get_cmap("tab10")


def set_ax_azim(ax_azim, set_xlabel: bool = True):
    ax_azim.grid(which='major', color='black', linestyle=':')
    ax_azim.set_yticks(np.linspace(-np.pi, np.pi, 9))
    ax_azim.set_yticklabels([r'$-180$', '', r'$-90$', '', r'$0$', '', r'$90$', '', r'$180$'])

    if set_xlabel:
        ax_azim.set_xlabel('Time [sec]')
    else:
        ax_azim.tick_params(labelbottom=False)
    ax_azim.set_ylabel('Azimuth angle [deg]')
    ax_azim.set_ylim([-np.pi, np.pi])
    ax_azim.set_xlim([0, 12 - 0.01])
    ax_azim.legend(loc='upper right', borderaxespad=0)


def set_ax_elev(ax_elev):
    ax_elev.grid(which='major', color='black', linestyle=':')
    ax_elev.set_yticks(np.linspace(-np.pi / 2, np.pi / 2, 5))
    ax_elev.set_yticklabels([r'$-90$', r'$-45$', r'$0$', r'$45$', r'$90$'])

    ax_elev.set_xlabel('Time [sec]')
    ax_elev.set_ylabel('Elevation angle [deg]')
    ax_elev.set_ylim([-np.pi / 2, np.pi / 2])
    ax_elev.set_xlim([0, 12 - 0.01])


def generate_figure(sedmat: np.ndarray, doamat: np.ndarray, fn: str, title: str):
    fig_all = plt.figure(figsize=(18.0, 12.0))

    nrow, ncol = 2, 1
    gs = matplotlib.gridspec.GridSpec(nrow, ncol,
                                      wspace=0.0, hspace=0.1,
                                      top=0.95, bottom=0.1,
                                      left=0.1, right=0.99)

    all_ax_azim = fig_all.add_subplot(gs[0, 0])
    all_ax_elev = fig_all.add_subplot(gs[1, 0])

    ax_t = np.arange(sedmat.shape[0]) * 0.02
    for j, (sed, doa) in enumerate(zip(sedmat.T, doamat.transpose(1, 0, 2))):
        azim = np.arctan2(doa[:, 1], doa[:, 0]) * (sed > 0.5)
        elev = np.arctan2(doa[:, 2], np.sqrt(doa[:, 1] ** 2 + doa[:, 0] ** 2)) * (sed > 0.5)
        azim[sed <= 0.5] = -np.inf
        elev[sed <= 0.5] = -np.inf
        azim -= 0.15
        azim[azim < -np.pi] += np.pi * 2
        all_ax_azim.plot(ax_t, azim, color=cmap(j),
                         linewidth=6, linestyle=styles[j % len(styles)],
                         label=classes[j] if np.max(azim) > -10 else None)
        all_ax_elev.plot(ax_t, elev, color=cmap(j),
                         linewidth=6, linestyle=styles[j % len(styles)],
                         label=classes[j] if np.max(azim) > -10 else None)

    set_ax_azim(all_ax_azim, False)
    set_ax_elev(all_ax_elev)
    all_ax_azim.set_title(title)
    fig_all.savefig(fn + '.pdf')
    fig_all.clf()


if __name__ == '__main__':
    logger.setLevel(INFO)
    handler = StreamHandler()
    handler.setFormatter(Formatter('%(asctime)s %(name)s,l%(lineno)03d[%(levelname)s]%(message)s'))
    logger.addHandler(handler)

    args = parser.parse_args()

    launched_time_str = '{0:%Y-%m-%d}_{0:%H-%M-%S}_{0:%f}'.format(datetime.datetime.now())
    file_formatter = Formatter('%(asctime)s %(name)s,l%(lineno)03d[%(levelname)s]%(message)s')
    dirpath = os.path.join('./article_figure/', args.eid)
    try:
        os.mkdir(dirpath)
    except FileExistsError:
        logger.info('Output directory ({:s}) already exists.'.format(dirpath))

    dataset = TorchDataSet(
        len_restrict=0,
        seq_len=500,  # Length: 10 sec.
        splits=[0],  # split 0: evaluation dataset
        with_conj=True,
        output_trim=0,
        nb_freq_bins_use=1024,
        direction_bias='virtual_rot',
        direction_bias_additional=np.pi,
        single_source_case_only=False,
        test_mode=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    exp_conds = [
        # Enumerate experimental conditions and the corresponding trained model file paths here like below:
        # ('Conv', './checkpoints/2020-08-05_18-39-45_357551_epoch254.checkpoint', 'Baseline method'),
        # ('ConvWithRot', './checkpoints/2020-08-05_18-40-43_873531_epoch236.checkpoint',
        #  'Baseline method (with rotation-based data augmentation)'),
        # ('Prop', './checkpoints/2020-07-30_21-32-37_017604_epoch100.checkpoint', 'Proposed method'),
    ]

    idx = 0  # 0 -> split0_57.npy in one of the authors' environments

    for i, (input_, label) in enumerate(dataloader):
        if i != idx:
            continue
        fid_, init_frame_ = dataset._available_cases[i]
        print(dataset._filenames_list[fid_])
        label_sed = label[0][0].numpy()
        label_doa = label[1][0].numpy()
        generate_figure(label_sed, label_doa, os.path.join(dirpath, 'Fig4-Groundtruth-' + str(i)), 'Ground truth')

        for (exp_cond, path_cp, title) in exp_conds:
            checkpoint = torch.load(path_cp)
            params = checkpoint['params']
            model = get_model(params['model'])
            net = model(cgnet_params=params['cgnet_params']).cuda()
            net.load_state_dict(checkpoint['model_state_dict'])
            net.eval()
            with torch.no_grad():
                ret_sed, ret_doa = net(input_.cuda(), update=False)

                ret_doa = ret_doa.cpu().numpy()[0]
                ret_sed = torch.sigmoid(ret_sed).cpu().numpy()[0]
                generate_figure(ret_sed, ret_doa, os.path.join(dirpath, 'Fig4-' + exp_cond + '-' + str(i)), title)
        break
