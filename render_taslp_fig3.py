"""
Generate Fig.3 of the article
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from logging import Formatter, StreamHandler, getLogger, INFO
logger = getLogger(__name__)

plt.rcParams.update({'font.size': 26})

parser = argparse.ArgumentParser(description='options')
parser.add_argument('--eid', type=str, default='taslp', help='experiment id for log')

markers = ['o', 'v', '^', '<', '>', 's', '*', '+', 'x', 'd', 'o', 'v']
styles = [
    'solid',
    (0, (2, 2)),
    (0, (3, 1)),
    'dotted',
    (0, (5, 1)),
    (0, (3, 1, 1, 1, 1, 1)),
    (0, (4, 1, 1, 1)),
]
cmap = plt.get_cmap("tab20")
cids = [7, 6, 9, 8, 1, 0]


def read_logfile(log_fn: str):

    with open(log_fn, 'r') as f:
        lines = f.readlines()

    ax_t, train_loss_sed, train_loss_doa, valid_loss_sed, valid_loss_doa = [], [], [], [], []
    flg = False
    for line in lines:
        ss = line.split()
        if '[INFO]Epoch' in line:
            ax_t.append(int(ss[-5].replace(',', '')))
            train_loss_sed.append(float(ss[-3].replace(',', '')))
            train_loss_doa.append(float(ss[-1].replace(',', '')))
            flg = True
        elif flg:
            valid_loss_sed.append(float(ss[3].replace(',', '')))
            valid_loss_doa.append(float(ss[5].replace(',', '')))
            flg = False

    return [train_loss_sed, train_loss_doa, valid_loss_sed, valid_loss_doa]


if __name__ == '__main__':
    logger.setLevel(INFO)
    handler = StreamHandler()
    handler.setFormatter(Formatter('%(asctime)s %(name)s,l%(lineno)03d[%(levelname)s]%(message)s'))
    logger.addHandler(handler)
    args = parser.parse_args()

    experiment_id = args.eid
    dirpath = os.path.join('./article_figure/', experiment_id)
    try:
        os.mkdir(dirpath)
    except FileExistsError:
        logger.info('Output directory ({:s}) already exists.'.format(dirpath))

    exp_conds = [
        # Enumerate experimental conditions and the corresponding training log file paths here like below:
        # ('Base.', './result/2020-08-05_18-42-39_017557.log'),
        # ('Base. (w time equiv.)', './result/2020-08-05_18-39-45_357551.log'),
        # ('Base. (w rot. equiv.)', './result/2020-08-05_18-41-56_214096.log'),
        # ('Base. (w rot. & time equiv.)', './result/2020-08-05_18-40-43_873531.log'),
        # ('Prop. (w scale equiv.)', './result/2020-08-05_18-35-38_538725.log'),
        # ('Prop. (w time & scale equiv.)', './result/2020-07-30_21-32-37_017604.log'),
    ]

    train_loss_sed, train_loss_doa, validation_loss_sed, validation_loss_doa = [], [], [], []
    for _, log_fn in exp_conds:
        a, b, c, d = read_logfile(log_fn)
        train_loss_sed.append(a)
        train_loss_doa.append(b)
        validation_loss_sed.append(c)
        validation_loss_doa.append(d)

    losses = [train_loss_sed, train_loss_doa, validation_loss_sed, validation_loss_doa]

    fig, axes = plt.subplots(1, 4, figsize=(36.0, 12))

    titles = [
        'Training loss (SED)',
        'Training loss (DOA)',
        'Validation loss (SED)',
        'Validation loss (DOA)',
    ]
    yranges = [
        [0.0, 0.25],
        [0.0, 1.4],
        [0.0, 0.25],
        [0.0, 1.4],
    ]
    for j, c in enumerate(exp_conds):
        for d, (loss_d, ax) in enumerate(zip(losses, axes.flatten())):
            ax.plot(np.arange(len(loss_d[j])) + 1, loss_d[j], color=cmap(cids[j]),
                    linewidth=5, linestyle=styles[j], label=c[0])

    for d, (ax, title, yrange) in enumerate(zip(axes.flatten(), titles, yranges)):
        ax.grid(which='major', color='black', linestyle=':', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_xlim([0, 100])
        ax.set_title(title)
        ax.set_ylim(yrange)
        if d == 0:
            ax.set_ylabel('Loss')
            ax.legend(loc='upper right', borderaxespad=0)
    fig.tight_layout()
    fig.savefig(os.path.join(dirpath, 'Fig3-Lossinfo.pdf'))
    fig.clf()
