"""
Run estimation using evaluation dataset and the trained model,
and evaluate the model performance
"""
import argparse
import datetime
import logging
import numpy as np
import sys
import torch
import torch.nn as nn
from torch.utils import data

import dcase19_dataset
from math_util import unit_vec_distance, unitvec2azimuthelevation
from models import get_model
import importlib
sys.path.append('seld-dcase2019/metrics')
evaluation_metrics = importlib.import_module('evaluation_metrics')

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


parser = argparse.ArgumentParser(description='options')
parser.add_argument('--eid', type=str, default=None, help='')
parser.add_argument('--resume', type=str, default=None, help='experiment identifier (used for loading)')


if __name__ == '__main__':

    launched_time_str = '{0:%Y-%m-%d}_{0:%H-%M-%S}_{0:%f}'.format(datetime.datetime.now())
    args = parser.parse_args()
    experiment_id = launched_time_str if args.eid is None else args.eid
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_formatter = logging.Formatter('%(asctime)s %(name)s,l%(lineno)03d[%(levelname)s]%(message)s')
    filehandler = logging.FileHandler(filename='ret_eval/' + experiment_id + '.log')
    filehandler.setLevel(logging.DEBUG)
    filehandler.setFormatter(file_formatter)
    logger.addHandler(filehandler)

    checkpoint = torch.load(args.resume)
    params = checkpoint['params']
    model = get_model(params['model'])
    net = model(cgnet_params=params['cgnet_params']).cuda()
    net.load_state_dict(checkpoint['model_state_dict'])
    criterion_sed = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([params['bce_weight']])).cuda()
    criterion_sed.load_state_dict(checkpoint['criterion_sed_state_dict'])

    logger.info('Parameters: ' + str(params))
    logger.info('Checkpoint: {}'.format(args.resume))
    logger.info('Experiment id: {}'.format(experiment_id))

    Dataset = dcase19_dataset.TorchDataSet
    evaluation_dataset = Dataset(len_restrict=0, seq_len=3000, splits=params['evaluation_splits'], with_conj=True,
                                 output_trim=params['label_trim'], nb_freq_bins_use=params['nb_freq_bin_use'],
                                 direction_bias=params['train_rotation_bias'], direction_bias_additional=np.pi,
                                 single_source_case_only=(params['dataset_type'] == 'singlesource'))
    evaluation_dataloader = data.DataLoader(evaluation_dataset, batch_size=1, drop_last=False, shuffle=False)
    logger.info('Evaluation dataset: {} minibatches'.format(len(evaluation_dataset)))

    torch.autograd.set_detect_anomaly(True)

    old_er_score = 1e9
    er_score = 1e8
    net.eval()
    with torch.no_grad():
        loss_sed_validation = list()
        loss_doa_validation = list()

        doa_preds = []
        doa_gts = []
        sed_preds = []
        sed_gts = []
        for i, (input_, label) in enumerate(evaluation_dataloader):
            ret_sed, ret_doa = net(input_.cuda(), update=False)
            doa_preds.append(evaluation_metrics.reshape_3Dto2D(unitvec2azimuthelevation(ret_doa)).cpu().numpy())
            doa_gts.append(evaluation_metrics.reshape_3Dto2D(unitvec2azimuthelevation(label[1])).numpy())
            loss_doa_validation.append(float(unit_vec_distance(ret_doa, label[1].cuda(), label[0].cuda())))

            loss_sed_validation.append(float(criterion_sed(ret_sed, label[0].cuda())))
            sed_preds.append(evaluation_metrics.reshape_3Dto2D(torch.sigmoid(ret_sed).cpu().numpy()) > 0.5)
            sed_gts.append(evaluation_metrics.reshape_3Dto2D(label[0].numpy()) > 0.5)

        sed_preds = np.concatenate(sed_preds)
        sed_gts = np.concatenate(sed_gts)
        doa_preds = np.concatenate(doa_preds)
        doa_gts = np.concatenate(doa_gts)
        loss_sed_validation = np.mean(loss_sed_validation)
        loss_doa_validation = np.mean(loss_doa_validation)

        [er_score, f1_score] = evaluation_metrics.compute_sed_scores(sed_preds, sed_gts, 50)
        doa_metric = evaluation_metrics.compute_doa_scores_regr(doa_preds, doa_gts, sed_preds, sed_gts)
        seld_metric = evaluation_metrics.compute_seld_metric([er_score, f1_score], doa_metric)

        logger.info('EvaluationSED: {:.4f}, EvaluationDOA: {:.4f}, SED ER: {:.4f}, F1: {:.4f}, DOA: {:s}, SELD: {:.4f}'.format(
            loss_sed_validation, loss_doa_validation, er_score, f1_score, str(doa_metric), seld_metric))
