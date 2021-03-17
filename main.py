"""
Train the SELD models using DCASE 2019 Task3 Dataset
"""
import argparse
import datetime
import logging
import os
import shutil
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

import dcase19_dataset
import parameter
from math_util import unit_vec_distance, unitvec2azimuthelevation
from models import get_model
import importlib
sys.path.append('seld-dcase2019/metrics')
evaluation_metrics = importlib.import_module('evaluation_metrics')

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


parser = argparse.ArgumentParser(description='training options')
parser.add_argument('--resume', type=str, default=None, help='checkpoint file path to resume')
parser.add_argument('--eid', type=str, default=None, help='experiment identifier (used for saving)')


if __name__ == '__main__':

    launched_time_str = '{0:%Y-%m-%d}_{0:%H-%M-%S}_{0:%f}'.format(datetime.datetime.now())
    args = parser.parse_args()
    experiment_id = launched_time_str if args.eid is None else args.eid
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_formatter = logging.Formatter('%(asctime)s %(name)s,l%(lineno)03d[%(levelname)s]%(message)s')
    filehandler = logging.FileHandler(filename='result/' + experiment_id + '.log')
    filehandler.setLevel(logging.DEBUG)
    filehandler.setFormatter(file_formatter)
    logger.addHandler(filehandler)

    if args.resume:
        checkpoint = torch.load(args.resume)
        params = checkpoint['params']
        model = get_model(params['model'])
        net = model(cgnet_params=params['cgnet_params']).cuda()
        net.load_state_dict(checkpoint['model_state_dict'])
        criterion_sed = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([params['bce_weight']])).cuda()
        criterion_sed.load_state_dict(checkpoint['criterion_sed_state_dict'])
    else:
        params = parameter.get_params()
        model = get_model(params['model'])
        net = model(cgnet_params=params['cgnet_params']).cuda()
        criterion_sed = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([params['bce_weight']])).cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    if params['learning_rate_scheduling']:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    logger.info('Parameters: ' + str(params))
    shutil.copy2('./parameter.py', os.path.join('result', experiment_id + '_param.py'))

    Dataset = dcase19_dataset.TorchDataSet

    dataset = Dataset(len_restrict=0, seq_len=params['seq_len'], splits=params['train_splits'], with_conj=True,
                      output_trim=params['label_trim'], nb_freq_bins_use=params['nb_freq_bin_use'],
                      direction_bias=params['train_rotation_bias'],
                      single_source_case_only=(params['dataset_type'] == 'singlesource'))
    dataloader = data.DataLoader(dataset, batch_size=params['batch_size'], drop_last=False, shuffle=True)
    logger.info('Training dataset: {} minibatches'.format(len(dataset)))

    validation_dataset = Dataset(len_restrict=0, seq_len=params['seq_len'], splits=params['validation_splits'], with_conj=True,
                                 output_trim=params['label_trim'], nb_freq_bins_use=params['nb_freq_bin_use'],
                                 direction_bias=params['train_rotation_bias'], direction_bias_additional=np.pi,
                                 single_source_case_only=(params['dataset_type'] == 'singlesource'))
    validation_dataloader = data.DataLoader(validation_dataset, batch_size=params['batch_size'], drop_last=False, shuffle=False)
    logger.info('Validation dataset: {} minibatches'.format(len(validation_dataset)))

    writer = SummaryWriter('tbx/' + experiment_id, flush_secs=30)
    torch.autograd.set_detect_anomaly(True)

    old_er_score = 1e9
    er_score = 1e8
    latest_improved_epoch = -1

    for epoch in range(params['max_epoch']):
        running_loss_sed = list()
        running_loss_doa = list()
        if er_score < old_er_score:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'criterion_sed_state_dict': criterion_sed.state_dict(),
                    'params': params,
                }, os.path.join('checkpoints', experiment_id + '_epoch' + str(epoch) + '.checkpoint'))
            old_er_score = er_score
            latest_improved_epoch = epoch

        if epoch - latest_improved_epoch > params['wait_limit_epoch']:
            logger.info('Give up at epoch {}'.format(epoch))
            sys.exit()

        net.train()
        for i, (input_, label) in enumerate(dataloader):
            y_sed, y_doa = net(input_.cuda(), update=True)
            optimizer.zero_grad()

            loss_sed = criterion_sed(y_sed, label[0].cuda())
            loss_doa = unit_vec_distance(y_doa, label[1].cuda(), label[0].cuda())

            loss = loss_sed + params['doa_loss_weight'] * loss_doa
            loss.backward()
            optimizer.step()

            running_loss_sed.append(loss_sed.item())
            running_loss_doa.append(loss_doa.item())
        running_loss_sed = np.mean(running_loss_sed)
        running_loss_doa = np.mean(running_loss_doa)

        writer.add_scalar('loss/sed_train', running_loss_sed, epoch)
        writer.add_scalar('loss/doa_train', running_loss_doa, epoch)

        logger.info('Epoch {:3d}, loss_sed: {:.4f}, loss_doa: {:.4f}'.format(epoch + 1, running_loss_sed, running_loss_doa))

        net.eval()
        with torch.no_grad():
            loss_sed_validation = list()
            loss_doa_validation = list()

            doa_preds = []
            doa_gts = []
            sed_preds = []
            sed_gts = []
            for i, (input_, label) in enumerate(validation_dataloader):
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

            writer.add_scalar('loss/sed_valid', loss_sed_validation, epoch)
            writer.add_scalar('loss/doa_valid', loss_doa_validation, epoch)
            writer.add_scalar('criteria/1_ER', er_score, epoch)
            writer.add_scalar('criteria/2_F1', f1_score, epoch)
            writer.add_scalar('criteria/3_DOA', doa_metric[0] / 90.0, epoch)
            writer.add_scalar('criteria/4_FR', doa_metric[1], epoch)
            writer.add_scalar('criteria/5_SELD', seld_metric, epoch)

            logger.info('ValidationSED: {:.4f}, ValidationDOA: {:.4f}, SED ER: {:.4f}, F1: {:.4f}, DOA: {:s}, SELD: {:.4f}'.format(
                loss_sed_validation, loss_doa_validation, er_score, f1_score, str(doa_metric), seld_metric))

        if params['learning_rate_scheduling']:
            scheduler.step()

    torch.save(
        {
            'epoch': params['max_epoch'],
            'model_state_dict': net.state_dict(),
            'criterion_sed_state_dict': criterion_sed.state_dict(),
            'params': params,
        }, os.path.join('checkpoints', experiment_id + '.checkpoint'))
