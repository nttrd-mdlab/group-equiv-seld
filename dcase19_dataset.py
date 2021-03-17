"""
DCASE 2019 Task 3 Dataset
"""

from logging import getLogger
import os
import random
import numpy as np
from torch.utils import data

import parameter
from math_util import WXYZ2Sph, rotate_vec, rotate_foa, \
    phase_alignment, decide_rotation
logger = getLogger(__name__)


class TorchDataSet(data.Dataset):
    def __init__(self, seq_len, splits=[1, ], random_shuffle=False,
                 len_restrict=0, with_conj=False, rotate=None, output_trim=0,
                 nb_freq_bins_use=None, direction_bias=None,
                 direction_bias_additional=None,
                 single_source_case_only=False,
                 test_mode=False):
        self.params = parameter.get_params()
        self._seq_len = seq_len
        self._splits = np.array(splits)
        self._data_dir = self.params["dataset_dir"]
        self._label_dir = os.path.join(self._data_dir, 'label')
        self._feat_dir = os.path.join(self._data_dir, 'foa_norm')
        self._nb_classes = 11
        self._nb_ch = 4
        self._2_nb_ch = 2 * self._nb_ch
        self._nondeteministic_shuffle = random_shuffle
        self._filenames_list = list()
        self.gen_data_file_name_list(len_restrict)
        self._available_cases = list()
        self.single_source_case_only = single_source_case_only
        self.gen_available_cases()
        self.with_conj = with_conj
        self.rotate = rotate
        self.output_trim = output_trim
        self._nb_freq_bins_use = nb_freq_bins_use
        self._direction_bias = direction_bias
        self._direction_bias_additional = direction_bias_additional

    def gen_data_file_name_list(self, len_restrict):
        self._filenames_list = list()
        for filename in os.listdir(self._feat_dir):
            if len(filename) > 5 and int(filename[5]) in self._splits:
                self._filenames_list.append(filename)

        logger.debug('Dataset File List Generated: {:d} files.'
                     .format(len(self._filenames_list)))
        if self._nondeteministic_shuffle:
            random.shuffle(self._filenames_list)
        else:
            random.Random(1).shuffle(self._filenames_list)

        sample_feature = np.load(os.path.join(self._feat_dir, self._filenames_list[0]))  # 3000 x 8192
        self._nb_frames_file = sample_feature.shape[0]
        self._nb_freq_bins = sample_feature.shape[1] // self._2_nb_ch

        if len_restrict > 0 and len(self._filenames_list) > len_restrict:
            self._filenames_list = self._filenames_list[:len_restrict]

    def gen_available_cases(self):
        self._available_cases = list()
        for fn_id, filename in enumerate(self._filenames_list):
            class_info = np.load(os.path.join(self._label_dir, filename)).astype('float32')[:, :self._nb_classes]
            last = -1
            for t, vec in enumerate(class_info):
                if self.single_source_case_only and np.sum(vec) > 1.5:
                    last = t
                if t - last == self._seq_len:
                    self._available_cases.append((fn_id, last + 1))
                    last = t
        logger.debug('Valid Segment List Generated: {:d} segments.'.format(len(self._available_cases)))

    def __len__(self):
        return len(self._available_cases)

    def __getitem__(self, idx, rotate=None):
        filename_id, init_frame = self._available_cases[idx]
        filename = self._filenames_list[filename_id]

        feature = np.load(os.path.join(self._feat_dir, filename)).astype('float32')[init_frame: init_frame + self._seq_len, :]
        label = np.load(os.path.join(self._label_dir, filename)).astype('float32')[init_frame: init_frame + self._seq_len, :]

        # ret_in: [(BATCHSIZE,) CH, TIME, FREQ, Re/Im]
        ret_in = feature.reshape(self._seq_len, 2, self._nb_freq_bins, self._nb_ch).transpose(3, 0, 2, 1)
        ret_in = ret_in[:, :, :self._nb_freq_bins_use]

        ret_in = phase_alignment(ret_in, tdiff=self.params['feature_phase_different_bin'])
        label = label[self.params['feature_phase_different_bin']:]

        class_ = label[self.output_trim:label.shape[0] - self.output_trim, :self._nb_classes]
        azi_rad = label[self.output_trim:label.shape[0] - self.output_trim, self._nb_classes:2 * self._nb_classes] * np.pi / 180
        ele_rad = label[self.output_trim:label.shape[0] - self.output_trim, 2 * self._nb_classes:3 * self._nb_classes] * np.pi / 180

        x = np.cos(ele_rad) * np.cos(azi_rad)
        y = np.cos(ele_rad) * np.sin(azi_rad)
        z = np.sin(ele_rad)
        xyz = np.stack([x, y, z], axis=-1)

        if rotate is not None:
            pass
        elif self._direction_bias == 'virtual_rot':
            rotate = decide_rotation(azi_rad, class_, 241)
        elif self._direction_bias == 'azi_random':
            rotate = [np.random.rand() * 2 * np.pi, 0.0, 0.0]

        if self._direction_bias_additional is not None:
            if rotate is None:
                rotate = [0.0, 0.0, 0.0]
            rotate[0] += self._direction_bias_additional

        if rotate is not None:
            xyz = rotate_vec(xyz.transpose(2, 0, 1), rotate).transpose(1, 2, 0)
            ret_in = rotate_foa(ret_in, rotate)

        if self.with_conj:
            ret_in_conj = np.array(ret_in)
            ret_in_conj[:, :, :, 1] *= -1
            ret_in = WXYZ2Sph(ret_in)
            ret_in_conj = WXYZ2Sph(ret_in_conj)
            assert ret_in.shape[0] == 4 and ret_in.shape[3] == 2
            ret = np.concatenate((ret_in[:1], ret_in_conj[:1], ret_in[1:], ret_in_conj[1:]), axis=0)
            assert ret.shape[0] == 8
            return ret, (class_, xyz)
        else:
            return ret_in, (class_, xyz)
