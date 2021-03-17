from logging import Formatter, StreamHandler, getLogger, INFO
import librosa
import joblib
import numpy as np
from scipy.io import wavfile
from sklearn import preprocessing
import os
logger = getLogger(__name__)

input_root_dir = './tausse_raw'  # REWRITE HERE
output_root_dir = './tausse_extracted'

# Put the dataset files downloaded from
# https://zenodo.org/record/2599196 and https://zenodo.org/record/3377088#.YENjV3X7QUE
# to the directories below:
input_foa_dev_dir = os.path.join(input_root_dir, 'foa_dev')
input_foa_eval_dir = os.path.join(input_root_dir, 'foa_eval')
input_metadata_dev_dir = os.path.join(input_root_dir, 'metadata_dev')
input_metadata_eval_dir = os.path.join(input_root_dir, 'metadata_eval')

output_foa_dir = os.path.join(output_root_dir, 'foa_norm')
output_label_dir = os.path.join(output_root_dir, 'label')


def make_dir(dirpath: str) -> None:
    try:
        os.mkdir(dirpath)
        logger.info('Directory {:s} created.'.format(dirpath))
    except FileExistsError:
        logger.info('Output directory ({:s}) already exists.'.format(dirpath))


fs_hz = 48000
hop_len_sec = 0.02
hop_len_samples = int(fs_hz * hop_len_sec)
frame_per_sec = fs_hz / float(hop_len_samples)
duration_sec = 60
max_len_samples = duration_sec * fs_hz

window_len_samples = hop_len_samples * 2
fftlen = 1 << (window_len_samples - 1).bit_length()
eps = np.spacing(1e-16)
nb_ch = 4  # FOA
max_frames = int(np.ceil(fs_hz * duration_sec / float(hop_len_samples)))

classes = [
    'knock',
    'drawer',
    'clearthroat',
    'phone',
    'keysDrop',
    'speech',
    'keyboard',
    'pageturn',
    'cough',
    'doorslam',
    'laughter'
]
classname2id = dict()
for i, name in enumerate(classes):
    classname2id[name] = i


def read_metadata_csv(filename: str) -> np.ndarray:
    sed_gt = np.zeros((max_frames, len(classes)))
    azi_gt = 180 * np.ones((max_frames, len(classes)))
    ele_gt = 50 * np.ones((max_frames, len(classes)))
    with open(filename, 'r') as f:
        next(f)
        for line in f:
            classname, start_sec, end_sec, ele, azi, _ = line.strip().split(',')
            if classname in classname2id:
                class_id = classname2id[classname]
            else:
                logger.error('Classname search failed: {:s}'.format(classname))
            start_fr, end_fr, ele_deg, azi_deg = int(float(start_sec) * frame_per_sec), int(float(end_sec) * frame_per_sec), int(ele), int(azi)
            sed_gt[start_fr:end_fr + 2, class_id] = 1
            azi_gt[start_fr:end_fr + 2, class_id] = azi_deg
            ele_gt[start_fr:end_fr + 2, class_id] = ele_deg
    return np.concatenate((sed_gt, azi_gt, ele_gt), axis=1)


if __name__ == '__main__':
    logger.setLevel(INFO)
    handler = StreamHandler()
    handler.setFormatter(Formatter('%(asctime)s %(name)s,l%(lineno)03d[%(levelname)s]%(message)s'))
    logger.addHandler(handler)

    make_dir(output_root_dir)
    make_dir(output_foa_dir)
    make_dir(output_label_dir)

    # Convert label files
    for input_dir in (input_metadata_dev_dir, input_metadata_eval_dir):
        for fn in os.listdir(input_dir):
            logger.info(fn)
            c = read_metadata_csv(os.path.join(input_dir, fn))
            np.save(os.path.join(output_label_dir, '{}.npy'.format(fn.split('.')[0])), c)

    spec_reim_scaler = preprocessing.StandardScaler()

    def get_spec(target_file_path: str):
        _, sig = wavfile.read(target_file_path)
        sig = sig / 32768.0
        if sig.shape[0] < max_len_samples:
            sig = np.vstack((sig, np.zeros((max_len_samples - sig.shape[0], nb_ch))))
        else:
            sig = sig[:max_len_samples, :]
        nb_bins = fftlen // 2
        spec = np.zeros((max_frames, nb_bins, nb_ch), dtype=complex)
        for ch, v in enumerate(sig.T):
            stft_ch = librosa.core.stft(v, n_fft=fftlen, hop_length=hop_len_samples, win_length=window_len_samples, window='hann')
            spec[:, :, ch] = stft_ch[1:, :max_frames].T
        return spec.reshape(max_frames, -1)

    for wav_filename in sorted(os.listdir(input_foa_dev_dir)):
        logger.info('{}'.format(wav_filename))
        specgram = get_spec(os.path.join(input_foa_dev_dir, wav_filename))
        spec_reim_scaler.partial_fit(np.concatenate((np.real(specgram), np.imag(specgram)), axis=1))

    joblib.dump(spec_reim_scaler, os.path.join(output_root_dir, 'foa_wts'))
    logger.info('foa_wts saved.')

    for input_dir in (input_foa_dev_dir, input_foa_eval_dir):
        for wav_fn in os.listdir(input_dir):
            logger.info('{}'.format(wav_fn))
            specgram = get_spec(os.path.join(input_dir, wav_fn))
            specgram = spec_reim_scaler.transform(np.concatenate((np.real(specgram), np.imag(specgram)), axis=1))
            np.save(os.path.join(output_foa_dir, '{:s}.npy'.format(wav_fn.split('.')[0])), specgram)
