import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
import scipy.io as scio
import numpy as np
import os
from dataset.ebg1 import EBG1
from dataset.ebg1_tfr import EBG1TFR
from dataset.ebg3 import EBG3TFR
import dataset.data_utils as data_utils


def load(dataset_name: str, path: str, batch_size: int, seed: int, device: str, **kwargs):
    g = torch.Generator().manual_seed(seed)

    if dataset_name == 'dataset1':
        data = EBG1(root_path=path, tmin=kwargs['tmin'], tmax=kwargs['tmax'], fmin=kwargs['fmin'], fmax=kwargs['fmax'],
                    binary=kwargs['binary'], transform=kwargs['ebg_transform'], freqs=kwargs['tfr_freqs'],
                    include_eeg=kwargs['include_eeg'], shuffle_labels=kwargs['shuffle_labels'])
    elif dataset_name == 'dataset1_tfr':
        data = EBG1TFR(root_path=path, tmin=kwargs['tmin'], tmax=kwargs['tmax'], fmin=kwargs['fmin'],
                       fmax=kwargs['fmax'], shuffle_labels=kwargs['shuffle_labels'])
    elif dataset_name == 'dataset3_tfr':
        data = EBG3TFR(root_path=path, tmin=kwargs['tmin'], tmax=kwargs['tmax'], fmin=kwargs['fmin'],
                       fmax=kwargs['fmax'], shuffle_labels=kwargs['shuffle_labels'],
                       baseline_type=kwargs['baseline_type'])
    else:
        raise NotImplementedError

    n_time_samples = len(data.time_vec)
    train_size = int(kwargs['train_size'] * len(data))
    val_size = int(kwargs['val_size'] * len(data))
    test_size = int(len(data) - train_size - val_size)

    print(f"train_size={train_size}, val_size={val_size}, test_size={test_size}")

    train_data, val_data, test_data = torch.utils.data.random_split(
        data, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed))

    train_labels = list(np.array(train_data.dataset.labels)[train_data.indices])
    count_0 = train_labels.count(0.)
    count_1 = train_labels.count(1.)
    class_weights = (count_0 + count_1) / np.array([count_0, count_1])
    sample_weights = np.array([class_weights[int(t)] for t in train_labels])
    sample_weights = torch.from_numpy(sample_weights).double()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False,
                                   drop_last=False,
                                   pin_memory=True if device == 'cuda' else False,
                                   generator=g, sampler=sampler)
    if val_size > 0:
        val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True,
                                     drop_last=False,
                                     pin_memory=True if device == 'cuda' else False,
                                     generator=g)
    else:
        val_data_loader = None

    if test_size > 0:
        test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True,
                                      drop_last=False, pin_memory=True if device == 'cuda' else False,
                                      generator=g)
    else:
        test_data_loader = None

    loaders = {
        'train_loader': train_data_loader,
        'val_loader': val_data_loader,
        'test_loader': test_data_loader
    }

    return data, data.class_weight, n_time_samples


def load_ebg1_ml(root_path: str, **kwargs):
    recordings = ['SL06_' + str("{:02d}".format(subject_id)) + '.mat' for subject_id in range(1, 31) if
                  subject_id != 4]
    indices_to_keep = scio.loadmat(os.path.join(root_path, 'kept_indices_dataset1.mat'))
    indices_to_keep = indices_to_keep['kept_trials']

    ebg_all = {}
    ebg_labels = None
    sampling_freq = None
    time_vector = None
    subject_id = None

    for i, recording in enumerate(recordings):
        file = os.path.join(root_path, recording)
        eeg, ebg, label, time_vec, fs = data_utils.load_ebg1_mat(file, indices_to_keep[0][i])

        if sampling_freq is None:
            sampling_freq = fs
            time_vector = time_vec
            if 'tmin' not in kwargs.keys():
                t_min = 0
            else:
                t_min = np.abs(time_vector - kwargs['tmin']).argmin()

            if 'tmax' not in kwargs.keys():
                t_max = len(time_vector) - 1
            else:
                t_max = np.abs(time_vector - kwargs['tmax']).argmin()

            time_vector = time_vector[t_min:t_max]

        ebg_all[str(i)] = {
            'ebg': ebg[..., t_min:t_max],
            'label': np.array([0. if l == 40 else 1. for l in label])
        }

    return ebg_all, time_vector, sampling_freq

    # if transforms == 'tfr_morlet':
    #     ebg_all = data_utils.apply_tfr_morlet(ebg_all, sampling_freq, kwargs['tfr_freqs'])
    #     ebg_all = ebg_all[..., t_min:t_max]
    #
    #     if reduction == 'window_avg':
    #         window = kwargs['window']
    #         stride = kwargs['stride']
    #         ebg_all = ebg_all.mean(axis=1).squeeze()
    #
    #         ebg_features = np.zeros(
    #             (len(ebg_all), 1 + (ebg_all.shape[1] - window.shape[0]) // stride, 1 +
    #              (ebg_all.shape[2] - window.shape[1]) // stride))
    #         for epoch in len(ebg_all):
    #             ebg_features[epoch, ...] = data_utils.strided_convolution(ebg_all[epoch, ...], window, window.shape[0])
    #
    #         ebg_features.reshape((len(ebg_all), -1))
    #         return ebg_features
    #
    #     elif reduction == 'pca':
    #         pca = PCA(n_components=0.95)
    #         ebg_train_pca = pca.fit_transform(ebg_train)
    #         ebg_test_pca = pca.transform(ebg_test)
    #
    #         return ebg_train_pca, ebg_test_pca, y_train, y_test

# def load_ebg1_tfr_ml(root_path: str, **kwargs):
#     pass
#
#
# def load_ebg3_tfr(root_path: str, **kwargs):
#     # read data from MATLAB matrices
#     # No. 15 doesn't exist, No. 13 had thick hair, and No.21 was anosmic
#     recordings = ['CNTLSL13_control_' + str("{:02d}".format(subject_id)) for subject_id in range(1, 22) if
#                   subject_id != 15 and subject_id != 13 and subject_id != 21]
#
#     ebg_all = {}
#     ebg_labels = None
#     tfr_freqs = None
#     time_vector = None
#     subject_id = None
#     for i, recording in enumerate(recordings):
#         filename = os.path.join(root_path, recording, 'tfr_CNT_2s_EOG_corrected.mat')
#         # load MATLAB matrices
#         ebg_subj, labels_subj, time_vec, freqs = data_utils.load_ebg3_tfr(filename)
#
#         if time_vector is None:
#             time_vector = time_vec
#         if tfr_freqs is None:
#             tfr_freqs = freqs
#
#         if ebg is None:
#             self.ebg = ebg_subj
#             self.labels = labels_subj
#         else:
#             self.ebg = np.vstack((self.ebg, ebg_subj))
#             self.labels = np.vstack((self.labels, labels_subj))
#         self.subject_id.extend(len(ebg_subj)*[i])
#
#     self.subject_id = np.array(self.subject_id)
#
#     # Remove NaN values
#     self.ebg = self.ebg[:, :, 13:, :371]
#     self.time_vec = self.time_vec[:371]
#     self.freqs = self.freqs[13:]