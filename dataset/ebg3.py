import sys
import os

sys.path.append(os.getcwd())

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import random
import dataset.data_utils as data_utils


class EBG3TFR(Dataset):
    def __init__(
            self, root_path: str,
            tmin: float = None,
            tmax: float = None,
            fmin: float = None,
            fmax: float = None,
            shuffle_labels: bool = False,
            baseline_type: str = 'db',
            seed: int = 42
    ):

        self.time_vec = None
        self.freqs = None
        self.baseline_min = -0.5
        self.baseline_max = -0.2
        self.ebg = None
        self.labels = None
        self.subject_id = []
        self.class_weight = None

        self.root_path = root_path

        # read data from MATLAB matrices
        # No. 15 doesn't exist, No. 13 had thick hair, and No.21 was anosmic
        # recordings = ['CNTLSL13_control_' + str("{:02d}".format(subject_id)) for subject_id in range(1, 22) if
        #               subject_id != 15 and subject_id != 13 and subject_id != 21]
        #
        # for i, recording in enumerate(recordings):
        #     filename = os.path.join(root_path, recording, 'tfr_CNT_2s_EOG_corrected.mat')
        #     # load MATLAB matrices
        #     ebg_subj, labels_subj, time_vec, freqs = data_utils.load_ebg3_tfr(filename)
        #
        #     if self.time_vec is None:
        #         self.time_vec = time_vec
        #     if self.freqs is None:
        #         self.freqs = freqs
        #
        #     if self.ebg is None:
        #         self.ebg = ebg_subj
        #         self.labels = labels_subj
        #     else:
        #         self.ebg = np.vstack((self.ebg, ebg_subj))
        #         self.labels = np.vstack((self.labels, labels_subj))
        #     self.subject_id.extend(len(ebg_subj)*[i])
        #
        # self.subject_id = np.array(self.subject_id)
        #
        # # Remove NaN values
        # self.ebg = self.ebg[:, :, 13:, :371]
        # self.time_vec = self.time_vec[:371]
        # self.freqs = self.freqs[13:]
        #
        # np.save(os.path.join(self.root_path, 'ebg_tfrs_dataset3.npy'), self.ebg)
        # np.save(os.path.join(self.root_path, 'ebg_labels_dataset3.npy'), self.labels)
        # np.save(os.path.join(self.root_path, 'ebg_subjects_dataset3.npy'), self.subject_id)
        # np.save(os.path.join(self.root_path, 'ebg_time_vec_dataset3.npy'), self.time_vec)
        # np.save(os.path.join(self.root_path, 'ebg_freqs_dataset3.npy'), self.freqs)

        # load NumPy arrays
        self.ebg = np.load(os.path.join(self.root_path, 'ebg_tfrs_dataset3.npy'), mmap_mode='r')
        self.labels = np.load(os.path.join(self.root_path, 'ebg_labels_dataset3.npy'), mmap_mode='r')
        self.subject_ids = np.load(os.path.join(self.root_path, 'ebg_subjects_dataset3.npy'), mmap_mode='r')
        self.time_vec = np.load(os.path.join(self.root_path, 'ebg_time_vec_dataset3.npy'), mmap_mode='r')
        self.freqs = np.load(os.path.join(self.root_path, 'ebg_freqs_dataset3.npy'), mmap_mode='r')

        self.labels = np.squeeze(self.labels)

        if tmin is None:
            self.t_min = 0
        else:
            self.t_min = np.abs(self.time_vec - tmin).argmin()

        if tmax is None:
            self.t_max = len(self.time_vec)
        else:
            self.t_max = np.abs(self.time_vec - tmax).argmin()

        if fmin is None:
            self.f_min = 0
        else:
            self.f_min = np.abs(self.freqs - fmin).argmin()

        if fmax is None:
            self.f_max = len(self.freqs)
        else:
            self.f_max = np.abs(self.freqs - fmax).argmin()

        self.baseline_min = np.abs(self.time_vec - self.baseline_min).argmin()
        self.baseline_max = np.abs(self.time_vec - self.baseline_max).argmin()
        self.time_vec = self.time_vec[self.t_min:self.t_max]

        # self.class_weight = torch.tensor([
        #     len(new_labels) / (new_labels.count(0.) * 2),
        #     len(new_labels) / (new_labels.count(1.) * 2)
        # ])
        class_0_count = list(self.labels).count(0)
        class_1_count = list(self.labels).count(1)
        self.class_weight = torch.tensor(class_0_count / class_1_count)

        self.data = self.ebg
        # self.baseline = np.mean(self.data, axis=(0, -1),
        #                         keepdims=True)
        self.baseline = np.mean(self.data[..., self.baseline_min:self.baseline_max],
                                axis=(-1),
                                keepdims=True)

        # baseline correction
        if baseline_type == 'absolute':
            self.data = self.data - self.baseline
        elif baseline_type == 'relative':
            self.data = self.data / self.baseline
        elif baseline_type == 'relchange':
            self.data = (self.data - self.baseline) / self.baseline
        elif baseline_type == 'normchange' or baseline_type == 'vssum':
            self.data = (self.data - self.baseline) / (self.data + self.baseline)
        elif baseline_type == 'db':
            self.data = 10 * np.log10(self.data / self.baseline)
        elif baseline_type == 'zscore':
            std_vals = np.std(self.data[..., self.baseline_min:self.baseline_max],
                              axis=(-1),
                              keepdims=True)
            self.data = (self.data - self.baseline) / (std_vals+1e-09)
        else:
            raise NotImplementedError

        self.data = self.data[:, :, self.f_min:self.f_max, self.t_min:self.t_max]

        if shuffle_labels:
            self.labels = random.Random(seed).sample(self.labels, len(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        sample = np.mean(np.expand_dims(self.data[item, ...], axis=0), axis=1)
        # mean = sample.mean(axis=(1, 2), keepdims=True)
        # std = sample.std(axis=(1, 2), keepdims=True)
        # sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))  # min-max normalization
        sample = torch.from_numpy(sample).double()

        return sample, self.labels[item]


if __name__ == "__main__":
    # data_args = {'tmin': -0.10, 'tmax': 0.30, 'fmin': 55, 'fmax':70}
    data_args = {}
    ebg_dataset = EBG3TFR(root_path='/Users/nonarajabi/Desktop/KTH/Smell/paper3/TFRs/', **data_args)
    # ebg_dataset = EBG1TFR(root_path='/local_storage/datasets/nonar/ebg/', **data_args)
    # np.save(os.path.join("/Users/nonarajabi/Desktop/KTH/Smell/ebg_out/", 'ebg1_tfr_20_100_ebg.npy'), ebg_dataset.ebg)
    # np.save(os.path.join("/Users/nonarajabi/Desktop/KTH/Smell/ebg_out/", 'ebg1_tfr_20_100_labels.npy'),
    #         np.array(ebg_dataset.labels))
    ebg_sample, ebg_label = ebg_dataset[0]
