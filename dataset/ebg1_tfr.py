import sys
import os

sys.path.append(os.getcwd())

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import random
import dataset.data_utils as data_utils


class EBG1TFR(Dataset):
    def __init__(
            self, root_path: str,
            tmin: float = None,
            tmax: float = None,
            fmin: float = None,
            fmax: float = None,
            shuffle_labels: bool = False,
            seed: int = 42
    ):

        self.root_path = root_path

        # load MATLAB matrices
        # ebg, labels, subject_ids, time_vec, freqs = data_utils.load_ebg1_tfr(
        #     filename_air=os.path.join(self.root_path, 'air_trials_tfr.mat'),
        #     filename_odor=os.path.join(self.root_path, 'odor_trials_tfr.mat')
        # )

        # np.save(os.path.join(self.root_path, 'ebg_tfrs_dataset1.npy'), ebg)
        # np.save(os.path.join(self.root_path, 'ebg_labels_dataset1.npy'), labels)
        # np.save(os.path.join(self.root_path, 'ebg_subjects_dataset1.npy'), subject_ids)
        # np.save(os.path.join(self.root_path, 'ebg_time_vec_dataset1.npy'), time_vec)
        # np.save(os.path.join(self.root_path, 'ebg_freqs_dataset1.npy'), freqs)

        # load NumPy arrays
        ebg = np.load(os.path.join(self.root_path, 'ebg_tfrs_dataset1.npy'), mmap_mode='r')
        labels = np.load(os.path.join(self.root_path, 'ebg_labels_dataset1.npy'), mmap_mode='r')
        subject_ids = np.load(os.path.join(self.root_path, 'ebg_subjects_dataset1.npy'), mmap_mode='r')
        time_vec = np.load(os.path.join(self.root_path, 'ebg_time_vec_dataset1.npy'), mmap_mode='r')
        freqs = np.load(os.path.join(self.root_path, 'ebg_freqs_dataset1.npy'), mmap_mode='r')

        self.baseline_min = -0.09
        self.baseline_max = -0.01
        self.ebg = ebg
        self.labels = np.squeeze(labels)
        self.subject_id = subject_ids
        self.time_vec = time_vec
        self.class_weight = None
        self.freqs = freqs

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
        self.class_weight = torch.tensor(class_0_count/class_1_count)

        self.data = self.ebg

        # self.baseline = np.mean(self.data, axis=(0, -1),
        #                         keepdims=True)
        self.baseline = np.mean(self.data[..., self.baseline_min:self.baseline_max],
                                axis=(-1),
                                keepdims=True)
        self.data = 10 * np.log10(self.data / self.baseline)
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
    ebg_dataset = EBG1TFR(root_path='/Users/nonarajabi/Desktop/KTH/Smell/Novel_Bulb_measure/data/', **data_args)
    # ebg_dataset = EBG1TFR(root_path='/local_storage/datasets/nonar/ebg/', **data_args)
    # np.save(os.path.join("/Users/nonarajabi/Desktop/KTH/Smell/ebg_out/", 'ebg1_tfr_20_100_ebg.npy'), ebg_dataset.ebg)
    # np.save(os.path.join("/Users/nonarajabi/Desktop/KTH/Smell/ebg_out/", 'ebg1_tfr_20_100_labels.npy'),
    #         np.array(ebg_dataset.labels))
    ebg_sample, ebg_label = ebg_dataset[0]
