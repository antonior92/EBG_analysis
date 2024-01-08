import matplotlib.pyplot as plt
import numpy as np
from dataset import data_utils
from dataset.load_data import load_ebg1_ml


cluster_data_path = '/local_storage/datasets/nonar/ebg/'
cluster_save_path = '/Midgard/home/nonar/data/ebg/ebg_out/'
local_data_path = "/Users/nonarajabi/Desktop/KTH/Smell/Novel_Bulb_measure/data/"
local_save_path = "/Users/nonarajabi/Desktop/KTH/Smell/ebg_out/"


if __name__ == "__main__":
    ebg_all, time_vec, fs = load_ebg1_ml(root_path=local_data_path, tmin=-1., tmax=1.)
    fs = fs.astype(float)

    baseline_min = np.abs(time_vec - (-.5)).argmin()
    baseline_max = np.abs(time_vec - (-.2)).argmin()

    tfr_freqs = np.linspace(20, 100, 160)
    gamma_band = (55, 65)
    gamma_band_min = np.abs(tfr_freqs - gamma_band[0]).argmin()
    gamma_band_max = np.abs(tfr_freqs - gamma_band[1]).argmin()

    t_min_list = np.linspace(0.0, 0.2, 4)
    window_size = 0.05

    for subj in ebg_all.keys():

        print(f"------- Subject {subj} -------")
        ebg_data = ebg_all[subj]['ebg']
        ebg_labels = ebg_all[subj]['label']
        # ebg_data = ebg_data[..., t_min:t_max]

        class0 = np.where(ebg_labels == 0.)
        class1 = np.where(ebg_labels == 1.)

        ebg_data = data_utils.apply_tfr(ebg_data, fs, tfr_freqs)
        ebg_baseline = np.mean(ebg_data[..., baseline_min:baseline_max], axis=(0, -1), keepdims=True)
        ebg_data = 10 * np.log10(ebg_data/ebg_baseline)
        ebg_data = np.mean(ebg_data, axis=1)
        # ebg_data = (ebg_data - np.min(ebg_data))/(np.max(ebg_data) - np.min(ebg_data))

        fig, axs = plt.subplots(1, 4, figsize=(30, 10))
        axs = axs.flatten()

        for i, t_min in enumerate(t_min_list):

            t_max = t_min + window_size
            tmin = np.abs(time_vec - t_min).argmin()
            tmax = np.abs(time_vec - t_max).argmin()
            ebg_window = np.mean(ebg_data[:, gamma_band_min:gamma_band_max, tmin:tmax], axis=(-1))
            ebg_window = np.amax(ebg_window, axis=-1)
            # ebg_window = np.median(ebg_window, axis=-1)
            ebg_window_0 = ebg_window[class0, ...].flatten()
            ebg_window_1 = ebg_window[class1, ...].flatten()

            axs[i].hist(ebg_window_0, label='class 0', alpha=0.5, density=True, histtype="stepfilled", bins=10)
            axs[i].hist(ebg_window_1, label='class 1', alpha=0.5, density=True, histtype="stepfilled", bins=10)
            axs[i].set_title('window '+'{:.2f}'.format(t_min)+'-'+'{:.2f}'.format(t_max))
            axs[i].legend()

        plt.suptitle(f'subject {subj}')
        plt.show()


