import numpy as np
from scipy.signal import convolve2d, decimate
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, normalize, minmax_scale
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, RepeatedStratifiedKFold, cross_val_score
from imblearn.over_sampling import RandomOverSampler
import random
from dataset.load_data import load_ebg1_ml
from dataset import data_utils
import matplotlib.pyplot as plt


cluster_data_path = '/local_storage/datasets/nonar/ebg/'
cluster_save_path = '/Midgard/home/nonar/data/ebg/ebg_out/'
local_data_path = "/Users/nonarajabi/Desktop/KTH/Smell/Novel_Bulb_measure/data/"
local_save_path = "/Users/nonarajabi/Desktop/KTH/Smell/ebg_out/"


def weighted_f1_score(binary_conf_mat, weight_0, weight_1):

    f1_0 = binary_conf_mat[0, 0] / (binary_conf_mat[0, 0] + 0.5 * (binary_conf_mat[0, 1] + binary_conf_mat[1, 0]))
    # f1_1 = binary_conf_mat[1, 1] / (binary_conf_mat[1, 1] + 0.5 * (binary_conf_mat[0, 1] + binary_conf_mat[1, 0]))
    # return (weight_0 * f1_0 + weight_1 * f1_1) / (weight_0 + weight_1)
    return f1_0


def evaluate_metrics(y_pred, y_true):
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    confusion_matrix_val = confusion_matrix(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    f1_score = weighted_f1_score(confusion_matrix_val,
                                 np.sum(confusion_matrix_val, axis=1)[0], np.sum(confusion_matrix_val, axis=1)[1])

    return balanced_accuracy, confusion_matrix_val, roc_auc, f1_score


def apply_training_pipeline(data, labels, training_indices, test_indices, **kwargs):
    oversample = RandomOverSampler(sampling_strategy='minority')

    # train-test split
    ebg_train, y_train = data[training_indices, ...], labels[training_indices]
    ebg_test, y_test = data[test_indices, ...], labels[test_indices]

    ebg_train, y_train = oversample.fit_resample(ebg_train, y_train)

    # feature extraction
    scaler = StandardScaler()
    ebg_train = scaler.fit_transform(ebg_train)
    ebg_test = scaler.transform(ebg_test)

    # ebg_train = minmax_scale(ebg_train)
    # ebg_test = minmax_scale(ebg_test)

    # pca = PCA(n_components=0.95)
    # ebg_train_pca = pca.fit_transform(ebg_train)
    # ebg_test_pca = pca.transform(ebg_test)
    lda = LinearDiscriminantAnalysis(n_components=1)
    ebg_train_pca = lda.fit_transform(ebg_train, y_train)
    ebg_test_pca = lda.transform(ebg_test)
    # ebg_train_pca = ebg_train
    # ebg_test_pca = ebg_test

    model = svm.SVC(class_weight='balanced', kernel='linear')
    # model.fit(ebg_train_pca, y_train)
    # model = tree.DecisionTreeClassifier(max_depth=3)
    model = model.fit(ebg_train_pca, y_train)
    y_pred = model.predict(ebg_test_pca)

    return evaluate_metrics(y_pred, y_test)


if __name__ == "__main__":

    shuffle_labels = False
    label_seeds = np.arange(0, 250) if shuffle_labels else np.array([0])

    t_min = 0.1
    t_max = 0.15

    ebg_all, time_vec, fs = load_ebg1_ml(root_path=local_data_path, tmin=-1., tmax=1.)
    fs = fs.astype(float)

    t_min = np.abs(time_vec - t_min).argmin()
    t_max = np.abs(time_vec - t_max).argmin()

    baseline_min = np.abs(time_vec - (-0.5)).argmin()
    baseline_max = np.abs(time_vec - (-0.2)).argmin()

    time_vec = time_vec[t_min:t_max]

    tfr_freqs = np.linspace(20, 100, 800)
    gamma_band = (50, 75)
    # gamma_band = (30, 100)
    f_min = np.abs(tfr_freqs - gamma_band[0]).argmin()
    f_max = np.abs(tfr_freqs - gamma_band[1]).argmin()

    acc = {}
    auc_roc = {}
    conf_mat = {}
    f1 = {}
    class_0_ebg = []
    class_1_ebg = []
    training_kwargs = {
        'n_splits': 5 if not shuffle_labels else 2,
        'n_repeats': 10 if not shuffle_labels else 1,
    }

    for subj in ebg_all.keys():

        print(f"------- Subject {subj} -------")
        ebg_data = ebg_all[subj]['ebg']
        ebg_labels = ebg_all[subj]['label']
        # ebg_data = ebg_data[..., t_min:t_max]

        ebg_data = data_utils.apply_tfr(ebg_data, fs, tfr_freqs, method='dpss')
        # ebg_baseline = np.mean(ebg_data[..., f_min:f_max, baseline_min:baseline_max], axis=(0, -1), keepdims=True)
        ebg_baseline = np.mean(ebg_data[..., f_min:f_max, baseline_min:baseline_max], axis=-1, keepdims=True)
        ebg_data = 10 * np.log10(ebg_data[..., f_min:f_max, t_min:t_max] / ebg_baseline)

        class0 = np.where(ebg_labels == 0.)
        class1 = np.where(ebg_labels == 1.)
        #
        # class1_avg = np.mean(ebg_data[class1, ...].squeeze(), axis=(0, -1), keepdims=True)
        # class0_avg = np.mean(ebg_data[class0, ...].squeeze(), axis=(0, -1), keepdims=True)

        #
        # ebg_data[class1] = 10 * np.log10(ebg_data[class1]/class1_avg)
        # ebg_data[class0] = 10 * np.log10(ebg_data[class0]/class0_avg)

        class_0_ebg.append(np.mean(ebg_data[class0, ...].squeeze(), axis=(0, 1)))
        class_1_ebg.append(np.mean(ebg_data[class1, ...].squeeze(), axis=(0, 1)))

        # average over channels
        ebg_data = ebg_data.mean(axis=1).squeeze()
        # ebg_data = ebg_data[..., np.abs(time_vec - 0.05).argmin():np.abs(time_vec - 0.15).argmin()].mean(axis=-1)
        ebg_data = ebg_data.reshape((len(ebg_data), -1))

        # steps = [('scaler', StandardScaler()), ('lda', LinearDiscriminantAnalysis(n_components=1)), ('svm', svm.SVC(class_weight='balanced', kernel='linear'))]
        # model = Pipeline(steps=steps)
        # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
        # n_scores = cross_val_score(model, ebg_data, ebg_labels, scoring='balanced_accuracy', cv=cv, n_jobs=-1, error_score='raise')
        # print('Balanced Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
        # sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

        for seed_value in label_seeds:
            if len(label_seeds) > 1:
                new_ebg_labels = random.Random(seed_value).sample(list(ebg_labels), len(ebg_labels))
                new_ebg_labels = np.array(new_ebg_labels)
            else:
                new_ebg_labels = ebg_labels
            sss = RepeatedStratifiedKFold(n_splits=training_kwargs['n_splits'],
                                          n_repeats=training_kwargs['n_repeats'], random_state=0)
            for _, (train_index, test_index) in enumerate(sss.split(ebg_data, new_ebg_labels)):
                balanced_acc, confusion_mat, auc, f1_val = \
                    apply_training_pipeline(ebg_data, new_ebg_labels, train_index, test_index)

                if subj not in acc.keys():
                    # acc[subj] = [model.score(ebg_test_pca, y_test)]
                    acc[subj] = [balanced_acc]
                    conf_mat[subj] = np.expand_dims(confusion_mat, axis=0)
                    auc_roc[subj] = [auc]
                    f1[subj] = [f1_val]
                else:
                    # acc[subj].append(model.score(ebg_test_pca, y_test))
                    acc[subj].append(balanced_acc)
                    conf_mat[subj] = np.vstack((conf_mat[subj], np.expand_dims(confusion_mat, axis=0)))
                    auc_roc[subj].append(auc)
                    f1[subj].append(f1_val)

            # print(f'----------- Fold {i} -----------')
        print(f'Balanced Accuracies: {np.mean(acc[subj])}')
        print(f'AUC ROCs: {np.mean(auc_roc[subj])}')
        print(f'F1 Score: {np.mean(f1[subj])}')
        # print(f'Confusion Matrices: \n{conf_mat[subj]}')

    fig, ax = plt.subplots(figsize=(20, 10))
    acc_vals = np.zeros((len(acc['0']), len(acc.keys())))
    for i, key in enumerate(acc.keys()):
        acc_vals[:, i] = acc[key]
    ax.boxplot(acc_vals, labels=acc.keys(), vert=True, patch_artist=True, boxprops=dict(facecolor="blue"))
    ax.set_title(f"f = {f_min}-{f_max} Hz and t = {t_min}-{t_max} and Scale+PCA+SVM")
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Balanced Acc.")
    plt.show()

    # tfr_freqs = tfr_freqs[10:-10]
    # fig, axs = plt.subplots(6, 5, figsize=(20, 15))
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # axs = axs.flatten()
    # for s in range(29):
    #     ebg_0_subj = np.array(class_0_ebg)[s, ...]
    #     ebg_1_subj = np.array(class_1_ebg)[s, ...]
    #     diff = ebg_1_subj - ebg_0_subj
    #     # diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
    #     axs[s].imshow(diff)
    #     axs[s].set_xticks([0, np.abs(time_vec - 0.).argmin(), len(time_vec)],
    #                    [str('{:.2f}'.format(time_vec[0])), '0.00', str('{:.2f}'.format(time_vec[-1]))])
    #     axs[s].set_yticks([0, 20, 40, 60, 80, 100, 120],
    #                    [str('{:.2f}'.format(tfr_freqs[0])),
    #                     str('{:.2f}'.format(tfr_freqs[20])),
    #                     str('{:.2f}'.format(tfr_freqs[40])),
    #                     str('{:.2f}'.format(tfr_freqs[60])),
    #                     str('{:.2f}'.format(tfr_freqs[80])),
    #                     str('{:.2f}'.format(tfr_freqs[100])),
    #                     str('{:.2f}'.format(tfr_freqs[120]))])
    #     axs[s].set_title(f"subject {s}")
    # plt.tight_layout()
    # plt.show()

    # from scipy.ndimage import gaussian_filter
    #
    # class_0_ebg = np.array(class_0_ebg).mean(axis=0)
    # class_1_ebg = np.array(class_1_ebg).mean(axis=0)
    # print(np.abs(time_vec - 0.).argmin())
    # fig, axs = plt.subplots()
    # diff = class_1_ebg-class_0_ebg
    # diff = gaussian_filter(diff, sigma=5)
    # axs.imshow(diff, vmin=0, aspect=0.25, cmap='jet')
    # axs.set_xticks([0, np.abs(time_vec - 0.).argmin(), len(time_vec)],
    #                [str('{:.2f}'.format(time_vec[0])), '0.00', str('{:.2f}'.format(time_vec[-1]))])
    # axs.set_yticks([99, 199, 299, 399, 499, 599, 699],
    #                [# str('{:.2f}'.format(tfr_freqs[0])),
    #                 str('{:.2f}'.format(tfr_freqs[199])),
    #                 str('{:.2f}'.format(tfr_freqs[299])),
    #                 str('{:.2f}'.format(tfr_freqs[399])),
    #                 str('{:.2f}'.format(tfr_freqs[499])),
    #                 str('{:.2f}'.format(tfr_freqs[599])),
    #                str('{:.2f}'.format(tfr_freqs[699])),
    #                str('{:.2f}'.format(tfr_freqs[799]))])
    # plt.show()
