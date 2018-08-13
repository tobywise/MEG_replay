import matplotlib as mpl
mpl.use('Agg')

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from mne.decoding import (SlidingEstimator, cross_val_multiscore)
import mne
import matplotlib.pyplot as plt
import numpy as np
from mne.decoding import UnsupervisedSpatialFilter
import pandas as pd
from joypy import joyplot
from matplotlib import cm
from bayes_opt import BayesianOptimization
import argparse
import os
os.environ['OMP_NUM_THREADS'] = '1'
# DATA / FEATURE AUGMENTATION FUNCTIONS


def augment_samples(X_data, shifts=(), y_data=None):

    """
    Adds samples from timepoints either side of the timepoint of interest as targets.

    E.g. For timepoints [1, 2, 3], if interested in 2, 1 and 3 will be relabelled as 2 and added to the target array

    Args:
        X_data: MEG data
        shifts: Timepoints to include, must include zero
        y_data: Target data

    Returns:
        Augmented X and y arrays

    """

    shifted_data = []
    shifted_ys = []
    if len(shifts):
        for i in shifts:
            shifted_data.append(np.roll(X_data, i, 2))
            if y_data is not None:
                shifted_ys.append(y_data)
        X_data = np.vstack(shifted_data)

    if y_data is not None and len(shifts):
        y_data = np.hstack(shifted_ys)

    return X_data, y_data

def add_features(X_data, shifts, y_data=None):

    """
    Adds timepoints either side of current timepoint as features for current timepoint

    Args:
        X_data: MEG data
        shifts: Timepoints to include, must include zero
        y_data: Target data

    Returns:
        Augmented X array, unmodified y array

    """

    shifted_data = []
    for i in shifts:
        shifted_data.append(np.roll(X_data, i, 2))
    X_data = np.hstack(shifted_data)

    return X_data, y_data

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def plot_state_detection(state_probabilities, start=0, end=100, **kwargs):

    state_probabilities = pd.DataFrame(state_probabilities[start:end, :11])  # convert to dataframe and remove null

    joyplot(state_probabilities, kind='values', x_range=range(end-start), colormap=cm.cool,
            title='State reactivation probability', alpha=1,
            **kwargs)
    plt.xlabel("Timepoint")

def exclude_eyes(X_data, y_data, threshold=1.0, timepoints=30, pupil_diff_thresh=0.1):

    for i in [302, 303]:

        sds = np.mean(X_data[:, i, 10:timepoints + 10], axis=1)
        X_data = X_data[sds < threshold, ...]
        y_data = y_data[sds < threshold]

    mean_pupil_response = X_data[:, 304, :].mean(axis=0)
    diff = np.subtract(X_data[:, 304, :], mean_pupil_response)

    X_data = X_data[np.abs(diff.mean(axis=1)) < pupil_diff_thresh, ...]
    y_data = y_data[np.abs(diff.mean(axis=1)) < pupil_diff_thresh]

    return X_data, y_data

###############################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("subject")
    parser.add_argument("n_iter", type=int)
    args = parser.parse_args()

    shifts = np.arange(-5, 6)
    np.random.seed(100)

    # LOAD EPOCHS
    subjectID = args.subject
    epochs = mne.read_epochs(r'/data/twise/{0}/post_ICA_localiser-epo.fif.gz'.format(subjectID))

    # ar = AutoReject(n_jobs=4)
    # epochs_clean = ar.fit_transform(epochs.pick_types(meg=True))
    epochs_clean = epochs

    drop_idx = [n for n, i in enumerate(epochs_clean.drop_log) if len(i)]

    print("LOADED DATA")

    #####################################################################################
    # RUN CLASSIFICATION WITH BASIC SETTINGS TO FIND BEST TIME POINT FOR CLASSIFICATION #
    #####################################################################################

    # Get epoch data
    X_raw = epochs_clean.get_data()[:, :, :]  # MEG signals: n_epochs, n_channels, n_times (exclude non MEG channels)
    y_raw = epochs_clean.events[:, 2]  # Get event types
    # y_raw = np.array([i for n, i in enumerate(y_raw) if n not in drop_idx])

    # select events and time period of interest
    event_selector = (y_raw < 23) | (y_raw == 99)
    X_raw = X_raw[event_selector, ...]
    y_raw = y_raw[event_selector]
    X_raw = X_raw[:, 29:302, :]

    # print("Number of unique events = {0}\n\nEvent types = {1}".format(len(np.unique(y_raw)),
    #                                                                   np.unique(y_raw))

    # Do PCA with 50 components
    pca = UnsupervisedSpatialFilter(PCA(50), average=False)
    pca_data = pca.fit_transform(X_raw)
    X_raw = pca_data

    # CLASSIFIER
    # Logistic regression with L2 penalty, multi-class classification performed as one-vs-rest
    # Data is transformed to have zero mean and unit variance before being passed to the classifier

    clf = make_pipeline(StandardScaler(), LogisticRegression(multi_class='multinomial', C=0.1, penalty='l2',
                                                             solver='saga', tol=0.01))

    # Try classifying at all time points with 5 fold CV
    time_decod = SlidingEstimator(clf, n_jobs=1, scoring='accuracy')
    scores = cross_val_multiscore(time_decod, X_raw, y_raw,
                                  cv=5, n_jobs=1)

    # Mean scores across cross-validation splits
    mean_scores = np.mean(scores, axis=0)
    best_idx = np.where(mean_scores == mean_scores.max())[0][0]
    best_idx = 33

    if best_idx < 5: best_idx = 5

    print("Best classification at index {0}, {1}ms".format(best_idx, (best_idx * 10) - 133))

    # Plot
    # fig, ax = plt.subplots()
    # # ax.plot(range(10), mean_scores, label='Score')
    # ax.axhline(1 / 12., color='#a8a8a8', linestyle='--', label='Chance')
    # ax.set_xlabel('Times')
    # ax.set_ylabel('Subset accuracy')
    # ax.axvline(.0, color='#515151', linestyle='-')
    # ax.set_title('Decoding accuracy')
    # ax.plot(epochs.times[:len(mean_scores)], mean_scores, label='Score')
    # ax.axvline(epochs.times[best_idx], color='#76b9e8', linestyle='--')
    #
    # ax.legend()
    # plt.tight_layout()

    ###############################################################################
    # OPTIMISE HYPERPARAMETERS USING 3 FOLD CV WITH DATA AND FEATURE AUGMENTATION #
    ###############################################################################

    # Get epoch data
    X_raw = epochs_clean.get_data()[:, :, :]  # MEG signals: n_epochs, n_channels, n_times (exclude non MEG channels)
    # X_raw = np.hstack(
    #     [X_raw, epochs.get_data()[np.array([i for i in range(X_raw.shape[0]) if i not in drop_idx]), 302:305, :]])
    y_raw = epochs_clean.events[:, 2]  # Get event types

    # select events and time period of interest
    event_selector = (y_raw < 23) | (y_raw == 99)
    X_raw = X_raw[event_selector, ...]
    y_raw = y_raw[event_selector]
    X_raw = X_raw[:, 29:302, :]

    print("Number of unique events = {0}\n\nEvent types = {1}".format(len(np.unique(y_raw)),
                                                                      np.unique(y_raw)))


    def cvlr(C, n_features, pupil_diff_thresh=None, threshold=None, X_raw=X_raw, y_raw=y_raw):
        n_features = int(n_features)
        # X, y = exclude_eyes(X_raw, y_raw, threshold=threshold, pupil_diff_thresh=pupil_diff_thresh, timepoints=10)
        X, y = (X_raw, y_raw)
        X = X[:, 29:302, best_idx - 5:best_idx + 5]
        if X.shape[0] > 200:

            pca = UnsupervisedSpatialFilter(PCA(n_features), average=False)
            pca_data = pca.fit_transform(X)
            # X_raw = pca_data

            clf = make_pipeline(StandardScaler(), LogisticRegression(multi_class='multinomial', C=C, penalty='l2',
                                                                     solver='saga', tol=0.01))
            # clf = make_pipeline(StandardScaler(), LogisticRegression(multi_class='ovr', C=C, penalty='l2', tol=0.01))

            cv = KFold(3)  # CV
            shifts = np.arange(-4, 5)  # Additional timepoints to use as features
            # shifts = [0]
            accuracy = []

            for n, (train_index, test_index) in enumerate(cv.split(pca_data[..., 0])):
                print("Fold {0} / 3".format(n + 1))

                # Add features + samples to X/y training data and test data
                X_train, y_train = add_features(pca_data[train_index, :, :], shifts, y[train_index])
                X_test, y_test = add_features(pca_data[test_index, :, :], shifts, y[test_index])

                # Add samples to training data
                # X_train, y_train = augment_samples(X_train, shifts, y_train)

                # Fit the classifier to training data and predict on held out data
                clf.fit(X_train[..., 5], y_train)  # X represents timepoints 5 either side of the best index
                y_pred = clf.predict(X_test[..., 5])

                accuracy.append(recall_score(y_test, y_pred, average='macro'))

            acc = np.mean(accuracy)

        else:

            acc = 0

        return acc


    gp_params = {"alpha": 1e-5}

    svcBO = BayesianOptimization(cvlr, {'C': (0.001, 50), 'n_features': (10, 150)})
    # svcBO.explore({'C': [0.001, 0.01, 0.1]})

    svcBO.maximize(n_iter=5)
    print('-' * 53)
    print('Final Results')
    print('SVC: %f' % svcBO.res['max']['max_val'])
    print(svcBO.res['max']['max_params'])


    # CONFUSION MATRIX - 5 FOLD CV WITH DATA AND FEATURE AUGMENTATION

    # This is done in a weird way because we're adding neighbouring timepoints as additional samples and we need to avoid
    # Mixing training and testing data. We split into CV folds before adding samples

    # Select data at timepoint with best classification accuracy (plus neighbouring timepoints)

    # USE OPTIMISED VALUES

    pca = UnsupervisedSpatialFilter(PCA(int(svcBO.res['max']['max_params']['n_features'])), average=False)
    print(X_raw.shape)
    pca_data = pca.fit_transform(X_raw)

    # CLASSIFIER
    # Logistic regression with L2 penalty, multi-class classification performed as one-vs-rest
    # Data is transformed to have zero mean and unit variance before being passed to the classifier

    clf = make_pipeline(StandardScaler(), LogisticRegression(multi_class='multinomial',
                                                             C=svcBO.res['max']['max_params']['C'], penalty='l2',
                                                             solver='saga', tol=0.01))

    X_raw = pca_data[..., best_idx - 10:best_idx + 10]

    cv = KFold(5)  # CV
    shifts = np.arange(-5, 6)  # Additional timepoints to use as features
    # shifts = [0]

    conf_matrices = []
    accuracy = []
    ts_scores = []

    for n, (train_index, test_index) in enumerate(cv.split(X_raw[..., 0])):
        print("Fold {0} / 5".format(n + 1))

        # Add features + samples to X/y training data and test data
        X_train, y_train = add_features(X_raw[train_index, :, :], shifts, y_raw[train_index])
        X_test, y_test = add_features(X_raw[test_index, :, :], shifts, y_raw[test_index])

        # Add samples to training data
        # X_train, y_train = augment_samples(X_train, np.arange(-2, 2), y_train)

        # Fit the classifier to training data and predict on held out data
        clf.fit(X_train[..., 5], y_train)  # X represents timepoints 5 either side of the best index
        y_pred = clf.predict(X_test[..., 5])

        # Produce confusion matrix
        conf_mat = confusion_matrix(y_test, y_pred)
        accuracy.append(accuracy_score(y_test, y_pred))

        # Report accuracy on this fold
        print("Accuracy = {0}%".format(accuracy[n]))

        # Add normalised matrix to list
        conf_matrices.append(conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis])

    # Get mean confusion matrix
    mean_conf_mat = np.dstack(conf_matrices).mean(2)

    print("Mean accuracy = {0}%".format(np.mean(accuracy)))

    # Plot mean confusion matrix
    plt.figure()
    plot_confusion_matrix(mean_conf_mat, title='Normalised confusion matrix')
    plt.savefig(r'/data/twise/{0}/confusion_matrix.pdf'.format(subjectID))


    # DETECT STATES IN REST DATA

    task_data = r'/data/twise/{0}/post_ICA_task-epo.fif.gz'.format(subjectID)
    task_epochs = mne.read_epochs(task_data)
    print(task_epochs)

    # Get epoch data
    task_X_raw = task_epochs.get_data()[:, 29:302, :]  # MEG signals: n_epochs, n_channels, n_times
    task_y_raw = task_epochs.events[:, 2]  # Get event types

    # select events and time period of interest - 60 = planning, 30 = rest
    planning_X = task_X_raw[task_y_raw == 60, :, :]
    rest_X = task_X_raw[task_y_raw == 30, :, :]

    print("Number of planning trials = {0}\n" \
          "Number of rest trials = {1}".format(planning_X.shape[0], rest_X.shape[0]))

    # PCA on planning and rest data
    print(rest_X.shape)
    rest_X = pca.transform(rest_X)


    def augmentation(X, y, samples=False):

        X_augmented, y_augmented = add_features(X, shifts, y)

        if samples:
            # X_augmented, y_augmented = augment_samples(X_augmented, np.arange(-1, 1), y_augmented)
            X_augmented, y_augmented = augment_samples(X_augmented, [0], y_augmented)

        return X_augmented, y_augmented


    ## AUGMENTATION
    planning_X, planning_y = augmentation(planning_X, None)  # planning
    rest_X, rest_y = augmentation(rest_X, None)  # rest
    X_augmented, y_augmented = augmentation(X_raw, y_raw, samples=True)

    # FIT CLASSIFIER ON FULL LOCALISER DATA

    clf_full = clf.fit(X_augmented[..., 5], y_augmented)

    # PREDICT ON PLANNING AND REST
    planning_preds = []
    rest_preds = []

    for i in range(planning_X.shape[0]):  # predict on every trial
        planning_pred = clf_full.predict_proba(planning_X[i, ...].T)
        planning_preds.append(planning_pred)

    for i in range(rest_X.shape[0]):  # predict on every trial
        rest_pred = clf_full.predict_proba(rest_X[i, ...].T)
        rest_preds.append(rest_pred)

    np.save(os.path.join('/data/twise/{0}/'.format(subjectID), '{0}_rest_states'.format(subjectID)), rest_preds)
    np.save(os.path.join('/data/twise/{0}/'.format(subjectID), '{0}_planning_states'.format(subjectID)), planning_preds)
