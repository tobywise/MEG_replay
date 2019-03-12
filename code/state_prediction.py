import numpy as np
import copy
from tqdm import tqdm

def predict_states(X, clf, n_stim=8, shifts=(-5, 6), remove=0):

    """

    Args:
        X: MEG data
        clf: Classifier trained on localiser data
        n_stim: Number of states
        shifts: Number of adjacent states to use. Tuple of (previous states, subsequent states)

    Returns:
        Numpy array of state activation probabilities

    """

    n_tp = X.shape[2]  # Number of timepoints
    state_probabilities = np.zeros((X.shape[0], n_tp + shifts[0] - shifts[1] + 1, n_stim))

    # clf = copy.deepcopy(clf)

    # pca = clf.steps[0][1] # get pca
    # clf.steps.pop(0)  # remove pca from classifier
    # clf.steps.pop(0)  # remove timepoint selection from classifier
    # print(clf.steps)

    for i in tqdm(range(X.shape[0])):  # predict on every trial
        trial_X = np.expand_dims(X[i, ...], 0)
        # trial_X = pca.transform(trial_X)
        # exclude first and last few timepoints as we don't have any adjacent data to add as features
        timepoints = []
        for j in range(n_tp)[0 - shifts[0]:n_tp - shifts[1] + 1]:
            tp_X = trial_X[..., j + shifts[0]:j + shifts[1]]
            timepoints.append(tp_X)
        timepoints = np.stack(timepoints).squeeze()
        if timepoints.ndim < 3:
            timepoints = timepoints[..., np.newaxis]
        pred = clf.predict_proba(timepoints)
        state_probabilities[i, :, :] = pred
            # print(tp_X.shape)
            # pred = clf.predict_proba(tp_X)
            # state_probabilities[i, j, :] = pred
    return state_probabilities[..., :n_stim - remove]


