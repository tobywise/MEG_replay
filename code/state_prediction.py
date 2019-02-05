import numpy as np
import copy

def predict_states(X, clf, n_stim=8, shifts=(-5, 6)):

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
    state_probabilities = np.zeros((X.shape[0], n_tp, n_stim))

    for i in range(X.shape[0]):  # predict on every trial
        trial_X = np.expand_dims(X[i, ...], 0)
        # exclude first and last few timepoints as we don't have any adjacent data to add as features
        for j in range(n_tp)[0 - shifts[0]:n_tp - shifts[1] + 1]:
            tp_X = trial_X[..., j + shifts[0]:j + shifts[1]]
            pred = clf.predict_proba(tp_X)
            state_probabilities[i, j, :] = pred
    return state_probabilities[:, 0 - shifts[0]:n_tp - shifts[1] + 1, :n_stim - 1]


# class StateArray(np.ndarray):
#
#