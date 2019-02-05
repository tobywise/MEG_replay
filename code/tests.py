import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'code')
from utils import add_features, select_path, generate_test_matrix
from sequenceness import sequenceness_regression
from glms import get_chosen_unchosen


@pytest.fixture()
def PCA_MEG_data():

    """
    Creates an array a shape similar to what we get from PCA, i.e. 3D (trials X components X timepoints)
    Data is ordered

    Each component is a range of 50 numbers (one per timepoint), starting at the end of the last (i.e. PCA1 = 0-50, PCA2 = 50-100)
    Each trial increases by 1000

    """

    X = np.arange(0, 50)
    X = np.tile(X, (5, 1))
    for i in range(5):
        X[i, :] += i * 50
    X = np.tile(X, (10, 1, 1))
    for i in np.arange(10):
        X[i, ...] += i * 1000

    return X

@pytest.fixture()
def transition_matrix():

    """
    Return a transition matrix
    """

    return np.array([[0., 1., 1., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0.]])

@pytest.fixture()
def subset_transition_matrices():

    transition_matrix = np.array([[0., 1., 1., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0.]])

    matrices = [transition_matrix]

    for n, i in enumerate([5, 6]):
        matrices.append(select_path(transition_matrix, i))

    return matrices

@pytest.fixture()
def state_activation_array_forwards():

    transition_matrix = np.array([[0., 1., 1., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0.]])

    return generate_test_matrix(transition_matrix, [5], lag=6)

@pytest.fixture()
def state_activation_array_backwards():
    transition_matrix = np.array([[0., 1., 1., 0., 0., 0., 0.],
                                  [0., 0., 0., 1., 0., 0., 0.],
                                  [0., 0., 0., 0., 1., 0., 0.],
                                  [0., 0., 0., 0., 0., 1., 0.],
                                  [0., 0., 0., 0., 0., 0., 1.],
                                  [0., 0., 0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0., 0., 0.]])

    return generate_test_matrix(transition_matrix.T, [0], start_state=5, lag=6)



# Add features

def test_add_features_linear_increase(PCA_MEG_data):

    """ Checks that when adding features our timepoints remain in order """

    Xf = add_features(PCA_MEG_data)

    assert np.all(np.diff(Xf[0, :]) == 1)

def test_add_features_trial_order(PCA_MEG_data):

    """ Checks that when adding features our timepoints remain in order """

    Xf = add_features(PCA_MEG_data)

    assert np.all(np.diff(Xf, axis=0) == 1000)

# Transition matrix

def test_select_path(transition_matrix):

    subset_matrix = select_path(transition_matrix, 5)

    true = np.array([[0., 1., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 1., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 1., 0.],
                   [0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0.]])

    assert np.all(subset_matrix - true == 0)



"""
SEQUENCENESS TESTS

These test various permutations of "true" sequenceness (forwards or backwards), the matrix we've used (also forwards/backwards), and the measure we're taking (forwards/backwards/difference)


"""

def test_sequenceness_forward_replay_forward_matrix_correct_path(subset_transition_matrices,
                                                                 state_activation_array_forwards):

    # Should detect forwards replay

    forwards, backwards, difference = sequenceness_regression(state_activation_array_forwards,
                                                              subset_transition_matrices[1], max_lag=40, alpha=True)

    assert np.where(forwards == np.max(forwards))[0] == 5
    assert np.all(np.abs(backwards) < 0.1)
    assert np.where(difference == np.max(difference))[0] == 5


def test_sequenceness_forward_replay_backward_matrix_correct_path(subset_transition_matrices,
                                                                  state_activation_array_forwards):

    # Should detect reverse replay

    forwards, backwards, difference = sequenceness_regression(state_activation_array_forwards,
                                                              subset_transition_matrices[1].T, max_lag=40, alpha=True)

    assert np.all(np.abs(forwards) < 0.1)
    assert np.where(backwards == np.max(backwards))[0] == 5
    assert np.where(difference == np.min(difference))[0] == 5

def test_sequenceness_forward_replay_forward_matrix_incorrect_path(subset_transition_matrices,
                                                                 state_activation_array_forwards):

    # Should detect nothing (wrong path)

    forwards, backwards, difference = sequenceness_regression(state_activation_array_forwards,
                                                              subset_transition_matrices[2], max_lag=40, alpha=True)

    assert np.all(np.abs(forwards) < 0.1)
    assert np.all(np.abs(backwards) < 0.1)
    assert np.all(np.abs(difference) < 0.1)


def test_sequenceness_backward_replay_forward_matrix_correct_path(subset_transition_matrices,
                                                                  state_activation_array_backwards):

    # Should detect reverse replay

    forwards, backwards, difference = sequenceness_regression(state_activation_array_backwards,
                                                              subset_transition_matrices[1], max_lag=40, alpha=True)

    assert np.all(np.abs(forwards) < 0.1)
    assert np.where(backwards == np.max(backwards))[0] == 5
    assert np.where(difference == np.min(difference))[0] == 5


def test_sequenceness_backward_replay_backward_matrix_correct_path(subset_transition_matrices,
                                                                   state_activation_array_backwards):

    # Should detect forwards replay

    forwards, backwards, difference = sequenceness_regression(state_activation_array_backwards,
                                                              subset_transition_matrices[1].T, max_lag=40, alpha=True)

    assert np.where(forwards == np.max(forwards))[0] == 5
    assert np.all(np.abs(backwards) < 0.1)
    assert np.where(difference == np.max(difference))[0] == 5


def test_sequenceness_backward_replay_forward_matrix_incorrect_path(subset_transition_matrices,
                                                                    state_activation_array_backwards):

    # Should detect nothing (wrong path)

    forwards, backwards, difference = sequenceness_regression(state_activation_array_backwards,
                                                              subset_transition_matrices[2], max_lag=40, alpha=True)

    assert np.all(np.abs(forwards) < 0.1)
    assert np.all(np.abs(backwards) < 0.1)
    assert np.all(np.abs(difference) < 0.1)


""""
REGRESSIONS
"""

@pytest.fixture()
def sequenceness_array_different_intensity_each_arm():

    seq = np.zeros(40)
    seq[4] = -0.1
    seq[5] = -0.2
    seq[6] = -0.1
    seq[7] = -0.05

    sequenceness = np.tile(seq, (6, 3, 1)).transpose(0, 2, 1)
    sequenceness[..., 2] *= 3

    return sequenceness

@pytest.fixture()
def sequenceness_array_different_arm_intensity_each_trial():

    seq = np.zeros(40)
    seq[4] = -0.1
    seq[5] = -0.2
    seq[6] = -0.1
    seq[7] = -0.05

    sequenceness = np.tile(seq, (6, 3, 1)).transpose(0, 2, 1)
    sequenceness[::2, :, 1] *= 3
    sequenceness[1::2, :, 2] *= 3

    return sequenceness

@pytest.fixture()
def behaviour_df():

    df = pd.DataFrame(dict(trial_number=np.arange(6),
                           next_move=[0, 1, 0, 1, 0, 1]))
    return df


def test_chosen_unchosen_separation(sequenceness_array_different_arm_intensity_each_trial, behaviour_df):

    chosen_unchosen_sequenceness = get_chosen_unchosen(sequenceness_array_different_arm_intensity_each_trial,
                                                       behaviour_df)

    assert np.sum(chosen_unchosen_sequenceness[..., 1].mean(axis=0) - chosen_unchosen_sequenceness[0, :, 1]) == 0 and \
           np.sum(chosen_unchosen_sequenceness[..., 2].mean(axis=0) - chosen_unchosen_sequenceness[0, :, 2])















































