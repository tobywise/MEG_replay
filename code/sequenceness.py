from numba import jit
import numpy as np
from scipy.linalg import toeplitz, pinv
from sklearn.preprocessing import minmax_scale
from itertools import permutations
import networkx as nx
from tqdm import tqdm
from sympy.utilities.iterables import multiset_permutations

@jit
def betas_func(max_lag, nstates, dm, y, alpha=True):
    """
    Fits GLM and gives beta values for regressors

    Args:
        max_lag: Maximum time lag to include (in samples)
        nstates: Number of states
        dm: State reactivation matrix with something done to it
        y: State reactivation matrix
        alpha: Filter alpha oscillations

    Returns:
        Betas from GLM

    """

    betas = np.empty((nstates * max_lag, nstates))
    betas[:] = np.nan

    bins = 10

    if alpha:

        for ilag in range(0, bins):
            temp_zinds = np.arange(1, nstates * max_lag, bins) + ilag - 1
            temp = np.matmul(np.linalg.pinv(np.hstack([dm[:, temp_zinds], np.ones((dm.shape[0], 1))])), y)
            betas[temp_zinds, :] = temp[:-1, :]

    else:

        for ilag in range(0, max_lag):
            zinds = np.arange(0, nstates * max_lag, max_lag) + ilag - 1
            temp = np.matmul(np.linalg.pinv(np.hstack([dm[:, zinds], np.ones((dm.shape[0], 1))])), y)
            betas[zinds, :] = temp[:-1, :]

    return betas


def test_transitions(betas, transition_matrix, max_lag=40, constant=False):

    nstates = transition_matrix.shape[0]

    # betasr = betas.reshape((max_lag, nstates, nstates), order='F')
    betasnbins64 = betas.reshape((max_lag, nstates ** 2), order='F')

    T1 = transition_matrix
    T2 = transition_matrix.T

    CC = np.ones((nstates, nstates))
    II = np.eye(nstates)

    if constant:
        bbb = np.matmul(pinv(np.vstack([T1.flatten(order='F'), T2.flatten(order='F'),
                                        II.flatten(order='F'), CC.flatten(order='F')]).T), betasnbins64.T)
    else:
        bbb = np.matmul(pinv(np.vstack([T1.flatten(order='F'), T2.flatten(order='F'),
                                        II.flatten(order='F')]).T), betasnbins64.T)       

    fb = bbb[0, :]
    bb = bbb[1, :]

    return fb, bb, (fb - bb)

def lagged_GLM(X_data, max_lag=40, nstates=7, alpha=True):

    X = X_data.copy()
    nbins = max_lag + 1

    dm = toeplitz(X[:, 0], np.zeros((nbins, 1)))
    dm = dm[:, 1:]

    for kk in range(1, nstates):
        temp = toeplitz(X[:, kk], np.zeros((nbins, 1)))
        temp = temp[:, 1:]
        dm = np.hstack([dm, temp])

    dm = minmax_scale(dm, axis=0)  # Scale the matrix

    y = minmax_scale(X, axis=0)  # Scale the reactivation array - probably not necessary

    betas = betas_func(max_lag, nstates, dm, y, alpha=alpha)

    return betas

def check_descendents_and_identity(G, permuted):
    for i in range(permuted.shape[0]):
        desc = nx.descendants(G, i)
        if (desc and np.any(permuted[i, tuple(desc)])) or permuted[i, i]:
            return False
    return True

def check_permuted_matrix(true, permuted, matrices):
   
    # for m in [true, true.T]:
    #     G = nx.DiGraph(m)
    #     if not check_descendents_and_identity(G, permuted):
    #         return False
    #     elif not check_descendents_and_identity(G, permuted.T):
    #         return False

    if len(matrices):
        matrices = np.array(matrices)
        diff = np.abs(matrices.reshape((matrices.shape[0], matrices.shape[1] ** 2)) - permuted.reshape((permuted.shape[0] ** 2)))
        if np.any((np.mean(diff, axis=1) == 0)):
            return False

    
    return True

def generate_permuted_matrices(transition_matrix, n_permutations=100):
    
    permuted_matrices = []

    valid = np.ones_like(transition_matrix)

    for m in [transition_matrix, transition_matrix.T]:
        G = nx.DiGraph(m)
        for i in range(transition_matrix.shape[0]):
            desc = nx.descendants(G, i)
            valid[i, list(desc)] = 0
    valid -= np.eye(valid.shape[0])

    new = np.zeros(valid.sum().astype(int))
    new[:transition_matrix.sum().astype(int)] = 1

    mp = multiset_permutations(new)

    pbar = tqdm()

    for new_m in mp:
        if len(permuted_matrices) < n_permutations:
            m = np.zeros_like(transition_matrix)
            m[valid == 1] = new_m
            pbar.update(1)
            if check_permuted_matrix(transition_matrix, m, permuted_matrices):
                permuted_matrices.append(m)

    pbar.close()

    return permuted_matrices


class StateReactivation(object):

    def __init__(self, reactivation_array):
        
        self.reactivation_array = reactivation_array

    def sequenceness_regression(self, X_data, transition_matrix, max_lag=10, nstates=7, alpha=True, remove_first=False, constant=False):

        """
        Generates sequenceness measures using GLM

        Args:
            X_data: State probability matrix
            transition_matrix: Transition matrix
            max_lag: Maximum time lag to include (in samples)
            nstates: Number of states
            alpha: Filter out alpha oscillations
            remove_first: Removes the first state
            constant: Adds a constant term to the GLM

        Returns:
            Forward sequenceness, reverse sequenceness, subtraction

        """

        # Remove first row of the transition matrix
        if remove_first:
            transition_matrix[0, :] = 0

        betas = lagged_GLM(X_data, max_lag, nstates, alpha=alpha)

        fb, bb, difference = test_transitions(betas, transition_matrix, max_lag=max_lag, constant=constant)

        return fb, bb, difference, betas

    def null_sequenceness_regression(self, betas, matrix, max_lag, constant):

        fb, bb, difference = test_transitions(betas, matrix, max_lag=max_lag, constant=constant)

        return fb, bb, difference

    def get_sequenceness(self, max_lag, matrices, alpha=True, remove_first=False, constant=False, permuted_matrices=()):

        """
        Gets forwards, backwards, and reverse sequenceness on each trial

        Args:
            max_lag: Maximum lag to test
            matrices: List of transition matrices
            alpha: If true, removes alpha-related oscillations
            remove_first: Removes the first state from the transition matrix # TODO also remove from the state reactivation array?
            constant: Add a constant to the regression
        
        Returns:
            Sequenceness dictionary, keys = forwards, backwards, difference, values = array of shape (n_trials, n_lags, n_matrices)
        """

        n_permutations = len(permuted_matrices)

        sequenceness_null = None

        forwards, backwards, difference = [np.empty((self.reactivation_array.shape[0], max_lag, len(matrices))) for i in range (3)]
        if n_permutations:
            forwards_null, backwards_null, difference_null = [np.empty((self.reactivation_array.shape[0], max_lag, n_permutations)) for i in range (3)]

        # Loop over matrices and trials
        for n, matrix in enumerate(matrices):
            print("Testing matrix {0}".format(n+1))

            # Generate permuted matrices
            for trial in tqdm(range(self.reactivation_array.shape[0])):
                # Get sequenceness for real matrix (and its reverse)
                forwards[trial, :, n], backwards[trial, :, n], \
                difference[trial, :, n], betas = self.sequenceness_regression(self.reactivation_array[trial, ...],
                                                 matrix, max_lag, alpha=alpha, remove_first=remove_first, constant=constant)
        
        # Test permuted matrices
        print("Testing permuted matrices")
        for trial in tqdm(range(self.reactivation_array.shape[0])):
            for nperm, p in enumerate(permuted_matrices):
                forwards_null[trial, :, nperm], backwards_null[trial, :, nperm], \
                difference_null[trial, :, nperm] = self.null_sequenceness_regression(betas, p, max_lag=max_lag, constant=constant)

        # Put the results into a dictionary
        sequenceness = dict(forwards=forwards, backwards=backwards, difference=difference)
        if n_permutations:
            sequenceness_null = dict(forwards=forwards_null, backwards=backwards_null, difference=difference_null)

        return sequenceness, sequenceness_null





