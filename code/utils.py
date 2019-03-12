import networkx as nx
import numpy as np
from scipy.stats import norm
import seaborn as sns
from scipy.stats import gaussian_kde
from sequenceness import plot_matrices
import matplotlib.pyplot as plt

def select_timepoints(X, idx=33, embedding=10):
    idx = int(idx)
    shifts = (int(0 - embedding / 2), int(embedding / 2 + 1))
    return X[..., idx + shifts[0]:idx + shifts[1]] 


def add_features(X):
    return X.reshape(X.shape[0], -1)

def select_path(transition_matrix, outcome_state):

    G = nx.DiGraph(transition_matrix)

    path = nx.shortest_path(G, 0, outcome_state)

    m = transition_matrix.copy()

    for i in range(len(m)):
        if i not in path:
            m[i, :] = 0
            m[:, i] = 0

    return m

def path_to_matrix(path, nstates):

    matrix = np.zeros((nstates, nstates))

    for i in range(len(path) - 1):
        matrix[path[i], path[i+1]] = 1

    return matrix

from sklearn.model_selection import RandomizedSearchCV


def generate_test_matrix(transition_matrix, start_end, lag=5, ntimepoints=1000, gap_mean=5, gap_sd=1,
                         plot=False, all_state_reactivation=0., plot_paths=False, activation_width=5, noise=0,
                         other_state_activation=0, noise_states=None):

    """
    Creates an array of state reactivations given a transition matrix and paths within that matrix.

    Args:
        transition_matrix: A numpy array defining a transition matrix
        start_end: List of tuples representing start and end states for each path
        lag: Gap between reactivations within a sequence
        ntimepoints: Number of timepoints to include in the output array
        gap_mean: Mean of the gaussian distribution used to generate gaps between replay events
        gap_sd: SD of the gap distribution
        plot: If true, plots a heatmap showing the generated data
    
    Returns:
        [type]: [description]
    """

    nstates = transition_matrix.shape[0]

    G = nx.Graph(transition_matrix)

    rv = norm(loc=activation_width/2, scale=activation_width/4)

    X = np.zeros((ntimepoints * 2, nstates))

    paths = []
    for s, e in start_end:
        paths.append(nx.shortest_path(G, s, e))
    paths = np.array(paths)

    # Convert paths for matrices
    path_matrices = []
    for p in paths:
        path_matrices.append(path_to_matrix(p, nstates))

    if plot_paths:
        plot_matrices(path_matrices)
    
    max_seq_length = np.max(paths.shape)
    n_replay_events = int(ntimepoints / (lag * max_seq_length + gap_mean))

    for e in range(n_replay_events):
        gap = np.random.normal(gap_mean, gap_sd, ntimepoints).astype(int)  # Gap between replay events
        for n, s in enumerate(paths[np.random.choice(range(len(paths)))]):
            X[(lag * max_seq_length + gap[e]) * e + lag * n:
              (lag * max_seq_length + gap[e]) * e + lag * n + lag, s] = \
                rv.pdf(np.arange(lag)) 
            # X[(lag * max_seq_length + gap[e]) * e + lag * n:
            #   (lag * max_seq_length + gap[e]) * e + lag * n + lag, :] = \
            #     rv.pdf(np.arange(lag)) #* all_state_reactivation

    X = X[:ntimepoints, :]

    if noise_states is None:
        noise_states = np.arange(X.shape[1])

    if noise is not 0:
        for n, i in enumerate(noise_states):
            activations = np.zeros(X.shape[0])
            if isinstance(noise, list):
                state_noise = noise[n]
            else:
                state_noise = noise
            activations[np.random.randint(0, X.shape[0], int(X.shape[0] * state_noise))] = 1
            activations = np.convolve(activations, rv.pdf(np.arange(lag)))[:X.shape[0]]
            X[:, i] += activations

    if other_state_activation != 0:
        for i in range(X.shape[0]):
            X[i, X[i, :] == 0] = X[i, X[i, :] != i].mean() * other_state_activation
            # if i < 5:
            #     print(X[i, X[i, :] != i].mean())

    # X = (X + X.min()) / X.max()
    X = (X + X.min())
    X /= X.max()

    return X

    if plot:
        plt.figure()
        sns.heatmap(X)



class PositionDecoder(object):

    # THIS MIGHT ALL BE UNNECESSARY - COULD PROBABLY JUST USE REACTIVATION MATRIX

    def __init__(self, y_pred, y):
        self.y_pred = y_pred
        self.y = y


    def decode_position(self, reactivations):

        # One trial of reactivations, shape (n_timepoints, n_states)
        location_prediction = np.empty(reactivations.shape)

        # Loop through time points
        for i in range(reactivations.shape[0]):

            # Probability of state occupancy
            pX = np.ones(reactivations.shape[1])  # Uniform distribution

            # Probability of reactivations given state
            pNc_X = np.zeros(reactivations.shape[1])
            for j in range(reactivations.shape[1]):
                kde = gaussian_kde(self.y_pred[self.y == j, j])  # J prediction probabilities on trials where we were at position J
                pNc_X[j] = kde.pdf(reactivations[i, j])  # Probability of the current state J prediction given these values

            pX_nc = pNc_X * pX  # TODO what type of multiplication?



