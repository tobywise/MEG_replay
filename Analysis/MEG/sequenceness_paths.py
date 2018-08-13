import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
import os
import re
from scipy.stats import linregress
from sklearn.preprocessing import scale

# TODO look at "preactivation" of next state during state walk through

def cross_correlation(X_data, transition_matrix, maxlag=60, shuffles=0):
    diff_array = []
    ff_array = []
    fb_array = []

    X_data = X_data.copy()

    for i in range(np.max([1, shuffles])):

        if shuffles > 0:
            print "Shuffling data"
            X_data = X_data.T
            np.random.shuffle(X_data)
            X_data = X_data.T

        X_dataf = np.matmul(X_data, transition_matrix)
        X_datar = np.matmul(X_data, transition_matrix.T)

        forward_rs = []
        backward_rs = []

        ff = []
        fb = []
        diffs = []

        for lag in range(maxlag):

            forward_corr_array = []

            for i in range(X_data.shape[1]):
                r = np.corrcoef(X_data[lag:, i], np.roll(X_dataf[lag:, i], lag))[0][1]
                if np.isnan(r):
                    r = 0
                forward_corr_array.append(r)

            forward_mean_corr = np.mean(forward_corr_array)

            backward_corr_array = []

            for i in range(X_data.shape[1]):
                r = np.corrcoef(X_data[lag:, i], np.roll(X_datar[lag:, i], lag))[0][1]
                if np.isnan(r):
                    r = 0
                backward_corr_array.append(r)

            backward_mean_corr = np.mean(backward_corr_array)

            forward_rs.append(forward_mean_corr)
            backward_rs.append(backward_mean_corr)

            diff = np.array(forward_corr_array) - np.array(backward_corr_array)
            diffs.append(np.mean(diff))
            ff.append(np.mean(np.array(forward_corr_array)))
            fb.append(np.mean(np.array(backward_corr_array)))

        diffs = np.array(diffs)
        diff_array.append(diffs)
        ff = np.array(ff)
        ff_array.append(ff)
        fb = np.array(fb)
        fb_array.append(fb)

    if len(diff_array) == 1:
        diff_array = diff_array[0]
        ff_array = ff_array[0]
        fb_array = fb_array[0]
    else:
        diff_array = np.vstack(diff_array)
        ff_array = np.vstack(ff_array)
        fb_array = np.vstack(fb_array)

    return ff_array, fb_array, diff_array


from scipy.linalg import toeplitz, pinv
from sklearn.preprocessing import minmax_scale

from numba import jit


@jit
def betas_func(max_lag, nstates, dm, y, alpha=True):
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


def sequenceness_regression(X_data, transition_matrix, max_lag=10, nstates=11, alpha=True):
    # start = time()

    X = X_data.copy()
    nbins = max_lag + 1

    dm = toeplitz(X[:, 0], np.zeros((nbins, 1)))
    dm = dm[:, 1:]

    for kk in range(1, nstates):
        temp = toeplitz(X[:, kk], np.zeros((nbins, 1)))
        temp = temp[:, 1:]
        dm = np.hstack([dm, temp])

    dm = minmax_scale(dm, axis=0)

    y = X

    betas = betas_func(max_lag, nstates, dm, y, alpha=alpha)

    betasr = betas.reshape((max_lag, nstates, nstates), order='F')
    betasnbins64 = betas.reshape((max_lag, nstates ** 2), order='F')

    T1 = transition_matrix
    T2 = transition_matrix.T

    CC = np.ones((nstates, nstates))
    II = np.eye(nstates)

    bbb = np.matmul(pinv(np.vstack([T1.flatten(order='F'), T2.flatten(order='F'),
                                    II.flatten(order='F'), CC.flatten(order='F')]).T), betasnbins64.T)

    fb = bbb[0, :]
    bb = bbb[1, :]

    return fb, bb, (fb - bb)

import networkx as nx

def select_path(transition_matrix, outcome_state):

    G = nx.DiGraph(transition_matrix)

    path = nx.shortest_path(G, 0, outcome_state)

    m = transition_matrix.copy()

    for i in range(len(m)):
        if i not in path:
            m[i, :] = 0
            m[:, i] = 0

    return m

#####
max_lag = 40

transition_matrix = np.loadtxt(r'C:\Users\Toby\OneDrive - University College London\MEG'
                               r'\task\Task_information\transition_matrix.txt')

matrices = []

for n, i in enumerate([7, 8, 9, 10]):
    matrices.append(select_path(transition_matrix, i))

from scipy.stats import zscore
from time import time

# EVERY SUBJECT
data_dir = 'C:\Users\Toby\OneDrive - University College London\MEG\Analysis\MEG\detected_states/'
behav_dir = r'C:\Users\Toby\OneDrive - University College London\MEG\task\Data\behavioural\pilot\scan_ids'

files = os.listdir(data_dir)

rest_files = [f for f in files if 'MG0' in f and 'plann' in f][:12]

trial_replay = []
mean_replay = np.empty((len(rest_files), max_lag, 4))

import mne
aa = mne.read_epochs(r'C:\Users\Toby\Desktop\post_ICA_task-epo.fif.gz')

for n, f in enumerate(rest_files[:]):

    sub = re.search('MG.+(?=_plann)', f).group()
    print "Subject {0}".format(sub)

    rest_preds = np.load(os.path.join(data_dir, f))

    sequenceness = np.empty((len(rest_preds), max_lag, 4))

    for trial_number, i in enumerate(rest_preds):
        if not trial_number % 10:
            print "Trial {0} of {1}".format(trial_number, len(rest_preds))
        for nn, matrix in enumerate(matrices):
            _, _, sequenceness[trial_number, :, nn] = sequenceness_regression(i[00:600, :11], matrix,
                                                                       max_lag=max_lag, alpha=True, nstates=11)


    trial_replay.append(sequenceness)
    for j in range(4):
        mean_replay[n, :, j] = sequenceness[:, :, j].mean(axis=0)

plt.figure()
for i in range(4):
    plt.plot(mean_replay[..., i].mean(axis=0))
plt.tight_layout()

trial_info = pd.read_csv('task/Task_information/trial_info.csv')

sub_trial_replay_A = np.stack([i[:99] for i in trial_replay['A']], axis=2)
sub_trial_replay_B = np.stack([i[:99] for i in trial_replay['B']], axis=2)

# Equalish replay early on
plt.plot(sub_trial_replay_A[10:30].mean(axis=0).mean(axis=1))
plt.plot(sub_trial_replay_B[10:30].mean(axis=0).mean(axis=1))

# Less equal later on
plt.plot(sub_trial_replay_A[80:].mean(axis=0).mean(axis=1))
plt.plot(sub_trial_replay_B[80:].mean(axis=0).mean(axis=1))

# TODO look at actual path taken and deal with outcome only trials

import statsmodels.api as sm

shock_data = {'condition': [], 'replay': []}

group_slopes_shock = []
group_slopes_shock2 = []
group_slopes_shock_null = []
group_slopes_reward = []

replay_chosen_means = []
replay_unchosen_means = []
replay_unchosen2_means = []

for n, i in enumerate(trial_replay):
    print n

    try:

        replay_chosen = []
        replay_unchosen = []  # other branch
        replay_unchosen2 = []  # other state, same branch

        dropped = pd.read_csv(
            os.path.join(data_dir, rest_files[n].replace('planning_states.npy', 'task_dropped_trials.csv')))
        dropped = dropped[dropped.event_id == 30]
        dropped = dropped[dropped.event_id == 60]

        sub = re.search('MG.+(?=_plann)', rest_files[n]).group()
        behavioural = pd.read_csv(os.path.join(behav_dir, '{0}_behavioural.csv'.format(sub)))
        behavioural = behavioural.drop(dropped.trial.values)
        behavioural.reset_index()

        behavioural = pd.merge(behavioural, trial_info, on='trial_number')

        behavioural.loc[behavioural['Reward_received'].isnull(), 'State_3'] = behavioural.end_state
        sel = behavioural['Reward_received'].isnull()

        behavioural['reward'] = np.nan

        behavioural.loc[behavioural['State_3'] == 7, 'reward'] = behavioural['State_1_reward'][
            behavioural['State_3'] == 7]
        behavioural.loc[behavioural['State_3'] == 8, 'reward'] = behavioural['State_2_reward'][
            behavioural['State_3'] == 8]
        behavioural.loc[behavioural['State_3'] == 9, 'reward'] = behavioural['State_3_reward'][
            behavioural['State_3'] == 9]
        behavioural.loc[behavioural['State_3'] == 10, 'reward'] = behavioural['State_4_reward'][
            behavioural['State_3'] == 10]
        behavioural.loc[~sel, 'reward'] = behavioural['Reward_received']

        behavioural['shock'] = np.nan
        behavioural.loc[behavioural['State_3'] == 7, 'shock'] = behavioural['State_1_shock'][
            behavioural['State_3'] == 7]
        behavioural.loc[behavioural['State_3'] == 8, 'shock'] = behavioural['State_2_shock'][
            behavioural['State_3'] == 8]
        behavioural.loc[behavioural['State_3'] == 9, 'shock'] = behavioural['State_3_shock'][
            behavioural['State_3'] == 9]
        behavioural.loc[behavioural['State_3'] == 10, 'shock'] = behavioural['State_4_shock'][
            behavioural['State_3'] == 10]
        behavioural.loc[~sel, 'shock'] = behavioural['Shock_received']
        behavioural['State_3'] -= 7

        outcome_only = np.where(behavioural.trial_type_x)[0]
        # outcome_only = np.arange(0, 200)

        behavioural = behavioural[~behavioural.trial_type_x.astype(bool)]

        for j in range(len(behavioural))[:]:
            if j in np.where(behavioural.shock < 2)[0]:
                state = int(behavioural.State_3.values[j])
                replay_chosen.append(i[j, :, state])
                if state < 2:
                    ru = i[j, :, 2:]
                    if len(ru.shape) > 1:
                        ru = ru.mean(axis=1)
                    ru2 = i[j, :, 1 - state]
                    if len(ru2.shape) > 1:
                        ru2 = ru2.mean(axis=1)
                    replay_unchosen.append(ru)
                    replay_unchosen2.append(ru2)
                else:
                    ru = i[j, :, 2:]
                    if len(ru.shape) > 1:
                        ru = ru.mean(axis=1)
                    ru2 = i[j, :, [k for k in [2, 3] if k != state]]
                    if len(ru2.shape) > 1 and not any([v == 1 for v in ru2.shape]):
                        ru2 = ru2.mean(axis=1)
                    replay_unchosen.append(ru)
                    replay_unchosen2.append(ru2)

        # plt.figure()
        # plt.plot(replay_chosen)

        replay_chosen = np.vstack(replay_chosen)
        replay_unchosen = np.vstack(replay_unchosen)
        replay_unchosen2 = np.vstack(replay_unchosen2)

        replay_chosen_means.append(replay_chosen.mean(axis=0))
        replay_unchosen_means.append(replay_unchosen.mean(axis=0))
        replay_unchosen2_means.append(replay_unchosen2.mean(axis=0))


        # Effect of shock
        slopes_shock = np.zeros(replay_chosen.shape[1])
        slopes_shock2 = np.zeros(replay_unchosen2.shape[1])
        # slopes_shock_null = np.zeros((i.shape[1], 100))
        slopes_reward = np.zeros(replay_unchosen.shape[1])

        for t in range(0, replay_chosen.shape[1]):
            if not t % 10:
                print t
            # slopes_shock[t] = linregress(scale(behavioural['pe_abs'][5:100]),
            #                              scale(replay_chosen[5:100, t])).slope
            # slopes_reward[t] = linregress(scale(behavioural['pe_abs'][5:100]),
            #                              scale(replay_unchosen[5:100, t])).slope
            X = scale(behavioural[:][behavioural.trial_type_x <2][['reward', 'shock']])
            X = sm.add_constant(X)
            y = scale(replay_chosen[:100, t])
            m = sm.OLS(y, X).fit()
            slopes_shock[t] = m.params[2]

            y = scale(replay_unchosen[:100, t])
            m = sm.OLS(y, X).fit()
            slopes_reward[t] = m.params[2]

            y = scale(replay_unchosen2[:100, t])
            m = sm.OLS(y, X).fit()
            slopes_shock2[t] = m.params[2]


        #
        #     # permute
        #     for p in range(50):
        #         shuffled_trials = behavioural['State_3'].values.copy()
        #         np.random.shuffle(shuffled_trials)
        #         slopes_shock_null[t, p] = linregress(scale(shuffled_trials[5:]),
        #                                              scale(diff_replay[5:, t])).slope
        #
        #
        group_slopes_shock.append(slopes_shock)
        group_slopes_reward.append(slopes_reward)
        group_slopes_shock2.append(slopes_shock2)
        # group_slopes_shock_null.append(slopes_shock_null)

    except Exception as e:
        print e
        print "ARGH"


plt.plot(np.vstack(replay_chosen_means).mean(axis=0), label='Chosen')
plt.plot(np.vstack(replay_unchosen_means).mean(axis=0), label='Unchosen, other branch')
plt.plot(np.vstack(replay_unchosen2_means).mean(axis=0), label='Unchosen, same branch')
plt.legend()
plt.tight_layout()

group_slopes_shock = np.vstack(group_slopes_shock)
group_slopes_shock_mean = group_slopes_shock.mean(axis=0)
group_slopes_shock_sd = group_slopes_shock.std(axis=0)

# plt.plot(group_slopes_shock.mean(axis=0))

# null_slopes = np.vstack([k.mean(axis=1) for k in group_slopes_shock_null])
# null_slopes_mean = null_slopes.mean(axis=0)
# null_slopes_sd = null_slopes.std(axis=0)

# PLOT
plt.figure()
# plt.fill_between(range(60), null_slopes_mean - null_slopes_sd, null_slopes_mean + null_slopes_sd, alpha=0.2)
# plt.plot(range(60), null_slopes_mean, label='Permuted data')
# plt.fill_between(range(60), group_slopes_shock_mean - group_slopes_shock_sd, group_slopes_shock_mean + group_slopes_shock_sd, alpha=0.2)
plt.plot(range(max_lag), group_slopes_shock_mean, label='Real data')
plt.legend()
plt.title("Effect of shock on replay")
plt.ylabel(r'$\beta$')
plt.xlabel(r"Lag (ms)")
plt.xticks(np.arange(0, max_lag, 10), np.arange(0, max_lag, 10) * 10)
plt.tight_layout()


plt.figure()
plt.plot(group_slopes_shock[:, :].T)


##

group_slopes_reward = np.vstack(group_slopes_reward)
group_slopes_reward_mean = group_slopes_reward.mean(axis=0)

group_slopes_shock = np.vstack(group_slopes_shock)
group_slopes_shock_mean = group_slopes_shock.mean(axis=0)

group_slopes_shock2 = np.vstack(group_slopes_shock2)
group_slopes_shock2_mean = group_slopes_shock2.mean(axis=0)


plt.figure()
plt.plot(range(max_lag), group_slopes_shock_mean, label='Chosen sequence')
plt.plot(range(max_lag), group_slopes_reward_mean, label='Unchosen sequence')
plt.plot(range(max_lag), group_slopes_shock2_mean, label='Unchosen sequence, same branch')
plt.legend()
plt.title("Effect of shock on replay")
plt.ylabel(r'$\beta$')
plt.xlabel(r"Lag (ms)")
plt.xticks(np.arange(0, max_lag, 10), np.arange(0, max_lag, 10) * 10)
plt.tight_layout()

