from mne.io import read_raw_ctf
import mne
import matplotlib.pyplot as plt
from mne.preprocessing import ICA, create_eog_epochs
from scipy.stats import ttest_1samp
import numpy as np
import os
import pandas as pd

"""
PREPROCESSING
"""

subject_dir = r'/Users/dancingdiva12/Desktop/MEG Data'
behavioural_dir = r'/Users/dancingdiva12/Desktop/Behavioral Data'
subject_id = '5379_M009'
behavioural_id = 'M009'

# Get behavioural 
behaviour_file = [i for i in os.listdir(os.path.join(behavioural_dir, behavioural_id)) if 'csv' in i][0]

# This code will find any .ds directories in a folder and put them into a list
# Make sure you have one folder per subject containing all their .ds files
data_dir = os.path.join(subject_dir, subject_id)
data = os.listdir(data_dir)
data = sorted([i for i in data if '.ds' in i])

# This then loops through these files, reads them, and adds the raw data to a list before concatenating the runs
raws = []
for i in range(len(data)):
    raws.append(read_raw_ctf(os.path.join(data_dir, data[i]), preload=True))

raw = mne.concatenate_raws(raws)  # Raw is the full task raw data
del raws  # delete the individual raw data files to save memory

# Find events based on triggers
events = mne.find_events(raw, stim_channel='UPPT001', shortest_event=0)
events[:, 0] += 20  # adjust for presentation latency

# Remove the CTF compensation
raw.apply_gradient_compensation(0)

# Select MEG channels
raw = raw.pick_types('mag')
raw.pick_channels(raw.info['ch_names'][29:])

# Do some other cleaning
mf_kwargs = dict(origin=(0., 0., 0.), st_duration=10.)
raw = mne.preprocessing.maxwell_filter(raw, **mf_kwargs)

# Plot the raw data - this produces an interactive plot
raw.plot()

# bad channels
raw.info['bads'] = []

# Filter the data
raw.filter(1, 30, method='fir', fir_design='firwin')  # Filter is set to 1-30hz

# We can plot the power spectrum to see the effect of the filter
raw.plot_psd(area_mode='range', tmax=10.0,  average=False)


# Set epoch rejection criteria - this can be changed depending on how stringent you want to be
reject = dict(mag=5e-9)

# Split up into epochs - we're interested in events 40, 8, and 50
epochs = mne.Epochs(raw, events, tmin=-0.1, tmax=4.01, preload=True,
                              event_id=[40, 8, 50],
                              reject=reject)

# Downsample to 100 Hz
epochs.resample(100, npad="auto")

# Run ICA
ica = ICA(n_components=0.95, method='fastica',
          random_state=0, max_iter=100).fit(epochs, decim=1, reject=reject)

# Plot components
ica.plot_components()

# You can also plot the timecourses of the components - the component numbers you want to see are provided as a list
ica.plot_sources(epochs, range(0, ica.n_components_))

# And various other things like the power spectrum - the list given to the picks argument here should contain the components you want to look at
ica.plot_properties(epochs, picks=[0])

# Remove components
ica.exclude = [0]  # ICA components to remove
ica.apply(epochs)  # Remove these components

# Save the data (MNE likes epoch files to end in -epo)
epochs.save(os.path.join(data_dir, 'epoched_data-epo.fif.gz'))


"""
ERP analysis
"""

# Read in the epoched data if you need to
epochs = mne.read_epochs(os.path.join(data_dir, 'epoched_data-epo.fif.gz'))

behaviour = pd.read_csv(os.path.join(behavioural_dir, behavioural_id, behaviour_file))
tt = behaviour['trial_type'].values # trial type
#
# if len(epochs['40']) + len(epochs['8']) < len(behaviour):
#     raise ValueError("SUBJECT IS MISSING EVENTS - LEAVE THIS SUBJECT UNTIL TOBY FIXES THINGS")

# Remove extra events - hopefully this will work...
if len(epochs['40']) + len(epochs['8']) > len(behaviour):

    count = 0

    for i in range(len(epochs)):

        if i < len(epochs):

            if epochs.events[i, 2] == 8:
                if tt[count] != 0:
                    print tt[count]
                    print i
                    epochs.drop(i)
                    count += 1
                else:
                    # print tt[count]
                    count += 1

            elif epochs.events[i, 2] == 40:
                # print tt[count]
                count += 1

else:

    drops = []

    temp_epochs = epochs[['40', '8']].events

    for n, trial in enumerate(tt):
        n -= len(drops)
        if (temp_epochs[n, 2] == 8 and trial == 1) or (temp_epochs[n, 2] == 40 and trial == 0):
            print "Dropping {0}".format(n)
            drops.append(n)
            behaviour = behaviour.drop(n)


# Get outcome events, 8 = normal outcomes (when the final state is displayed), 40 = outcome only warnings (we'll
# get the outcomes by starting after 2 seconds), 50 = shock outcomes
normal_outcomes = epochs['8']
outcome_only_outcomes = epochs['40']
shock_outcomes = epochs['50']

# Crop the epochs so that they start 0.1 seconds before onset and end after 1 second
normal_outcomes = normal_outcomes.crop(-0.1, 1)
outcome_only_shock_outcomes = outcome_only_outcomes.copy().crop(2.9, 4)
outcome_only_outcomes = outcome_only_outcomes.crop(1.9, 3)
shock_outcomes = shock_outcomes.crop(-0.1, 0.8)

#normal_outcomes = normal_outcomes.crop(-0.1, 1)
#outcome_only_outcomes = outcome_only_outcomes.crop(1.9, 3)
#outcome_only_shock_outcomes = outcome_only_outcomes.crop(2.9, 4)
#shock_outcomes = shock_outcomes.crop(-0.1, 0.8)

# Adjust the times for the outcome only outcome
outcome_only_outcomes.times = outcome_only_outcomes.times - 2

# Combine all outcome trials
outcomes = mne.concatenate_epochs([normal_outcomes, outcome_only_outcomes]).pick_types('mag')

# Save these for use later
outcomes.save(os.path.join(data_dir, 'outcomes-epo.fif.gz'))
shock_outcomes.save(os.path.join(data_dir, 'shock_outcomes-epo.fif.gz'))

"""
Look at some ERFs!
"""

shock_outcomes = mne.read_epochs(os.path.join(data_dir, 'shock_outcomes-epo.fif.gz'))

# Get the evoked response for shock outcomes
shock_evoked = shock_outcomes.average()
shock_evoked.plot_joint(ts_args=dict(time_unit='s'), topomap_args=dict(time_unit='s'))
plt.savefig('/Users/dancingdiva12/Desktop/UCL/Research Project/thesis/figures/{0}_shocks.png'.format(subject_id))

# If you want to look at power


# from mne.time_frequency import tfr_morlet
# freqs = np.logspace(*np.log10([6, 35]), num=8)
# n_cycles = freqs / 2.  # different number of cycle per frequency
#
# power, itc = tfr_morlet(shock_outcomes, freqs=freqs, n_cycles=n_cycles, use_fft=True,
#                         return_itc=True, decim=3, n_jobs=1)
# power.plot_topo(baseline=(-0.5, 0), mode='logratio', title='Average power')


########################################################

# And the same for reward outcomes
reward_evoked = outcomes.average()
reward_evoked.plot_joint(ts_args=dict(time_unit='s'), topomap_args=dict(time_unit='s'))
plt.savefig('/Users/dancingdiva12/Desktop/UCL/Research Project/thesis/figures/{0}_rewards.png'.format(subject_id))


# Load behaviour data and recode things
behaviour.reset_index()
trial_info = pd.read_csv('task/Task_information/trial_info.csv')
behaviour = pd.merge(behaviour, trial_info, on='trial_number')

behaviour['State_3'][behaviour['Reward_received'].isnull()] = behaviour.end_state
sel = behaviour['Reward_received'].isnull()

behaviour['reward'] = np.nan
behaviour['reward'][behaviour['State_3'] == 7] = behaviour['State_1_reward'][behaviour['State_3'] == 7]
behaviour['reward'][behaviour['State_3'] == 8] = behaviour['State_2_reward'][behaviour['State_3'] == 8]
behaviour['reward'][behaviour['State_3'] == 9] = behaviour['State_3_reward'][behaviour['State_3'] == 9]
behaviour['reward'][behaviour['State_3'] == 10] = behaviour['State_4_reward'][behaviour['State_3'] == 10]
behaviour['reward'][~sel] = behaviour['Reward_received']

behaviour['shock'] = np.nan
behaviour['shock'][behaviour['State_3'] == 7] = behaviour['0_shock'][behaviour['State_3'] == 7]
behaviour['shock'][behaviour['State_3'] == 8] = behaviour['1_shock'][behaviour['State_3'] == 8]
behaviour['shock'][behaviour['State_3'] == 9] = behaviour['2_shock'][behaviour['State_3'] == 9]
behaviour['shock'][behaviour['State_3'] == 10] = behaviour['3_shock'][behaviour['State_3'] == 10]
behaviour['shock'][~sel] = behaviour['Shock_received']

print "Number of shocks"
behaviour.shock.sum()
print "Total reward"
behaviour.reward.sum()

# Drop the first trial
behaviour = behaviour[1:]
outcomes.drop(0)

# Assign behavioural data to metadata attribute
outcomes.metadata = behaviour.assign(Intercept=1)

# Linear regression
# The regression has two terms - the intercept (not interesting) and the reward level (interesting)

from mne.stats import linear_regression, fdr_correction
names = ["Intercept", 'reward']
res = linear_regression(outcomes, outcomes.metadata[names], names=names)
# Create an "evoked" object for the reward predictor - this represents the beta value for reward level in the model
# across time points and gradiometers
reward_evoked = res["reward"].beta
reward_evoked.plot_joint(ts_args=dict(time_unit='s'), topomap_args=dict(time_unit='s'))
plt.savefig('/Users/dancingdiva12/Desktop/UCL/Research Project/thesis/figures/{0}_evoked_rewards.png'.format(subject_id))


# You can choose which time points the plot shows scalp plots for using the times argument
reward_evoked.plot_joint(ts_args=dict(time_unit='s'), topomap_args=dict(time_unit='s'), times=[.09, .45])

shock_evoked.save(os.path.join(data_dir, 'shock_evoked-ave.fif.gz'))
reward_evoked.save(os.path.join(data_dir, 'reward_evoked-ave.fif.gz'))

"""
Group analysis
"""

# SHOCK

evokeds = []

for root, dir, files in os.walk(subject_dir):
    for f in files:
        if 'ave' in f and 'shock' in f:
            evokeds.append(os.path.join(root, f))

evokeds = [mne.read_evokeds(i)[0] for i in evokeds] * 2
#evokeds = mne.combine_evoked(evokeds, 'equal')

grand_average = mne.combine_evoked(evokeds, 'equal')

grand_average.plot_joint(ts_args=dict(time_unit='s'), topomap_args=dict(time_unit='s'))

subject_data = [np.array([i.data.T]) for i in evokeds]
subject_data = np.concatenate(subject_data)


###########################
# OLD CONSERVATIVE METHOD #
###########################

ps = ttest_1samp(subject_data, 0, axis=0)[1]
reject_H0, fdr_pvals = fdr_correction(ps)
grand_average.plot_image(mask=reject_H0, time_unit='s')

############################
# LESS CONSERVATIVE METHOD #
############################

# Get connectivity matrix (represents spatial relationship between sensors)
connectivity, ch_names = mne.channels.find_ch_connectivity(evokeds[0].info, ch_type='mag')

# Remove missing channels
idx = []
for n, i in enumerate(ch_names):
    if i + '-2910' in evokeds[0].info['ch_names']:
        idx.append(n)
mask = np.zeros(connectivity.shape[0], dtype=bool)
mask[idx] = True
connectivity = connectivity[mask, :]
connectivity = connectivity[:, mask]

# HERE ARE TWO METHDOS OF THRESHOLDING

# METHOD A
# Relatively quick - try this first

T_obs, clusters, cluster_p_values, H0 = \
    mne.stats.spatio_temporal_cluster_1samp_test(subject_data, n_permutations=1000, threshold=None, n_jobs=4,
                                             connectivity=connectivity)

# METHOD B
# Can be very slow
threshold_tfce = dict(start=.2, step=.2)
T_obs, clusters, cluster_p_values, H0 = \
    mne.stats.spatio_temporal_cluster_1samp_test(subject_data, n_permutations=1000, threshold=threshold_tfce, n_jobs=4,
                                             connectivity=connectivity)


# Plot significant clusters
good_cluster_inds = []
mask = np.zeros_like(T_obs, dtype=bool)
for n, (c, p_val) in enumerate(zip(clusters, cluster_p_values)):
    if p_val <= 0.05:
        mask[c] = True
        good_cluster_inds.append(n)

grand_average.plot_image(mask=mask.T, time_unit='s')

# PLOT RESPONSES AT SENSORS CONTRIBUTING TO EACH CLUSTER (if there are any clusters...)
# This is all taken from this example https://mne-tools.github.io/dev/auto_tutorials/plot_stats_spatio_temporal_cluster_sensors.html

from mne.viz import plot_topomap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mne.viz import plot_compare_evokeds

pos = mne.find_layout(grand_average.info).pos[idx, :]

# loop over clusters
for i_clu, clu_idx in enumerate(good_cluster_inds):
    # unpack cluster information, get unique indices
    time_inds, space_inds = np.squeeze(clusters[clu_idx])
    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)

    # get topography for F stat
    f_map = T_obs[time_inds, ...].mean(axis=0)

    # get signals at the sensors contributing to the cluster
    sig_times = grand_average.times[time_inds]

    # create spatial mask
    mask = np.zeros((f_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True

    # initialize figure
    fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))

    # plot average test statistic and mark significant sensors
    image, _ = plot_topomap(f_map, pos, mask=mask, axes=ax_topo, cmap='coolwarm',
                            vmin=np.min, vmax=np.max, show=False)

    # create additional axes (for ERF and colorbar)
    divider = make_axes_locatable(ax_topo)

    # add axes for colorbar
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_topo.set_xlabel(
        'Averaged T-map ({:0.3f} - {:0.3f} s)'.format(*sig_times[[0, -1]]))

    # add new axis for time courses and plot time courses
    ax_signals = divider.append_axes('right', size='300%', pad=1.2)
    title = 'Cluster #{0}, {1} sensor'.format(i_clu + 1, len(ch_inds))
    if len(ch_inds) > 1:
        title += "s (mean)"
    # plot_compare_evokeds(grand_average, title=title, picks=ch_inds, axes=ax_signals,
    #                      show=True, truncate_yaxis='max_ticks')
    grand_average.plot(picks=ch_inds, axes=ax_signals, show=False)

    # plot temporal cluster extent
    ymin, ymax = ax_signals.get_ylim()
    ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                             color='orange', alpha=0.3)

    # clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=.05)
    plt.show()


############################################################################################################
############################################################################################################

##########
# REWARD #
##########

evokeds = []

for root, dir, files in os.walk(subject_dir):
    for f in files:
        if 'ave' in f and 'reward' in f:
            evokeds.append(os.path.join(root, f))

evokeds = [mne.read_evokeds(i)[0] for i in evokeds]

grand_average = mne.combine_evoked(evokeds, 'equal')

grand_average.plot_joint(ts_args=dict(time_unit='s'), topomap_args=dict(time_unit='s'))


subject_data = [np.array([i.data.T]) for i in evokeds]
subject_data = np.concatenate(subject_data)


###########################
# OLD CONSERVATIVE METHOD #
###########################

ps = ttest_1samp(subject_data, 0, axis=0)[1]
reject_H0, fdr_pvals = fdr_correction(ps)
grand_average.plot_image(mask=reject_H0, time_unit='s')

############################
# LESS CONSERVATIVE METHOD #
############################

# Get connectivity matrix (represents spatial relationship between sensors)
connectivity, ch_names = mne.channels.find_ch_connectivity(evokeds[0].info, ch_type='mag')

# Remove missing channels
idx = []
for n, i in enumerate(ch_names):
    if i + '-2910' in evokeds[0].info['ch_names']:
        idx.append(n)
mask = np.zeros(connectivity.shape[0], dtype=bool)
mask[idx] = True
connectivity = connectivity[mask, :]
connectivity = connectivity[:, mask]

# THRESHOLDING
T_obs, clusters, cluster_p_values, H0 = \
    mne.stats.spatio_temporal_cluster_1samp_test(subject_data, n_permutations=1000, threshold=None, n_jobs=4,
                                             connectivity=connectivity)

# Plot significant clusters
good_cluster_inds = []
mask = np.zeros_like(T_obs, dtype=bool)
for n, (c, p_val) in enumerate(zip(clusters, cluster_p_values)):
    if p_val <= 0.05:
        mask[c] = True
        good_cluster_inds.append(n)

grand_average.plot_image(mask=mask.T, time_unit='s')

# PLOT RESPONSES AT SENSORS CONTRIBUTING TO EACH CLUSTER (if there are any clusters...)
# This is all taken from this example https://mne-tools.github.io/dev/auto_tutorials/plot_stats_spatio_temporal_cluster_sensors.html

from mne.viz import plot_topomap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mne.viz import plot_compare_evokeds

pos = mne.find_layout(grand_average.info).pos[idx, :]

# loop over clusters
for i_clu, clu_idx in enumerate(good_cluster_inds):
    # unpack cluster information, get unique indices
    time_inds, space_inds = np.squeeze(clusters[clu_idx])
    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)

    # get topography for F stat
    f_map = T_obs[time_inds, ...].mean(axis=0)

    # get signals at the sensors contributing to the cluster
    sig_times = grand_average.times[time_inds]

    # create spatial mask
    mask = np.zeros((f_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True

    # initialize figure
    fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))

    # plot average test statistic and mark significant sensors
    image, _ = plot_topomap(f_map, pos, mask=mask, axes=ax_topo, cmap='coolwarm',
                            vmin=np.min, vmax=np.max, show=False)

    # create additional axes (for ERF and colorbar)
    divider = make_axes_locatable(ax_topo)

    # add axes for colorbar
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_topo.set_xlabel(
        'Averaged T-map ({:0.3f} - {:0.3f} s)'.format(*sig_times[[0, -1]]))

    # add new axis for time courses and plot time courses
    ax_signals = divider.append_axes('right', size='300%', pad=1.2)
    title = 'Cluster #{0}, {1} sensor'.format(i_clu + 1, len(ch_inds))
    if len(ch_inds) > 1:
        title += "s (mean)"
    # plot_compare_evokeds(grand_average, title=title, picks=ch_inds, axes=ax_signals,
    #                      show=True, truncate_yaxis='max_ticks')
    grand_average.plot(picks=ch_inds, axes=ax_signals, show=False)

    # plot temporal cluster extent
    ymin, ymax = ax_signals.get_ylim()
    ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                             color='orange', alpha=0.3)

    # clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=.05)
    plt.show()




