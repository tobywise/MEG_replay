from mne.io import read_raw_ctf
import mne
import matplotlib.pyplot as plt
from mne.preprocessing import ICA, create_eog_epochs
import numpy as np
import os


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

import pandas as pd
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
behavioural = pd.read_csv(r'/Users/dancingdiva12/Desktop/Behavioral Data')
behavioural.reset_index()
trial_info = pd.read_csv('task/Task_information/trial_info.csv')
behavioural = pd.merge(behavioural, trial_info, on='trial_number')

behavioural['State_3'][behavioural['Reward_received'].isnull()] = behavioural.end_state
sel = behavioural['Reward_received'].isnull()

behavioural['reward'] = np.nan
behavioural['reward'][behavioural['State_3'] == 7] = behavioural['State_1_reward'][behavioural['State_3'] == 7]
behavioural['reward'][behavioural['State_3'] == 8] = behavioural['State_2_reward'][behavioural['State_3'] == 8]
behavioural['reward'][behavioural['State_3'] == 9] = behavioural['State_3_reward'][behavioural['State_3'] == 9]
behavioural['reward'][behavioural['State_3'] == 10] = behavioural['State_4_reward'][behavioural['State_3'] == 10]
behavioural['reward'][~sel] = behavioural['Reward_received']

behavioural['shock'] = np.nan
behavioural['shock'][behavioural['State_3'] == 7] = behavioural['0_shock'][behavioural['State_3'] == 7]
behavioural['shock'][behavioural['State_3'] == 8] = behavioural['1_shock'][behavioural['State_3'] == 8]
behavioural['shock'][behavioural['State_3'] == 9] = behavioural['2_shock'][behavioural['State_3'] == 9]
behavioural['shock'][behavioural['State_3'] == 10] = behavioural['3_shock'][behavioural['State_3'] == 10]
behavioural['shock'][~sel] = behavioural['Shock_received']

# Drop the first trial
behavioural = behavioural[1:]

# Assign behavioural data to metadata attribute
outcomes.metadata = behaviour.assign(Intercept=1)[1:]

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

evokeds = [mne.read_evokeds(i)[0] for i in evokeds]
#evokeds = mne.combine_evoked(evokeds, 'equal')

grand_average = mne.combine_evoked(evokeds, 'equal')

grand_average.plot_joint(ts_args=dict(time_unit='s'), topomap_args=dict(time_unit='s'))

subject_data = [i.data for i in evokeds]
subject_data = np.dstack(subject_data)

from scipy.stats import ttest_1samp
ps = ttest_1samp(subject_data, 0, axis=2)[1]

reject_H0, fdr_pvals = fdr_correction(ps)

grand_average.plot_image(mask=reject_H0, time_unit='s')

# REWARD

evokeds = []

for root, dir, files in os.walk(subject_dir):
    for f in files:
        if 'ave' in f and 'reward' in f:
            evokeds.append(os.path.join(root, f))

evokeds = [mne.read_evokeds(i)[0] for i in evokeds]

grand_average = mne.combine_evoked(evokeds, 'equal')

grand_average.plot_joint(ts_args=dict(time_unit='s'), topomap_args=dict(time_unit='s'))

subject_data = [i.data for i in evokeds]
subject_data = np.dstack(subject_data)

from scipy.stats import ttest_1samp
ps = ttest_1samp(subject_data, 0, axis=2)[1]

reject_H0, fdr_pvals = fdr_correction(ps)

grand_average.plot_image(mask=reject_H0, time_unit='s')


ev = events[:, 2]

ev = ev[(ev == 30) | (ev == 50)]


shock_list = []

for i in range(1, len(ev)):
    if ev[i] == 30 and ev[i-1] != 50:
        shock_list.append(0)
    elif ev[i] == 30 and ev[i-1] == 50:
        shock_list.append(1)


plt.scatter(range(100), behaviour.shock)
plt.scatter(range(len(shock_list)), shock_list)





