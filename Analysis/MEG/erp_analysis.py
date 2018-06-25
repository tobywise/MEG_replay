from mne.io import read_raw_ctf
import mne
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from mne.preprocessing import ICA, create_eog_epochs
import numpy as np
import os
import argparse

"""
PREPROCESSING
"""

# This code will find any .ds directories in a folder and put them into a list
data_dir = 'path/to/meg'
data = os.listdir(data_dir)
data = sorted([i for i in data if '.ds' in i])

# This then loops through these files, reads them, and adds the raw data to a list before concatenating the runs
raws = []
for i in range(len(data)):
    raws.append(read_raw_ctf(os.path.join(data_dir, data[i]), preload=True))

raw = mne.concatenate_raws(raws)  # Raw is the full task raw data

# Remove the CTF compensation
raw.apply_gradient_compensation(0)

# Do some other cleaning
mf_kwargs = dict(origin=(0., 0., 0.), st_duration=10.)
raw = mne.preprocessing.maxwell_filter(raw, **mf_kwargs)

# Plot the raw data
raw.plot()

# Manually set EOG channels
raw.set_channel_types({'UADC001-2910': 'eog', 'UADC002-2901': 'eog', 'UADC003-2901': 'eog'})

# Filter the data
raw.filter(1, 30, method='fir', fir_design='firwin')  # Filter is set to 1-30hz

# Find events based on triggers
events = mne.find_events(raw, stim_channel='UPPT001', shortest_event=1)

# Set epoch rejection criteria - this can be changed depending on how stringent you want to be
reject = dict(mag=5e-9, eog=20)

# Split up into epochs
epochs = mne.Epochs(raw, events, tmin=-0.1, tmax=0.8, preload=True,
                              event_id=[40, 8],
                              reject=reject)


# Downsample to 100 Hz
epochs = epochs.copy().resample(100, npad='auto')

# Run ICA
ica = ICA(n_components=0.95, method='fastica',
          random_state=0, max_iter=100).fit(raw, decim=1, reject=reject)

# Plot components
ica.plot_components()

# ICA results can be saved if needed
ica.save(os.path.join(data_dir, 'meg-ica.fif.gz'))

# Detect eye movement related components by correlation
eog_inds, scores = ica.find_bads_eog(raw)
print(eog_inds)

ica.exclude = [0, 1, 2]  # ICA components to remove

ica.apply(epochs)  # Remove these components

# Save the data (MNE likes epoch files to end in -epo)
epochs.save('epoched_data-epo.fif.gz')



