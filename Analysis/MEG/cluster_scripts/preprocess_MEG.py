# -*- coding: utf-8 -*-
import matplotlib as mpl
mpl.use('Agg')

from mne.io import read_raw_ctf
import mne
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from mne.preprocessing import ICA, create_eog_epochs
import numpy as np
import os
import argparse

data_dir = r'C:\Users\Toby\Desktop\meg_18_06_18'

def preproc_meg(data_dir):

    data = os.listdir(data_dir)
    data = sorted([i for i in data if '.ds' in i])

    localiser_raws = []
    task_raws = []

    localiser_idx = range(0, 6)

    for i in localiser_idx:
        localiser_raws.append(read_raw_ctf(os.path.join(data_dir, data[i]), preload=True))

    task_idx = range(6, 11)

    for i in task_idx:
        task_raws.append(read_raw_ctf(os.path.join(data_dir, data[i]), preload=True))

    print(localiser_raws)
    print(task_raws)
    localiser_raw = mne.concatenate_raws(localiser_raws)
    task_raw = mne.concatenate_raws(task_raws)

    localiser_raw.apply_gradient_compensation(0)
    task_raw.apply_gradient_compensation(0)

    mf_kwargs = dict(origin=(0., 0., 0.), st_duration=10.)
    localiser_raw = mne.preprocessing.maxwell_filter(localiser_raw, **mf_kwargs)
    localiser_raw.plot()
    plt.savefig(os.path.join(data_dir, 'localiser_raw.pdf'))

    task_raw = mne.preprocessing.maxwell_filter(task_raw, **mf_kwargs)
    task_raw.plot()
    plt.savefig(os.path.join(data_dir, 'task_raw.pdf'))

    localiser_raw.set_channel_types({'UADC001-2910': 'eog', 'UADC002-2901': 'eog', 'UADC003-2901': 'eog'})
    task_raw.set_channel_types({'UADC001-2910': 'eog', 'UADC002-2901': 'eog', 'UADC003-2901': 'eog'})

    # localiser_raws.info['bads'] = ['MLT54']

    # picks = mne.pick_types(localiser_raw.info, meg='mag', eog=True)
    # REJECT BAD CHANNELS
    # eyes = mne.pick_channels(raw.info['ch_names'], ['UADC001-2910', 'UADC002-2901'])

    print("FILTERING")
    localiser_raw.filter(0.5, 45, method='fir', fir_design='firwin')
    task_raw.filter(0.5, 45, method='fir', fir_design='firwin')

    print("FINDING EVENTS")
    localiser_events = mne.find_events(localiser_raw, stim_channel='UPPT001', shortest_event=1)
    task_events = mne.find_events(task_raw, stim_channel='UPPT001', shortest_event=1)
    reject = dict(mag=5e-9, eog=20)

    print("EPOCHING")
    localiser_epochs = mne.Epochs(localiser_raw, localiser_events, tmin=-0.1, tmax=0.8, preload=True,
                                  event_id=[i for i in list(range(2, 23, 2))] + [99],
                        reject=reject)
    localiser_epochs.drop_bad()

    task_epochs = mne.Epochs(task_raw, task_events, tmin=0, tmax=8, preload=True,
                        reject=reject)
    task_epochs.drop_bad()

    # Downsample to 100 Hz
    print("DOWNSAMPLING")
    print('Original sampling rate:', task_epochs.info['sfreq'], 'Hz')

    localiser_epochs = localiser_epochs.copy().resample(100, npad='auto')

    task_epochs = task_epochs.copy().resample(100, npad='auto')
    print('New sampling rate:', task_epochs.info['sfreq'], 'Hz')

    print("ICA")
    reject = dict(mag=5e-9)
    full_raw = mne.concatenate_raws([localiser_raw, task_raw])

    ica = ICA(n_components=0.95, method='fastica',
              random_state=0, max_iter=100).fit(full_raw, decim=1, reject=reject)
    ica.plot_components()

    pp = PdfPages(os.path.join(data_dir, 'ICA.pdf'))
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

    ica.save(os.path.join(data_dir, 'meg-ica.fif.gz'))

    # detect EOG by correlation
    eog_inds, scores = ica.find_bads_eog(full_raw)
    print(eog_inds)

    print("SAVING")
    localiser_epochs.save(os.path.join(data_dir, 'pre_ICA_localiser-epo.fif.gz'))
    task_epochs.save(os.path.join(data_dir, 'pre_ICA_task-epo.fif.gz'))

    print("DONE")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    args = parser.parse_args()

    preproc_meg(args.data_dir)
