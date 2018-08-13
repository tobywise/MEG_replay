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
import pandas as pd

import os
os.environ['OMP_NUM_THREADS'] = '1'

def preproc_meg(data_dir, session_id, task=True, n_loc=6, n_stim=11):

    data = os.listdir(data_dir)
    data = sorted([i for i in data if '.ds' in i and str(session_id) in i])

    output_dir = os.path.join('/data/twise/', str(session_id))

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print(data)

    localiser_raws = []
    task_raws = []

    localiser_idx = range(0, n_loc)

    for i in localiser_idx:
        localiser_raws.append(read_raw_ctf(os.path.join(data_dir, data[i]), preload=True))

    if task:

        task_idx = range(n_loc, n_loc + 5)

        for i in task_idx:
            task_raws.append(read_raw_ctf(os.path.join(data_dir, data[i]), preload=True))

    localiser_raw = mne.concatenate_raws(localiser_raws)

    if task:
        task_raw = mne.concatenate_raws(task_raws)

    localiser_raw.apply_gradient_compensation(0)

    if task:
        task_raw.apply_gradient_compensation(0)

    mf_kwargs = dict(origin=(0., 0., 0.), st_duration=10.)
    localiser_raw = mne.preprocessing.maxwell_filter(localiser_raw, **mf_kwargs)
    localiser_raw.plot()
    plt.savefig(os.path.join(output_dir, 'localiser_raw.pdf'))

    if task:

        task_raw = mne.preprocessing.maxwell_filter(task_raw, **mf_kwargs)
        task_raw.plot()
        plt.savefig(os.path.join(output_dir, 'task_raw.pdf'))
        task_raw.set_channel_types({'UADC001-2910': 'eog', 'UADC002-2901': 'eog', 'UADC003-2901': 'eog'})

    localiser_raw.set_channel_types({'UADC001-2910': 'eog', 'UADC002-2901': 'eog', 'UADC003-2901': 'eog'})


    # localiser_raws.info['bads'] = ['MLT54']

    # picks = mne.pick_types(localiser_raw.info, meg='mag', eog=True)
    # REJECT BAD CHANNELS
    # eyes = mne.pick_channels(raw.info['ch_names'], ['UADC001-2910', 'UADC002-2901'])

    print("FILTERING")
    localiser_raw.filter(0.5, 45, method='fir', fir_design='firwin')

    if task:
        task_raw.filter(0.5, 45, method='fir', fir_design='firwin')

    print("FINDING EVENTS")
    localiser_events = mne.find_events(localiser_raw, stim_channel='UPPT001', shortest_event=1)
    if task: task_events = mne.find_events(task_raw, stim_channel='UPPT001', shortest_event=1)
    reject = dict(mag=5e-9, eog=20)

    print("EPOCHING")
    localiser_epochs = mne.Epochs(localiser_raw, localiser_events, tmin=-0.1, tmax=0.8, preload=True,
                                  event_id=[i for i in list(range(2, n_stim * 2 + 1, 2))] + [99],
                        reject=None)

    if task:
        task_dropped = {'trial': [], 'event_id': []}

        task_epochs = mne.Epochs(task_raw, task_events, tmin=0, tmax=8, preload=True,
                            reject=None, event_id=[60, 30], consecutive=True)
        print(task_epochs)

        for n, i in enumerate(task_epochs.drop_log):
            if len(i) and i != ['IGNORED']:
                task_dropped['trial'].append(np.where(task_events[task_events[:, 2] == task_events[n][2]][:, 0] == task_events[n][0])[0][0])
                task_dropped['event_id'].append(task_events[n][2])

        task_dropped = pd.DataFrame(task_dropped)
        print(task_dropped)

        task_dropped.to_csv(os.path.join(output_dir, '{0}_task_dropped_trials.csv'.format(str(session_id))))

    # Downsample to 100 Hz
    print("DOWNSAMPLING")
    print('Original sampling rate:', localiser_epochs.info['sfreq'], 'Hz')

    localiser_epochs = localiser_epochs.copy().resample(100, npad='auto')

    if task: task_epochs = task_epochs.copy().resample(100, npad='auto')
    print('New sampling rate:', localiser_epochs.info['sfreq'], 'Hz')

    print("ICA")
    ica = ICA(n_components=0.95, method='fastica',
              random_state=0, max_iter=100).fit(task_epochs, decim=1, reject=None)
    ica.plot_components()

    pp = PdfPages(os.path.join(output_dir, 'ICA.pdf'))
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

    ica.save(os.path.join(output_dir, 'meg-ica.fif.gz'))

    # detect EOG by correlation
    eog_inds, scores = ica.find_bads_eog(task_epochs)
    print(eog_inds)

    print("SAVING")
    localiser_epochs.save(os.path.join(output_dir, 'pre_ICA_localiser-epo.fif.gz'))
    if task: task_epochs.save(os.path.join(output_dir, 'pre_ICA_task-epo.fif.gz'))

    print("DONE")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("session_id")
    parser.add_argument("n_loc", type=int)
    parser.add_argument("n_stim", type=int)
    args = parser.parse_args()

    preproc_meg(args.data_dir, args.session_id, task=True, n_loc=args.n_loc, n_stim=args.n_stim)
