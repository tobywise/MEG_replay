import yaml
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

with open('task/replay_task_settings.yaml') as f:
    config = yaml.load(f)

phase_durations = OrderedDict()

phase_durations['Planning'] = config['MEG_durations']['start_duration']
phase_durations['Move entering'] = config['MEG_durations']['move_entering_duration']
phase_durations['Pre move fixation'] = config['MEG_durations']['pre_move_fixation_duration']
phase_durations['Move_1'] = config['MEG_durations']['move_durations'][0]
phase_durations['Move_2'] = config['MEG_durations']['move_durations'][1]
phase_durations['Move_3'] = config['MEG_durations']['move_durations'][2]
phase_durations['Final move'] = config['MEG_durations']['move_durations'][3]
phase_durations['Reward'] = config['MEG_durations']['shock_symbol_delay']
phase_durations['Shock symbol'] = config['MEG_durations']['shock_delay']
phase_durations['Rest'] = config['MEG_durations']['rest_duration']

starts = [phase_durations['Planning'],
          phase_durations['Move entering'],
          phase_durations['Pre move fixation'],
          phase_durations['Move_1'],
          phase_durations['Move_2'],
          phase_durations['Move_3'],
          phase_durations['Reward'],
          phase_durations['Shock symbol'],
          phase_durations['Final move'] - (phase_durations['Reward'] + phase_durations['Shock symbol']),
          phase_durations['Rest']]

phases = ['Planning', 'Move entering', 'Pre move fixation', 'State 1', 'State 2', 'State 3', 'Reward', 'Shock symbol',
          'Shocks', 'Rest']

starts = np.cumsum(starts)
ends = starts.copy()
starts = np.hstack([0, starts[:-1]])


plt.figure(figsize=(15, 3))
for i in ends:
    plt.axvline(i, linestyle='--', color='gray')
bars = plt.barh(range(len(starts)),  ends-starts, left=starts)
plt.yticks(range(len(starts)), phases)
plt.title("Trial phase durations")

cmap = mpl.cm.cool
colours = []
for n, i in enumerate(np.arange(0, 1, 1. / len(phases))):
    bars[n].set_color(cmap(i))
plt.tight_layout()

plt.savefig('task/trial_durations.pdf')
plt.savefig('task/trial_durations.png')