import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as font_manager
prop = font_manager.FontProperties(fname="C:\WINDOWS\Fonts\opensans-light.ttf")
matplotlib.rcParams['font.family'] = prop.get_name()
matplotlib.rcParams['axes.facecolor'] = '#fbfbfb'


# Load trial info and specify subjects
subjects = ['M006', 'M007', 'M008']
data_dir = 'task\Data/behavioural\pilot'
trial_info = pd.read_csv('task\Task_information/trial_info.csv')
trial_info['end_state'] = trial_info['end_state'] - 7


# Plot things
f, ax = plt.subplots(len(subjects), 1, figsize=(15, 8))
gap = 0.4

for n, i in enumerate(subjects):

    # Find subject data
    data = [j for j in os.listdir(data_dir) if 'Subject{0}'.format(i) in j and not 'localiser' in j]
    data = pd.read_csv(os.path.join(data_dir, data[0]))

    # Recode final state variable (7-10 -> 0-3)
    data['State_3'] = data['State_3'] - 7

    # Plot
    # Reward levels
    for k in range(4):
        if k == 0:
            r_label = 'Reward level'
            s_label = 'Shock'
        else:
            r_label = None
            s_label = None

        trial_info.loc[
            trial_info['{0}_shock'.format(k)] == 0, '{0}_shock'.format(k)] = np.nan  # recode zero shocks to nan

        ax[n].scatter(range(len(trial_info)), trial_info['{0}_shock'.format(k)] * k + gap,
                      facecolors='#c87200', label=s_label, marker='x')
        ax[n].scatter(range(len(trial_info)), np.ones(len(trial_info)) * k, edgecolors='gray', linewidth=0.5,
                      c=trial_info['{0}_reward'.format(k)], cmap='inferno', label=r_label, alpha=0.5)

    # Choices
    ax[n].scatter(np.arange(len(data))[~data['Reward_received'].isnull().values],
                  data['State_3'][~data['Reward_received'].isnull().values],
                  edgecolors='#3d3d3d', linewidths=1.2,
                  label='Chosen state', facecolors='none')

    # Outcome only / incorrect trial outcomes
    ax[n].scatter(np.arange(len(data))[data['Reward_received'].isnull().values],
                  trial_info['end_state'][data['Reward_received'].isnull().values],
                  edgecolors='#7e7e7f', linewidths=1.2, marker='s',
                  label='Forced state', facecolors='none')

    ax[n].set_yticks(range(4), range(4))
    ax[n].set_xlabel("Trial", fontweight='light')
    ax[n].set_ylabel("Terminal state", fontweight='light')
    ax[n].set_yticks(range(0, 4))
    ax[n].set_yticklabels(range(1, 5))

    # outcome only trials
    for j in range(len(data)):
        if data.trial_type[j] == 1:
            ax[n].axvline(j, color='#f4f4f4', lw=8, zorder=0, ymin=(trial_info.end_state[j] - 7) / 4.,
                          ymax=(trial_info.end_state[j] - 6) / 4.)

    # legend
    if n == 0:
        legend = ax[n].legend()
        legend.legendHandles[1].set_color(plt.cm.inferno(.2))
    ax[n].set_title('Subject {0}, total reward = {1}, number of shocks = {2}'.format(i, data.Reward_received.sum(),
                                                                                     data.Shock_received.sum().astype(
                                                                                         int)))

plt.tight_layout()
plt.show()