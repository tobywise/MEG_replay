import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as font_manager
from collections import OrderedDict
from matplotlib import gridspec
import pandas as pd

prop = font_manager.FontProperties(fname="C:\WINDOWS\Fonts\opensans-light.ttf")
matplotlib.rcParams['font.family'] = prop.get_name()

# VARIABLES

n_trials = 100  # number of trials
n_blocks = 4  # number of blocks (reversals)
scale = 0.03  # mean step size of random walk
sd = 0.05  # SD of step size
diff = 0.04  # difference between random walks in the same tree branch
n_outcome_trials = 25

save = True


# CALCULATE MEAN INNOVATIONS FOR RANDOM WALK

trials_per_block = n_trials / n_blocks
a_innov_mean = []
b_innov_mean = []

for i in range(n_blocks):
    b_innov_mean.append(np.ones(trials_per_block) * (i % 2 * 2 - 1) * scale)  # random walk mean step size follows alternating pattern
    a_innov_mean.append(np.ones(trials_per_block) * ((i + 1) % 2 * 2 - 1) * scale)

a_innov_mean = np.hstack(a_innov_mean)
b_innov_mean = np.hstack(b_innov_mean)

# CREATE RANDOM WALKS

values = OrderedDict([('h', .25), ('i', .23), ('j', .90), ('k', .96)])

for state, start in values.iteritems():
    print state
    if state == 'h':
        random_walk = np.cumsum(np.hstack([start, np.random.normal(a_innov_mean, sd, n_trials)[1:]]))
        values[state] = random_walk + np.random.normal(0, diff, n_trials)
        values['i'] = random_walk + np.random.normal(0, diff, n_trials)
    elif state == 'j':
        random_walk = np.cumsum(np.hstack([start, np.random.normal(b_innov_mean, sd, n_trials)[1:]]))
        values[state] = random_walk + np.random.normal(0, diff, n_trials)
        values['k'] = random_walk + np.random.normal(0, diff, n_trials)
    # Scale values to between zero and one
    values[state] = (values[state] + -values[state].min()) / values[state].max()


plt.figure(figsize=(8, 3))
for state in values.keys():
    plt.plot(values[state], label='State {0}'.format(state.capitalize()))
plt.xlabel("Trial")
plt.ylabel("Reward")
plt.legend(loc='upper right')
plt.tight_layout()
if save:
    plt.savefig('Slides/random_walks.svg')

# SHOCKS

n_shocks_min = 3  # minimum number of consecutive shocks from one state
n_shocks_max = 7  # maximum number

shocks = []

while np.sum(shocks) < n_trials:
    shocks.append(np.random.randint(n_shocks_min, n_shocks_max))

shock_outcomes = OrderedDict([('h', []), ('i', []), ('j', []), ('k', [])])

prev_shocked_state = 'a'

for i in shocks:
    shocked_state = [s for s in shock_outcomes.keys() if s!= prev_shocked_state][np.random.randint(0, len(shock_outcomes.keys()) - 1)]
    for s in shock_outcomes.keys():
        if s == shocked_state:
            shock_outcomes[s] += np.ones(i).tolist()
        else:
            outcomes = np.zeros(i)
            outcomes[:] = np.nan
            shock_outcomes[s] += outcomes.tolist()
        if shock_outcomes[s] > n_trials:
            shock_outcomes[s] = shock_outcomes[s][:n_trials]
    prev_shocked_state = shocked_state

for k, v in shock_outcomes.iteritems():
    nans = np.where(np.isnan(shock_outcomes[k]))[0]
    np.random.shuffle(nans)
    extra_shock_idx = nans[:int(len(nans) * .15)]
    shock_outcomes[k] = np.array(shock_outcomes[k])
    shock_outcomes[k][extra_shock_idx] = 1

# CREATE TRIAL OUTCOMES

reward_df = pd.DataFrame(values)
reward_df.columns = ['0_reward', '1_reward', '2_reward', '3_reward']

shock_df = pd.DataFrame(shock_outcomes)
shock_df.columns = ['0_shock', '1_shock', '2_shock', '3_shock']

trial_info = pd.concat([reward_df, shock_df], axis=1)
trial_info['trial_number'] = range(0, len(trial_info))

trial_type = np.hstack([np.ones(n_outcome_trials), np.zeros(n_trials - n_outcome_trials)])
np.random.shuffle(trial_type)
trial_info['trial_type'] = trial_type
trial_info[trial_info.isnull()] = 0
trial_info['end_state'] = np.random.randint(7, 11, len(trial_info))


if save:
    trial_info.to_csv('task/Task_information/trial_info.csv', index=False)

## PLOT THINGS

import matplotlib.font_manager as font_manager
prop = font_manager.FontProperties(fname="C:\WINDOWS\Fonts\opensans-light.ttf")
matplotlib.rcParams['font.family'] = prop.get_name()
matplotlib.rcParams['axes.facecolor'] = '#fbfbfb'


plt.figure(figsize=(15, 3))

gap = 0.4

# Plot
# Reward levels
for k in range(4):
    if k == 0:
        r_label = 'Reward level'
        s_label = 'Shock'
    else:
        r_label = None
        s_label = None

    trial_info.loc[trial_info['{0}_shock'.format(k)] == 0, '{0}_shock'.format(k)] = np.nan  # recode zero shocks to nan

    plt.scatter(range(len(trial_info)), trial_info['{0}_shock'.format(k)] * k + gap,
                  facecolors='#c87200', label=s_label, marker='x')
    plt.scatter(range(len(trial_info)), np.ones(len(trial_info)) * k, edgecolors='gray', linewidth=0.5,
                  c=trial_info['{0}_reward'.format(k)], cmap='inferno', label=r_label, alpha=0.5)

# Choices
plt.yticks(range(4), range(4))
plt.xlabel("Trial", fontweight='light')
plt.ylabel("Terminal state", fontweight='light')
plt.yticks(range(0, 4))
# plt.yticklabels(range(1, 5))

# outcome only trials
for j in range(len(trial_info)):
    if trial_info.trial_type[j] == 1:
        plt.axvline(j, color='#f4f4f4', lw=8, zorder=0, ymin=(trial_info.end_state[j] - 7) / 4.,
                      ymax=(trial_info.end_state[j] - 6) / 4.)
plt.tight_layout()

if save:
    plt.savefig('Slides/random_walks_shocks.svg')
