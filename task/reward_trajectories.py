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
scale = 0.04  # mean step size of random walk
sd = 0.05  # SD of step size
diff = 0.04  # difference between random walks in the same tree branch
n_outcome_trials = 20


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
plt.savefig('Slides/random_walks.svg')

# SHOCKS

n_shocks_min = 2  # mininmum number of consecutive shocks from one state
n_shocks_max = 6  # maximum number

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


# Plot shocks and rewards
plt.figure(figsize=(8, 3))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

for state in shock_outcomes.keys():
    ax0.scatter(range(n_trials), shock_outcomes[state], edgecolors='white')

ax0.set_yticks([])
ax0.set_xticks([])
ax0.set_title("Shock outcomes")

for state in values.keys():
    ax1.plot(values[state], label='State {0}'.format(state.capitalize()))
ax1.set_xlabel("Trial")
ax1.set_ylabel("Reward")
ax1.legend(loc='upper right')
ax1.set_title("Reward outcomes")

plt.tight_layout()

plt.savefig('Slides/random_walks_shocks.svg')


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

trial_info.to_csv('task/Task_information/trial_info.csv', index=False)