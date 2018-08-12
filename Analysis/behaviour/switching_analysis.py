import pandas as pd
import os
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import ttest_1samp

behavioural_dir ='/path/to/behavioural/files'
trial_info = pd.read_csv('task/Task_information/trial_info.csv')

behavioural_files = os.listdir(behavioural_dir)[:-3]

reward_betas = []
shock_betas = []
interaction_betas = []

for i in behavioural_files:

    try:

        behaviour = pd.read_csv(os.path.join(behavioural_dir, i))

        behaviour.reset_index()
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

        behaviour['switch'] = np.hstack([(np.diff(behaviour.State_3) != 0), [np.nan]])
        behaviour['switch'][np.hstack([(behaviour.Reward_received.isnull())[1:], [np.nan]]).astype(bool)] = np.nan

        behaviour['switch'][behaviour.reward > 0.5].sum()
        behaviour['switch'][behaviour.reward <= 0.5].sum()



        model = smf.Logit.from_formula(formula='switch ~ reward + shock + reward * shock',
                                       data=behaviour[~behaviour['switch'].isnull()])
        result = model.fit()
        # print result.summary()
        
        reward_betas.append(result.params.reward)
        shock_betas.append(result.params.shock)
        interaction_betas.append(result.params['reward:shock'])

    except Exception as e:
        print "Failed on subject {0}".format(i)
        print e

reward = ttest_1samp(reward_betas, 0)
shock = ttest_1samp(shock_betas, 0)
interaction = ttest_1samp(interaction_betas, 0)

template = "EFFECT OF {0} ON SWITCHING\n" \
           "--------------------------\n" \
           "T value: {1}\n" \
           "Degrees of freedom: {2}\n" \
           "P value: {3}\n\n"

print template.format('REWARD', reward.statistic, len(reward_betas) - 1, reward.pvalue)
print template.format('SHOCK', shock.statistic, len(shock_betas) - 1, shock.pvalue)
print template.format('REWARD X SHOCK INTERACTION', interaction.statistic, len(interaction_betas) - 1, interaction.pvalue)










