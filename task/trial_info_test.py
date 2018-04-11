import pandas as pd
import yaml
import numpy as np
import os

with open('task/replay_task_settings.yaml') as f:
    config = yaml.load(f)

n_trials = config['number training trials']['n_test_trials']

df = pd.DataFrame(dict(trial_number=np.arange(0, n_trials),
                       end_state=np.random.randint(7, 11, n_trials),
                       trial_type=np.zeros(n_trials)))

df.to_csv(os.path.join('task', config['directories']['trial_info_test']))