import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import scale
from sklearn.externals import joblib

def check_dimensions(sequenceness, phase, measure, n_trials, n_lags, n_arms):
    if sequenceness.ndim != 3:
        raise AttributeError("{0} {1} sequenceness should have 3 dimensions, found {2}".format(phase, measure, sequenceness.ndim))
    if sequenceness.shape[0] != n_trials:
        raise AttributeError('Too few trials in {0} {1} sequenceness, expected {2}, found {3}'.format(phase, measure, n_trials, sequenceness.shape[0]))
    if sequenceness.shape[1] != n_lags:
        raise AttributeError("Too few lags in {0} {1} sequenceness, expected {2}, found {3}".format(phase, measure, n_lags, sequenceness.shape[1]))
    if sequenceness.shape[2] != n_arms:
        raise AttributeError("Too few arms in {0} {1} sequenceness, expected {2}, found {3}".format(phase, measure, n_arms, sequenceness.shape[2]))

def check_missing(sequenceness, phase, measure):

    if np.any(np.isnan(sequenceness)):
        raise ValueError("{0} {1} sequenceness contains NaNs".format(phase, measure))
    if np.any(np.isinf(sequenceness)):
        raise ValueError("{0} {1} sequenceness contains infs".format(phase, measure))        
    
def get_chosen_unchosen(sequenceness, behaviour, exclude_outcome_only, type='shown'):

    """
    Converts arm 1/2 to chosen/unchosen. 
    TODO check that unchosen/chosen are the correct way round
    TODO chosen moves on next trial
    
    Args:
        sequenceness: Array of sequenceness, shape (n_subjects, n_trials, 3) where the last dimension represents (both arms, arm 1, arm 2)
        behaviour: Behavioural data
        exclude_outcome_only: Exclude outcome only trials
        type: 'shown' or 'chosen' - if 'shown' uses the shown move in failed trials

    Returns:
        Modified version of the input array, where the last dimension now represents (both arms, chosen arm, unchosen arm)
    """

    if exclude_outcome_only:
        behaviour = behaviour[behaviour['trial_type'] == 0].copy()

    # Sequenceness should be n_trials X n_lags X n_arms
    chosen = sequenceness[..., 1].copy()
    unchosen = sequenceness[..., 2].copy()

    # Whether we're using the move that was shown or the move they chose
    if type == 'chosen':
        column = 'next_move'
    elif type == 'shown':
        column = 'shown_move'

    chosen[behaviour[column] == 1, :] = sequenceness[behaviour[column] == 1, :, 2]
    unchosen[behaviour[column] == 1, :] = sequenceness[behaviour[column] == 1, :, 1]

    sequenceness[..., 1] = chosen
    sequenceness[..., 2] = unchosen

    return sequenceness


class Sequenceness(object):

    def __init__(self, sequenceness, behaviour, subject, n_trials=100, chosen=True, accuracy=None):

        """
        Args:
            sequenceness: List of (phase, sequenceness dictionary pickle, expected shape, exclude outcome only trials) tuples
            behaviour: Either pandas dataframe or path to csv file containing behavioural data
            n_trials: Number of trials in the behavioural data
            chosen: If true, changes the third dimension in the sequenceness data to chosen/unchosen arm rather than arm 1/2
        """
        
        # Subject ID
        self.subject = subject
        
        # Accuracy
        self.accuracy = accuracy

        # N trials
        self.n_trials = n_trials

        # Load behaviour
        if isinstance(behaviour, str):
            self.behaviour = pd.read_csv(behaviour)
        elif isinstance(behaviour, pd.DataFrame):
            self.behaviour = behaviour

        # Remove trials that need to be removed
        self.behaviour = self.behaviour[~self.behaviour['trial_number'].isnull()]

        # Calculate useful things
        self.behaviour['shown_move'] = self.behaviour['State_3_shown'] - 5  # Get the state that was show on each trial
        self.behaviour['chosen_move'] = self.behaviour['State_1_chosen'] - 1  # Get the state that the subject chose (coded as 1/2 originally)

        # Check behaviour shapes etc
        if len(self.behaviour) != n_trials:
            raise AttributeError("Behavioural data should have {0} trials, found {1}".format(n_trials, len(self.behaviour)))
        if np.any(np.diff(self.behaviour['trial_number']) != 1):
            raise ValueError("Trial numbers in behavioural data don't increase linearly, some trials may be missing")

        # Load sequenceness
        self.sequenceness = dict()

        for seq in sequenceness:
            phase = seq[0]
            data = seq[1]
            shape = seq[2]
            exclude_outcome_only = seq[3]

            # Read in pickle
            self.sequenceness[phase] = joblib.load(data)

            # Go through measures (i.e. forwards/backwards/difference)
            for measure in self.sequenceness[phase].keys():
                # Check sequenceness shapes and missing data
                check_dimensions(self.sequenceness[phase][measure] , phase, measure, *shape)
                check_missing(self.sequenceness[phase][measure], phase, measure)

                # Change to chosen / unchosen
                if chosen:
                    self.sequenceness[phase][measure] = get_chosen_unchosen(self.sequenceness[phase][measure], self.behaviour, exclude_outcome_only)

    def __repr__(self):
        return '<' + self.subject + ' sequenceness data | {0} trials>'.format(self.n_trials)

    def trialwise(self, phase, measure='difference', exclude_outcome_only=False, predictor_shifts=()):

        """
        Produces a trialwise dataframe of behaviour and sequenceness data
        
        Returns:
            phase (str): Task phase
            measure (str): Sequenceness measure (e.g. difference)
            exclude_outcome_only (bool): Exclude outcome only trials
            predictor_shifts (list): List of dictionaries containing two keys, 'name' and 'shift'. Name specifies the column in the behavioural dataframe, 
        """

        # Copy the behavioural data to avoid changing the original
        behaviour = self.behaviour.copy()
        # Copy sequenceness so we don't affect the original data
        sequenceness = self.sequenceness[phase][measure].copy()

        # Exclude outcome only trials from behaviour if needed
        if exclude_outcome_only:
            if sequenceness.shape[0] == len(behaviour):
                sequenceness = sequenceness[behaviour.trial_type == 0]
            behaviour = behaviour[behaviour.trial_type == 0]
        # Get data for each arm
        seq_dfs = []

        for arm in range(sequenceness.shape[2]):
            # Convert sequenceness to pandas dataframe with each lag for each arm as a different column
            seq_dfs.append(pd.DataFrame(sequenceness[..., arm], columns=['arm_{0}__lag_'.format(arm) + str(i) for i in range(sequenceness.shape[1])]))

        # Concatenate behaviour and sequenceness
        behav_seq = pd.concat([behaviour.reset_index()] + [seq_df.reset_index() for seq_df in seq_dfs], axis=1)

        # Shift predictors if needed
        for pred in predictor_shifts:
            behav_seq[pred['name']] = np.roll(behav_seq[pred['name']],
                                            pred['shift'])  # shift any predictors from the previous trial
            behav_seq[pred['name']][pred['shift']] = np.nan  # Used to drop trials that get shifted to weird places

        # Drop trials that we've lost when shifting predictors
        behav_seq = behav_seq.dropna(axis=0, subset=[i['name'] for i in predictor_shifts])

        behav_seq['trial_number_new'] = np.arange(len(behav_seq))

        return behav_seq

        
def create_df_dict(predictors, arms=['both', 'chosen', 'unchosen'], n_lags=40):

    """
    Creates a dictionary to hold results of GLM - later turned into a DataFrame

    Args:
        predictors: List of predictors

    Returns:
        Dictionary with keys (arm, lag, predictor, beta)
    
    """

    result_df = dict()
    result_df['lag'] = np.tile(np.repeat(np.arange(n_lags), len(predictors)), 1)
    result_df['predictor'] = []
    result_df['beta'] = []

    return result_df


def trialwise_glm(sequenceness, formula, phase, measure='difference', predictor_shifts=(), exclude_outcome_only=False, n_lags=40):
    """
    Runs a trialwise GLMs predicting replay on a given trial from behavioural predictors. This is run across
    all time lags to demonstrate where replay intensity is related to a variable of interest across trials.

    Args:
        sequenceness: Object of the Sequenceness class or a dataframe with behavioural data and sequenceness data
        formula: Formula of the form arm_n__lag ~ predictors, where n is the arm number of interest
        phase (str): Phase of the task. Accepts either 'rest' or 'planning'
        predictor_shifts (list): List of dictionaries containing two keys, 'name' and 'shift'. Name specifies the column in the behavioural dataframe, 
        shift specifies in which direction the values of this columnn should be shifted (e.g. moving back one trial to line up MEG data with behaviour 
        on the subsequent trial).
        exclude_outcome_only (bool): If true and using 'rest' phase, excludes outcome only trials
        n_lags: Number of time lags in the sequenceness data

    Returns:
        A dataframe containing beta values

    """

    # Check things
    if not isinstance(predictor_shifts, list) and not isinstance(predictor_shifts, tuple):
        raise TypeError("Predictors should be specified as a list of dictionaries")

    if not all([isinstance(i['name'], str) for i in predictor_shifts]):
        raise TypeError("Predictor list should contain dictionaries with keys 'name' and 'shift'")

    if not all([isinstance(i['shift'], int) for i in predictor_shifts]):
        raise TypeError("Predictor shift values should be specified as integers")

    # Stringify predictors
    # pred_string = stringify_predictors(predictors)
    predictor_names = re.findall('(?<=[~+] )[A-Za-z_:]+', formula)
    arm = re.search('arm_[0-9]', formula).group()
    formula = re.search('(?<=~ ).+', formula).group()

    # Dictionary to store coefficients, which will later be turned into a pandas DataFrame
    result_df = create_df_dict(predictor_names, arms=['both', 'chosen', 'unchosen'], n_lags=40)

    if isinstance(sequenceness, pd.DataFrame):
        behav_seq = sequenceness
    else:
        # Create a dataframe containing the necessary data
        behav_seq = sequenceness.trialwise(phase, measure, exclude_outcome_only, predictor_shifts)
        try:
            behav_seq[[i for i in predictor_names if not ':' in i]] = scale(behav_seq[[i for i in predictor_names if not ':' in i]])
        except Exception as e:
            print(behav_seq)
            raise e

    # Run GLM across all time lags
    for lag in range(n_lags):
        model = smf.ols(formula='{0}__lag_{1} ~ {2}'.format(arm, lag, formula), data=behav_seq)
        res = model.fit()
        for p in predictor_names:
            result_df['predictor'].append(p)
            result_df['beta'].append(res.params[p])

    result_df = pd.DataFrame(result_df)
    result_df['Subject'] = sequenceness.subject

    return result_df

