# -*- coding: utf-8 -*-

from __future__ import division
from psychopy import core, visual, event, gui, monitors, data
from psychopy.iohub import launchHubServer
import pandas as pd
import os, csv
import yaml
import numpy as np
import random


class ReplayExperiment(object):

    def __init__(self, config=None):

        # All of this code will run when we load an instance of this class

        # Load config
        # this sets self.config (the config attribute of our experiment class) to a dictionary containing all the values
        # in our config file. The keys of this dictionary correspond to the section headings in the config file, and
        # each value is another dictionary with keys that refer to the subheadings in the config file. This means you can
        # reference things in the dictionary by e.g. self.config['heading']['subheading']

        # with open(config) as f:
        #     self.config = yaml.load(f)

        subject_id = 10
        random.seed(subject_id)

        # # This part sets up various things to allow us to save the data
        #
        # self.script_location = os.path.dirname(__file__)
        #
        # # Folder for saving data
        # self.save_folder = 'replay_behavioural_data'
        # if not os.path.isdir(self.save_folder):
        #     os.makedirs(self.save_folder)
        #
        # # Data dictionary - used for saving data
        # self.data_keys = ['Subject', 'Trial']
        # self.trial_data = dict(zip(self.data_keys, [None for i in self.data_keys]))
        #
        # # Enter subject ID for saving
        # dialogue = gui.Dlg()
        # dialogue.addField('Subject')
        # dialogue.show()
        #
        # if dialogue.OK:
        #     self.subject = dialogue.data[0]
        # else:
        #     core.quit()

        # This part sets up information about the monitor

        monitor = monitors.Monitor('test2', width=40.92, distance=74)
        monitor.setSizePix((1024, 768))
        self.win = visual.Window(monitor=monitor, size=(1024, 768), fullscr=True, allowGUI=False, color='#606060', units='deg',
                            colorSpace='hex')
        self.win.mouseVisible = True

        self.response_keys = ['down', 'left', 'right', 'up']

        self.arrow_images = {}

        for n in self.response_keys:
            self.arrow_images[n] = 'Stimuli/Arrows/{0}.png'.format(n)

        # Here we define devices we're using - this might get more complicated later

        # self.io = launchHubServer(session_code='A', experiment_code='1')
        # self.keyboard = self.io.devices.keyboard  # iohub keyboard


        # -------------------------------------------------------------------------------#
        # Here we want to define all the stimuli we're going to be showing on the screen #
        # -------------------------------------------------------------------------------#

        # Fixation cross
        self.fixation = visual.TextStim(win=self.win, height=0.8, color='white', text="+")

        # # Text stimuli
        self.main_text = visual.TextStim(win=self.win, height=0.8, color='white',
                                         alignVert='center', alignHoriz='center', wrapWidth=30)
        self.outcome_text = visual.TextStim(win=self.win, height=0.8, color='white',
                                         pos=(0, 7), wrapWidth=30)
        self.move_text = visual.TextStim(win=self.win, height=0.8, color='white',
                                         pos=(0, -7), wrapWidth=30)
        self.instruction_text = visual.TextStim(win=self.win, height=0.8, color='white', wrapWidth=30)

        self.reward_text = visual.TextStim(win=self.win, height=0.8, color='white', pos=(-12, 0), wrapWidth=30)


        # self.main_text.fontFiles = ["fonts/OpenSans-Regular.ttf"]  # Arial is horrible
        # self.main_text.font = 'Open Sans'
        #
        # self.inst_text = visual.TextStim(win=self.win, height=0.7, color='white', pos=(0, -7), wrapWidth=30)
        # self.inst_text.fontFiles = ["C:\Users\Toby\Downloads/fonts\Open_Sans\OpenSans-Regular.ttf"]
        # self.inst_text.font = 'Open Sans'

        # Stimulus location and size information - allows this to be easily set and reused later
        stimuli_location = 'Stimuli'
        self.stimuli = [os.path.join(stimuli_location, i) for i in os.listdir(stimuli_location)
                   if ('.png' in i or '.jpg' in i or '.jpeg' in i) and 'shock' not in i]
        print self.stimuli
        random.shuffle(self.stimuli)

        # positions are given as (x units, y units) - here we're putting three of these coordinate pairs in a list
        self.locs = [(-9, 0), (0, 0), (9, 0)]  # store preset positions for use later

        # sizes are given as (width, height)
        self.image_size = (6, 6)

        self.outcome_image = visual.ImageStim(win=self.win, size=(2, 3), image='Stimuli/shock.png', pos=(0, 7))

        # use imagestim to set up image stimuli - you'll need to fill in some details here
        self.display_image = visual.ImageStim(win=self.win, size=(11, 11))

        self.arrow_move = visual.ImageStim(win=self.win, size=(2, 3))

        # TRANSITION MATRIX

        self.matrix, self.matrix_keys = self.create_matrix()



    def get_responses(self, response_event, response_keys, start_time=0, mapping=None):
        """
        Watches for keyboard responses and returns the response and RT if one is made

        Args:
            response_event: Iohub response event (must not be None)
            response_keys: Allowed keys
            start_time: Trial start time, allows calculation of RT from current time - start time
            mapping: Dictionary of keys to map responses to, one per response key. E.g response keys [a, l] could be mapped
                     to [1, 2] by providing the dictionary {'a': '1', 'l': '2'}

        Returns:
            response: key pressed (mapped if mapping provided)
            rt: response time (minus start time if provided)

        """
        for kb_event in response_event:
            if kb_event.key in response_keys:
                response_time = kb_event.time - start_time
                response = kb_event.key
                if mapping is not None:
                    response = mapping[response]
            else:
                response = None
                response_time = None

            return response, response_time


    def show_stimuli(self, stimuli):

        """
        Shows stimuli. This allows us to simply write
        >>> self.show_stimuli(stimulus_list)

        rather than
        >>> stimulus_A.draw()
        >>> stimulus_B.draw()
        >>> stimulus_C.draw()

        etc...

        Args:
            stimuli: A list of stimuli we want to draw - e.g. [stim_A, stim_B, stim_C]

        Returns:

        """

        for stim in stimuli:
            stim.draw()


    def show_move(self, outcome, picture, move, reward):


        # set image
        self.display_image.image = picture

        # set outcome text either as value or shock
        #self.outcome_image.image = outcome


        if outcome == 0:
            self.outcome_image.draw()
        else:
            self.outcome_text.text = outcome
            self.outcome_text.draw()

        # set move text
        self.arrow_move.image = self.arrow_images[move]

        #show reward text
        self.reward_text.text = reward

        # draw everything
        self.arrow_move.draw()
        self.display_image.draw()
        self.reward_text.draw()


    def instructions(self, text):

        # set text
        self.instruction_text.text = text
        # draw
        self.instruction_text.draw()
        self.win.flip()
        # waitkeys
        event.waitKeys(maxWait=5, keyList='space')


    def create_matrix(self):

        # T MAZE
        matrix = np.zeros((13, 13))

        matrix[0, 1] = 1
        matrix[[1, 1], [0, 2]] = 1
        matrix[[2, 2], [1, 3]] = 1
        matrix[[3, 3, 3, 3], [2, 4, 9, 10]] = 1
        matrix[[4, 4], [3, 5]] = 1
        matrix[[5, 5], [4, 6]] = 1
        matrix[6, 5] = 1

        matrix[7, 8] = 1
        matrix[[8, 8], [7, 9]] = 1
        matrix[[9, 9], [8, 3]] = 1
        matrix[[10, 10], [3, 11]] = 1
        matrix[[11, 11], [10, 12]] = 1
        matrix[12, 11] = 1

        matrix_keys = matrix.astype(int).astype(str)
        keys = ['up', 'down', 'left', 'right']
        random.shuffle(keys)

        for i in range(matrix.shape[0]):
            one_idx = np.where(matrix[i, :] == 1)
            random.shuffle(keys)
            for j in range(len(one_idx[0])):
                matrix_keys[i, one_idx[0][j]] = keys[j]

        return matrix, matrix_keys


    def moves_to_states(self, trial_moves, start):

        state = start
        previous_state = None

        moves_states = []

        for n, move in enumerate(trial_moves):
            # print "Move {0}, move = {1}".format(n, move)
            allowed_moves = [i for n, i in enumerate(self.matrix_keys[state, :]) if not '0' in i
                             and not n == previous_state]
            if move not in allowed_moves:
                return False
            row = self.matrix_keys[state, :]
            next_state = np.where(row == move)[0][0]
            moves_states.append((move, next_state))
            previous_state = state
            state = next_state

        return moves_states

    def test_moves(self, start):

        state = start
        previous_state = None

        moves_states = []

        for n in range(self.n_moves):
            # print "Move {0}, move = {1}".format(n, move)
            allowed_moves = [i for n, i in enumerate(self.matrix_keys[state, :]) if not '0' in i
                             and not n == previous_state]
            print allowed_moves
            if len(allowed_moves) == 0:
                return False
            move = allowed_moves[np.random.randint(0, len(allowed_moves))]
            row = self.matrix_keys[state, :]
            next_state = np.where(row == move)
            moves_states.append(move)
            previous_state = state
            state = next_state[0][0]

        return moves_states

    def run_task(self, training=False, structured=False):

        # Clock
        self.clock = core.Clock()

        # Read trial info
        # trial_info = pd.read_csv(self.config['trial_info']['trial_info'])

        trial_info = range(0, 10)  # placeholder, gives us 10 trials
        trial_number = range(0, 10)  # placeholder

        # Convert each column to a python list
        # trial_number = trial_info.trial_number.tolist()
        # trial_type = trial_info.trial_type.tolist()

        # Create data file
        # if not training:
        #     fname = '{0}/UCO_uncertainty_Subject{1}_{2}.csv'.format(self.save_folder, self.subject, data.getDateStr())
        #     csvWriter = csv.writer(open(fname, 'wb'), delimiter=',').writerow
        #     csvWriter(self.data_keys)


        self.start_duration = 5
        self.pre_move_duration = 3
        self.move_entering_duration = 3
        self.move_duration = 2
        self.n_moves = 5
        self.move_period_duration = self.move_duration * self.n_moves

        test_moves = np.repeat(['up', 'down', 'left', 'right'], 5)


        #if self.instruction_duration is True:
        #     self.instruction_text.text = "Welcome to the MEG thing"
        #     self.instruction_text.draw()
        #
        # key = event.getKeys(keyList=['space'])
        #
        # if key is not None:
        #     self.instruction_duration = False

        ####
        # instructions
        text = "Welcome to MEG thing! Insert instructions here. Press spacebar to continue."

        self.instructions(text)


        for i in range(len(trial_info)):  # TRIAL LOOP - everything in here is repeated each trial

            print "Trial {0} / {1}".format(i, len(trial_info))

            # self.io.clearEvents('all')  # clear keyboard events

            continue_trial = True  # this variables changes to False when we want to stop the trial
            self.clock.reset()

            change_times = [0, self.start_duration, self.start_duration + self.pre_move_duration, self.start_duration +
                            self.pre_move_duration + self.move_entering_duration, self.start_duration +
                            self.pre_move_duration + self.move_entering_duration + self.move_period_duration]  # list of times at which the screen should change

            # set text
            welcome_text = 'Welcome to MEG thing!'

            left_text = 'left'
            right_text = 'right'
            up_text = 'up'
            down_text = 'down'

            # Starting state
            start_state = [0, 1, 11, 12, 8, 7, 5, 6]
            random.shuffle(start_state)
            start_state = start_state[0]
            self.display_image.setImage(self.stimuli[start_state])
            row = self.matrix_keys[start_state, :]
            states = None

            # rewards
            outcome = [0.12, 0.18, 0.23, 0.31, 0.35, 0.41, 0.44, 0.48, 0.56, 0.62, 0.74, 0.85, 0.91, 0]
            random.shuffle(outcome)
            reward = [0] * self.n_moves

            # Default values for responses in case the subject makes no response
            response = None
            rt = None

            key = None

            trial_moves = []

            moves_found = False

            moves_to_enter = []

            while not moves_found:
                random.shuffle(test_moves)
                moves_to_enter = self.test_moves(start_state)
                print moves_to_enter
                if moves_to_enter is not False:
                    moves_found = True

            key_text = 'Enter key movements\n{0}'.format(moves_to_enter)



            while continue_trial:  # run the trial

                t = self.clock.getTime()  # get the time

                # SCREEN 1
                if change_times[0] <= t < change_times[1]:

                    self.display_image.draw()


                #SCREEN 2
                elif change_times[1] <= t < change_times[2]:
                    self.main_text.text = key_text
                    self.main_text.draw()
                    event.clearEvents()

                #SCREEN 3
                elif change_times[2] <= t < change_times[3]:

                    # if not len(trial_moves):
                    #     self.main_text.text = ''

                    raw_keys = event.getKeys(keyList=['left', 'right', 'up', 'down'], timeStamped=True)

                    if len(raw_keys):
                        key, rt = raw_keys[0]

                        print key, rt

                    if key is not None and len(trial_moves) < self.n_moves:
                        self.arrow_move.image = self.arrow_images[key]

                        trial_moves.append(key)
                        key = None

                    if len(trial_moves) > 0:
                        self.arrow_move.draw()

                elif change_times[3] <= t < change_times[4]:
                    if states is None:
                        moves_states = self.moves_to_states(trial_moves, start_state)
                        # when wrong moves are entered have screen to display "You entered wrong movements"
                        # and wait for the same duration of time before next trial
                    if moves_states is False:
                        self.main_text.text = "Wrong moves entered"
                        self.main_text.draw()
                    elif len(trial_moves) < self.n_moves:
                        self.main_text.text = "Too few moves entered"
                        self.main_text.draw()
                    else:
                        for n, (move, state) in enumerate(moves_states):
                            print move, state
                            if change_times[3] + n * self.move_duration <= t < change_times[3] + (n + 1) * self.move_duration:
                                reward[n] = outcome[state]
                                self.show_move(outcome[state], self.stimuli[state], move, sum(reward))


                elif t >= change_times[-1]:
                    print trial_moves

                    continue_trial = False


                # # SCREEN 2
                # elif change_times[1] <= t < change_times[2]:
                #
                #     self.show_stimuli()
                #
                # # ITI
                # elif change_times[2] <= t < change_times[3]:
                #
                #     self.fixation.draw()

                # End trial
                # elif t >= change_times[-1]:
                #     print trial_moves
                #     continue_trial = False

                # flip to draw everything
                self.win.flip()

                # Add remaining data to dictionary for saving
                # Responses
                # self.trial_data['response'] = response
                # self.trial_data['rt'] = rt

                # If the trial has ended, save data to csv
                # if not continue_trial:
                #     if not training:
                #         csvWriter([self.trial_data[category] for category in self.data_keys])

                # quit if subject pressed scape
                if event.getKeys(["escape"]):
                    core.quit()


## RUN THE EXPERIMENT

experiment = ReplayExperiment('replay_task_settings.yaml')

experiment.run_task(training=False)

