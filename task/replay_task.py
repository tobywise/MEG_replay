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
        # print self.stimuli
        random.shuffle(self.stimuli)

        # positions are given as (x units, y units) - here we're putting three of these coordinate pairs in a list
        self.locs = [(-9, 0), (0, 0), (9, 0)]  # store preset positions for use later

        # sizes are given as (width, height)
        self.image_size = (6, 6)

        self.outcome_image = visual.ImageStim(win=self.win, size=(2, 3), image='Stimuli/shock.png', pos=(0, 7))

        # use imagestim to set up image stimuli - you'll need to fill in some details here
        self.display_image = visual.ImageStim(win=self.win, size=(11, 11))

        self.arrow_display = visual.ImageStim(win=self.win, size=(2, 3))

        self.arrow_gap = 3
        self.n_moves = 5

        self.n_training_trials = 10

        # on each loop, append an image stimulus to the list

        # TRANSITION MATRIX

        self.matrix, self.matrix_keys = self.create_matrix()

        #circle

        self.circle = visual.Circle(win=self.win, radius=1, fillColor=None, lineColor=[1, 1, 1], pos=(0, -8))

        self.arrow_positions, self.arrow_progress = self.create_arrows(self.n_moves)


    def create_arrows(self, n):

        """

        :param n:
        :return:
        """

        arrow_positions = np.arange(-np.abs(((n - 1) * self.arrow_gap) / 2.),
                                         np.abs(((n - 1) * self.arrow_gap) / 2.) + 0.1, self.arrow_gap)

        arrow_progress = []
        # loop through range(number of moves)

        for i in range(n):
            arrow_progress.append(visual.ImageStim(win=self.win, size=(1.5, 2), pos=(arrow_positions[i], -8)))

        return arrow_positions, arrow_progress

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

    def show_move(self, outcome, shock, picture, move, reward, t, shock_time):


        # set image
        self.display_image.image = picture

        # set outcome text either as value or shock
        #self.outcome_image.image = outcome

        if t <= shock_time:
            self.outcome_text.text = outcome
            self.outcome_text.draw()
        elif t > shock_time and shock == 1:
            self.outcome_image.draw()



        #show reward text
        self.reward_text.text = reward

        for i in range(self.n_moves):
            self.arrow_progress[i].draw()

        # draw on each iteration

        # draw everything
        self.display_image.draw()
        self.reward_text.draw()

    def instructions(self, text, max_wait=2):

        # set text
        self.instruction_text.text = text
        # draw
        self.instruction_text.draw()
        self.win.flip()
        # waitkeys
        event.waitKeys(maxWait=max_wait, keyList='space')


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
            allowed_moves = [i for nn, i in enumerate(self.matrix_keys[state, :]) if not '0' in i
                             and not nn == previous_state]

            if move not in allowed_moves:
                return False
            row = self.matrix_keys[state, :]
            next_state = np.where(row == move)[0][0]
            moves_states.append((move, next_state))
            previous_state = state
            state = next_state

        return moves_states

    def test_moves(self, start, n_moves, single_move=True):

        state = start
        previous_state = None

        moves_states = []

        for n in range(n_moves):
            allowed_moves = [i for n, i in enumerate(self.matrix_keys[state, :]) if not '0' in i
                             and not n == previous_state]
            # print allowed_moves
            if len(allowed_moves) == 0:
                return False
            if single_move:
                move = allowed_moves[np.random.randint(0, len(allowed_moves))]
            else:
                move = allowed_moves
            if n_moves > 1 and single_move:
                row = self.matrix_keys[state, :]
                next_state = np.where(row == move)
                previous_state = state
                state = next_state[0][0]
                move = [move]

            moves_states += move

        return moves_states

    def run_training(self):

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



        text = "Begin training by pressing the spacebar"

        self.instructions(text, max_wait=5)



        # Starting state
        start_state = [0, 1, 11, 12, 8, 7, 5, 6]
        random.shuffle(start_state)
        start_state = start_state[0]
        self.display_image.setImage(self.stimuli[start_state])
        row = self.matrix_keys[start_state, :]
        states = None

        for trial in range(self.n_training_trials):

            text = "Starting new trial"
            self.instructions(text)

            for i in range(len(trial_info)):  # TRIAL LOOP - everything in here is repeated each trial

                print "Trial {0} / {1}".format(i, len(trial_info))

                # self.io.clearEvents('all')  # clear keyboard events

                continue_trial = True  # this variables changes to False when we want to stop the trial
                self.clock.reset()

                change_times = [0, self.start_duration, self.start_duration + self.pre_move_duration, self.start_duration +
                                self.pre_move_duration + self.move_entering_duration, self.start_duration +
                                self.pre_move_duration + self.move_entering_duration + self.move_period_duration]  # list of times at which the screen should change


                # Default values for responses in case the subject makes no response
                response = None
                rt = None

                key = None

                trial_moves = []

                moves_found = False

                moves_to_enter = []

                while not moves_found:
                    random.shuffle(test_moves)
                    moves_to_enter = self.test_moves(start_state, 1, single_move=False)
                    # print moves_to_enter
                    if moves_to_enter is not False:
                        moves_found = True


                training_arrow_positions, training_arrows = self.create_arrows(len(moves_to_enter))

                training_move_positions = {}

                for n, i in enumerate(moves_to_enter):
                    training_arrows[n].image = self.arrow_images[i]
                    training_arrows[n].draw()
                    training_move_positions[i] = training_arrows[n].pos

                self.display_image.setImage(self.stimuli[start_state])
                self.display_image.draw()
                #self.arrow_display.image = self.arrow_images[key]
                # self.arrow_display.setImage(training_arrows)


                ## THIS IS WHERE THEY MAKE THE MOVE
                self.win.flip()
                key = event.waitKeys(keyList=moves_to_enter + ['escape', 'esc'])[0]
                start_state = self.moves_to_states([key], start_state)[0][1]


                self.circle.pos = training_move_positions[key]
                self.circle.draw()
                self.display_image.draw()
                for n, i in enumerate(moves_to_enter):
                    training_arrows[n].draw()
                self.win.flip()
                core.wait(1)

                # quit if subject pressed scape
                if key in ['escape', 'esc']:
                    core.quit()

    def run_task(self):

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



            # Starting state
            start_state = [0, 1, 11, 12, 8, 7, 5, 6]
            random.shuffle(start_state)
            start_state = start_state[0]
            self.display_image.setImage(self.stimuli[start_state])
            row = self.matrix_keys[start_state, :]
            states = None

            # rewards
            outcome = [0.12, 0.18, 0.23, 0.31, 0.35, 0.41, 0.44, 0.48, 0.56, 0.62, 0.74, 0.85, 0.91]
            random.shuffle(outcome)
            reward = [0] * self.n_moves

            shock_outcome = [0] * 13
            shock_outcome[0] = 1
            random.shuffle(shock_outcome)
            shock = [0] * self.n_moves


            # Default values for responses in case the subject makes no response
            response = None
            rt = None

            key = None

            trial_moves = []

            moves_found = False

            moves_to_enter = []

            while not moves_found:
                random.shuffle(test_moves)
                moves_to_enter = self.test_moves(start_state, self.n_moves)
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

                    raw_keys = event.getKeys(keyList=['left', 'right', 'up', 'down'], timeStamped=self.clock)

                    if len(raw_keys):
                        key, rt = raw_keys[0]

                    if key is not None and len(trial_moves) < self.n_moves:
                        self.arrow_display.image = self.arrow_images[key]

                        trial_moves.append(key)
                        key = None

                    if len(trial_moves) > 0:
                        if t < rt + 0.2:
                            self.arrow_display.draw()

                elif change_times[3] <= t < change_times[4]:
                    if states is None:
                        moves_states = self.moves_to_states(trial_moves, start_state)
                        # loop through their moves (trial_moves)
                    for n, key in enumerate(trial_moves):
                        self.arrow_progress[n].image = self.arrow_images[key]


                    #wrong moves or too few moves
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
                                shock[n] = shock_outcome[state]
                                self.show_move(reward[n], shock[n], self.stimuli[state], move, sum(reward), t,
                                               change_times[3] + n * self.move_duration + self.move_duration / 2.)
                                self.circle.pos = (self.arrow_positions[n], self.circle.pos[1])

                                self.circle.draw()

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

#experiment.run_task()
experiment.run_training()

