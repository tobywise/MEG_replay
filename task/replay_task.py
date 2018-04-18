# -*- coding: utf-8 -*-

from __future__ import division
from psychopy import core, visual, event, gui, monitors, data
import pandas as pd
import os, csv
import yaml
import numpy as np
import random
import networkx as nx
import json
import ctypes

class ParallelPort(object):

    def __init__(self, port=888, test=False):
        if not test:
            self._parallel = cytpes.WinDLL('')
        self.port = port

    def setData(self, data=0):

        if not test:
            self._parallel.outp(self.port, data)
        else:
            print "-"


class ReplayExperiment(object):

    def __init__(self, config=None):

        # Load config
        # this sets self.config (the config attribute of our experiment class) to a dictionary containing all the values
        # in our config file. The keys of this dictionary correspond to the section headings in the config file, and
        # each value is another dictionary with keys that refer to the subheadings in the config file. This means you can
        # reference things in the dictionary by e.g. self.config['heading']['subheading']

        with open(config) as f:
            self.config = yaml.load(f)

        # ------------------------------------#
        # Subject/task information and saving #
        # ------------------------------------#

        # Enter subject ID and other information
        dialogue = gui.Dlg()
        dialogue.addText("Subject info")
        dialogue.addField('Subject ID')
        dialogue.addField('Shock level')
        dialogue.addText("Task info")
        dialogue.addField('Mode', choices=['Experiment', 'Testing'])
        dialogue.addField('Instructions', initial=True)
        dialogue.addField('Training', initial=True)
        dialogue.addField('Test', initial=True)
        dialogue.addField('Task', initial=True)
        dialogue.show()

        # check that values are OK and assign them to variables
        if dialogue.OK:
            self.subject_id = dialogue.data[0]
            self.shock_level = dialogue.data[1]
            self.mode = dialogue.data[2]
            self._run_instructions = dialogue.data[3]
            self._run_training = dialogue.data[4]
            self._run_test = dialogue.data[5]
            self._run_main_task = dialogue.data[6]
        else:
            core.quit()

        # Set task mode - used for testing etc
        # 'Experiment' = normal, 'Testing' = show valid moves before move entering phase
        if self.mode == 'Testing':
            self.show_valid_moves = True
        else:
            self.show_valid_moves = False

        random.seed(self.subject_id)  # all randomness will be the same every time the subject does the task

        # This part sets up various things to allow us to save the data
        self.script_location = os.path.dirname(__file__)

        # Folder for saving data
        self.save_folder = self.config['directories']['saved_data']
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)

        # ------------- #
        # Task settings #
        # ------------- #

        self.n_moves = self.config['durations']['n_moves']  # number of moves subjects are expected to make
        self.n_training_trials = self.config['number training trials']['n_training_trials']  # number of training trials

        # Transition matrix
        self.matrix, self.matrix_keys, self.matrix_asgraph = self.create_matrix()

        # Number of trials in training phase
        self.n_training_trials = self.config['number training trials']['n_training_trials']

        # Number of moves that subjects make
        self.n_moves = self.config['durations']['n_moves']

        # Load trial information
        self.trial_info = pd.read_csv(self.config['directories']['trial_info'])
        self.trial_info = self.trial_info.round(2)  # ensures reward values are displayed nicely

        self.trial_info_test = pd.read_csv(self.config['directories']['trial_info_test'])

        # Check for missing data
        assert np.all(~self.trial_info.isnull())
        assert np.all(~self.trial_info_test.isnull())
        assert len(self.trial_info) > 0
        assert len(self.trial_info_test) > 0

        # Get reward and shock outcomes
        self.reward_info = self.trial_info[[c for c in self.trial_info.columns if 'reward' in c or c == 'trial_number']]
        self.shock_info = self.trial_info[[c for c in self.trial_info.columns if 'shock' in c or c == 'trial_number']]

        # Things to save
        self.data_keys = ['Subject', 'trial_number', 'Move_1', 'Move_2', 'Move_3', 'State_1', 'State_2', 'State_3',
                          'RT_1', 'RT_2', 'RT_3', 'Reward', 'Shock', 'trial_type']
        self.response_data = dict(zip(self.data_keys, [None for i in self.data_keys]))

        # Save file information
        self.save_folder = self.config['directories']['saved_data']
        self.save_prefix = self.config['filenames']['save_prefix']

        # -----------------------#
        # Monitor & window setup #
        # -----------------------#

        monitor = monitors.Monitor('test2', width=40.92, distance=74)
        monitor.setSizePix((1024, 768))
        self.win = visual.Window(monitor=monitor, size=(1024, 768), fullscr=True, allowGUI=False, color='gray', units='deg',
                            colorSpace='hex')
        self.win.mouseVisible = False  # make the mouse invisible

        # Keys used for making moves
        self.response_keys = self.config['response keys']['response_keys']


        # --------#
        # Stimuli #
        # --------#

        # Text stimuli #

        # Fixation cross
        self.fixation = visual.TextStim(win=self.win, height=0.8, color='white', text="+")
        # Text stimuli
        self.main_text = visual.TextStim(win=self.win, height=0.8, color='white',
                                         alignVert='center', alignHoriz='center', wrapWidth=30)
        self.main_text.fontFiles = [self.config['fonts']['font_path']]  # Arial is horrible
        self.main_text.font = self.config['fonts']['font_name']
        # Reward value text
        self.outcome_text = visual.TextStim(win=self.win, height=1.5, color='white',
                                         pos=(0, 7), wrapWidth=30)
        self.outcome_text.fontFiles = [self.config['fonts']['font_path']]
        self.outcome_text.font = self.config['fonts']['font_name']
        # Instruction text
        self.instruction_text = visual.TextStim(win=self.win, height=0.8, color='white', wrapWidth=30)
        self.instruction_text.fontFiles = [self.config['fonts']['font_path']]
        self.instruction_text.font = self.config['fonts']['font_name']


        # Image stimuli #

        # Image-related settings
        self.image_size = self.config['image sizes']['size_image_size']
        self.arrow_gap = self.config['arrow positions']['arrow_gap']  # gap between arrows

        # Find image files

        # Images used for displaying arrows
        self.arrow_images = {}
        for n in self.response_keys:
            self.arrow_images[n] = os.path.join(self.config['directories']['arrow_path'], '{0}.png'.format(n))

        # Circle used to indicate the current move
        self.circle = visual.Circle(win=self.win, radius=1, fillColor=None, lineColor=[1, 1, 1], pos=(0, -8))

        # State image files
        stimuli_location = self.config['directories']['stimuli_path']
        self.stimuli = [os.path.join(stimuli_location, i) for i in os.listdir(stimuli_location)
                   if ('.png' in i or '.jpg' in i or '.jpeg' in i) and 'shock' not in i]

        random.shuffle(self.stimuli)  # make sure stimuli are randomly assigned to states

        # Create some ImageStim instances

        # Image to display shock outcomes
        self.outcome_image = visual.ImageStim(win=self.win, size=self.config['image sizes']['size_outcome_image'],
                                              image='Stimuli/shock.png', pos=(0, 7))
        # State image
        self.display_image = visual.ImageStim(win=self.win, size=self.config['image sizes']['size_display_image'])
        # End state image in test phase
        self.display_image_test = visual.ImageStim(win=self.win, size=self.config['image sizes']['size_display_image'],
                                                   pos=(10, 0))
        # Image to display the key being pressed
        self.arrow_display = visual.ImageStim(win=self.win, size=self.config['image sizes']['size_arrow_display'])

        # Create the arrow stimuli
        self.arrow_positions, self.arrow_progress = self.create_arrows(self.n_moves)

        # Arrow used in test phase to indicate the end state
        self.test_arrow = visual.ImageStim(win=self.win, size=(1.5, 2),
                                           image=os.path.join(self.config['directories']['arrow_path'],
                                                              'right.png'))

        # Instruction text
        self.main_instructions = self.load_instructions(self.config['directories']['main_instructions'])
        self.training_instructions = self.load_instructions(self.config['directories']['training_instructions'])
        self.task_instructions = self.load_instructions(self.config['directories']['task_instructions'])
        self.test_instructions = self.load_instructions(self.config['directories']['test_instructions'])

    def run(self):

        """
        Runs the experiment
        """

        # Show main instructions
        print "A"
        if self._run_instructions:
            self.show_starting_instructions()
        print "B"
        # Run training
        if self._run_training:
            self.run_training()
        print "C"
        # Run test phase
        if self._run_test:
            self.run_test()

        # Run task
        if self._run_main_task:
            self.run_task()

        # Show end screen
        self.show_end_screen()

    def show_starting_instructions(self):

        """
        Used to show the starting instructions, calls the inner instructions function

        """

        self.grand_instructions(self.main_instructions)

    def show_end_screen(self):

        """
        Shows a screen for 10 seconds telling subjects that the experiment has ended

        """

        self.grand_instructions(['End of experiment'])
        self.win.flip()
        core.wait(10)

    def run_training(self):

        try:
            self.__run_training()
        except:
            self.save_json(1, 1, 'Crash', False, None, None, None, self.subject_id, stopped='Crash')
            raise

    def run_test(self):

        """
        Runs the test phase of the experiment, where subjects are shown a start state and have to navigate to an end state

        """

        try:
            self.__run_task(test=True, instructions=self.test_instructions, trial_info=self.trial_info_test)
        except:
            self.save_json(1, 1, 'Crash', False, None, None, None, self.subject_id, stopped='Crash')
            raise

    def run_task(self):

        """
        Runs the main task

        """

        try:
            self.__run_task(instructions=self.task_instructions, trial_info=self.trial_info)
        except:
            self.save_json(1, 1, 'Crash', False, None, None, None, self.subject_id, stopped='Crash')
            raise


    def __run_training(self):

        """
        Runs the training phase of the experiment

        """


        # Clock
        self.clock = core.Clock()

        self.start_duration = self.config['durations']['start_duration']
        self.pre_move_duration = self.config['durations']['pre_move_duration']
        self.move_entering_duration = self.config['durations']['move_entering_duration']
        self.move_duration = self.config['durations']['move_durations']
        self.move_period_duration = self.move_duration * self.n_moves

        test_moves = np.repeat(['up', 'down', 'left', 'right'], 5)

        if self.mode == 'Experiment':
            self.grand_instructions(self.training_instructions)

        start_state = [0]
        random.shuffle(start_state)
        start_state = start_state[0]
        self.display_image.setImage(self.stimuli[start_state])
        # row = self.matrix_keys[start_state, :]
        states = None

        for trial in range(self.n_training_trials):

            text = "Starting new trial"
            self.instructions(text)

            monitoring_saved = {'Training': False}

            for i in range(self.n_moves + 1):  # TRIAL LOOP - everything in here is repeated each trial

                print "Move {0} / {1}".format(i + 1, self.n_training_trials)

                # self.io.clearEvents('all')  # clear keyboard events

                continue_trial = True  # this variables changes to False when we want to stop the trial
                self.clock.reset()

                # Check whether we're at a terminal state
                if self.matrix[start_state, :].mean() == 0:
                    terminal = True
                else:
                    terminal = False

                moves_found = False
                moves_to_enter = []

                if not terminal:
                    while not moves_found:
                        random.shuffle(test_moves)
                        moves_to_enter = self.test_moves(start_state)
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

                if not terminal:
                    key = event.waitKeys(keyList=moves_to_enter + ['escape', 'esc'])[0]
                    if key in ['escape', 'esc']:
                        self.save_json(i + 1, 0, 'Rest', False, None, None, None, self.subject_id, stopped='Space')
                        core.quit()
                    start_state = self.moves_to_states([key], start_state)[0][1]
                    self.circle.pos = training_move_positions[key]
                    self.circle.draw()
                    for n, i in enumerate(moves_to_enter):
                        training_arrows[n].draw()

                    # if not monitoring_saved['Training']:
                    #     monitoring_saved['Training'] = self.save_json(i + 1, self.n_training_trials, 'Outcome only',
                    #                                                   True, start_state, None, None, self.subject_id)
                else:
                    start_state = 0

                self.display_image.draw()

                self.win.flip()
                core.wait(1)

                # quit if subject pressed scape
                if key in ['escape', 'esc']:
                    self.save_json(i + 1, 0, 'Rest', False, None, None, None, self.subject_id, stopped='Space')
                    core.quit()

            core.wait(1)


    def __run_task(self, test=False, instructions=None, trial_info=None):

        """
        Method used to run the main task - used by both the task and test phases

        Args:
            test: Whether this is being run as the test phase, boolean
            instructions: Instructions to show at the start of the task
            trial_info: Trial information

        Returns:

        """

        # Clock
        self.clock = core.Clock()

        # Durations
        self.start_duration = self.config['durations']['start_duration']
        self.pre_move_duration = self.config['durations']['pre_move_duration']
        self.move_entering_duration = self.config['durations']['move_entering_duration']
        self.move_duration = self.config['durations']['move_durations']
        self.move_period_duration = self.move_duration * self.n_moves

        test_moves = np.repeat(['up', 'down', 'left', 'right'], 5)

        # Show instructions for the actual task (not shown if in testing mode)
        if instructions is not None and not self.show_valid_moves:
            self.grand_instructions(instructions)

        end_state = None

        # Set up saving
        if not test:
            # Create the data file
            fname = '{0}/{1}_Subject{2}_{3}_behaviour.csv'.format(self.save_folder, self.save_prefix, self.subject_id,
                                                                  data.getDateStr())
            csvWriter = csv.writer(open(fname, 'wb'), delimiter=',').writerow
            csvWriter(self.data_keys)  # Write column headers

        # Number of successes in test phase
        n_successes = 0

        # Track which monitoring data we've saved
        monitoring_saved = {'Planning': False, 'Moves': False, 'Rest': False, 'Outcome': False, 'Pause': False}

        for i in range(len(trial_info)):  # TRIAL LOOP - everything in here is repeated each trial

            # In the test phase, break when subject gets enough trials correct
            if n_successes == self.config['number training trials']['n_test_successes']:
                break

            print "Trial {0} / {1}".format(i + 1, len(trial_info))

            # self.io.clearEvents('all')  # clear keyboard events

            continue_trial = True  # this variables changes to False when we want to stop the trial
            self.clock.reset()

            change_times = list(np.cumsum([0, self.start_duration, self.pre_move_duration, self.move_entering_duration,
                                           self.move_period_duration, self.config['durations']['rest_duration']]))

            outcome_only_change_times = list(np.cumsum([0, self.config['durations']['outcome_only_text_duration'],
                                                        self.config['durations']['outcome_only_duration'],
                                                        self.config['durations']['rest_duration']]))

            # Starting state
            start_state = 0
            self.display_image.setImage(self.stimuli[start_state])
            states = None

            # End state on test trials
            if test:
                end_state = trial_info['end_state'][i]
                self.display_image_test.setImage(self.stimuli[end_state])


            # If running the test phase, we want all outcome states except the correct one to say "incorrect"
            if test:
                outcome = [''] * (self.matrix.shape[0] - (self.reward_info.shape[1] - 1))
                outcome += ['Incorrect'] * (self.matrix.shape[0] - len(outcome))
                outcome[end_state] = 'Correct'
                shock_outcome = [0] * self.matrix.shape[0]

            else:
                # get reward values
                outcome = [''] * (self.matrix.shape[0] - (self.reward_info.shape[1] - 1))
                outcome += self.reward_info[[c for c in self.reward_info.columns if 'reward' in c]].iloc[i, :].tolist()
                shock_outcome = [0] * (self.matrix.shape[0] - (self.shock_info.shape[1] - 1))
                shock_outcome += self.shock_info[[c for c in self.shock_info.columns if 'shock' in c]].iloc[i, :].tolist()

            # Default values for responses in case the subject makes no response
            rt = None
            key = None
            trial_moves = []
            move_rts = []
            moves_found = False
            moves_to_enter = []
            valid_moves = False

            # Identifies and shows valid trajectories - used for testing
            if self.show_valid_moves:
                if not test:
                    end_state = np.random.randint(7, 11)
                while not moves_found:
                    random.shuffle(test_moves)
                    moves_to_enter = self.test_moves(start_state, end_state)
                    if moves_to_enter is not False:
                        moves_found = True
            else:
                moves_to_enter = ''

            # Key entering stage warning text
            key_text = 'Enter key movements\n{0}'.format(moves_to_enter)

            # Make sure the start state image is reset to the center of the screen
            self.display_image.setPos((0, 0))

            # Outcome only trials outcome state
            if trial_info['trial_type'][i] == 1:
                outcome_state = trial_info['end_state'][i]

            while continue_trial:  # run the trial

                t = self.clock.getTime()  # get the time

                # OUTCOME ONLY TRIALS
                if trial_info['trial_type'][i] == 1:

                    # Show text to indicate this is an outcome only trial
                    if t < outcome_only_change_times[1]:

                        text = "Outcome only"
                        self.instructions(text, max_wait=0)

                        # if not monitoring_saved['Outcome']:
                        #     monitoring_saved['Outcome'] = self.save_json(i+1, len(trial_info), 'Outcome only', None, None,
                        #                                                  outcome, shock_outcome, self.subject_id)

                    # Show outcome
                    elif outcome_only_change_times[1] <= t < outcome_only_change_times[2]:
                        outcome_only = outcome[outcome_state]
                        shock_only = shock_outcome[outcome_state]
                        self.show_move(outcome_only, shock_only, self.stimuli[outcome_state], t,
                                       outcome_only_change_times[1] + self.config['durations']['outcome_only_duration'] / 2.,
                                       show_moves=False)

                    # End trial
                    elif outcome_only_change_times[2] <= t < outcome_only_change_times[3]:
                        self.fixation.draw()

                    # Rest period
                    elif t >= outcome_only_change_times[3]:
                        continue_trial = False

                else:
                    # Show start state
                    if change_times[0] <= t < change_times[1]:
                        if test:
                            self.show_start_end_move()
                        else:
                            self.display_image.draw()

                        if not monitoring_saved['Planning']:
                            monitoring_saved['Planning'] = self.save_json(i+1, len(trial_info), 'Planning', None, None,
                                                                         outcome, shock_outcome, self.subject_id)

                    # Move entering warning
                    elif change_times[1] <= t < change_times[2]:
                        self.main_text.text = key_text
                        self.main_text.draw()
                        event.clearEvents()
                        self.display_image.setPos((0, 0))

                    # Move entering period
                    elif change_times[2] <= t < change_times[3]:

                        raw_keys = event.getKeys(keyList=['left', 'right', 'up', 'down'], timeStamped=self.clock)

                        if len(raw_keys):
                            key, rt = raw_keys[0]

                        if key is not None and len(trial_moves) < self.n_moves:
                            self.arrow_display.image = self.arrow_images[key]

                            trial_moves.append(key)
                            move_rts.append(rt)
                            key = None

                        if len(trial_moves) > 0:
                            if t < rt + 0.2:
                                self.arrow_display.draw()

                    # Show moves
                    elif change_times[3] <= t < change_times[4]:
                        if states is None:
                            moves_states = self.moves_to_states(trial_moves, start_state)
                            # loop through their moves (trial_moves)
                        for n, key in enumerate(trial_moves):
                            self.arrow_progress[n].image = self.arrow_images[key]


                        # Wrong moves or too few moves
                        if moves_states is False:
                            self.main_text.text = "Wrong moves entered"
                            self.main_text.draw()
                        elif len(trial_moves) < self.n_moves:
                            self.main_text.text = "Too few moves entered"
                            self.main_text.draw()
                        else:
                            valid_moves = True
                            for n, (move, state) in enumerate(moves_states):
                                if change_times[3] + n * self.move_duration <= t < change_times[3] + (n + 1) * self.move_duration:
                                    self.show_move(outcome[state], shock_outcome[state], self.stimuli[state], t,
                                                   change_times[3] + n * self.move_duration + self.move_duration / 2.)
                                    self.circle.pos = (self.arrow_positions[n], self.circle.pos[1])

                                    self.circle.draw()

                        # if not monitoring_saved['Moves']:
                        #     monitoring_saved['Moves'] = self.save_json(i+1, len(trial_info), 'Moves', valid_moves,
                        #                                                [i[1] for i in moves_states], outcome,
                        #                                                shock_outcome, self.subject_id)

                    # Rest period
                    elif change_times[4] <= t < change_times[5]:
                        self.fixation.draw()

                        # if not monitoring_saved['Rest']:
                        #     monitoring_saved['Rest'] = self.save_json(i+1, len(trial_info), 'Rest', valid_moves,
                        #                                                [i[1] for i in moves_states], outcome,
                        #                                                shock_outcome, self.subject_id)

                    # End trial
                    elif t >= change_times[-1]:

                        continue_trial = False

                # flip to draw everything
                self.win.flip()

                # If the trial has ended, save data to csv
                if not continue_trial:
                    # Responses
                    self.response_data['trial_number'] = i
                    self.response_data['Subject'] = self.subject_id
                    if trial_info['trial_type'][i] == 1:
                        for n in range(0, 3):
                            self.response_data['Move_{0}'.format(n + 1)] = np.nan
                            self.response_data['State_{0}'.format(n + 1)] = np.nan
                            self.response_data['RT_{0}'.format(n + 1)] = np.nan
                        self.response_data['Reward'] = outcome_only
                        self.response_data['Shock'] = shock_only
                    elif moves_states is not False:
                        for n, (move, state) in enumerate(moves_states):
                            self.response_data['Move_{0}'.format(n + 1)] = move
                            self.response_data['State_{0}'.format(n + 1)] = state
                            self.response_data['RT_{0}'.format(n + 1)] = move_rts[n]
                        self.response_data['Reward'] = outcome[state]
                        self.response_data['Shock'] = shock_outcome[state]
                    self.response_data['trial_type'] = trial_info['trial_type'][i]
                    if not test:
                        csvWriter([self.response_data[category] for category in self.data_keys])  # Write data
                    else:
                        # In the test phase, figure out whether they got to the correct state
                        if moves_states is not False and moves_states[-1][1] == end_state:
                            n_successes += 1
                        else:
                            n_successes = 0

                        print n_successes

                if event.getKeys(["space", ' ']):
                    self.instruction_text.setText("Experiment paused")
                    self.instruction_text.draw()
                    self.win.flip()
                    if not monitoring_saved['Pause']:
                        monitoring_saved['Pause'] = self.save_json(i + 1, len(trial_info), 'Pause', valid_moves,
                                                                  None, None, None, self.subject_id, stopped='Space')
                    event.waitKeys(['space', ' '])

                # quit if subject pressed scape
                if event.getKeys(["escape"]):
                    self.save_json(i + 1, len(trial_info), 'Escape', valid_moves, None, None, None, self.subject_id,
                                   stopped='Escape')
                    core.quit()


    def load_instructions(self, text_file):

        """
        Loads a text file containing instructions and splits it into a list of strings

        Args:
            text_file: A text file containing instructions, with each page preceded by an asterisk

        Returns: List of instruction strings

        """

        with open(text_file, 'r') as f:
            instructions = f.read()

        return instructions.split('*')

    def create_arrows(self, n):

        """
        Creates a list of arrow stimuli representing moves the subject has made. Arrow directions are not determined at
        this point - this is just a correctly positioned list of arrow stimuli to be given images later.

        Args:
            n: number of arrows to show

        Returns:
            Positions of the arrows
            Arrow stimuli in correct positions

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


    def show_move(self, outcome, shock, picture, t, shock_time, show_moves=True):

        """
        Shows the image and (potentially) outcome associated with a state

        Args:
            outcome: The reward value of the state
            shock: Whether the state is associated with a shock (binary)
            picture: The image to be displayed (path to the image)
            t: Current time
            shock_time: Time at which the shock outcome should be displayed (this occurs after the reward outcome)
            show_moves: Whether or not to show the moves leading to a particular state underneath the image (boolean)


        """

        # set image
        self.display_image.image = picture

        # set outcome text either as value or shock
        #self.outcome_image.image = outcome

        if t <= shock_time:
            self.outcome_text.text = outcome
            self.outcome_text.draw()
        elif t > shock_time and shock == 1:
            self.outcome_image.draw()

        if show_moves:
            for i in range(self.n_moves):
                self.arrow_progress[i].draw()

        # draw everything
        self.display_image.draw()

    def instructions(self, text, max_wait=2):

        """
        Shows instruction text

        Args:
            text: Text to display
            max_wait: The maximum amount of time to wait for a response before moving on

        Returns:

        """

        # set text
        self.instruction_text.text = text
        # draw
        self.instruction_text.draw()
        if max_wait > 0:
            self.win.flip()
        # waitkeys
        event.waitKeys(maxWait=max_wait, keyList='space')

    def grand_instructions(self, text_file):

        """
        Displays instruction text files for main instructions, training instructions, task instructions

        Args:
            text_file: A list of strings

        Returns:

        """

        if not isinstance(text_file, list):
            raise TypeError("Input is not a list")

        for i in text_file:
            self.instruction_text.text = i
            self.instruction_text.draw()
            self.win.flip()
            core.wait(1)
            key = event.waitKeys(keyList=['space', 'escape', 'esc'])
            if key[0] in ['escape', 'esc']:
                core.quit()

    def create_matrix(self):

        """
        Assigns keys to the transition matrix provided in the config file

        Returns: The matrix and a matrix of keys subjects can press from each state, and where that key takes them

        """

        matrix = np.loadtxt(self.config['directories']['matrix'])


        matrix_keys = matrix.astype(int).astype(str)
        keys = ['up', 'down', 'left', 'right']
        random.shuffle(keys)

        for i in range(matrix.shape[0]):
            one_idx = np.where(matrix[i, :] == 1)
            random.shuffle(keys)
            for j in range(len(one_idx[0])):
                matrix_keys[i, one_idx[0][j]] = keys[j]

        G = nx.DiGraph(matrix)

        return matrix, matrix_keys, G


    def moves_to_states(self, trial_moves, start):

        """
        Converts a series of moves to a sequence of states

        Args:
            trial_moves: The moves made by the subject
            start: The starting state

        Returns:
            A list of tuples of the form (move made by the subject, the state this move took them to)

        """

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

    def test_moves(self, start, end=None):

        """
        For a given start and end state, works out a series of allowable moves. If end is None, returns all possible moves
        from the current state

        Args:
            start: Starting state
            end: End state

        Returns:
            A list of allowed moves

        """

        moves_states = []

        if end is not None:
            path = nx.shortest_path(self.matrix_asgraph, start, end)

            for n, i in enumerate(path[:-1]):
                moves_states.append(self.matrix_keys[i, path[n + 1]])

        else:
            neighbours = self.matrix_asgraph.neighbors(start)
            for i in neighbours:
                moves_states.append(self.matrix_keys[start, i])

        return moves_states


    def show_start_end_move(self):

        """
        Displays both the start and end move, used for the test phase

        """

        self.display_image.setPos((-10, 0))
        self.display_image.draw()
        self.display_image_test.draw()
        self.test_arrow.draw()

    def save_json(self, trial_number, total_trials, phase, valid_moves, moves, reward, shock, subject, stopped="None"):

        json_data = {"Trial": trial_number, "Total_trials": total_trials, "Stopped": stopped, "Phase": phase,
                     "Valid": valid_moves, "Moves": moves, "Reward": reward, "Shock": shock,  "Subject": subject}

        try:
            with open('//cher/twise/web/replay_task_output_json.txt', 'w') as f:
                json.dump(json_data, f)
            return True
        except:
            pass


## RUN THE EXPERIMENT

# Create experiment class
experiment = ReplayExperiment('replay_task_settings.yaml')

# Run the experiment
experiment.run()

