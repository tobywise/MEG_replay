# -*- coding: utf-8 -*-

from __future__ import division
from psychopy import core, visual, event, gui, monitors, data, logging
import pandas as pd
import os, csv
import yaml
import numpy as np
import random
import networkx as nx
import json
import ctypes
import time
import copy
import re
import warnings


class ParallelPort(object):

    """
    Used to interact with the parallel port. If no parallel port is present, just prints triggers

    """

    def __init__(self, port=888, trigger_record_file=None):

        """
        Args:
            port: Parallel port number

        """

        try:
            self._parallel = ctypes.WinDLL('simpleio.dll')
            self.test = False
        except:
            self.test = True
            warnings.warn("NO PARALLEL PORT FOUND: RUNNING IN TEST MODE")

        self.port = port
        self.value = 0

        self.trigger_record_file = trigger_record_file

        # Send 0 when initialising
        if not self.test:
            self._parallel.outp(self.port, self.value)


    def setData(self, data=0):

        if data != self.value:

            if not self.test:
                self._parallel.outp(self.port, data)
                print "-- Sending value {0} to parallel port -- ".format(data)
                if self.trigger_record_file:
                    self.trigger_record_file.write("-- Sending value {0} to parallel port -- \n".format(data))
            else:
                print "-- Sending value {0} to parallel port -- ".format(data)
                if self.trigger_record_file:
                    self.trigger_record_file.write("-- Sending value {0} to parallel port -- \n".format(data))

            self.value = data



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
        dialogue.addField('Monitor', initial=False)
        dialogue.addField('MEG', initial=False)
        dialogue.addField('Show instructions', initial=True)
        dialogue.addField('Show photodiode square', initial=False)
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
            self.monitoring = dialogue.data[7]
            self.MEG_mode = dialogue.data[8]
            self.show_instructions = dialogue.data[9]
            self.photodiode = dialogue.data[10]
        else:
            core.quit()

        # Set task mode - used for testing etc
        # 'Experiment' = normal, 'Testing' = show valid moves before move entering phase
        if self.mode == 'Testing':
            self.testing_mode = True
        else:
            self.testing_mode = False

        # MEG mode
        if self.MEG_mode:
            self.durations = 'MEG_durations'
        else:
            self.durations = 'durations'

        # Recode blank subject ID to zero - useful for testing
        if self.subject_id == '':
            self.subject_id = '0'

        random.seed(int(re.search('\d+', self.subject_id).group()))  # all randomness will be the same every time the subject does the task

        # This part sets up various things to allow us to save the data
        self.script_location = os.path.dirname(__file__)

        # Folder for saving data
        self.save_folder = self.config['directories']['saved_data']
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)

        # ------------- #
        # Task settings #
        # ------------- #

        self.n_moves = self.config[self.durations]['n_moves']  # number of moves subjects are expected to make
        self.n_training_trials = self.config['number training trials']['n_training_trials']  # number of training trials
        self.n_test_trials = self.config['number training trials']['n_test_trials']  # number of training trials

        # Transition matrix
        self.matrix, self.matrix_keys, self.matrix_asgraph = self.create_matrix()

        # Number of trials in training phase
        self.n_training_trials = self.config['number training trials']['n_training_trials']

        # Number of moves that subjects make
        self.n_moves = self.config[self.durations]['n_moves']

        # Load trial information
        self.trial_info = pd.read_csv(self.config['directories']['trial_info'])
        self.trial_info = self.trial_info.round(2)  # ensures reward values are displayed nicely
        self.trial_info_test = pd.read_csv(self.config['directories']['trial_info_test'])

        # Number of trials per block
        self.trials_per_block = self.config[self.durations]['trials_per_block']

        # Check for missing data
        for c in self.trial_info.columns:
            self.trial_info[c][self.trial_info[c].isnull()] = 0

        print self.trial_info

        assert np.all(~self.trial_info.isnull())
        assert np.all(~self.trial_info_test.isnull())
        assert len(self.trial_info) > 0
        assert len(self.trial_info_test) > 0

        # Get reward and shock outcomes
        self.reward_info = self.trial_info[[c for c in self.trial_info.columns if 'reward' in c or c == 'trial_number']]
        self.shock_info = self.trial_info[[c for c in self.trial_info.columns if 'shock' in c or c == 'trial_number']]

        # Get maximum available reward
        self.max_reward = self.get_max_reward()
        print "Maximum available reward = {0}".format(self.max_reward)

        # Things to save
        self.data_keys = ['Subject', 'trial_number', 'State_1', 'State_2', 'State_3', 'outcome_type',
                          'RT_1', 'RT_2', 'RT_3', 'Reward_received', 'Shock_received', 'trial_type',
                          'State_1_shock', 'State_2_shock']
        self.response_data = dict(zip(self.data_keys, [None for i in self.data_keys]))

        # Save file information
        self.save_folder = self.config['directories']['saved_data']
        self.save_prefix = self.config['filenames']['save_prefix']

        # -----------------------#
        # Monitor & window setup #
        # -----------------------#

        monitor = monitors.Monitor('test2', width=40.92, distance=74)
        monitor.setSizePix((1024, 768))
        self.win = visual.Window(monitor=monitor, size=(1024, 768), fullscr=True, allowGUI=False, color='#616161',
                                 units='deg',
                                 colorSpace='hex')
        # self.win.mouseVisible = False  # make the mouse invisible

        # self.win.recordFrameIntervals = True

        self.mouse = event.Mouse()

        self.win.refreshThreshold = 1 / 60. + 0.004  # for checking for dropped frames

        # Set the log module to report warnings to the standard output window
        # (default is errors only).
        logging.console.setLevel(logging.WARNING)

        # Keys used for making moves
        self.response_keys = self.config['response keys']['response_keys']
        self.response_phases = self.config['response keys']['response_phases']  # levels of the tree

        # Set up parallel port

        # File to save triggers
        self.trigger_record_file = open('{0}/{1}_Subject{2}_{3}_triggers.txt'.format(self.save_folder, self.save_prefix, self.subject_id,
                                                                    data.getDateStr()), 'w')

        self.parallel_port = ParallelPort(port=888, trigger_record_file=self.trigger_record_file)
        self.n_shocks = self.config[self.durations]['n_shocks']
        self.shock_delay = self.config[self.durations]['shock_delay']

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

        # Find image files

        # Circle used to indicate the current move
        self.circle = visual.Circle(win=self.win, radius=1, fillColor=None, lineColor=[1, 1, 1], pos=(0, -8))

        # State image files
        stimuli_location = self.config['directories']['stimuli_path']
        self.stimuli = [os.path.join(stimuli_location, i) for i in os.listdir(stimuli_location)
                        if ('.png' in i or '.jpg' in i or '.jpeg' in i) and 'shock' not in i][:self.matrix.shape[0]]

        # Save stimulu order
        stim_fname = '{0}/{1}_Subject{2}_{3}_localiser_stimuli.txt'.format(self.save_folder, self.save_prefix, self.subject_id,
                                                                    data.getDateStr())

        with open(stim_fname, 'wb') as f:
            f.write(str(self.stimuli))


        random.shuffle(self.stimuli)  # make sure stimuli are randomly assigned to states

        # State selection images
        self.state_selection_images = []  # list of tuples, (image id, imagestim)

        for i in range(4):
            pos = (((i % 4) - 1.5) * self.config['image sizes']['state_selection_spacing'], 0)
            self.state_selection_images.append((i, visual.ImageStim(win=self.win,
                                                                    size=self.config['image sizes'][
                                                                        'size_selection_image'],
                                                                    pos=pos),
                                                visual.TextStim(win=self.win, pos=pos, height=5, color='yellow')))

        self.state_selection_dict = None

        # Create some ImageStim instances

        # Image to display shock outcomes
        self.outcome_image = visual.ImageStim(win=self.win, size=self.config['image sizes']['size_outcome_image'],
                                              image=self.config['stimuli']['shock'], pos=(0, 7))
        self.outcome_image_noshock = visual.ImageStim(win=self.win, size=self.config['image sizes']['size_outcome_image_noshock'],
                                              image=self.config['stimuli']['noshock'], pos=(0, 7))
        # State image
        self.display_image = visual.ImageStim(win=self.win, size=self.config['image sizes']['size_display_image'])
        # End state image in test phase
        self.display_image_test = visual.ImageStim(win=self.win, size=self.config['image sizes']['size_display_image'],
                                                   pos=(10, 0))

        # Arrow used in test phase to indicate the end state
        self.test_arrow = visual.ImageStim(win=self.win, size=(1.5, 2),
                                           image=os.path.join(self.config['directories']['arrow_path'],
                                                              'right.png'))

        # Instruction text
        self.main_instructions = self.load_instructions(self.config['directories']['main_instructions'])
        self.training_instructions = self.load_instructions(self.config['directories']['training_instructions'])
        self.task_instructions = self.load_instructions(self.config['directories']['task_instructions'])
        self.test_instructions = self.load_instructions(self.config['directories']['test_instructions'])

        # Photodiode things
        self.photodiode_square = visual.Rect(self.win, 1, 1, fillColor='black', pos=(-15.5, -9.5))
        if self.photodiode:
            self.photodiode_square.opacity = 0

    def run(self):

        """
        Runs the experiment
        """

        test_passed = False

        # Show main instructions
        if self._run_instructions:
            self.show_starting_instructions()

        # Run training
        if self._run_training:
            self.run_training()
            core.wait(1)

        # Run test phase
        if self._run_test:
            self.move_entering_duration = self.config[self.durations]['move_entering_duration_initial']
            self.move_entering_duration_step = (self.config[self.durations]['move_entering_duration_initial'] -
                                                self.config[self.durations]['move_entering_duration'])  / \
                                               float(self.config[self.durations]['move_entering_reduction_length'])
            test_passed = self.run_test(show_instructions=self.show_instructions)
            core.wait(1)

        # Rerun training and test if not passed, until they pass the test
        if self._run_training and self._run_test:
            while not test_passed:
                self.run_training(show_instructions=False, show_intro=True)
                core.wait(1)
                test_passed = self.run_test(show_instructions=False, show_intro=True)
                core.wait(1)
                print "TEST DONE"
                print test_passed

        # Run task
        if self._run_main_task:
            self.move_entering_duration = self.config[self.durations]['move_entering_duration']
            self.run_task(show_instructions=self.show_instructions, show_intro=True)

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

        self.grand_instructions(['End of experiment\n'])
        self.win.flip()
        core.wait(10)

    def run_training(self, show_instructions=True, show_intro=False):

        try:
            self.__run_training(show_instructions=show_instructions, show_intro=show_intro)
        except:
            if self.monitoring:
                self.save_json(1, 1, 'Crash', False, None, None, None, self.subject_id, stopped='Crash')
            raise

    def run_test(self, show_instructions=True, show_intro=False):

        """
        Runs the test phase of the experiment, where subjects are shown a start state and have to navigate to an end state

        """

        try:
            result = self.__run_task(test=True, instructions=self.test_instructions, trial_info=self.trial_info_test,
                                     show_instructions=show_instructions, show_intro=show_intro)
            return result
        except:
            if self.monitoring:
                self.save_json(1, 1, 'Crash', False, None, None, None, self.subject_id, stopped='Crash')
            raise

    def run_task(self, show_instructions=True, show_intro=True):

        """
        Runs the main task

        """

        try:
            self.__run_task(instructions=self.task_instructions, trial_info=self.trial_info,
                            show_instructions=show_instructions, show_intro=show_intro)
        except:
            if self.monitoring:
                self.save_json(1, 1, 'Crash', False, None, None, None, self.subject_id, stopped='Crash')
            raise

    def __run_training(self, show_instructions=True, show_intro=False):

        """
        Runs the training phase of the experiment

        """

        # Clock
        self.clock = core.Clock()

        self.start_duration = self.config[self.durations]['start_duration']
        self.pre_move_duration = self.config[self.durations]['pre_move_duration']
        self.move_entering_duration = self.config[self.durations]['move_entering_duration']
        self.move_durations = self.config[self.durations]['move_durations']
        self.cumulative_move_durations = np.cumsum([0] + self.move_durations)
        self.move_period_duration = np.sum(self.move_durations)

        test_moves = np.repeat(['up', 'down', 'left', 'right'], 5)

        if self.mode == 'Experiment' and show_instructions:
            self.grand_instructions(self.training_instructions)

        if show_intro:
            self.grand_instructions(["Starting training phase, press space to begin"])

        for trial in range(self.n_training_trials):

            text = "Starting new trial"
            self.instructions(text)

            monitoring_saved = {'Training': False}

            moves = []

            current_state = 0
            self.display_image.setImage(self.stimuli[current_state])

            for i in range(self.n_moves + 1):  # TRIAL LOOP - everything in here is repeated each trial

                print "Move {0} / {1}".format(i + 1, self.n_training_trials)

                pos_selected = []
                move_rts = []

                # self.io.clearEvents('all')  # clear keyboard events

                continue_trial = True  # this variables changes to False when we want to stop the trial
                self.clock.reset()

                # Check whether we're at a terminal state
                if self.matrix[current_state, :].mean() == 0:
                    terminal = True
                else:
                    terminal = False

                if not terminal:
                    valid_states = True
                else:
                    valid_states = False

                self.display_image.setImage(self.stimuli[current_state])
                self.display_image.draw()

                ## THIS IS WHERE THEY MAKE THE MOVE
                self.win.flip()
                core.wait(1.3)

                if trial == 0:
                    self.setup_state_selection_grid(random_positions=True, valid=valid_states, test=self.testing_mode,
                                                    initial_state=current_state)
                else:
                    self.setup_state_selection_grid(random_positions=False, valid=valid_states, test=self.testing_mode,
                                                    initial_state=current_state)
                core.wait(0.5)


                self.win.mouseVisible = True

                if not terminal:

                    # If a key is pressed, work out which state and position was selected
                    self.draw_state_selection_grid(selected=pos_selected, test=self.testing_mode)
                    self.win.flip()

                    raw_keys = event.waitKeys(timeStamped=self.clock, keyList=self.valid_moves + ['escape', 'esc'])

                    if len(raw_keys) and len(moves) < 3:

                        key, rt = raw_keys[0]

                        if key in ['escape', 'esc']:
                            # self.save_json(trial + 1, 0, 'Rest', False, None, None, None, self.subject_id,
                            #                stopped='Space')
                            core.quit()
                            self.trigger_record_file.close()

                        # get selected state
                        phase = len(moves)
                        try:
                            phase = len(moves)
                            selected_state = self.phase_key_state_mapping[phase][key]
                            moves.append(selected_state)
                            move_rts.append(rt)

                            # get selected state position on grid
                            pos = self.state_selection_dict[selected_state]
                            pos_selected.append(pos)
                            current_state = selected_state
                        except Exception as e:  # if the chosen key doesn't exist in the dictionary for this phase
                            pass

                    # if not monitoring_saved['Training']:
                    #     monitoring_saved['Training'] = self.save_json(i + 1, self.n_training_trials, 'Outcome only',
                    #                                                   True, start_state, None, None, self.subject_id)


                # self.display_image.draw()
                self.win.flip()
                core.wait(1)

            core.wait(1)

    def __run_task(self, test=False, instructions=None, trial_info=None, show_instructions=True, show_intro=False):

        """
        Method used to run the main task - used by both the task and test phases

        Args:
            test: Whether this is being run as the test phase, boolean
            instructions: Instructions to show at the start of the task
            trial_info: Trial information

        Returns:

        """

        # Shuffle test trial info

        trial_info = trial_info.sample(frac=1)

        self.send_trigger(0, False)

        # Clock
        self.clock = core.Clock()

        # Durations
        if test:
            # Shorter planning and rest periods in test phase
            self.start_duration = self.config[self.durations]['start_duration_test']
            self.rest_duration = self.config[self.durations]['rest_duration_test']
        else:
            self.start_duration = self.config[self.durations]['start_duration']
            self.rest_duration = self.config[self.durations]['rest_duration']
        self.pre_move_duration = self.config[self.durations]['pre_move_duration']
        # self.move_entering_duration = self.config[self.durations]['move_entering_duration']
        self.pre_move_fixation_duration = self.config[self.durations]['pre_move_fixation_duration']
        self.move_durations = self.config[self.durations]['move_durations']
        self.cumulative_move_durations = np.cumsum([0] + self.move_durations)
        self.move_period_duration = np.sum(self.move_durations)
        self.shock_symbol_delay = self.config[self.durations]['shock_symbol_delay']

        test_moves = np.repeat(['up', 'down', 'left', 'right'], 5)

        # Show instructions for the actual task (not shown if in testing mode)
        if instructions is not None and not self.testing_mode and show_instructions:
            self.grand_instructions(instructions)
            self.win.flip()

        if show_intro:
            if test:
                self.grand_instructions(["Starting test phase, press 1 to begin"])
                self.win.flip()
            else:
                self.grand_instructions(['Starting task'])
                self.win.flip()

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

        # Rewards collected
        self.reward_value = 0.0

        # core.wait(2)  # let things load before starting

        if not test:
            n_trials = len(trial_info)
        else:
            n_trials = self.n_test_trials

        previously_shocked = False

        for i in range(n_trials):  # TRIAL LOOP - everything in here is repeated each trial

            print "Trial {0} / {1}".format(i + 1, n_trials)
            self.trigger_record_file.write("Trial {0} / {1}".format(i + 1, n_trials))

            self.send_trigger(0, False)

            # self.io.clearEvents('all')  # clear keyboard events

            continue_trial = True  # this variable changes to False when we want to stop the trial

            change_times = list(np.cumsum([0, self.start_duration, self.pre_move_duration, self.move_entering_duration,
                                           self.pre_move_fixation_duration, self.move_period_duration,
                                           self.rest_duration]))

            outcome_only_change_times = list(np.cumsum([0, self.config[self.durations]['outcome_only_text_duration'],
                                                        self.config[self.durations]['outcome_only_duration'],
                                                        self.config[self.durations]['rest_duration']]))

            # Track which monitoring data we've saved
            monitoring_saved = {'Planning': False, 'Moves': False, 'Rest': False, 'Outcome': False, 'Pause': False}

            # Starting state
            start_state = 0
            self.display_image.setImage(self.stimuli[start_state])
            states = None
            self.outcome_text.text = ''

            # End state on test trials
            if test:
                end_state = trial_info['end_state'][i]
                self.display_image_test.setImage(self.stimuli[end_state])

            # If running the test phase, we want all outcome states except the correct one to say "incorrect"
            if test:
                outcome = [''] * (self.matrix.shape[0] - (self.reward_info.shape[1] - 1))
                outcome += ['Incorrect'] * (self.matrix.shape[0] - len(outcome))
                outcome[end_state] = 'Correct'
                shock_outcome = [None] * self.matrix.shape[0]

            else:
                # get reward values
                outcome = [''] * (self.matrix.shape[0] - (self.reward_info.shape[1] - 1))
                outcome += self.reward_info[[c for c in self.reward_info.columns if 'reward' in c]].iloc[i, :].tolist()
                shock_outcome = [None] * (self.matrix.shape[0] - (self.shock_info.shape[1] - 1))
                shock_outcome += self.shock_info[[c for c in self.shock_info.columns if 'shock' in c]].iloc[i,
                                 :].tolist()
                print shock_outcome
                # shock_outcome = [1] * self.matrix.shape[0]

            # Default values for responses in case the subject makes no response
            move_rts = []
            moves_found = False
            moves_to_enter = []
            valid_moves = False
            moves_validated = False
            self.shocks_given = 0
            outcome_type = 'moves_entered'

            # Generated moves for incorrect trials
            moves = None

            # Setup selection grid
            #self.setup_state_selection_grid(random_positions=True, test=self.testing_mode)
            setup_check_grid = [True, False, False]
            self.setup_state_selection_grid(random_positions=True, test=self.testing_mode)
            selected_state = 0
            pos = 10

            moves = []
            moves_states = []
            pos_selected = []  # image positions that have been selected - these are faded out on the grid
            too_few_moves = False

            # Identifies and shows valid trajectories - used for testing
            if self.testing_mode:
                if not test:
                    end_state = np.random.randint(5, 7)
                while not moves_found:
                    random.shuffle(test_moves)
                    moves_to_enter, _ = self.test_moves(start_state, end_state)
                    if moves_to_enter is not False:
                        moves_found = True
            else:
                moves_to_enter = ''

            # Key entering stage warning text
            key_text = 'Get ready to enter key movements\n{0}'.format(moves_to_enter)

            # Make sure the start state image is reset to the center of the screen
            self.display_image.setPos((0, 0))

            if test:
                trial_type = 0
            else:
                trial_type = trial_info['trial_type'][i]

            # Outcome only trials outcome state
            outcome_state = trial_info['end_state'][i]

            # Trigger dict - lets us know whether we've sent triggers yet
            self.trigger_dict = {'Planning': False, 'Move_entering': False, 'State_0': False, 'State_1': False,
                            'State_2': False, 'NoShock': False, 'Trial_start': False,
                            'State_3': False, 'Shock': False, 'Rest': False, 'Outcome_only_warning': False,
                            'Outcome_only_outcome': False}


            if not i % self.trials_per_block and self.MEG_mode and not test:
                if i == 0:
                    self.grand_instructions(["We are about to start the experiment"])
                else:
                    self.grand_instructions(["Take a break"])
                self.grand_instructions(["We are about to start a new block, please keep as "
                                         "still as possible for the duration of the block"])

            self.clock.reset()

            # RUN THE TRIAL

            self.send_trigger(74, self.trigger_dict['Trial_start'])
            self.send_trigger(74, self.trigger_dict['Trial_start'])

            while continue_trial:

                t = self.clock.getTime()  # get the time

                # OUTCOME ONLY TRIALS
                if trial_type == 1:

                    # Show text to indicate this is an outcome only trial
                    if t < outcome_only_change_times[1]:

                        text = "Outcome only"
                        self.instructions(text, max_wait=0)

                        self.send_trigger(self.config['triggers']['outcome_only_warning'],
                                          self.trigger_dict['Outcome_only_warning'])
                        self.trigger_dict['Outcome_only_warning'] = True

                        if not monitoring_saved['Outcome'] and self.monitoring:
                            monitoring_saved['Outcome'] = self.save_json(i + 1, len(trial_info), 'Outcome', True,
                                                                         [int(outcome_state)],
                                                                         outcome, shock_outcome, self.subject_id)

                        if self.MEG_mode and self.photodiode:
                            if t < 0.5:
                                self.photodiode_square.fillColor = 'white'
                                self.photodiode_square.draw()
                            else:
                                self.photodiode_square.fillColor = 'black'

                    # Show outcome
                    elif outcome_only_change_times[1] <= t < outcome_only_change_times[2]:
                        outcome_only = outcome[outcome_state]
                        shock_only = shock_outcome[outcome_state]
                        self.show_move(outcome_only, shock_only, self.stimuli[outcome_state], t,
                                       outcome_only_change_times[1] + self.config[self.durations][
                                           'outcome_only_duration'] / 2.,
                                       shock_delay=self.shock_delay)

                        self.send_trigger(self.config['triggers']['outcome_only_outcome'],
                                          self.trigger_dict['Outcome_only_outcome'])
                        self.trigger_dict['Outcome_only_outcome'] = True

                        if self.MEG_mode and self.photodiode:
                            if change_times[1] < t < change_times[1] + 0.5:
                                self.photodiode_square.fillColor = 'white'
                                self.photodiode_square.draw()
                            else:
                                self.photodiode_square.fillColor = 'black'

                    # End trial
                    elif outcome_only_change_times[2] <= t < outcome_only_change_times[3]:
                        self.fixation.draw()

                    # Rest period
                    elif t >= outcome_only_change_times[3]:
                        self.send_trigger(self.config['triggers']['rest'], self.trigger_dict['Rest'])
                        self.trigger_dict['Rest'] = True
                        continue_trial = False

                else:
                    # Show start state
                    if change_times[0] <= t < change_times[1]:
                        if test:
                            self.show_start_end_move()
                        else:
                            self.display_image.draw()
                        if not monitoring_saved['Planning'] and self.monitoring:
                            monitoring_saved['Planning'] = self.save_json(i + 1, len(trial_info), 'Planning', None,
                                                                          None,
                                                                          outcome, shock_outcome, self.subject_id)

                        self.send_trigger(self.config['triggers']['planning'], self.trigger_dict['Planning'])
                        self.trigger_dict['Planning'] = True

                        if self.MEG_mode and self.photodiode:
                            if change_times[0] < t < change_times[0] + 0.5:
                                self.photodiode_square.fillColor = 'white'
                                self.photodiode_square.draw()
                            else:
                                self.photodiode_square.fillColor = 'black'

                    # Move entering warning
                    elif change_times[1] <= t < change_times[2]:
                        self.main_text.text = key_text
                        self.main_text.draw()
                        event.clearEvents()
                        self.display_image.setPos((0, 0))

                        self.send_trigger(self.config['triggers']['move_entering'], self.trigger_dict['Move_entering'])
                        self.trigger_dict['Move_entering'] = True

                        if self.MEG_mode and self.photodiode:
                            if change_times[1] < t < change_times[1] + 0.5:
                                self.photodiode_square.fillColor = 'white'
                                self.photodiode_square.draw()
                            else:
                                self.photodiode_square.fillColor = 'black'

                    # Move entering period
                    elif change_times[2] <= t < change_times[3]:

                        for n in range(self.n_moves):
                            if change_times[2] + n * self.move_entering_duration / 3. <= t < \
                                    change_times[2] + (n + 1) * self.move_entering_duration / 3.:
                                if not setup_check_grid[n]:
                                    pos = 10
                                    self.setup_state_selection_grid(random_positions=True, test=self.testing_mode,
                                                                    initial_state=selected_state, assign_keys=False)
                                    setup_check_grid[n] = True
                                # Draw state selection grid
                                if selected_state > 0 or n == 0:  # only show if the previous move was made
                                    self.draw_state_selection_grid(selected=[pos], test=self.testing_mode)

                                if len(moves) == n:
                                    # Watch for key presses
                                    # raw_keys = event.getKeys(keyList=self.response_keys, timeStamped=self.clock)
                                    raw_keys = event.getKeys(timeStamped=self.clock)

                                    # If a key is pressed, work out which state and position was selected
                                    if len(raw_keys) and len(moves) < self.n_moves:

                                        key, rt = raw_keys[0]

                                        # get selected state
                                        phase = len(moves)

                                        try:
                                            selected_state = self.phase_key_state_mapping[phase][key]
                                            moves.append(selected_state)
                                            move_rts.append(rt)

                                            # get selected state position on grid
                                            pos = self.state_selection_dict[selected_state]
                                            pos_selected.append(pos)

                                        except:  # if the chosen key doesn't exist in the dictionary for this phase
                                            pass


                        event.clearEvents()

                        if self.MEG_mode and self.photodiode:
                            if change_times[2] < t < change_times[2] + 0.5:
                                self.photodiode_square.fillColor = 'white'
                                self.photodiode_square.draw()
                            else:
                                self.photodiode_square.fillColor = 'black'

                    elif change_times[3] <= t < change_times[4]:  # Fixation before moves are shown

                        self.fixation.draw()

                    # Show moves

                    elif change_times[4] <= t < change_times[5]:

                        # Validate moves
                        if not moves_validated:
                            valid_moves = self.validate_moves([0] + moves)
                            moves_validated = True
                            if len(moves) < self.n_moves: too_few_moves = True


                        # Show generated moves and warning if too few moves entered
                        if too_few_moves:

                            if test:
                                self.main_text.text = "Too few moves entered"
                                self.main_text.draw()

                            else:
                                if len(moves) < self.n_moves:
                                    self.outcome_text.text = "Too few moves entered"
                                    self.outcome_text.draw()
                                    _, moves = self.test_moves(0, outcome_state)
                                    moves = moves[1:]
                                    move_rts = [np.nan] * len(moves)
                                    outcome_type = 'too_few_moves'

                                self.show_move_sequence(moves, change_times[4], change_times[5], t, outcome,
                                                        shock_outcome, test)

                        # Show generated moves and warning if wrong moves entered
                        elif not valid_moves:

                            if test:
                                self.main_text.text = "Wrong moves entered"
                                self.main_text.draw()

                            else:
                                if not len(moves):
                                    self.outcome_text.text = "Wrong moves entered"
                                    self.outcome_text.draw()
                                    _, moves = self.test_moves(0, outcome_state)
                                    moves = moves[1:]
                                    move_rts = [np.nan] * len(moves)
                                    outcome_type = 'invalid_moves'

                                self.show_move_sequence(moves, change_times[4], change_times[5], t, outcome,
                                                        shock_outcome, test)

                        # Show entered moves
                        else:
                            self.show_move_sequence(moves, change_times[4], change_times[5], t,
                                                    outcome, shock_outcome, test)

                        # Monitor
                        if not monitoring_saved['Moves'] and self.monitoring:
                            if moves:
                                monitoring_saved['Moves'] = self.save_json(i + 1, len(trial_info), 'Moves', valid_moves,
                                                                           moves, outcome,
                                                                           shock_outcome, self.subject_id)
                            else:
                                monitoring_saved['Moves'] = self.save_json(i + 1, len(trial_info), 'Moves', valid_moves,
                                                                           None, outcome,
                                                                           shock_outcome, self.subject_id)

                    # Rest period
                    elif change_times[5] <= t < change_times[6]:
                        self.fixation.draw()

                        if not monitoring_saved['Rest'] and self.monitoring:
                            if moves:
                                monitoring_saved['Rest'] = self.save_json(i + 1, len(trial_info), 'Rest', valid_moves,
                                                                          moves, outcome,
                                                                          shock_outcome, self.subject_id)
                            else:
                                monitoring_saved['Rest'] = self.save_json(i + 1, len(trial_info), 'Rest', valid_moves,
                                                                          None, outcome,
                                                                          shock_outcome, self.subject_id)

                        self.send_trigger(self.config['triggers']['rest'], self.trigger_dict['Rest'])
                        self.trigger_dict['Rest'] = True

                        if self.MEG_mode and self.photodiode:
                            if change_times[4] < t < change_times[4] + 0.5:
                                self.photodiode_square.fillColor = 'white'
                                self.photodiode_square.draw()
                            else:
                                self.photodiode_square.fillColor = 'black'

                    # End trial
                    elif t >= change_times[-1]:
                        continue_trial = False

                # flip to draw everything
                self.win.flip()

                # If the trial has ended, save data to csv
                if not continue_trial:

                    # if not test and valid_moves and len(moves) == self.n_moves:
                    #     self.reward_value += float(outcome[moves[-1]])  # add reward to total
                    #     print "Current accumulated reward = {0}".format(self.reward_value)

                    # Record whether they got shocked
                    if len(moves) and shock_outcome[moves[-1]]:
                        previously_shocked = True

                    # Responses
                    self.response_data['trial_number'] = i
                    self.response_data['Subject'] = self.subject_id

                    # Reduce move entering duration
                    if test and self.move_entering_duration > self.config[self.durations]['move_entering_duration']:
                        self.move_entering_duration -= self.move_entering_duration_step

                    if trial_info['trial_type'][i] == 1:
                        for n in range(0, 3):
                            self.response_data['State_{0}'.format(n + 1)] = np.nan
                            self.response_data['RT_{0}'.format(n + 1)] = np.nan
                        self.response_data['Shock_received'] = shock_only
                    else:
                        print moves
                        for n, state in enumerate(moves):
                            print n, state
                            self.response_data['State_{0}'.format(n + 1)] = state
                            self.response_data['RT_{0}'.format(n + 1)] = move_rts[n]
                        if len(moves) == self.n_moves:
                            print shock_outcome[state]
                            self.response_data['Shock_received'] = shock_outcome[state]
                        else:
                            self.response_data['Shock_received'] = np.nan
                    self.response_data['trial_type'] = trial_type
                    for i in range(2):
                        self.response_data['State_{0}_shock'.format(i + 1)] = shock_outcome[i + 5]
                    self.response_data['outcome_type'] = outcome_type

                    if not test:
                        print self.response_data
                        csvWriter([self.response_data[category] for category in self.data_keys])  # Write data
                    else:
                        # In the test phase, figure out whether they got to the correct state
                        if len(moves) > 0 and moves[-1] == end_state:
                            n_successes += 1
                        else:
                            n_successes = 0

                        print "Number of successes = {0}".format(n_successes)

                        # In the test phase, break when subject gets enough trials correct
                        if test and n_successes == self.config['number training trials']['n_test_successes']:
                            return True

                if event.getKeys(["space", ' ']):
                    self.instruction_text.setText("Experiment paused")
                    self.instruction_text.draw()
                    self.win.flip()
                    if not monitoring_saved['Pause'] and self.monitoring:
                        monitoring_saved['Pause'] = self.save_json(i + 1, len(trial_info), 'Pause', valid_moves,
                                                                   None, None, None, self.subject_id, stopped='Space')
                    event.waitKeys(['space', ' ', '1'])
                    self.send_trigger(0, False)
                    self.trigger_dict = {'Planning': False, 'Move_entering': False, 'State_0': False, 'State_1': False,
                                         'State_2': False, 'NoShock': False,
                                         'State_3': False, 'Shock': False, 'Rest': False, 'Outcome_only_warning': False,
                                         'Outcome_only_outcome': False}
                    self.clock.reset()

                # quit if subject pressed scape
                if event.getKeys(["escape", "esc"]):
                    # if self.monitoring:
                    #     self.save_json(i + 1, len(trial_info), 'Escape', valid_moves, None, None, None, self.subject_id,
                    #                    stopped='Escape')
                    core.quit()
                    self.trigger_record_file.close()


        return False

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

    def send_shocks(self, shocks_given, n_shocks):

        """
        Sends a train of shocks via the parallel port

        Args:
            shocks_given: Number of shocks given
            n_shocks: Number of shocks in the shocks train

        Returns:
            Updated number of shocks given

        """

        if shocks_given < n_shocks:
            print "Shock {0}".format(shocks_given + 1)
            self.parallel_port.setData(255)
            self.parallel_port.setData(0)
            shocks_given += 1

        return shocks_given

    def show_move(self, outcome, shock, picture, t, shock_time, shock_delay):

        """
        Shows the image and (potentially) outcome associated with a state

        Args:
            outcome: The reward value of the state
            shock: Whether the state is associated with a shock (binary)
            picture: The image to be displayed (path to the image)
            t: Current time
            shock_time: Time at which the shock outcome should be displayed (this occurs after the reward outcome)
            shock_delay: Delay until shocks are given after showing shock icon
            shocks_given: Number of shocks given so far

        """

        # set image
        if self.display_image.image != picture:
            self.display_image.image = picture

        # set outcome text either as value or shock
        # self.outcome_image.image = outcome
        if t <= shock_time:  # Show reward outcome
            if outcome:
                self.outcome_text.text = outcome
            self.outcome_text.draw()
        elif t > shock_time + shock_delay and shock == 1:  # Show shock outcome and give shocks
            self.outcome_image.draw()
            self.shocks_given = self.send_shocks(self.shocks_given, n_shocks=self.n_shocks)
        elif t > shock_time and shock == 1:  # Show shock outcome
            self.outcome_image.draw()
            self.send_trigger(self.config['triggers']['shock_outcome'], self.trigger_dict['Shock'])
            self.trigger_dict['Shock'] = True
        elif t > shock_time and shock == 0:  # Send trigger for no shock outcome
            self.outcome_image_noshock.draw()
            self.send_trigger(self.config['triggers']['no_shock_outcome'], self.trigger_dict['NoShock'])
            self.trigger_dict['NoShock'] = True

        # draw everything
        self.display_image.draw()


    def show_move_sequence(self, moves, start_time, end_time, t, outcome, shock_outcome, test):

        """
        Shows a sequence of states with associated outcome for the final state

        Args:
            moves: Sequence of state IDs to show
            start_time: Time at which the move sequence should start
            end_time: Time at which the move sequence should end
            t: Current time, taken from the clock
            outcome: Outcomes for all states
            shock_outcome: Shock outcomes for all states
            test: Test setting

        """

        if not moves[0] == 0:
            moves = [0] + moves

        for n, state in enumerate(moves):

            if start_time + self.cumulative_move_durations[n] <= t < start_time + self.cumulative_move_durations[n + 1]:

                self.show_move(outcome[state], shock_outcome[state], self.stimuli[state], t,
                               start_time + self.cumulative_move_durations[n] + self.shock_symbol_delay, self.shock_delay)

                self.send_trigger((n + 1) * 2, self.trigger_dict['State_{0}'.format(n)])
                self.trigger_dict['State_{0}'.format(n)] = True

                if self.MEG_mode and self.photodiode:
                    if start_time < t < start_time + 0.5:
                        self.photodiode_square.fillColor = 'white'
                        self.photodiode_square.draw()
                    else:
                        self.photodiode_square.fillColor = 'black'


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
        event.waitKeys(maxWait=max_wait, keyList=['space', '1'])

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
            key = event.waitKeys(keyList=['space', 'escape', 'esc', '1'])
            if key[0] in ['escape', 'esc']:
                core.quit()
                self.trigger_record_file.close()

    def create_matrix(self):

        """
        Assigns keys to the transition matrix provided in the config file

        Returns: The matrix and a matrix of keys subjects can press from each state, and where that key takes them

        """

        matrix = np.loadtxt(self.config['directories']['matrix'])

        print matrix

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

    def test_moves(self, start, end=None, return_keys=True):

        """
        For a given start and end state, works out a series of allowable moves. If end is None, returns all possible moves
        from the current state

        Args:
            start: Starting state
            end: End state

        Returns:
            A list of allowed moves

        """

        if end is not None:
            states = nx.shortest_path(self.matrix_asgraph, start, end)

        else:
            states = [i for i in self.matrix_asgraph.neighbors(start)]

        if end is None:  # task - we don't need the first state
            moves = [self.state_selection_keys[i] for i in states]
        else:  # training, we do
            moves = [self.state_selection_keys[i] for i in states[1:]]

        return moves, states

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
                     "Valid": valid_moves, "Moves": moves, "Reward": reward, "Shock": shock, "Subject": subject}

        try:
            with open('//cher/twise/web/replay_task_output_json.txt', 'w') as f:
                json.dump(json_data, f)
            return True
        except:
            pass

    def get_max_reward(self):

        reward_data = self.trial_info[self.trial_info.trial_type == 0][[i for i in
                                                                        self.trial_info.columns if 'reward' in i]]

        return reward_data.max(axis=1).sum()

    def setup_state_selection_grid(self, random_positions=True, valid=False, test=False,
                                   initial_state=None, assign_keys=True):

        """
        Assign state images to the imagestim instances that make up the 3 x 3 state selection grid

        """

        # Shuffle imagestim/number order so images are randomly assigned to positions on the grid

        if random_positions:
            random.shuffle(self.state_selection_images)

        # Create a dictionary that looks like {state id: grid id}
        self.state_selection_dict = {}

        # Assign keys for each state
        if assign_keys:
            self.state_selection_keys, self.phase_key_state_mapping = self.assign_response_keys()

        # Get valid moves
        if initial_state is None:
            valid_moves, valid_states = self.test_moves(0)
        else:
            valid_moves, valid_states = self.test_moves(initial_state)
        self.valid_moves = valid_moves

        displayed_states = valid_states + random.sample([i for i in range(1, len(self.stimuli)) if i not in valid_states],
                                                        4 - len(valid_states))

        # Make invalid images smaller
        if valid:
            invalid_size = [i / 2. for i in self.config['image sizes']['size_selection_image']]

        for grid_index, state in enumerate(displayed_states):
            self.state_selection_images[grid_index][1].image = self.stimuli[state]
            self.state_selection_images[grid_index][2].text = self.state_selection_keys[state] # TODO ???
            if valid and state not in valid_states:
                self.state_selection_images[grid_index][1].size = invalid_size
            else:
                self.state_selection_images[grid_index][1].size = self.config['image sizes']['size_selection_image']
            self.state_selection_dict[state] = self.state_selection_images[grid_index][0]

        # for n, i in enumerate(self.stimuli[1:]):
        #     self.state_selection_images[n][1].image = i
        #     self.state_selection_images[n][2].text = self.state_selection_keys[n + 1]
        #     if valid and n + 1 not in valid_states:
        #         self.state_selection_images[n][1].size = invalid_size
        #     else:
        #         self.state_selection_images[n][1].size = self.config['image sizes']['size_selection_image']
        #     self.state_selection_dict[n + 1] = self.state_selection_images[n][0]

    def draw_state_selection_grid(self, selected=(), test=False):

        for n, i, j in self.state_selection_images:
            if n in selected:
                i.opacity = 0.3
            else:
                i.opacity = 1
            i.draw()
            j.draw()

    def validate_moves(self, moves):

        for n in range(len(moves) - 1):

            if not self.matrix[moves[n], moves[n + 1]] == 1:
                return False

        return True

    def assign_response_keys(self):

        # Assign keys to each state option - must have distinct keys for each option at each level
        shuffled_response_keys = copy.copy(self.response_keys)

        state_keys = {}  # maps states to keys, used for labelling
        phase_key_state_mapping = {}  # maps phases (levels of the tree) and keys to states

        prev_phase = -1
        prev_n = 0

        for i in range(self.matrix.shape[0]):

            next_states = np.where(self.matrix[i, :])[0]

            for n, j in enumerate(next_states):

                phase = self.response_phases[j - 1]
                if phase != prev_phase:
                    random.shuffle(shuffled_response_keys)
                    prev_n = 0

                state_keys[j] = shuffled_response_keys[n + prev_n]

                if not phase in phase_key_state_mapping:
                    phase_key_state_mapping[phase] = {}
                phase_key_state_mapping[phase][shuffled_response_keys[n + prev_n]] = j

                prev_phase = phase
            prev_n += n + 1

        return state_keys, phase_key_state_mapping


    def send_trigger(self, data, set):

        if not set:
            self.win.callOnFlip(self.parallel_port.setData, data)
        else:
            self.win.callOnFlip(self.parallel_port.setData, 0)


def trigger(port, data):

    port.setData(data)

## RUN THE EXPERIMENT

# Create experiment class
experiment = ReplayExperiment('replay_task_settings.yaml')

# Run the experiment
experiment.run()

