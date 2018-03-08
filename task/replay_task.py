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

        with open(config) as f:
            self.config = yaml.load(f)


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


        # self.main_text.fontFiles = ["fonts/OpenSans-Regular.ttf"]  # Arial is horrible
        # self.main_text.font = 'Open Sans'
        #
        # self.inst_text = visual.TextStim(win=self.win, height=0.7, color='white', pos=(0, -7), wrapWidth=30)
        # self.inst_text.fontFiles = ["C:\Users\Toby\Downloads/fonts\Open_Sans\OpenSans-Regular.ttf"]
        # self.inst_text.font = 'Open Sans'

        # Stimulus location and size information - allows this to be easily set and reused later
        stimuli_location = '/Stimuli'
        stimuli = [os.path.join(stimuli_location, i) for i in os.listdir(stimuli_location)
                   if '.png' in i or '.jpg' in i or '.jpeg' in i]
        random.shuffle(stimuli)

        # positions are given as (x units, y units) - here we're putting three of these coordinate pairs in a list
        self.locs = [(-9, 0), (0, 0), (9, 0)]  # store preset positions for use later

        # sizes are given as (width, height)
        self.image_size = (6, 6)

        self.outcome_image = visual.ImageStim(win=self.win, size=(2, 3), image='shock.png', pos=(0, 7))

        # use imagestim to set up image stimuli - you'll need to fill in some details here
        self.display_image = visual.ImageStim(win=self.win, size=(12, 8))

        # TRANSITION MATRIX

        matrix, matrix_keys = self.create_matrix()



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


    def show_move(self, outcome, picture, move):


        # set image
        self.display_image.image = picture

        # set outcome text either as value or shock
        #self.outcome_image.image = outcome
        if outcome == -99:
            self.outcome_image.draw()
        else:
            self.outcome_text.text = outcome
            self.outcome_text.draw()

        # set move text
        self.move_text.text = move

        # draw everything
        self.move_text.draw()
        self.display_image.draw()


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
        self.pre_move_duration = 1
        self.move_entering_duration = 3
        self.move_duration = 2
        self.n_moves = 5
        self.move_period_duration = self.move_duration * self.n_moves

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

            change_times = [0, self.start_duration, self.start_duration + self.pre_move_duration, self.start_duration +
                            self.pre_move_duration + self.move_entering_duration, self.start_duration +
                            self.pre_move_duration + self.move_entering_duration + self.move_period_duration]  # list of times at which the screen should change

            # set text
            welcome_text = 'Welcome to MEG thing!'
            key_text = 'Enter key movements'
            left_text = 'left'
            right_text = 'right'
            up_text = 'up'
            down_text = 'down'



            # Default values for responses in case the subject makes no response
            response = None
            rt = None

            key = None

            trial_moves = []

            while continue_trial:  # run the trial

                t = self.clock.getTime()  # get the time

                # SCREEN 1
                if change_times[0] <= t < change_times[1]:

                    self.main_text.text = welcome_text
                    self.main_text.draw()


                #SCREEN 2
                elif change_times[1] <= t < change_times[2]:
                    self.main_text.text = key_text
                    self.main_text.draw()

                #SCREEN 3
                elif change_times[2] <= t < change_times[3]:

                    if not len(trial_moves):
                        self.main_text.text = ''

                    raw_keys = event.getKeys(keyList=['left', 'right', 'up', 'down'], timeStamped=True)

                    if len(raw_keys):
                        key, rt = raw_keys[0]

                        print key, rt

                    if key is not None and len(trial_moves) < self.n_moves:
                        if key == 'left':
                            self.main_text.text = left_text
                        elif key == 'right':
                            self.main_text.text = right_text
                        elif key == 'up':
                            self.main_text.text = up_text
                        elif key == 'down':
                            self.main_text.text = down_text

                        trial_moves.append(key)
                        key = None

                    self.main_text.draw()

                elif change_times[3] <= t < change_times[4]:
                    for n, move in enumerate(trial_moves):
                        if change_times[3] + n * self.move_duration <= t < change_times[3] + (n + 1) * self.move_duration:
                            self.show_move(-99, 'cat.jpeg', move)

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

