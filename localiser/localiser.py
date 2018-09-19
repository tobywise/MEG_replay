
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

    def __init__(self, port=888):

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

    def setData(self, data=0):

        if not self.test:
            self._parallel.outp(self.port, data)
            print "-- Sending value {0} to parallel port -- ".format(data)
        else:
            print "-- Sending value {0} to parallel port -- ".format(data)


class Localiser(object):

    def __init__(self, config=None):

        """

        Args:
            config: Path to config file
        """

        with open(config) as f:
            self.config = yaml.load(f)

        # Enter subject ID and other information
        dialogue = gui.Dlg()
        dialogue.addText("Subject info")
        dialogue.addField('Subject ID')
        dialogue.addField('Number of blocks', 2)
        dialogue.show()

        # check that values are OK and assign them to variables
        if dialogue.OK:
            self.subject_id = dialogue.data[0]
            self.n_blocks = dialogue.data[1]
        else:
            core.quit()

        # Recode blank subject ID to zero - useful for testing
        if self.subject_id == '':
            self.subject_id = '0'

        # Monitor nad window
        monitor = monitors.Monitor('test2', width=40.92, distance=74)
        monitor.setSizePix((1024, 768))
        self.win = visual.Window(monitor=monitor, size=(1024, 768), fullscr=True, allowGUI=False, color='#616161',
                                 units='deg',
                                 colorSpace='hex')

        # Set up parallel port
        self.parallel_port = ParallelPort(port=888)
        self.parallel_port.setData(0)

        # Variables used to keep track of responses
        self.n_correct = 0
        self.possible_correct = 0

        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

        # Create the data file
        self.data_keys = ['True_answer', 'Response', 'Image_idx', 'RT']
        self.save_folder = self.config['directories']['saved_data']
        self.save_prefix = self.config['filenames']['save_prefix']

        # LOCALISER STIMULI

        # Image stimulus for displaying stimuli
        self.image = visual.ImageStim(win=self.win, size=self.config['image_sizes']['size_display_image'])

        # Fixation cross
        self.fixation = visual.TextStim(win=self.win, height=0.8, color='white', text="+")

        # Text used for instructions etc
        self.main_text = visual.TextStim(win=self.win, height=0.8, color='white',
                                         alignVert='center', alignHoriz='center', wrapWidth=30)


        # PERFORMANCE DISPLAY STIMULI

        # Title text
        self.title = visual.TextStim(win=self.win, text="Your performance so far:", pos=(0, 9))

        # Axis and lines
        self.y_axis = visual.Rect(win=self.win, width=0.1, height=8, pos=(-10, 0), fillColor='white', lineColor=None)
        self.marker_lines = [visual.Rect(win=self.win, width=0.1, height=8, pos=(-10 + 5 * (i + 1), 0), opacity=0.25,
                                    fillColor='white', lineColor=None) for i in range(4)]
        self.marker_labels = [visual.TextStim(win=self.win, pos=(-10 + 5 * (i), 5), height=0.6,
                                         text=str(i * 25) + '%') for i in range(5)]

        # Bars and labels
        self.subject_bar = visual.Rect(win=self.win, width=0, height=2, fillColor='#FF7F0E', lineColor=None,
                                  pos=(-10., 2))

        self.average_bar = visual.Rect(win=self.win, width=0, height=2, fillColor='#1F77B4', lineColor=None,
                                  pos=(-10., -2))

        self.subject_bar_label = visual.TextStim(win=self.win, pos=(-8.5, 2), text='You', height=0.8)
        self.average_bar_label = visual.TextStim(win=self.win, pos=(-7.5, -2), text='Average', height=0.8)


    def run_localiser(self):

        # Path to save data to
        fname = '{0}/{1}_Subject{2}_{3}_behaviour.csv'.format(self.save_folder, self.save_prefix, self.subject_id,
                                                              data.getDateStr())

        # Create file and write header
        csvWriter = csv.writer(open(fname, 'wb'), delimiter=',').writerow
        csvWriter(self.data_keys)  # Write column headers
        self.response_data = dict(zip(self.data_keys, [None for i in self.data_keys]))

        faded = False

        rt = -999

        # Loop through trials
        for i in range(self.config['task_settings']['n_trials']):

            self.parallel_port.setData(0)

            print "\nTrial {0} / {1}".format(i+1, self.config['task_settings']['n_trials'])

            # Get random image
            image_idx = random.randint(0, self.config['task_settings']['n_stimuli'] - 1)
            self.image.image = self.stimuli[image_idx]

            # Determine whether this will be a null trial
            null_rng = random.randint(0, 100)
            if null_rng > 100 - self.config['task_settings']['percentage_null']:  # null trials
                null = True
                # self.fixation.draw()
                self.win.callOnFlip(self.parallel_port.setData, 99)

            # If not, show an image
            else:
                # Determine whether this will be a faded image
                faded_rng = random.randint(0, 100)
                if faded_rng < self.config['task_settings']['percentage_faded']:
                    self.image.opacity = self.config['image_sizes']['test_alpha']
                    faded = True
                    self.possible_correct += 1
                    print "FADED IMAGE"
                else:
                    self.image.opacity = 1
                    faded = False

                self.image.draw()
                self.win.callOnFlip(self.parallel_port.setData, (image_idx + 1) * 2)  # Trigger value = (idx + 1) * 2

            fliptime = self.win.flip()
            self.parallel_port.setData(0)

            # ISI
            duration = self.config['durations']['image_duration'] + \
                       (random.randint(-self.config['durations']['image_jitter'] * 1000,
                                self.config['durations']['image_jitter'] * 1000)) / 1000.
            core.wait(duration)

            # Check for key presses
            key = event.getKeys(timeStamped=True, keyList=['1', 'escape', 'esc'])
            if key:
                key, rt = key[0]
                if key in ['escape', 'esc']:
                    core.quit()
                rt -= fliptime
            if key and faded:
                print "Key pressed, correct answer"
                self.n_correct += 1
                self.tp += 1
            elif key and not faded:
                print "Key pressed, incorrect answer"
                self.n_correct -= 1
                self.fp += 1
            elif not key and faded:
                self.fn += 1
            elif not key and not faded:
                self.tn += 1

            print "Number correct = {0}".format(self.n_correct)
            print "Accuracy = {0}".format((self.tp + float(self.tn)) / (self.tp + self.fp + self.fn + self.tn))

            # Fixation
            self.fixation.draw()
            # self.win.callOnFlip(self.parallel_port.setData, 80)
            self.win.flip()
            self.parallel_port.setData(0)
            fix_duration = self.config['durations']['fixation_duration'] + \
                       (random.randint(-self.config['durations']['fixation_jitter'] * 1000,
                                self.config['durations']['fixation_jitter'] * 1000)) / 1000.
            core.wait(fix_duration)

            # If no response was made, record it as a zero
            if key == []:
                key = 0

            # Save data
            self.response_data['True_answer'] = faded
            self.response_data['Response'] = key
            self.response_data['Image_idx'] = image_idx
            self.response_data['RT'] = rt
            csvWriter([self.response_data[category] for category in self.data_keys])  # Write data


    def instructions(self, text, fixation=True):

        """
        Shows an instruction screen

        Args:
            text: Text to show
            fixation: If true, shows a fixation cross for one second after a key is pressed

        """

        self.main_text.text = text
        self.main_text.draw()
        self.win.flip()
        event.waitKeys(keyList=['space'])

        if fixation:
            self.fixation.draw()
            self.win.flip()
            core.wait(1)

    def run_resting(self):

        for i in range(10):

            self.fixation.draw()

            self.parallel_port.setData(30)
            self.parallel_port.setData(0)

            core.wait(8)

    def run_task(self):

        # Welcome screen
        self.instructions("Welcome to the task", fixation=False)
        self.instructions("We are about to begin, please keep your head still until the next break")


        # Resting
        # self.run_resting()
        self.instructions("We are now starting the real task, get ready to spot the faded images", fixation=False)

        # Get location of stimuli
        stimuli_location = self.config['directories']['stimuli_path']

        # Assign stimuli
        self.stimuli = [os.path.join(stimuli_location, i) for i in os.listdir(stimuli_location)
                        if ('.png' in i or '.jpg' in i or '.jpeg' in i) and 'shock' not in i][
                       :self.config['task_settings']['n_stimuli']]


        # Set random seed based on subject ID
        random.seed(int(re.search('\d+', self.subject_id).group()))

        # Shuffle stimuli
        random.shuffle(self.stimuli)

        # Save stimuli order
        stim_fname = '{0}/{1}_Subject{2}_{3}_localiser_stimuli.txt'.format(self.save_folder, self.save_prefix, self.subject_id,
                                                                    data.getDateStr())

        with open(stim_fname, 'wb') as f:
            f.write(str(self.stimuli))

        for i in range(self.n_blocks):

            print "Block {0} of {1}".format(i + 1, self.n_blocks)

            # Run the localiser block
            self.run_localiser()

            # Show current performance relative to average performance
            # current_performance = self.get_performance()
            # if current_performance < i * 4:
            #     current_performance = i * 4 + 5
            # average_performance = np.max([current_performance + ((self.n_blocks / 2) - (self.n_blocks - i)), (i + 1) * 5])

            # print "CURRENT PERFORMANCE"
            # print current_performance
            # print "AVERAGE PERFORMANCE"
            # print average_performance

            # current_performance, average_performance = raw_input("Enter performance").split(',')
            # current_performance = int(re.search('\d+', current_performance).group())
            # average_performance = int(re.search('\d+', average_performance).group())

            # if current_performance > 100 or average_performance > 100:
            #     current_performance, average_performance = raw_input("Enter performance").split(',')
            #     current_performance = int(re.search('\d+', current_performance).group())
            #     average_performance = int(re.search('\d+', average_performance).group())

            # self.show_performance(current_performance, average_performance)

            # self.instructions("Take a break", fixation=False)
            self.instructions("We are about to begin, please keep your head still until the next break")

        self.run_resting()

        # End screen
        self.instructions("End of task, thank you for participating!\n\n"
                          "You collected {0} points out of a possible {1}".format(self.n_correct,
                                                                                  self.possible_correct, fixation=False))

    def get_performance(self):

        """
        Calculates performance percentage
        """

        return (np.max([self.n_correct, 1]) / float(self.possible_correct)) * 100


    def show_performance(self, subject, average):

        """
        Shows subject's performance as a percentage alongside the average performance

        Args:
            subject: Subject's percentage correct
            average: Average percentage correct

        """

        # Calculate bar widths
        self.subject_bar_width = 20 * (subject / 100.)
        self.average_bar_width = 20 * (average / 100.)

        # Grow bars
        while self.subject_bar.width < self.subject_bar_width or self.average_bar.width < self.average_bar_width:

            [i.draw() for i in self.marker_lines]
            [i.draw() for i in self.marker_labels]

            if self.average_bar.width < self.average_bar_width:
                self.average_bar.width += 0.15
                self.average_bar.pos = (-10 + self.average_bar.width / 2., -2)

            self.average_bar.draw()

            if self.subject_bar.width < self.subject_bar_width and not self.average_bar.width < self.average_bar_width:
                self.subject_bar.width += 0.05
                self.subject_bar.pos = (-10 + self.subject_bar.width / 2., 2)

            self.subject_bar.draw()

            self.subject_bar_label.draw()
            self.average_bar_label.draw()

            self.y_axis.draw()

            self.title.draw()
            self.win.flip()

        # Move on if space is pressed
        event.waitKeys(keyList=['space'])
        self.subject_bar.width = 0
        self.average_bar.width = 0

        core.wait(1)


localiser = Localiser('localiser_config.yaml')
localiser.run_task()