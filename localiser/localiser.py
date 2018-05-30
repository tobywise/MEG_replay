
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

class ParallelPort(object):

    def __init__(self, port=888, test=False):
        self.test = test
        if not self.test:
            self._parallel = ctypes.WinDLL('simpleio.dll')
        self.port = port

    def setData(self, data=0):

        if not self.test:
            self._parallel.outp(self.port, data)
        else:
            print "-- {0} -- ".format(data)


class Localiser(object):

    def __init__(self, config=None, block=0):

        with open(config) as f:
            self.config = yaml.load(f)


        # ------------------------------------#
        # Subject/task information and saving #
        # ------------------------------------#

        # Enter subject ID and other information
        dialogue = gui.Dlg()
        dialogue.addText("Subject info")
        dialogue.addField('Subject ID')
        dialogue.addText("Task info")
        dialogue.addField('Order', choices=[0, 1])
        dialogue.addField('Parallel port testing', initial=False)
        dialogue.show()

        # check that values are OK and assign them to variables
        if dialogue.OK:
            self.subject_id = dialogue.data[0]
            self.order = dialogue.data[1]
            self._parallel_port = dialogue.data[2]
        else:
            core.quit()

        monitor = monitors.Monitor('test2', width=40.92, distance=74)
        monitor.setSizePix((1024, 768))
        self.win = visual.Window(monitor=monitor, size=(1024, 768), fullscr=True, allowGUI=False, color='gray',
                                 units='deg',
                                 colorSpace='hex')

        self.parallel_port = ParallelPort(port=888, test=self._parallel_port)

        self.image = visual.ImageStim(win=self.win, size=self.config['image_sizes']['size_display_image'])
        self.fixation = visual.TextStim(win=self.win, height=0.8, color='white', text="+")

        self.main_text = visual.TextStim(win=self.win, height=0.8, color='white',
                                         alignVert='center', alignHoriz='center', wrapWidth=30)


        self.n_correct = 0
        self.possible_correct = 0
        self.block = block

        self.variants = ['standard', 'text']

        # Create the data file
        self.data_keys = ['True_answer', 'Response']
        self.save_folder = self.config['directories']['saved_data']
        self.save_prefix = self.config['filenames']['save_prefix']

    def run_localiser(self, variant=None, block=0):

        self.block = block

        if variant == 'text':
            text = True
        else:
            text = False

        fname = '{0}/{1}_Subject{2}_{3}_behaviour.csv'.format(self.save_folder, self.save_prefix, self.subject_id,
                                                              data.getDateStr())
        csvWriter = csv.writer(open(fname, 'wb'), delimiter=',').writerow
        csvWriter(self.data_keys)  # Write column headers
        self.response_data = dict(zip(self.data_keys, [None for i in self.data_keys]))

        for i in range(self.config['task_settings']['n_trials_{0}'.format(variant)]):

            self.parallel_port.setData(0)

            print "Trial {0} / {1}, block {2}".format(i+1, self.config['task_settings']['n_trials_{0}'.format(variant)], self.block)

            null = False

            # Get random image
            image_idx = random.randint(0, self.config['task_settings']['n_stimuli'] - 1)
            self.image.image = self.stimuli[image_idx]

            key = False

            small = random.randint(0, 100)
            if small < self.config['task_settings']['percentage_small']:
                self.image.opacity = self.config['image_sizes']['test_alpha']
                # self.main_text.text = 'test'
                # self.main_text.draw()
                small = True
                self.possible_correct += 1
                print "TEST TRIAL"
            elif small > 100 - self.config['task_settings']['percentage_null']:  # null trials
                null = True
            else:
                self.image.opacity = 1
                small = False

            if text and not null:
                self.main_text.text = self.image.image.split('\\')[-1].split('.')[0].lower()
                self.main_text.draw()

                self.win.callOnFlip(self.parallel_port.setData, 20)
                self.win.flip()
                self.parallel_port.setData(0)

                text_duration = self.config['durations']['text_duration'] + \
                                            (random.randint(-self.config['durations']['text_jitter'] * 1000,
                                            self.config['durations']['text_jitter'] * 1000)) / 1000.
                core.wait(text_duration)


            if not null:
                self.image.draw()
                self.win.callOnFlip(self.parallel_port.setData, image_idx + 1)
            else:
                self.fixation.draw()
                self.win.callOnFlip(self.parallel_port.setData, 99)
            fliptime = self.win.flip()
            self.parallel_port.setData(0)

            duration = self.config['durations']['image_duration'] + \
                       (random.randint(-self.config['durations']['image_jitter'] * 1000,
                                self.config['durations']['image_jitter'] * 1000)) / 1000.
            core.wait(duration)
            key = event.getKeys(timeStamped=True, keyList=['1', 'escape', 'esc'])
            if key:
                key, rt = key[0]
                if key in ['escape', 'esc']:
                    core.quit()
                rt -= fliptime
            if key and small:
                self.n_correct += 1
            elif key and not small:
                self.n_correct -= 1

            print "Number correct = {0}".format(self.n_correct)

            self.fixation.draw()
            self.win.callOnFlip(self.parallel_port.setData, 30)
            self.win.flip()
            self.parallel_port.setData(0)
            fix_duration = self.config['durations']['fixation_duration'] + \
                       (random.randint(-self.config['durations']['fixation_jitter'] * 1000,
                                self.config['durations']['fixation_jitter'] * 1000)) / 1000.
            core.wait(fix_duration)

            if key == []:
                key = 0
            self.response_data['True_answer'] = small
            self.response_data['Response'] = key
            csvWriter([self.response_data[category] for category in self.data_keys])  # Write data


    def instructions(self, text, fixation=True):

        self.main_text.text = text
        self.main_text.draw()
        self.win.flip()
        event.waitKeys(keyList=['space'])
        if fixation:
            self.fixation.draw()
            self.win.flip()
            core.wait(1)

    def run_task(self):

        self.instructions("Welcome to the task", fixation=False)
        self.instructions("We are about to begin, please keep your head still until the next break")

        n_runs = [4, 4]

        stimuli_location = self.config['directories']['stimuli_path']

        self.stimuli = [os.path.join(stimuli_location, i) for i in os.listdir(stimuli_location)
                        if ('.png' in i or '.jpg' in i or '.jpeg' in i) and 'shock' not in i][
                       :self.config['task_settings']['n_stimuli']]

        self.order = int(self.order)               

        for i in range(n_runs[self.order]):

            print "Block {0} of {1}, {2} variant".format(i + 1, n_runs[self.order], self.variants[self.order])

            self.run_localiser(self.variants[self.order], i+1)
            self.instructions("Take a break", fixation=False)
            self.instructions("We are about to begin, please keep your head still until the next break")


        self.stimuli = [os.path.join(stimuli_location, i) for i in os.listdir(stimuli_location)
                        if ('.png' in i or '.jpg' in i or '.jpeg' in i) and 'shock' not in i][
                       self.config['task_settings']['n_stimuli']:self.config['task_settings']['n_stimuli'] * 2]

        self.instructions("The images you see in the next block will be different from those in the previous block")

        for i in range(n_runs[1 - self.order]):

            print "Block {0} of {1}, {2} variant".format(i + 1, n_runs[1 - self.order], self.variants[1 - self.order])

            self.run_localiser(self.variants[1 - self.order], i+1)

            if i < n_runs[1 - self.order] - 1:
                self.instructions("Take a break", fixation=False)
                self.instructions("We are about to begin, please keep your head still until the next break")

        self.instructions("End of task, thank you for participating!\n\n"
                          "You collected {0} points out of a possible {1}".format(self.n_correct,
                                                                                  self.possible_correct, fixation=False))

localiser = Localiser('localiser_config.yaml')
localiser.run_task()