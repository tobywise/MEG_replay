# MEG replay task
## Dependencies
Requires PsychoPy, which can be installed by following the instructions here http://psychopy.org/installation.html
We've had trouble installing the latest versions, so we're using 1.85.3. This can be installed with `pip install psychopy==1.85.3`

Additionally, the task requires NetworkX, which can be installed with `pip install networkx`

The git repository contains everything else necessary to run the task.

## Running the task
The task can be run with the script `replay_task.py`, which must be run from the same directory as the script.

### Options
When run, the task gives a few options:
* Subject ID - the ID of the subject
* Shock level - the current given by the shock box
* Mode - the mode to run the task in, if *Experiment* is chosen, the task runs as designed for the subject. If *Testing* is chosen, it skips instruction screens and provides example sequences of movements.
* Task phase checkboxes
	* Instructions - instructions at the start that explain the basics of the task
	* Training - training phase, where subjects freely move from stimulus to stimulus
	* Test - test phase, where subjects are tested on their knowledge of transitions. In this phase they are given a start state and a target terminal state and asked to enter the moves necessary to get them to the terminal state.
	* Task - the task phase, where subjects are able to move through the tree to collect rewards and avoid shocks
 * Parallel port testing - if checked, turns off the shocks (otherwise the task crashes on computers without a parallel port)


