import ctypes
from psychopy import core, visual, monitors, event
import random

test = False

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
            pass

port = ParallelPort(888, test)

# psychopy visual stuff
monitor = monitors.Monitor('test2', width=40.92, distance=74)
monitor.setSizePix((1024, 768))

# TODO MEG SCREEN 0

win = visual.Window(monitor=monitor, size=(1024, 768), fullscr=True, allowGUI=False, color='#606060', units='deg',
                    colorSpace='hex', screen=0)
vas = visual.RatingScale(win, low=1, high=10, scale='How painful was the shock?', acceptText='Press 4 to continue',
                         acceptKeys=['4'], showAccept=False, markerStart=5)
main_text = visual.TextStim(win=win, height=0.8, color='white', alignVert='center', alignHoriz='center', wrapWidth=30)
main_text.fontFiles = ["fonts/OpenSans-Regular.ttf"]  # Arial is horrible
main_text.font = 'Open Sans'

clock = core.Clock()

def give_shock(n_shocks=5, gap=0.0166):

    for i in range(n_shocks):
        print i
        port.setData(255)
        port.setData(0)
        core.wait(gap)

continue_calibration = True
shocked = False
start = True

while continue_calibration:

    if start:
        main_text.text = 'Get ready'
        main_text.draw()
        win.flip()
        event.waitKeys(keyList=['1', ' '])
        main_text.text = 'How painful was the shock?'
        start = False

    continue_shock = True
    reset = False

    while continue_shock:
        if not shocked:
            win.flip()
            delay = 0.5 + (random.randint(0, 100) / 100. * 1)
            print delay
            core.wait(delay)
            give_shock()
            core.wait(1)
            shocked = True
        while vas.noResponse:

            key = event.getKeys(['esc', 'escape', 'q', '1', '2'])
            if len(key):
                if key[0] == '1'and vas.getRating() >= vas.low + 1:
                    vas.setMarkerPos(vas.getRating() - 2)
                elif key[0] =='2' and vas.getRating() <= vas.high - 1:
                    vas.setMarkerPos(vas.getRating())

                if key[0] in ['esc', 'escape']:
                    core.quit()
            vas.draw()
            win.flip()

        if vas.getRating() == 10 and not reset:
            main_text.setText("Shock intensity rated 10/10")
            main_text.draw()
            win.flip()
            if len(event.getKeys()):
                reset = True
                continue_shock = False
                shocked = False
                vas.reset()
                event.clearEvents()
        else:
            continue_shock = False
            shocked = False
            vas.reset()

        if len(event.getKeys(['esc', 'escape', 'q'])):
            core.quit()

