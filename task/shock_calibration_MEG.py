import ctypes
from psychopy import core, visual, monitors, event

class ParallelPort(object):

    def __init__(self, port=888):
        self._parallel = ctypes.WinDLL('simpleio.dll')
        self.port = port

    def setData(self, data=0):
        self._parallel.outp(self.port, data)

port = ParallelPort(888)

# psychopy visual stuff
monitor = monitors.Monitor('test2', width=40.92, distance=74)
monitor.setSizePix((1024, 768))

# TODO MEG SCREEN 0

win = visual.Window(monitor=monitor, size=(1024, 768), fullscr=True, allowGUI=False, color='#606060', units='deg',
                    colorSpace='hex', screen=0)
vas = visual.RatingScale(win, low=1, high=10, scale='How painful was the shock?', acceptText='Press space to continue', acceptKeys=['space'], showAccept=False)
main_text = visual.TextStim(win=win, height=0.8, color='white', alignVert='center', alignHoriz='center', wrapWidth=30)
main_text.fontFiles = ["fonts/OpenSans-Regular.ttf"]  # Arial is horrible
main_text.font = 'Open Sans'

clock = core.Clock()

def give_shock(n_shocks=5, gap=0.016):

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
        event.waitKeys(keyList=['1', ' '])
        start = False

    continue_shock = True
    reset=False

    while continue_shock:
        if not shocked:
            win.flip()
            core.wait(1)
            give_shock()
            core.wait(1)
            shocked = True

        main_text.setText("Rating")
        main_text.draw()
        win.flip()
        event.waitKeys(keyList=['1', ' '])
        continue_shock = False
        shocked = False

        if len(event.getKeys(['esc', 'escape', 'q'])):
            core.quit()
        
