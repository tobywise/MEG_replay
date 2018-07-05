import ctypes
from psychopy import core

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
            print "-"

parallel_port = ParallelPort(port=888)

for i in range(0, 255):
    print "NUMBER {0}".format(i)
    parallel_port.setData(i)
    core.wait(2)
    parallel_port.setData(0)