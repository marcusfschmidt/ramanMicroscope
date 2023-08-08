#%%
import matplotlib.pyplot as plt
import numpy as np
import OscilloscopeInterface as Oscilloscope
import time


class AcqScope(object):

    def __init__(self, channelNumber = 1, scale = 10, timebase = 200e-6, avgNumber = 16, acqMode = "average"):
        """ Function to initialize the oscilloscope and set its parameters.

        Parameters
        ----------
        channelNumber : int
            The channel number to acquire data from.
        scale : float or str
            The scale of the oscilloscope. If "auto", the scale will be set to auto.
        timebase : float
            The timebase of the oscilloscope.
        avgNumber : int
            The number of averages to take.
        acqMode : str
            The acquisition mode. Can be "average", "normal", "peak" og "hires".
        
        ----------
        """


        print("Connecting to oscilloscope...")
        try:
            scope = Oscilloscope.IV()
            idstr = scope.scope.query("*IDN?")
            print("Oscilloscope connection established, ID: " + idstr)
        except:
            print("Oscilloscope connection not established. Check connection and try again.")
            raise SystemExit("ERROR: could not connect to oscilloscope") from None

        self.scope = scope
        self.channelNumber = channelNumber
        self.timebase = timebase
        self.avgNumber = 1
        self.scale = scale

        self.channelDict = {1: scope.Channel1, 2: scope.Channel2, 3: scope.Channel3, 4: scope.Channel4}
        self.channelBooleanArray = np.zeros(4, dtype=bool)
        self.channelBooleanArray[channelNumber-1] = True

        scope.reset()
        scope.do_command("WAVeform:SOURce CHANnel%d" % channelNumber)
        scope.display(self.channelBooleanArray)
        scope.triggerEdge(self.channelDict[self.channelNumber], 30)
        scope.setWaveIntensity()

        if scale == "auto":
            scope.autoscale()
        else:
            scope.scale(self.channelNumber, self.scale)

        scope.timebase(timebase)
        scope.timebaseOffset(0)
        self.scope.do_command(":TRIG:FORC")



        if acqMode == "normal":
            scope.acquireMode(scope.NORMal)
        elif acqMode == "average":
            scope.acquireMode(scope.AVERage)
            self.avgNumber = avgNumber
            scope.do_command("ACQuire:COUNt %d" % avgNumber)
        elif acqMode == "peak":
            scope.acquireMode(scope.PEAK)
        elif acqMode == "hires":
            scope.acquireMode(scope.HRESolution)

    def acquire(self):
            self.scope.single()
            self.scope.do_command(":TRIG:FORC")


            time.sleep(self.avgNumber*self.timebase*10+1e-6)
            _,voltage = self.scope.captureWaveform([self.channelNumber], False,'',True)
            voltage = voltage.flatten()
            return voltage

    def close(self):
        self.scope.scope.close()