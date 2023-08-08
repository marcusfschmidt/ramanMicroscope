#%%

import pyvisa as visa
import sys
import os
import time
import numpy as np

####### 
# PI stage must be connected using the IEEE interface (GPIB). This setting can change if adjusting a knob on the stage. If it is connected using IEEE, the upper right corner of the screen should show IEEE as opposed to RS232.
# In order to change it from RS232 to IEEE, turn the "Next" knob twice to "Communication".
# Here, press the button to the left "Select" until the screen shows IEEE. Then press the "esc" button.


# Depending on which GPIB cable is used, it may be necessary to change the commands. The controllers from Prologix work with their own commands which must be used prior to sending standard device commands.

# Note that the PI stage seems to be very unstable. Sometimes the GPIB connection is lost and the stage must be restarted manually (back of the stage). This typically also means that the controller itself loses connection, and the USB must be unplugged and replugged.

######
class PI(object):

    def __init__(self, prologixBool = True, Address = "ASRL3::INSTR", prologGPIBAddress = '7', X = 'A', Y = 'B', correctionAngle = 0, velocity = 100):

        """
        prologixBool: True if using a prologix controller, False if not
        GPIBAddress: GPIB address or COM port of the stage. If using a prologix controller, the GPIB address must be specified using the prologGPIBAddress variable.
        prologGPIBAddress: GPIB address of the stage if using a prologix controller
        X: X-axis channel (A or B)
        Y: Y-axis channel (A or B)
        correctionAngle: angle of the stage relative to the microscopy platform (if 0, assuming stage is perpendicular to the beam)
        velocity: velocity of the stage in microns/s
        """


        rm = visa.ResourceManager()
        self.visaInstrList = rm.list_resources()

        if len(self.visaInstrList) == 0:
            print("ERROR: no instrument found!")
            print("Exited because of error.")
            sys.exit(1)

        print("Connecting to GPIB controller and stage...")
        try:
            self.PI = rm.open_resource(Address)
        except:
            raise SystemExit("Could not connect to GPIB controller.") from None

        self.PI.timeout = 5000
        self.PI.write_termination = '\n'
        self.PI.read_termination = '\n'

        # If using the prologix controller, check if connection has been made
        if prologixBool:
            try:
                self.PI.write("++addr " + prologGPIBAddress)
                self.PI.write("++mode 1")
                self.PI.write("++ver")
                print("Controller connection established, version: " + self.PI.read())

                try:
                    idn_string = self.PI.query("*IDN?")
                    print("GPIB connection established, ID: " + idn_string)
                            
                except visa.VisaIOError:
                    print("Exited because of error. Try restarting the stage and the Python kernel.")
                    raise SystemExit("ERROR: could not connect to stage") from None


            except visa.VisaIOError:
                print("Exited because of error. Try unplugging and replugging the USB cable and restarting the Python kernel.")
                raise SystemExit("ERROR: could not connect to controller") from None
        
        #otherwise if using the agilent cable, check if connection has been made
        else:
            try:
                idn_string = self.PI.query("*IDN?")
                print("GPIB connection established, ID: " + idn_string)
            except visa.VisaIOError:
                print("Exited because of error. Try restarting the stage and the Python kernel.")
                raise SystemExit("ERROR: could not connect to stage") from None


        self.X = X
        self.Y = Y
        self.correctionAngle = correctionAngle
        self.velocity = velocity


        self.PI.clear()
        #Swithes servo-controlled motion and goes to (0,0)
        self.PI.write("ONL 1")
        self.PI.write("VEL A {} B {}".format(self.velocity,self.velocity))
        self.PI.write("MOV A 0 B 0")

        #velocity control and drift compensation
        self.PI.write("VCO A 1 B 1") 
        self.PI.write("DCO A 1 B 1")

    def getPos(self, channel):
        return float(self.PI.query("POS? %s" % channel))

    def moveToPos(self, channel, position):
        currentPos = self.getPos(channel)
        deltaPos = abs(position - currentPos)
        approximateTime = deltaPos/self.velocity

        self.PI.write("MOV %s %s" % (channel, str(position)))
        time.sleep(approximateTime)

    def moveToCoordinates(self, coordinates):
        x = coordinates[0]
        y = coordinates[1]
        theta = self.correctionAngle

        #Move x-coordinate
        self.moveToPos(self.X, np.cos(theta)*x + np.sin(theta)*y)

        #Move y-coordinate
        self.moveToPos(self.Y, -np.sin(theta)*x + np.cos(theta)*y)

    def close(self):
        self.PI.close()
