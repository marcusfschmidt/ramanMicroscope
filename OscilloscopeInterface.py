#%%
import pyvisa
import matplotlib.pyplot as plt
import numpy as np
import string
import struct
import sys
import time

rm =  pyvisa.ResourceManager()

debug = 0

class IV(object):
    
    NORMal = "NORMal"
    PEAK = "PEAK"
    AVERage = "AVERage"
    HRESolution = "HRESolution"
    Channel1 = 1
    Channel2 = 2
    Channel3 = 3
    Channel4 = 4
    AllChannels = [Channel1, Channel2, Channel3, Channel4]
    
    def __init__(self):
        self.visaInstrList = rm.list_resources(query='USB?*')
        if len(self.visaInstrList) == 0:
            print("ERROR: no instrument found!")
            print("Exited because of error.")
            sys.exit(1)
        myScope = self.visaInstrList[0]
        self.scope = rm.open_resource(myScope)
        self.scope.timeout = 15000
        self.scope.write_termination = '\n'
        self.scope.read_termination = '\n'
        self.scope.clear()

        # Get and display the device's *IDN? string.
        idn_string = self.scope.query("*IDN?")
        # print("Identification string:\n%s" % idn_string)
        
    def autoscale(self):
        """
            trigger autoscale.
        """
        self.do_command(":AUToscale")
    
    def display(self, channelBooleanArray, func1 = False, func2 = False):
        """
            enable or disable channels
        """
        channel1, channel2, channel3, channel4 = channelBooleanArray
        self.scope.write(":CHANnel1:DISPlay %s"%self.bool2ONOFF(channel1))
        self.scope.write(":CHANnel2:DISPlay %s"%self.bool2ONOFF(channel2))
        self.scope.write(":CHANnel3:DISPlay %s"%self.bool2ONOFF(channel3))
        self.scope.write(":CHANnel4:DISPlay %s"%self.bool2ONOFF(channel4))
        self.scope.write(":FUNCtion1:DISPlay %s"%self.bool2ONOFF(func1))
        self.scope.write(":FUNCtion2:DISPlay %s"%self.bool2ONOFF(func2))


            
    # def display(self, channel1 = False, channel2 = False, channel3 = False, channel4 = False, func1 = False, func2 = False):
    #     """
    #         enable or disable channels
    #     """        
    #     self.scope.write(":CHANnel1:DISPlay %s"%self.bool2ONOFF(channel1))
    #     self.scope.write(":CHANnel2:DISPlay %s"%self.bool2ONOFF(channel2))
    #     self.scope.write(":CHANnel3:DISPlay %s"%self.bool2ONOFF(channel3))
    #     self.scope.write(":CHANnel4:DISPlay %s"%self.bool2ONOFF(channel4))
    #     self.scope.write(":FUNCtion1:DISPlay %s"%self.bool2ONOFF(func1))
    #     self.scope.write(":FUNCtion2:DISPlay %s"%self.bool2ONOFF(func2))


        
    def bool2ONOFF(self, value):
        if (value):
            return "ON"
        else:
            return "OFF"
        
    # =========================================================
    # Send a command and check for errors:
    # =========================================================
    
    def do_command(self, command, hide_params=False):
        if hide_params:
            (header, data) = string.split(command, " ", 1)
            if debug:
                print("\nCmd = '%s'" % header)
        else:
            if debug:
                print("\nCmd = '%s'" % command)
        self.scope.write("%s" % command)
        if hide_params:
            self.check_instrument_errors(header)
        else:
            self.check_instrument_errors(command)
            
            
    # =========================================================
    # Check for instrument errors:
    # =========================================================
    def check_instrument_errors(self, command):
        while True:
            error_string = self.scope.query(":SYSTem:ERRor?")
            if error_string:   # If there is an error string value.
                if error_string.find("+0,", 0, 3) == -1:   # Not "No error".
                    print("ERROR: %s, command: '%s'" % (error_string, command))
                    print("Exited because of error.")
                    sys.exit(1)
                else:   # "No error"
                    break
            else:   # :SYSTem:ERRor? should always return string.
                print("ERROR: :SYSTem:ERRor? returned nothing, command: '%s'" % command)
                print("Exited because of error.")
                sys.exit(1)
                
    # =========================================================
    # Send a query, check for errors, return string:
    # =========================================================
    def do_query_string(self, query):
        if debug:
            print("Qys = '%s'" % query)
        result = self.scope.query("%s" % query)
        self.check_instrument_errors(query)
        return result
    
    
    # =========================================================
    # Send a query, check for errors, return values:
    # =========================================================
    def do_query_binary(self, query):
        if debug:
            print("Qyb = '%s'" % query)
        results = self.scope.query_binary_values("%s" % query, datatype='c', is_big_endian=True)
        self.check_instrument_errors(query)
        return results
    
    # =========================================================
    # Send a query, check for errors, return values:
    # =========================================================
    def do_query_double(self, query):
        if debug:
            print("Qyd = '%s'" % query)
        results = self.scope.query("%s" % query)
        self.check_instrument_errors(query)
        return float(results)


    # =========================================================
    # Returns data from definite-length block.
    # =========================================================
    def get_definite_length_block_data(self, sBlock):    
        return b''.join(sBlock)

    def reset(self):
        # Clear status and load the default setup.
        self.do_command("*CLS")
        self.do_command("*RST")

    def takeScreenshot(self):
        # Download the screen image.
        # --------------------------------------------------------
        self.do_command(":HARDcopy:INKSaver OFF")
        sDisplay = self.do_query_binary(":DISPlay:DATA? PNG, COLor")
        sDisplay = self.get_definite_length_block_data(sDisplay)
        # Save display data values to file.

        f = open("screen_image.png", "wb")
        f.write(sDisplay)
        f.close()
        print("Screen image written to screen_image.png.")

    def saveSetup(self, filename):
        # Save oscilloscope setup.
        sSetup = self.do_query_binary("SYSTem:SETup?")
        sSetup = self.get_definite_length_block_data(sSetup)
        f = open(filename, "wb")
        f.write(sSetup)
        f.close()
        print("Setup bytes saved: %d" % len(sSetup))

    def restoreSetup(self, filename):
        # Or, set up oscilloscope by loading a previously saved setup.
        f = open(filename, "rb")
        sSetup = f.read()
        f.close()
        self.do_command(":SYSTem:SETup #8%08d%s" % (len(sSetup), sSetup), hide_params=True)
        print("Setup bytes restored: %d" % len(sSetup))

    def triggerEdge(self, channelNumber, level):

        if (not channelNumber in self.AllChannels):
            raise ValueError('Channel not available')

        # Set trigger mode.
        self.do_command(":TRIGger:MODE EDGE")
        qresult = self.do_query_string(":TRIGger:MODE?")
        if debug:
            print("Trigger mode: %s" % qresult)
        # Set EDGE trigger parameters.
        self.do_command(":TRIGger:EDGE:SOURCe CHANnel%d"%channelNumber)
        qresult = self.do_query_string(":TRIGger:EDGE:SOURce?")
        if debug:
            print("Trigger edge source: %s" % qresult)
        self.do_command(":TRIGger:EDGE:LEVel %2.f"%level)
        qresult = self.do_query_string(":TRIGger:EDGE:LEVel?")
        if debug:
            print("Trigger edge level: %s" % qresult)
        self.do_command(":TRIGger:EDGE:SLOPe POSitive")
        qresult = self.do_query_string(":TRIGger:EDGE:SLOPe?")
        if debug:
            print("Trigger edge slope: %s" % qresult)

    def do_captureWaveformData(self, channelNumber):
        # Set the waveform source.
        self.do_command(":WAVeform:SOURce CHANnel%d" % channelNumber)
        qresult = self.do_query_string(":WAVeform:SOURce?")
        if debug:
            print("Waveform source: %s" % qresult)
         
        # Choose the format of the data returned:
        self.do_command(":WAVeform:FORMat BYTE")
        if debug:
            print("Waveform format: %s" % self.do_query_string(":WAVeform:FORMat?"))

        # Get the waveform data.
        sData = self.do_query_binary(":WAVeform:DATA?")
        sData = self.get_definite_length_block_data(sData)
        # Unpack unsigned byte data.
        values = struct.unpack("%dB" % len(sData), sData)
         # print("Number of data values: %d" % len(values))
        return values

    def showWaveformPreamble(self, channelNumber):
        # Display the waveform settings from preamble:
        wav_form_dict = {
            0 : "BYTE",
            1 : "WORD",
            4 : "ASCii",
        }
        acq_type_dict = {
            0 : "NORMal",
            1 : "PEAK",
            2 : "AVERage",
            3 : "HRESolution",
        }
        preamble_string = self.do_query_string(":WAVeform:PREamble?")
        (
            wav_form, acq_type, wfmpts, avgcnt, x_increment, x_origin,
            x_reference, y_increment, y_origin, y_reference
        ) = preamble_string.split(",")
        if debug:
            print("Waveform format: %s" % wav_form_dict[int(wav_form)])
            print("Acquire type: %s" % acq_type_dict[int(acq_type)])
            print("Waveform points desired: %s" % wfmpts)
            print("Waveform average count: %s" % avgcnt)
            print("Waveform X increment: %s" % x_increment)
            print("Waveform X origin: %s" % x_origin)
            print("Waveform X reference: %s" % x_reference)
            print("Waveform Y increment: %s" % y_increment)
            print("Waveform Y origin: %s" % y_origin)
            print("Waveform Y reference: %s" % y_reference)

    def single(self):
        self.do_command(":SINGle")

    def run(self):
        self.do_command(":RUN")

    def stop(self):
        self.do_command(":STOP")

    def timebaseOffset(self, offset = 0.0):
        self.do_command(":TIMebase:POSition %f" % offset)
 
    def queryTimebaseOffset(self):
        return float(self.do_query_string(":TIMebase:POSition?"))

    def timebase(self, scale):
        self.do_command(":TIMebase:SCALe %f" % scale)
 
    def queryTimebase(self):
         return float(self.do_query_string(":TIMebase:SCALe?"))

    def acquireMode(self, mode):
        self.do_command(":ACQuire:TYPE %s" % mode)

    def queryAcquireMode(self):
        return self.do_query_string(":ACQuire:TYPE?")

    def offset(self, channelNumber, offset):
        if (not channelNumber in self.AllChannels):
            raise ValueError('Channel not available')
        self.do_command(":CHANnel%d:OFFSet %f" % (channelNumber, offset))

    def queryOffset(self, channelNumber):
        if (not channelNumber in self.AllChannels):
            raise ValueError('Channel not available')
        return float(self.do_query_string(":CHANnel%d:OFFSet?" % channelNumber))

    def scale(self, channelNumber, scale):
        if (not channelNumber in self.AllChannels):
            raise ValueError('Channel not available')
        self.do_command(":CHANnel%d:SCALe %f" % (channelNumber, scale))

    def queryScale(self, channelNumber):
        if (not channelNumber in self.AllChannels):
            raise ValueError('Channel not available')
        return float(self.do_query_string(":CHANnel%d:SCALe?" % channelNumber))

    def captureWaveform(self, channelNumbers, csvBool, filename, setSettingsBool,maxPoints = 0):

        for channelNumber in channelNumbers:
            if (not channelNumber in self.AllChannels):
                raise ValueError('Channel not available')
                    
        if setSettingsBool:
            # Set the waveform points mode.
            # self.do_command(":WAVeform:POINts:MODE RAW")
            # qresult = self.do_query_string(":WAVeform:POINts:MODE?")
            # print("Waveform points mode: %s" % qresult)
             
            # Get the number of waveform points available.
            if maxPoints == 0:
                self.do_command(":WAVeform:POINts:MODE MAXimum")
            else:
                self.do_command(":WAVeform:POINts %d" % maxPoints)
            # qresult = self.do_query_string(":WAVeform:POINts?")
            # print("Waveform points available: %s" % qresult)
        
        values = {}
        y_increment = {}
        y_origin = {}
        y_reference = {}
        for channelNumber in channelNumbers:
            # self.showWaveformPreamble(channelNumber)
            
            # Get numeric values for later calculations.
            x_increment = self.do_query_double(":WAVeform:XINCrement?")
            x_origin = self.do_query_double(":WAVeform:XORigin?")
            
            y_increment[channelNumber] = self.do_query_double(":WAVeform:YINCrement?")
            y_origin[channelNumber] = self.do_query_double(":WAVeform:YORigin?")
            y_reference[channelNumber] = self.do_query_double(":WAVeform:YREFerence?")

            values[channelNumber] = self.do_captureWaveformData(channelNumber)

        # Figure out the lenght of the shortest array
        l = 16 * 1024**2
        for channelNumber in channelNumbers:
            if len(values[channelNumber]) < l:
                l = len(values[channelNumber])

        if csvBool:
            # Save waveform data values to CSV file.
            f = open(filename, "w")
            for i in range(0, l):
                time_val = x_origin + (i * x_increment) # assumes time is shared
                
                f.write("%E" % time_val)
                for channelNumber in channelNumbers:
                    voltage = ((values[channelNumber][i] - y_reference[channelNumber]) * y_increment[channelNumber]) + y_origin[channelNumber]
                    f.write(", %f" % voltage)
                f.write("\n")
            f.close()
            print("Waveform format BYTE data written to %s." % filename)
        else:
            time = x_origin+np.arange(0,l)*x_increment
            voltage = np.zeros((l,len(channelNumbers)))      
            for i,channelNumber in enumerate(channelNumbers):
                voltage[:,i] = ((np.asarray(values[channelNumber]) - y_reference[channelNumber]) * y_increment[channelNumber]) + y_origin[channelNumber]
            return time,voltage
        
        
    def setWaveFormCaptureSettings(self,maxPoints = 0):
        # Set the waveform points mode.
        self.do_command(":WAVeform:POINts:MODE RAW")
        qresult = self.do_query_string(":WAVeform:POINts:MODE?")
        print("Waveform points mode: %s" % qresult)
         
        # Get the number of waveform points available.
        if maxPoints == 0:
            self.do_command(":WAVeform:POINts:MODE MAXimum")
        else:
            self.do_command(":WAVeform:POINts %d" % maxPoints)
        qresult = self.do_query_string(":WAVeform:POINts?")
        print("Waveform points available: %s" % qresult)
        
    def getWaveFormReferences(self,channelNumbers):
        y_increment = {}
        y_origin = {}
        y_reference = {}
        
        x_increment = self.do_query_double(":WAVeform:XINCrement?")
        x_origin = self.do_query_double(":WAVeform:XORigin?")
        for channelNumber in channelNumbers:

            # Get numeric values for later calculations.
            y_increment[channelNumber] = self.do_query_double(":WAVeform:YINCrement?")
            y_origin[channelNumber] = self.do_query_double(":WAVeform:YORigin?")
            y_reference[channelNumber] = self.do_query_double(":WAVeform:YREFerence?")
            
        return x_increment, x_origin, y_increment, y_origin, y_reference
    
    
    def captureWaveNoTime(self,channelNumbers,x_increment,x_origin,y_increment,y_origin,y_reference):
        values = {}
        voltage = {}
        for channelNumber in channelNumbers:
            values[channelNumber] = np.array(self.do_captureWaveformData(channelNumber))
            voltage[channelNumber] = (((values[channelNumber]) - y_reference[channelNumber]) * y_increment[channelNumber]) + y_origin[channelNumber]
        return voltage
       
        
       
    def setChannelImpedance(self,impedance="FIFTy"):
    
        self.do_command(":CHANnel1:IMPedance %s"%impedance)
        self.do_command(":CHANnel2:IMPedance %s"%impedance)
        self.do_command(":CHANnel3:IMPedance %s"%impedance)
        self.do_command(":CHANnel4:IMPedance %s"%impedance)
        
    def setWaveIntensity(self,intensity=85):
        self.do_command("DISP:INT:WAV %s"%intensity)
        
    def mathCoincidence(self,channels,offset=20,scale=20):
        
        CH1 = str(channels[0])
        CH2 = str(channels[1])
        # print(type(CH1))
        self.do_command("FUNC1:OPER MULT")
        self.do_command("FUNC1:SOURce1 CHAN%s"%CH1)
        self.do_command("FUNC1:SOURce2 CHAN%s"%CH2)
        self.do_command("FUNC1:DISP ON")
        time.sleep(0.5)
        self.do_command("FUNC1:OFFS %s"%offset)
        self.do_command("FUNC1:SCAL %s"%scale)
        
        # time.sleep(0.01)

        
      
        
# scope = IV()

# scope.reset()
# scope.triggerEdge(IV.Channel1, 0)
# scope.acquireMode(IV.NORMal)
# scope.timebase(100e-6)
# scope.single()
# _,voltage = scope.captureWaveform([1], False,'',False)

# # channelNumbers = [1,2,3]
# # time,voltage = scope.captureWaveform(channelNumbers, False,'',True,maxPoints=0)
    # def captureWaveform(self, channelNumbers, csvBool, filename, setSettingsBool,maxPoints = 0):