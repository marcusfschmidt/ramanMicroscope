 #%%
import numpy as np
import StageInterface
import AcquisitionOscilloscope
import MicroscopyInterface
import keyboard
import time
import sys
import os

if __name__ == '__main__':

    #change working directory to file location
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)     


    #Initialize scope, stage and microscope.
    #The parameters of the scope can be changed when loading it. Acquiremode can be "normal", "average", "peak" or "hires".
    #The microscope loglevel parameter can be set to 'debug', 'info', 'warning', 'error' or 'critical' and determines what information is printed to the console when the server is running.
    scope = AcquisitionOscilloscope.AcqScope(channelNumber = 4, scale = 10, timebase = 10e-3, avgNumber = 16, acqMode='hires')
    stage = StageInterface.PI(prologixBool = True, Address = "ASRL3::INSTR", prologGPIBAddress = '7', X = 'A', Y = 'B', correctionAngle = 0, velocity = 100)
    microscope = MicroscopyInterface.Microscope(stage,scope, loglevel='critical', ramanPath='csv/', fastScanPath='fastScan/')

    # Set parameters for the scan below
    #______________________________________________________________________________________________________________________
    #set to True if you want to scan in a snake-like pattern
    snakeLike = True
    #print or not print the steps in the first scan
    printBool = True

    #The below boolean determines whether the outut data should multiplied by -1
    #"True" typically corresponds to fluorescence or Raman signal from a suitable sample
    #"False" typically corresponds to bright-field microscopy, where the lack of sample allows light through
    negativeSignalBoolean = True

    #Boolean to enable or disable the merging of adjacent clusters
    mergeBoolean = False

    #resolution for the first scan in microns
    resolution = 1

    #length of scanning area in microns
    lenx = 30
    leny = 30

    #Title of the fast scan file, if saved
    fastScanTitleString = "verdi_poly_40nm_gold_test1"

    #padding to be added around the search grid. Note that if the grid is close to the edge of the travel range (0 to 100 microns) of the stage, the padding will be ignored 
    padding = 1 #in micros
    #resolution for the Raman scan with the spectrometer in microns. 
    ramanScanningResolution = 1 #in microns
    #boolean to indicate if the saved coordinates are in physical coordinates or in grid coordinates
    physicalCoordinatesBool = False
    #______________________________________________________________________________________________________________________


    #resolution checks
    if  ramanScanningResolution > resolution:
        print("The Raman scanning resolution must be numerically smaller than the old resolution, exiting")
        sys.exit()

    ratio = resolution / ramanScanningResolution
    if not ratio.is_integer():
        print("In order for the new grid to be centered properly, the new resolution must be a factor of the old resolution. If you use this resolution, the grid will be offset by a small amount. Appropriate padding should be added to the grid to account for this.")
        print("Press [c] to continue")

        while True:
            if keyboard.is_pressed('c'):
                break
        print("")

    #find the grid size
    nx = int(np.ceil(lenx/resolution)) + 1
    ny = int(np.ceil(leny/resolution)) + 1

    #check if res and grid size exceeds maximum travel (= 100 microns)
    if resolution*nx > 100 or resolution*ny > 100:
        print("ERROR: Grid size exceeds maximum travel")
        sys.exit()
    

    threshold, cluster = microscope.findBacteriaClusters(nx, ny, resolution, saveTitleString = fastScanTitleString, printBool = printBool, snakeLike = snakeLike, negativeSignalBoolean = negativeSignalBoolean, mergeBoolean = mergeBoolean)

    #wait for user input with key press "c" to continue and ask the user to switch out the fiber between PMT and spectrometer
    print("Moving to Raman Scan, please switch out the fiber between PMT and spectrometer.")
    print("Press [c] to continue")

    while True:
        if keyboard.is_pressed('c'):
            break
    print("Scanning for Raman signal in chosen cluster...")
    #cluster = (0,0,0,0)
    microscope.ramanScan(cluster, resolution, ramanScanningResolution, padding, snakeLike = snakeLike, physicalCoordinatesBool = physicalCoordinatesBool)















# #%%
# def plotFunc(filename, xmax, ymax, res):

#     file = np.load(filename)
#     file = np.transpose(file)
#     file = np.rot90(file, 2)


#     x = np.flip(np.arange(0, (xmax+1)*res, res))
#     y = np.flip(np.arange(0, (ymax+1)*res, res))

#     extent = [x[-1], x[0], y[-1], y[0]]

#     return file, extent, x ,y


# volt, _, x,y = plotFunc("2PA 35x35 res 1 maybe work (damage from 1PA).npy", 35, 35, 1)
# fig, ax = plt.subplots(1,1)
# ax.pcolor(x,y,volt)
# ax.invert_yaxis()
# ax.invert_xaxis()
# plt.gca().set_aspect("equal")
