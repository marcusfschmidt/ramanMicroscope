#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import labspecserver
from fastapi import FastAPI, Header, HTTPException, File, UploadFile, BackgroundTasks
import sys
import os
import traceback
import datetime
import glob
import shutil


class Microscope(object):

    def __init__(self, stage, oscilloscope, loglevel='critical', ramanPath = 'csv/', fastScanPath = 'fastScan/'):
        self.stage = stage
        self.scope = oscilloscope
        self.loglevel = loglevel
        self.csvPath = ramanPath
        self.fastScanPath = fastScanPath



    # From chatGPT. I did not write this nor did I attempt to understand it. Seems to work; finds clusters and determines if any clusters are adjacent (including diagonal adjacency)
    # If you need to change it, consider using chatGPT again. When merging, note the importance of comparing each cluster with every other, including new clusters previously merged.
    
    ##########################################################
    #utility functions for the finding and merging of clusters
    def bfs(self, x, y, matrix, threshold, visited):
            q = [(x, y)]
            x_min, y_min, x_max, y_max = x, y, x, y
            while q:
                x, y = q.pop(0)
                if (x, y) not in visited and matrix[x][y] >= threshold:
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                    visited.add((x, y))  # add to visited after checking neighbors
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        if 0 <= x+dx < len(matrix) and 0 <= y+dy < len(matrix[0]):
                            q.append((x+dx, y+dy))
            return x_min, y_min, x_max, y_max 
    
    
    def is_adjacent(self, cluster1, cluster2):
            for i in range(cluster1[0], cluster1[2] + 1):
                for j in range(cluster1[1], cluster1[3] + 1):
                    if (i, j) in [(cluster2[0] - 1, k) for k in range(cluster2[1], cluster2[3] + 1)] \
                            + [(cluster2[2] + 1, k) for k in range(cluster2[1], cluster2[3] + 1)] \
                            + [(k, cluster2[1] - 1) for k in range(cluster2[0], cluster2[2] + 1)] \
                            + [(k, cluster2[3] + 1) for k in range(cluster2[0], cluster2[2] + 1)] \
                            + [(cluster2[0] - 1, cluster2[1] - 1), (cluster2[0] - 1, cluster2[3] + 1), \
                            (cluster2[2] + 1, cluster2[1] - 1), (cluster2[2] + 1, cluster2[3] + 1)]:
                        return True
                    if (i, j) in [(x, y) for x in range(cluster2[0], cluster2[2] + 1) for y in range(cluster2[1], cluster2[3] + 1)]:
                        return True
            return False

    def merge(self, c1, c2, matrix, threshold):
            x_min = min(c1[0], c2[0])
            y_min = min(c1[1], c2[1])
            x_max = max(c1[2], c2[2])
            y_max = max(c1[3], c2[3])
            
            # extend the merged cluster to form a square
            for i in range(x_min, x_max + 1):
                for j in range(y_min, y_max + 1):
                    if matrix[i][j] >= threshold:
                        x_min = min(x_min, i)
                        y_min = min(y_min, j)
                        x_max = max(x_max, i)
                        y_max = max(y_max, j)
            
            return x_min, y_min, x_max, y_max
    #utility functions end
    ##########################################################


    def find_clusters(self, matrix, threshold):
        visited = set()  # use a set to store visited indices
        clusters = []

        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if (i, j) not in visited and matrix[i][j] >= threshold:
                    y_min, x_min, y_max, x_max = self.bfs(i, j, matrix, threshold, visited)
                    clusters.append((x_min, y_min, x_max, y_max))
        
        return clusters
    
    def merge_clusters(self, clusters, matrix, threshold):
        """
        Find clusters of adjacent pixels with values >= threshold in a 2D matrix.
        Return a list of clusters, where each cluster is a tuple of (x_min, y_min, x_max, y_max).
        """          
        # Copy the input list to avoid modifying the original list
        clusters_copy = list(clusters)

        # Initialize a list to store the merged clusters
        merged_clusters = []

        # Merge adjacent clusters until no more merges are possible
        while True:
            merged = False
            n_clusters = len(clusters_copy)
            for i in range(n_clusters):
                for j in range(i+1, n_clusters):
                    c1 = clusters_copy[i]
                    c2 = clusters_copy[j]
                    if self.is_adjacent(c1, c2):
                        merged_cluster = self.merge(c1, c2, matrix, threshold)
                        clusters_copy.remove(c1)
                        clusters_copy.remove(c2)
                        clusters_copy.append(merged_cluster)
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                break

        # Return the final list of merged clusters
        merged_clusters = clusters_copy
 
        return np.array(merged_clusters)
   
    def findAndMerge(self, matrix, threshold, mergeBoolean):
        clusters = self.find_clusters(matrix, threshold)
        if mergeBoolean:
            clusters = self.merge_clusters(clusters, matrix, threshold)
        return clusters

    #Function to define a scanning grid
    def defineGrid(self, size, startpoint, resolution, offset = 0, snakeLike = False, lowerbound = 0, upperbound = 100):
        """
        Function to define a scanning grid

        Args:
            size (tuple): Size of the scanning grid in x and y direction in microns
            startpoint (tuple): Coordinates of the startpoint of the scanning grid in microns
            resolution (float): Resolution of the scanning grid in microns
            offset (float): Offset of the scanning grid in microns
            snakeLike (bool): Whether the scanning grid should be snake-like
            lowerbound (float): Lower bound of the scanning grid in microns. Beware the travel range of the stage.
            upperbound (float): Upper bound of the scanning grid in microns. Beware the travel range of the stage.

        Returns:
            physicalGrid (np.array): Array containing the physical coordinates of the scanning grid, 1D
            scanningGrid (np.array): Array containing the scanning coordinates of the scanning grid, 1D
            xaxis (np.array): Array containing the x-axis coordinates of the physical grid, from a meshgrid
            yaxis (np.array): Array containing the y-axis coordinates of the physical grid, from a meshgrid
        """
        sizex, sizey = size
        x0, y0 = startpoint

        #define x,y meshgrids and include a potential offset
        xaxis = np.arange(0, sizex, resolution)
        yaxis = np.arange(0, sizey, resolution)
        x, y = np.meshgrid(xaxis, yaxis)
        x = x + x0 - offset
        y = y + y0 - offset

        #Check whether we are exceeding the scanning area
        x_keep = (x >= lowerbound) & (x <= upperbound)
        y_keep = (y >= lowerbound) & (y <= upperbound)
        # Keep only the rows and columns that meet the condition
        x = x[y_keep[:, 0], :][:, x_keep[0, :]]
        y = y[y_keep[:, 0], :][:, x_keep[0, :]]

        #Define the physical grid (i.e. physical coordinates) after removing the rows and columns that exceed the scanning area
        physicalGrid = np.dstack((x,y))

        #make new meshgrid wih scanning grid integer coordinate points based on the length of x,y
        xScan, yScan = np.meshgrid(np.arange(x.shape[1]), np.arange(y.shape[0]))
        scanningGrid = np.dstack((xScan,yScan))

        # if snakelike, flip every second row
        if snakeLike:
            scanningGrid[1::2, :] = scanningGrid[1::2, ::-1]
            physicalGrid[1::2, :] = physicalGrid[1::2, ::-1]

        #reshape the grids to be 1D for easy indexing when communicating with the labspec server
        scanningGrid = scanningGrid.reshape(-1,2)
        physicalGrid = physicalGrid.reshape(-1,2)

        return scanningGrid, physicalGrid, xaxis, yaxis


    # acquire data and move the stage
    def acquireAndMove(self, nx, ny, resolution, snakeLike = False, printBool = False):
        """
        Function to acquire data and move the stage for a search with the PMT

        Args:
            nx (int): Number of points in the x direction
            ny (int): Number of points in the y direction
            resolution (float): Resolution of the scanning grid in microns
            snakeLike (bool): Whether the scanning grid should be snake-like
            printBool (bool): Whether to print the progress of the search
            
        Returns:
            dataOut (np.array): Array containing the acquired data
            xaxis (np.array): Array containing the x-axis coordinates of the physical grid, from a meshgrid
            yaxis (np.array): Array containing the y-axis coordinates of the physical grid, from a meshgrid
        """

        dataOut = np.zeros((nx, ny))

        size = np.array([nx, ny])*resolution
        grid, physicalGrid,xaxis,yaxis = self.defineGrid(size, (0,0),resolution, offset = 0, snakeLike = snakeLike)
        dataToFile = np.zeros((len(grid),3))
        import time
        for n,i in enumerate(grid):
            if printBool:
                print("", end="\r")
                print(f"Step %s out of %s" % (n+1, len(grid)), end = "")
            
            y,x = physicalGrid[n]
            self.stage.moveToCoordinates((x, y))
            
            meanData = np.mean(self.scope.acquire())
            dataOut[i[0],i[1]] = meanData

            dataToFile[n, 0:2] = i
            dataToFile[n, 2] = meanData

        print("")
        return dataOut,dataToFile,xaxis,yaxis

    def findBacteriaClusters(self, nx, ny, resolution, saveTitleString, printBool = True, snakeLike = False, negativeSignalBoolean = True, mergeBoolean = True):
        """
        Function to find bacteria clusters with the PMT

        Args:
            nx (int): Number of points in the x direction
            ny (int): Number of points in the y direction
            resolution (float): Resolution of the scanning grid in microns
            printBool (bool): Whether to print the progress of the search
            snakeLike (bool): Whether the scanning grid should be snake-like
            negativeSignalBoolean (bool): Whether to multiply the data from the PMT with -1 (true) or not (false)
            mergeBoolean (bool): Whether to merge adjacent clusters by default or not

        Returns:
            clusters (np.array): Array containing the coordinates of the clusters
        """

        print("Locating bacteria clusters...\n")
        # acquire data
        

        voltageData,dataToFile,x,y = self.acquireAndMove(nx, ny, resolution, snakeLike, printBool)
        voltageData = np.where(negativeSignalBoolean, -voltageData, voltageData)
        dataToFile[:,2] = np.where(negativeSignalBoolean, -dataToFile[:,2], dataToFile[:,2])

        self.voltageData = voltageData
        self.dataToFile = dataToFile
        self.currentMergeBoolean = mergeBoolean

    
        def on_confirm_clicked(event, fig):
            plt.ioff()
            plt.close(fig)

        def onMaskRemoveClick(event, patches):
            for patch in patches:
                patch.remove()
            patches.clear()

        def onSaveFileClick(event, saveTitleString):
            clusterFindTime = datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S")
            fileName = self.fastScanPath + saveTitleString + "_" + clusterFindTime + '.csv'
            pd.DataFrame(self.dataToFile).to_csv(fileName, header=None, index = False)

        def onNegativeMultiplyClick(event, data,im,ax, patches):
            im.remove()
            self.voltageData = data*(-1)
            self.dataToFile[:,2] = self.dataToFile[:,2]*(-1)
            self.im = ax.imshow(self.voltageData, origin='lower', cmap='viridis')
            for patch in patches:
                patch.remove()
            patches.clear()

        def onMergeToggleClick(event, mergeBoolean, patches, fig):
            self.currentMergeBoolean = not mergeBoolean
            print(self.currentMergeBoolean)
            onMaskRemoveClick(event, patches)
            fig.canvas.draw()
            fig.canvas.flush_events()

            
        #interactive plot to select clusters from threshold
        fig, ax = plt.subplots()
        self.fig = fig
        fig.set_size_inches(8, 6)
        self.im = ax.imshow(voltageData, origin='lower', cmap='viridis')
        #make colorbar close to the right side of the plot
        cbar = fig.colorbar(self.im, ax = ax, pad = 0.005)
        #remove tick labels from colorbar
        cbar.ax.set_yticklabels([])

        patches = []

        #confirm button
        button_ax = fig.add_axes([0.83, 0.02+0.4, 0.15, 0.05])
        button = plt.Button(button_ax, 'Confirm')
        button.on_clicked(lambda event: on_confirm_clicked(event, fig))

        remove_ax = fig.add_axes([0.83, 0.5, 0.15, 0.05])
        removebutton = plt.Button(remove_ax, 'Clear')
        removebutton.on_clicked(lambda event: onMaskRemoveClick(event, patches))

        save_ax = fig.add_axes([0.83, 0.58, 0.15, 0.05])
        saveButton = plt.Button(save_ax, 'Save')
        saveButton.on_clicked(lambda event: onSaveFileClick(event, saveTitleString))

        negative_ax = fig.add_axes([0.83, 0.58+0.08, 0.15, 0.05])
        negativeButton = plt.Button(negative_ax, 'Multiply by -1')
        negativeButton.on_clicked(lambda event: onNegativeMultiplyClick(event,self.voltageData,self.im,ax, patches))

        mergeToggle_ax = fig.add_axes([0.83, 0.58+0.08*2, 0.15, 0.05])
        mergeToggleButton = plt.Button(mergeToggle_ax, 'Toggle merging')
        mergeToggleButton.on_clicked(lambda event: onMergeToggleClick(event,self.currentMergeBoolean, patches, self.fig))



        plt.ion()
        plt.show()

        selected_threshold = np.min(self.voltageData)
        confirmed_threshold = selected_threshold
        confirmedCluster = None
        selectedCluster = None
        
        ax.set_title("Threshold chosen: {:.4f}".format(selected_threshold) + " (all)")
        #run as long as figure exists:
        while plt.fignum_exists(fig.number):

            # check if the confirm button was clicked
            # when the button is clicked, the ginput function runs randomly one last time. the below code is necessary to confirm the wanted threshold and cluster
            if button.on_clicked:
                confirmed_threshold = selected_threshold
                confirmedCluster = selectedCluster


            x, y = plt.ginput(n = 1, timeout=0)[0]
            i = int(np.round(y))
            j = int(np.round(x))
            selected_threshold = self.voltageData[i, j]

            #each overlay is removed when the loop runs again
            for patch in patches:
                patch.remove()
            patches = []

            #overlay the mask on the image without changing the image
            self.clusters = self.findAndMerge(self.voltageData, selected_threshold, self.currentMergeBoolean)
            for n,c in enumerate(self.clusters):
                x1, y1, x2, y2 = c
                x = np.arange(x1, x2+1)
                y = np.arange(y1, y2+1)
                xpoint,ypoint = np.meshgrid(x,y)
                if i in y and j in x:
                    patch = ax.scatter(xpoint, ypoint, s = 20, color= 'red', label = "Selected cluster", alpha = 0.7)
                    selectedCluster = self.clusters[n]
                    patches.append(patch)
                else: 
                    patch = ax.scatter(xpoint, ypoint, s = 20, color= 'black',alpha=0.5, label = "Other cluster")
                    patches.append(patch)

            #add title that shows the current chosen threhsold
            ax.set_title("Threshold chosen: {:.4f}".format(selected_threshold) + ", number of clusters detected: {}".format(len(self.clusters)))
            fig.canvas.draw()
            fig.canvas.flush_events()


        print("")
        print("Threshold chosen: {:.4f}".format(confirmed_threshold))       
        print("Cluster chosen as: {}".format(confirmedCluster))
        print("Continuing...")
        return confirmed_threshold, confirmedCluster*resolution


    #function to add padding given a set of physical coordinates and a scanning resolution
    def addPadding(self, cluster, scanningResolution, padding, roundUpBool = False):
        """
        Parameters
        ----------
        cluster : the physical coordinates of the cluster, x1, y1, x2, y2
        scanningResolution : the scanning resolution in microns
        padding : the padding in microns
        roundUpBool : if True, round the padding to the nearest multiple of the scanning resolution
        
        Returns xsize, ysize, x1_padded, y1_padded
        -------
        """

        x1, y1, x2, y2 = cluster

        if roundUpBool:
            #round padding to nearest multiple of scanning resolution
            padding = np.ceil(padding/scanningResolution) * scanningResolution


        x1_padded = x1 - padding
        y1_padded = y1 - padding
        x2_padded = x2 + padding
        y2_padded = y2 + padding

        xsize = int(np.ceil((x2_padded - x1_padded + 1)/scanningResolution))
        ysize = int(np.ceil((y2_padded - y1_padded + 1)/scanningResolution))

        return xsize, ysize, x1_padded, y1_padded   
    

    # function to perform the Raman scan using server endpoints - program must be rerun after this runs, since I can't shut down the server programmatically and I am not software engineer enough to figure out async stuff
    def ramanScan(self, cluster, oldResolution, newResolution, padding, snakeLike = False, physicalCoordinatesBool = False):
        """
        Parameters
        ----------
        cluster : the physical coordinates of the cluster, x1, y1, x2, y2
        oldResolution : the scanning resolution in microns
        newResolution : the desired scanning resolution in microns
        padding : the padding in microns
        snakeLike : if True, scan in a snake-like pattern
        physicalCoordinatesBool : if True, the file saves the physical coordinates of the grid, if False, the file saves the scanning coordinates of the grid

        Returns None
        -------
        """
   
        #add the desired padding to the chosen cluster
        nx, ny, x0, y0 = self.addPadding(cluster, oldResolution, padding, roundUpBool = False)
        
        #calculate the resolution ratio and the required offset to center the grid properly
        ratio = newResolution/oldResolution
        offset = ((ratio - 1) * newResolution) / 2
        #physical size of the grid in microns
        sizenew = np.array([nx, ny])*oldResolution
        #define scanning grids
        scanningGrid, physicalGrid, x, y = self.defineGrid(sizenew, (x0,y0), newResolution, offset = offset, snakeLike = snakeLike)

        #In order to communicate with the labspec computer, we use a uvicorn server.
        #This means that in order to communicate, we utilize VB scripts to send HTTP requests over the local network to the PC running this script.
        #the @ is a decorator, so that once a request reaches the endpoint e.g. IP/initialize, the function defined w/ 'async def initialize' is called
        #Using this, we send and receive data between the two computers. If troubleshooting, be sure to enable "loglevel = 'info'" when defining this class, otherwise the server will not print anything to the console    
        app = FastAPI()
        server = labspecserver.labspecserver(app, logLevel=self.loglevel)    

        server.setGrid(scanningGrid, physicalGrid)

        

        @app.on_event("startup")
        async def startup_event():
            print("Server started.")
            print("Start the script on LabSpec and acquisiton will start.")

        #Function to run prior to starting the scan
        @app.post("/initialize", status_code=200)
        async def initialize(Title: str):
            self.now =  datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S")
            self.title = Title + "_" + self.now
            self.subcsvPath = self.csvPath + "/" + self.title 

            #check if csvpath/id/ exists, and if not, create it
            if not os.path.exists(self.subcsvPath):
                os.makedirs(self.subcsvPath)
            return {"points": server.numberOfPoints}

        #The files are individually saved to a folder. Upon calling /acqFinished, the files are concatenated into one and the folder is deleted.
        #This approach just happened for easier testing and allowed me to more easily get the accumulations and acqtime without thinking too much.
        #If this proves problematic some day, the other code (i.e. labspec server mmapp) has implementation with writing to csv with mode = 'a' i.e. append mode.
        #in this case you would need to consider how the content is read to ensure you still read the acq time and accumulation
        @app.post("/acquireSpectrum", status_code=200)
        async def acquireSpectrum(file: UploadFile = File(...)):
            try:
                contents = await file.read()
                contents = contents.decode(errors='ignore').splitlines()
            except Exception as ex:
                traceback.print_exception(sys.exc_info())

            npArr = []
            df = pd.DataFrame(contents)

            #find filename without extension
            fileName = file.filename.split('.')[0]
            title = fileName.split('_')[:-1]
            title = '_'.join(title)
            coordinateCounter = (fileName.split('_')[-1])
            # coordinate = str(server.scanningGrid[coordinateCounter])
            fileName = title + '_' + coordinateCounter


            csv = self.subcsvPath + "/" + fileName +  '.csv'
 
            retObj =  {
                'success': True,
                'csvPath': csv
            }
            
            try:
                df.to_csv(csv, header=False, index=False)
            except:
                retObj['success'] = False

            return retObj

        #function that moves the stage to the next grid point.
        @app.post("/moveStage", status_code=200)
        async def moveStage(currentPoint: int):
            newPoint = server.physicalGrid[currentPoint]
            y = newPoint[0]
            x = newPoint[1]

            print("Moving to point " + str(currentPoint+1) + " out of " + str(len(server.scanningGrid)) + ", grid point " + str(server.scanningGrid[currentPoint]) + ", physical coordinates: " + str(newPoint))
            self.stage.moveToCoordinates((x,y))
            return {"success": True}

        #Function to run after the scan is finished. Here, the files are concatenated and the folder is deleted.
        @app.post("/acqFinished", status_code=200)
        async def acqFinished():
            #find all files in subcsvPath and load them with glob
            csvFiles = glob.glob(self.subcsvPath + "/*.csv")
        
            #create an empty list to store the dataframes
            dfList = []
            #read the x and y coordinates from the filenames and concatenate all csv files into one dataframe
            for n,f in enumerate(csvFiles):
                print(f)
                currentPoint = int(f.split("_")[-1].split(".")[0])
                if physicalCoordinatesBool:
                    x,y = server.physicalGrid[currentPoint]
                else:
                    x,y = server.scanningGrid[currentPoint]

                if n == 0: 
                    acquisitionParameters = pd.read_csv(f, sep="=", nrows = 27, header = None).to_numpy()
                    acqtime = acquisitionParameters[0,1].replace("\t", "")
                    accum = acquisitionParameters[1,1].replace("\t", "")

                df = pd.read_csv(f, skiprows=range(0,28), sep='\t', header = None).transpose()
                #set first row as header in df and drop first row
                df.columns = df.iloc[0]
                df = df.drop(df.index[0])

                df.insert(0, 'id', 0)
                df.insert(1, 'x', x)
                df.insert(2, 'y', y)

                #append the dataframe to the list
                dfList.append(df) 

            #concatenate df
            df = pd.concat(dfList, ignore_index=True) 

            startPointStr = str(server.physicalGrid[0])
            resolutionStr = str(newResolution)

            scanParameterString = "x0y0=" + startPointStr + "_res=" + resolutionStr + "_acqtime=" + acqtime + "_accum=" + accum  

            #save the dataframe as a csv file
            df.to_csv(self.csvPath + "/" + self.title + "_" + scanParameterString + ".csv", index=False, header = True)
            shutil.rmtree(self.subcsvPath)
            print("Acquisition finished. To quit the program, press Ctrl+C.")
            return {"success": True}

        #The run server statement. This will start the server and wait for requests. After it is run, it can not be programmatically stopped without killing the process or using asynchroneous programming.
        #This means that the script "locks up", and you can not run any other code until the server is stopped.
        server.runServer()




