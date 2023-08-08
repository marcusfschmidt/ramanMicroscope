#%%
from typing import List
from fastapi import FastAPI, Header, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.responses import HTMLResponse

from pydantic import BaseModel
import uvicorn

import os
import numpy as np
import pandas as pd

from math import copysign
import sys
import traceback

import subprocess
import time


app = FastAPI()
class labspecserver(object):

    def __init__(self, app, port = 5500, logLevel = 'critical'):
        self.numberOfPoints = 0
        self.coordinateSet = []
        self.port = port
        self.logLevel = logLevel
        self.app = app
 
    def runServer(self):
        uvicorn.run(self.app, host="0.0.0.0", port=self.port, timeout_keep_alive=0, log_level=self.logLevel)
    
    def setGrid(self, scanningGrid, physicalGrid):
        self.numberOfPoints = len(scanningGrid)
        self.scanningGrid = scanningGrid
        self.physicalGrid = physicalGrid

# app = FastAPI()

# @app.on_event("startup")
# async def startup_event():
#     print("Server started.")
#     print("Start the script on LabSpec and acquisiton will start. The program will terminate when the scan is complete.")




# @app.post("/initialize", status_code=200)
# async def initialize():
#     return {"points": labspecserver.numberOfPoints}

# @app.post("/quit", status_code=199)
# async def quit(self):
#     print("Scan complete, quitting...")
#     sys.exit(0)


# @app.post("/acquireSpectrum", status_code=200)
# async def acquireSpectrum(self, file: UploadFile = File(...)):
#     try:
#         contents = await file.read()
#         contents = contents.decode(errors='ignore').splitlines()
#     except Exception as ex:
#         traceback.print_exception(sys.exc_info())

#     npArr = []

#     for line in range(28, len(contents)-1):
#         lineArr = contents[line].split('\t')
#         npArr.append(lineArr)
#     tpArr = np.transpose(np.array(npArr))
#     df = pd.DataFrame([tpArr[1]], columns=tpArr[0])

#     #find filename without extension
#     fileName = file.filename.split('.')[0]
#     title = fileName.split('_')[0]
#     id = fileName.split('_')[1]
#     coordinateCounter = int(fileName.split('_')[2])
#     coordinate = str(self.scanningGrid[coordinateCounter])

#     fileName = title + '_' + id + '_(' + coordinate + ')'

#     csv = self.csvPath + fileName +  '.csv'

#     retObj =  {
#         'success': True,
#         'csvPath': csv
#     }
    
#     try:
#         header = False
#         if not os.path.isfile(csv):
#             header = True
#         df.to_csv(csv, mode='a', header=header, index=False)
#     except:
#         retObj['success'] = False

#     return retObj


# @app.post("/moveStage", status_code=200)
# async def moveStage(self, currentPoint: int):
#     newPoint = self.physicalGrid[currentPoint]
#     print("Moving to grid point: " + str(self.scanningGrid[currentPoint]))
#     self.stage.moveToCoordinate(newPoint)
#     return {"success": True}

# @app.post("/acqFinished", status_code=200)
# async def acqFinished(self):
#     print("Acquisition finished")
#     return {"success": True}

