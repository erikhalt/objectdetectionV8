import cv2,numpy as np, time
from ultralytics import YOLO

class Detector: 
    def __init__(self,classpath,modelname):
        self.modelName = modelname
        self.classesPath = classpath
        self.readClasses()
        self.model = self.initModel()

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classeslist = f.read().splitlines()

    def initModel(self):
        return YOLO(self.modelName)