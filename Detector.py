import cv2,numpy as np, time
from ultralytics import YOLO

class Detector: 
    def __init__(self,classpath,modelname):
        self.modelName = modelname
        self.classesPath = classpath
        self.readClasses()
        self.model = self.initModel()
        print('Init done')

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classeslist = f.read().splitlines()

    def initModel(self):
        model = YOLO(self.modelName)
        return model
    
    def predict_img(self,img):
        result = self.model(img)
        image = cv2.imread(img)
        for r in result:
            if len(r.boxes.cls) != 0:
                for coords in r.boxes.xyxy:
                    cv2.rectangle(image,(int(coords[0]),int(coords[1])),(int(coords[2]),int(coords[3])),(255,0,0),1)
            cv2.imshow('window',img)
            cv2.waitKey(1)   
            cv2.destroyAllWindows

    def video(self,videopath):
        cap = cv2.VideoCapture(videopath)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,600)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,600)
        if (cap.isOpened()==False):
            print('Error !!!!!')
            return
        
        
        while cap.isOpened():
            success,frame = cap.read()
            if success:
                result = self.model(frame)
                annotated_frame = result[0].plot()
                annotated_frame = cv2.resize(annotated_frame, (780, 540), 
                                                interpolation = cv2.INTER_AREA)
                cv2.imshow('window',annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

