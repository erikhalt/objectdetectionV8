import cv2,numpy as np, time
from ultralytics import YOLO
from time import sleep

class Detector: 
    def __init__(self,classpath,modelname,dataset,videopath,stream):
        self.modelName = modelname
        self.classesPath = classpath
        self.datasetpath = dataset
        self.stream = stream
        self.videopath = videopath
        self.fps_list = []

        self.readClasses()
        self.model = self.initModel()
    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classeslist = f.read().splitlines()
    

    def initModel(self):
        if self.stream:
            model = YOLO(self.modelName)
            return model
        model = YOLO(self.modelName)
        return model
    
    def predict_img(self,img):
        result = self.model(img)
        image = cv2.imread(img)

        for r in result:
            print(r.boxes)
            if len(r.boxes.cls) != 0:
                for i in range(len(r.boxes.cls)):
                    coords = r.boxes.xyxy[i]
                    cls = int(r.boxes.cls[i])
                    cv2.rectangle(image,
                                  (int(coords[0]),int(coords[1])),(int(coords[2]),int(coords[3])),
                                  (255,0,0),
                                  1)
                    cv2.putText(image,f'{self.classeslist[cls]}', 
                            (int(coords[0]),int(coords[1])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5,
                            (255,0,0),
                            1,
                            2)
                    

            
        cv2.imshow('window',image)
        cv2.waitKey(1)
        sleep(2)   
        cv2.destroyAllWindows

    def video(self):
        cap = cv2.VideoCapture(self.videopath)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,600)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,600)
        
        if (cap.isOpened()==False):
            print('Error !!!!!')
            return
        
        pre_time = 0
        post_time = 0
        while cap.isOpened():
            success,frame = cap.read()
            
            pre_time = time.time()
            
            if success:
                result = self.model(frame,stream=self.stream)
                for r in result:
            
                    if len(r.boxes.cls) != 0:
                        for i in range(len(r.boxes.cls)):
                            coords = r.boxes.xyxy[i]
                            cls = int(r.boxes.cls[i])
                            cv2.rectangle(frame,
                                          (int(coords[0]),int(coords[1])),(int(coords[2]),int(coords[3])),
                                          (255,0,0),
                                          1)
                            cv2.putText(frame,f'{self.classeslist[cls]}', 
                                    (int(coords[0]),int(coords[1])), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5,
                                    (255,0,0),
                                    1,
                                    2)
                        fps = int(1/(pre_time-post_time))
                        self.fps_list.append(fps)
                        cv2.putText(frame,f'FPS: {fps}', 
                                    (10,30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    1,
                                    (0,255,0),
                                    1,
                                    2)
                        cv2.imshow('window',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
            post_time = pre_time
        if len(self.fps_list) != None:
            mean = sum(self.fps_list)/len(self.fps_list)
            print(f'Mean fps during video is {mean}')
