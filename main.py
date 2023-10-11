from Detector import Detector
import cv2 

class Main:
    def __init__(self) -> None:
        """
        This Program will focus on pretrained models just for concept uses.
        For this one we will try out Ultralytics YoloV8n
        """
        self.Classpath = 'class.names'
        self.modelname = 'yolov8n.yaml'
        self.datasetpath = 'coco.yaml'
        # self.videopath = 'data/P1033684.mp4'
        self.videopath = 'data/production_id_4405593 (360p).mp4'


    def run(self,video:False,img:False):
        self.detector = Detector(self.Classpath,self.modelname,self.datasetpath,self.videopath)
        
        if img:
            pics = [
                'data/vid_4_1000.jpg',
                'data/vid_4_1020.jpg',
                'data/vid_4_1040.jpg',
                'data/vid_4_1060.jpg',
                'data/vid_4_1080.jpg',
            ]
            for i in pics:
                img = self.detector.predict_img(i)
        
        if video:
            self.detector.video()

if __name__ == "__main__":
    """
    The code down below is to run the YOLOV8 model from ultralytics.
    Its tested and works pretty good. Still a few small changes that has to be made
    but program runs.

    """
    program = Main()
    program.run(video=True,img=False)
