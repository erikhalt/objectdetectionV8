from Detector import Detector
import cv2 

class Main:
    def __init__(self) -> None:
        """
        This Program will focus on pretrained models just for concept uses.
        For this one we will try out Ultralytics YoloV8n
        """
        self.Classpath = 'class.names'
        self.modelname = 'yolov8n.pt'
        self.videopath = 'data/P1033684.mp4'


    def run(self,video:False,img:False):
        self.detector = Detector(self.Classpath,self.modelname)
        
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
            self.detector.video(self.videopath)

if __name__ == "__main__":
    program = Main()
    program.run(video=True,img=False)
