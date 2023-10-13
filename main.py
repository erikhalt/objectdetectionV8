from Detector import Detector
import cv2
from time import sleep 

class Main:
    def __init__(self) -> None:
        """
        This Program will focus on pretrained models just for concept uses.
        For this one we will try out Ultralytics YoloV8n

        videopath: path to saved mp4 on local pc

        picslist: path to saved images to detect on local pc
        """
        self.Classpath = 'class.names'
        self.modelname = 'yolov8n.pt'
        self.datasetpath = 'coco.yaml'
        self.videopath = 'data/pexels_videos_4698 (360p).mp4'


    def run(self):

        stream_choice = input('Run with Stream activated? [y/n]')
        if stream_choice == 'y' or stream_choice == 'Y':
            stream = True
        else:
            stream = False

        if stream:
            self.detector = Detector(self.Classpath,self.modelname,self.datasetpath,self.videopath,stream)    
        else:
            self.detector = Detector(self.Classpath,self.modelname,self.datasetpath,self.videopath,stream)

        while True:
            cv2.destroyAllWindows()
            choice = input('Img[1], vid[2] "q" to quit')
            
            if choice == 'q':
                break

            if choice == '1':
                pics = [
                    'data/pexels-tyler-tornberg-1587267.jpg',
                ]
                for i in pics:
                    img = self.detector.predict_img(i)
                    continue

            if choice == '2':
                print('To cancel video during runtime press Q on keyboard')
                sleep(5)
                self.detector.video()
                continue
            
            print('Please input 1,2 or q')


if __name__ == "__main__":
    """
    The code down below is to run the YOLOV8 model from ultralytics.
    Its tested and works pretty good. Still a few small changes that has to be made
    but program runs.

    """
    program = Main()
    program.run()
