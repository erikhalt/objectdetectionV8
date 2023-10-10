from Detector import Detector

class Main:
    def __init__(self) -> None:
        """
        This Program will focus on pretrained models just for concept uses.
        For this one we will try out Ultralytics YoloV8n
        """
        self.Classpath = 'class.names'
        self.modelname = 'yolov8n.pt'


    def run(self):
        self.detector = Detector(self.Classpath,self.modelname)

if __name__ == "__main__":
    program = Main()
    program.run()
