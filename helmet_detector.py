import os
import sys
from pathlib import Path
import numpy as np


import cv2
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_download, attempt_load
from utils.torch_utils import select_device
from utils.augmentations import letterbox

from utils.general import (check_img_size, non_max_suppression)
import base64



class Yolo_Detector():
    @torch.no_grad()
    def __init__(self):
        self.weights= "/mnt/Source/Helmet_detection/best.pt"
        # self.data='coco128.yaml'
        self.imgsz=(640, 640)
        self.conf_thres=0.5  
        self.iou_thres=0.35
        self.half=False
        self.classes=None
        self.agnostic_nms=False 
        self.augment=False 
        engine_type = "0"



        # self.device = select_device(device)
        # self.model = DetectMultiBackend(self.weights, device="0", dnn=False, data=self.data)
        #  if pt:  # PyTorch
        self.device = select_device(engine_type)

        model = attempt_load(self.weights if isinstance(self.weights, list) else self.weights, map_location=self.device)
        stride = max(int(model.stride.max()), 32)  # model stride
        self.names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        self.model = model  # explicitly assign for to(), cpu(), cuda(), half()

        imgsz = check_img_size((640, 640), s= 32)  # check image size
        # webcam = source.isnumeric() 


        # self.half &= (self.pt or jit or onnx or engine) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        self.half = False if engine_type is not "cpu" else True

        self.model.model.half() if self.half else self.model.model.float()

        cudnn.benchmark = True
        
        
    def detect(self, content):
        formate = content.split(".")[-1]
        if formate == "mp4":
            cap = cv2.VideoCapture(content)
            while(cap.isOpened()):
                ret, img = cap.read()
                if ret:
                    img_cp = img
                    img = letterbox(img, (640, 640), stride= 32, auto= True)[0]
                    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                    img = np.ascontiguousarray(img)
                    
                    
                    im = torch.from_numpy(img).to(self.device)
                    im = im.half() if self.half else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]

                    pred = self.model(im, augment=self.augment)[0]#, visualize=visualize)
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det= 1000)
                    for i, dets in enumerate(pred):  # per image
            
                        box = []
                        if len(dets)>0:
                            for det in dets:
                                no = int(det[-1])
                                result = self.names[no]
                                conf = float(det[-2])
                                if conf >= 0.85:
                                    x1 = int(det[0])
                                    y1 = int(det[1])
                                    x2 = int(det[2])
                                    y2 = int(det[3])
                                    rect=cv2.rectangle(img_cp,(x1,y1),(x2,y2),(0,255,0),2)
                                    cv2.putText(img_cp, (result+"  "+str(conf)), (x1,y1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 34,0))
                                    cv2.imshow("Rectangled", rect) 
                                    if cv2.waitKey(25) & 0xFF == ord('q'):break
                            # cv2.waitKey(0)
        else:
            img = cv2.imread(content)
            img_cp = img
            img = letterbox(img, (640, 640), stride= 32, auto= True)[0]
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            
            
            im = torch.from_numpy(img).to(self.device)
            im = im.half() if self.half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]

            pred = self.model(im, augment=self.augment)[0]#, visualize=visualize)
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det= 1000)
            for i, dets in enumerate(pred):  # per image
    
                box = []
                if len(dets)>0:
                    for det in dets:
                        no = int(det[-1])
                        result = self.names[no]
                        conf = float(det[-2])
                        if conf >= 0.85:
                            x1 = int(det[0])
                            y1 = int(det[1])
                            x2 = int(det[2])
                            y2 = int(det[3])
                            rect=cv2.rectangle(img_cp,(x1,y1),(x2,y2),(0,255,0),2)
                            cv2.putText(img_cp, (result+"  "+str(conf)), (x1,y1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 34,0))
                    cv2.imshow("image", rect)
                    cv2.waitKey(0) # waits until a key is pressed
                    cv2.destroyAllWindows() 
            
            
            
            
            
            
            
            
            

                  
                
                
                
                
                
                # cv2.imshow('Frame',frame)
                
def main():
    yolo_detector = Yolo_Detector()
    yolo_detector.detect("/mnt/Source/Helmet_detection/Arvig Construction Our Purpose, Our People.mp4")
    
    
if __name__ == "__main__":
    main()  