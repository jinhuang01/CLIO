import torch
import cv2
import argparse
import numpy as np
from vision.ssd.multi_headed_ssd import MultiHeadedSSD
from vision.ssd.predictor import Predictor
import os
from glob import glob
import time 
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.config import multi_headed_config



class eight_unit():
    def __init__(self, Qin=None, Qout=None, args=None, IMG_W=244, IMG_H = 244, HID_W    = 30, HID_H    = 30, MAX_HIDS = 8):

        self.Qin      = Qin
        self.Qout     = Qout
        self.args     = args
        self.IMG_W    = IMG_W
        self.IMG_H    = IMG_H
        self.HID_W    = HID_W
        self.HID_H    = HID_H
        self.MAX_HIDS = MAX_HIDS
        
        self.class_names = [name.strip() for name in open(args.label_file).readlines()]

        self.net = MultiHeadedSSD(len(self.class_names), width_mult=1.0, 
                config=multi_headed_config, is_test=True, 
                widths=[2,4,8], shared_layer_conf='halving_24')
        self.net.default_width = 8

        self.net.load(self.args.trained_model)
        self.net = self.net.eval()
        self.net.default_width = 8
        self.predictor = Predictor(
            self.net, 
            self.net.config.image_size, 
            self.net.config.image_mean,
            self.net.config.image_std, 
            nms_method=None,
            iou_threshold=self.net.config.iou_threshold,
            candidate_size=200,
            sigma=0.5,
        )

    def predict(self):

        while True:
                
            #Check qin.size(). qin.get blocks forever and prevents shutdown
            if self.Qin.qsize()==0:
                time.sleep(0.1)
                continue
        
            print("CM: cloud process got input")
                
            val = self.Qin.get()  
            if(val["quit"]):          
                print("CM: cloud process got quit signal. Quitting.")
                return;
            
            start = time.time() 
            
            #Check if image was transmitted.
            #Otherwise, convert to numpy from bytes
            if("img" in val.keys()):
                img = val["img"]
                img = np.frombuffer(img, dtype=np.uint8, count=-1, offset=0)
                img = img.reshape(self.IMG_H,self.IMG_W)
                img = np.stack([img] * 3, axis=-1)
            else:
                img = np.zeros((self.IMG_H,self.IMG_W,3),dtype=np.uint8)
            
            hid = np.frombuffer(val["hid"], dtype=np.int16, count=-1, offset=0)
            num_channels = int(len(hid)/(self.HID_W*self.HID_H))
            hid = hid.reshape(num_channels, self.HID_W,self.HID_H) * 2**-11
            hid = torch.from_numpy(hid).float()
            
            boxes, labels, probs = self.predictor.predict(hid, 10, 0.4, is_feats=True)
            for i in range(boxes.size(0)):
                box = boxes[i, :]
                cv2.rectangle(
                    img, 
                    (box[0], box[1]), 
                    (box[2], box[3]), 
                    (255,255,255),
                    thickness=1)
                label = "%s: %.2f"%(class_names[labels[i]], probs[i])
                cv2.putText(img, label,
                    (box[0]+20 ,box[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # font scale
                    (255,255,255),
                    1)  # line type
        
            hid = hid.data.numpy()
            
            
            time_bytes = val["time_vals"]
            time_vals = np.frombuffer(time_bytes, dtype=np.uint32)
            time_vals = time_vals * 10**-6 #microseconds to seconds
            time_vals = {
                "gap8_sensing" : time_vals[0],
                "gap8_compute" : time_vals[1],
                "gap8_comm" : time_vals[2],
                "cloud_compute":time.time() - start
            }
                        
            self.Qout.put({"img":img, "hid":hid, "time":time_vals})
        
            print("CM: cloud process finished computing.")
    
        return