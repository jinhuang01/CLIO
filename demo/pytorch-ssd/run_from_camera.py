import cv2
import argparse
import numpy as np
from vision.ssd.multi_headed_ssd import MultiHeadedSSD
from vision.ssd.predictor import Predictor
import os
from glob import glob
import time
from vision.ssd.config import mobilenetv1_ssd_config
import serial
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--net', default='mb2-ssd-lite', type=str)
parser.add_argument('--trained_model', type=str)
parser.add_argument('--label_file', type=str)
args = parser.parse_args()


class_names = [name.strip() for name in open(args.label_file).readlines()]

net = MultiHeadedSSD(len(class_names), width_mult=1.0, config=mobilenetv1_ssd_config, is_test=True)

net.load(args.trained_model)
net.default_width = 32
predictor = Predictor(
    net, 
    net.config.image_size, 
    net.config.image_mean,
    net.config.image_std, 
    nms_method=None,
    iou_threshold=net.config.iou_threshold,
    candidate_size=200,
    sigma=0.5
)


W = 324
H = 244
n   = W*H  
baudrate = int(3 * 10**6)
ser = serial.Serial('/dev/ttyUSB1', baudrate ,timeout=1)#625000)
ser.flushInput()
time.sleep(0.1)

print("Python serial reader started using baudrate %d" % baudrate)

i = 0
delta=1
plt.ion()
plt.figure(1)
while(True):
    #input("Press Enter to continue...")
    
    start = time.time()
    ser.flushInput()
    ser.write(i.to_bytes(1,'little'))
    time.sleep(0.01)    
    
    buff  = ser.read(n)
    
    if(len(buff)==H*W): 
        orig_image = np.frombuffer(buff, dtype=np.uint8, count=-1, offset=0)
        orig_image = orig_image.reshape(H, W)
        img = np.stack([orig_image] * 3, axis=-1)
        #orig_image = np.log(1.0 + 1.0*orig_image)
        orig_image = img
        
        boxes, labels, probs = predictor.predict(img, 10, 0.4)
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            cv2.rectangle(
                    orig_image, 
                    (box[0], box[1]), 
                    (box[2], box[3]), 
                    (255,255,255),
                    thickness=1)
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.putText(orig_image, label,
                (box[0]+20 ,box[1]+40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # font scale
                (255,255,255),
                1)  # line type


        print(boxes.shape, labels)
        delta1=time.time()-start
        plt.clf()
        #plt.imshow(x, cmap="gray")
        plt.imshow(orig_image, cmap='gray')
        plt.title("Frame %d. Capture %.2fs. %.2ffps" % (i, delta1, 1/delta))
        plt.pause(0.001)
        delta = time.time() - start
        i=i+1
    else:
        print("Recieved buffer was wrong size.")
        print("Waited %fs for %d /%d bytes"%(delta,len(buff),n))
