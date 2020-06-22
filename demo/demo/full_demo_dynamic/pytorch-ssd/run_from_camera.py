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
import serial
#import threading
#from queue import Queue
from multiprocessing import Process, Queue

#GUI imports
import matplotlib, sys
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pylab as plt
from scipy import ndimage
import tkinter as Tk
from mpl_toolkits.axes_grid1 import make_axes_locatable
import im_tools

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='mb2-ssd-lite', type=str)
parser.add_argument('--trained_model', type=str)
parser.add_argument('--label_file', type=str)
args = parser.parse_args()

imgH    = 244
imgMaxW = 324
crop    = 40
imgW    = imgMaxW - 2*crop
mult    = 1 
dtype   = np.int16
img_bytes = imgH*imgW*mult
hid_bytes = 30*30*8*2


baudrate = int(3 * 10**6)
ser = serial.Serial('/dev/ttyUSB1',baudrate=baudrate, timeout=2)
ser.flushInput()
time.sleep(0.1)

print("Python serial reader started using baudrate %d" % baudrate)


def network_forward(qin, qout, args):
    #load declare and load network 
    class_names = [name.strip() for name in open(args.label_file).readlines()]
    net = MultiHeadedSSD(len(class_names), width_mult=1.0, 
            config=multi_headed_config, is_test=True, 
            widths=[2,4,8], shared_layer_conf='halving_24')
    net.load(args.trained_model)
    net = net.eval()
    net.default_width = 8
    
    #create predictor
    predictor = Predictor(net, 
            net.config.image_size, net.config.image_mean,
            net.config.image_std, nms_method=None,
            iou_threshold=net.config.iou_threshold,
            candidate_size=200, sigma=0.5)

    while True:
        #Check qin.size(). qin.get blocks forever and prevents shutdown
        if qin.qsize() == 0:
            time.sleep(0.1)
            continue
        
        #print("  NN process got input")

        start = time.time()    
        val = qin.get()            
        img, hid, time_vals = val
        boxes, labels, probs = predictor.predict(hid, 10, 0.4, is_feats=True)
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
        #predict_time = time.time() - start
        #print(predict_time)
        time_vals['nn_pytorch'] = time.time() - start
        qout.put((img, hid, time_vals))
    
            
def read_bytes(ser, qout, qcontrol):
    control_config = {'include_img': True, 'num_channels': 8, 'quit': False}
    
    #Make sure all writing has stopped
    while(ser.in_waiting>0):
        print("Flushing...")
        ser.reset_input_buffer()
        time.sleep(0.01)
    
    #t = threading.currentThread()
    #while getattr(t, "do_run", True):
    while True:
        if qcontrol.qsize() > 0:
            control_config = qcontrol.get()    
        if control_config['quit']:
            print('Stopping serial I/0')
            return
        include_img = control_config['include_img']
        
        #Send the ready signal
        #ser.write(b'\x00')
        sig = int(include_img)
        send_bytes = sig.to_bytes(1, 'little')
        send_bytes += control_config['num_channels'].to_bytes(1, 'little')
        ser.write(send_bytes)
        start = time.time()
        #print("  Sending ready signal")

        print(control_config)
        
         
        hid_bytes = control_config['num_channels']*30*30*2
        num_bytes = hid_bytes + 12 #12 bytes for gap8 time measurement
        if include_img:
            num_bytes += img_bytes
        
        buff = ser.read(num_bytes)
        if(len(buff)==num_bytes):
            #print("  Recevided image")
            if include_img:
                offset = img_bytes
                img = buff[0:img_bytes]
                hid = buff[img_bytes : img_bytes + hid_bytes]
                img = np.frombuffer(img, dtype=np.uint8, count=-1, offset=0)
                img = img.reshape(imgH, imgW)
                img = np.stack([img] * 3, axis=-1)
            else:
                offset = 0
                img = np.ones([244,244])
                hid = buff[0:hid_bytes]
            
            hid = np.frombuffer(hid, dtype=np.int16, count=-1, offset=0)
            hid = hid.reshape(control_config['num_channels'], 30,30) * 2**-10
            hid = torch.from_numpy(hid).float()

            time_bytes = buff[offset+hid_bytes:]
            time_vals = np.frombuffer(time_bytes, dtype=np.uint32)
            time_vals = time_vals * 10**-6 #microseconds to seconds
            time_vals = {
                'camera' : time_vals[0],
                'gap8_nn' : time_vals[1],
                'uart_write' : time_vals[2]
            }
            qout.put((img, hid, time_vals))
        else:
            print("  Got %d bytes expected %d bytess"%(len(buff),img_bytes+hid_bytes))


Q1 = Queue()
Q2 = Queue()
controlQ = Queue()


MAX_HIDS = 8
MIN_HIDS = 2
SKIP_HID = 2
HID_W    = 30
HID_H    = 30
HID_GRID = 3

stop_threads=False

#Set winodow properties
root = Tk.Tk()
root.wm_title("Distributed Prediction GUI")
root.configure(bg='white')

#Define top Frame
top_frame = Tk.Frame(root)
top_frame.configure(bg='white')
top_frame.pack(side="top", fill="x")

#Define bottom  Frame
bot_frame = Tk.Frame(root)
bot_frame.pack(side="top", fill="x")
bot_frame.config(relief=Tk.GROOVE, bd=2)

#Create canvas to display Image
root.image     = np.ones((244,244))
fig_image      = plt.figure(figsize=(6,6))
ax             = plt.gca()
plt.title("Remote Image")
im_image       = ax.imshow(root.image,cmap="gray") 
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
divider        = make_axes_locatable(ax)
cax            = divider.append_axes("right", size="5%", pad=0.05)
im_image.set_clim(0,255)
plt.colorbar(im_image, cax=cax)


#Embed the Image
canvas_image = FigureCanvasTkAgg(fig_image, master=top_frame)
canvas_image.draw()
canvas_image.get_tk_widget().pack(side=Tk.LEFT, pady=5, padx=5, fill=Tk.BOTH, expand=1)

#Create canvas to Diaply Hids
root.hids = im_tools.gallery(np.zeros((MAX_HIDS,HID_W,HID_H)), ncols=HID_GRID, nrows=HID_GRID)
fig_hids = plt.figure(figsize=(6,6))
ax             = plt.gca()
im_hids        = ax.imshow(root.hids,cmap="jet") 
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
plt.title("Remote Hidden Activations")
divider        = make_axes_locatable(ax)
cax            = divider.append_axes("right", size="5%", pad=0.05)
im_hids.set_clim(0,6)
plt.colorbar(im_hids, cax=cax)

#Embed the Hids
canvas_hids = FigureCanvasTkAgg(fig_hids, master=top_frame)
canvas_hids.draw()
canvas_hids.get_tk_widget().pack(side=Tk.LEFT, pady=5, padx=5, fill=Tk.BOTH, expand=1)

#Create control check box
state_vars={}
state_vars["Transmit Image"] = Tk.IntVar()
state_vars["Transmit Image"].set(1)
chk_transmit = Tk.Checkbutton(master = bot_frame, text="Send Image", variable=state_vars["Transmit Image"])
chk_transmit.pack(side=Tk.LEFT, expand=1, pady=5)

#Create hidden unit slider
state_vars["Num Hids"] = Tk.IntVar()
state_vars["Num Hids"].set(MAX_HIDS)
hid_select = Tk.Scale(master = bot_frame, label="Hidden Dimension", showvalue=False, length=200, sliderlength=20, from_=MIN_HIDS, to=MAX_HIDS, tickinterval=SKIP_HID, orient=Tk.HORIZONTAL, variable=state_vars["Num Hids"])
hid_select.set(state_vars["Num Hids"].get())
hid_select.pack(side=Tk.LEFT, expand=1, pady=5)

#Create a quit button
button_quit = Tk.Button(master = bot_frame, text = 'Quit', command = quit)
button_quit.pack(side=Tk.LEFT,expand=1, pady=5)

#Define quit function
def quit(*args):
    global stop_threads
    print('quit button press...')
    stop_threads = True
    controlQ.put({'include_img': False, 'num_channels':8, 'quit':True})
    t1.do_run = False
    t2.do_run = False
    
    t1.join()
    t2.join()
    
    root.quit()     
    root.destroy() 

#Define refresh function
#This is the main event loop for displaying updates in the GUI
def do_refresh():
    global start
    
    if Q2.qsize()>0:


        img, hid, time_vals = Q2.get()

        print(time_vals)

        control_config = {'include_img': state_vars['Transmit Image'].get() == 1}
        
        #map slide value to the nearest power of 2
        #only a width of 2, 4, and 8 are supported
        mapping = {2:2, 3:2, 4:4, 5:4, 6:8, 7:8, 8:8}
        num_channels = state_vars['Num Hids'].get()

        #control_config['num_channels'] = state_vars['Num Hids'].get()
        control_config['num_channels'] = mapping[num_channels]
        control_config['quit'] = False

        #controlQ.put(state_vars["Transmit Image"].get())
        controlQ.put(control_config)
    
        root.image = img
        im_image.set_data(root.image)
        canvas_image.draw()  
        
        #take the sum of each channel (sum.shape = (8,))
        #sort by the sum before plotting
        #this forces the dead channels (has sum of 0) to appear last
        selected_hids = hid[0:state_vars["Num Hids"].get()]
        sort_channels = False
        if sort_channels:
            sums = selected_hids.sum(axis=-1).sum(axis=-1)
            idx = np.argsort(-sums)
            selected_hids = selected_hids[idx]
    
        root.hids = im_tools.gallery(selected_hids, ncols=HID_GRID, nrows=HID_GRID)
        im_hids.set_data(root.hids)
        canvas_hids.draw()
        
        proc_time = time.time() - start
        print('approx fps = ', 1/proc_time)
        start = time.time() 
    root.after(200,do_refresh)
    
    #print("App control state variables")
    #for key in state_vars:
    #    print("  %s %d"%(key, state_vars[key].get()))

#Start serial transfer thread
#t1 = threading.Thread(target=read_bytes, args=(ser, Q1))
t1 = Process(target=read_bytes, args=(ser, Q1, controlQ))
t1.start()

#Start neural network thread
#t2 = threading.Thread(target=network_forward, args=(Q1, Q2))
t2 = Process(target=network_forward, args=(Q1, Q2, args))
t2.start()

#Start gui thread with display refresh
start = time.time()
root.after(200,do_refresh)
root.mainloop()
