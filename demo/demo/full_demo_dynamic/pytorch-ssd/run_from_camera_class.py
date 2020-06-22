import os
import argparse
import numpy as np
from glob import glob
from multiprocessing import Process, Queue
import tkinter as Tk

from gap8_gui import gui
from cloud_model import eight_unit as model
from serial_reader import simulated_joint_serial_reader as reader
#from serial_reader import simple_joint_serial_reader as reader

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='mb2-ssd-lite', type=str)
parser.add_argument('--trained_model', type=str)
parser.add_argument('--label_file', type=str)
args = parser.parse_args()

Q1          = Queue()
Q2          = Queue()
Qcontrol    = Queue()
root        = Tk.Tk()
this_reader = reader(Qcontrol=Qcontrol, Qout=Q1)
p_reader    = Process(target=this_reader.read_bytes,args=())
this_model  = model(Qin=Q1, Qout=Q2, args=args)
p_model    = Process(target=this_model.predict,args=())
this_gui    = gui(root, Qin=Q2, Qcontrol=Qcontrol,reader_process=p_reader,model_process=p_model)


#Start reader, model, and gui threads 
p_reader.start() #Start reader process
p_model.start() #Start model process
root.after(200,this_gui.do_refresh) 
root.mainloop()






