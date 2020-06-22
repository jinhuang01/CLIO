import serial
import time
import numpy as np

class simple_joint_serial_reader():
    
    def __init__(self, Qcontrol=None, Qout=None, baudrate=int(3 * 10**6), device = '/dev/ttyUSB1', read_timeout = 2, IMG_W=244, IMG_H = 244, HID_W    = 30, HID_H    = 30, MAX_HIDS = 8):
    
        self.baudrate = baudrate
        self.Qcontrol = Qcontrol
        self.Qout     = Qout
        self.device   = device
        self.read_timeout=read_timeout
    
        self.IMG_W    = IMG_W
        self.IMG_H    = IMG_H
        self.HID_W    = HID_W
        self.HID_H    = HID_H
        self.MAX_HIDS = MAX_HIDS

        self.ser = serial.Serial(device,baudrate=self.baudrate, timeout=self.read_timeout)
        self.control_config = {'include_img': True, 'num_channels': self.MAX_HIDS, 'quit': False}
        time.sleep(0.1)

    def flush_loop(self):
        #Make sure all writing has stopped
        while(self.ser.in_waiting>0):
            print("Flushing...")
            self.ser.flushInput()
            time.sleep(0.01)
    
    def update_config(self):
        #Read all config change requests
        while(self.Qcontrol.qsize() > 0):
            self.control_config = Qcontrol.get() 
            

    def read_bytes(self):
        
        print("PSR: Starting Python Simple Joint Serial Reader")
        
        while(True):

            #Check for updated control info
            self.update_config()        
            if self.control_config['quit']:
                print("PSR: Got quit control signal. Quitting.")  
                return
            
            #Flush serial input incase of pending writes
            self.flush_loop()
            
            #Send the data read ready signal
            print("PSR: Sending data read ready signal to Gap8")        
            sig = int(self.control_config['include_img'])
            send_bytes = sig.to_bytes(1, 'little')
            send_bytes += self.control_config['num_channels'].to_bytes(1, 'little')
            ser.write(send_bytes)
            start = time.time()
        
            #Determine number of bytes to read
            hid_bytes  = self.control_config['num_channels']*self.HID_W*self.HID_H*2
            time_bytes = 12
            num_bytes = hid_bytes + time_bytes
            if include_img:
                img_bytes = self.IMG_W*self.IMG_H
                num_bytes += img_bytes
        
            buff = ser.read(num_bytes)
            out={}
            out["quit"]=False
            if(len(buff)==num_bytes):
                print("  PSR: Recevided image and hids")
                
                if include_img:
                    out["img"] = buff[0:img_bytes]
                    out["hid"] = buff[img_bytes:img_bytes+hid_bytes]
                    out["time_vals"] = buff[img_bytes+hid_bytes:]
                else:
                    out["hid"] = buff[:hid_bytes]
                    out["time_vals"] = buff[hid_bytes:]
            
                self.Qout.put(out)
            else:
                print("  PSR: Got %d bytes expected %d bytess"%(len(buff),img_bytes+hid_bytes))
                continue
        return   
        
        
        
        
        
        
class simulated_joint_serial_reader():
    
    def __init__(self, Qcontrol=None, Qout=None, baudrate=int(3 * 10**6), device = '/dev/ttyUSB1', read_timeout = 2, IMG_W=244, IMG_H = 244, HID_W    = 30, HID_H    = 30, MAX_HIDS = 8,):
    
        self.baudrate = baudrate
        self.Qcontrol = Qcontrol
        self.Qout     = Qout
        self.device   = device
    
        self.IMG_W    = IMG_W
        self.IMG_H    = IMG_H
        self.HID_W    = HID_W
        self.HID_H    = HID_H
        self.MAX_HIDS = MAX_HIDS
        
        self.control_config = {'include_img': True, 'num_channels': self.MAX_HIDS, 'quit': False}
    
    def update_config(self):
        #Read all config change requests
        while(self.Qcontrol.qsize() > 0):
            self.control_config = self.Qcontrol.get() 
            

    def read_bytes(self):
        
        print("PSR: Starting Python Simulated Joint Serial Reader")
        
        while(True):

            #Check for updated control info
            self.update_config()        
            if self.control_config['quit']:
                print("PSR: Got quit control signal. Quitting.")  
                self.Qout.put({"quit":True})
                return
            
            out={}
            out["quit"]=False
        
            #Generate synthetic data to send through
            hid_state = self.control_config['num_channels']*self.HID_W*self.HID_H
            hid = np.random.choice(range(5000),hid_state).astype(np.int16).tobytes()
            out["hid"]=hid
            if self.control_config['include_img']:
                img = np.random.choice(range(255),self.IMG_W*self.IMG_H).astype(np.uint8).tobytes()
                out["img"]=img

            out["time_vals"] = np.random.choice(range(1000000),3).astype(np.uint32).tobytes()

            self.Qout.put(out)
            
            time.sleep(1)
            
        return   