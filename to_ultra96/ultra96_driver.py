import json
import time
import socket
import numpy as np
from pynq import Overlay, allocate          # pynq logic cannot be tested outside the Ultra96

# TODO: Consider logging?

class GestureModel:
    """
    Encapsulate FPGA AI logic
    """
    def __init__(self, bitstream_path):
        self.overlay = Overlay(bitstream_path)
        self.dma = self.overlay.axi_dma_0       # TODO: match names with Vivado  
        
        # PYNQ buffers for AI prediction
        self.in_buffer = allocate(shape=(25, 8), dtype='f4')
        self.out_buffer = allocate(shape=(8,), dtype='f4')

    def predict(self, window_data):
        # TODO: ensure model can accept the in_buffer shape
        self.in_buffer[:] = window_data
        
        self.dma.sendchannel.transfer(self.in_buffer)
        self.dma.recvchannel.transfer(self.out_buffer)
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()
        
        return np.argmax(self.out_buffer)

# --------------------------------------------- COMMS CODE ---------------------------------------------
"""
TODO: Yiting

Feel free to change whatever btw, this is just some boilerplate
"""
class ClientConnection:
    def __init__(self, host, port, retries=3, retry_time=3):
        self.RECVSIZE = 1024

        for i in range(retry_time):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                print(f"Attempt #{i+1} connecting to {host}:{port}")
                self.client_socket.connect((host, port))
                print("Connection successful")
                break
            except socket.error as e:
                print(f"Connection failed, retrying in {retry_time}s")
                time.sleep(retry_time)

    def receive_live_data(self, buffer):
        """Receive HW JSON data"""
        # add on to the end of buffer
        # I'm thinking we receive the latest sensor data
        # Maybe also timestamped so the code knows whether to add to sliding window?
        data, addr = self.sock.recvfrom(self.RECVSIZE)

    def transmit_result(self, result):
        self.sock.sendall(result)

# --------------------------------------------- MAIN ---------------------------------------------
"""
1. Receive sensor data from ESP32 via TCP socket
2. Update 500ms sliding window buffer by managing old and new data
3. Trigger FPGA fabric to process current window by streaming buffer via AXI DMA X
4. Retrieve classification result and process against confidence threshold
5. Transmit result to Unity game engine for game state updates
"""
def main(bitstream_path):
    """Execute driver script"""
    main_buffer = np.zeroes((100, 8), dtype='f4')
    window = np.zeroes((25, 8), dtype='f4')
    model = GestureModel(bitstream_path)
    conn = ClientConnection()
    
    while True:
        # TODO: continue implementing architecture

        # 1. Ingest data
        conn.receive_live_data(main_buffer)
        # Into main_buffer

        # 2. Update sliding window
        window = ...

        # 3. Predict and process
        result = model.predicts(window)
        # TODO: process result against some threshold or whatever
        
        # 4. Transmit to Unity
        conn.transmit_result(result)

