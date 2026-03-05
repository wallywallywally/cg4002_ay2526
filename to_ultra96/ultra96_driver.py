import os
import time
import json
import socket
import pandas as pd
import numpy as np
from model import CNN, process_window           # Import our custom CNN

# --------------------------------------------- COMMS CODE ---------------------------------------------

"""
TODO (Yiting)
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

    def receive_input(self):
        data, addr = self.sock.recvfrom(self.RECVSIZE)              # blocking
        sample = np.fromstring(data.decode('utf-8'), sep=',')       # e.g. "1,2,3,4,5,6"

        # Basic validation
        if sample.size == CNN.RAW_CH:
            return sample, addr
        return None, None

    def send_output(self, result):
        self.sock.sendall(result)

# --------------------------------------------- MAIN ---------------------------------------------

LOG_FILE = f"log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
COLUMN_ORDER = ["timestamp", "id", "confidence", "PS_power", "PL_power", "move_in", "inference", "move_out", "total"]

def main(bitstream_path):
    # Setup logging
    perf_log = []

    with open("gesture_map.json", "r") as file:
        raw_map = json.load(file)
    gesture_map = {int(v): k for k, v in raw_map.items()}

    cnn = CNN(bitstream_path)
    window = np.zeros((CNN.IN_LEN, CNN.IN_CH), dtype='f4')

    # TODO (Yiting): connection setup
    conn = ClientConnection()

    # Measure idle power over first 5s
    samples = list()
    start_time = time.time()
    while (time.time() - start_time) < 5:
        samples.append((cnn.get_current_power()))
        time.sleep(0.2)
    print(f"Avg PS power: {sum(s[0] for s in samples) / len(samples)} W")
    print(f"Avg PL power: {sum(s[1] for s in samples) / len(samples)} W")

    while True:
        # TODO (Yiting): input
        new_sample, addr = conn.receive_input()

        if new_sample is not None:
            # Update sliding window
            window = np.roll(window, -1, axis=0)
            window[-1, :] = new_sample
            # window[-1, :] = np.random.uniform(low=-1.0, high=1.0, size=(CNN.IN_CH,)).astype('f4')           # stub for testing
            sample_for_pred = process_window(window)

            # Predict and process (or done by Vis?)
            pred_id, logits, metrics = cnn.predict_timed(sample_for_pred.T)
            metrics["PS_power"], metrics["PL_power"] = cnn.get_current_power()

            # Log
            metrics["timestamp"] = time.time()
            metrics["id"] = pred_id
            metrics["confidence"] = logits[pred_id]
            perf_log.append(metrics)

            if len(perf_log) >= 500:                # to tweak?
                df = pd.DataFrame(perf_log)
                df = df[COLUMN_ORDER]
                df.to_csv(LOG_FILE, mode='a', index=False, header=not os.path.exists(LOG_FILE))
                perf_log = []

            # Transmit output
            result = {
                "id": pred_id,
                "gesture": gesture_map[pred_id],
                "confidence": logits[pred_id]
            }

            # TODO (Yiting): output
            conn.send_output(result)

bitstream_path = "cnn.bit"
main(bitstream_path)

# Run this file with:
# sudo -E $(which python) ultra96_driver.py