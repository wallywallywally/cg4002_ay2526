import time
import json
import numpy as np
from scipy.ndimage import uniform_filter1d
from pynq import Overlay, allocate, get_rails

# --------------------------------------------- MODEL ---------------------------------------------

class CNN:
    RAW_CH = 10
    IN_CH = 30
    IN_LEN = 25
    NUM_CLASSES = 9

    CONTROL_REGISTER = 0x00

    with open("gesture_map.json", "r") as file:
        raw_map = json.load(file)
    GESTURE_MAP = {int(v): k for k, v in raw_map.items()}
    
    """
    Setup CNN IP block and input/output buffers
    """
    def __init__(self, bitstream_path):
        self.cnn = Overlay(bitstream_path).cnn_top_0
        self.rails = get_rails()
        
        # Allocate DMA memory
        # int32 matches ap_fixed<32,12>
        self.input_buffer = allocate(shape=(CNN.IN_CH, CNN.IN_LEN), dtype=np.int32)
        self.output_buffer = allocate(shape=(CNN.NUM_CLASSES,), dtype=np.int32)
        
        # Tell the IP where the data is in physical memory -> see "registers" in overlay
        in_addr = self.input_buffer.device_address
        out_addr = self.output_buffer.device_address

        # Input Address (0x10 is low 32 bits, 0x14 is high 32 bits)
        self.cnn.write(0x10, in_addr & 0xFFFFFFFF)
        self.cnn.write(0x14, in_addr >> 32)

        # Output Address (0x1c is low 32 bits, 0x20 is high 32 bits)
        self.cnn.write(0x1c, out_addr & 0xFFFFFFFF)
        self.cnn.write(0x20, out_addr >> 32)

    # In ap_fixed<32,12>, we have 20 bits
    def to_fixed(self, float_val, frac_bits=20):
        return np.int32(np.round(float_val * (2**frac_bits)))

    def from_fixed(self, int_val, frac_bits=20):
        return int_val.astype(float) / (2**frac_bits)

    # def predict(self, data):
    #     self.input_buffer[:] = self.to_fixed(data)

    #     # Start HW
    #     self.cnn.write(CNN.CONTROL_REGISTER, 1) # ap_start

    #     # Wait for it to finish (poll ap_done)
    #     while not (self.cnn.read(CNN.CONTROL_REGISTER) & 0x2): pass

    #     logits = self.from_fixed(self.output_buffer.copy())
    #     prediction = np.argmax(logits)
    #     return prediction, logits
    
    def predict_timed(self, data):
        fixed_data = self.to_fixed(data)
        
        # CPU -> FPGA
        t0 = time.time()
        self.input_buffer[:] = fixed_data
        t1 = time.time()

        # Prediction
        self.cnn.write(CNN.CONTROL_REGISTER, 1) # ap_start
        t2 = time.time()
        while not (self.cnn.read(CNN.CONTROL_REGISTER) & 0x2): pass
        t3 = time.time()

        # FPGA -> CPU
        t4 = time.time()
        out_buf_copy = self.output_buffer.copy()
        t5 = time.time()

        logits = self.from_fixed(out_buf_copy)
        prediction = np.argmax(logits)
        metrics = {
            "move_in": t1 - t0,
            "inference": t3 - t2,
            "move_out": t5 - t4,
            "total": t5 - t0
        }

        return prediction, logits, metrics
    
    def get_current_power(self):
        ps_watt = self.rails["PSINT_FP"].power.value + self.rails["PSINT_LP"].power.value
        pl_watt = self.rails["INT"].power.value
        return ps_watt, pl_watt

    def get_idle_power(self, time=5, interval=0.2):
        samples = list()
        start_time = time.time()
        while (time.time() - start_time) < time:
            samples.append((self.get_current_power()))
            time.sleep(interval)
        print(f"Avg PS power: {sum(s[0] for s in samples) / len(samples)} W")
        print(f"Avg PL power: {sum(s[1] for s in samples) / len(samples)} W")
        
# --------------------------------------------- DATA PROCESSING ---------------------------------------------

def process_raw_signal(df):
    df = df.astype(np.float32)
    sensors_only = df[:, 1:9]

    # Smooth
    cleaned_sensors = uniform_filter1d(sensors_only, size=3, axis=0)

    return cleaned_sensors

def _get_normalization_scale(min_val, max_val):
    abs_min = abs(min_val)
    abs_max = abs(max_val)
    return max(abs_min, abs_max)

def engineer_features(df):
    SENSOR_RANGES = {
        'accel': (-28.683, 24.72),
        'gyro': (-8.731, 8.731),
        'flex_min': 800,
        'flex_max': 3268,
        'press': 4095
    }

    df = df.astype(np.float32)
    
    accel_raw = df[:, 0:3]
    gyro_raw = df[:, 3:6]
    flex_raw = df[:, 6:7]
    press_raw = df[:, 7:8]
    
    # Handle sensor ranges
    accel_scale = _get_normalization_scale(SENSOR_RANGES['accel'][0], SENSOR_RANGES['accel'][1])
    gyro_scale = _get_normalization_scale(SENSOR_RANGES['gyro'][0], SENSOR_RANGES['gyro'][1])
    
    # Normalize to [-1, 1]
    accel = np.clip(accel_raw / accel_scale, -1.0, 1.0)
    gyro = np.clip(gyro_raw / gyro_scale, -1.0, 1.0)
    
    flex = (flex_raw - SENSOR_RANGES['flex_min']) / (SENSOR_RANGES['flex_max'] - SENSOR_RANGES['flex_min'])
    flex = np.clip(flex, 0, 1)
    
    press = press_raw / SENSOR_RANGES['press']
    press = np.clip(press, 0, 1)

    # Magnitudes
    accel_mag = np.linalg.norm(accel, axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(gyro, axis=1, keepdims=True)
    
    features_base = np.hstack([accel, gyro, flex, press, accel_mag, gyro_mag])
    
    # AGGREGATE STATISTICS (global, across entire window)
    w_mean = np.mean(features_base, axis=0)
    w_std = np.std(features_base, axis=0)
    w_std[w_std == 0] = 1e-6  # Avoid division by zero
    mean_feat = np.tile(w_mean, (features_base.shape[0], 1))
    std_feat = np.tile(w_std, (features_base.shape[0], 1))
    return np.hstack([features_base, mean_feat, std_feat])

def process_window(window):
    processed = process_raw_signal(window)
    engineered = engineer_features(processed)
    return engineered