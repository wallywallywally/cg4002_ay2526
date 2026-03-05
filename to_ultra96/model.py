import time
import numpy as np
from scipy.ndimage import uniform_filter1d
from pynq import Overlay, allocate, get_rails

# --------------------------------------------- MODEL ---------------------------------------------

class CNN:
    RAW_CH = 10
    IN_CH = 30
    IN_LEN = 25
    NUM_CLASSES = 10

    CONTROL_REGISTER = 0x00
    
    """
    Setup CNN IP block and input/output buffers
    """
    def __init__(self, bitstream_path):
        self.cnn = Overlay("cnn.bit").cnn_top_0
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
        
# --------------------------------------------- DATA PROCESSING ---------------------------------------------

def process_raw_signal(df):
    df = df.astype(np.float32)
    sensors_only = df[:, 1:9]

    # Smooth
    cleaned_sensors = uniform_filter1d(sensors_only, size=3, axis=0)

    return cleaned_sensors

def engineer_features(df):
    # Extract and normalise to [-1, 1]
    SENSOR_RANGES = {
        'accel': 8.0 * 9.80665,            # +/-8 g
        'gyro': 500 * (np.pi / 180),       # +/-500 deg/s -> convert to rad/s
        'flex_min': 800,                   # some buffer for anyone
        'flex_max': 2100,    
        'press': 4095.0                    # 12b
    }

    df = df.astype(np.float32)
    accel = df[:, 0:3] / SENSOR_RANGES['accel']
    gyro = df[:, 3:6] / SENSOR_RANGES['gyro']
    flex = (df[:, 6:7] - SENSOR_RANGES['flex_min']) / (SENSOR_RANGES['flex_max'] - SENSOR_RANGES['flex_min'])
    flex = np.clip(flex, 0, 1)
    press = df[:, 7:8] / SENSOR_RANGES['press']
    
    # Get magnitudes
    accel_mag = np.linalg.norm(accel, axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(gyro, axis=1, keepdims=True)

    features_base = np.hstack([accel, gyro, flex, press, accel_mag, gyro_mag])

    # Get mean and std dev for all
    w_mean = np.mean(features_base, axis=0)
    mean_feat = np.tile(w_mean, (features_base.shape[0], 1))
    w_std  = np.std(features_base, axis=0)
    std_feat  = np.tile(w_std, (features_base.shape[0], 1))
    
    return np.hstack([features_base, mean_feat, std_feat])

def process_window(window):
    processed = process_raw_signal(window)
    engineered = engineer_features(processed)
    return engineered