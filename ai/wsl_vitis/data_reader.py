import numpy as np

class DataReader:
    def __init__(self, model_path):
        self.data = np.load("calibration_data.npy")
        self.counter = 0
        self.max_batches = len(self.data)

    def get_next(self):
        if self.counter < self.max_batches:
            input_name = "input"            # "input" must match the name used in torch.onnx.export
            batch = self.data[self.counter : self.counter + 1].astype(np.float32)
            self.counter += 1
            return {input_name: batch}
        return None