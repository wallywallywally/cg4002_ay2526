import torch
from pytorch_nndct.apis import torch_quantizer
import numpy as np
from model_definition import CNN1DClassifier

model = CNN1DClassifier(input_size=44, num_classes=8)
model.load_state_dict(torch.load("cnn_weights.pth", map_location="cpu"))
model.eval()

# 2. Setup Quantizer
inputs = torch.randn([1, 44, 25])                   # Dummy input for CNN
quantizer_calib = torch_quantizer('calib', model, (inputs))
quantizer_test = torch_quantizer('test', model, (inputs))
quant_model_calib = quantizer_calib.quant_model
quant_model_test = quantizer_test.quant_model

# 3. Load your calibration data (The .npy file you made!)
calib_data = np.load("calibration_data.npy") # Shape (100, 44, 25)

# 4. Calibration Loop
print("Starting Calibration...")
with torch.no_grad():
    for i in range(len(calib_data)):
        # Take one sample, add batch dim -> (1, 44, 25)
        sample = torch.tensor(calib_data[i:i+1], dtype=torch.float32)
        quant_model_calib(sample)

with torch.no_grad():
    quant_model_test(inputs)

# 5. Export the quantized configuration
quantizer_calib.export_quant_config()
quantizer_test.export_xmodel(deploy_check=False, output_dir="quantize_result")
print("Quantization Config Exported!")