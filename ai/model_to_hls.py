import hls4ml
import torch
import os
from model_definition import CNN1DClassifier

OUTPUT_DIR = "hls_model"

INPUT_SIZE = 8
OUTPUT_SIZE = 8

model = CNN1DClassifier(input_size=INPUT_SIZE, num_classes=OUTPUT_SIZE)
weights_pth = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cnn_weights.pth')
model.load_state_dict(torch.load(weights_pth))
model.eval()

config = hls4ml.utils.config_from_pytorch_model(model, (INPUT_SIZE, 25))
config['Model']['Precision'] = 'ap_fixed<16,6>'
config['Model']['ReuseFactor'] = 1

# Convert
hls_model = hls4ml.converters.convert_from_pytorch_model(
    model,
    hls_config=config,
    input_shape=(INPUT_SIZE, 25),
    output_dir=OUTPUT_DIR,
    project_name="cnn_hls",
    backend="Vitis",
    part="xczu3eg-sbva484-1-i" # Ultra96v2
)

### THESE METHODS NEED VITIS 2024.1 IN WSL
hls_model.compile()
hls_model.build(
    reset=True,           # Clears previous runs to ensure fresh reports
    csim=True,            # Behavioral Sim: Checks accuracy (Math check)
    synth=True,           # Synthesis: Generates RTL (Area/Resource estimate)
    cosim=True,           # C/RTL Cosim: Measures clock cycle latency (Timing check)
    validation=True,      # Compares HLS output vs PyTorch output automatically
    export=True,          # Creates the .zip IP-Core for Vivado
    vsynth=True           # Runs a "Vivado Synthesis" check (more accurate than HLS synth)          !!! takes forever...
)

hls4ml.report.read_vivado_report(OUTPUT_DIR)