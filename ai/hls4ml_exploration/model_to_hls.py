"""
hls4ml ROUTE

Prerequisites:
- Install Vitis 2024.1 on WSL
- Fix locale issues
  > sudo locale-gen en_US.UTF-8
  > sudo update-locale LANG=en_US.UTF-8

Setup Vitis/Vivado for every new shell: 
- source ~/AMDDesignTools/Vivado/2024.1/settings64.sh
- source ~/AMDDesignTools/Vitis/2024.1/settings64.sh

Launch GUI: vivado &
Note that vitis IDE doesn't work, need to install X11 server, but technically I should not need it...

Model_to_hls script:
- cd /mnt/c/Users/Willson/Desktop/cg4002_ay2526/ai
- With miniconda, python model_to_hls.py
"""

"""
Verdict: Discontinue
- HLS synthesised uses >100% of Ultra96's area (LUTs, DSPs, etc...)
- I need more control -> manual Vitis HLS...
"""

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

config = hls4ml.utils.config_from_pytorch_model(
    model,
    input_shape=(None, INPUT_SIZE, 25),
    # channels_last_conversion='full',
    default_reuse_factor=1
)

config['Model']['Precision'] = "ap_fixed<12,4>"
config['Model']['ReuseFactor'] = 64
config['Model']['ConfigArrayPartition'] = False
config['Model']['BramFactor'] = 0

# IO_STREAM
config['Model']['Strategy'] = 'Latency'
config['Model']['FIFO_depth'] = 100                 # Must be 100

config['LayerName'] = {}
for name in ['conv1', 'conv2', 'fc']:
    config['LayerName'][name] = {}
    config['LayerName'][name]['StorageConfig'] = {'StorageType': 'BRAM'}
config['LayerName']['conv1']['ReuseFactor'] = 8
config['LayerName']['conv2']['ReuseFactor'] = 8
config['LayerName']['fc']['ReuseFactor'] = 128
    
# Convert
hls_model = hls4ml.converters.convert_from_pytorch_model(
    model,
    hls_config=config,
    output_dir=OUTPUT_DIR,
    project_name="cnn_hls",
    backend="Vitis",
    io_type="io_stream",
    part="xczu3eg-sbva484-1-i" # Ultra96v2
)

### THESE METHODS NEED VITIS 2024.1 IN WSL
hls_model.compile()
hls_model.build(
    reset=True,           # Clears previous runs to ensure fresh reports
    csim=False,            # Behavioral Sim: Checks accuracy (Math check)
    synth=True,           # Synthesis: Generates RTL (Area/Resource estimate)
    cosim=False,           # C/RTL Cosim: Measures clock cycle latency (Timing check)
    validation=False,      # Compares HLS output vs PyTorch output automatically                    fails with io_stream
    vsynth=True,           # Runs a "Vivado Synthesis" check (more accurate than HLS synth)          !!! takes forever and needs a lot of RAM
    export=False          # Creates the .zip IP-Core for Vivado
)

hls4ml.report.read_vivado_report(OUTPUT_DIR)