import os
import torch
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, 'cnn_weights.pth')
output_path = os.path.join(script_dir, 'weights.h')

model_data = torch.load(input_path, map_location='cpu')

def to_cpp(tensor, name):
    flattened = tensor.detach().numpy().flatten()
    cpp_str = f"const data_t {name}[] = {{" + ",".join(map(str, flattened)) + "};"
    return cpp_str

def check_weights():
    for key in model_data.keys():
        print(f"{key}: {len(model_data[key].detach().numpy().flatten())}")

def get_weights_header():
    with open(output_path, "w") as f:
        f.write("#ifndef WEIGHT_H\n")
        f.write("#define WEIGHT_H\n")
        f.write('#include "ap_fixed.h"\n')
        f.write('typedef ap_fixed<16, 6> data_t;\n')
        for key in model_data.keys():
            f.write(to_cpp(model_data[key], key.replace(".", "_")) + "\n")
        f.write('#endif\n')

# check_weights()
# get_weights_header()