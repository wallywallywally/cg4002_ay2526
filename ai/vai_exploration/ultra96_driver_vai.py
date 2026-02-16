from pynq_dpu import DPU
import numpy as np
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Vitis AI Inference using .npy files')
    parser.add_argument('--file', type=str, required=True, help='Path to the .npy data file')
    parser.add_argument('--model', type=str, default='cnn.xmodel', help='Path to .xmodel file')
    args = parser.parse_args()

    # 1. Validation
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        sys.exit(1)

    # 2. Load DPU & Model
    # Note: 'dpu.xclbin' must be in the same folder
    try:
        dpu = DPU('dpu.xclbin') 
        dpu.load_model(args.model)
    except Exception as e:
        print(f"Hardware Error: {e}")
        sys.exit(1)

    # 3. Load .npy Data
    # This file is already (44, 25) thanks to your extraction script
    raw_data = np.load(args.file)
    
    # Add batch dimension -> (1, 44, 25)
    input_data = np.expand_dims(raw_data, axis=0).astype(np.float32)

    # 4. Run on FPGA
    dpu.set_input_tensor(0, input_data)
    dpu.execute()
    predictions = dpu.get_output_tensor(0)

    # 5. Output Results
    result = np.argmax(predictions)
    
    print("-" * 30)
    print(f"FPGA INFERENCE RESULTS")
    print(f"Source File: {os.path.basename(args.file)}")
    print(f"Predicted Class: {result}")
    print("-" * 30)

if __name__ == "__main__":
    main()