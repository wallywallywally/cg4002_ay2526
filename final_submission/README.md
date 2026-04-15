# Model training
1. Run all the cells in model.ipynb in order
2. After training the final model, export its weights as cnn_weights.pth
3. Convert weights to C++ header (weights.h)


# Vitis + Vivado Route
In Vitis IDE:
1. Create new HLS component
2. Add source files
   - Convert model from Python to C++, then set as hls.syn.top function in hls_config.cfg
   - Copy weights.h into source folder
3. Add test bench files
   - Pass data into the model and check output
   - Used for C Simulation and C/RTL Cosimulation
4. Run flow programmes
   - NOTE: If C-Sim fails after the above steps, delete model_to_hls/cg4002_cnn/cg4002_cnn and let Vitis regenerate
   - C Simulation: check if C++ model runs and results are in line with Pytorch model
   - C Synthesis: generate RTL and check utilisation estimates (area)
   - C/RTL Cosimulation: measure inference latency in clock cycles
   - Implementation (up to an hour): map RTL to HW resources
      - Place and Route gives the most accurate resource usages and power utilisation
      - Power report found in hls/impl/verilog/report/cnn_top_power_routed.rpt
      - See hls/impl/report/verilog/export_impl.rpt for other report locations
   - Package: export to IP block (.zip) for Vivado

In Vivado:
1. Create new project
   - Default part: Boards > Ultra96v2
2. Create block design
   - Add Zynq Ultrascale+ MPSoC, then Run Block Automation
   - Add model IP block to Settings > IP > Repository, then Run Connection Automation > M_AXI_HPM0_FPD
      - If a new version of the IP block is created, Settings > IP > Repository > Refresh All
      - Report IP Status should show up at the top, and then Upgrade Selected
   - Customise Zynq block > PS-PL Configuration> PS-PL Interfaces
      - Slave Interface > AXI HP > HPC0 FPD > enable
      - Master Interface > AXI HPM1 FPD > disable
   - Run Connection Automation > S_AXI_HPC0_FPD
   - No more suggestions for connection automations, and address editor shows no incomplete
   - Validate Design (blue checkbox on the toolbar)
3. Sources > Design Sources > cnn > right-click and Create HDL Wrapper
4. Generate bitstream

Move to Ultra96:
1. Copy necessary files to /to_ultra96
   - [Project_Name].runs/impl_1/cnn_wrapper.bit
   - [Project_Name].gen/sources_1/bd/cnn/hw_handoff/cnn.hwh
2. ```scp -r to_ultra96 xilinx@makerslab-fpga-17.ddns.comp.nus.edu.sg:/home/b10```
3. Rename both files to the same name e.g. cnn.bit, cnn.hwh

On Ultra96:
1. Access Jupyter
   - On laptop: ```ssh -L 9090:localhost:9090 xilinx@172.26.191.142```
      - Normal ssh won't work
   - On browser: http://localhost:9090/ ; pw: xilinx
   - This can only see files in /home/xilinx/jupyter_notebooks
2. Upload files
3. Run driver script