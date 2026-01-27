"""
1. Receive sensor data from ESP32 via TCP socket
2. Update 500ms sliding window buffer by managing old and new data
3. Trigger FPGA fabric to process current window by streaming buffer via AXI DMA
4. Retrieve classification result and process against confidence threshold
5. Transmit result to Unity game engine for game state updates

TODO: create methods/classes for these functions
"""
