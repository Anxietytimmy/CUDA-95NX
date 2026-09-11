# CUDA-95NX
- A Stress Testing Utility for the Nintendo Switch.
- This tool is designed to fully stress and stability test the GPU inside the Nintendo Switch.
- This tool can and will cause extreme stress to the GPU and its power delivery components if sufficiently clocked. 
- If using for stability testing, use Core with 64 repeats at high temperatures. 

# Modes
- 4 Profiles, Balanced, Core, Mem, and L2
- Balanced tries to evenly stress memory and the gpu core, with 64ALU Iterations and 32 random memory gather iterations
- Core has 512ALU iterations and 4 RMG iterations, useful for stability (GPU UV) and by far the most intensive mode.
- Mem has 8ALU and 256 RMG Iterations, testing over 4MB
- L2 is identical to mem, however only tests over L2's 256KB

# Display
- The main console displays stats of the current profile, with the numeber of batches and memory rates below
- Errors are displayed both as a log and a general counter, with mismatches, transient/persistent, ALU, input data and timeouts listed.

# Controls
- Dpad U/D controls the number of repeats, default of 4. 
- Y switches between the profiles, going from Balanced -> Core -> Mem -> L2.
- X injects faults into the computation
- Pressing + stops the comp, with a second press exiting.
