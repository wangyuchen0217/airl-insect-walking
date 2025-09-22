# This script is to automatcally start the simulation for a certain times 
# to collect dataset with write-in logging mode. 


# TARGET_RUNS = 1  # total number of runs

# def sysCall_init():
#     sim = require('sim')
    
#     # Initialize signals only if they do not exist yet
#     if sim.getInt32Signal('run_counter') is None:
#         sim.setInt32Signal('run_counter', 0)
#     if sim.getInt32Signal('needs_restart') is None:
#         sim.setInt32Signal('needs_restart', 0)

# def sysCall_nonSimulation():
#     # Executed periodically while simulation is not running
#     if sim.getSimulationState() == sim.simulation_stopped:
#         needs = sim.getInt32Signal('needs_restart') or 0
#         cnt   = sim.getInt32Signal('run_counter') or 0
#         if needs == 1:
#             sim.setInt32Signal('needs_restart', 0)  # clear the flag
#             if cnt < TARGET_RUNS:
#                 print(f'[CUSTOM] Start run {cnt+1}/{TARGET_RUNS}')
#                 sim.startSimulation()
#             else:
#                 print(f'[CUSTOM] DONE: total runs = {cnt}')

# def sysCall_beforeSimulation():
#     # Executed before a simulation starts
#     pass

# def sysCall_afterSimulation():
#     # Executed after each simulation ends
#     cnt = sim.getInt32Signal('run_counter') or 0
#     sim.setInt32Signal('run_counter', cnt + 1)
#     sim.setInt32Signal('needs_restart', 1)  # request next run

# def sysCall_cleanup():
#     # Executed when the script is destroyed (not needed for per-run logic)
#     pass