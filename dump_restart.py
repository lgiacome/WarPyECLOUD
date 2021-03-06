from warp import dump as warpdump
from io import StringIO
import sys
from warp_pyecloud_sim import warp_pyecloud_sim

def dump(sim, filename):
    sim.solver.solver.laser_func = None
    del sim.solver.em3dfft_args['laser_func']
    sim.laser_func = None
    sim.text_trap = None
    sim.original = None        
#    sim.custom_plot = None
#    sim.custom_time_prof = None
    sim.chamber = None
    warpdump(filename)

def reinit(sim, laser_func, custom_plot, custom_time_prof = None):
    sim.laser_func = laser_func
    sim.custom_plot = custom_plot
    sim.custom_time_prof = custom_time_prof
    if custom_time_prof is None:
        sim.time_prof = sim.gaussian_time_prof
    else:
        sim.time_prof = sim.self_wrapped_custom_time_prof

    sim.solver.solver.laser_func = sim.laser_func
    sim.solver.em3dfft_args['laser_func'] = sim.laser_func
    sim.print_solvers_info() 
    sim.text_trap = {True: StringIO(), False: sys.stdout}[sim.enable_trap]
    sim.original = sys.stdout
