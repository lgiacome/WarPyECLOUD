import os
import numpy as np
import replaceline as rl
from pathlib import Path

from scipy.constants import c as clight

    
tobecopied1 = 'transient_sim.py'
tobecopied2 = 'restart_sim.py'
tobecopied3 = 'restart_sim_nobunch.py'

current_dir = os.getcwd()
study_folder =  current_dir.split('/config')[0]
scan_folder = study_folder+'/simulations'
if not os.path.exists(scan_folder):
    os.mkdir(scan_folder)
    os.mkdir(scan_folder+'/progress')

launch_file_lines1 = []
launch_file_lines2 = []
launch_file_lines3 = []
#launch_file_lines1 +=['#!/bin/bash\n']
#launch_file_lines2 +=['#!/bin/bash\n']
#launch_file_lines3 +=['#!/bin/bash\n']

exps = np.linspace(-6,6,13)
exps_str = np.array(['m6','m5','m4','m3','m2','m1',
                     '0','1','2','3','4','5','6'])
Emax_vect = 10**exps

# If user defined env variable NCPUS use it otherwise use 10 as default
NCPUS = os.environ.get('NCPUS')
if NCPUS is None:
    NCPUS = '10'

for i, Emax in enumerate(Emax_vect):
    current_sim_ident = 'complete_cavity_Emax_'+exps_str[i]
    print(current_sim_ident)
    current_sim_folder = scan_folder+'/'+current_sim_ident
    if not os.path.exists(current_sim_folder):
        os.mkdir(current_sim_folder)
    os.system('cp -r %s %s'%(tobecopied1, current_sim_folder))
    os.system('cp -r %s %s'%(tobecopied2, current_sim_folder))
    os.system('cp -r %s %s'%(tobecopied3, current_sim_folder))

    curr_sim = current_sim_folder+'/transient_sim.py'
    
    rl.replaceline_and_save(fname = curr_sim,
     findln = 'laser_emax = ', newline = 'laser_emax = %1.0e\n'%Emax)
      
    rl.replaceline_and_save(fname = curr_sim,
     findln = "images_dir = ", newline = "images_dir = '"+current_sim_folder+"/images'\n")

    rl.replaceline_and_save(fname = curr_sim,
     findln = "temps_filename = ", newline = "temps_filename = '"+current_sim_folder+"/transient_temp.h5'\n")

    rl.replaceline_and_save(fname = curr_sim,
     findln = "output_filename = ", newline = "output_filename = '"+current_sim_folder+"/transient_out.h5'\n")
 
    launch_lines1 = ['bsub -B -N -R "rusage[mem=4096]" -o '
                    + current_sim_folder +
                    '/crab.out -n ' + NCPUS + ' -W 1440 mpirun -np '
                    + NCPUS + ' python ' + current_sim_folder
                    +'/transient_sim.py -p 1 1 ' + NCPUS + '\n']
                    
    launch_lines2 = ['bsub -B -N -R "rusage[mem=4096]" -o '
                    + current_sim_folder +
                    '/crab.out -n ' + NCPUS + ' -W 1440 mpirun -np '
                    + NCPUS + ' python ' + current_sim_folder
                    +'/restart_sim.py -p 1 1 ' + NCPUS + '\n']
                    
    launch_lines3 = ['bsub -B -N -R "rusage[mem=4096]" -o '
                    + current_sim_folder +
                    '/crab.out -n ' + NCPUS + ' -W 1440 mpirun -np '
                    + NCPUS + ' python ' + current_sim_folder
                    +'/restart_sim_nobunch.py -p 1 1 ' + NCPUS + '\n']

    with open('simulations/' + current_sim_ident
              + '/run_transient_single','w') as fid:
        fid.writelines(launch_lines1)
        
    with open('simulations/' + current_sim_ident
              + '/run_restart_single','w') as fid:
        fid.writelines(launch_lines2)
        
    with open('simulations/' + current_sim_ident
              + '/run_restart_nobunch_single','w') as fid:
        fid.writelines(launch_lines3)
        
    launch_file_lines1 += launch_lines1
    launch_file_lines2 += launch_lines2
    launch_file_lines3 += launch_lines3

with open(study_folder+'/run_transient_scan', 'w') as fid:
    fid.writelines(launch_file_lines1)
    
with open(study_folder+'/run_restart_scan', 'w') as fid:
    fid.writelines(launch_file_lines2)

with open(study_folder+'/run_restart_nobunch_scan', 'w') as fid:
    fid.writelines(launch_file_lines3)
