import os
import numpy as np
import replaceline as rl
from pathlib import Path

from scipy.constants import c as clight

    
tobecopied = 'complete_sim.py'

current_dir = os.getcwd()
study_folder =  current_dir.split('/config')[0]
scan_folder = study_folder+'/simulations'
os.mkdir(scan_folder)
os.mkdir(scan_folder+'/progress')

launch_file_lines = []
launch_file_lines +=['#!/bin/bash\n']

Emax_vect = np.linspace(0,55e6,12)

prog_num = 0 
for Emax in Emax_vect:                
    prog_num +=1
    current_sim_ident = 'complete_cavity_Emax_'+str(Emax/1e6)
    print(current_sim_ident)
    current_sim_folder = scan_folder+'/'+current_sim_ident
    os.mkdir(current_sim_folder)
    
    rl.replaceline_and_save(fname = 'complete_sim.py',
     findln = 'E_field_max = ', newline = 'E_field_max = %d,\n'%Emax)            
    
    rl.replaceline_and_save(fname = 'complete_sim.py',
     findln = "'images_dir': ", newline = "\t'images_dir': '"+current_sim_folder+"/images',\n")

    rl.replaceline_and_save(fname = 'complete_sim.py',
     findln = "'temps_filename': ", newline = "\t'temps_filename': '"+current_sim_folder+"/complete_temp.mat',\n")

    rl.replaceline_and_save(fname = 'complete_sim.py',
     findln = "'output_filename': ", newline = "\t'output_filename': '"+current_sim_folder+"/complete_out.mat',\n")
 
    launch_lines = ['bsub -B -N -R "rusage[mem=4096]" -o '
                    + current_sim_folder +
                    '/crab.out -n 24 -W 1440 mpirun -np 24 python ' 
                    + current_sim_folder 
                    +'/complete_sim.py -p 1 1 24\n']   
                 
    with open('launch_single','w') as fid:
        fid.writelines(launch_lines)                     

    os.system('cp -r %s %s'%(tobecopied, current_sim_folder))
    
    launch_file_lines += launch_lines

            
                                    
with open(study_folder+'/run_CC_scan', 'w') as fid:
    fid.writelines(launch_file_lines)

