import os

def htcondor_config(scan_folder, time_requirement_days, ncpus = 1, job_filename = 'job.job',
                    list_filename = 'list_sim_folders.txt', sub_filename = 'htcondor.sub', 
                    out_filename = 'htcondor.out', err_filename = 'htcondor.err', 
                    log_filename = 'htcondor.log', run_filename = 'run_htcondor'):

    list_folders= os.listdir(scan_folder)
    list_submit = []
    for folder in list_folders:
        if os.path.isfile('simulations/'+folder+'/'+job_filename):
            list_submit.append(folder+'\n')
    with open(list_filename, 'w') as fid:
        fid.writelines(list_submit)

    with open(sub_filename, 'w') as fid:
        fid.write("universe = vanilla\n")
        fid.write("executable = "+ scan_folder+"/$(dirname)/" + job_filename + "\n")
        fid.write('arguments = ""\n')
        fid.write("output = " + scan_folder + "/$(dirname)/" + out_filename + "\n")
        fid.write("error = " + scan_folder + "/$(dirname)/" + err_filename + "\n")
        fid.write("log = "+scan_folder+"/$(dirname)/" + log_filename +  "\n")
        fid.write("RequestCpus = " + ncpus + "\n")
        fid.write('transfer_output_files = ""\n')
        fid.write("+MaxRuntime = %d\n"%(time_requirement_days*24*3600))
        fid.write("queue dirname from "+list_filename+"\n")
    with open(run_filename, 'w') as fid:
        fid.write('condor_submit '+sub_filename+'\n')
        fid.write('condor_q --nobatch\n')
#    os.chmod('../run_htcondor',0755)

