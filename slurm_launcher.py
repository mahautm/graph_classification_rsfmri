import json
import os


def run_slurm_job(
    sub_name,
    email="mmahaut@ensc.fr",
    logs_dir="/scratch/mmahaut/scripts/logs",
    python_path="python",
    slurm_dir="/scratch/mmahaut/scripts/slurm",
    code_dir="/scratch/mmahaut/scripts/graph_classification_rsfmri",
    script_name_1="01_parcellation.py",
    script_name_2="02_compute_parcel_adjacency.py",
    time_wall="12:00:00",
):
    """
    This will write a .sh file to launch a script on a specific subject.

    Parameters
    ----------
    sub_name : string
        name of subjet to execute script on

    email : string, email address, default "mmahaut@ensc.fr"
        email to which notifications will be sent at the end of the execution of the script for the first fold of a given dimension

    logs_dir : string, path, default "/scratch/mmahaut/scripts/logs"
        path to where both the output and the error logs will be saved.

    python_path : string, path, default "python"
        path to where the python version used to run the mdae_step.py script is found. 3.7 & 3.8 work.
    
    slurm_dir : string, path, default "/scratch/mmahaut/scripts/slurm"
        path to where the .sh script used to enter the slurm queue will be saved before being executed

    code_dir : string, path, default "/scratch/mmahaut/scripts/INT_fMRI_processing"
        path to where the script can be found

    script_name : string, path, default "mdae_step.py"
        script file name
    
    time_wall : string, hh:mm:ss time format, default "12:00:00"
        time after which running script will be killed
    """

    job_name = "{}_graph".format(sub_name)
    slurmjob_path = os.path.join(slurm_dir, "{}.sh".format(job_name))
    create_slurmjob_cmd = "touch {}".format(slurmjob_path)
    os.system(create_slurmjob_cmd)

    # write arguments into the slurmjob file
    with open(slurmjob_path, "w") as fh:
        fh.writelines("#!/bin/sh\n")
        fh.writelines("#SBATCH --job-name={}\n".format(job_name))
        fh.writelines("#SBATCH -o {}/{}_%j.out\n".format(logs_dir, job_name))
        fh.writelines("#SBATCH -e {}/{}_%j.err\n".format(logs_dir, job_name))
        fh.writelines("#SBATCH --time={}\n".format(time_wall))
        fh.writelines("#SBATCH --account=b125\n")
        fh.writelines("#SBATCH --partition=skylake\n")
        # fh.writelines("#SBATCH --gres-flags=enforce-binding\n")
        # number of nodes for this job
        fh.writelines("#SBATCH --nodes=1\n")
        # number of cores for this job
        fh.writelines("#SBATCH --ntasks-per-node=10\n")  # ??
        # email alerts
        fh.writelines("#SBATCH --mail-type=END\n")
        fh.writelines("#SBATCH --mail-user={}\n".format(email))
        # making sure group is ok for data sharing within group
        batch_cmd = (
            'eval "$(/scratch/mmahaut/tools/Anaconda3/bin/conda shell.bash hook)"\n'
            + "conda activate tf\n"
            + "{} {}/{} {}\n".format(python_path, code_dir, script_name_1, sub_name)
            + "{} {}/{} {}".format(python_path, code_dir, script_name_2, sub_name)
        )
        fh.writelines(batch_cmd)

    os.system("sbatch %s" % slurmjob_path)


if __name__ == "__main__":
    subs_list_file = open(
        "/scratch/mmahaut/scripts/graph_classification_rsfmri/subs_list_asd.json"
    )
    subs_list = json.load(subs_list_file)
    for sub_name in subs_list:
        run_slurm_job(sub_name)
