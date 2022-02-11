#!/bin/bash
#SBATCH --time 14-00:00:00     # Time limit for the job (REQUIRED)
#SBATCH --job-name=j20_bin   # Job name
#SBATCH --nodes=1 # Number of nodes to allocate. Same as SBATCH -N
#SBATCH --ntasks=5# Number of cores to allocate. Same as SBATCH -n
#SBATCH --partition=normal    # Partition/queue to run the job in. (REQUIRED)
#SBATCH -e slurm-%j.err  # Error file for this job.
#SBATCH -o slurm-%j.out  # Output file for this job.
#SBATCH -A coa_rya225_uksr  # Project allocation account name (REQUIRED)
#SBATCH --mail-type ALL    # Send email when job starts/ends
#SBATCH --mail-user naj222@g.uky.edu   # Where email is sent to (optional)
python /mnt/gpfs3_amd/scratch/naj222/dir1/code/norm2_all_lick_bin/code20.py
