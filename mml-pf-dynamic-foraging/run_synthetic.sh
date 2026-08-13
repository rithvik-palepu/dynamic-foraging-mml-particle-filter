#!/bin/bash
#SBATCH --job-name=synth_boundary
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8      # Requests 8 dedicated cores for our workers=-1 parallelization
#SBATCH --mem=16GB             # Requests 16GB of RAM
#SBATCH --time=02:00:00        # Sets a 2-hour maximum time limit
#SBATCH --output=synth_%j.log  # Saves the terminal output to a log file

# Initialize conda and activate your specific environment
eval "$(conda shell.bash hook)"
conda activate cv_env

# Execute the Python script
python synthetic_drifting_agent.py