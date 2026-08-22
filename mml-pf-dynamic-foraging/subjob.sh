#!/bin/bash
#SBATCH -p celltypes
#SBATCH --job-name blockcv
#SBATCH --array=0-49
#SBATCH --time 24:00:00
#SBATCH --nodes 1
#SBATCH --mem 32G
#SBATCH -c 8
#SBATCH -o blockcv_%A_task_%a.out

# Interleaved-block cross-validation (Pillow scheme): MMLPF + perseveration
# vs PsyTrack. One array task per mouse.
#
# BEFORE SUBMITTING, run the prepare stage once on hpc-login:
#
#   source ~/miniconda3/bin/activate cv_env
#   cd ~/dynamic-foraging-mml-particle-filter/mml-pf-dynamic-foraging/
#   python hpc_block_cv.py prepare --cache-dir ./cache --subjects 50 --sessions 20
#
# That queries the database once and caches per-subject trial arrays. Compute
# nodes frequently cannot reach the database, and 50 tasks querying it at once
# is antisocial where they can. It also prints the real number of qualifying
# subjects -- set --array below to (that number - 1), since 50 mice with 20
# sessions each at foraging_eff > 0.65 may not exist.
#
# AFTER the array finishes:
#
#   python hpc_block_cv.py combine --out-dir ./results

# Update the temporary path used by pip/conda
export TMPDIR="/allen/programs/celltypes/workgroups/mousecelltypes/rithvik.palepu/"

# The parallelism is across placements within each task (-c 8), so keep BLAS
# single-threaded per worker: 8 workers x 8 BLAS threads oversubscribes the
# cores and runs slower than serial.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Activate designated virtual environment
source ~/miniconda3/bin/activate cv_env

# Navigate to the specific project directory where you cloned the repo
cd ~/dynamic-foraging-mml-particle-filter/mml-pf-dynamic-foraging/

# Execute the python script, passing the array ID as an argument
python hpc_block_cv.py run \
    --array_id $SLURM_ARRAY_TASK_ID \
    --cache-dir ./cache \
    --out-dir ./results \
    --cores 8 \
    --seeds 10 \
    --block 20 \
    --test-frac 0.2