#!/bin/bash
#SBATCH --job-name=vmogeAD
#SBATCH --output=/scratch/delmiari/thesisproject/pipeline_results/slurm_%j.out
#SBATCH --error=/scratch/delmiari/thesisproject/pipeline_results/slurm_%j.err
#SBATCH --time=24:00:00                 # Keep at 24 hours (safest maximum allocation walltime)
#SBATCH --cpus-per-task=8               # Feeds into NUM_WORKERS=8 in Python for fast data loading
#SBATCH --mem=32G                       # Massive 32GB RAM memory footprint runway
#SBATCH --gres=gpu:1                    # Allocate 1 enterprise GPU node
#SBATCH --partition=bmeph               # Your lab's dedicated partition on Hinton


# ── Activate your environment ──────────────────────────────────────────────────
# If you use conda:
#   conda activate your_env_name
# If you use a venv:
source /scratch/delmiari/thesisproject/thesis_env/bin/activate

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "Working dir: $SLURM_SUBMIT_DIR"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

# ── Run the pipeline ──────────────────────────────────────────────────────────
echo "Starting prototyp pipeline training..."
python vmogeAD.py

echo "Job finished at $(date)"