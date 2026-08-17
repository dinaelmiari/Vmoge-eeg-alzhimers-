#!/bin/bash
#SBATCH --job-name=vmoge_ablation
#SBATCH --output=/scratch/delmiari/thesisproject/pipeline_results/ablation_%j.out
#SBATCH --error=/scratch/delmiari/thesisproject/pipeline_results/ablation_%j.err
#SBATCH --time=24:00:00                 # Keep at 24 hours (safest maximum allocation walltime)
#SBATCH --cpus-per-task=8               # Feeds into NUM_WORKERS=8 in Python for fast data loading
#SBATCH --mem=32G                       # Massive 32GB RAM memory footprint runway
#SBATCH --gres=shard:12                    # Allocate 1 enterprise GPU node
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
python classifier_ablation.py

echo "Job finished at $(date)"