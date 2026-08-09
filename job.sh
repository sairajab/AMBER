#!/bin/bash
#SBATCH --output=sweep.log
#SBATCH --error=sweep.err
#SBATCH --job-name="Optuna"	 # job name
#SBATCH --partition=kestrel-gpu 		 # partition to which job should be submitted
#SBATCH --qos=gpu_long			  		 # qos type
#SBATCH --nodes=1                 		 # node count
#SBATCH --ntasks=1                		 # total number of tasks across all nodes
#SBATCH --cpus-per-task=4         		 # cpu-cores per task
#SBATCH --mem-per-cpu=7G
#SBATCH --gres=gpu:3090:1               #gpu:nvidia_a100_3g.39gb:1# Request 1 GPU (A100 40GB)
#SBATCH --time=3-12:27:14				 #  wall time

source /s/chromatin/a/nobackup/Saira/anaconda3/etc/profile.d/conda.sh
conda activate aam
#attention fit-sample-regressor --i-table filtered_table.biom --m-metadata-file data/metadata_sheds_hip_skin_2222.tsv --m-metadata-column outdoor_add_0 --output-dir output_run3 --p-missing-samples ignore --p-tree data/deblur-gg2-aligned-150bp-table.nwk  --p-penalty 0.1 --p-taxonomy-level 5 --p-patience 30 --p-intermediate-activation gelu --p-warmup-steps 100000
python src/microbiome_model/training/train_orig.py