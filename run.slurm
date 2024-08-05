#!/bin/bash
#SBATCH --job-name=paddle       # job name
#SBATCH --nodes=1               # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --cpus-per-task=24      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --partition=DGX
#SBATCH --qos=lv0b
#SBATCH --account=research
#SBATCH --time=10:00:00         # total run time limit (HH:MM:SS)
#SBATCH --mem=500G              # memory
#SBATCH --output=outputs/convert_translator/log.txt

module load cuda11.8
module load cudnn8.6-cuda11.8
module load gcc11
module load nccl2-cuda11.8-gcc11
source /scratch/zhaozijing/anaconda3/bin/activate
conda activate paddle

sh configs/convert_translator.sh
